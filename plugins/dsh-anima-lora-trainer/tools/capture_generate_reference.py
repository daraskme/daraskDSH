"""Run a second ComfyUI instance with hooks that record the Anima text-to-image reference tensors.

Usage: comfy-venv/bin/python capture_generate_reference.py OUT_DIR [PORT]
Then POST an Anima T2I API graph to http://127.0.0.1:PORT/prompt. Saved under OUT_DIR/reference.safetensors:
tokens/weights, Qwen hidden states, the c_crossattn the DiT receives (positive and negative), the initial
noise, sigmas, er_sde inputs/denoised/noise per step (first 3 and last), DiT calls (first 3 and last),
node outputs (KSampler latent, decoded image) and per-node wall times (timings.json).
Not for speed measurements: tensors are copied to the CPU.
"""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

COMFY = Path(__file__).resolve().parents[3] / ".runtime/ComfyUI"
OUT = Path(sys.argv[1]).resolve()
PORT = sys.argv[2] if len(sys.argv) > 2 else "8189"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(COMFY)
sys.path.insert(0, str(COMFY))
sys.argv = ["main.py", "--listen", "127.0.0.1", "--port", PORT, "--disable-auto-launch"]

saved, timings, counters = {}, [], {}


def tensors(value, prefix):
    import torch
    found = {}
    if hasattr(value, "tensors") and isinstance(getattr(value, "tensors"), list):
        for index, item in enumerate(value.tensors):
            found.update(tensors(item, f"{prefix}.{index}"))
    elif isinstance(value, torch.Tensor):
        found[prefix] = value.detach().to("cpu").clone()
    elif isinstance(value, dict):
        for key, item in value.items():
            found.update(tensors(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(tensors(item, f"{prefix}.{index}"))
    return found


def keep(name, value):
    saved.update(tensors(value, name))


def count(name):
    counters[name] = counters.get(name, 0) + 1
    return counters[name] - 1


def flush():
    from safetensors.torch import save_file
    if saved:
        save_file({k: v.contiguous() for k, v in saved.items()}, str(OUT / "reference.safetensors"))
    (OUT / "timings.json").write_text(json.dumps(timings, indent=1))


def install_hooks(execution):
    import torch
    import comfy.k_diffusion.sampling as kd
    import comfy.ldm.anima.model as anima
    import comfy.sample
    import comfy.text_encoders.anima as te

    tokenize = te.AnimaTokenizer.tokenize_with_weights

    def tokens(self, text, *args, **kwargs):
        out = tokenize(self, text, *args, **kwargs)
        index = count("prompt")
        for encoder, batches in out.items():
            keep(f"prompt{index}.{encoder}.ids", torch.tensor([t[0] for b in batches for t in b], dtype=torch.int64))
            keep(f"prompt{index}.{encoder}.weights", torch.tensor([float(t[1]) for b in batches for t in b]))
        (OUT / f"prompt{index}.txt").write_text(text)
        return out
    te.AnimaTokenizer.tokenize_with_weights = tokens

    encode = te.AnimaTEModel.encode_token_weights

    def encoded(self, token_weight_pairs):
        out = encode(self, token_weight_pairs)
        index = count("encode")
        keep(f"encode{index}.hidden", out[0])
        keep(f"encode{index}.extra", out[2])
        return out
    te.AnimaTEModel.encode_token_weights = encoded

    preprocess = anima.Anima.preprocess_text_embeds

    def crossattn(self, text_embeds, text_ids, t5xxl_weights=None):
        out = preprocess(self, text_embeds, text_ids, t5xxl_weights=t5xxl_weights)
        index = count("crossattn")
        keep(f"crossattn{index}.in.embeds", text_embeds)
        keep(f"crossattn{index}.in.ids", text_ids)
        if t5xxl_weights is not None:
            keep(f"crossattn{index}.in.weights", t5xxl_weights)
        keep(f"crossattn{index}.out", out)
        return out
    anima.Anima.preprocess_text_embeds = crossattn

    forward = anima.Anima.forward

    def dit(self, x, timesteps, context, **kwargs):
        index = count("dit")
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = forward(self, x, timesteps, context, **kwargs)
        torch.cuda.synchronize()
        timings.append({"what": f"dit.call{index}", "seconds": time.perf_counter() - started})
        if index < 3 or os.environ.get("DSH_CAPTURE_ALL_DIT") == "1":
            keep(f"dit.call{index}.x", x)
            keep(f"dit.call{index}.timesteps", timesteps)
            keep(f"dit.call{index}.context", context)
            keep(f"dit.call{index}.out", out)
            keep(f"dit.call{index}.kwargs", {k: v for k, v in kwargs.items() if k != "transformer_options"})
        keep("dit.last.x", x)
        keep("dit.last.timesteps", timesteps)
        keep("dit.last.out", out)
        return out
    anima.Anima.forward = dit

    prepare_noise = comfy.sample.prepare_noise

    def noise(latent_image, seed, batch_inds=None):
        out = prepare_noise(latent_image, seed, batch_inds)
        keep("noise.initial", out)
        (OUT / "seed.txt").write_text(str(seed))
        return out
    comfy.sample.prepare_noise = noise

    er_sde = kd.sample_er_sde

    def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, **kwargs):
        keep("er_sde.x", x)
        keep("er_sde.sigmas", sigmas)
        steps = {"n": 0}

        def wrapped_model(xi, sigma, **kw):
            i = steps["n"]
            steps["n"] += 1
            out = model(xi, sigma, **kw)
            if i < 3 or i == len(sigmas) - 2:
                keep(f"er_sde.step{i}.x", xi)
                keep(f"er_sde.step{i}.sigma", sigma)
                keep(f"er_sde.step{i}.denoised", out)
            return out
        for attr in ("inner_model", "model_sampling"):
            if hasattr(model, attr):
                setattr(wrapped_model, attr, getattr(model, attr))
        if noise_sampler is None:
            noise_sampler = kd.default_noise_sampler(x, seed=extra_args.get("seed", None) if extra_args else None)
        drawn = {"n": 0}

        def wrapped_noise(sigma, sigma_next):
            out = noise_sampler(sigma, sigma_next)
            if drawn["n"] < 3:
                keep(f"er_sde.noise{drawn['n']}", out)
            drawn["n"] += 1
            return out
        result = er_sde(wrapped_model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                        s_noise=s_noise, noise_sampler=wrapped_noise, **kwargs)
        keep("er_sde.out", result)
        return result
    kd.sample_er_sde = sampler

    get_output_data = execution.get_output_data

    async def node_output(prompt_id, unique_id, obj, input_data_all, *args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = await get_output_data(prompt_id, unique_id, obj, input_data_all, *args, **kwargs)
        torch.cuda.synchronize()
        name = type(obj).__name__ if not isinstance(obj, type) else obj.__name__
        timings.append({"what": f"node.{unique_id}.{name}", "seconds": time.perf_counter() - started})
        keep(f"node.{unique_id}.{name}.out", result[0])
        flush()
        return result
    execution.get_output_data = node_output


class HookExecution:
    def find_spec(self, name, path=None, target=None):
        if name != "execution":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            install_hooks(module)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, HookExecution())
runpy.run_path(str(COMFY / "main.py"), run_name="__main__")
