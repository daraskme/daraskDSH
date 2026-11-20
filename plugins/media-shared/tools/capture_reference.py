"""Run ComfyUI (second instance) with hooks that record the H3 reference tensors.

Usage: comfy-venv/bin/python capture_reference.py OUT_DIR [PORT]
Then POST the API graph to http://127.0.0.1:PORT/prompt. Every executed node's tensor
inputs/outputs and wall time, the DiT calls (steps 0, 1 and last), the Qwen token IDs and
hidden states, the token-refiner output and the sampling noise are saved under OUT_DIR.
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

saved = {}
timings = []


def tensors(value, prefix):
    """Flatten nested node values into {name: tensor} (CPU copies)."""
    import torch
    found = {}
    if hasattr(value, "tensors") and isinstance(getattr(value, "tensors"), list):  # comfy NestedTensor (AV latent)
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


TIMINGS_ONLY = os.environ.get("DSH_CAPTURE_TIMINGS_ONLY") == "1"


def keep(name, value):
    if not TIMINGS_ONLY:
        saved.update(tensors(value, name))


def install_hooks(execution):
    # Runs right after ComfyUI's main.py imports `execution`, so its environment setup came first.
    import torch
    import comfy.ldm.minimax.model as dit
    import comfy.text_encoders.minimax as te
    import comfy_extras.nodes_custom_sampler as custom

    tokenize = te.MiniMaxH3Tokenizer.tokenize_with_weights

    def tokens(self, text, *args, **kwargs):
        out = tokenize(self, text, *args, **kwargs)
        # {encoder: [[(token, weight, ...), ...]]} -> int64 token IDs per encoder
        for encoder, batches in out.items():
            ids = [item[0] for batch in batches for item in batch if isinstance(item[0], int)]
            keep(f"tokenizer.{encoder}.ids", torch.tensor(ids, dtype=torch.int64))
            # -1 marks a vision entry (its image goes to the vision tower)
            marked = [item[0] if isinstance(item[0], int) else -1 for batch in batches for item in batch]
            keep(f"tokenizer.{encoder}.marked", torch.tensor(marked, dtype=torch.int64))
            images = [item[0]["data"] for batch in batches for item in batch if isinstance(item[0], dict)]
            for index, image in enumerate(images):
                keep(f"tokenizer.{encoder}.image{index}", image)
        (OUT / "prompt.txt").write_text(text)
        return out
    te.MiniMaxH3Tokenizer.tokenize_with_weights = tokens

    qwen_forward = te.MiniMaxQwen3VL.forward

    def qwen(self, input_ids, *args, **kwargs):
        out = qwen_forward(self, input_ids, *args, **kwargs)
        keep("te.input_ids", input_ids)
        keep("te.embeds", kwargs.get("embeds"))
        info = kwargs.get("embeds_info") or []
        (OUT / "embeds_info.json").write_text(json.dumps([{k: v for k, v in e.items() if isinstance(v, (int, str))} for e in info]))
        keep("te.output", out)
        return out
    te.MiniMaxQwen3VL.forward = qwen

    import comfy.text_encoders.qwen3vl as qvl
    import comfy.ldm.minimax.vae as mvae
    vision_forward = qvl.Qwen3VLVisionModel.forward
    vision_calls = {"n": 0}

    def vision(self, x, grid, *args, **kwargs):
        out = vision_forward(self, x, grid, *args, **kwargs)
        i = vision_calls["n"]
        vision_calls["n"] += 1
        keep(f"vision{i}.pixels", x)
        keep(f"vision{i}.grid", grid)
        keep(f"vision{i}.out", out)
        return out
    qvl.Qwen3VLVisionModel.forward = vision
    build_inputs = qvl.Qwen3VL.build_image_inputs

    def image_inputs(self, embeds, embeds_info):
        out = build_inputs(self, embeds, embeds_info)
        keep("te.position_ids", out[0])
        keep("te.visual_pos_masks", out[1])
        return out
    qvl.Qwen3VL.build_image_inputs = image_inputs
    encode_calls = {"n": 0}
    for cls in (mvae.MiniMaxH3VideoVAE, mvae.H3ImageVAE):
        def encode(self, x, *args, _original=cls.encode, **kwargs):  # H3ImageVAE.encode does not call super
            out = _original(self, x, *args, **kwargs)
            i = encode_calls["n"]
            encode_calls["n"] += 1
            keep(f"vae_encode{i}.in", x)
            keep(f"vae_encode{i}.out", out)
            return out
        cls.encode = encode

    refiner_forward = dit.TokenRefiner.forward
    inside = {"refiner": False, "linear": 0}

    def refiner(self, x, *args, **kwargs):
        keep("refiner.input", x)
        inside["refiner"] = os.environ.get("DSH_CAPTURE_REFINER") == "1"
        try:
            out = refiner_forward(self, x, *args, **kwargs)
        finally:
            inside["refiner"] = False
        keep("refiner.output", out)
        return out
    dit.TokenRefiner.forward = refiner

    # With DSH_CAPTURE_REFINER=1, record every module and linear inside the token refiner.
    for cls_name in ("RefinerBlock", "Attention", "MLP"):
        cls = getattr(dit, cls_name)
        original = cls.forward

        def wrapped(self, x, *args, _original=original, _name=cls_name, **kwargs):
            out = _original(self, x, *args, **kwargs)
            if inside["refiner"]:
                index = inside.setdefault(_name, 0)
                inside[_name] += 1
                keep(f"refiner.{_name}{index}.in", x)
                keep(f"refiner.{_name}{index}.out", out)
            return out
        cls.forward = wrapped
    # With DSH_CAPTURE_BLOCK0=1, record the first DiT block of the first forward (and its attention/MLP).
    block_forward = dit.DiTBlock.forward
    seen = {"block": 0}

    def block(self, x, t_emb, mod_segments, rope_freqs, *args, **kwargs):
        first = os.environ.get("DSH_CAPTURE_BLOCK0") == "1" and seen["block"] == 0
        seen["block"] += 1
        if first:
            keep("block0.x", x.clone())
            keep("block0.t_emb", t_emb)
            keep("block0.rope_freqs", rope_freqs)
            (OUT / "block0_segments.json").write_text(json.dumps(
                [(a, b, r if isinstance(r, int) else r.tolist()) for a, b, r in mod_segments]))
            inside["block0"] = True
        out = block_forward(self, x, t_emb, mod_segments, rope_freqs, *args, **kwargs)
        if first:
            inside["block0"] = False
            keep("block0.out", out)
        return out
    dit.DiTBlock.forward = block
    for cls_name in ("Attention", "MLP"):
        cls = getattr(dit, cls_name)
        original = cls.forward

        def wrapped0(self, x, *args, _original=original, _name=cls_name, **kwargs):
            out = _original(self, x, *args, **kwargs)
            if inside.get("block0"):
                keep(f"block0.{_name}.in", x)
                keep(f"block0.{_name}.out", out)
            return out
        cls.forward = wrapped0
    functional_linear = torch.nn.functional.linear

    def linear(x, weight, bias=None):
        out = functional_linear(x, weight, bias)
        if inside["refiner"]:
            index = inside["linear"]
            inside["linear"] += 1
            keep(f"refiner.linear{index}.x", x)
            keep(f"refiner.linear{index}.weight", weight)
            keep(f"refiner.linear{index}.out", out)
        return out
    torch.nn.functional.linear = linear

    model_forward = dit.MiniMaxH3Model.forward
    calls = {"n": 0}

    def model(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        index = calls["n"]
        calls["n"] += 1
        torch.cuda.synchronize()
        started = time.perf_counter()
        if os.environ.get("DSH_CAPTURE_PROFILE") == "1" and index == 3:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = model_forward(self, x, timestep, context, transformer_options=transformer_options,
                                    minimax_payload=minimax_payload, **kwargs)
                torch.cuda.synchronize()
            rows = [(e.self_device_time_total / 1000, e.count, e.key) for e in prof.key_averages() if e.self_device_time_total > 0]
            rows.sort(reverse=True)
            (OUT / "step_profile.json").write_text(json.dumps(rows[:60], indent=1))
        else:
            out = model_forward(self, x, timestep, context, transformer_options=transformer_options,
                                minimax_payload=minimax_payload, **kwargs)
        torch.cuda.synchronize()
        timings.append({"what": f"dit.call{index}", "seconds": time.perf_counter() - started})
        layout = getattr(self, "_last_layout", None)
        if index in (0, 1) or index >= 7:
            keep(f"dit.call{index}.x", x)
            keep(f"dit.call{index}.timestep", timestep)
            keep(f"dit.call{index}.context", context)
            keep(f"dit.call{index}.payload", minimax_payload or {})
            layout = (minimax_payload or {}).get("layout")
            if layout is not None and index == 0:
                keep("layout.position_ids", layout.position_ids)
                (OUT / "layout_segments.json").write_text(json.dumps(layout.segments))
            keep(f"dit.call{index}.out", out)
        return out
    dit.MiniMaxH3Model.forward = model

    noise = custom.Noise_RandomNoise.generate_noise

    def generate(self, input_latent):
        out = noise(self, input_latent)
        keep("noise.input", input_latent)
        keep("noise.output", out)
        return out
    custom.Noise_RandomNoise.generate_noise = generate

    get_output_data = execution.get_output_data

    async def node_output(prompt_id, unique_id, obj, input_data_all, *args, **kwargs):
        import torch
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = await get_output_data(prompt_id, unique_id, obj, input_data_all, *args, **kwargs)
        torch.cuda.synchronize()
        name = type(obj).__name__ if not isinstance(obj, type) else obj.__name__
        timings.append({"what": f"node.{unique_id}.{name}", "seconds": time.perf_counter() - started})
        keep(f"node.{unique_id}.{name}.in", {k: v for k, v in input_data_all.items() if k not in ("model", "clip", "vae")})
        keep(f"node.{unique_id}.{name}.out", result[0])
        flush()
        return result
    execution.get_output_data = node_output


def flush():
    from safetensors.torch import save_file
    if saved:
        save_file({k: v.contiguous() for k, v in saved.items()}, str(OUT / "reference.safetensors"))
    (OUT / "timings.json").write_text(json.dumps(timings, indent=1))


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


def install_guide_node():
    """DSH_CAPTURE_GUIDE=file.safetensors: a "DSHCaptureGuide" node that appends raw latent keyframes like the long-video
    joins (video: 1-step blocks at frame offsets; audio: one audio_latent keyframe), read from the file's
    video.<frame index> / audio.<frame index> tensors."""
    path = os.environ.get("DSH_CAPTURE_GUIDE")
    if not path:
        return
    import nodes
    import node_helpers
    from safetensors.torch import load_file

    class DSHCaptureGuide:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"positive": ("CONDITIONING",)}}
        RETURN_TYPES = ("CONDITIONING",)
        FUNCTION = "run"
        CATEGORY = "dsh"

        def run(self, positive):
            data = load_file(path)
            keyframes = list(positive[0][1].get("minimax_keyframes", []))
            for key in sorted((k for k in data if k.startswith("video.")), key=lambda k: float(k[6:])):
                keyframes.append({"resolved_frame_index": int(float(key[6:])), "latent": data[key]})
            for key in (k for k in data if k.startswith("audio.")):
                keyframes.append({"resolved_frame_index": float(key[6:]), "audio_latent": data[key]})
            return (node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes}),)

    nodes.NODE_CLASS_MAPPINGS["DSHCaptureGuide"] = DSHCaptureGuide


class HookNodes:
    def find_spec(self, name, path=None, target=None):
        if name != "nodes":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            sys.modules["nodes"] = module
            install_guide_node()
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, HookExecution())
sys.meta_path.insert(0, HookNodes())
runpy.run_path(str(COMFY / "main.py"), run_name="__main__")
