"""Native Anima text-to-image job (no ComfyUI code): prompt -> Qwen3 + LLM adapter -> ER-SDE/CFG with the DiT -> VAE.

Usage: comfy-venv/bin/python generate.py JOB_DIR   (JOB_DIR/request.json is written by jobs.mjs)
request: prompt, negative, seed, count, steps, cfg, width, height, loras [{path, strength}], output_dir.
Progress goes to JOB_DIR/status.json; images to output_dir as NNN-<seed>.png with the settings in a PNG text chunk.
Bit-exact with ComfyUI's UNETLoader/LoraLoaderModelOnly/KSampler(er_sde, simple)/VAEDecode graph
(docs/anima-generate-native-plan.md). The caller cancels by killing the process group.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
MODELS = ROOT / "plugins/dsh-comfyui/models"
FRONTEND = ROOT / ".runtime/anima-native/frontend"
KITCHEN_BUILD = ROOT / ".runtime/h3-native/build"  # libdsh_kitchen.so (vendored comfy-kitchen kernels)
BUILD = ROOT / ".runtime/anima-native/generate-build"
MAX_T5_TOKENS = 512

JOB = Path(sys.argv[1]).resolve()
status = {"stage": "starting", "done": 0, "total": 0, "images": [], "started": time.time()}


def update(**changes):
    status.update(changes, updated=time.time())
    with tempfile.NamedTemporaryFile("w", dir=JOB, delete=False, encoding="utf-8") as temporary:
        json.dump(status, temporary, ensure_ascii=False)
    os.replace(temporary.name, JOB / "status.json")


def build():
    import torch
    from torch.utils import cpp_extension
    BUILD.mkdir(parents=True, exist_ok=True)
    cu13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
    cpp_extension.load(name="dsh_anima_generate", is_python_module=False, with_cuda=False, build_directory=str(BUILD),
                       sources=[str(HERE / f"{n}.cpp") for n in ("text", "dit", "vae")] +
                               [str(ROOT / "plugins/dsh-minimax-h3/native/kitchen/kitchen_ops.cpp")],
                       extra_cflags=["-O3"],
                       extra_include_paths=[str(cu13 / "include"), str(ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/include")],
                       extra_ldflags=[str(KITCHEN_BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{KITCHEN_BUILD}",
                                      f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"])


def main():
    request = json.loads((JOB / "request.json").read_text())
    count = int(request["count"])
    update(stage="loading", total=count, message="モデルを読み込んでいます")
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from safetensors import safe_open
    from tokenizers import Tokenizer
    import torch
    import lora
    import sampler
    from tokens import prompt_tokens

    started = time.time()
    build()
    with safe_open(str(MODELS / "text_encoders/qwen_3_06b_base.safetensors"), "pt", device="cuda") as f:
        qwen_w = {k: f.get_tensor(k).float() for k in f.keys()}  # ComfyUI runs this encoder in FP32
    with safe_open(str(MODELS / "diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
        dit_w = {k: f.get_tensor(k) for k in f.keys()}
    with safe_open(str(MODELS / "vae/qwen_image_vae.safetensors"), "pt", device="cuda") as f:
        vae_w = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys() if k.startswith(("decoder.", "conv2."))}
    if request.get("loras"):
        dit_w = lora.merge(dit_w, [(lora.load(item["path"]), float(item["strength"])) for item in request["loras"]])
    qwen_tok = Tokenizer.from_file(str(FRONTEND / "qwen_tokenizer.json"))
    t5_tok = Tokenizer.from_file(str(FRONTEND / "t5_tokenizer.json"))

    def condition(text, label):
        qwen_ids, t5_ids, t5_weights = prompt_tokens(qwen_tok, t5_tok, text)
        if len(t5_ids) > MAX_T5_TOKENS:
            raise ValueError(f"{label}が長すぎます（T5 {len(t5_ids)} トークン、上限 {MAX_T5_TOKENS}）")
        return torch.ops.dsh_anima_gen.crossattn(qwen_w, dit_w, torch.tensor(qwen_ids, device="cuda"),
                                                 torch.tensor(t5_ids, device="cuda"), torch.tensor(t5_weights))

    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        cond, uncond = condition(request["prompt"], "プロンプト"), condition(request.get("negative", ""), "ネガティブ")
        update(stage="generating", message=None, load_seconds=time.time() - started)
        for index in range(count):
            seed = int(request["seed"]) + index
            step_started = time.time()

            def progress(i, x, denoised):
                update(step=i + 1, steps=request["steps"])
            latent = sampler.sample(dit_w, cond, uncond, seed, steps=int(request["steps"]), cfg=float(request["cfg"]),
                                    height=int(request["height"]), width=int(request["width"]), on_step=progress)
            image = torch.ops.dsh_anima_gen.vae_decode(vae_w, latent).cpu().numpy()
            info = PngInfo()
            info.add_text("parameters", json.dumps({k: request[k] for k in ("prompt", "negative", "steps", "cfg", "width", "height")}
                                                   | {"seed": seed, "sampler": "er_sde", "scheduler": "simple",
                                                      "loras": [{"name": i["name"], "strength": i["strength"]} for i in request.get("loras", [])]},
                                                   ensure_ascii=False))
            path = output / f"{index + 1:03d}-{seed}.png"
            Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8)).save(path, pnginfo=info)
            status["images"].append({"file": path.name, "seed": seed, "seconds": round(time.time() - step_started, 2)})
            update(done=index + 1)
    update(stage="done", message="完了", finished=time.time())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # reported to the panel; the traceback goes to stderr for jobs.mjs
        import traceback
        traceback.print_exc()
        update(stage="error", message=str(error), finished=time.time())
        sys.exit(1)
