"""FP8 DiT (dit_fp8) vs the bit-exact BF16 DiT: per-forward error, speed, and full images (PSNR) on a few seeds.

Run with ComfyUI's venv:  comfy-venv/bin/python generate_fp8_quality.py [OUT_DIR]
Writes exact/fp8 PNG pairs to OUT_DIR for viewing.
"""
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "plugins/image-generation/native/anima"))
import generate  # noqa: E402
import sampler  # noqa: E402
from tokens import prompt_tokens  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/anima-native/fp8-quality"
OUT.mkdir(parents=True, exist_ok=True)
REF = ROOT / ".runtime/anima-native/generate-reference"
M = generate.MODELS
generate.build()
with safe_open(str(M / "text_encoders/qwen_3_06b_base.safetensors"), "pt", device="cuda") as f:
    qwen_w = {k: f.get_tensor(k).float() for k in f.keys()}
with safe_open(str(M / "diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
    dit_w = {k: f.get_tensor(k) for k in f.keys()}
with safe_open(str(M / "vae/qwen_image_vae.safetensors"), "pt", device="cuda") as f:
    vae_w = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys() if k.startswith(("decoder.", "conv2."))}
fp8_w = torch.ops.dsh_anima_gen.prepare_fp8(dit_w)
qt, tt = (Tokenizer.from_file(str(generate.FRONTEND / f"{n}_tokenizer.json")) for n in ("qwen", "t5"))
ref = load_file(str(REF / "reference.safetensors"))


def condition(text):
    q, t, w = prompt_tokens(qt, tt, text)
    return torch.ops.dsh_anima_gen.crossattn(qwen_w, dit_w, torch.tensor(q, device="cuda"), torch.tensor(t, device="cuda"),
                                             torch.tensor(w))


def timed(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / n * 1000


with torch.inference_mode():
    print("per forward (captured inputs, batch 2): relative L2 of FP8 vs exact")
    for call in ("call0", "call1", "call2", "last"):
        x, t = ref[f"dit.{call}.x"].cuda(), ref[f"dit.{call}.timesteps"].cuda()
        c = ref["dit.call0.context"].cuda()
        exact = torch.ops.dsh_anima_gen.dit(dit_w, x, t, c).float()
        fast = torch.ops.dsh_anima_gen.dit_fp8(fp8_w, x, t, c).float()
        print(f"  {call} sigma {t[0].item():.4f}: {((fast - exact).norm() / exact.norm()).item():.3e}")
    print(f"forward: exact {timed(lambda: torch.ops.dsh_anima_gen.dit(dit_w, x, t, c)):.1f} ms, "
          f"fp8 {timed(lambda: torch.ops.dsh_anima_gen.dit_fp8(fp8_w, x, t, c)):.1f} ms")

    prompts = [(REF / "prompt1.txt").read_text(),
               "1girl, solo, school uniform, cherry blossoms, wind, looking at viewer, smile, detailed background, masterpiece",
               "a cozy cafe interior at night, warm lights, rain on the window, a cat sleeping on a chair, anime style"]
    negative = condition("")
    for index, prompt in enumerate(prompts):
        cond = condition(prompt)
        images = {}
        for mode in ("exact", "fp8"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            latent = sampler.sample(fp8_w if mode == "fp8" else dit_w, cond, negative, 1000 + index, fp8=mode == "fp8")
            image = torch.ops.dsh_anima_gen.vae_decode(vae_w, latent)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            images[mode] = (image.cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(images[mode]).save(OUT / f"{index}-{mode}.png")
            print(f"prompt {index} {mode}: {seconds:.2f} s")
        mse = np.mean((images["exact"].astype(float) - images["fp8"].astype(float)) ** 2)
        print(f"prompt {index}: PSNR fp8 vs exact {10 * np.log10(255 ** 2 / max(mse, 1e-9)):.2f} dB")
