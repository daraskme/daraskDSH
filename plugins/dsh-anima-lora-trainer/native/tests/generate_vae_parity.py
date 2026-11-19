"""Anima VAE decode (generate/vae.cpp) vs ComfyUI's VAEDecode on the captured KSampler latent.

Run with ComfyUI's venv:  comfy-venv/bin/python generate_vae_parity.py [REFERENCE_DIR]
ComfyUI decodes this VAE in BF16 (its first working dtype); FP16 and FP32 are shown for comparison.
"""
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[4]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/anima-native/generate-reference"
sys.path.insert(0, str(ROOT / "plugins/dsh-anima-lora-trainer/native/generate"))
import generate  # noqa: E402

generate.build()  # the C++ ops and FP8 kernels, shared with the runner
ref = load_file(str(REF / "reference.safetensors"))
latent = ref["node.13.KSampler.out.0.0.samples"].cuda()
want = ref["node.14.VAEDecode.out.0.0"][0]
with safe_open(str(ROOT / "plugins/dsh-comfyui/models/vae/qwen_image_vae.safetensors"), "pt", device="cuda") as f:
    base = {k: f.get_tensor(k) for k in f.keys() if k.startswith(("decoder.", "conv2."))}

with torch.inference_mode():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        weights = {k: v.to(dtype) for k, v in base.items()}
        got = torch.ops.dsh_anima_gen.vae_decode(weights, latent)
        torch.cuda.synchronize()
        started = time.perf_counter()
        got = torch.ops.dsh_anima_gen.vae_decode(weights, latent)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - started) * 1000
        got = got.cpu()
        diff = (got - want).abs()
        print(f"{str(dtype):15s} shape {tuple(got.shape)} bit-identical {torch.equal(got, want)}, max abs {diff.max().item():.3e}, "
              f"mean abs {diff.mean().item():.3e}, 8-bit pixels differing {((got * 255).round() != (want * 255).round()).sum().item()}, {ms:.0f} ms")
