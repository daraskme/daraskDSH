"""Anima VAE encode (generate/vae.cpp vae_encode) vs ComfyUI's VAEEncode on a real picture.

Run with ComfyUI's venv:  comfy-venv/bin/python generate_encode_parity.py [IMAGE]
ComfyUI encodes this VAE in BF16 (its first working dtype).
"""
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "plugins/image-generation/native/anima"))
sys.path.insert(0, str(ROOT / ".runtime/ComfyUI"))
import generate  # noqa: E402

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next((ROOT / "outputs/images/anima").rglob("*.png"))
img = Image.open(image_path).convert("RGB")
img = img.crop((0, 0, img.width // 16 * 16, img.height // 16 * 16))
pixels = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)

generate.build()
with safe_open(str(ROOT / "models/vae/qwen_image_vae.safetensors"), "pt", device="cuda") as f:
    sd = {k: f.get_tensor(k) for k in f.keys()}

import comfy.sd  # noqa: E402
vae = comfy.sd.VAE(sd={k: v.clone() for k, v in sd.items()})
with torch.inference_mode():
    want = vae.encode(pixels[None]).float().cpu()           # [1, 16, 1, h, w]
    weights = {k: v.to(vae.vae_dtype) for k, v in sd.items() if k.startswith(("encoder.", "conv1."))}
    got = torch.ops.dsh_anima_gen.vae_encode(weights, pixels.cuda())
    torch.cuda.synchronize()
    started = time.perf_counter()
    got = torch.ops.dsh_anima_gen.vae_encode(weights, pixels.cuda())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - started) * 1000
    got = got.cpu()
diff = (got - want).abs()
print(f"{image_path.name} {tuple(pixels.shape)} vae dtype {vae.vae_dtype}: shape {tuple(got.shape)} vs {tuple(want.shape)}, "
      f"bit-identical {torch.equal(got, want)}, max abs {diff.max().item():.3e}, mean abs {diff.mean().item():.3e}, {ms:.0f} ms")
