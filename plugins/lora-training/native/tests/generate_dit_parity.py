"""Anima generation DiT (generate/dit.cpp) vs ComfyUI's captured forward calls (CFG batch 2: cond + uncond).

Run with ComfyUI's venv:  comfy-venv/bin/python generate_dit_parity.py [REFERENCE_DIR]
Feeds each captured call's x / timesteps / context (dit.call0-2 and the last call, which reuses call0's
context) and compares the velocity prediction. DSH_PROFILE=1 also times one forward.
"""
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[4]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/anima-native/generate-reference"
NATIVE = ROOT / "plugins/lora-training/native"
MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT / "plugins/image-generation/native/anima"))
import generate  # noqa: E402

generate.build()  # the C++ ops and FP8 kernels, shared with the runner
with safe_open(str(MODELS / "diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
    weights = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("net.llm_adapter.")}
ref = load_file(str(REF / "reference.safetensors"))


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


with torch.inference_mode():
    for call in ("call0", "call1", "call2", "last"):
        x = ref[f"dit.{call}.x"].cuda()
        t = ref[f"dit.{call}.timesteps"].cuda()
        context = ref[f"dit.{call}.context" if call != "last" else "dit.call0.context"].cuda()
        out = torch.ops.dsh_anima_gen.dit(weights, x, t, context).cpu()
        want = ref[f"dit.{call}.out"]
        same = [torch.equal(out[i], want[i]) for i in range(out.shape[0])]
        print(f"{call}: sigma {t.tolist()} bit-identical cond/uncond {same}, relative L2 {rel(out, want):.3e}, "
              f"max abs {(out.float() - want.float()).abs().max().item():.3e}")
    if os.environ.get("DSH_PROFILE") == "1":
        for _ in range(2):
            torch.ops.dsh_anima_gen.dit(weights, x, t, context)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            torch.ops.dsh_anima_gen.dit(weights, x, t, context)
        torch.cuda.synchronize()
        print(f"forward (batch 2): {(time.perf_counter() - started) / 5 * 1000:.1f} ms")
