"""Anima LoRA merge (generate/lora.py) vs ComfyUI's LoraLoaderModelOnly: @darask at strength 0.8.

Run with ComfyUI's venv:  comfy-venv/bin/python generate_lora_parity.py
Checks the first DiT call of the LoRA capture (none -> with -> none on the same resident base weights).
"""
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[4]
REF = ROOT / ".runtime/anima-native/generate-reference-lora"
PLAIN = ROOT / ".runtime/anima-native/generate-reference"
NATIVE = ROOT / "plugins/dsh-anima-lora-trainer/native"
BUILD = ROOT / ".runtime/h3-native/build"
MODELS = ROOT / "plugins/dsh-comfyui/models"
sys.path.insert(0, str(NATIVE / "generate"))
import lora  # noqa: E402

CU13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
cpp_extension.load(name="dsh_anima_gen_dit", sources=[str(NATIVE / "generate/dit.cpp"),
                   str(ROOT / "plugins/dsh-minimax-h3/native/kitchen/kitchen_ops.cpp")],
                   is_python_module=False, with_cuda=False, build_directory=str(BUILD), extra_cflags=["-O3"],
                   extra_include_paths=[str(CU13 / "include"), str(ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/include")],
                   extra_ldflags=[str(BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{BUILD}",
                                  f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"], verbose=False)
with safe_open(str(MODELS / "diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
    base = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("net.llm_adapter.")}
ref, plain = load_file(str(REF / "reference.safetensors")), load_file(str(PLAIN / "reference.safetensors"))
tensors = lora.load(MODELS / "loras/trained/@darask.safetensors")


def run(weights, r):
    return torch.ops.dsh_anima_gen.dit(weights, r["dit.call0.x"].cuda(), r["dit.call0.timesteps"].cuda(),
                                       r["dit.call0.context"].cuda()).cpu()


def report(name, got, want):
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    print(f"{name}: bit-identical {torch.equal(got, want)}, relative L2 {rel:.3e}")


with torch.inference_mode():
    report("none (plain capture)", run(base, plain), plain["dit.call0.out"])
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        merged = lora.merge(base, [(tensors, 0.8)], intermediate=dtype)
        report(f"with @darask 0.8 (intermediate {name})", run(merged, ref), ref["dit.call0.out"])
        del merged
    report("none again (base untouched)", run(base, plain), plain["dit.call0.out"])
    report("LoRA effect (with vs plain capture)", ref["dit.call0.out"], plain["dit.call0.out"])
