"""H3 DiT parity (dense attention) against ComfyUI with the BlockSparseAttention node removed.

Run with ComfyUI's venv:  comfy-venv/bin/python test_dit_parity.py [DENSE_REFERENCE_DIR]
Order (Astra review): one block -> full forward (video and audio outputs) at steps 0, 1 and 7.
"""
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/h3-native/reference-small-dense"
BUILD = ROOT / ".runtime/h3-native/build"
DIT = ROOT / "plugins/dsh-comfyui/models/diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors"
LORA = ROOT / "plugins/dsh-comfyui/loras/H3_VBVR_Pro-V2.safetensors"
SHIFT_V, SHIFT_A = 10.0, 3.0  # MiniMaxH3SigmaShift in the reference graph


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"{name}: bit-identical {torch.equal(got, ref)}, relative L2 {rel:.3e}, max abs {(got - ref).abs().max().item():.3e}")


CU13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
TOOLCHAIN = ROOT / ".runtime/h3-native/toolchain/nvidia/cu13"
cpp_extension.load(name="dsh_h3_dit", sources=[str(HERE / f) for f in ("te.cpp", "condition.cpp", "dit.cpp", "kitchen/kitchen_ops.cpp")],
                   is_python_module=False, with_cuda=False, build_directory=str(BUILD), extra_cflags=["-O3"],
                   extra_include_paths=[str(CU13 / "include"), str(TOOLCHAIN / "include")],
                   extra_ldflags=[str(BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{BUILD}",
                                  f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"], verbose=False)
ref = load_file(str(REF / "reference.safetensors"), device="cuda")
with safe_open(str(DIT), "pt", device="cuda") as f:
    weights = {k: f.get_tensor(k) for k in f.keys()}
count = torch.ops.dsh_h3.merge_loras(weights, load_file(str(LORA), device="cuda"), 1.0)
print(f"LoRA merged into {count} projections")

segments = [v for seg in json.loads((REF / "block0_segments.json").read_text()) for v in seg]
out = torch.ops.dsh_h3.dit_block(weights, 0, ref["block0.x"], ref["block0.t_emb"], segments, ref["block0.rope_freqs"])
report("block 0", out, ref["block0.out"])

for call in (0, 1, 7):
    p = f"dit.call{call}"
    torch.cuda.synchronize(); start = time.perf_counter()
    video, audio = torch.ops.dsh_h3.dit_forward(weights, ref[p + ".x.0"], ref[p + ".x.1"], ref[p + ".timestep"],
                                                ref[p + ".context"], ref[p + ".payload.text_token_tags"],
                                                SHIFT_V, SHIFT_A, SHIFT_V / SHIFT_A)
    torch.cuda.synchronize()
    print(f"step {call} ({(time.perf_counter() - start) * 1000:.0f} ms, sigma {ref[p + '.timestep'].item() / 1000:.4f})")
    report("  video", video, ref[p + ".out.0"])
    report("  audio", audio, ref[p + ".out.1"])

if __import__("os").environ.get("DSH_PROFILE") == "1":
    from torch.profiler import ProfilerActivity, profile
    p = "dit.call1"
    args = (weights, ref[p + ".x.0"], ref[p + ".x.1"], ref[p + ".timestep"], ref[p + ".context"],
            ref[p + ".payload.text_token_tags"], SHIFT_V, SHIFT_A, SHIFT_V / SHIFT_A)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        torch.ops.dsh_h3.dit_forward(*args)
        torch.cuda.synchronize()
    averages = prof.key_averages()
    gpu = sorted(((e.self_device_time_total / 1000, e.count, e.key) for e in averages if e.self_device_time_total > 0), reverse=True)
    print(f"profile step 1: GPU {sum(r[0] for r in gpu):.0f} ms")
    for ms, n, key in gpu[:8]:
        print(f"  GPU {ms:8.1f} ms {n:6d}x {key[:96]}")
    for ms, n, key in sorted(((e.self_cpu_time_total / 1000, e.count, e.key) for e in averages), reverse=True)[:6]:
        print(f"  CPU {ms:8.1f} ms {n:6d}x {key[:80]}")
