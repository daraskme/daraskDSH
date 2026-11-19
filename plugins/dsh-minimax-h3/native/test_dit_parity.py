"""H3 DiT parity against ComfyUI: dense attention (BlockSparseAttention node removed) and the SLA path.

Run with ComfyUI's venv:  comfy-venv/bin/python test_dit_parity.py [DENSE_REFERENCE_DIR] [SLA_REFERENCE_DIR]
Order (Astra review): one block -> full forward (video and audio outputs) at steps 0, 1 and 7 (dense);
SLA: step 0 bootstraps the per-block statistics, step 1 reuses them (step 7 needs steps 2-6: sampler test).
"""
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

ROOT = HERE.parents[2]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/h3-native/reference-small-dense"
SLA_REF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".runtime/h3-native/reference-small"
DIT = ROOT / "plugins/dsh-comfyui/models/diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors"
LORA = ROOT / "plugins/dsh-comfyui/loras/H3_VBVR_Pro-V2.safetensors"
SHIFT_V, SHIFT_A = 10.0, 3.0  # MiniMaxH3SigmaShift in the reference graph


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"{name}: bit-identical {torch.equal(got, ref)}, relative L2 {rel:.3e}, max abs {(got - ref).abs().max().item():.3e}")


build.build()
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
                                                SHIFT_V, SHIFT_A, SHIFT_V / SHIFT_A, [], False)
    torch.cuda.synchronize()
    print(f"step {call} ({(time.perf_counter() - start) * 1000:.0f} ms, sigma {ref[p + '.timestep'].item() / 1000:.4f})")
    report("  video", video, ref[p + ".out.0"])
    report("  audio", audio, ref[p + ".out.1"])

sla = load_file(str(SLA_REF / "reference.safetensors"), device="cuda")
stats = [torch.zeros(56, 128, device="cuda") for _ in range(100)]
for call in (0, 1):
    p = f"dit.call{call}"
    torch.cuda.synchronize(); start = time.perf_counter()
    video, audio = torch.ops.dsh_h3.dit_forward(weights, sla[p + ".x.0"], sla[p + ".x.1"], sla[p + ".timestep"],
                                                sla[p + ".context"], sla[p + ".payload.text_token_tags"],
                                                SHIFT_V, SHIFT_A, SHIFT_V / SHIFT_A, stats, call == 0)
    torch.cuda.synchronize()
    print(f"SLA step {call} ({(time.perf_counter() - start) * 1000:.0f} ms)")
    report("  video", video, sla[p + ".out.0"])
    report("  audio", audio, sla[p + ".out.1"])

if __import__("os").environ.get("DSH_PROFILE") == "1":
    from torch.profiler import ProfilerActivity, profile
    p = "dit.call1"
    args = (weights, ref[p + ".x.0"], ref[p + ".x.1"], ref[p + ".timestep"], ref[p + ".context"],
            ref[p + ".payload.text_token_tags"], SHIFT_V, SHIFT_A, SHIFT_V / SHIFT_A, stats, False)
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
