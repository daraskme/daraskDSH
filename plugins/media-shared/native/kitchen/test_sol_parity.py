"""Our libdsh_kitchen.so build vs the comfy-kitchen wheel, through the same sol_attn_chunked driver.

Run with ComfyUI's venv (torch 2.15 + cu132):  comfy-venv/bin/python test_sol_parity.py
Shapes follow H3 T2V at 1280x704x121 (124 frames): 33 text + 414 audio + 32,560 video rows.
"""
from pathlib import Path
import time

import torch
from torch.utils import cpp_extension

import comfy_kitchen.backends.cuda as kitchen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # plugins/media-shared/native/kitchen
BUILD = ROOT / ".runtime/h3-native/build"
CU13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
TOOLCHAIN = ROOT / ".runtime/h3-native/toolchain/nvidia/cu13"  # crt/ headers (nvidia-cuda-crt 13.2)

cpp_extension.load(name="dsh_kitchen_ops", sources=[str(HERE / "kitchen_ops.cpp")], is_python_module=False,
                   with_cuda=False, build_directory=str(BUILD), extra_cflags=["-O3"],
                   extra_include_paths=[str(CU13 / "include"), str(TOOLCHAIN / "include")],
                   extra_ldflags=[str(BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{BUILD}",
                                  f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"], verbose=False)
ops = torch.ops.dsh_kitchen
original = kitchen._C


class Ours:
    """Drop-in for the wheel's _C entry points the chunked driver uses."""
    def sol_attn_plan(self, batch, t, h, token_aug=0):
        names = list(original.sol_attn_plan(batch, t, h, token_aug=token_aug).keys())
        return dict(zip(names, ops.plan(batch, t, h, token_aug)))

    def sol_producer_begin(self, ws, batch, t, h, stream, token_aug=0):
        ops.producer_begin(ws, batch, t, h, token_aug)

    def sol_producer_chunk(self, ws, chunk, fab, qw, kw, km, vsc, rope_eps, rot, t0, m, batch, t, h, stream,
                           block_len=None, token_aug=0):
        ops.producer_chunk(ws, chunk, fab, qw, kw, km, vsc, block_len, rope_eps, rot, t0, m, batch, t, h, token_aug)

    def sol_attn_core(self, ws, out, vscale, kmean_next, vamax, batch, t, h, tau, scale, sb0, sb1, sq0, sq1, stream,
                      threshold=None, block_len=None, tail=True, token_aug=0):
        ops.attn_core(ws, out, vscale, kmean_next, vamax, block_len, tail, batch, t, h, tau, scale, threshold,
                      sb0, sb1, sq0, sq1, token_aug)


def run(backend, chunks, t, h, rope, norms, kmean=None, vscale=None):
    kitchen._C = backend
    wrap = kitchen._wrap_for_dlpack
    if backend is not original:
        kitchen._wrap_for_dlpack = lambda x: x
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = kitchen.sol_attn_chunked(chunks, t, h, rope, norms, kmean=kmean, vscale=vscale, topk_ratio=0.1,
                                          sink_blocks=[0, 7], sink_q=[0, 7])
        torch.cuda.synchronize()
        return result, time.perf_counter() - start
    finally:
        kitchen._C = original
        kitchen._wrap_for_dlpack = wrap


torch.manual_seed(0)
t, h, d, chunk = 33 + 414 + 32560, 56, 128, 4096
qkv = (torch.randn(t, 3 * h * d, device="cuda") * 0.5).to(torch.bfloat16)
chunks = [qkv[i:i + chunk] for i in range(0, t, chunk)]
angle = torch.randn(t, 1, d // 2, device="cuda")
rope = torch.stack([torch.stack([angle.cos(), -angle.sin()], -1), torch.stack([angle.sin(), angle.cos()], -1)], -2)
norms = (torch.rand(d, device="cuda") + 0.5, torch.rand(d, device="cuda") + 0.5)

print(torch.__version__, torch.cuda.get_device_name(), "T", t)
for step in ("step 1 (bootstrap statistics)", "step 2 (previous statistics)"):
    km = None if step.startswith("step 1") else ref[1]
    vs = None if step.startswith("step 1") else ref[2]
    for _ in range(2):  # warm both paths
        run(original, chunks, t, h, rope, norms, km, vs); run(Ours(), chunks, t, h, rope, norms, km, vs)
    (ref, ref_s) = run(original, chunks, t, h, rope, norms, km, vs)
    (got, got_s) = run(Ours(), chunks, t, h, rope, norms, km, vs)
    same = [torch.equal(a, b) for a, b in zip(ref, got)]
    print(f"{step}: out/kmean/vscale bit-identical {same}; wheel {ref_s * 1000:.1f} ms, ours {got_s * 1000:.1f} ms")
