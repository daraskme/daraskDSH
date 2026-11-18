"""Anima sampling (generate/sampler.py + dit.cpp) vs the ComfyUI capture: sigmas, noise, CFG denoised, ER-SDE updates.

Run with ComfyUI's venv:  comfy-venv/bin/python generate_sampler_parity.py [REFERENCE_DIR]
1. Injected: the captured x / sigma of steps 0, 1, 2, 33 -> CFG denoised; captured denoised -> the next x.
2. End to end from the seed with the captured c_crossattn: solver output and the KSampler latent.
"""
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[4]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/anima-native/generate-reference"
NATIVE = ROOT / "plugins/dsh-anima-lora-trainer/native"
BUILD = ROOT / ".runtime/h3-native/build"
sys.path.insert(0, str(NATIVE / "generate"))
import sampler  # noqa: E402

CU13 = Path(torch.__file__).resolve().parents[1] / "nvidia/cu13"
cpp_extension.load(name="dsh_anima_gen_dit", sources=[str(NATIVE / "generate/dit.cpp"),
                   str(ROOT / "plugins/dsh-minimax-h3/native/kitchen/kitchen_ops.cpp")],
                   is_python_module=False, with_cuda=False, build_directory=str(BUILD), extra_cflags=["-O3"],
                   extra_include_paths=[str(CU13 / "include"), str(ROOT / ".runtime/h3-native/toolchain/nvidia/cu13/include")],
                   extra_ldflags=[str(BUILD / "libdsh_kitchen.so"), f"-Wl,-rpath,{BUILD}",
                                  f"-L{Path(torch.__file__).parent / 'lib'}", "-lc10_cuda"], verbose=False)
with safe_open(str(ROOT / "plugins/dsh-comfyui/models/diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
    weights = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("net.llm_adapter.")}
ref = load_file(str(REF / "reference.safetensors"))
seed = int((REF / "seed.txt").read_text())


def report(name, got, want):
    got, want = got.float().cpu(), want.float().cpu()
    rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    print(f"  {name}: bit-identical {torch.equal(got, want)}, relative L2 {rel:.3e}, max abs {(got - want).abs().max().item():.3e}")


context = ref["dit.call0.context"].cuda()
# which batch row is the positive prompt: the one equal to the longer (non-empty) c_crossattn
positive = max((0, 1), key=lambda i: ref[f"crossattn{i}.in.ids"].shape[1])
row = next(r for r in (0, 1) if torch.equal(ref["dit.call0.context"][r], ref[f"crossattn{positive}.out"][0]))
cond, uncond = context[row:row + 1], context[1 - row:2 - row]
print(f"batch row of the positive prompt: {row}")

print("schedule and noise")
sigmas = sampler.simple_sigmas(34)
report("simple sigmas (35)", sigmas, ref["er_sde.sigmas"])
report("initial noise (CPU)", sampler.initial_noise(seed, 1216, 832), ref["noise.initial"])
gen = torch.Generator(device="cuda")
gen.manual_seed(seed)
for i in range(3):
    report(f"er_sde noise {i}", torch.randn(ref["er_sde.x"].shape, generator=gen, device="cuda"), ref[f"er_sde.noise{i}"])

denoise = sampler.cfg_denoiser(weights, cond, uncond, 4.5)
print("injected steps")
with torch.inference_mode():
    for i in (0, 1, 2, 33):
        got = denoise(ref[f"er_sde.step{i}.x"].cuda(), ref[f"er_sde.step{i}.sigma"].cuda())
        report(f"step {i} CFG denoised", got, ref[f"er_sde.step{i}.denoised"])
    # replay the solver on the captured denoised values to check the updates 0->1 and 1->2
    captured = {i: ref[f"er_sde.step{i}.denoised"].cuda() for i in (0, 1)}
    seen = {}

    def replay(x, sigma):
        i = len(seen)
        seen[i] = x.clone()
        if i in captured:
            return captured[i]
        raise StopIteration
    try:
        sampler.er_sde(replay, ref["er_sde.x"].cuda(), ref["er_sde.sigmas"].cuda(), seed)
    except StopIteration:
        pass
    report("ER-SDE update 0->1", seen[1], ref["er_sde.step1.x"])
    report("ER-SDE update 1->2", seen[2], ref["er_sde.step2.x"])

    print("end to end from the seed")
    torch.cuda.synchronize()
    started = time.perf_counter()
    steps = {}
    latent = sampler.sample(weights, cond, uncond, seed,
                            on_step=lambda i, x, d: steps.update({i: (x.clone(), d.clone())}) if i in (0, 1, 2, 33) else None)
    torch.cuda.synchronize()
    print(f"  34 steps: {time.perf_counter() - started:.2f} s")
    for i in (0, 1, 2, 33):
        report(f"step {i} x", steps[i][0], ref[f"er_sde.step{i}.x"])
    report("solver output (= step 33 denoised)", steps[33][1], ref["er_sde.out"])
    report("KSampler latent", latent, ref["node.13.KSampler.out.0.0.samples"])
