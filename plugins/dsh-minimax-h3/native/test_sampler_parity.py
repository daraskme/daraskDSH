"""H3 sampling (sampler.py + dit.cpp with SLA) vs the ComfyUI capture: sigmas, noise, the DiT inputs of steps 1
and 7 and the final AV latent, end to end from the seed with the captured conditioning.

Run with ComfyUI's venv:  comfy-venv/bin/python test_sampler_parity.py [REFERENCE_DIR]
"""
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import sampler  # noqa: E402

ROOT = HERE.parents[2]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/h3-native/reference-small"
DIT = ROOT / "plugins/dsh-comfyui/models/diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors"
LORA = ROOT / "plugins/dsh-comfyui/loras/H3_VBVR_Pro-V2.safetensors"


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"{name}: bit-identical {torch.equal(got, ref)}, relative L2 {rel:.3e}, max abs {(got - ref).abs().max().item():.3e}")


build.build()
ref = load_file(str(REF / "reference.safetensors"), device="cuda")
report("simple sigmas (9)", sampler.simple_sigmas(8, 10.0), ref["node.9.BasicScheduler.out.0.0"])
video_shape, audio_shape = sampler.latent_shapes(640, 384, 39)
noise = sampler.initial_noise(42, video_shape, audio_shape)
report("noise video (= step 0 DiT input)", noise[0].to(torch.bfloat16), ref["dit.call0.x.0"])
report("noise audio", noise[1].to(torch.bfloat16), ref["dit.call0.x.1"])

with safe_open(str(DIT), "pt", device="cuda") as f:
    weights = {k: f.get_tensor(k) for k in f.keys()}
torch.ops.dsh_h3.merge_loras(weights, load_file(str(LORA), device="cuda"), 1.0)
steps = {}
with torch.inference_mode():
    torch.cuda.synchronize(); start = time.perf_counter()
    latent = sampler.sample(weights, ref["dit.call0.context"], ref["dit.call0.payload.text_token_tags"], 42, 640, 384, 39,
                            on_step=lambda i, x, d: steps.update({i: [t.clone() for t in x]}))
    torch.cuda.synchronize()
print(f"8 steps: {time.perf_counter() - start:.2f} s")
for i in (1, 7):
    report(f"step {i} DiT input video", steps[i][0].to(torch.bfloat16), ref[f"dit.call{i}.x.0"])
    report(f"step {i} DiT input audio", steps[i][1].to(torch.bfloat16), ref[f"dit.call{i}.x.1"])
report("final latent video", latent[0], ref["node.14.SamplerCustomAdvanced.out.0.0.samples.0"])
report("final latent audio", latent[1], ref["node.14.SamplerCustomAdvanced.out.0.0.samples.1"])
