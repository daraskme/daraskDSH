"""H3 video and audio VAE decode (vae.py, audio.py) vs the ComfyUI capture, from the captured final AV latent.

Run with ComfyUI's venv:  comfy-venv/bin/python test_vae_parity.py [REFERENCE_DIR]
Compares the uncompressed RGB [frames, H, W, 3] and the loudness-normalized 32 kHz stereo waveform.
"""
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audio  # noqa: E402
import build  # noqa: E402
import vae  # noqa: E402

ROOT = HERE.parents[2]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/h3-native/reference-small"
MODELS = ROOT / "models/vae"


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"{name}: bit-identical {torch.equal(got, ref)}, relative L2 {rel:.3e}, max abs {(got - ref).abs().max().item():.3e}")


def load(path, keep):
    with safe_open(str(path), "pt", device="cuda") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.startswith(keep)}


build.build()
ref = load_file(str(REF / "reference.safetensors"))
video_w = {k: v.half() for k, v in load(MODELS / "minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors",
                                        ("decoder.", "post_quant_conv.", "latents_")).items()}
audio_w = load(MODELS / "minimax_h3_audio_vae_fp32.safetensors", ("decoder.", "dec_in_proj.", "latents_"))
with torch.inference_mode():
    for _ in range(2):
        torch.cuda.synchronize(); start = time.perf_counter()
        rgb = vae.decode_video(video_w, ref["node.10.VAEDecode.in.samples.0.samples.0"].cuda())
        torch.cuda.synchronize(); mid = time.perf_counter()
        wave = audio.decode(audio_w, ref["node.23.VAEDecodeAudio.in.samples.0.samples.1"].cuda())
        torch.cuda.synchronize()
    print(f"video {mid - start:.2f} s, audio {time.perf_counter() - mid:.2f} s")
report("RGB", rgb, ref["node.10.VAEDecode.out.0.0"])
report("waveform", wave, ref["node.23.VAEDecodeAudio.out.0.0.waveform"])
