"""H3 video (34 frames) and audio VAE encoders (encode.py) vs ComfyUI's MiniMaxH3VideoVAE / MiniMaxH3AudioVAE encode.

Run with ComfyUI's venv:  comfy-venv/bin/python test_encode_av_parity.py VIDEO.mp4
"""
from pathlib import Path
import sys

import av
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / ".runtime/ComfyUI"))
import encode  # noqa: E402
import comfy.sd  # noqa: E402

MODELS = ROOT / "plugins/dsh-comfyui/models/vae"
VIDEO = MODELS / "minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors"
AUDIO = MODELS / "minimax_h3_audio_vae_fp32.safetensors"


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    print(f"{name}: shape ok, relative L2 {((got - ref).norm() / ref.norm()).item():.3e}, "
          f"max abs {(got - ref).abs().max().item():.3e}")


def read(path, frames, size):
    with av.open(str(path)) as c:
        rgb = [f.to_ndarray(format="rgb24", width=size[0], height=size[1]) for _, f in zip(range(frames), c.decode(video=0))]
    with av.open(str(path)) as c:
        wave = np.concatenate([f.to_ndarray() for f in c.decode(audio=0)], axis=1)
    return torch.from_numpy(np.stack(rgb)).float() / 255.0, torch.from_numpy(wave[:, :int(frames / 24 * 32000)]).float()


rgb, wave = read(sys.argv[1], 34, (512, 288))
with torch.inference_mode():
    vw = load_file(str(VIDEO), device="cuda")
    ref_vae = comfy.sd.VAE(sd=vw)
    ref_vae.first_stage_model.to("cuda")
    want = ref_vae.first_stage_model.encode(rgb.permute(3, 0, 1, 2).unsqueeze(0).cuda().half() * 2 - 1)
    with safe_open(str(VIDEO), "pt", device="cuda") as f:
        w = {k: f.get_tensor(k).half() if k.startswith(("encoder.", "quant_conv.")) else f.get_tensor(k)
             for k in f.keys() if k.startswith(("encoder.", "quant_conv.", "latents_"))}
    report(f"video {tuple(want.shape)}", encode.encode_video(w, rgb), want)
    report("image (1 frame)", encode.encode_video(w, rgb[:1]),
           ref_vae.first_stage_model.encode(rgb[:1].permute(3, 0, 1, 2).unsqueeze(0).cuda().half() * 2 - 1))
    del ref_vae, vw
    aw = load_file(str(AUDIO), device="cuda")
    ref_audio = comfy.sd.VAE(sd=aw)
    ref_audio.first_stage_model.to("cuda")
    want = ref_audio.first_stage_model.encode(wave.unsqueeze(0).cuda())
    report(f"audio {tuple(want.shape)}", encode.encode_audio(aw, wave), want)
