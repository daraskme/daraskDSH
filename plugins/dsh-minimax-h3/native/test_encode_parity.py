"""VAE encoder (one frame) against ComfyUI's captured encodes: vae_encodeN.in -> vae_encodeN.out."""
import sys
from pathlib import Path
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vae

ROOT = HERE.parents[2]
MODELS = ROOT / "plugins/dsh-comfyui/models/vae"
CASES = {"cond-h3-i2v": ("minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors", False),
         "cond-h3-ref": ("minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors", False),
         "cond-ph3-edit": ("p-h3-image_vae_int8_convrot.safetensors", True),
         "cond-ph3-editref": ("p-h3-image_vae_int8_convrot.safetensors", True)}

weights = {}
with torch.inference_mode():
    for case, (name, sample) in CASES.items():
        if name not in weights:
            with safe_open(str(MODELS / name), "pt", device="cuda") as f:
                weights[name] = {k: f.get_tensor(k).half() if k.startswith(("encoder.", "quant_conv.")) else f.get_tensor(k)
                                 for k in f.keys() if k.startswith(("encoder.", "quant_conv.", "latents_"))}
        with safe_open(str(ROOT / ".runtime/h3-native" / case / "reference.safetensors"), "pt") as f:
            i = 0
            while f"vae_encode{i}.in" in f.keys():
                x = f.get_tensor(f"vae_encode{i}.in")[:, :, 0].cuda()
                want = f.get_tensor(f"vae_encode{i}.out").cuda()
                got = vae.encode_signed(weights[name], x, sample)
                err = ((got - want).norm() / want.norm()).item()
                print(f"{case} encode{i} {tuple(got.shape)} rel {err:.2e} max {(got - want).abs().max().item():.2e}")
                i += 1
