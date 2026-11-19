"""Qwen3-VL vision tower against ComfyUI's captured calls: tokenizer imageN -> visionN (patches, merged, deepstack)."""
import sys
from pathlib import Path
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vision

ROOT = HERE.parents[2]
TE = ROOT / "plugins/dsh-comfyui/models/text_encoders"
CASES = {"cond-h3-i2v": ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "visual.", "qwen3vl_32b"),
         "cond-h3-ref": ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "visual.", "qwen3vl_32b"),
         "cond-ph3-editref": ("p-h3-image_text_encoder_int8_convrot.safetensors", "model.visual.", "qwen3vl_4b")}


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


loaded = {}
with torch.inference_mode():
    for case, (name, prefix, encoder) in CASES.items():
        if name not in loaded:
            with safe_open(str(TE / name), "pt", device="cuda") as f:
                loaded[name] = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        w = loaded[name]
        with safe_open(str(ROOT / ".runtime/h3-native" / case / "reference.safetensors"), "pt", device="cuda") as f:
            i = 0
            while f"vision{i}.pixels" in f.keys():
                image = f.get_tensor(f"tokenizer.{encoder}.image{i}")[0]
                x, grid = vision.patches(image)
                merged, deep = vision.encode(w, prefix, image)
                print(f"{case} image{i} grid {grid} patches {rel(x, f.get_tensor(f'vision{i}.pixels')):.1e} "
                      f"merged {rel(merged, f.get_tensor(f'vision{i}.out.0')):.1e} "
                      f"deepstack {[f'{rel(d, f.get_tensor(f'vision{i}.out.1.{j}')):.1e}' for j, d in enumerate(deep)]}")
                i += 1
