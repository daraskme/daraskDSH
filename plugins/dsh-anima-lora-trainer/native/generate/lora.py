"""Anima LoRA merge as ComfyUI's LoraLoaderModelOnly does it (comfy/weight_adapter/lora.py, GPL-3.0).

Keys: "diffusion_model.<path>.lora_A.weight" (down) / ".lora_B.weight" (up), optional ".alpha".
weight + ((strength * alpha / rank) * (up @ down)).to(weight dtype), with alpha / rank = 1 when there is no alpha.
The base weights are never modified: the result is a new dict sharing every untouched tensor.
"""
import torch
from safetensors import safe_open


def load(path, device="cuda"):
    with safe_open(str(path), "pt", device=device) as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def merge(base, loras, intermediate=torch.bfloat16):
    """base: DiT weights ("net.*"); loras: [(tensors, strength)]. Unknown or unpaired keys raise."""
    merged = dict(base)
    for tensors, strength in loras:
        stems = {k[:-len(".lora_A.weight")] for k in tensors if k.endswith(".lora_A.weight")}
        extra = set(tensors) - {s + suffix for s in stems for suffix in (".lora_A.weight", ".lora_B.weight", ".alpha")}
        if extra:
            raise ValueError(f"unsupported LoRA keys: {sorted(extra)[:5]}")
        for stem in sorted(stems):
            if not stem.startswith("diffusion_model.") or stem + ".lora_B.weight" not in tensors:
                raise ValueError(f"unpaired or non-DiT LoRA key: {stem}")
            name = "net." + stem[len("diffusion_model."):] + ".weight"
            if name not in base:
                raise ValueError(f"LoRA targets a missing weight: {name}")
            down, up = tensors[stem + ".lora_A.weight"], tensors[stem + ".lora_B.weight"]
            alpha = tensors[stem + ".alpha"].item() / down.shape[0] if stem + ".alpha" in tensors else 1.0
            weight = merged[name]
            diff = torch.mm(up.to(weight.device, intermediate).flatten(1), down.to(weight.device, intermediate).flatten(1))
            merged[name] = weight + ((strength * alpha) * diff.reshape(weight.shape)).type(weight.dtype)
    return merged
