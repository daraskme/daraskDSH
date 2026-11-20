"""Apply LoRA files to the DiT weight dict for generation: BF16 weights take the merge (condition.cpp merge_loras,
kohya or diffusion-pipe / PEFT keys); int8 convrot weights (P-H3 blocks) get a run-time branch "<path>.lora_a" [r, in] /
"<path>.lora_b" [out, r] (scale folded into b, several LoRAs concatenated along r) that dit.cpp's proj adds."""
import torch


def apply(weights, lora, strength):
    """-> number of projections the LoRA touched."""
    count = torch.ops.dsh_h3.merge_loras(weights, lora, float(strength))
    for name in [k[:-len(".weight_scale")] for k in weights if k.endswith(".weight_scale")]:
        kohya = "lora_unet_" + name.replace(".", "_")
        if kohya + ".lora_up.weight" in lora:
            up, down = lora[kohya + ".lora_up.weight"], lora[kohya + ".lora_down.weight"]
            scale = float(lora[kohya + ".alpha"]) / down.shape[0]
        elif f"diffusion_model.{name}.lora_B.weight" in lora:
            up, down = lora[f"diffusion_model.{name}.lora_B.weight"], lora[f"diffusion_model.{name}.lora_A.weight"]
            scale = 1.0
        else:
            continue
        a = down.to(torch.bfloat16)
        b = (up.float() * (scale * float(strength))).to(torch.bfloat16)
        if name + ".lora_a" in weights:
            a = torch.cat([weights[name + ".lora_a"], a])
            b = torch.cat([weights[name + ".lora_b"], b], dim=1)
        weights[name + ".lora_a"], weights[name + ".lora_b"] = a.contiguous(), b.contiguous()
        count += 1
    return count
