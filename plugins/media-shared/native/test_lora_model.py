"""lora_model.py against the inference ops (dit.cpp dense forward, condition.cpp refine_text) and FP32 gradients.

Run with ComfyUI's venv (ninja on PATH):  comfy-venv/bin/python test_lora_model.py h3|ph3
1. refiner: refine(w, {}, states) vs refine_text;  2. single blocks vs dit_block, and the whole forward vs dit_forward
   (dense, audio_scale 1; end to end only loosely, 50 BF16 blocks amplify rounding);
3. a random LoRA (refiner + blocks) vs the inference ops with it merged (H3) or attached at run time (P-H3, loras.py);
4. LoRA gradients of the first 2 blocks in BF16 vs the same model in FP32 (H3), or the int8 W8A8 path vs the
   dequantized BF16 weights (P-H3).
"""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import lora_model as M  # noqa: E402
import train_lora  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "h3"


def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).norm() / b.norm()).item()


def check(name, value, limit):
    print(f"{name}: {value:.3e} (limit {limit:.0e})", "ok" if value <= limit else "FAIL")
    assert value <= limit, name


build.build()
torch.manual_seed(0)
settings = train_lora.H3 if MODEL == "h3" else train_lora.PH3
sv, sa = settings["shift_v"], settings["shift_a"]
w = train_lora.load_dit(MODEL)
states = torch.randn(23, 5120, device="cuda") * 3
tags = torch.ones(23, dtype=torch.long)
video = torch.randn(1, 24, 2 if MODEL == "h3" else 1, 16, 24, device="cuda").bfloat16()
audio = torch.randn(1, 32, 2, 7 if MODEL == "h3" else 2, device="cuda").bfloat16()
sigma = 0.62

def dit_block_inputs(inputs):
    """t_emb, flat segments and the BF16 rope table for torch.ops.dsh_h3.dit_block."""
    c, s_ = inputs["cos"][:, 0], inputs["sin"][:, 0]
    table = torch.stack([c, -s_, s_, c], -1).reshape(1, -1, 1, 48, 2, 2).bfloat16()
    t_table = w["adaln_t_table"].float()
    sig = torch.tensor(sigma)
    unique_t = torch.tensor(sorted({float(1 - sig), float(1 - M.time_shift_sigma(sig, sv, sa))}), device="cuda")
    pos = unique_t.clamp(0, 1) * (t_table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=t_table.shape[0] - 2)
    t_emb = torch.lerp(t_table[i0], t_table[i0 + 1], (pos - i0)[:, None])
    return t_emb, [v for seg in inputs["segments"] for v in seg], table


with torch.no_grad():
    context = torch.ops.dsh_h3.refine_text(w, states)
    check("refiner vs refine_text", rel(M.refine(w, {}, states), context), 1e-3)
    inputs = M.prepare(w, video, audio, sigma, context[None], tags, sv, sa)
    x = torch.cat([context, inputs["h"][context.shape[0]:]])
    t_emb, flat, table = dit_block_inputs(inputs)
    probe = [0, 1, 25, M.block_count(w) - 1]
    for i in probe:  # one block from the same input: BF16 rounding only
        want = torch.ops.dsh_h3.dit_block(w, i, x, t_emb, flat, table)
        got = M.block(w, {}, i, x, inputs["mods"][i], inputs["segments"], inputs["cos"], inputs["sin"])
        # int8 W8A8 (P-H3) turns 1-ulp input differences into whole quantization steps (W8A8 itself is ~1% from BF16)
        check(f"block {i} vs dit_block", rel(got - x, want - x), 2e-3 if MODEL == "h3" else 1e-2)
    # end to end: 50 BF16 blocks amplify rounding (the reference itself moves ~10% for 1e-3 input noise on random data)
    want = torch.ops.dsh_h3.dit_forward(w, video, audio, torch.tensor([sigma * 1000], device="cuda"), context[None], tags,
                                        sv, sa, 1.0, [], False, None, None)
    got = M.forward(w, {}, inputs, states)
    check("forward video vs dit_forward (end to end)", rel(got[0], want[0]), 0.15)
    check("forward audio vs dit_forward (end to end)", rel(got[1], want[1]), 0.15)

    # a random LoRA on the refiner and some blocks vs the same LoRA merged into the BF16 weights
    lora = {}
    names = ["token_refiner.blocks.0.attn.qkv_proj", "token_refiner.blocks.1.mlp.fc2", "blocks.0.attn.qkv_proj",
             "blocks.1.attn.out_proj", "blocks.25.mlp.fc1", f"blocks.{M.block_count(w) - 1}.mlp.fc2"]
    for name in names:
        out_f, in_f = w[name + ".weight"].shape
        lora[name] = (torch.randn(8, in_f, device="cuda") * 0.1, torch.randn(out_f, 8, device="cuda") * 0.1)
    # the refiner's BF16 output rounding swamps a LoRA's change (both the branch and the merge are ~15% off FP32), so the
    # refiner LoRA math is checked in FP32: the branch vs refine_text on FP32 weights with the update merged exactly
    import loras
    w32 = {k: v.float() for k, v in w.items() if k.startswith(("condition_proj.", "token_refiner."))}
    refined = M.refine(w32, lora, states)
    for name, (a, b) in lora.items():
        if name in w32 or name + ".weight" in w32:
            w32[name + ".weight"] = w32[name + ".weight"] + b @ a
    check("LoRA refiner vs merged refine_text (FP32)", rel(refined, torch.ops.dsh_h3.refine_text(w32, states)), 1e-4)
    del w32
    if MODEL == "h3":  # BF16 merging rounds the update away too: block LoRA math in FP32, branch vs exact merge
        for i in (0, 1):
            w32 = {k: v.float() for k, v in w.items() if k.startswith(f"blocks.{i}.")}
            got = M.block(w32, lora, i, x.float(), inputs["mods"][i], inputs["segments"], inputs["cos"], inputs["sin"])
            for name, (a, b) in lora.items():
                if name + ".weight" in w32:
                    w32[name + ".weight"] = w32[name + ".weight"] + b @ a
            want = M.block(w32, {}, i, x.float(), inputs["mods"][i], inputs["segments"], inputs["cos"], inputs["sin"])
            check(f"LoRA block {i} vs merged (FP32)", rel(got - x.float(), want - x.float()), 1e-4)
        # and the generation side takes the PEFT keys
        peft = {f"diffusion_model.{name}.lora_{ab}.weight": t for name, pair in lora.items() for ab, t in zip("AB", pair)}
        check("PEFT LoRA projections merged", abs(loras.apply(w, peft, 1.0) - len(lora)), 0)
    else:  # P-H3: the int8 blocks' run-time LoRA (loras.py, dit.cpp proj): the LoRA output matches within the W8A8 noise
        base = [M.block(w, {}, i, x, inputs["mods"][i], inputs["segments"], inputs["cos"], inputs["sin"]) for i in probe]
        got = [M.block(w, lora, i, x, inputs["mods"][i], inputs["segments"], inputs["cos"], inputs["sin"]) for i in probe]
        peft = {f"diffusion_model.{name}.lora_{ab}.weight": t for name, pair in lora.items() for ab, t in zip("AB", pair)}
        loras.apply(w, peft, 1.0)
        for i, g, b0 in zip(probe, got, base):
            want = torch.ops.dsh_h3.dit_block(w, i, x, t_emb, flat, table)
            effect = rel(g - x, b0 - x)
            print(f"  block {i}: the LoRA moves the block output by {effect:.1%}")
            check(f"LoRA block {i} vs run-time LoRA dit_block", rel(g - x, want - x), 1e-2)

# gradients on a 2-block model
small = {k: v for k, v in w.items() if (not k.startswith("blocks.") or k.split(".")[1] in ("0", "1")) and "lora_" not in k}
del w
torch.cuda.empty_cache()
lora = M.lora_parameters(small, 4, seed=1)
with torch.no_grad():
    for a, b in lora.values():
        b.normal_(0, 0.01)
if MODEL == "h3":
    reference = {k: v.float() if v.is_floating_point() else v for k, v in small.items()}
else:  # the int8 projections as the BF16 weights they stand for
    reference = dict(small)
    for k in [k for k in small if k.endswith(".weight_scale")]:
        name = k[:-len("_scale")]
        reference.pop(k)
        reference[name] = torch.ops.dsh_kitchen.dequantize_int8_convrot(small[name], small[k])


def grads(weights, dtype):
    for a, b in lora.values():
        a.grad = b.grad = None
    inputs = M.prepare(weights, video.to(dtype), audio.to(dtype), sigma, torch.empty(1, 23, 0, dtype=dtype), tags, sv, sa)
    if dtype == torch.float32:
        inputs["h"] = inputs["h"].float()
    v, a = M.forward(weights, lora, inputs, states.to(dtype))
    (v.square().mean() + a.square().mean()).backward()
    return torch.cat([p.grad.flatten() for pair in lora.values() for p in pair])


ref_grad = grads(reference, torch.float32 if MODEL == "h3" else torch.bfloat16)
got_grad = grads(small, torch.bfloat16)
cos = F.cosine_similarity(got_grad.float(), ref_grad.float(), dim=0).item()
check("LoRA gradient 1 - cosine", 1 - cos, 1e-2)
check("LoRA gradient norm ratio - 1", abs(got_grad.norm().item() / ref_grad.norm().item() - 1), 5e-2)
