"""H3 text conditioning parity against the ComfyUI reference capture.

Run with ComfyUI's venv:  comfy-venv/bin/python test_text_parity.py [REFERENCE_DIR]
Checks (Astra review): token ids exact -> layer-50 unnormalized hidden state (5120)
-> condition_proj + LoRA-merged token refiner (5376).
Also writes the tokenizer.json the native runtime uses.
"""
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils import cpp_extension

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/h3-native/reference-small"
BUILD = ROOT / ".runtime/h3-native/build"
COMFY = ROOT / ".runtime/ComfyUI"
TE = ROOT / "models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
EXTRA = {"<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671, "<|lyrics_start|>": 151672,
         "<|lyrics_end|>": 151673, "<|caption_start|>": 151674, "<|caption_end|>": 151675}


def tokenizer_json():
    out = BUILD / "qwen3vl_tokenizer.json"
    if not out.exists():
        from transformers import Qwen2TokenizerFast
        fast = Qwen2TokenizerFast.from_pretrained(str(COMFY / "comfy/text_encoders/qwen25_tokenizer"))
        fast.add_special_tokens({"additional_special_tokens": list(EXTRA)})
        assert all(fast.convert_tokens_to_ids(k) == v for k, v in EXTRA.items())
        fast.backend_tokenizer.save(str(out))
    return out


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    print(f"{name}: relative L2 {rel:.3e}, cosine {cos:.6f}, max abs {(got - ref).abs().max().item():.3e}")


ref = load_file(str(REF / "reference.safetensors"))
prompt = (REF / "prompt.txt").read_text()

# 1. token ids (exact)
from tokenizers import Tokenizer
ids = Tokenizer.from_file(str(tokenizer_json())).encode(prompt, add_special_tokens=False).ids
expected = ref["tokenizer.qwen3vl_32b.ids"].tolist()
print(f"token ids: {len(ids)} tokens, exact match {ids == expected}")

# 2. layer-50 hidden state
cpp_extension.load(name="dsh_h3_text", sources=[str(HERE / "te.cpp"), str(HERE / "condition.cpp")], is_python_module=False, with_cuda=False,
                   build_directory=str(BUILD), extra_cflags=["-O3"], verbose=False)
weights = {}
with safe_open(str(TE), "pt", device="cuda") as f:
    for key in f.keys():
        if key.startswith("model.") and not key.endswith("comfy_quant"):
            weights[key] = f.get_tensor(key)
token_ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
out = torch.ops.dsh_h3.encode_text(weights, token_ids)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(5):
    out = torch.ops.dsh_h3.encode_text(weights, token_ids)
torch.cuda.synchronize()
print(f"encode: {(time.perf_counter() - start) / 5 * 1000:.1f} ms for {len(ids)} tokens, output {tuple(out.shape)} {out.dtype}")
report("layer-50 hidden vs ComfyUI", out, ref["te.output.0"][0])
got, want = out.float().cpu(), ref["te.output.0"][0].float()
print("per-token |ref|:", [round(v) for v in want.norm(dim=-1).tolist()[:4]], "...")
print("per-token relative error:", [round(v, 4) for v in ((got - want).norm(dim=-1) / want.norm(dim=-1)).tolist()])
report("tokens 1.. (without the first)", got[1:], want[1:])

# 3. condition_proj + LoRA-merged token refiner (exact)
DIT = ROOT / "models/diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors"
LORA = ROOT / "models/loras/H3_VBVR_Pro-V2.safetensors"
del weights
torch.cuda.empty_cache()
with safe_open(str(DIT), "pt", device="cuda") as f:
    dit = {k: f.get_tensor(k) for k in f.keys() if k.startswith(("condition_proj", "token_refiner"))}
lora = load_file(str(LORA), device="cuda")
count = torch.ops.dsh_h3.merge_loras(dit, lora, 1.0)
merged = dit
print(f"LoRA merged into {count} refiner projections")
refined = torch.ops.dsh_h3.refine_text(merged, ref["te.output.0"][0].cuda())
print("refiner output bit-identical to ComfyUI:", torch.equal(refined.cpu(), ref["refiner.output"]))
report("refiner output", refined, ref["refiner.output"])
end_to_end = torch.ops.dsh_h3.refine_text(merged, out)
report("end to end (native Qwen -> refiner) vs ComfyUI", end_to_end, ref["refiner.output"])
