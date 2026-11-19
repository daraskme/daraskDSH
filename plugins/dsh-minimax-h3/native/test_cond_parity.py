"""Picture conditioning against ComfyUI: token ids / tags / MRoPE positions, then the Qwen output (H3: layer 50,
P-H3: the text connector input and the DiT context)."""
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build
import cond

ROOT = HERE.parents[2]
TE = ROOT / "plugins/dsh-comfyui/models/text_encoders"
tok = Tokenizer.from_file(str(ROOT / ".runtime/h3-native/build/qwen3vl_tokenizer.json"))
build.build()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check(case, name, prefix, vprefix, encoder, layers, taps, prompt_prefix=""):
    ref = safe_open(str(ROOT / ".runtime/h3-native" / case / "reference.safetensors"), "pt", device="cuda")
    prompt = (ROOT / ".runtime/h3-native" / case / "prompt.txt").read_text()
    images = []
    while f"tokenizer.{encoder}.image{len(images)}" in ref.keys():
        images.append(ref.get_tensor(f"tokenizer.{encoder}.image{len(images)}")[0])
    items = cond.presentation(tok, prompt, len(images))
    marked = ref.get_tensor(f"tokenizer.{encoder}.marked").tolist()
    print(case, "tokens", "same" if items == marked else f"DIFF {items} vs {marked}")
    with safe_open(str(TE / name), "pt", device="cuda") as f:
        te = {k: f.get_tensor(k) for k in f.keys() if k.startswith((prefix, vprefix)) and not k.endswith("comfy_quant")}
    with torch.inference_mode():
        hidden, tags = cond.encode(te, prefix, vprefix, tok, prompt, images, layers, taps)
    print(case, "tags", torch.equal(tags.cpu(), ref.get_tensor("dit.call0.payload.text_token_tags").cpu()))
    return te, hidden, ref


te, hidden, ref = check("cond-h3-i2v", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "model.", "visual.", "qwen3vl_32b", 50, [])
print("h3 i2v layer 50", rel(hidden[-1], ref.get_tensor("te.output.0")[0]))
del te
te, hidden, ref = check("cond-h3-ref", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "model.", "visual.", "qwen3vl_32b", 50, [])
print("h3 ref layer 50", rel(hidden[-1], ref.get_tensor("te.output.0")[0]))
del te

sys.path.insert(0, str(HERE))
import ph3
dit = ph3.load_dit()
for case in ("cond-ph3-edit", "cond-ph3-editref"):
    ref = safe_open(str(ROOT / ".runtime/h3-native" / case / "reference.safetensors"), "pt", device="cuda")
    prompt = (ROOT / ".runtime/h3-native" / case / "prompt.txt").read_text()
    images = []
    while f"tokenizer.qwen3vl_4b.image{len(images)}" in ref.keys():
        images.append(ref.get_tensor(f"tokenizer.qwen3vl_4b.image{len(images)}")[0])
    print(case, "tokens", cond.presentation(tok, prompt, len(images)) == ref.get_tensor("tokenizer.qwen3vl_4b.marked").tolist())
    with torch.inference_mode():
        context, tags, connected = ph3.condition(ph3.tensors(ph3.TEXT_ENCODER), dit, tok, prompt, images)
    print(case, "tags", torch.equal(tags, ref.get_tensor("dit.call0.payload.text_token_tags")),
          "connector", rel(connected, [ref.get_tensor(k) for k in ref.keys() if k.endswith("Conditioning.out.0.0.0.0")][0][0]),
          "context", rel(context, ref.get_tensor("dit.call0.context")))
