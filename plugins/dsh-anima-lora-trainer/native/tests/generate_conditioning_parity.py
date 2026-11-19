"""Anima generation text conditioning vs the ComfyUI capture (positive with emphasis, empty negative).

Run with ComfyUI's venv:  comfy-venv/bin/python generate_conditioning_parity.py [REFERENCE_DIR]
Checks: Qwen ids -> T5 ids/weights -> the c_crossattn the DiT receives ([1, 512, 1024] BF16), using the
tokenization rules of ComfyUI's SDTokenizer (emphasis parsing, per-segment tokenization) and the frontend's
native Qwen3-0.6B + LLM adapter (generate/text.cpp), then T5 weights and zero padding.
"""
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from safetensors import safe_open
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[4]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/anima-native/generate-reference"
FRONTEND = ROOT / ".runtime/anima-native/frontend"
NATIVE = ROOT / "plugins/dsh-anima-lora-trainer/native"
MODELS = ROOT / "plugins/dsh-comfyui/models"
sys.path.insert(0, str(NATIVE / "generate"))
from tokens import QWEN_PAD, T5_END, tokenize  # noqa: E402


ref = load_file(str(REF / "reference.safetensors"))
qwen_tok = Tokenizer.from_file(str(FRONTEND / "qwen_tokenizer.json"))
t5_tok = Tokenizer.from_file(str(FRONTEND / "t5_tokenizer.json"))
sys.path.insert(0, str(ROOT / "plugins/dsh-anima-lora-trainer/native/generate"))
import generate  # noqa: E402

generate.build()  # the C++ ops and FP8 kernels, shared with the runner
with safe_open(str(MODELS / "text_encoders/qwen_3_06b_base.safetensors"), "pt", device="cuda") as f:
    qwen_w = {k: f.get_tensor(k).float() for k in f.keys()}  # ComfyUI runs this encoder in FP32
with safe_open(str(MODELS / "diffusion_models/anima-base-v1.0.safetensors"), "pt", device="cuda") as f:
    dit_w = {k: f.get_tensor(k) for k in f.keys() if k.startswith("net.llm_adapter.")}

for index in (0, 1):
    prompt = (REF / f"prompt{index}.txt").read_text()
    qwen = tokenize(qwen_tok, prompt, False, None, QWEN_PAD)
    t5 = tokenize(t5_tok, prompt, True, T5_END, 0)
    qwen_ids = [t for t, _ in qwen]
    t5_ids, t5_w = [t for t, _ in t5], [w for _, w in t5]
    ok = (qwen_ids == ref[f"prompt{index}.qwen3_06b.ids"].tolist(), t5_ids == ref[f"prompt{index}.t5xxl.ids"].tolist(),
          torch.equal(torch.tensor(t5_w, dtype=torch.float32), ref[f"prompt{index}.t5xxl.weights"]))
    print(f"prompt{index} {prompt[:40]!r}: qwen ids {ok[0]} ({len(qwen_ids)}), t5 ids {ok[1]} ({len(t5_ids)}), t5 weights {ok[2]}")

    def rel(a, b):
        return ((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm().clamp_min(1e-12)).item()
    q = torch.tensor(qwen_ids, device="cuda")
    t = torch.tensor(t5_ids, device="cuda")
    enc = next(i for i in (0, 1) if ref[f"encode{i}.hidden"].shape[1] == len(qwen_ids))
    cross = next(i for i in (0, 1) if ref[f"crossattn{i}.in.ids"].shape[1] == len(t5_ids))
    with torch.no_grad():
        hidden = torch.ops.dsh_anima_gen.qwen(qwen_w, q)
        print(f"  qwen hidden: bit-identical {torch.equal(hidden.float().cpu(), ref[f'encode{enc}.hidden'])}, "
              f"relative L2 {rel(hidden, ref[f'encode{enc}.hidden']):.3e}")
        adapted = torch.ops.dsh_anima_gen.adapter(dit_w, ref[f"crossattn{cross}.in.embeds"].cuda(), t)
        weighted = torch.nn.functional.pad(adapted * ref[f"crossattn{cross}.in.weights"].cuda(), (0, 0, 0, 512 - adapted.shape[1]))
        want = ref[f"crossattn{cross}.out"]
        print(f"  adapter on ComfyUI's Qwen hidden: bit-identical {torch.equal(weighted.cpu(), want)}, relative L2 {rel(weighted, want):.3e}")
        out = torch.ops.dsh_anima_gen.crossattn(qwen_w, dit_w, q, t, torch.tensor(t5_w))
        print(f"  c_crossattn end to end: shape {tuple(out.shape)}, bit-identical {torch.equal(out.cpu(), want)}, relative L2 {rel(out, want):.3e}")
