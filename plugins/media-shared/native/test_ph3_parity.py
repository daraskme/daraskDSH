"""P-H3 image parity against the ComfyUI capture (T2I without the prompt upsampler).

Run with ComfyUI's venv:  comfy-venv/bin/python test_ph3_parity.py [REFERENCE_DIR]
Text: token ids -> 5 Qwen3-VL 4B hidden states -> text_connector -> DiT context. DiT (int8 convrot W8A8):
block 0, then the full forward at steps 0, 1 and 7 (docs/ph3-image-native-plan.md).
"""
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

ROOT = HERE.parents[2]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".runtime/h3-native/ph3-reference-1024"
DIT = ROOT / "models/diffusion_models/p-h3-image_int8_convrot.safetensors"
TE = ROOT / "models/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors"
TOKENIZER = ROOT / ".runtime/h3-native/build/qwen3vl_tokenizer.json"
TAPS = [13, 19, 25, 31]  # + the final-norm output (layer 36): PH3_IMAGE_HIDDEN_STATES


def report(name, got, ref):
    got, ref = got.float().flatten().cpu(), ref.float().flatten().cpu()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"{name}: bit-identical {torch.equal(got, ref)}, relative L2 {rel:.3e}, max abs {(got - ref).abs().max().item():.3e}")


build.build()
ref = load_file(str(REF / "reference.safetensors"), device="cuda")
import ph3  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
prompt = (REF / "prompt.txt").read_text()
tokenizer = Tokenizer.from_file(str(TOKENIZER))
ids = tokenizer.encode(prompt, add_special_tokens=False).ids
print("token ids exact:", ids == ref["tokenizer.qwen3vl_4b.ids"].tolist())
weights = ph3.load_dit()
with torch.inference_mode():
    context, _, connected = ph3.condition(ph3.tensors(TE), weights, tokenizer, prompt)
report("text_connector output", connected, ref["node.26.PH3ImageConditioning.out.0.0.0.0"][0])
report("DiT context (connector -> refiner)", context, ref["dit.call0.context"])
with torch.inference_mode():
    report("DiT context from the captured connector output",
           torch.ops.dsh_h3.refine_text(weights, ref["node.26.PH3ImageConditioning.out.0.0.0.0"][0]).unsqueeze(0), ref["dit.call0.context"])
with torch.inference_mode():
    segments = [v for seg in json.loads((REF / "block0_segments.json").read_text()) for v in seg]
    out = torch.ops.dsh_h3.dit_block(weights, 0, ref["block0.x"], ref["block0.t_emb"], segments, ref["block0.rope_freqs"])
    report("block 0", out, ref["block0.out"])
    for call in (0, 1, 7):
        p = f"dit.call{call}"
        video, audio = torch.ops.dsh_h3.dit_forward(weights, ref[p + ".x.0"], ref[p + ".x.1"], ref[p + ".timestep"], ref[p + ".context"],
                                                    ref[p + ".payload.text_token_tags"], 1.0, 1.0, 1.0, [], False)
        report(f"step {call} video", video, ref[p + ".out.0"])
        report(f"step {call} audio", audio, ref[p + ".out.1"])
    steps = {}
    latent = ph3.sample(weights, ref["dit.call0.context"], ref["dit.call0.payload.text_token_tags"], 42, 1024, 1024,
                        on_step=lambda i, x, d: steps.update({i: x}))
    report("noise (= step 0 DiT input)", steps[0][0].to(torch.bfloat16), ref["dit.call0.x.0"])
    report("step 7 DiT input", steps[7][0].to(torch.bfloat16), ref["dit.call7.x.0"])
    report("final latent", latent[0], ref["node.10.SamplerCustomAdvanced.out.0.0.samples.0"])
    del weights
    rgb = ph3.decode(ph3.load_vae(), ref["node.11.VAEDecode.in.samples.0.samples.0"])
    want = ref["node.11.VAEDecode.out.0.0"][0]
    report("VAE RGB (int8 weights dequantized to FP16)", rgb, want)
    print(f"  PSNR {10 * torch.log10(1 / ((rgb.float() - want.float()) ** 2).mean()).item():.1f} dB")
