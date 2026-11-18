"""BF16 accuracy gate: native must be no less accurate than the reference BF16 run.

Both the native engine and diffusion-pipe (PEFT, CUDA autocast) are compared with
an FP32 diffusion-pipe run on Anima Base. The native engine deliberately rounds
differently from autocast (shared casts, fused RMSNorm), so it is judged against
FP32 rather than bit-for-bit against the reference.

Usage: bf16_accuracy.py [--checkpoint PATH] [--sample PATH]
"""
import argparse
from pathlib import Path
import sys

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_parity as mp

ROOT = mp.ROOT
SLACK = 1.05  # native error may exceed the reference's by at most 5%


def build(pipe, checkpoint, dtype, rank):
    model = mp.reference_model(pipe, checkpoint, "cuda", dtype)
    weights = dict(model.state_dict())
    targets = [n for n, m in model.named_modules() if n.startswith("blocks.") and isinstance(m, torch.nn.Linear)]
    peft = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0, target_modules=targets),
                          autocast_adapter_dtype=False)
    adapters = {n.removeprefix("base_model.model.").replace(".default.", "."): t
                for n, t in peft.named_parameters() if t.requires_grad}
    return weights, peft, adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=ROOT / ".runtime/anima-native/extension/dsh_anima_native.so")
    parser.add_argument("--diffusion-pipe", type=Path, default=ROOT / ".runtime/diffusion-pipe")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "plugins/dsh-comfyui/models/diffusion_models/anima-base-v1.0.safetensors")
    parser.add_argument("--sample", type=Path,
                        help="preprocessed safetensors with latents and context (default: first reference-cache file)")
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    sample = args.sample or sorted((ROOT / ".runtime/anima-native/reference-cache").glob("*.safetensors"))[0]
    torch.ops.load_library(str(args.library))
    torch.manual_seed(534924)

    weights, peft, adapters = build(args.diffusion_pipe, args.checkpoint, torch.bfloat16, args.rank)
    generator = torch.Generator(device="cuda").manual_seed(8792)
    with torch.no_grad():
        for name, tensor in adapters.items():
            if name.endswith(".lora_B.weight"):
                tensor.normal_(std=0.0007, generator=generator)
    initial = {n: t.detach().clone() for n, t in adapters.items()}
    native = {n: t.clone().requires_grad_() for n, t in initial.items()}

    cached = load_file(sample, device="cuda")
    latent, context = cached["latents"].float(), cached["context"].to(torch.bfloat16)
    batch, _, _, height, width = latent.shape
    noise = torch.randn(latent.shape, device="cuda", generator=generator)
    time = torch.linspace(0.17, 0.73, batch, device="cuda").unsqueeze(1)
    fraction = time.reshape(batch, 1, 1, 1, 1)
    noisy, target = (1 - fraction) * latent + fraction * noise, noise - latent
    padding = torch.zeros(batch, 1, height, width, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = peft(noisy, time, context, padding_mask=padding)
    torch.nn.functional.mse_loss(out.float(), target).backward()
    reference_prediction = out.detach().float()
    reference_grads = {n: t.grad.float() for n, t in adapters.items()}
    out = torch.ops.anima_native.predict(weights, native, noisy, time, context)
    torch.nn.functional.mse_loss(out.float(), target).backward()
    native_prediction = out.detach().float()
    native_grads = {n: t.grad.float() for n, t in native.items()}
    del weights, peft, adapters, native
    torch.cuda.empty_cache()

    _, peft, adapters = build(args.diffusion_pipe, args.checkpoint, torch.float32, args.rank)
    with torch.no_grad():
        for name, tensor in adapters.items():
            tensor.copy_(initial[name].float())
    out = peft(noisy, time, context.float(), padding_mask=padding)
    torch.nn.functional.mse_loss(out, target).backward()
    truth_prediction = out.detach().float()
    truth = {n: t.grad.float() for n, t in adapters.items()}

    flat = lambda grads: torch.cat([grads[n].flatten() for n in truth])
    results = {
        "prediction": (mp.normalized_error(native_prediction, truth_prediction),
                       mp.normalized_error(reference_prediction, truth_prediction)),
        "gradients (combined)": (mp.normalized_error(flat(native_grads), flat(truth)),
                                 mp.normalized_error(flat(reference_grads), flat(truth))),
        "gradients (sum of per-layer)": (sum(mp.normalized_error(native_grads[n], truth[n]) for n in truth),
                                         sum(mp.normalized_error(reference_grads[n], truth[n]) for n in truth)),
    }
    failed = []
    for label, (native_error, reference_error) in results.items():
        ok = native_error <= reference_error * SLACK
        print(f"{label}: error vs FP32 native {native_error:.5g}, reference BF16 {reference_error:.5g} "
              f"{'OK' if ok else 'WORSE'}", flush=True)
        if not ok:
            failed.append(label)
    assert not failed, failed


if __name__ == "__main__":
    main()
