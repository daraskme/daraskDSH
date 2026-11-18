"""Compare the native prediction and every LoRA gradient to diffusion-pipe with PEFT."""

import argparse
import os
from pathlib import Path
import sys

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[4]


def reference_model(diffusion_pipe, checkpoint, device, dtype):
    sys.path.insert(0, str(diffusion_pipe))
    from models.cosmos_predict2_modeling import MiniTrainDIT

    config = dict(
        max_img_h=1024,
        max_img_w=1024,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        pos_emb_cls="rope3d",
        use_adaln_lora=True,
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_enable_fps_modulation=False,
    )
    if checkpoint is None:
        return MiniTrainDIT(
            **config, model_channels=64, num_blocks=2, num_heads=1, adaln_lora_dim=12
        ).to(device=device, dtype=dtype)

    with torch.device("meta"):
        model = MiniTrainDIT(**config, model_channels=2048, num_blocks=28, num_heads=16)
    keys = set(model.state_dict())
    state = {}
    with safe_open(checkpoint, framework="pt") as file:
        for key in file.keys():
            name = key.removeprefix("net.").removeprefix("model.diffusion_model.")
            if name in keys:
                state[name] = file.get_tensor(key).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    with torch.device(device):
        model.build_pos_embed()
    return model


def normalized_error(actual, expected):
    actual = actual.float()
    expected = expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-20)).item()


def check_prediction(library, diffusion_pipe, checkpoint, sample, rank, device):
    if device == "cuda":
        # Several BF16 CUDA backward reductions can vary even when comparing
        # identical PEFT runs. Use the deterministic kernels for exact parity.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.ops.load_library(str(library))
    torch.manual_seed(534924)
    torch.set_num_threads(2)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = reference_model(diffusion_pipe, checkpoint, device, dtype)
    weights = dict(model.state_dict())
    targets = [
        name for name, module in model.named_modules()
        if name.startswith("blocks.") and isinstance(module, torch.nn.Linear)
    ]
    reference = get_peft_model(
        model,
        LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0, target_modules=targets),
        autocast_adapter_dtype=False,
    )
    reference_adapters = {
        name.removeprefix("base_model.model.").replace(".default.", "."): tensor
        for name, tensor in reference.named_parameters() if tensor.requires_grad
    }
    assert len(reference_adapters) == 2 * len(targets)
    generator = torch.Generator(device=device).manual_seed(8792)
    with torch.no_grad():
        for name, tensor in reference_adapters.items():
            if name.endswith(".lora_B.weight"):
                tensor.normal_(std=0.0007 if checkpoint else 0.01, generator=generator)
    native_adapters = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in reference_adapters.items()
    }
    if checkpoint is None:
        # The native implementation should still freeze the base if a caller
        # hands it differentiable weights.
        for weight in weights.values():
            weight.requires_grad_()
    versions = {name: tensor._version for name, tensor in weights.items()}

    if sample is None:
        latent = torch.randn((2, 16, 1, 6, 8), device=device, generator=generator)
        context = torch.randn(2, 512, 1024, device=device, dtype=dtype, generator=generator)
    else:
        cached = load_file(sample, device=device)
        latent = cached["latents"].float()
        context = cached["context"].to(dtype)
    shape = latent.shape
    batch, _, _, height, width = shape
    noise = torch.randn(shape, device=device, generator=generator)
    time = torch.linspace(0.17, 0.73, batch, device=device).unsqueeze(1)
    expanded_time = time.reshape(batch, 1, 1, 1, 1)
    noisy = (1 - expanded_time) * latent + expanded_time * noise
    target = noise - latent
    padding_mask = torch.zeros(batch, 1, height, width, device=device)

    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        expected = reference(noisy, time, context, padding_mask=padding_mask)
    expected_loss = torch.nn.functional.mse_loss(expected.float(), target)
    expected_loss.backward()
    actual = torch.ops.anima_native.predict(weights, native_adapters, noisy, time, context)
    actual_loss = torch.nn.functional.mse_loss(actual.float(), target)
    actual_loss.backward()

    assert actual.shape == expected.shape == shape
    assert actual.dtype == expected.dtype == dtype
    prediction_error = normalized_error(actual.detach(), expected.detach())
    loss_error = normalized_error(actual_loss.detach(), expected_loss.detach())
    tolerance = 1e-5
    print(f"Prediction relative L2 {prediction_error:.6g}, loss relative error {loss_error:.6g}", flush=True)
    assert prediction_error <= tolerance, prediction_error
    assert loss_error <= tolerance, loss_error
    gradient_errors = {}
    actual_gradients = []
    reference_gradients = []
    for name, tensor in native_adapters.items():
        grad = tensor.grad
        reference_grad = reference_adapters[name].grad
        assert grad is not None and reference_grad is not None, name
        assert torch.isfinite(grad).all() and grad.count_nonzero(), name
        gradient_errors[name] = normalized_error(grad, reference_grad)
        actual_gradients.append(grad.detach().float().flatten())
        reference_gradients.append(reference_grad.detach().float().flatten())
    worst = max(gradient_errors, key=gradient_errors.get)
    combined_error = normalized_error(torch.cat(actual_gradients), torch.cat(reference_gradients))
    print(f"{len(native_adapters)} LoRA gradients, combined relative L2 {combined_error:.6g}, "
          f"worst relative L2 {gradient_errors[worst]:.6g} ({worst})", flush=True)
    actual_flat, reference_flat = torch.cat(actual_gradients), torch.cat(reference_gradients)
    cosine = torch.nn.functional.cosine_similarity(actual_flat, reference_flat, dim=0).item()
    norm_ratio = (actual_flat.norm() / reference_flat.norm()).item()
    print(f"Combined gradient cosine {cosine:.6f}, norm ratio {norm_ratio:.6f}", flush=True)
    if dtype == torch.float32:
        assert gradient_errors[worst] <= tolerance, (worst, gradient_errors[worst])
    else:
        # The native backward shares BF16 casts, so gradient sums round in a
        # different order than autocast. Small, cancellation-heavy adaLN LoRA
        # gradients then differ by ~10% per layer, while the reference BF16 run
        # itself is ~25% (combined) away from an FP32 reference. Judge the whole.
        assert combined_error <= 0.02, combined_error
        assert cosine >= 0.9999, cosine
        assert abs(norm_ratio - 1) <= 0.01, norm_ratio
    assert all(tensor.grad is None and tensor._version == versions[name] for name, tensor in weights.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / ".runtime/anima-native/extension/dsh_anima_native.so")
    parser.add_argument("--diffusion-pipe", type=Path, default=ROOT / ".runtime/diffusion-pipe")
    parser.add_argument("--checkpoint", type=Path, help="Use Anima Base v1.0 on CUDA (default uses a small CPU model)")
    parser.add_argument("--sample", type=Path, help="Use a preprocessed safetensors file with latents and context")
    parser.add_argument("--rank", type=int, help="LoRA rank (default: 2 for CPU, 8 with --checkpoint)")
    check = parser.parse_args()
    check_prediction(
        check.library, check.diffusion_pipe, check.checkpoint, check.sample,
        check.rank or (8 if check.checkpoint else 2),
        device="cuda" if check.checkpoint else "cpu",
    )


if __name__ == "__main__":
    main()
