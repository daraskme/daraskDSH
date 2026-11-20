"""Command line for Anima LoRA training and reference comparisons."""

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from tempfile import NamedTemporaryFile
import time

from safetensors import SafetensorError
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if __package__:
    from .common import clip_gradients as _clip_gradients, publish_new as _publish_new, sample_order as _sample_order
    from .dataset import DTYPE, ModelPaths, find_samples, load_reference, prepare_dataset, read_dit_checkpoint
else:
    from common import clip_gradients as _clip_gradients, publish_new as _publish_new, sample_order as _sample_order
    from dataset import DTYPE, ModelPaths, find_samples, load_reference, prepare_dataset, read_dit_checkpoint


def load_dit_weights(path):
    weights = read_dit_checkpoint(path)
    if any(value.dtype not in (torch.float32, torch.bfloat16, torch.float16) for value in weights.values()):
        raise ValueError(f"Expected an unquantized Anima DiT checkpoint: {path}")
    return {name: value.to(DTYPE) for name, value in weights.items()}


def initialize_adapters(weights, rank, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    adapters = {}
    for name, weight in sorted(weights.items()):
        # diffusion-pipe targets all nn.Linear modules within each DiT block.
        if not name.startswith("blocks.") or weight.ndim != 2 or not name.endswith(".weight"):
            continue
        outputs, inputs = weight.shape
        prefix = name.removesuffix(".weight")
        a = torch.empty((rank, inputs), dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        adapters[prefix + ".lora_A.weight"] = a.to(weight.dtype)
        adapters[prefix + ".lora_B.weight"] = torch.zeros((outputs, rank), dtype=weight.dtype)
    if not adapters:
        raise ValueError("The Anima DiT checkpoint has no compatible blocks")
    return adapters


def dit_config(weights):
    # From diffusion-pipe's get_dit_config, with the LLM adapter supplied separately.
    width, input_size = weights["x_embedder.proj.1.weight"].shape
    head_dim = weights["blocks.0.self_attn.q_norm.weight"].numel()
    blocks = set()
    for name in weights:
        match = re.match(r"blocks\.(\d+)\.", name)
        if match:
            blocks.add(int(match[1]))
    if input_size != 68 or width % head_dim or blocks != set(range(len(blocks))):
        raise ValueError("Expected Anima image DiT weights with 16 latent channels and contiguous blocks")
    return dict(max_img_h=1024, max_img_w=1024, max_frames=128,
                in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
                model_channels=width, concat_padding_mask=True, crossattn_emb_channels=1024,
                pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
                min_fps=1, max_fps=30, use_adaln_lora=True,
                adaln_lora_dim=weights["blocks.0.adaln_modulation_self_attn.1.weight"].shape[0],
                mlp_ratio=weights["blocks.0.mlp.layer1.weight"].shape[0] / width,
                num_blocks=len(blocks), num_heads=width // head_dim,
                extra_per_block_abs_pos_emb=False, rope_h_extrapolation_ratio=4.0,
                rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
                extra_h_extrapolation_ratio=1.0, extra_w_extrapolation_ratio=1.0,
                extra_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=False, use_llm_adapter=False)


class LoRALinear(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(a.to(base.weight).clone())
        self.b = nn.Parameter(b.to(base.weight).clone())

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.a), self.b)


def build_reference_model(reference, weights, adapters, device, compile_blocks=False, *, model_config=None):
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = reference.MiniTrainDIT(**(dit_config(weights) if model_config is None else model_config))
    model.load_state_dict(weights, assign=True)
    model.requires_grad_(False).train().to(device)
    for key, a in adapters.items():
        if key.endswith(".lora_A.weight"):
            prefix = key.removesuffix(".lora_A.weight")
            linear = model.get_submodule(prefix)
            b = adapters[prefix + ".lora_B.weight"]
            model.set_submodule(prefix, LoRALinear(linear, a, b))
    if compile_blocks:
        for block in model.blocks:
            block.compile(fullgraph=True, dynamic=False)
    return model


def flow_inputs(latents, noise, timestep):
    latents, noise, timestep = latents.float(), noise.float(), timestep.float().reshape(-1, 1)
    interpolation = timestep.view(-1, 1, 1, 1, 1)
    return (1 - interpolation) * latents + interpolation * noise, timestep, noise - latents


def train_reference(reference, weights, adapters, latents, contexts, steps, learning_rate, seed,
                    *, device="cuda", compile_blocks=False, noises=None, times=None, log_every=0):
    from optimi import AdamW

    if (noises is None) != (times is None) or (noises is not None and (len(noises) != steps or len(times) != steps)):
        raise ValueError("Provide one noise and timestep tensor per step, or omit both")
    device = torch.device(device)
    timestep_generator = torch.Generator(device="cpu").manual_seed(seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    order = _sample_order(len(latents), seed)
    model = build_reference_model(reference, weights, adapters, device, compile_blocks)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.01, eps=1e-8)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
    loss_values, elapsed_ms = [], []
    shape_count = len({latent.shape for latent in latents})
    compile_context = (torch._dynamo.config.patch(recompile_limit=max(8, 2 * shape_count))
                       if compile_blocks else nullcontext())
    with compile_context:
        for step in range(steps):
            started = time.perf_counter()
            index = next(order) if noises is None else step % len(latents)
            latent = latents[index].to(device=device, dtype=torch.float32)
            context = contexts[index].to(device=device, dtype=next(model.parameters()).dtype)
            if noises is None:
                timestep = torch.randn((latent.shape[0], 1), generator=timestep_generator).sigmoid().to(device)
                noise = torch.randn(latent.shape, generator=noise_generator, device=device, dtype=torch.float32)
            else:
                timestep, noise = times[step].to(device), noises[step].to(device)
            noisy, timestep, target = flow_inputs(latent, noise, timestep)
            padding = noisy.new_zeros((noisy.shape[0], 1, *noisy.shape[-2:]))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=DTYPE, enabled=use_cuda):
                prediction = model(noisy, timestep, context, padding_mask=padding)
            loss = F.mse_loss(prediction.float(), target)
            loss.backward()
            _clip_gradients(params)
            optimizer.step()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite training loss on step {step + 1}")
            loss_values.append(loss_value)
            elapsed_ms.append((time.perf_counter() - started) * 1000)
            if log_every and ((step + 1) % log_every == 0 or step + 1 == steps):
                print(f"step {step + 1}/{steps} loss={loss_value:.6f} ms={elapsed_ms[-1]:.1f}", file=sys.stderr, flush=True)

    results = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            results[name + ".lora_A.weight"] = module.a.detach().cpu()
            results[name + ".lora_B.weight"] = module.b.detach().cpu()
    return results, torch.tensor(loss_values, dtype=torch.float32), torch.tensor(elapsed_ms, dtype=torch.float64)


def train_native(weights, adapters, latents, contexts, steps, learning_rate, seed, *, noises=None, times=None, log_every=0):
    if __package__:
        from .build import load_extension
    else:
        from build import load_extension

    load_extension()
    return torch.ops.anima_native.train(weights, adapters, latents, contexts, steps, learning_rate,
                                        seed, noises, times, log_every)


def save_adapter(path, adapters):
    parameters = {"diffusion_model." + key: value.detach().to(device="cpu", dtype=DTYPE).contiguous()
                  for key, value in adapters.items()}
    _publish_new(Path(path), lambda temporary: save_file(parameters, temporary, metadata={"format": "pt"}))


def _positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _model_paths(args):
    config_directory = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    config_path = (args.config or Path(os.environ.get("COMFY_FOLDER_LORA_CONFIG") or config_directory / "comfy-folder-lora/config.json")).expanduser()
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif args.config or os.environ.get("COMFY_FOLDER_LORA_CONFIG"):
        raise ValueError(f"Anima configuration does not exist: {config_path}")
    else:
        config = {}
    models = config.get("models", {}).get("anima", {})
    pipe = args.diffusion_pipe or config.get("diffusionPipe") or ROOT / ".runtime/diffusion-pipe"
    diffusion = args.diffusion or models.get("diffusion")
    vae = args.vae or models.get("vae")
    encoder = args.text_encoder or models.get("textEncoder")
    for flag, path in (("--diffusion", diffusion), ("--vae", vae), ("--text-encoder", encoder)):
        if not path or not Path(path).expanduser().is_file():
            raise ValueError(f"Anima checkpoint is missing for {flag}: {path or config_path}")
    return ModelPaths(*(Path(path).expanduser().resolve() for path in (pipe, diffusion, vae, encoder)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="dsh anima train", description="Train a ComfyUI-compatible Anima LoRA from a folder of images.")
    parser.add_argument("--folder", required=True, type=Path, help="image directory with same-stem UTF-8 .txt captions")
    parser.add_argument("--output", required=True, type=Path, help="new .safetensors file")
    parser.add_argument("--recursive", action="store_true", help="include images in subdirectories")
    parser.add_argument("--resolution", default=512, type=_positive_integer, help="nominal square resolution for aspect ratio buckets (default: 512)")
    parser.add_argument("--rank", default=16, type=_positive_integer, help="LoRA rank (default: 16)")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=_positive_integer, help="number of optimizer steps")
    duration.add_argument("--epochs", default=1, type=_positive_integer, help="number of passes over the image folder (default: 1)")
    parser.add_argument("--learning-rate", default=2e-5, type=float, help="constant AdamW learning rate (default: 2e-5)")
    parser.add_argument("--seed", default=42, type=int, help="adapter initialization, shuffling, and flow sampling seed (default: 42)")
    parser.add_argument("--backend", choices=("native", "eager", "compile"), default="native")
    parser.add_argument("--config", type=Path, help="Folder LoRA config.json (overrides COMFY_FOLDER_LORA_CONFIG)")
    parser.add_argument("--diffusion-pipe", type=Path, help="diffusion-pipe checkout with Anima model and tokenizer definitions")
    parser.add_argument("--diffusion", type=Path, help="Anima base DiT safetensors")
    parser.add_argument("--vae", type=Path, help="Anima VAE safetensors")
    parser.add_argument("--text-encoder", type=Path, help="Anima Qwen3-0.6B text encoder safetensors")
    parser.add_argument("--cache-dir", type=Path, help="reuse encoded samples from this directory across training runs")
    parser.add_argument("--metrics", type=Path, help="save the JSON run summary to a new file")
    parser.add_argument("--summary-only", action="store_true", help="omit individual steps from stdout JSON")
    parser.add_argument("--log-every", default=1, type=int, help="print progress every N steps (0 disables)")
    args = parser.parse_args(argv)
    if args.resolution % 16 or args.resolution > 5792:
        parser.error("--resolution must be a multiple of 16 no greater than 5792")
    if args.seed < 0 or args.seed >= 2**63:
        parser.error("--seed must be between 0 and 2^63-1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if args.log_every < 0:
        parser.error("--log-every must be nonnegative")
    if args.output.suffix.lower() != ".safetensors":
        parser.error("--output must end in .safetensors")
    if args.metrics is not None and args.metrics.resolve() == args.output.resolve():
        parser.error("--output and --metrics must use different paths")
    return args


def run(args):
    output = args.output.expanduser().absolute()
    metrics_path = args.metrics.expanduser().absolute() if args.metrics is not None else None
    for path in (output, metrics_path):
        if path is not None and os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite {path}")
    samples = find_samples(args.folder.expanduser(), args.recursive)
    paths = _model_paths(args)
    reference = load_reference(paths.diffusion_pipe)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Anima training requires a CUDA device with native bfloat16 support")

    device = torch.device("cuda")
    steps = args.steps if args.steps is not None else args.epochs * len(samples)
    torch.cuda.reset_peak_memory_stats(device)
    preprocessing_started = time.perf_counter()
    print(f"Encoding {len(samples)} image(s) at {args.resolution}px", file=sys.stderr, flush=True)
    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    latents, contexts, cache_hits = prepare_dataset(samples, args.resolution, paths, device, cache_dir)
    torch.cuda.synchronize(device)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    preprocessing_peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    preprocessing_peak_reserved_vram_bytes = torch.cuda.max_memory_reserved(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    weights = load_dit_weights(paths.diffusion)
    adapters = initialize_adapters(weights, args.rank, args.seed)
    train_args = (weights, adapters, latents, contexts, steps, args.learning_rate, args.seed)
    print(f"Training {steps} step(s) with the {args.backend} backend", file=sys.stderr, flush=True)
    training_started = time.perf_counter()
    if args.backend == "native":
        trained, losses, elapsed_ms = train_native(*train_args, log_every=args.log_every)
    else:
        trained, losses, elapsed_ms = train_reference(reference, *train_args, device=device,
                                                      compile_blocks=args.backend == "compile", log_every=args.log_every)
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    timings, step_loss = elapsed_ms.tolist(), losses.tolist()
    result = {"output": str(output), "backend": args.backend, "gpu": torch.cuda.get_device_name(device),
              "steps": steps, "images": len(samples), "cache_hits": cache_hits, "resolution": args.resolution,
              "rank": args.rank, "learning_rate": args.learning_rate, "seed": args.seed,
              "preprocessing_seconds": preprocessing_seconds, "preprocessing_peak_vram_bytes": preprocessing_peak_vram_bytes,
              "preprocessing_peak_reserved_vram_bytes": preprocessing_peak_reserved_vram_bytes,
              "training_seconds": training_seconds, "training_peak_vram_bytes": torch.cuda.max_memory_allocated(device),
              "training_peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
              "first_loss": step_loss[0], "last_loss": step_loss[-1], "first_step_ms": timings[0],
              "step_loss": step_loss, "step_ms": timings, "median_step_ms": statistics.median(timings),
              "median_step_ms_after_first": statistics.median(timings[1:]) if len(timings) > 1 else None}
    summary = json.dumps(result, ensure_ascii=False, allow_nan=False)
    save_adapter(output, trained)
    if metrics_path is not None:
        _publish_new(metrics_path, lambda temporary: temporary.write_text(summary + "\n", encoding="utf-8"))
    if args.summary_only:
        result.pop("step_loss")
        result.pop("step_ms")
        summary = json.dumps(result, ensure_ascii=False, allow_nan=False)
    print(summary, flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (ImportError, KeyError, OSError, RuntimeError, SafetensorError, ValueError) as error:
        print(f"Anima training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
