"""Native LoRA training for MiniMax H3 (videos with sound, and pictures) and P-H3 (pictures).

Usage (ComfyUI's venv, ninja on PATH):
  train_lora.py --model h3|ph3 --folder DIR --output NEW.safetensors [--steps N | --epochs N] [--rank 32] ...

Same objective as diffusion-pipe's models/minimax_h3.py: flow matching (x_t = (1 - t) x0 + t noise, target noise - x0),
uniform t with shift 8 for videos and 1 for pictures, the audio at time_shift_sigma(t, 12, 3) (the model's own shifts;
P-H3 1 / 1 as its sampler), MSE weighted by element count across video and audio, LoRA on the 208 projections of the
blocks and the token refiner, and CFG-augmented training (cfg 4: (cond + 3 uncond) / 4, the uncond pass without gradient
on the empty prompt, one padding token masked out as a key) to keep the 8-step distillation. What is faster: every weight stays on the GPU (no block
swap, no pipeline engine), activation checkpointing only for the blocks that would not fit, mask-free flash attention,
and cached text / latents (lora_model.py, encode.py).
Captions are the same-stem .txt files. Progress goes to stderr; the JSON summary to stdout.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # mixed video / picture shapes
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.append(str(ROOT / "plugins/lora-training/native"))  # after HERE: both have a build.py
import common  # noqa: E402  (shared with the Anima trainer)

MODELS = ROOT / "models"
H3 = dict(dit=MODELS / "diffusion_models/10Eros_Max_h3_8stepTURBO_beta7.safetensors",
          base_loras=[MODELS / "loras/H3_VBVR_Pro-V2.safetensors"],
          text_encoder=MODELS / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
          video_vae=MODELS / "vae/minimax_h3_video_vae_dully_seamless_v1_fp16.safetensors",
          audio_vae=MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors",
          shift_v=12.0, shift_a=3.0, image_resolution=768, video_resolution=512)
PH3 = dict(shift_v=1.0, shift_a=1.0, image_resolution=1024)
TOKENIZER = ROOT / ".runtime/h3-native/build/qwen3vl_tokenizer.json"
CACHE = ROOT / ".runtime/h3-lora/cache"
FRAME_BUCKETS = (17, 34, 68, 102, 136)  # the VAE encodes 17-frame clips
FPS, SAMPLE_RATE = 24, 32000
VIDEO_SHIFT, IMAGE_SHIFT = 8.0, 1.0
CFG = 4.0
# Activation memory per packed token (measured on H3): a block kept for backward ~275 KB, a checkpointed one (its
# input) ~30 KB, plus ~0.5 MB transient while one block recomputes and runs backward. Blocks that do not fit in the
# free memory are checkpointed (recomputed in backward).
KEPT_BYTES, CHECKPOINTED_BYTES, TRANSIENT_BYTES = 280_000, 32_000, 600_000


def log(message):
    print(message, file=sys.stderr, flush=True)


def load(path, keep=lambda k: True, cast=None):
    from safetensors import safe_open
    with safe_open(str(path), "pt", device="cuda") as f:
        return {k: (f.get_tensor(k) if cast is None else f.get_tensor(k).to(cast))
                for k in f.keys() if keep(k) and not k.endswith("comfy_quant")}


# ---- media -----------------------------------------------------------------------------------------

def read_picture(path, resolution):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as source:
        image = common.fit_rgb(source, common.bucket_size(*source.size, resolution, 32))
    return torch.from_numpy(np.asarray(image).copy()).float() / 255.0  # [H, W, 3]


def read_video(path, resolution):
    """First FRAME_BUCKETS-sized clip ([T, H, W, 3] in [0, 1]) and its 32 kHz stereo audio ([2, L]) or None."""
    import av
    import numpy as np
    import torch
    from PIL import Image
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        size = common.bucket_size(stream.codec_context.width, stream.codec_context.height, resolution, 32)
        frames = []
        for frame in container.decode(video=0):
            frames.append(np.asarray(common.fit_rgb(frame.to_image(), size)))
            if len(frames) == FRAME_BUCKETS[-1]:
                break
    usable = [n for n in FRAME_BUCKETS if n <= len(frames)]
    if not usable:
        return None, None
    count = usable[-1]
    video = torch.from_numpy(np.stack(frames[:count])).float() / 255.0
    wave = None
    with av.open(str(path)) as container:
        if container.streams.audio:
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SAMPLE_RATE)
            chunks, need = [], math.ceil(count / FPS * SAMPLE_RATE)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
                if sum(c.shape[1] for c in chunks) >= need:
                    break
            if chunks:
                wave = np.concatenate(chunks, axis=1)[:, :need]
                wave = torch.from_numpy(np.pad(wave, ((0, 0), (0, need - wave.shape[1])))).float()
    return video, wave


# ---- caching ------------------------------------------------------------------------------------------

def cache_key(model, sample, settings):
    stat = sample.image.stat()
    text = json.dumps([model, str(sample.image.resolve()), stat.st_size, stat.st_mtime_ns, sample.caption, settings])
    return sha256(text.encode()).hexdigest()[:32]


def encode_texts(model, captions):
    """{caption: (text states [T, 5120] FP32 before condition_proj, tags [T])}; the DiT's refiner runs in training."""
    import torch
    from tokenizers import Tokenizer
    import cond
    import ph3
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    out = {}
    with torch.inference_mode():
        if model == "h3":
            te = load(H3["text_encoder"], lambda k: k.startswith("model."))
            for c in sorted(set(captions)):
                states, tags = cond.encode(te, "model.", "visual.", tokenizer, c, [], 50, [])
                out[c] = (states[-1].float().cpu(), tags.cpu())
        else:
            te = ph3.tensors(ph3.TEXT_ENCODER)
            dit = {k: v.bfloat16() for k, v in ph3.tensors(ph3.DIT, lambda k: k.startswith(("condition_proj.", "token_refiner."))).items()}
            for c in sorted(set(captions)):
                _, tags, states = ph3.condition(te, dit, tokenizer, c)
                out[c] = (states.float().cpu(), tags.cpu())
            del dit
        del te
    torch.cuda.empty_cache()
    return out


def prepare(model, samples):
    """Encode (or read from the cache) every sample -> list of dicts with CPU tensors."""
    import torch
    from safetensors.torch import load_file, save_file
    import encode
    import ph3
    import vae
    settings = H3 if model == "h3" else PH3
    keys = {s: cache_key(model, s, [settings["image_resolution"], settings.get("video_resolution"), FRAME_BUCKETS])
            for s in samples}
    CACHE.mkdir(parents=True, exist_ok=True)
    missing = [s for s in samples if not (CACHE / f"{keys[s]}.safetensors").is_file()]
    hits = len(samples) - len(missing)
    if missing:
        log(f"Encoding {len(missing)} sample(s) (cached {hits})")
        texts = encode_texts(model, [s.caption for s in missing])
        with torch.inference_mode():
            if model == "h3":
                video_w = load(H3["video_vae"], lambda k: k.startswith(("encoder.", "quant_conv.", "latents_")), torch.float16)
                audio_w = load(H3["audio_vae"], lambda k: not k.startswith(("decoder.", "dec_in_proj.")))
            else:
                video_w = ph3.load_vae()
            for s in missing:
                entry = {"states": texts[s.caption][0].contiguous(), "tags": texts[s.caption][1].contiguous()}
                if s.image.suffix.lower() in common.VIDEO_EXTENSIONS:
                    frames, wave = read_video(s.image, settings["video_resolution"])
                    if frames is None:
                        log(f"skipped (shorter than {FRAME_BUCKETS[0]} frames): {s.image.name}")
                        continue
                    entry["latent"] = encode.encode_video(video_w, frames).cpu()
                    if wave is not None:
                        entry["audio"] = encode.encode_audio(audio_w, wave).cpu()
                elif model == "h3":
                    entry["latent"] = encode.encode_video(video_w, read_picture(s.image, settings["image_resolution"])[None]).cpu()
                else:
                    entry["latent"] = vae.encode_image(video_w, read_picture(s.image, settings["image_resolution"]), sample=True).cpu()
                entry = {k: v.contiguous() for k, v in entry.items()}
                common.publish_new(CACHE / f"{keys[s]}.safetensors", lambda path: save_file(entry, str(path)))
        del video_w
        if model == "h3":
            del audio_w
        torch.cuda.empty_cache()
    data = [load_file(str(CACHE / f"{keys[s]}.safetensors")) for s in samples if (CACHE / f"{keys[s]}.safetensors").is_file()]
    return data, hits


# ---- training ------------------------------------------------------------------------------------------

def load_dit(model, ph3_bf16=False):
    import torch
    import ph3
    if model == "ph3":
        w = ph3.load_dit()
        if ph3_bf16:  # the int8 convrot projections as the BF16 weights they stand for, once
            for key in [k for k in w if k.endswith(".weight_scale")]:
                name = key[:-len("_scale")]
                w[name] = torch.ops.dsh_kitchen.dequantize_int8_convrot(w[name], w.pop(key))
        return w
    w = load(H3["dit"])
    for path in H3["base_loras"]:
        torch.ops.dsh_h3.merge_loras(w, load(path), 1.0)
    return w


def checkpoint_count(tokens, blocks):
    import torch
    free, _ = torch.cuda.mem_get_info()
    free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()  # the caching allocator's spare blocks
    budget = free - 4 * 2**30 - tokens * (TRANSIENT_BYTES + blocks * CHECKPOINTED_BYTES)
    keep = int(max(0, budget) // (tokens * (KEPT_BYTES - CHECKPOINTED_BYTES)))
    return max(0, blocks - keep)


def train(args):
    import torch
    import torch.nn.functional as F
    import build
    import lora_model as M

    build.build()
    torch.manual_seed(args.seed)
    settings = H3 if args.model == "h3" else PH3
    extensions = common.IMAGE_EXTENSIONS | (common.VIDEO_EXTENSIONS if args.model == "h3" else set())
    samples = common.find_samples(args.folder, args.recursive, extensions)
    started = time.perf_counter()
    data, hits = prepare(args.model, samples)
    preprocessing_seconds = time.perf_counter() - started
    if not data:
        raise ValueError("no usable samples")
    steps = args.steps or args.epochs * len(data)

    started = time.perf_counter()
    w = load_dit(args.model, args.ph3_bf16)
    lora = M.lora_parameters(w, args.rank, args.seed)
    params = [p for pair in lora.values() for p in pair]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.01, eps=1e-8, fused=True)
    blocks = M.block_count(w)
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    log(f"Training {steps} step(s) on {len(data)} sample(s), rank {args.rank}")
    torch.cuda.synchronize()

    order = common.sample_order(len(data), args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    losses, step_ms = [], []
    empty = torch.ones(1, dtype=torch.long)  # the empty prompt: one text-tagged padding row
    for step in range(steps):
        begin = time.perf_counter()
        item = data[next(order)]
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * min(1.0, (step + 1) / args.warmup) if args.warmup else args.learning_rate
        x0 = item["latent"].cuda().float()
        is_image = x0.shape[2] == 1
        if "audio" in item:
            a0 = item["audio"].cuda().float()
        elif args.model == "ph3":
            a0 = torch.zeros(1, 32, 2, 2, device="cuda")  # p-h3-image samples a 2-step dummy audio latent
        else:
            a0 = torch.zeros(1, 32, 2, 0, device="cuda")
        shift = IMAGE_SHIFT if is_image else VIDEO_SHIFT
        t = torch.rand((), device="cuda", generator=generator)
        t = t * shift / (1 + (shift - 1) * t)
        noise = torch.randn(x0.shape, device="cuda", generator=generator)
        audio_noise = torch.randn(a0.shape, device="cuda", generator=generator)
        t_a = M.time_shift_sigma(t, settings["shift_v"], settings["shift_a"])
        xt = ((1 - t) * x0 + t * noise).to(torch.bfloat16)
        at = ((1 - t_a) * a0 + t_a * audio_noise).to(torch.bfloat16)
        sigma = float(t)

        states = item["states"].cuda()
        text = torch.empty(1, states.shape[0], 0, dtype=torch.bfloat16)
        inputs = M.prepare(w, xt, at, sigma, text, item["tags"], settings["shift_v"], settings["shift_a"])
        tokens = inputs["h"].shape[0]
        if args.cfg > 1:
            with torch.no_grad():
                u = M.prepare(w, xt, at, sigma, text[:, :1], empty, settings["shift_v"], settings["shift_a"])
                video_u, audio_u = M.forward(w, lora, u)
                del u
        checkpointed = checkpoint_count(tokens, blocks)
        video, audio = M.forward(w, lora, inputs, states, checkpointed)
        if args.cfg > 1:
            video = (video + (args.cfg - 1) * video_u) / args.cfg
            audio = (audio + (args.cfg - 1) * audio_u) / args.cfg
        video_loss = F.mse_loss(video, noise - x0)
        loss = video_loss
        if "audio" in item:
            nv, na = video.numel(), audio.numel()
            loss = (video_loss * nv + F.mse_loss(audio, audio_noise - a0) * na) / (nv + na)
        loss.backward()
        common.clip_gradients(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step_ms.append((time.perf_counter() - begin) * 1000)
        if args.log_every and (step + 1) % args.log_every == 0:
            # the same progress line as the Anima trainer (pipeline.py parses it)
            log(f"step {step + 1}/{steps} loss={losses[-1]:.6f} elapsed_ms={step_ms[-1]:.1f} tokens={tokens} checkpointed={checkpointed}")

    adapters = {}
    for name, (a, b) in lora.items():
        adapters[name + ".lora_A.weight"] = a
        adapters[name + ".lora_B.weight"] = b
    common.save_lora(args.output, adapters, {"dsh_model": args.model, "rank": str(args.rank)})
    return {"output": str(args.output), "model": args.model, "steps": steps, "images": len(data), "cache_hits": hits,
            "rank": args.rank, "learning_rate": args.learning_rate, "preprocessing_seconds": preprocessing_seconds,
            "load_seconds": load_seconds, "training_seconds": sum(step_ms) / 1000,
            "training_peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "first_loss": losses[0], "last_loss": losses[-1], "median_step_ms_after_first":
            statistics.median(step_ms[1:]) if len(step_ms) > 1 else step_ms[0], "step_ms": step_ms, "step_loss": losses}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="train_lora.py", description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("h3", "ph3"), required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int)
    duration.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=CFG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--ph3-bf16", action="store_true", help="P-H3: train on BF16 copies of the int8 weights")
    parser.add_argument("--metrics", type=Path, help="also save the JSON summary to this new file")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = train(args)
    if args.metrics:
        common.publish_new(args.metrics, lambda path: path.write_text(json.dumps(result) + "\n", encoding="utf-8"))
    if args.summary_only:
        result.pop("step_ms")
        result.pop("step_loss")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
