"""Build the frozen Anima frontend modules for the standalone C++ trainer."""

import argparse
import ctypes
import fcntl
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch
from torch import nn

if __package__:
    from . import dataset
else:
    import dataset


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRONTEND = ROOT / ".runtime/anima-native/frontend"
DEFAULT_CONFIG = Path("~/.config/comfy-folder-lora/config.json").expanduser()
ARCHIVE_NAMES = {"vae.pt", "conditioner.pt", "metadata.json"}
TOKENIZER_NAMES = ("qwen_tokenizer", "qwen_tokenizer_config", "t5_tokenizer")
DEVICE = torch.device("cuda:0")


class ImageEncoder(nn.Module):
    def __init__(self, vae, scale):
        super().__init__()
        self.encoder = vae.encoder
        self.projection = vae.conv1
        self.register_buffer("mean", scale[0].reshape(1, 16, 1, 1, 1))
        self.register_buffer("inverse_std", scale[1].reshape(1, 16, 1, 1, 1))
        self.compute_dtype = vae.conv1.weight.dtype

    def forward(self, pixels):
        # The first frame of WanVAE_ uses an empty temporal feature cache.
        pixels = pixels.to(self.compute_dtype) * 2 - 1
        mean, _ = self.projection(self.encoder(pixels)).chunk(2, dim=1)
        return ((mean - self.mean) * self.inverse_std).to(torch.bfloat16)


class QwenEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        length = dataset.CONTEXT_SHAPE[1]
        causal = torch.ones(length, length, device=DEVICE, dtype=torch.bool).tril()
        self.register_buffer("causal", causal[None, None])

    def forward(self, qwen_ids, qwen_mask):
        # Supply the dynamic mask directly to SDPA. Transformers' ordinary
        # 2-D masking path has data-dependent branches during tracing.
        allowed = self.causal & qwen_mask.bool()[:, None, None, :]
        return self.encoder(
            input_ids=qwen_ids, attention_mask={"full_attention": allowed}, use_cache=False, is_causal=False
        ).last_hidden_state

    def full_forward(self, qwen_ids):
        # An unpadded sequence uses the reference causal GQA kernel.
        return self.encoder(
            input_ids=qwen_ids, attention_mask={"full_attention": None}, use_cache=False, is_causal=True
        ).last_hidden_state


class CaptionConditioner(nn.Module):
    def __init__(self, encoder, adapter):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, qwen_ids, qwen_mask, t5_ids, t5_mask):
        if bool(qwen_mask.all()):
            hidden = self.encoder.full_forward(qwen_ids)
        else:
            hidden = self.encoder(qwen_ids, qwen_mask)
        hidden = hidden.masked_fill(~qwen_mask.to(torch.bool).unsqueeze(-1), 0)
        context = self.adapter(hidden, t5_ids, t5_mask, qwen_mask)
        return context.masked_fill(~t5_mask.to(torch.bool).unsqueeze(-1), 0).to(torch.bfloat16).contiguous()


def save_trace(module, inputs, path):
    traced = torch.jit.trace(module.eval(), inputs, check_trace=False)
    return save_archive(traced, path)


def save_archive(module, path):
    torch.jit.freeze(module.eval(), optimize_numerics=False).save(str(path))
    return torch.jit.load(str(path), map_location=DEVICE).eval()


def assert_equivalent(actual, expected, shape, label):
    if actual.shape != shape or actual.dtype != torch.bfloat16 or not torch.isfinite(actual).all():
        raise RuntimeError(f"Invalid {label} output: {actual.shape}, {actual.dtype}")
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
    if not torch.isfinite(error) or error > 0.01:
        raise RuntimeError(f"{label} export disagrees with the eager model (relative L2 {error.item():.5g})")
    return error.item()


@torch.no_grad()
def export_image_encoder(paths, destination):
    vae, scale = dataset._load_vae(paths.vae, DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(5602)
    shapes = [(64, 96), (128, 128), (448, 576)]
    images = [
        torch.rand((1, 3, 1, height, width), device=DEVICE, generator=generator)
        for height, width in shapes
    ]
    encoder = ImageEncoder(vae, scale)
    archive = save_trace(encoder, (images[0],), destination)
    error = 0
    for (height, width), image in zip(shapes, images, strict=True):
        eager = vae.encode(image.to(encoder.compute_dtype) * 2 - 1, scale).to(torch.bfloat16)
        with torch.jit.optimized_execution(False):
            actual = archive(image)
        error = max(error, assert_equivalent(
            actual, eager, (1, 16, 1, height // 8, width // 8), "VAE"
        ))
    return error


def caption_inputs(qwen_tokenizer, t5_tokenizer, caption):
    options = dict(return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    qwen = qwen_tokenizer([caption], **options).to(DEVICE)
    t5 = t5_tokenizer([caption], **options).to(DEVICE)
    return qwen.input_ids, qwen.attention_mask, t5.input_ids, t5.attention_mask


@torch.no_grad()
def export_caption_conditioner(paths, destination):
    qwen_tokenizer, t5_tokenizer, encoder, adapter = dataset._load_text_models(paths, DEVICE)
    if encoder.config._attn_implementation != "sdpa" or encoder.has_sliding_layers:
        raise RuntimeError("The native frontend requires the Qwen3-0.6B SDPA configuration")
    first = caption_inputs(qwen_tokenizer, t5_tokenizer, "portrait, window light")
    other = caption_inputs(qwen_tokenizer, t5_tokenizer, "青い空の下で歩く二人。 A long street lined with trees.")
    qwen_ids, qwen_mask, t5_ids, t5_mask = first
    fully_padded = torch.zeros_like(qwen_mask)
    no_padding = torch.ones_like(qwen_mask)
    cases = {
        "short caption": first,
        "multilingual caption": other,
        "all padding": (qwen_ids, fully_padded, t5_ids, fully_padded),
        "empty Qwen context": (qwen_ids, fully_padded, t5_ids, t5_mask),
        "no padding": (qwen_ids, no_padding, t5_ids, no_padding),
    }
    qwen = torch.jit.trace_module(
        QwenEncoder(encoder).eval(),
        {"forward": (qwen_ids, qwen_mask), "full_forward": (qwen_ids,)},
        check_trace=False,
    )
    conditioned = torch.jit.trace(
        adapter, (qwen(qwen_ids, qwen_mask), t5_ids, t5_mask, qwen_mask), check_trace=False,
    )
    archive = save_archive(torch.jit.script(CaptionConditioner(qwen, conditioned)), destination)
    error = 0
    for label, (qwen_ids, qwen_mask, t5_ids, t5_mask) in cases.items():
        hidden = encoder(input_ids=qwen_ids, attention_mask=qwen_mask, use_cache=False).last_hidden_state
        hidden = hidden.masked_fill(~qwen_mask.bool().unsqueeze(-1), 0)
        expected = adapter(
            source_hidden_states=hidden, target_input_ids=t5_ids,
            target_attention_mask=t5_mask, source_attention_mask=qwen_mask,
        ).masked_fill(~t5_mask.bool().unsqueeze(-1), 0)
        # Match the C++ frontend. The profiling executor can change how the
        # caption model rounds BF16 values even with unchanged attention masks.
        with torch.jit.optimized_execution(False):
            actual = archive(qwen_ids, qwen_mask, t5_ids, t5_mask)
        error = max(error, assert_equivalent(actual, expected, dataset.CONTEXT_SHAPE, label))
        if torch.count_nonzero(actual.masked_select(~t5_mask.bool().unsqueeze(-1))):
            raise RuntimeError("Caption conditioner output contains unmasked padding")
    return error


def input_files(paths):
    configs = paths.diffusion_pipe / "configs"
    qwen = configs / "qwen3_06b"
    t5 = configs / "t5_old"
    return {
        "diffusion": paths.diffusion,
        "vae": paths.vae,
        "text_encoder": paths.text_encoder,
        "qwen_config": qwen / "config.json",
        "qwen_tokenizer": qwen / "tokenizer.json",
        "qwen_tokenizer_config": qwen / "tokenizer_config.json",
        "qwen_merges": qwen / "merges.txt",
        "qwen_vocab": qwen / "vocab.json",
        "t5_tokenizer": t5 / "tokenizer.json",
        "t5_spiece": t5 / "spiece.model",
        "vae_source": paths.diffusion_pipe / "models/wan/vae2_1.py",
        "llm_adapter_source": paths.diffusion_pipe / "models/llm_adapter.py",
        "dataset_source": Path(dataset.__file__),
        "export_source": Path(__file__),
    }


def identify(path):
    path = path.resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def publish(stage, destination):
    if not destination.exists():
        stage.rename(destination)
        return
    # Exchange directories atomically so concurrent readers can open either
    # complete pack. TemporaryDirectory then reclaims the previous pack.
    libc = ctypes.CDLL(None, use_errno=True)
    exchange = libc.renameat2
    exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    exchange.restype = ctypes.c_int
    if exchange(-100, os.fsencode(stage), -100, os.fsencode(destination), 2):
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), str(destination))


def export(paths, destination):
    files = input_files(paths)
    identities = {name: identify(path) for name, path in files.items()}
    versions = {"format": 1, "torch": str(torch.__version__), "transformers": version("transformers")}
    artifact_names = ARCHIVE_NAMES | {name + ".json" for name in TOKENIZER_NAMES}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name(destination.name + ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise ValueError(f"Frontend directory cannot be a symlink: {destination}")
        if destination.exists():
            entries = {path.name for path in destination.iterdir()}
            if entries - artifact_names or (entries and "metadata.json" not in entries):
                raise ValueError(f"Existing directory contains files outside the Anima frontend: {destination}")
            if "metadata.json" in entries:
                old = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
                if (
                    all(old.get(key) == value for key, value in versions.items())
                    and all(
                        all(old.get("inputs", {}).get(name, {}).get(key) == value for key, value in identity.items())
                        for name, identity in identities.items()
                    )
                    and all((destination / name).is_file() for name in artifact_names)
                ):
                    print(f"Anima frontend is current at {destination}", flush=True)
                    return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to export the frozen Anima frontend")
        dataset.load_reference(paths.diffusion_pipe)
        with TemporaryDirectory(dir=destination.parent, prefix=".anima-frontend-") as temporary:
            stage = Path(temporary) / "pack"
            stage.mkdir()
            for name, path in files.items():
                with path.open("rb") as source:
                    identities[name]["sha256"] = file_digest(source, "sha256").hexdigest()
            for name in TOKENIZER_NAMES:
                shutil.copyfile(files[name], stage / (name + ".json"))
            image_error = export_image_encoder(paths, stage / "vae.pt")
            print(f"VAE validated (worst relative L2 {image_error:.6g})", flush=True)
            context_error = export_caption_conditioner(paths, stage / "conditioner.pt")
            print(f"Caption conditioner validated (worst relative L2 {context_error:.6g})", flush=True)
            for name, path in files.items():
                if any(identities[name][key] != value for key, value in identify(path).items()):
                    raise RuntimeError(f"An input changed during export: {path}")
            (stage / "metadata.json").write_text(
                json.dumps({**versions, "inputs": identities}, indent=2) + "\n", encoding="utf-8"
            )
            publish(stage, destination)
    print(f"Exported the frozen Anima frontend to {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=os.environ.get("COMFY_FOLDER_LORA_CONFIG", DEFAULT_CONFIG))
    parser.add_argument("--frontend-dir", type=Path, default=DEFAULT_FRONTEND)
    parser.add_argument("--diffusion-pipe", type=Path)
    parser.add_argument("--diffusion", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser()
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    anima = config.get("models", {}).get("anima", {})
    chosen = (
        args.diffusion_pipe or config.get("diffusionPipe"),
        args.diffusion or anima.get("diffusion"),
        args.vae or anima.get("vae"),
        args.text_encoder or anima.get("textEncoder"),
    )
    if not all(chosen):
        parser.error("Configure Anima or supply --diffusion-pipe, --diffusion, --vae and --text-encoder")
    paths = dataset.ModelPaths(*(Path(path).expanduser().resolve(strict=True) for path in chosen))
    export(paths, args.frontend_dir.expanduser().absolute())


if __name__ == "__main__":
    main()
