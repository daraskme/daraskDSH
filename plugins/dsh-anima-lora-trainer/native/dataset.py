"""Image and caption conditioning for the native Anima trainer."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AR_BUCKETS = tuple(2 ** (i / 4) for i in range(-4, 5))
CONTEXT_SHAPE = (1, 512, 1024)
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class ModelPaths:
    diffusion_pipe: Path
    diffusion: Path
    vae: Path
    text_encoder: Path


@dataclass(frozen=True)
class Sample:
    image: Path
    caption: str


def find_samples(folder, recursive=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    paths = folder.rglob("*") if recursive else folder.iterdir()
    images = sorted(path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())
    if not images:
        raise ValueError(f"No supported images in {folder}")
    samples = []
    for image in images:
        sidecar = image.with_suffix(".txt")
        caption = sidecar.read_text(encoding="utf-8-sig").strip() if sidecar.is_file() else ""
        if not caption:
            raise ValueError(f"Image requires a nonempty UTF-8 caption: {sidecar}")
        samples.append(Sample(image, caption))
    return samples


def bucket_size(width, height, resolution):
    log_aspect = math.log(width / height)
    aspect = min(AR_BUCKETS, key=lambda ratio: abs(log_aspect - math.log(ratio)))
    scale = math.sqrt(aspect)
    return round(resolution * scale / 16) * 16, round(resolution / scale / 16) * 16


def read_image(path, resolution):
    from torchvision.transforms.functional import to_tensor

    with Image.open(path) as source:
        size = bucket_size(*source.size, resolution)
        image = source
        if image.mode == "RGBA" or "transparency" in image.info:
            image = image.convert("RGBA")
            white = Image.new("RGBA", image.size, (255, 255, 255, 255))
            white.alpha_composite(image)
            image = white
        image = ImageOps.fit(image.convert("RGB"), size)
        return to_tensor(image)[None, :, None, :, :]


def load_reference(diffusion_pipe):
    root = Path(diffusion_pipe).resolve()
    if not (root / "models/cosmos_predict2_modeling.py").is_file():
        raise ValueError(f"Anima model definitions are missing from diffusion-pipe: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models import cosmos_predict2_modeling

    if not Path(cosmos_predict2_modeling.__file__).resolve().is_relative_to(root):
        raise ValueError(f"Anima model definitions are already loaded from another checkout: {cosmos_predict2_modeling.__file__}")
    return cosmos_predict2_modeling


def read_dit_checkpoint(path, *, conditioner=False):
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            name = key.removeprefix("net.").removeprefix("model.diffusion_model.")
            if name.endswith((".scale_weight", ".scale_input")):
                raise ValueError(f"Expected unquantized Anima weights: {path}")
            is_conditioner = name.startswith("llm_adapter.")
            if conditioner != is_conditioner:
                continue
            if name in tensors:
                raise ValueError(f"Duplicate Anima checkpoint key: {name}")
            tensors[name] = checkpoint.get_tensor(key)
    if not tensors:
        raise ValueError(f"Missing {'LLM adapter' if conditioner else 'DiT'} weights in {path}")
    return tensors


def _load_vae(path, device):
    from models.wan.vae2_1 import WanVAE_

    # Wan2.1 VAE settings and Anima latent normalization from CosmosPredict2Pipeline.
    with torch.device("meta"):
        vae = WanVAE_(dim=96, z_dim=16, temperal_downsample=[False, True, True])
    vae.load_state_dict(load_file(path), assign=True)
    vae.eval().requires_grad_(False).to(device)
    mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    scale = (torch.tensor(mean, device=device, dtype=DTYPE),
             1.0 / torch.tensor(std, device=device, dtype=DTYPE))
    return vae, scale


def _load_text_models(paths, device):
    from accelerate import init_empty_weights
    from models.llm_adapter import LLMAdapter
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

    configs = paths.diffusion_pipe / "configs"
    qwen_path = configs / "qwen3_06b"
    t5_path = configs / "t5_old"
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    t5_tokenizer = T5TokenizerFast(vocab_file=str(t5_path / "spiece.model"),
                                  tokenizer_file=str(t5_path / "tokenizer.json"))
    config = Qwen3Config.from_pretrained(qwen_path, local_files_only=True)
    config.use_cache = False
    with init_empty_weights():
        encoder = Qwen3Model(config)
        adapter = LLMAdapter(1024, 1024, 1024, num_layers=6, self_attn=True)
    with safe_open(str(paths.text_encoder), framework="pt", device="cpu") as checkpoint:
        encoder.load_state_dict({key.removeprefix("model."): checkpoint.get_tensor(key).to(DTYPE)
                                 for key in checkpoint.keys() if key != "lm_head.weight"}, assign=True)
    adapter.load_state_dict({key.removeprefix("llm_adapter."): value.to(DTYPE)
                             for key, value in read_dit_checkpoint(paths.diffusion, conditioner=True).items()},
                            assign=True)
    return (qwen_tokenizer, t5_tokenizer, encoder.eval().requires_grad_(False).to(device),
            adapter.eval().requires_grad_(False).to(device))


@torch.no_grad()
def encode_caption(caption, text_models, device):
    qwen_tokenizer, t5_tokenizer, encoder, adapter = text_models
    args = dict(return_tensors="pt", truncation=True, padding="max_length", max_length=CONTEXT_SHAPE[1])
    qwen = qwen_tokenizer([caption], **args).to(device)
    t5 = t5_tokenizer([caption], **args).to(device)
    hidden = encoder(input_ids=qwen.input_ids, attention_mask=qwen.attention_mask).last_hidden_state
    hidden.masked_fill_(~qwen.attention_mask.bool().unsqueeze(-1), 0)
    context = adapter(source_hidden_states=hidden, target_input_ids=t5.input_ids,
                      target_attention_mask=t5.attention_mask, source_attention_mask=qwen.attention_mask)
    return context.masked_fill_(~t5.attention_mask.bool().unsqueeze(-1), 0).to(device="cpu", dtype=DTYPE).contiguous()


def _file_identity(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _cache_dependencies(paths):
    from importlib.metadata import version

    root = paths.diffusion_pipe
    files = [Path(__file__), paths.diffusion, paths.vae, paths.text_encoder,
             root / "models/wan/vae2_1.py", root / "models/llm_adapter.py"]
    files.extend(sorted((root / "configs/qwen3_06b").glob("*")))
    files.extend(sorted((root / "configs/t5_old").glob("*")))
    return [torch.__version__, version("transformers"), *map(_file_identity, files)]


def _cache_path(directory, cache_dependencies, sample, image_identity, resolution):
    fingerprint = json.dumps([cache_dependencies, image_identity, sample.caption, resolution])
    return directory / (sha256(fingerprint.encode("utf-8")).hexdigest() + ".safetensors")


def _read_cache(path):
    try:
        data = load_file(path)
        latents, context = data["latents"], data["context"]
        if context.shape == CONTEXT_SHAPE and context.dtype == DTYPE and latents.dtype == DTYPE:
            if latents.ndim == 5 and latents.shape[:3] == (1, 16, 1) and all(x > 0 and x % 2 == 0 for x in latents.shape[-2:]):
                return latents, context
    except (FileNotFoundError, SafetensorError, KeyError):
        pass
    return None


def _save_cache(path, latents, context):
    with NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        save_file({"latents": latents, "context": context}, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


@torch.no_grad()
def prepare_dataset(samples, resolution, paths, device, cache_dir=None):
    """Return CPU latents and 512-token masked conditioning plus the number of cache hits."""
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    dependencies = _cache_dependencies(paths) if cache is not None else None

    latents, contexts = [], []
    vae = text_models = None
    cache_hits = 0
    for sample in samples:
        image_identity = _file_identity(sample.image) if cache is not None else None
        cache_path = _cache_path(cache, dependencies, sample, image_identity, resolution) if cache is not None else None
        cached = _read_cache(cache_path) if cache_path is not None else None
        if cached is not None:
            latent, context = cached
            cache_hits += 1
        else:
            if vae is None:
                load_reference(paths.diffusion_pipe)
                vae, scale = _load_vae(paths.vae, device)
                text_models = _load_text_models(paths, device)
            pixels = read_image(sample.image, resolution).to(device=device, dtype=next(vae.parameters()).dtype)
            latent = vae.encode(pixels * 2 - 1, scale).to(device="cpu", dtype=DTYPE).contiguous()
            context = encode_caption(sample.caption, text_models, device)
            if cache_path is not None and image_identity == _file_identity(sample.image):
                _save_cache(cache_path, latent, context)
        latents.append(latent)
        contexts.append(context)
    return latents, contexts, cache_hits
