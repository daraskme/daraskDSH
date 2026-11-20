"""Pieces shared by the native LoRA trainers (Anima here, MiniMax H3 / P-H3 in plugins/media-shared/native/train_lora.py):
captioned sample discovery, aspect-ratio buckets, picture loading, sample order, gradient clipping and atomic saving."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
AR_BUCKETS = tuple(2 ** (i / 4) for i in range(-4, 5))


@dataclass(frozen=True)
class Sample:
    image: Path  # a picture, or a video for the trainers that take them
    caption: str


def find_samples(folder, recursive=False, extensions=IMAGE_EXTENSIONS):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    paths = folder.rglob("*") if recursive else folder.iterdir()
    media = sorted(path for path in paths if path.suffix.lower() in extensions and path.is_file())
    if not media:
        raise ValueError(f"No supported images in {folder}")
    samples = []
    for path in media:
        sidecar = path.with_suffix(".txt")
        caption = sidecar.read_text(encoding="utf-8-sig").strip() if sidecar.is_file() else ""
        if not caption:
            raise ValueError(f"Image requires a nonempty UTF-8 caption: {sidecar}")
        samples.append(Sample(path, caption))
    return samples


def bucket_size(width, height, resolution, multiple=16):
    log_aspect = math.log(width / height)
    aspect = min(AR_BUCKETS, key=lambda ratio: abs(log_aspect - math.log(ratio)))
    scale = math.sqrt(aspect)
    return round(resolution * scale / multiple) * multiple, round(resolution / scale / multiple) * multiple


def fit_rgb(image, size):
    """PIL image -> RGB (transparency over white) center-cropped and resized to `size` (width, height)."""
    if image.mode == "RGBA" or "transparency" in image.info:
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        white.alpha_composite(image)
        image = white
    return ImageOps.fit(image.convert("RGB"), size)


def sample_order(size, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    while True:
        yield from torch.randperm(size, generator=generator).tolist()


def clip_gradients(params, max_norm=1.0):
    gradients = [p.grad for p in params if p.grad is not None]
    norms = torch.stack([norm.float() for norm in torch._foreach_norm(gradients, 2, dtype=torch.float32)])
    coefficient = (max_norm / (torch.linalg.vector_norm(norms) + 1e-6)).clamp(max=1)
    torch._foreach_mul_(gradients, coefficient)


def publish_new(path, write):
    """Write a new file through a temporary + fsync + hard link (never overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        write(temp_path)
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_lora(path, adapters, metadata=None):
    """{'<module>.lora_A.weight' | '<module>.lora_B.weight': tensor} -> BF16 'diffusion_model.' keys (ComfyUI / diffusion-pipe)."""
    from safetensors.torch import save_file
    parameters = {"diffusion_model." + key: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                  for key, value in adapters.items()}
    publish_new(path, lambda temporary: save_file(parameters, temporary, metadata={"format": "pt", **(metadata or {})}))
