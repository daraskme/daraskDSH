"""Input pictures for keyframes, references and edits: loading and the sizes each consumer sees.

Written from ComfyUI's LoadImage (nodes.py, decoded through PyAV like comfy_api VideoFromFile, PIL as the fallback),
comfy.utils.common_upscale / lanczos (PIL LANCZOS through 8 bit, optional center crop) and nodes_minimax_h3.py
(MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, PH3ImageConditioning). Pictures are FP32 [H, W, 3] in [0, 1] on the CPU.
"""
import math

import numpy as np
import torch

CANVAS = 32
PH3_SIZE = 2912
PH3_VISION_PIXELS = 1000000
EIGHT_BIT = ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8")


def load(path):
    """The first frame as RGB, like ComfyUI's LoadImage."""
    try:
        import av
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0))
            if frame.format.name in EIGHT_BIT or frame.format.name.startswith(("gray", "ya8")):
                return torch.from_numpy(np.divide(frame.to_ndarray(format="rgb24"), 255.0, dtype=np.float32))
            return torch.from_numpy(np.ascontiguousarray(frame.to_ndarray(format="gbrpf32le"), dtype=np.float32))
    except Exception:  # formats PyAV cannot open (animated WebP and the like)
        from PIL import Image, ImageOps
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)


def resize(image, width, height, crop=False):
    """common_upscale(lanczos): stretch, or with crop=True center-crop to the target aspect first."""
    from PIL import Image
    if crop:
        h, w = image.shape[0], image.shape[1]
        old, new = w / h, width / height
        x = round((w - w * (new / old)) / 2) if old > new else 0
        y = round((h - h * (old / new)) / 2) if old < new else 0
        image = image[y:h - y, x:w - x]
    pil = Image.fromarray(np.clip(255. * image.numpy(), 0, 255).astype(np.uint8))
    out = pil.resize((width, height), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(out).astype(np.float32) / 255.0)


def resize_to(image, size):
    """_resize_to: an image already at (width, height) stays untouched."""
    return image if (image.shape[1], image.shape[0]) == size else resize(image, *size)


def round_canvas(width, height, scale):
    return tuple(max(CANVAS, round(side * scale / CANVAS) * CANVAS) for side in (width, height))


def within_area(width, height, pixels):
    """_size_within_area: aspect-preserving scale (down only) to `pixels`."""
    return round_canvas(width, height, min(1.0, math.sqrt(pixels / (width * height))))


def h3_reference(image, width, height):
    """MiniMaxH3ReferenceToVideo, ref_image_size=match: down to the video's pixel area (the same picture feeds the
    vision tower and the VAE)."""
    h, w = image.shape[0], image.shape[1]
    return resize(image, *round_canvas(w, h, min(1.0, math.sqrt((width * height) / (w * h)))))


def ph3_sizes(image, width=0, height=0):
    """PH3ImageConditioning: (vision tower picture, VAE picture) for one input."""
    size = (image.shape[1], image.shape[0])
    area = (width * height) or PH3_SIZE ** 2
    return resize_to(image, within_area(*size, PH3_VISION_PIXELS)), resize_to(image, within_area(*size, area))
