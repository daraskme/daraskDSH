"""MiniMax H3 video VAE decoder (ViT3D, FP16) for the T2V latents, and the P-H3 image decoder.

Written from ComfyUI's comfy/ldm/minimax/vae.py (MiniMaxH3VideoVAE.decode: latent statistics, 256-px spatial
tiles in rows of up to 4, 17-frame temporal chunks with a 5-frame blend) and comfy/sd.py (FP16, flash-first SDPA,
no FP16 accumulation); same dtypes and order as ComfyUI (bit-exact, see test_vae_parity.py).
Weights: the safetensors keys as stored (decoder.*, post_quant_conv.*, latents_mean/std), FP16, on CUDA.
"""
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SDPA_PRIORITY = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PH3_IMAGE_FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0
RATIO, RATIO_T = 16, 4
CLIP, TOKEN_DROP, TILE, TILE_OVERLAP = 17, 3, 256, 64
PRE_PAD = (-CLIP) % RATIO_T                      # 3
CHUNK_TOKENS = math.ceil(CLIP / RATIO_T)         # 5
TOKEN_OVERLAP = (-TOKEN_DROP) % CHUNK_TOKENS     # 2
FRAME_OVERLAP = max(TOKEN_OVERLAP * RATIO_T - PRE_PAD, 0)  # 5


def token_ids(dims, device, dtype):
    coords = [2.0 * (torch.arange(0.5, n, dtype=dtype, device=device) / n) - 1.0 for n in dims]
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1).flatten(0, len(dims) - 1).unsqueeze(0)


def rope_table(ids, dim_head=64, ratio=0.75, theta=100.0):
    n_dim = ids.shape[-1]
    # the module buffer is cast to the VAE dtype with the rest of the model
    inv_freq = (1 / theta ** torch.arange(0, 1, 2 * n_dim / int(dim_head * ratio), dtype=torch.float32)).to(ids.dtype).to(ids.device)
    angles = (2.0 * math.pi * ids[:, :, :, None].float() * inv_freq[None, None, None, :]).flatten(2, 3)
    c, s = torch.cos(angles), torch.sin(angles)
    return torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2).to(ids.dtype)


def vit(w, x, ids, patch_t, heads=32, dim_head=64):
    """ViT3DDecoder.forward: [B, 24, T, H, W] -> [B, 3, T * patch_t, H * 16, W * 16] (raw, normalized pixels)."""
    B, _, T, H, W = x.shape
    h = F.linear(x.flatten(2).transpose(1, 2), w["decoder.x_embedder.weight"], w["decoder.x_embedder.bias"])
    patches = h.shape[1]
    h = torch.cat([h, w["decoder.register_tokens"].expand(B, -1, -1), torch.zeros_like(h[:, 0:1, :])], dim=1)
    ids = torch.cat([ids.expand(B, -1, -1), torch.zeros((B, 5, ids.shape[-1]), device=x.device, dtype=ids.dtype)], dim=1)
    table = rope_table(ids, dim_head)
    ones = torch.ones(dim_head, device=x.device, dtype=x.dtype)
    dim = heads * dim_head
    for i in range(sum(k.endswith(".scale1") for k in w)):
        p = f"decoder.transformer_blocks.{i}."
        qkv = F.linear(F.rms_norm(h, (dim,), w[p + "norm1.weight"], 1e-5), w[p + "attn.to_qkv.weight"], w[p + "attn.to_qkv.bias"])
        q, k, v = qkv.view(B, h.shape[1], heads, 3 * dim_head).chunk(3, dim=-1)
        torch.ops.dsh_kitchen.rms_rope_split_half_(q, k, table, ones, ones, 1e-5, table.shape[-3] * 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if q.nelement() < 1024 * 128:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
                out = F.scaled_dot_product_attention(q, k, v)
        out = torch.nan_to_num(out.transpose(1, 2).reshape(B, -1, dim))
        h = torch.addcmul(h, F.linear(out, w[p + "attn.to_out.weight"], w[p + "attn.to_out.bias"]), w[p + "scale1"])
        gate, up = F.linear(F.rms_norm(h, (dim,), w[p + "norm2.weight"], 1e-5), w[p + "ff.w1.weight"], w[p + "ff.w1.bias"]).chunk(2, -1)
        h = torch.addcmul(h, F.linear(F.silu(gate).mul_(up), w[p + "ff.w2.weight"], w[p + "ff.w2.bias"]), w[p + "scale2"])
    out = F.linear(F.layer_norm(h, (dim,), w["decoder.norm_out.weight"], w["decoder.norm_out.bias"], 1e-5),
                   w["decoder.proj_out.weight"], w["decoder.proj_out.bias"])[:, :patches]
    out = out.view(B, T, H, W, 3, patch_t, RATIO, RATIO)
    return out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, 3, T * patch_t, H * RATIO, W * RATIO)


def decode_pixels(w, z):
    """post_quant_conv + the video ViT on one (batched) clip."""
    z = F.conv3d(z, w["post_quant_conv.weight"], w["post_quant_conv.bias"])
    return vit(w, z, token_ids(z.shape[2:], z.device, z.dtype), RATIO_T)


def split_tiles(n):
    if TILE >= n:
        return [0], [n], []
    count = math.ceil(n / TILE)
    while True:
        overlaps = [TILE_OVERLAP] * (count - 1)
        remaining = TILE * count - sum(overlaps) - n
        if remaining >= 0:
            break
        count += 1
    for i in range(remaining // RATIO):
        overlaps[i % (count - 1)] += RATIO
    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + TILE - overlaps[i])
    return starts, [TILE] * count, overlaps


def blend(a, b, extent, dim):
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * (1 - positions / extent).view(shape) + \
        b.narrow(dim, 0, extent) * (positions / extent).view(shape)
    if extent < b.shape[dim]:
        return torch.cat([blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim)
    return blended


def tiled_decode(w, z, pixels=decode_pixels):
    height, width = z.shape[-2] * RATIO, z.shape[-1] * RATIO
    y_idx, y_len, y_overlap = split_tiles(height)
    x_idx, x_len, x_overlap = split_tiles(width)
    canvas = strip = None
    out_y = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        row = z[..., i_pos // RATIO:(i_pos + i_len) // RATIO, :]
        slices = [row[..., p // RATIO:(p + n) // RATIO] for p, n in zip(x_idx, x_len)]
        tiles = [t for k in range(0, len(slices), 4) for t in pixels(w, torch.cat(slices[k:k + 4])).chunk(len(slices[k:k + 4]))]
        new_strip = left_tail = None
        out_x = 0
        for j, tile in enumerate(tiles):
            if i > 0:
                tile = blend(strip[..., :, x_idx[j]:x_idx[j] + x_len[j]], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            last_x = j == len(x_idx) - 1
            left_tail = None if last_x else tile[..., :, -x_overlap[j]:].clone()
            if not last_x:
                tile = tile[..., :, :-x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
            if i < len(y_idx) - 1:
                if new_strip is None:
                    new_strip = torch.empty(*tile.shape[:-2], y_overlap[i], width, dtype=tile.dtype, device=tile.device)
                new_strip[..., :, out_x:out_x + tile.shape[-1]] = tile[..., -y_overlap[i]:, :]
                tile = tile[..., :-y_overlap[i], :]
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            tile_height = tile.shape[-2]
            out_x += tile.shape[-1]
        strip = new_strip
        out_y += tile_height
    return canvas


def finalize(part):
    std = torch.tensor(IMAGENET_STD, dtype=torch.float16).float().view(1, 3, 1, 1, 1).to(part.device)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float16).float().view(1, 3, 1, 1, 1).to(part.device)
    return (part * std).add_(mean).clamp_(0.0, 1.0)


def denormalize(w, z):
    return z.to(torch.float16) * w["latents_std"].view(1, -1, 1, 1, 1).to(torch.float16) + \
        w["latents_mean"].view(1, -1, 1, 1, 1).to(torch.float16)


def decode_video(w, z):
    """[1, 24, T, H, W] KSampler latent -> FP32 RGB [frames, H * 16, W * 16, 3] in [0, 1] (decode_temporal)."""
    z = denormalize(w, z)
    pseudo = z.shape[2] + TOKEN_DROP
    pad = (-pseudo) % CHUNK_TOKENS
    chunks = (pseudo + pad) // CHUNK_TOKENS - 1
    if chunks < 1:
        pad += CHUNK_TOKENS
        chunks += 1
    assert pad == 0, "H3 latents are 5k + 2 frames long"  # no pad-frame trimming needed for the T2V grid
    parts, overlap = [], None
    chunk_frames = CHUNK_TOKENS * RATIO_T
    for i in range(chunks):
        clip = tiled_decode(w, z[:, :, i * CHUNK_TOKENS:i * CHUNK_TOKENS + CHUNK_TOKENS + TOKEN_OVERLAP])
        for j in range(2):
            part = clip[:, :, j * chunk_frames:min(j * chunk_frames + chunk_frames, clip.shape[2])][:, :, PRE_PAD:]
            if j == 0:
                if overlap is not None:
                    part, overlap = blend(overlap, part, FRAME_OVERLAP, dim=-3), None
                parts.append(finalize(part))
            else:
                overlap = part.contiguous()
        if i == chunks - 1 and overlap is not None:
            parts.append(finalize(overlap))
    return torch.cat(parts, dim=2)[0].permute(1, 2, 3, 0)


def ph3_image_pixels(w, z):
    """PH3ImageDecoder on [B, 24, 1, H, W]: one frame per latent at a fixed rope time."""
    z = F.conv3d(z, w["post_quant_conv.weight"], w["post_quant_conv.bias"])
    ids = token_ids(z.shape[3:], z.device, z.dtype)
    ids = torch.cat([torch.full_like(ids[..., :1], PH3_IMAGE_FRAME_POSITION), ids], dim=-1)
    return vit(w, z, ids, 1)


# ---- encoder (keyframes and reference images: always one frame) ----------------------------------------------------
# EncoderFCN3D on a single frame: every causal 3x3x3 conv keeps only its last temporal tap (ComfyUI's "causal_zero"),
# so it is a 2D conv; GroupNorm statistics are per frame. FP16 like ComfyUI.

IMAGE_ENCODE_SEED = 42


def _conv(w, p, x, pad=None):
    weight = w[p + ".weight"]
    if weight.dim() == 5:
        weight = weight[:, :, -1]
    if pad:
        x = F.pad(x, pad, mode="reflect")
    return F.conv2d(x, weight.to(x.dtype), w[p + ".bias"].to(x.dtype), stride=1)


def _norm_silu(w, p, x):
    return F.silu(F.group_norm(x, 32, w[p + ".weight"].to(x.dtype), w[p + ".bias"].to(x.dtype), 1e-6))


def encode_moments(w, x):
    """[1, 3, H, W] normalized pixels -> [1, 48, H / 16, W / 16] moments."""
    h = _conv(w, "encoder.conv_in", x, (1, 1, 1, 1))
    levels = 1 + max(int(k.split(".")[2]) for k in w if k.startswith("encoder.down."))
    for level in range(levels):
        blocks = 1 + max(int(k.split(".")[4]) for k in w if k.startswith(f"encoder.down.{level}.block."))
        for block in range(blocks):
            p = f"encoder.down.{level}.block.{block}"
            shortcut = _conv(w, p + ".nin_shortcut", h) if p + ".nin_shortcut.weight" in w else h
            if p + ".conv1.weight" not in w:  # pruned to its shortcut (LightVAE Lite)
                h = shortcut
                continue
            inner = _conv(w, p + ".conv1", _norm_silu(w, p + ".norm1", h), (1, 1, 1, 1))
            h = _conv(w, p + ".conv2", _norm_silu(w, p + ".norm2", inner), (1, 1, 1, 1)) + shortcut
        p = f"encoder.down.{level}.downsample.conv"
        if p + ".weight" in w:  # every downsampling level halves H and W (the time stride is moot for one frame)
            h = F.conv2d(F.pad(h, (0, 1, 0, 1), mode="reflect"), w[p + ".weight"][:, :, -1].to(h.dtype),
                         w[p + ".bias"].to(h.dtype), stride=2)
    h = _conv(w, "encoder.conv_out", _norm_silu(w, "encoder.norm_out", h), (1, 1, 1, 1))
    return F.conv2d(h, w["quant_conv.weight"][:, :, 0].to(h.dtype), w["quant_conv.bias"].to(h.dtype))


def tiled_encode(w, x):
    """MiniMaxH3VideoVAE.tiled_encode on one frame: 256-px tiles, blended in latent space."""
    y_idx, y_len, y_overlap = split_tiles(x.shape[-2])
    x_idx, x_len, x_overlap = split_tiles(x.shape[-1])
    rows = [[encode_moments(w, x[..., i:i + a, j:j + b]) for j, b in zip(x_idx, x_len)] for i, a in zip(y_idx, y_len)]
    out_rows = []
    for i, row in enumerate(rows):
        out = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(rows[i - 1][j], tile, y_overlap[i - 1] // RATIO, dim=-2)
            if j > 0:
                tile = blend(row[j - 1], tile, x_overlap[j - 1] // RATIO, dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., :-(y_overlap[i] // RATIO), :]
            if j < len(row) - 1:
                tile = tile[..., :, :-(x_overlap[j] // RATIO)]
            out.append(tile)
        out_rows.append(torch.cat(out, dim=-1))
    return torch.cat(out_rows, dim=-2)


def encode_image(w, pixels, sample=False):
    """[H, W, 3] RGB in [0, 1] (already at the target size) -> normalized latent [1, 24, 1, H / 16, W / 16].

    sample=False: the H3 keyframe/reference encode (posterior mean). sample=True: the p-h3-image encode
    (mean + std * noise from a CPU generator seeded 42)."""
    return encode_signed(w, (pixels.to("cuda", torch.float16) * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0), sample)


def encode_signed(w, x, sample=False):
    """[1, 3, H, W] FP16 pixels in [-1, 1] (ComfyUI's VAE input) -> normalized latent."""
    mean = torch.tensor(IMAGENET_MEAN).to(x).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).to(x).view(1, 3, 1, 1)
    x = x.add(1.0).mul_(0.5).sub_(mean).div_(std)
    moments = tiled_encode(w, x).float().unsqueeze(2)
    mean_z, logvar = moments.chunk(2, dim=1)
    if sample:
        noise = torch.randn(mean_z.shape, generator=torch.Generator().manual_seed(IMAGE_ENCODE_SEED)).to(mean_z.device)
        mean_z = torch.addcmul(mean_z, torch.exp(0.5 * logvar.clamp(-30.0, 20.0)), noise)
    # the statistics are module buffers, cast to the VAE's FP16 with the rest of it
    lm = w["latents_mean"].half().view(1, -1, 1, 1, 1).to(mean_z)
    ls = w["latents_std"].half().view(1, -1, 1, 1, 1).to(mean_z)
    return (mean_z - lm) / ls
