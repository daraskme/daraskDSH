"""MiniMax H3 video VAE encoder (causal 3D convolutions, FP16) and audio VAE encoder (DAC + attention head, FP32),
for caching LoRA training latents.

Written from ComfyUI's comfy/ldm/minimax/vae.py (MiniMaxH3VideoVAE.encode: 17-frame clips, 256-px spatial tiles,
the last 3 latent steps dropped) and audio_vae.py (MiniMaxH3AudioVAE.encode). Same math and dtypes; ComfyUI's fused
kitchen kernels (NDHWC norm/pad, FP16-accumulate conv) are replaced by plain cuDNN, so results agree to FP16
rounding rather than bit for bit (test_encode_av_parity.py).
Weights: the safetensors keys as stored, on CUDA (video FP16, audio FP32).
"""
import math

import torch
import torch.nn.functional as F

import vae as video_vae

CLIP, TOKEN_DROP = 17, 3
HOP = 800  # audio samples per latent step (32 kHz -> 40 Hz)


# ---- video ---------------------------------------------------------------------------------------

def _norm_silu(w, p, x):
    # TemporalIsolatedGroupNorm: statistics per frame
    b, c, t, h, wd = x.shape
    y = F.group_norm(x.transpose(1, 2).reshape(b * t, c, h, wd), 32, w[p + ".weight"].to(x.dtype),
                     w[p + ".bias"].to(x.dtype), 1e-6)
    return F.silu(y.view(b, t, c, h, wd).transpose(1, 2))


def _causal_conv(w, p, x, stride=(1, 1, 1), spatial=(1, 1, 1, 1), time_pad=1):
    """CausalConv3d: reflect spatial padding, zero front-only temporal padding (2 * pad_t frames)."""
    weight = w[p + ".weight"]
    if x.shape[2] == 1 and weight.shape[2] > 1:  # one frame: drop the taps that would see the zero front padding
        weight = weight[:, :, -1:]
        time_pad = 0
    if any(spatial):
        x = F.pad(x, (*spatial, 0, 0), mode="reflect")
    if time_pad:
        x = F.pad(x, (0, 0, 0, 0, 2 * time_pad, 0))
    return F.conv3d(x, weight.to(x.dtype), w[p + ".bias"].to(x.dtype), stride=stride)


def _encode_moments(w, x):
    """[1, 3, T, H, W] normalized pixels -> [1, 48, T', H / 16, W / 16] moments."""
    h = _causal_conv(w, "encoder.conv_in", x)
    levels = 1 + max(int(k.split(".")[2]) for k in w if k.startswith("encoder.down."))
    for level in range(levels):
        blocks = 1 + max(int(k.split(".")[4]) for k in w if k.startswith(f"encoder.down.{level}.block."))
        for block in range(blocks):
            p = f"encoder.down.{level}.block.{block}"
            shortcut = _causal_conv(w, p + ".nin_shortcut", h, spatial=(0,) * 4, time_pad=0) \
                if p + ".nin_shortcut.weight" in w else h
            if p + ".conv1.weight" not in w:  # pruned to its shortcut (LightVAE Lite)
                h = shortcut
                continue
            inner = _causal_conv(w, p + ".conv1", _norm_silu(w, p + ".norm1", h))
            h = _causal_conv(w, p + ".conv2", _norm_silu(w, p + ".norm2", inner)) + shortcut
        p = f"encoder.down.{level}.downsample.conv"
        if p + ".weight" in w:
            time_stride = 2 if level in (1, 2) else 1  # time_down (1, 2, 2, 1, 1, 1)
            h = _causal_conv(w, p, h, stride=(time_stride, 2, 2), spatial=(0, 1, 0, 1))
    h = _causal_conv(w, "encoder.conv_out", _norm_silu(w, "encoder.norm_out", h))
    return F.conv3d(h, w["quant_conv.weight"].to(h.dtype), w["quant_conv.bias"].to(h.dtype))


def _tiled(w, x):
    y_idx, y_len, y_overlap = video_vae.split_tiles(x.shape[-2])
    x_idx, x_len, x_overlap = video_vae.split_tiles(x.shape[-1])
    rows = [[_encode_moments(w, x[..., i:i + a, j:j + b]) for j, b in zip(x_idx, x_len)] for i, a in zip(y_idx, y_len)]
    out_rows = []
    for i, row in enumerate(rows):
        out = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = video_vae.blend(rows[i - 1][j], tile, y_overlap[i - 1] // video_vae.RATIO, dim=-2)
            if j > 0:
                tile = video_vae.blend(row[j - 1], tile, x_overlap[j - 1] // video_vae.RATIO, dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., :-(y_overlap[i] // video_vae.RATIO), :]
            if j < len(row) - 1:
                tile = tile[..., :, :-(x_overlap[j] // video_vae.RATIO)]
            out.append(tile)
        out_rows.append(torch.cat(out, dim=-1))
    return torch.cat(out_rows, dim=-2)


@torch.no_grad()
def encode_video(w, frames):
    """[T, H, W, 3] RGB in [0, 1] (T a multiple of 17, or 1) -> normalized latent [1, 24, T_lat, H / 16, W / 16]."""
    x = frames.to("cuda", torch.float16).permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
    mean = torch.tensor(video_vae.IMAGENET_MEAN).to(x).view(1, 3, 1, 1, 1)
    std = torch.tensor(video_vae.IMAGENET_STD).to(x).view(1, 3, 1, 1, 1)
    x = x.add(1.0).mul_(0.5).sub_(mean).div_(std)
    if x.shape[2] == 1:
        moments = _tiled(w, x)[:, :, -1:]
    else:
        clips = []
        for i in range(math.ceil(x.shape[2] / CLIP)):
            clip = x[:, :, i * CLIP:(i + 1) * CLIP]
            if clip.shape[2] < CLIP:
                clip = torch.cat([clip, clip[:, :, -1:].repeat(1, 1, CLIP - clip.shape[2], 1, 1)], dim=2)
            clips.append(_tiled(w, clip))
        moments = torch.cat(clips, dim=2)[:, :, :-TOKEN_DROP]
    z = moments.float().chunk(2, dim=1)[0]
    lm = w["latents_mean"].half().view(1, -1, 1, 1, 1).to(z)
    ls = w["latents_std"].half().view(1, -1, 1, 1, 1).to(z)
    return (z - lm) / ls


# ---- audio ---------------------------------------------------------------------------------------

def _snake(x, alpha):
    t = torch.sin(alpha * x)
    return t.mul_(t).mul_((alpha + 1e-9).reciprocal()).add_(x)


def _conv1d(w, p, x, **kw):
    return F.conv1d(x, w[p + ".weight"], w[p + ".bias"], **kw)


def _residual(w, p, x, dilation):
    y = _snake(x, w[p + ".block.0.alpha"])
    y = _conv1d(w, p + ".block.1", y, dilation=dilation, padding=3 * dilation)
    y = _conv1d(w, p + ".block.3", _snake(y, w[p + ".block.2.alpha"]))
    pad = (x.shape[-1] - y.shape[-1]) // 2
    if pad > 0:
        x = x[..., pad:-pad]
    return y.add_(x)


def _layer_norm(w, p, x):
    return F.layer_norm(x, (x.shape[-1],), w[p + ".weight"], w[p + ".bias"], 1e-5)


def _attn_projection(w, x):
    """AttnProjection: [B, T, 2048] -> [B, T, 32]."""
    p = "pre_block"
    h = _layer_norm(w, p + ".norm1", x)
    b, n, c = h.shape
    bias = torch.cat((w[p + ".attn.q_bias"], w[p + ".attn.zero_k_bias"], w[p + ".attn.v_bias"]))
    q, k, v = F.linear(h, w[p + ".attn.qkv.weight"], bias).reshape(b, n, 3, 8, c // 8).permute(2, 0, 3, 1, 4).unbind(0)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    a = F.linear(F.adaptive_avg_pool1d(a.mean(dim=1), 32), w[p + ".attn.proj.weight"], w[p + ".attn.proj.bias"])
    x = F.linear(_layer_norm(w, p + ".norm3", x), w[p + ".proj.weight"], w[p + ".proj.bias"]).add_(a)
    m = _layer_norm(w, p + ".mlp.norm", _layer_norm(w, p + ".norm2", x))
    m = F.gelu(F.linear(m, w[p + ".mlp.w0.weight"], w[p + ".mlp.w0.bias"]), approximate="tanh") \
        .mul_(F.linear(m, w[p + ".mlp.w1.weight"], w[p + ".mlp.w1.bias"]))
    return x.add_(F.linear(m, w[p + ".mlp.w2.weight"], w[p + ".mlp.w2.bias"]))


@torch.no_grad()
def encode_audio(w, waveform):
    """[2, L] stereo float waveform at 32 kHz in [-1, 1] -> normalized latent [1, 32, 2, ceil(L / 800)]."""
    x = waveform.to("cuda", torch.float32)
    x = F.pad(x, (0, (-x.shape[-1]) % HOP)).reshape(2, 1, -1)
    x = _conv1d(w, "encoder.block.0", x, padding=3)
    for i, stride in enumerate((2, 4, 4, 5, 5), start=1):
        p = f"encoder.block.{i}.block"
        for j, dilation in enumerate((1, 3, 9)):
            x = _residual(w, f"{p}.{j}", x, dilation)
        x = _conv1d(w, f"{p}.4", _snake(x, w[f"{p}.3.alpha"]), stride=stride, padding=math.ceil(stride / 2))
    x = _conv1d(w, "encoder.block.7", _snake(x, w["encoder.block.6.alpha"]), padding=1)
    x = _attn_projection(w, x.transpose(1, 2)).transpose(1, 2)
    z = _conv1d(w, "mean_proj", x)
    z = (z - w["latents_mean"].view(1, -1, 1)) / w["latents_std"].view(1, -1, 1)
    return z.unsqueeze(0).permute(0, 2, 1, 3)  # [1, 32, 2, T]
