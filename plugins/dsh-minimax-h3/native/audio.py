"""MiniMax H3 audio VAE decoder (BigVGAN, FP32, 32 kHz stereo) and ComfyUI's VAEDecodeAudio loudness step.

Written from ComfyUI's comfy/ldm/minimax/audio_vae.py (MiniMaxH3AudioVAE.decode) and comfy_extras/nodes_audio.py
(vae_decode_audio); same dtypes and order (bit-exact, see test_vae_parity.py).
Weights: the safetensors keys as stored (dec_in_proj.*, decoder.*, latents_mean/std), FP32, on CUDA.
"""
import math

import torch
import torch.nn.functional as F

SAMPLE_RATE = 32000
UPSAMPLE_RATES = (5, 5, 2, 2, 2, 2, 2)
UPSAMPLE_KERNELS = (9, 9, 4, 4, 4, 4, 4)
RESBLOCK_KERNELS = (3, 7, 11)
DILATIONS = (1, 3, 5)


def kaiser_sinc_filter(cutoff, half_width, kernel_size=12):
    half = kernel_size // 2
    delta_f = 4 * half_width
    a = 2.285 * (half - 1) * math.pi * delta_f + 7.95
    beta = 0.1102 * (a - 8.7) if a > 50.0 else 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21.0) if a >= 21.0 else 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half, half) + 0.5
    filt = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filt /= filt.sum()
    return filt.view(1, 1, kernel_size)


FILTER = kaiser_sinc_filter(0.25, 0.3)  # up and down sampling x2 share the same design


def snake_beta(w, p, x):
    alpha = torch.exp(w[p + ".alpha"]).view(1, -1, 1)
    beta = torch.exp(w[p + ".beta"]).view(1, -1, 1)
    t = torch.sin(alpha * x)
    return t.mul_(t).mul_((beta + 1e-9).reciprocal()).add_(x)


def activation(w, p, x):
    """Anti-aliased SnakeBeta: upsample x2, activate, low-pass and downsample x2."""
    c = x.shape[1]
    filt = FILTER.to(x).expand(c, -1, -1)
    x = F.conv_transpose1d(F.pad(x, (5, 5), mode="replicate"), filt, stride=2, groups=c).mul_(2)[..., 15:-15]
    x = snake_beta(w, p + ".act", x)
    return F.conv1d(F.pad(x, (5, 6), mode="replicate"), filt, stride=2, groups=c)


def amp_block(w, p, x, kernel):
    for i, d in enumerate(DILATIONS):
        xt = activation(w, f"{p}.activations.{2 * i}", x)
        xt = F.conv1d(xt, w[f"{p}.convs1.{i}.weight"], w[f"{p}.convs1.{i}.bias"], dilation=d, padding=(kernel * d - d) // 2)
        xt = activation(w, f"{p}.activations.{2 * i + 1}", xt)
        xt = F.conv1d(xt, w[f"{p}.convs2.{i}.weight"], w[f"{p}.convs2.{i}.bias"], padding=(kernel - 1) // 2)
        x = xt.add_(x)
    return x


def decode(w, z):
    """[1, 32, 2, T] KSampler audio latent -> stereo waveform [1, 2, T * 800] (CPU, FP32), loudness-normalized."""
    b, c, s, t = z.shape
    z = z.float().permute(0, 2, 1, 3).reshape(b * s, c, t)
    z = z * w["latents_std"].view(1, -1, 1) + w["latents_mean"].view(1, -1, 1)
    x = F.conv1d(z, w["dec_in_proj.weight"], w["dec_in_proj.bias"])
    x = F.conv1d(x, w["decoder.conv_pre.weight"], w["decoder.conv_pre.bias"], padding=3)
    for i, (u, k) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNELS)):
        x = F.conv_transpose1d(x, w[f"decoder.ups.{i}.0.weight"], w[f"decoder.ups.{i}.0.bias"], stride=u, padding=(k - u) // 2)
        xs = None
        for j, kernel in enumerate(RESBLOCK_KERNELS):
            y = amp_block(w, f"decoder.resblocks.{i * len(RESBLOCK_KERNELS) + j}", x, kernel)
            xs = y if xs is None else xs.add_(y)
        x = xs.div_(len(RESBLOCK_KERNELS))
    x = activation(w, "decoder.activation_post", x)
    audio = F.conv1d(x, w["decoder.conv_post.weight"], padding=3).clamp_(-1.0, 1.0).reshape(b, s, -1).cpu()
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    return audio / std
