"""Anima sampling: simple scheduler, FLOW (shift 3) and the ER-SDE solver with CFG, in ComfyUI's op order.

Written from ComfyUI's comfy/samplers.py (simple_scheduler, cfg_function), comfy/model_sampling.py
(ModelSamplingDiscreteFlow, CONST) and comfy/k_diffusion/sampling.py (sample_er_sde) (GPL-3.0).
The DiT is torch.ops.dsh_anima_gen.dit (generate/dit.cpp).
"""
import torch

# Wan 2.1 latent normalization (ComfyUI latent_formats.Wan21)
LATENT_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
               0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
LATENT_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
              3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def flow_sigma(t, shift):
    return shift * t / (1 + (shift - 1) * t)


def simple_sigmas(steps, shift=3.0, timesteps=1000):
    table = flow_sigma(torch.arange(1, timesteps + 1, 1) / timesteps * 1.0 / 1.0, shift)
    stride = len(table) / steps
    return torch.FloatTensor([float(table[-(1 + int(x * stride))]) for x in range(steps)] + [0.0])


def initial_noise(seed, height, width):
    """832x1216 -> [1, 16, 1, 152, 104] FP32 on the CPU, as comfy.sample.prepare_noise."""
    generator = torch.manual_seed(seed)
    return torch.randn([1, 16, 1, height // 8, width // 8], dtype=torch.float32, generator=generator, device="cpu")


def latent_out(latent):
    mean = torch.tensor(LATENT_MEAN).view(1, 16, 1, 1, 1).to(latent.device, latent.dtype)
    std = torch.tensor(LATENT_STD).view(1, 16, 1, 1, 1).to(latent.device, latent.dtype)
    return latent * std / 1.0 + mean


def cfg_denoiser(weights, cond, uncond, scale, fp8=False):
    """denoised(x, sigma): one batch-2 DiT call (uncond, cond as ComfyUI batches them), x - v * sigma, then CFG.
    fp8: torch.ops.dsh_anima_gen.dit_fp8 (weights from prepare_fp8) instead of the bit-exact BF16 DiT."""
    context = torch.cat([uncond, cond])
    dit = torch.ops.dsh_anima_gen.dit_fp8 if fp8 else torch.ops.dsh_anima_gen.dit  # CUDA graphs measured no gain

    def denoise(x, sigma):
        xb = torch.cat([x, x])
        sb = torch.cat([sigma, sigma])
        out = dit(weights, xb.to(torch.bfloat16), (sb * 1.0).float(), context)
        denoised = xb - out.float() * sb.view(-1, 1, 1, 1, 1)
        u, c = denoised.chunk(2)
        return u + (c - u) * scale
    return denoise


def er_sde(denoise, x, sigmas, seed, s_noise=1.0, max_stage=3, shift=3.0, on_step=None):
    """comfy k_diffusion sample_er_sde for a CONST (flow) model; noise from a device generator seeded with `seed`."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed + 1 if x.device == torch.device("cpu") else seed)

    def noise_sampler():
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

    def scaler(v):
        return v * ((v ** 0.3).exp() + 10.0)

    s_in = x.new_ones([x.shape[0]])
    points = torch.arange(0, 200.0, dtype=torch.float32, device=x.device)
    if sigmas[0] >= 1:  # offset_first_sigma_for_snr: percent_to_sigma(1e-4)
        sigmas = sigmas.clone()
        sigmas[0] = flow_sigma(1.0 - 1e-4, shift)
    lambdas = sigmas.logit().exp()  # er_lambda = sigma / alpha = exp(-half_log_snr)

    old, old_d = None, None
    for i in range(len(sigmas) - 1):
        denoised = denoise(x, sigmas[i] * s_in)
        if on_step is not None:
            on_step(i, x, denoised)
        stage = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            ls, lt = lambdas[i], lambdas[i + 1]
            alpha_s = sigmas[i] / ls
            alpha_t = sigmas[i + 1] / lt
            r_alpha = alpha_t / alpha_s
            r = scaler(lt) / scaler(ls)
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised
            if stage >= 2:
                dt = lt - ls
                step = -dt / 200.0
                pos = lt + points * step
                scaled = scaler(pos)
                s = torch.sum(1 / scaled) * step
                d = (denoised - old) / (ls - lambdas[i - 1])
                x = x + alpha_t * (dt + s * scaler(lt)) * d
                if stage >= 3:
                    s_u = torch.sum((pos - ls) / scaled) * step
                    u = (d - old_d) / ((ls - lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * scaler(lt)) * u
                old_d = d
            if s_noise > 0:
                x = x + alpha_t * noise_sampler() * s_noise * (lt ** 2 - ls ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old = denoised
    return x


def sample(weights, cond, uncond, seed, steps=34, cfg=4.5, height=1216, width=832, on_step=None, fp8=False):
    """Text-to-image latent: KSampler(er_sde, simple) from an empty latent, returns the un-normalized latent."""
    sigmas = simple_sigmas(steps).cuda()
    noise = initial_noise(seed, height, width).cuda()
    x = sigmas[0].view(1, 1, 1, 1, 1) * (1.0 * noise) + (1.0 - sigmas[0].view(1, 1, 1, 1, 1)) * torch.zeros_like(noise)
    x = er_sde(cfg_denoiser(weights, cond, uncond, cfg, fp8), x, sigmas, seed, on_step=on_step)
    x = x / (1.0 - sigmas[-1].view(1, 1, 1, 1, 1))
    return latent_out(x.to(torch.float32))


def latent_in(latent):
    """Wan 2.1 process_in: the VAE mean -> the sampler's normalized latent."""
    mean = torch.tensor(LATENT_MEAN).view(1, 16, 1, 1, 1).to(latent.device, latent.dtype)
    std = torch.tensor(LATENT_STD).view(1, 16, 1, 1, 1).to(latent.device, latent.dtype)
    return (latent - mean) * 1.0 / std


def denoise_sigmas(steps, denoise):
    """KSampler with denoise < 1: the schedule of int(steps / denoise) steps, keeping its last steps + 1 sigmas."""
    if denoise > 0.9999:
        return simple_sigmas(steps)
    return simple_sigmas(int(steps / denoise))[-(steps + 1):]


def sample_image(weights, cond, uncond, seed, image_latent, steps=34, cfg=4.5, denoise=0.6, mask=None, on_step=None, fp8=False):
    """Image-to-image latent (KSampler with denoise on a VAEEncode latent), optionally only inside `mask`
    ([1, 1, 1, h, w] in [0, 1] at latent size, 1 = redraw) as ComfyUI's KSamplerX0Inpaint does with SetLatentNoiseMask:
    before each model call the unmasked part is reset to the source latent at the current noise level, and the
    model's denoised output keeps the source latent there. Returns the un-normalized latent."""
    sigmas = denoise_sigmas(steps, denoise).cuda()
    latent = latent_in(image_latent.cuda().float())
    _, _, _, h, w = latent.shape
    noise = initial_noise(seed, h * 8, w * 8).cuda()
    x = sigmas[0].view(1, 1, 1, 1, 1) * (1.0 * noise) + (1.0 - sigmas[0].view(1, 1, 1, 1, 1)) * latent
    inner = cfg_denoiser(weights, cond, uncond, cfg, fp8)
    if mask is None:
        model = inner
    else:
        keep = 1.0 - mask

        def model(x, sigma):
            s = sigma.view(-1, 1, 1, 1, 1)
            x = x * mask + (s * noise + (1.0 - s) * latent) * keep
            return inner(x, sigma) * mask + latent * keep
    x = er_sde(model, x, sigmas, seed, on_step=on_step)
    x = x / (1.0 - sigmas[-1].view(1, 1, 1, 1, 1))
    return latent_out(x.to(torch.float32))
