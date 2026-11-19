"""MiniMax H3 text-to-video sampling: SigmaShift (video 10 / audio 3), simple 8-step schedule, res_multistep.

Written from ComfyUI's model_sampling.py (ModelSamplingAV + CONST), samplers.py (simple_scheduler),
k_diffusion/sampling.py (res_multistep, eta 0), sample.py (prepare_noise) and nodes_minimax_h3.py
(the AV latent shapes); every op keeps ComfyUI's dtype and order (bit-exact, see test_sampler_parity.py).
The DiT is torch.ops.dsh_h3.dit_forward (dit.cpp) with the SLA statistics carried across steps.
"""
import torch

FPS = 24
AUDIO_LATENT_FPS = 40


def frame_count(length):
    n = max(5, length)
    while n % 17 != 5:
        n += 1
    return n


def latent_shapes(width, height, length):
    """(video [1, 24, T, H/16, W/16], audio [1, 32, 2, A]) for a width x height x length video."""
    frames = frame_count(length)
    latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return [1, 24, latent_t, height // 16, width // 16], [1, 32, 2, round(frames / FPS * AUDIO_LATENT_FPS)]


def flow_sigmas(shift):
    t = (torch.arange(1, 1001, 1) / 1000) * 1000 / 1000
    return shift * t / (1 + (shift - 1) * t)


def simple_sigmas(steps, shift):
    table = flow_sigmas(shift)
    stride = len(table) / steps
    return torch.FloatTensor([float(table[-(1 + int(x * stride))]) for x in range(steps)] + [0.0])


def initial_noise(seed, video_shape, audio_shape):
    generator = torch.manual_seed(seed)  # CPU, video then audio from the same generator
    return [torch.randn(shape, dtype=torch.float32, generator=generator, device="cpu") for shape in (video_shape, audio_shape)]


def sample(weights, context, token_tags, seed, width, height, length, steps=8, shift_video=10.0, shift_audio=3.0,
           sparse=True, on_step=None):
    """Returns the KSampler latent [video, audio] (FP32, audio back on its own scale)."""
    audio_scale = shift_video / shift_audio
    sigmas = simple_sigmas(steps, shift_video).cuda()
    noise = [n.cuda() for n in initial_noise(seed, *latent_shapes(width, height, length))]
    x = [sigmas[0] * n + (1.0 - sigmas[0]) * torch.zeros_like(n) for n in noise]
    stats = [torch.zeros(56, 128, device="cuda") for _ in range(100)] if sparse else []

    def denoise(x, sigma, first):
        out = torch.ops.dsh_h3.dit_forward(weights, x[0].to(torch.bfloat16), x[1].to(torch.bfloat16), sigma.view(1) * 1000,
                                           context, token_tags, shift_video, shift_audio, audio_scale, stats, first)
        return [xi - o.float() * sigma for xi, o in zip(x, out)]

    old_sigma_down = old_denoised = None
    for i in range(len(sigmas) - 1):
        sigma, sigma_down = sigmas[i], sigmas[i + 1]
        denoised = denoise(x, sigma, i == 0)
        if on_step:
            on_step(i, x, denoised)
        if sigma_down == 0 or old_denoised is None:  # Euler
            x = [xi + (xi - d) / sigma * (sigma_down - sigma) for xi, d in zip(x, denoised)]
        else:  # second-order multistep (res_multistep)
            t, t_old, t_next, t_prev = (-s.log() for s in (sigma, old_sigma_down, sigma_down, sigmas[i - 1]))
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1 = torch.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            x = [(-h).exp() * xi + h * (b1 * d + b2 * od) for xi, d, od in zip(x, denoised, old_denoised)]
        old_denoised, old_sigma_down = denoised, sigma_down
    return [x[0], x[1] * (1.0 / audio_scale)]

