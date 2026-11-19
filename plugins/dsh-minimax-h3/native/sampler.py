"""MiniMax H3 text-to-video sampling: SigmaShift (video 10 / audio 3), simple 8-step schedule, res_multistep.

Written from ComfyUI's model_sampling.py (ModelSamplingAV + CONST), samplers.py (simple_scheduler),
k_diffusion/sampling.py (res_multistep, eta 0), sample.py (prepare_noise) and nodes_minimax_h3.py
(the AV latent shapes); every op keeps ComfyUI's dtype and order (bit-exact, see test_sampler_parity.py).
The DiT is torch.ops.dsh_h3.dit_forward (dit.cpp) with the SLA statistics carried across steps.
"""
import torch

FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_T = 0.999


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


# ---- keyframes and references (ComfyUI's PackedLayout and MiniMaxH3Model._cond_video_rows) -------------------------

def _axis(dim, sqrt_area):
    ratio, n = dim / sqrt_area, dim // 2
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    area = (h * w) ** 0.5
    hh, ww = torch.meshgrid(_axis(h, area), _axis(w, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis(w, area)


def _video_grid(vt, frame, cursor):
    spans = torch.tensor([FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(vt)], dtype=torch.float64)
    t = float(cursor) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = t[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def _audio_grid(cursor, t, w_low, w_high):
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def patchify(z):
    """[1, 24, T, H, W] -> [T * H/2 * W/2, 96] rows (c, 1, 2, 2) in (t, h, w) order."""
    _, c, t, h, w = z.shape
    return z.reshape(1, c, t, 1, h // 2, 2, w // 2, 2).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, c * 4)


def conditioning(text_len, video_shape, audio_t, keyframes=(), refs=(), seed=0):
    """Condition rows and packed positions for dit_forward.

    keyframes: [(pixel frame index, normalized latent [1, 24, T, h, w])] on the target grid (I2V first / last frame);
    refs: normalized latents [1, 24, 1, h, w] (reference pictures, p-h3-image edit inputs).
    -> (cond_rows FP32 [N, 96] or None, positions [S, 3] float64 or None). Every condition restarts the seed's CPU
    noise stream: rows = 0.999 * rows + 0.001 * noise."""
    if not keyframes and not refs:
        return None, None
    _, _, latent_t, lat_h, lat_w = video_shape
    frame, w_grid = _frame_grid(lat_h, lat_w)
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len, dtype=torch.float64)
    pos = [text]
    cursor = float(text_len) + len(refs)  # references take one time step each ahead of the targets
    for index, z in keyframes:
        pos.append(_video_grid(z.shape[2], frame, cursor + FRAME_RESCALE * index))
    for i, z in enumerate(refs):
        r_frame, _ = _frame_grid(z.shape[3], z.shape[4])
        g = torch.empty(r_frame.shape[0], 3, dtype=torch.float64)
        g[:, 0] = float(text_len) + i
        g[:, 1:] = r_frame
        pos.append(g)
    pos.append(_audio_grid(cursor, audio_t, float(w_grid[0]), float(w_grid[-1])))
    pos.append(_video_grid(latent_t, frame, cursor))
    rows = []
    for z in [z for _, z in keyframes] + list(refs):
        r = patchify(z.to(torch.float32))
        noise = torch.randn(r.shape, generator=torch.Generator("cpu").manual_seed(int(seed)), dtype=torch.float32)
        rows.append(VISUAL_COND_T * r + (1.0 - VISUAL_COND_T) * noise.to(r.device))
    return torch.cat(rows).cuda(), torch.cat(pos)


def sample(weights, context, token_tags, seed, width, height, length, steps=8, shift_video=10.0, shift_audio=3.0,
           sparse=True, on_step=None, keyframes=(), refs=()):
    """Returns the KSampler latent [video, audio] (FP32, audio back on its own scale)."""
    audio_scale = shift_video / shift_audio
    sigmas = simple_sigmas(steps, shift_video).cuda()
    shapes = latent_shapes(width, height, length)
    noise = [n.cuda() for n in initial_noise(seed, *shapes)]
    cond_rows, positions = conditioning(context.shape[1], shapes[0], shapes[1][-1], keyframes, refs, seed)
    x = [sigmas[0] * n + (1.0 - sigmas[0]) * torch.zeros_like(n) for n in noise]
    stats = [torch.zeros(56, 128, device="cuda") for _ in range(100)] if sparse else []

    def denoise(x, sigma, first):
        out = torch.ops.dsh_h3.dit_forward(weights, x[0].to(torch.bfloat16), x[1].to(torch.bfloat16), sigma.view(1) * 1000,
                                           context, token_tags, shift_video, shift_audio, audio_scale, stats, first,
                                           cond_rows, positions)
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

