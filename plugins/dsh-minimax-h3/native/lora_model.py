"""MiniMax H3 / P-H3 DiT with LoRA for training (PyTorch autograd), text-to-video / text-to-image layout.

Same math and BF16 order as dit.cpp (forward_impl without condition rows, dense attention): the embeddings, time
embedding, rope table and AdaLN modulations need no gradient and are built under no_grad; the 50 blocks and the final
layer run with autograd. LoRA (A [r, in], B [out, r], scale 1) sits on qkv_proj, out_proj, mlp.fc1 and mlp.fc2 of the 50
blocks and of the 2 token_refiner blocks (208 projections, as diffusion-pipe), so the refiner runs here too, from the
cached 5120-wide text states (condition.cpp refine_text). P-H3's int8 convrot projections run forward exactly as inference (W8A8 int8_linear) and backward through the
dequantized BF16 weight (straight-through for the activation quantization).
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import sampler

HIDDEN, HEADS, HEAD_DIM, ROT = 5376, 56, 128, 96
EPS = 1e-5
TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2
TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def block_count(weights):
    return 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("blocks."))


def lora_names(weights):
    blocks = [f"token_refiner.blocks.{i}" for i in range(2)] + [f"blocks.{i}" for i in range(block_count(weights))]
    return [f"{b}.{t}" for b in blocks for t in TARGETS]


def lora_parameters(weights, rank, seed=0):
    """{module name: (A, B)} FP32 leaves; A kaiming-uniform (as PEFT), B zero."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    params = {}
    for name in lora_names(weights):
        out_f, in_f = weights[name + ".weight"].shape
        bound = 1.0 / math.sqrt(in_f)  # kaiming_uniform_(a=sqrt(5)) on [rank, in_f]
        a = (torch.rand(rank, in_f, generator=generator) * 2 - 1) * bound
        params[name] = (a.cuda().requires_grad_(), torch.zeros(out_f, rank, device="cuda", requires_grad=True))
    return params


class Int8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale):
        ctx.save_for_backward(weight, scale)
        return torch.ops.dsh_kitchen.int8_linear(x, weight, scale, None, 0)

    @staticmethod
    def backward(ctx, grad):
        weight, scale = ctx.saved_tensors
        return grad @ torch.ops.dsh_kitchen.dequantize_int8_convrot(weight, scale), None, None


class Int8SwiGLULinear(torch.autograd.Function):
    """fc2 on int8: SwiGLU fused into the quantizer exactly as inference (int8_linear act 2); backward through the
    dequantized weight and PyTorch's own BF16 SwiGLU backward (recomputed from the saved fc1 output)."""
    @staticmethod
    def forward(ctx, gate_up, weight, scale):
        ctx.save_for_backward(gate_up, weight, scale)
        return torch.ops.dsh_kitchen.int8_linear(gate_up, weight, scale, None, 2)

    @staticmethod
    def backward(ctx, grad):
        gate_up, weight, scale = ctx.saved_tensors
        dz = grad @ torch.ops.dsh_kitchen.dequantize_int8_convrot(weight, scale)
        with torch.enable_grad():
            x = gate_up.detach().requires_grad_()
            gate, up = x.chunk(2, -1)
            (d_gate_up,) = torch.autograd.grad(F.silu(gate) * up, x, dz)
        return d_gate_up, None, None


def proj(w, lora, name, x, swiglu=False):
    """x @ W^T (+ LoRA). swiglu: x is fc1's [gate | up] output and the projection input is silu(gate) * up."""
    weight = w[name + ".weight"]
    if weight.dtype == torch.int8 and swiglu:
        y = Int8SwiGLULinear.apply(x, weight, w[name + ".weight_scale"])
    elif swiglu:
        gate, up = x.chunk(2, -1)
        x = F.silu(gate) * up
        y = F.linear(x, weight)
    elif weight.dtype == torch.int8:
        y = Int8Linear.apply(x, weight, w[name + ".weight_scale"])
    else:
        y = F.linear(x, weight)
    if name in lora:
        if swiglu and weight.dtype == torch.int8:
            gate, up = x.chunk(2, -1)
            x = F.silu(gate) * up
        a, b = lora[name]
        y = torch.addmm(y, x @ a.to(x.dtype).t(), b.to(x.dtype).t())
    return y


def modulate(h, shift, scale, segments):
    return torch.cat([h[a:b] * (1.0 + scale[r].to(h.dtype)) + shift[r].to(h.dtype) for a, b, r in segments])


def gate_add(x, gate, other, segments):
    return torch.cat([torch.addcmul(x[a:b], other[a:b], gate[r].to(x.dtype)) for a, b, r in segments])


def rms_rope(x, weight, cos, sin):
    """Per-head BF16 RMSNorm, then partial split-half RoPE on the first 96 dims (pairs (i, i + 48)) in FP32, rounded once
    (matches comfy-kitchen's fused rms_rope_split_half to ~1e-5)."""
    xf = F.rms_norm(x, (HEAD_DIM,), weight, EPS).float()
    half = ROT // 2
    x1, x2 = xf[..., :half], xf[..., half:ROT]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, xf[..., ROT:]], -1).to(x.dtype)


def attention(w, lora, p, h, cos, sin, key_start=0):
    """key_start: rows before it are masked out as keys (the empty prompt's padding token)."""
    s = h.shape[0]
    q, k, v = proj(w, lora, p + ".attn.qkv_proj", h).split(HEADS * HEAD_DIM, -1)
    if cos is None:  # the token refiner: no rope (condition.cpp)
        q = F.rms_norm(q.view(s, HEADS, HEAD_DIM), (HEAD_DIM,), w[p + ".attn.q_norm.weight"], EPS)
        k = F.rms_norm(k.view(s, HEADS, HEAD_DIM), (HEAD_DIM,), w[p + ".attn.k_norm.weight"], EPS)
    else:
        q = rms_rope(q.view(s, HEADS, HEAD_DIM), w[p + ".attn.q_norm.weight"], cos, sin)
        k = rms_rope(k.view(s, HEADS, HEAD_DIM), w[p + ".attn.k_norm.weight"], cos, sin)
    v = v.view(s, HEADS, HEAD_DIM)
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k[key_start:].transpose(0, 1)[None],
                                         v[key_start:].transpose(0, 1)[None])
    return proj(w, lora, p + ".attn.out_proj", out[0].transpose(0, 1).reshape(s, HEADS * HEAD_DIM))


def block(w, lora, i, x, mod, segments, cos, sin, key_start=0):
    p = f"blocks.{i}"
    shift1, scale1, gate1, shift2, scale2, gate2 = mod
    h = modulate(F.rms_norm(x, (HIDDEN,), w[p + ".norm1.weight"], EPS), shift1, scale1, segments)
    x = gate_add(x, gate1, attention(w, lora, p, h, cos, sin, key_start), segments)
    h = modulate(F.rms_norm(x, (HIDDEN,), w[p + ".norm2.weight"], EPS), shift2, scale2, segments)
    return gate_add(x, gate2, proj(w, lora, p + ".mlp.fc2", proj(w, lora, p + ".mlp.fc1", h), swiglu=True), segments)


def refine(w, lora, states):
    """[T, 5120] text states -> [1, T, 5376] context: condition_proj + 2 refiner blocks + final norm (refine_text)."""
    proj_w = w["condition_proj.weight"]
    x = F.linear(states.to(proj_w.dtype), proj_w, w["condition_proj.bias"])
    for i in range(2):
        p = f"token_refiner.blocks.{i}"
        h = F.rms_norm(x, (HIDDEN,), w[p + ".norm1.weight"], EPS)
        x = attention(w, lora, p, h, None, None) + x
        h = F.rms_norm(x, (HIDDEN,), w[p + ".norm2.weight"], EPS)
        x = proj(w, lora, p + ".mlp.fc2", proj(w, lora, p + ".mlp.fc1", h), swiglu=True) + x
    return F.rms_norm(x, (HIDDEN,), w["token_refiner.final_norm.weight"], EPS)[None]


def time_shift_sigma(sigma, shift_from, shift_to):
    base = sigma / (shift_from + sigma * (1.0 - shift_from))
    return shift_to * base / (1.0 + (shift_to - 1.0) * base)


def positions(text_len, latent_t, lat_h, lat_w, audio_t):
    """PackedLayout positions [S, 3] for [text | audio | video] (dit.cpp make_layout)."""
    frame, w_grid = sampler._frame_grid(lat_h, lat_w)
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len, dtype=torch.float64)
    return torch.cat([text, sampler._audio_grid(float(text_len), audio_t, float(w_grid[0]), float(w_grid[-1])),
                      sampler._video_grid(latent_t, frame, float(text_len))])


@torch.no_grad()
def prepare(w, video_x, audio_x, sigma, context, tags, shift_v, shift_a):
    """Packed input rows, per-block modulations, segments and rope for one sample (no gradient needed).
    The context rows are filled in by forward (refined there with the LoRA); `context` here gives their count and dtype."""
    dtype = context.dtype
    _, _, latent_t, lat_h, lat_w = video_x.shape
    audio_t, text_len = audio_x.shape[-1], context.shape[1]
    sigma_v = torch.tensor(float(sigma), dtype=torch.float32).clamp_min(1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    unique_t = sorted({t_v, t_a})
    row = {t: i for i, t in enumerate(unique_t)}
    segments, run = [], 0
    tag = tags.flatten().tolist()
    for i in range(1, text_len + 1):
        if i == text_len or tag[i] != tag[run]:
            segments.append((run, i, row[t_v] * 3 + tag[run]))
            run = i
    a0, v0 = text_len, text_len + 2 * audio_t
    if audio_t:
        segments.append((a0, v0, row[t_a] * 3 + TAG_AUDIO))
    seq = v0 + latent_t * (lat_h // 2) * (lat_w // 2)
    segments.append((v0, seq, row[t_v] * 3 + TAG_VIDEO))

    f32 = lambda k: w[k].float()
    video_embed = F.linear(sampler.patchify(video_x.float()), f32("video_patch_proj.weight"), f32("video_patch_proj.bias")).to(dtype)
    audio_rows = audio_x.float()[0].permute(1, 2, 0).reshape(-1, 32)
    audio_embed = F.linear(audio_rows, f32("audio_patch_proj.weight"), f32("audio_patch_proj.bias")).to(dtype)
    h = torch.cat([audio_embed, video_embed])

    table = f32("adaln_t_table")
    t = torch.tensor(unique_t, dtype=torch.float32, device=table.device)
    pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    mods = []
    for i in range(block_count(w)):
        p = f"blocks.{i}.adaln_proj.linear"
        m = F.linear(t_emb, f32(p + ".weight"), f32(p + ".bias"))
        mods.append(m.view(m.shape[0] * 3, 6 * HIDDEN).chunk(6, -1))
    m = F.linear(t_emb, f32("final_layer.adaln_proj.linear.weight"), f32("final_layer.adaln_proj.linear.bias"))
    final = m.view(m.shape[0], 2 * HIDDEN).chunk(2, -1)

    position = positions(text_len, latent_t, lat_h, lat_w, audio_t).float().to(h.device)
    h = torch.cat([torch.zeros(text_len, HIDDEN, dtype=dtype, device=h.device), h])
    angles = torch.cat((position.unsqueeze(-1) * f32("rope.inv_freq").view(1, 1, -1)).unbind(1), -1)  # [S, 48]
    cos, sin = angles.cos().to(dtype).float()[:, None], angles.sin().to(dtype).float()[:, None]
    return dict(h=h, mods=mods, final=final, segments=segments, cos=cos, sin=sin, a0=a0, v0=v0,
                rows=(row[t_v], row[t_a]), shape=(latent_t, lat_h, lat_w, audio_t))


def forward(w, lora, inputs, states=None, checkpoint_blocks=0):
    """-> (video velocity [1, 24, T, H, W], audio velocity [1, 32, 2, A]) FP32, with autograd through the LoRA.
    states: the [T, 5120] text states to refine into the context rows; None for the empty prompt (one padding row,
    masked out as a key, as diffusion-pipe's empty caption)."""
    x = inputs["h"]
    key_start = 0
    if states is None:
        key_start = inputs["a0"]
    else:
        x = torch.cat([refine(w, lora, states)[0], x[inputs["a0"]:]])
    segments, cos, sin = inputs["segments"], inputs["cos"], inputs["sin"]
    for i, mod in enumerate(inputs["mods"]):
        if i < checkpoint_blocks and torch.is_grad_enabled():
            x = checkpoint(block, w, lora, i, x, mod, segments, cos, sin, key_start, use_reentrant=False)
        else:
            x = block(w, lora, i, x, mod, segments, cos, sin, key_start)
    shift, scale = inputs["final"]
    a0, v0 = inputs["a0"], inputs["v0"]
    rv, ra = inputs["rows"]
    latent_t, lat_h, lat_w, audio_t = inputs["shape"]

    def head(a, b, r, name):
        normed = F.rms_norm(x[a:b], (HIDDEN,), w["final_layer.norm.weight"], EPS)
        y = (normed * (1.0 + scale[r]) + shift[r]).float()
        return F.linear(y, w[name + ".weight"].float(), w[name + ".bias"].float())

    v = head(v0, x.shape[0], rv, "final_layer.video_out")
    video = v.reshape(1, latent_t, lat_h // 2, lat_w // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7) \
        .reshape(1, 24, latent_t, lat_h, lat_w)
    audio = head(a0, v0, ra, "final_layer.audio_out").reshape(2, audio_t, 32).permute(2, 0, 1)[None]
    return -video, -audio
