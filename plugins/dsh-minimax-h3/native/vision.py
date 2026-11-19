"""Qwen3-VL vision tower (Qwen3.5 vision + DeepStack) for the H3 (32B) and p-h3-image (4B) text encoders.

Written from ComfyUI's comfy/text_encoders/{qwen35,qwen3vl,qwen_vl}.py (Qwen35VisionModel, Qwen3VLDeepstackMerger,
process_qwen2vl_images); FP32 like ComfyUI's text encoder (the BF16 weights are upcast). Runs once per image.
Weights: the safetensors keys under PREFIX ("visual." for H3, "model.visual." for p-h3-image), on CUDA.
"""
import math

import torch
import torch.nn.functional as F

PATCH, TEMPORAL, MERGE, HEADS = 16, 2, 2, 16


def smart_resize(height, width, factor=PATCH * MERGE, min_pixels=3136, max_pixels=12845056):
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def grid(image):
    """(t, gh, gw) patch grid the tower sees for an [H, W, 3] image."""
    h_bar, w_bar = smart_resize(image.shape[0], image.shape[1])
    return (1, h_bar // PATCH, w_bar // PATCH)


def patches(image):
    """[H, W, 3] RGB in [0, 1] -> (flattened patches [N, 1536], (1, gh, gw)); mean/std 0.5, the frame repeated twice."""
    img = image[..., :3].permute(2, 0, 1).unsqueeze(0).float()
    h_bar, w_bar = smart_resize(img.shape[2], img.shape[3])
    img = F.interpolate(img, size=(h_bar, w_bar), mode="bilinear", align_corners=False)[0]
    img = (img - 0.5) / 0.5
    gh, gw = h_bar // PATCH, w_bar // PATCH
    x = img.unsqueeze(0).repeat(TEMPORAL, 1, 1, 1).reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    return x, (1, gh, gw)


def _f(w, key):
    return w[key].float()


def _pos_embed(w, p, gh, gw):
    table = w[p + "pos_embed.weight"]  # BF16: ComfyUI interpolates at the embedding's dtype
    side = int(table.shape[0] ** 0.5)
    device = table.device
    hi, wi = torch.linspace(0, side - 1, gh, device=device), torch.linspace(0, side - 1, gw, device=device)
    hf, wf = hi.int(), wi.int()
    hc, wc = (hf + 1).clip(max=side - 1), (wf + 1).clip(max=side - 1)
    dh, dw = hi - hf, wi - wf
    idx = [(hf[:, None] * side + wf[None]).flatten(), (hf[:, None] * side + wc[None]).flatten(),
           (hc[:, None] * side + wf[None]).flatten(), (hc[:, None] * side + wc[None]).flatten()]
    wts = [((1 - dh)[:, None] * (1 - dw)[None]).flatten(), ((1 - dh)[:, None] * dw[None]).flatten(),
           (dh[:, None] * (1 - dw)[None]).flatten(), (dh[:, None] * dw[None]).flatten()]
    # ComfyUI builds the weights through Python floats into a tensor of the table dtype
    wts = [torch.tensor(t.tolist(), dtype=table.dtype, device=device) for t in wts]
    pos = sum(table[i] * t[:, None] for i, t in zip(idx, wts))
    return pos.view(gh // MERGE, MERGE, gw // MERGE, MERGE, -1).permute(0, 2, 1, 3, 4).flatten(0, 3)


def _rotary(gh, gw, head_dim, device):
    dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    table = torch.outer(torch.arange(max(gh, gw), dtype=torch.float), inv_freq).to(device)
    rows = (torch.arange(gh // MERGE)[:, None, None, None] * MERGE + torch.arange(MERGE)[None, None, :, None])
    cols = (torch.arange(gw // MERGE)[None, :, None, None] * MERGE + torch.arange(MERGE)[None, None, None, :])
    rows = rows.expand(gh // MERGE, gw // MERGE, MERGE, MERGE).reshape(-1)
    cols = cols.expand(gh // MERGE, gw // MERGE, MERGE, MERGE).reshape(-1)
    emb = table[torch.stack((rows, cols), -1).to(device)].flatten(1)
    emb = torch.cat((emb, emb), dim=-1)
    return emb.cos().unsqueeze(-2), emb.sin().unsqueeze(-2)


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos[..., :half] - x2 * sin[..., half:], x1 * sin[..., :half] + x2 * cos[..., half:]), dim=-1)


def _ln(w, p, x):
    return F.layer_norm(x, (x.shape[-1],), _f(w, p + ".weight"), _f(w, p + ".bias"), 1e-6)


def _lin(w, p, x):
    return F.linear(x, _f(w, p + ".weight"), _f(w, p + ".bias"))


def encode(w, prefix, image):
    """[H, W, 3] in [0, 1] -> (merged [N / 4, D], [deepstack 3 x [N / 4, D]]) with D the LLM hidden size."""
    x, (_, gh, gw) = patches(image.to("cuda"))
    p = prefix
    proj = _f(w, p + "patch_embed.proj.weight")
    hidden = proj.shape[0]
    x = F.conv3d(x.view(-1, 3, TEMPORAL, PATCH, PATCH), proj, _f(w, p + "patch_embed.proj.bias"),
                 stride=(TEMPORAL, PATCH, PATCH)).view(-1, hidden)
    x = x + _pos_embed(w, p, gh, gw)
    cos, sin = _rotary(gh, gw, hidden // HEADS, x.device)
    n = x.shape[0]
    depth = 1 + max(int(k[len(p + "blocks."):].split(".")[0]) for k in w if k.startswith(p + "blocks."))
    deepstack_layers = {8: 0, 16: 1, 24: 2} if depth == 27 else {5: 0, 11: 1, 17: 2}
    deepstack = [None] * 3
    for i in range(depth):
        b = f"{p}blocks.{i}"
        q, k, v = _lin(w, b + ".attn.qkv", _ln(w, b + ".norm1", x)).reshape(n, 3, HEADS, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        x = x + _lin(w, b + ".attn.proj", out[0].transpose(0, 1).reshape(n, -1))
        x = x + _lin(w, b + ".mlp.linear_fc2", F.gelu(_lin(w, b + ".mlp.linear_fc1", _ln(w, b + ".norm2", x)), approximate="tanh"))
        if i in deepstack_layers:
            d = f"{p}deepstack_merger_list.{deepstack_layers[i]}"
            merged = _ln(w, d + ".norm", x.view(-1, hidden * MERGE * MERGE))
            deepstack[deepstack_layers[i]] = _lin(w, d + ".linear_fc2", F.gelu(_lin(w, d + ".linear_fc1", merged)))
    merged = _ln(w, p + "merger.norm", x).view(-1, hidden * MERGE * MERGE)
    return _lin(w, p + "merger.linear_fc2", F.gelu(_lin(w, p + "merger.linear_fc1", merged))), deepstack
