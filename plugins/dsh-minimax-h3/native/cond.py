"""Text + picture conditioning for H3 (Qwen3-VL 32B, layer 50) and p-h3-image (Qwen3-VL 4B, 5 hidden states).

The presentation is ComfyUI's MiniMaxH3Tokenizer (comfy/text_encoders/minimax.py): for every picture
"<Picture i>: " <|vision_start|> <image rows> <|vision_end|>, then the prompt; no chat template. Image rows come from the
vision tower, positions are Qwen3-VL MRoPE (qwen_vl.qwen2vl_mrope_position_ids), DeepStack features are added to the image
rows after the first three layers, and the DiT token tags are 0 over each vision block (start/end tokens included).
"""
import math

import torch

import vision

VISION_START, VISION_END = 151652, 151653
PICTURE = "<Picture %d>: "


def presentation(tokenizer, prompt, count):
    """-> list of token ids, with -1 where picture i's rows go (in order)."""
    def text(s):
        return tokenizer.encode(s, add_special_tokens=False).ids if s else []
    items = []
    for i in range(count):
        items += text(PICTURE % (i + 1)) + [VISION_START, -1, VISION_END]
    return items + text(prompt)


def mrope_positions(spans, length):
    """qwen2vl_mrope_position_ids: spans [(start, size, (t, gh, gw))] -> [3, length] float (None without pictures)."""
    if not spans:
        return None
    pos = torch.zeros((3, length))
    offset = 0
    for index, (start, size, grid) in enumerate(spans):
        if index == 0:
            pos[:, :start] = torch.arange(0, start)
        end = start + size
        len_max = max(grid) // 2
        start_next = len_max + start
        pos[:, end:] = torch.arange(start_next + offset, start_next + (length - end) + offset)
        pos[0, start:end] = start + offset
        rows = grid[1] // 2
        pos[1, start:end] = torch.arange(start + offset, start + rows + offset).unsqueeze(1).repeat(1, math.ceil(size / rows)).flatten(0)[:size]
        cols = grid[2] // 2
        pos[2, start:end] = torch.arange(start + offset, start + cols + offset).unsqueeze(0).repeat(math.ceil(size / cols), 1).flatten(0)[:size]
        offset += len_max - size
    return pos


def encode(te, prefix, vision_prefix, tokenizer, prompt, images, layers, taps):
    """images: [H, W, 3] RGB in [0, 1] at the size the vision tower sees.
    -> (hidden states (taps, then the last layer's output), token tags [T] int64 on CUDA)."""
    items = presentation(tokenizer, prompt, len(images))
    ids = torch.tensor([i for i in items if i >= 0], dtype=torch.int64, device="cuda")
    text_rows = torch.ops.dsh_h3.embed_tokens(te, prefix, ids)
    pictures = [vision.encode(te, vision_prefix, image) for image in images]
    rows, spans, used = [], [], 0
    for item in items:
        if item >= 0:
            rows.append(text_rows[used:used + 1])
            used += 1
        else:
            index = len(spans)
            start = sum(r.shape[0] for r in rows)
            rows.append(pictures[index][0])
            spans.append((start, rows[-1].shape[0], vision.grid(images[index])))
    embeds = torch.cat(rows)
    length = embeds.shape[0]
    tags = torch.ones(length, dtype=torch.int64)
    visual = torch.zeros(length, dtype=torch.bool)
    for start, size, _ in spans:
        tags[max(0, start - 1):start + size + 1] = 0
        visual[start:start + size] = True
    positions = mrope_positions(spans, length)
    deepstack = [torch.cat([d[j] for _, d in pictures]) for j in range(3)] if pictures else []
    hidden = torch.ops.dsh_h3.encode_embeds(te, prefix, embeds, None if positions is None else positions.cuda(), layers, taps,
                                            deepstack, visual.cuda() if pictures else None)
    return hidden, tags.cuda()
