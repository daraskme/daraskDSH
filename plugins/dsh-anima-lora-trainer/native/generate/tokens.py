"""Anima prompt tokenization with ComfyUI's SDTokenizer rules (comfy/sd1_clip.py, GPL-3.0).

`(text:1.3)` / `(text)` emphasis, `\\(` escapes, per-segment tokenization. The Qwen3 side has no end token and
pads an empty prompt to one token; the T5 side drops each segment's EOS and appends a single end token.
Weights apply to the T5 side (the LLM adapter output rows).
"""
QWEN_PAD, T5_END = 151643, 1


def parse_parentheses(text):
    result, current, level = [], "", 0
    for char in text:
        if char == "(":
            if level == 0:
                if current:
                    result.append(current)
                current = "("
            else:
                current += char
            level += 1
        elif char == ")":
            level -= 1
            if level == 0:
                result.append(current + ")")
                current = ""
            else:
                current += char
        else:
            current += char
    if current:
        result.append(current)
    return result


def token_weights(text, weight):
    out = []
    for part in parse_parentheses(text):
        if len(part) >= 2 and part[0] == "(" and part[-1] == ")":
            inner, w = part[1:-1], weight * 1.1
            colon = inner.rfind(":")
            if colon > 0:
                try:
                    w, inner = float(inner[colon + 1:]), inner[:colon]
                except ValueError:
                    pass
            out += token_weights(inner, w)
        else:
            out.append((part, weight))
    return out


def tokenize(tokenizer, text, adds_end, end_token, pad_token):
    text = text.replace("\\)", "\0\1").replace("\\(", "\0\2")
    tokens = []
    for segment, weight in token_weights(text, 1.0):
        segment = segment.replace("\0\1", ")").replace("\0\2", "(")
        if segment:
            ids = tokenizer.encode(segment, add_special_tokens=adds_end).ids
            tokens += [(t, weight) for t in (ids[:-1] if adds_end else ids)]
    if end_token is not None:
        tokens.append((end_token, 1.0))
    if not tokens:
        tokens.append((pad_token, 1.0))
    return tokens


def prompt_tokens(qwen_tokenizer, t5_tokenizer, text):
    """-> (qwen ids, t5 ids, t5 weights)"""
    qwen = tokenize(qwen_tokenizer, text, False, None, QWEN_PAD)
    t5 = tokenize(t5_tokenizer, text, True, T5_END, 0)
    return [t for t, _ in qwen], [t for t, _ in t5], [w for _, w in t5]
