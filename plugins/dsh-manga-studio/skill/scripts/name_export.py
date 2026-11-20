#!/usr/bin/env python3
"""ネーム（name TOML）を画像生成と写植のための形に書き出す.

  python name_export.py name/ep001.toml --to krea --out renders/ep001/krea [--width 864 --height 1216] [--text in-image|blank]
  python name_export.py name/ep001.toml --to panels --out renders/ep001/panels.md [--text blank|in-image]
  python name_export.py name/ep001.toml --to page-prompt --out renders/ep001/pages.md
  python name_export.py name/ep001.toml --to lettering --out renders/ep001/lettering.csv
  python name_export.py name/ep001.toml --to sections --out renders/ep001/sections.md
  python name_export.py name/ep001.toml --to compose --out renders/ep001/compose.json

krea:        ページごとの krea_layout_v1（絶対ピクセル xyxy の .layout.json と、norm_1000 の学習・推論用プロンプト .prompt.txt）。
             人物は character_id と lora 別名、pose を持ち、台詞は speakers で人物の lora 別名に結ぶ。
             --layout-prefix ph3 で PH3DaraskMix の manga lora 向け（ph3_layout_v1、整数の座標、character_id なし）。
             --krea-repo に krea2_layout パッケージのあるディレクトリを渡すと、その validate_layout でも検査する。
panels:      コマ 1 枚ずつ生成するモデル向けの英語プロンプト（画風ブロックと人物ブロックは全コマ同じ文面）。
page-prompt: ページを一括で生成するモデル向けの、コマの配置を設計図のように書いた英語プロンプト。
lettering:   写植用の文字一覧（CSV。ページ・コマ・順・種類・話者・文字）。
sections:    見出し（### 名前）で区切った文章。title、コマごとの英語の生成プロンプト panel1〜（全ページ通し番号、文字なし）、
             台詞 dialogue（1 ページ）/ dialogue-p1〜（ファイルの中のページの順に 1 から。「1: 台詞」「2!: 叫び」「3>: ナレーション」、/ で改列）、
             pages（見出しとページ・コマの対応）、notes（ページの組み立てに渡らない文字）。
compose:     ページを組む道具に渡す画素の配置（JSON。コマ・吹き出し・擬音・欄外の文字・人物・漫符の矩形）。--width の既定は 1600。

配置（人物・吹き出し・漫符の bbox）は推定の叩き台。各項目に box（コマ内の相対 0-1）を書けば固定できる。
終了コード: 成功 0 / 書き出した layout に検査エラー 1 / 入力エラー 2
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga_name as mn  # noqa: E402

PROMPT_PREFIX = "krea_layout_v1 "
KREA_LABELS = {"character", "text", "object"}
BALLOON_CAPTION = {
    "dialogue": "a white manga speech balloon with a dark outline",
    "shout": "a spiky manga shout balloon with a dark outline",
    "whisper": "a small manga speech balloon with a dashed outline",
    "thought": "a cloud-shaped manga thought balloon",
    "monologue": "a rectangular narration box",
    "narration": "a rectangular narration box",
}
TEXT_CAPTION = {
    "dialogue": "Japanese comic lettering",
    "shout": "bold Japanese comic lettering",
    "whisper": "small Japanese comic lettering",
    "thought": "Japanese comic lettering",
    "monologue": "Japanese comic lettering",
    "narration": "Japanese comic lettering",
    "sfx": "Japanese sound effect",
    "aside": "small handwritten Japanese text",
    "sign": "Japanese text written on an object in the scene",
}
POS_EN = {0: "on the right side", 1: "in the center", 2: "on the left side"}
SIDE_EN = {"right": "right-hand page", "left": "left-hand page"}


def cast_alias(cast, cid):
    c = cast.get(cid, {})
    return c.get("lora") or cid


def char_caption(cast, c):
    base = str(cast.get(c.get("id"), {}).get("appearance", "")).strip() or "a drawn manga character"
    extra = [x for x in (c.get("outfit_en"), c.get("expression")) if x]
    return ", ".join([base] + extra)


DEFAULT_STYLE = ("black-and-white Japanese manga, clean ink line art, solid black fills and sparse screentone, "
                 "plenty of white space, flat shading, dark hair as solid black fill and light hair as white with a few strands")
DEFAULT_NEGATIVE = ("grayscale painting, soft airbrush shading, smooth gradients, glossy highlights, rim light, bloom, "
                    "3d render, photorealistic, overly detailed background, extra characters, watermark, English text")
# background の値 → 英語の指示。場所が分かる最低限と、話に要る物だけを描かせる
BG_EN = {
    "detailed": "background: the location drawn clearly, with the objects that identify the place and matter to the scene",
    "simple": "background: only a few lines suggesting the location, mostly white",
    "white": "plain white background, no background drawing",
    "tone": "background: flat screentone only, no objects",
    "black": "solid black background",
    "effect": "background: manga effect lines only, no scenery",
}
BG_ALIASES = {"描き込み": "detailed", "簡略": "simple", "白": "white", "トーン": "tone", "ベタ": "black", "効果": "effect"}
# detail の値 → 英語の指示。決めコマだけ人物と背景をきちんと描く
DETAIL_EN = {
    "key": "key panel: draw the characters with care and full line work",
    "normal": "",
    "light": "simplified drawing with fewer lines and simple faces",
}
KEY_DEFAULT_BG = "background drawn clearly"   # 決めコマで background を書いていないとき（画材は画風ブロックに任せる）


def detail_and_bg(p):
    """detail（人物と線の仕上げ）と background（背景の有無と種類）を合わせた英語の指示のリスト."""
    d = mn.normalize_detail(p.get("detail")) or "normal"
    out = [DETAIL_EN.get(d, "")]
    if p.get("background"):
        out.append(background_en(p["background"]))
    elif d == "key":
        out.append(KEY_DEFAULT_BG)
    elif d == "light":
        out.append("minimal background")
    return [x for x in out if x]


def style_line(doc):
    meta = doc.get("meta", {})
    return meta.get("style") or DEFAULT_STYLE


def negative_line(doc):
    return doc.get("meta", {}).get("negative") or DEFAULT_NEGATIVE


def background_en(v):
    v = BG_ALIASES.get(str(v).strip(), str(v).strip())
    return BG_EN.get(v, f"background: {v}")


# ---------------------------------------------------------------- krea

def to_krea(doc, page, width, height, text_mode="in-image"):
    cast = doc["_cast"]
    fmt = mn.name_format(doc)
    mn.layout_page(page, fmt)
    aspect = height / width
    manpu = mn.manpu_index()
    sx, sy = width / 1000.0, height / 1000.0

    def absb(b):
        return [round(b[0] * sx, 2), round(b[1] * sy, 2), round(b[2] * sx, 2), round(b[3] * sy, 2)]

    panels_out, items = [], []
    order = mn.geometric_order(page.get("panels", []), fmt)
    order_ids = {id(p): k for k, p in enumerate(order)}
    page_chars = {c.get("id") for p in page.get("panels", []) for c in p.get("characters", []) or []}
    present_aliases = {cast_alias(cast, cid) for cid in page_chars if cid in cast}
    n_balloon = 0
    for k, p in enumerate(page.get("panels", [])):
        if "_bbox" not in p:
            continue
        pid = f"p{k + 1}"
        panels_out.append({"id": pid, "reading_order": order_ids.get(id(p), k), "bbox": absb(p["_bbox"])})
        pl = mn.place_panel(p, aspect, manpu)
        for o in pl["characters"]:
            c = o["src"]
            it = {"label": "character", "caption": char_caption(cast, c), "bbox": absb(o["bbox"]), "panel_id": pid}
            if c.get("id") in cast:
                it["character_id"] = c["id"]
                it["lora"] = cast_alias(cast, c["id"])
            if c.get("pose"):
                it["pose"] = str(c["pose"])
            items.append(it)
        for fx in pl["effects"]:
            kind = fx["def"].get("krea_kind")
            it = {"label": "object", "caption": fx["def"].get("prompt", fx["name"]), "bbox": absb(fx["bbox"]), "panel_id": pid}
            if kind:
                it["kind"] = kind
            items.append(it)
        for m in pl["marks"]:
            kind = m["def"].get("krea_kind")
            it = {"label": "object", "caption": m["def"].get("prompt", m["name"]), "bbox": absb(m["bbox"]), "panel_id": pid}
            if kind:
                it["kind"] = kind
            items.append(it)
        for bl in pl["balloons"]:
            t = bl["src"]
            ttype = t.get("type", "dialogue")
            n_balloon += 1
            bid = f"balloon_{n_balloon}"
            items.append({"id": bid, "label": "object", "kind": mn.TEXT_TYPES[ttype][1],
                          "caption": BALLOON_CAPTION[ttype], "bbox": absb(bl["bbox"]), "panel_id": pid})
            if text_mode == "blank":
                continue
            tb = _inset(bl["bbox"], 0.18, 0.12)
            it = {"label": "text", "kind": mn.TEXT_TYPES[ttype][2], "caption": TEXT_CAPTION[ttype],
                  "text": "　".join(bl["lines"]), "bbox": absb(tb), "panel_id": pid, "balloon_id": bid}
            spk = t.get("speaker")
            if spk in cast and cast_alias(cast, spk) in present_aliases and spk in page_chars:
                it["speakers"] = [cast_alias(cast, spk)]
            items.append(it)
        for tx in pl["texts"]:
            t = tx["src"]
            ttype = t.get("type", "sfx")
            if text_mode == "blank" and ttype != "sfx":
                continue
            it = {"label": "text", "caption": TEXT_CAPTION.get(ttype, "Japanese text"),
                  "text": "　".join(tx["lines"]), "bbox": absb(tx["bbox"]), "panel_id": pid}
            if mn.TEXT_TYPES[ttype][2]:
                it["kind"] = mn.TEXT_TYPES[ttype][2]
            items.append(it)
    n = len(panels_out)
    summary = page.get("summary") or f"{style_line(doc)}, {n} panel{'s' if n != 1 else ''}, read right to left"
    # コマごとの絵の中身（人物や文字の無い情景のコマでも、何を描くかが伝わるように）。p1 などの id で panels と対応づける
    scenes = []
    for k, p in enumerate(page.get("panels", [])):
        if "_bbox" not in p:
            continue
        bits = [x for x in (_cam(p), p.get("visual"), p.get("place_en") or p.get("place")) if x] + detail_and_bg(p)
        if bits:
            scenes.append(f"p{k + 1}: " + ", ".join(str(x) for x in bits))
    if scenes:
        summary = summary.rstrip(".") + ". " + "; ".join(scenes)
    layout = {"width": width, "height": height, "bbox_format": "xyxy", "bbox_coord_space": "abs",
              "summary": {"caption": summary}, "panels": panels_out, "items": items}
    return layout


def _inset(b, fx, fy):
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] + w * fx, b[1] + h * fy, b[2] - w * fx, b[3] - h * fy]


# krea2_layout/layout.py の KNOWN_KINDS（2026-11-03 の版）の写し
KREA_KNOWN_KINDS = {
    "speech_balloon", "dialogue", "sound_effect", "shout_balloon", "thought_balloon",
    "focus_lines", "emotion_mark", "speed_lines",
    "sweat_drops", "sweat_sheen", "steam_breath", "heavy_blush", "trembling_lines",
    "droplets_splash", "floating_hearts", "heart_in_eyes", "surprise_mark",
    "question_mark", "anger_vein", "dizzy_eyes", "motion_lines", "impact_burst",
    "vibration_marks", "oval_speech_bubble", "jagged_shout_bubble",
    "cloud_thought_bubble", "narration_box", "small_whisper_bubble",
}


def validate_krea(layout):
    """krea2_layout.layout.validate_layout（2026-11-03 の版）の規則を独立に写した検査.

    写した規則: width/height、bbox_format、panels の id の重複と reading_order の順列、bbox の数値・順序・範囲、
    label、panel_id、未知の kind（警告）、text の空（警告）、pose の付け先（警告）、lora と character_id（警告）、
    speakers（text 項目の、空でない・重複のない・定義済みの人物別名のリスト）。本家の版が変わったら --krea-repo で照合する。
    """
    errors, warnings = [], []
    w, h = layout.get("width"), layout.get("height")
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        return ["width/height は正の整数が必要"], warnings
    if layout.get("bbox_format", "xyxy") != "xyxy":
        errors.append("bbox_format は xyxy のみ対応")
    panels = layout.get("panels", [])
    ids = [p.get("id") for p in panels]
    if len(ids) != len(set(ids)):
        errors.append("panels の id が重複")
    orders = sorted(p.get("reading_order", -1) for p in panels)
    if panels and orders != list(range(len(panels))):
        errors.append(f"reading_order は 0..{len(panels) - 1} の順列: {orders}")

    def chk(b, where):
        if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            errors.append(f"{where}: bbox は数値 4 要素: {b!r}")
            return
        if not (b[0] < b[2] and b[1] < b[3]):
            errors.append(f"{where}: x1<x2, y1<y2 が必要: {b}")
        if b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h:
            errors.append(f"{where}: 画像範囲 {w}x{h} の外: {b}")
    for p in panels:
        chk(p.get("bbox"), f"panel {p.get('id')}")
    aliases = {it.get("lora") for it in layout.get("items", [])
               if it.get("label") == "character" and it.get("character_id") and it.get("lora")}
    for i, it in enumerate(layout.get("items", [])):
        where = f"items[{i}] label={it.get('label')}"
        if it.get("label") not in KREA_LABELS:
            errors.append(f"{where}: label は {sorted(KREA_LABELS)} のどれか")
        chk(it.get("bbox"), where)
        if it.get("panel_id") is not None and it["panel_id"] not in set(ids):
            errors.append(f"{where}: panel_id={it['panel_id']!r} が panels に無い")
        if it.get("kind") and it["kind"] not in KREA_KNOWN_KINDS:
            warnings.append(f"{where}: 未知の kind={it['kind']!r}")
        if it.get("label") == "text" and not str(it.get("text", "")).strip():
            warnings.append(f"{where}: text 項目に text が無い")
        if "pose" in it and not str(it["pose"]).strip():
            warnings.append(f"{where}: pose が空")
        if "pose" in it and it.get("label") != "character":
            warnings.append(f"{where}: pose は character 項目に付ける")
        if it.get("label") == "character" and "lora" in it and not it.get("character_id"):
            warnings.append(f"{where}: lora 指定には character_id の併記を推奨")
        if "speakers" in it:
            sp = it["speakers"]
            if it.get("label") != "text" or not isinstance(sp, list) or not sp or any(not isinstance(a, str) for a in sp):
                errors.append(f"{where}: speakers は text 項目の、空でない人物別名のリスト")
            elif len(set(sp)) != len(sp) or not set(sp) <= aliases:
                errors.append(f"{where}: speakers に重複か未定義の人物別名")
    return errors, warnings


LAYOUT_PREFIXES = {"krea": "krea_layout_v1 ", "ph3": "ph3_layout_v1 "}


def to_prompt(layout, flavor="krea"):
    w, h = layout["width"], layout["height"]
    sx, sy = 1000.0 / w, 1000.0 / h

    def conv(b):
        return [round(b[0] * sx, 2), round(b[1] * sy, 2), round(b[2] * sx, 2), round(b[3] * sy, 2)]
    out = json.loads(json.dumps(layout))
    for p in out["panels"]:
        p["bbox"] = conv(p["bbox"])
    for it in out["items"]:
        it["bbox"] = conv(it["bbox"])
    out["bbox_coord_space"] = "norm_1000"
    if flavor == "ph3":
        # PH3DaraskMix の manga lora: 同じキーと順、0〜1000 の整数、character_id は書かない（人物は lora 別名で結ぶ）
        for obj in out["panels"] + out["items"]:
            b = [min(1000, max(0, int(round(v)))) for v in obj["bbox"]]
            # 丸めで幅・高さが 0 にならないよう、最小 1 単位を残す
            for i in (0, 1):
                if b[i + 2] <= b[i]:
                    if b[i] >= 1000:
                        b[i] = 999
                    b[i + 2] = b[i] + 1
            obj["bbox"] = b
        for it in out["items"]:
            it.pop("character_id", None)
    keep = {k: out[k] for k in ("width", "height", "bbox_format", "bbox_coord_space", "summary", "items", "panels") if k in out}
    return LAYOUT_PREFIXES[flavor] + json.dumps(keep, ensure_ascii=False, separators=(",", ":"))


def adapters_template(doc):
    return {cast_alias(doc["_cast"], cid): {"path": f"/path/to/{cast_alias(doc['_cast'], cid)}.safetensors", "strength": 0.8}
            for cid, c in doc["_cast"].items() if c.get("lora")}


# ---------------------------------------------------------------- panels / page-prompt

def _cam(p):
    shot = mn.normalize_shot(p.get("shot"))
    angle = mn.normalize_angle(p.get("angle"))
    bits = []
    if shot in mn.SHOTS:
        bits.append(mn.SHOTS[shot][1])
    if angle in mn.ANGLES:
        bits.append(mn.ANGLES[angle][1])
    return ", ".join(bits)


def _panel_desc(doc, p, text_mode, manpu):
    cast = doc["_cast"]
    lines = []
    dbg = detail_and_bg(p)
    if dbg and (mn.normalize_detail(p.get("detail")) or "normal") != "normal":
        lines.append(dbg[0][0].upper() + dbg[0][1:] + ".")
        dbg = dbg[1:]
    cam = _cam(p)
    if cam:
        lines.append(f"Camera: {cam}.")
    if p.get("visual"):
        lines.append(f"Scene: {p['visual']}.")
    if p.get("place_en") or p.get("place"):
        lines.append(f"Location: {p.get('place_en') or p.get('place')}" + (f", {p.get('time_en') or p.get('time')}" if p.get("time") else "") + ".")
    chars = p.get("characters", []) or []
    if chars:
        lines.append(f"Exactly {len(chars)} character{'s' if len(chars) > 1 else ''} in this panel.")
    for k, c in enumerate(chars):
        slot = mn.POSITIONS.get(str(c.get("pos", "")))
        name = cast.get(c.get("id"), {}).get("name", c.get("id"))
        desc = [f"[{c.get('id')}] {name}", POS_EN.get(slot, "")]
        if c.get("outfit_en"):
            desc.append(f"wearing {c['outfit_en']} in this panel")
        if c.get("pose"):
            desc.append(f"pose: {c['pose']}")
        if c.get("expression"):
            desc.append(f"expression: {c['expression']}")
        marks = [manpu.get(m, {}).get("prompt", m) for m in c.get("manpu", []) or []]
        if marks:
            desc.append("manga symbols: " + "; ".join(marks))
        lines.append("- " + ", ".join(x for x in desc if x))
    fx = [manpu.get(e, {}).get("prompt", e) for e in p.get("effects", []) or []]
    if fx:
        lines.append("Effects: " + "; ".join(fx) + ".")
    for b in dbg:
        lines.append(b[0].upper() + b[1:] + ".")
    texts = p.get("text", []) or []
    balloons = [t for t in texts if t.get("type", "dialogue") in mn.BALLOON_TYPES]
    sfx = [t for t in texts if t.get("type") == "sfx"]
    present = {c.get("id") for c in chars}

    def where(t):
        """吹き出しの置き場所の指示. 話者が描かれている台詞は頭の上、画面外の声はしっぽを枠の外へ、ナレーションは角."""
        ttype = t.get("type", "dialogue")
        if ttype in ("narration", "monologue"):
            return "as a rectangular narration box in a corner of the panel"
        spk = t.get("speaker")
        if t.get("offscreen") or (spk is not None and spk not in present):
            return "with its tail pointing outside the panel (the speaker is off-panel)"
        if spk is not None:
            return f"above {cast.get(spk, {}).get('name', spk)}'s head"
        return "near the top of the panel"
    if balloons:
        if text_mode == "in-image":
            for t in balloons:
                spk = cast.get(t.get("speaker"), {}).get("name", t.get("speaker") or "")
                frm = f", from {spk}" if spk else ""
                lines.append(f"Speech balloon ({t.get('type', 'dialogue')}{frm}) {where(t)}: vertical Japanese text, exactly 「{'／'.join(mn.text_body(t))}」.")
            lines.append("Balloons are read right to left; the first balloon is the rightmost and highest.")
        else:
            kinds = "; ".join(f"balloon {k + 1} {where(t)}" for k, t in enumerate(balloons))
            lines.append(f"{len(balloons)} empty speech balloon{'s' if len(balloons) > 1 else ''} with no text inside "
                         f"(lettering is added later): {kinds}.")
    if sfx and text_mode == "in-image":
        lines.append("Hand-drawn sound effect lettering, mostly written vertically; it may overlap the figures: "
                     + ", ".join("「" + "".join(mn.text_body(t)) + "」" for t in sfx) + ".")
    elif sfx:
        lines.append("Leave room for hand-drawn sound effect lettering; do not draw any letters.")
    signs = [t for t in texts if t.get("type") == "sign"]
    asides = [t for t in texts if t.get("type") == "aside"]
    for t in signs:
        if text_mode == "in-image":
            lines.append(f"Japanese text written on an object in the scene (sign, screen or letter): exactly 「{''.join(mn.text_body(t))}」.")
        else:
            lines.append("An object in the scene carries text (sign, screen or letter); leave its surface blank for lettering later.")
    if asides:
        if text_mode == "in-image":
            lines.append("Small handwritten Japanese side note near the edge of the panel, outside any balloon: "
                         + ", ".join("「" + "".join(mn.text_body(t)) + "」" for t in asides) + ".")
        else:
            lines.append("Leave a little empty space near the panel edge for a small handwritten side note.")
    return lines


def to_panels_md(doc, text_mode="blank"):
    cast = doc["_cast"]
    manpu = mn.manpu_index()
    meta = doc.get("meta", {})
    out = [f"# {meta.get('title', 'name')} — panel prompts", "",
           "画風ブロックと人物ブロックは、全コマに同じ文面で貼る（変えるのは Panel の節だけ）。"
           "参照画像を使うモデルでは、人物ごとに同じキャラシートを毎回渡し、前のコマの出力を参照にしない。", "",
           "## Style block (same for every panel)", "", "```", style_line(doc) + ".", "Avoid: " + negative_line(doc) + ".", "```", "",
           "## Character blocks (same for every panel)", ""]
    for cid, c in cast.items():
        out.append(f"- [{cid}] {c.get('name', cid)}: {c.get('appearance', '(appearance not set)')}")
    out.append("")
    fmt = mn.name_format(doc)
    for p_i, page in enumerate(doc.get("pages", [])):
        mn.layout_page(page, fmt)
        number = page.get("number", p_i + 1)
        out.append(f"## P{number}")
        out.append("")
        for k, p in enumerate(page.get("panels", [])):
            b = p.get("_bbox")
            ratio = ""
            if b:
                aspect = mn.page_aspect(doc, page)
                w, h = b[2] - b[0], (b[3] - b[1]) * aspect
                ratio = f" — aspect {w / h:.2f}:1 (width:height)"
            out.append(f"### P{number}-{k + 1}{ratio}")
            out.append("```")
            out.extend(_panel_desc(doc, p, text_mode, manpu))
            out.append("```")
            out.append("")
    return "\n".join(out)


def to_page_prompt_md(doc, text_mode="blank"):
    manpu = mn.manpu_index()
    meta = doc.get("meta", {})
    fmt = mn.name_format(doc)
    cast = doc["_cast"]
    out = [f"# {meta.get('title', 'name')} — page prompts", ""]
    for p_i, page in enumerate(doc.get("pages", [])):
        mn.layout_page(page, fmt)
        number = page.get("number", p_i + 1)
        aspect = mn.page_aspect(doc, page)
        panels = [p for p in page.get("panels", []) if "_bbox" in p]
        order = mn.geometric_order(panels, fmt)
        out.append(f"## P{number}")
        out.append("")
        out.append("```")
        kind = {"page": "Japanese manga page", "yonkoma": "Japanese four-panel (yonkoma) comic strip", "webtoon": "vertical-scroll webtoon segment"}[fmt]
        side = f" ({SIDE_EN[mn.page_side(doc, number)]})" if fmt == "page" else ""
        out.append(f"{style_line(doc)}. A {kind}{side}, portrait, aspect ratio 1:{aspect:.2f}.")
        out.append(f"Avoid: {negative_line(doc)}.")
        if any(mn.normalize_detail(p.get("detail")) == "key" for p in panels):
            out.append("Only the key panel is fully rendered; the other panels use lighter line work.")
        if fmt != "webtoon":
            if page.get("order") == "written":
                out.append(f"Exactly {len(order)} panels. Read them in the numbered order below (some panels overlap or are inset; "
                           "follow the numbers, not the position).")
            elif fmt == "yonkoma" and "tiers" not in page and not any("bbox" in p for p in panels) and len(order) > 4:
                out.append(f"Exactly {len(order)} panels in vertical four-panel strips. Read the right strip top to bottom, "
                           "then the next strip to its left: panel 1 is at the top of the rightmost strip.")
            else:
                out.append(f"Exactly {len(order)} panels. Reading order is RIGHT TO LEFT, then TOP TO BOTTOM: panel 1 is at the top right.")
        out.append("Characters (keep identical across panels; use the matching reference image):")
        for cid in dict.fromkeys(c.get("id") for p in panels for c in p.get("characters", []) or []):
            out.append(f"- [{cid}] {cast.get(cid, {}).get('appearance', '')}")
        out.append("Panel layout (percent of the page: left, top, right, bottom):")
        for n, p in enumerate(order, 1):
            b = p["_bbox"]
            out.append(f"Panel {n}: box ({b[0] / 10:.0f}%, {b[1] / 10:.0f}%, {b[2] / 10:.0f}%, {b[3] / 10:.0f}%).")
            out.extend("  " + ln for ln in _panel_desc(doc, p, text_mode, manpu))
        out.append("Do not add panels, characters, captions or English text that are not listed.")
        out.append("```")
        out.append("")
    return "\n".join(out)


DIALOGUE_MARK = {"dialogue": "", "whisper": "", "thought": "", "shout": "!", "narration": ">", "monologue": ">"}


def panel_prompt(doc, p, manpu=None):
    """コマ 1 枚の英語の生成プロンプト（1 段落）. 画風 → 人物の外見 → カメラ → 画 → 人物の位置・ポーズ・表情 → 漫符 → 効果 → 背景
    → 絵の一部として描く文字（擬音・作中の文字）→ 吹き出しなし → 除外条件."""
    manpu = manpu if manpu is not None else mn.manpu_index()
    cast = doc["_cast"]
    parts = [style_line(doc)]
    dbg = detail_and_bg(p)
    if dbg:
        parts.append(dbg[0])
    chars = p.get("characters", []) or []
    for c in chars:
        app = cast.get(c.get("id"), {}).get("appearance")
        if app:
            parts.append(app + (f", wearing {c['outfit_en']}" if c.get("outfit_en") else ""))
    cam = _cam(p)
    if cam:
        parts.append(cam)
    if p.get("visual"):
        parts.append(str(p["visual"]))
    loc = p.get("place_en") or p.get("place")
    if loc:
        parts.append(f"location: {loc}" + (f", {p.get('time_en') or p.get('time')}" if p.get("time") else ""))
    if chars:
        parts.append(f"exactly {len(chars)} character{'s' if len(chars) > 1 else ''}")
    for c in chars:
        slot = mn.POSITIONS.get(str(c.get("pos", "")))
        bits = [x for x in (POS_EN.get(slot), c.get("pose"), c.get("expression")) if x]
        bits += [manpu.get(m, {}).get("prompt", m) for m in c.get("manpu", []) or []]
        name = cast.get(c.get("id"), {}).get("name") or c.get("id")
        if bits:
            parts.append(f"{c.get('id') or name}: " + ", ".join(bits))
    parts += [manpu.get(e, {}).get("prompt", e) for e in p.get("effects", []) or []]
    parts += dbg[1:]
    texts = p.get("text", []) or []
    drawn = []
    for t in texts:
        body = "".join(mn.text_body(t))
        if t.get("type") == "sfx":
            drawn.append(f"hand-drawn Japanese sound effect lettering 「{body}」")
        elif t.get("type") == "sign":
            drawn.append(f"Japanese text 「{body}」 written on an object in the scene")
    parts += drawn
    parts.append("no speech bubbles, no other text" if drawn else "no text, no letters, no speech bubbles")
    parts.append(f"avoid: {negative_line(doc)}")
    return ". ".join(str(x).strip().rstrip(".") for x in parts if str(x).strip()) + "."


def page_dialogue(page):
    """1 ページの台詞を「漫画のページを組む」の書き方にする（コマ番号はページの中で 1 から、/ で改列）.
    台詞の中の半角 / は改列と区別できないので全角 ／ にする。"""
    out = []
    for k, p in enumerate(page.get("panels", []), 1):
        for t in p.get("text", []) or []:
            mark = DIALOGUE_MARK.get(t.get("type", "dialogue"))
            if mark is None:
                continue
            out.append(f"{k}{mark}: " + "/".join(ln.replace("/", "／") for ln in mn.text_body(t)))
    return "\n".join(out)


def page_notes(page):
    """「漫画のページを組む」に渡らない文字の一覧（黙って消さないため）."""
    out = []
    for k, p in enumerate(page.get("panels", []), 1):
        for t in p.get("text", []) or []:
            ttype, body = t.get("type", "dialogue"), "／".join(mn.text_body(t))
            if ttype == "thought":
                out.append(f"コマ{k}: 心の声「{body}」は普通のフキダシで組まれる（雲形にするなら「ネームどおりにページを組む」）")
            elif ttype in ("sfx", "sign"):
                out.append(f"コマ{k}: {'擬音' if ttype == 'sfx' else '作中の文字'}「{body}」はコマの絵のプロンプトに入れた（絵として描かれる。崩れたら写植で直す）")
            elif ttype == "aside":
                out.append(f"コマ{k}: 欄外の小文字「{body}」は「漫画のページを組む」では入らない（「ネームどおりにページを組む」か写植で）")
    return out


def to_sections(doc):
    """見出し（### 名前）で区切った文章. title、コマごとの生成プロンプト panel1〜（全ページ通し番号）、
    台詞 dialogue（1 ページなら）または dialogue-p1〜（ファイルの中のページの順に 1 から）、ページとコマの対応 pages、
    ページの組み立てに渡らない文字の一覧 notes."""
    manpu = mn.manpu_index()
    title = doc.get("meta", {}).get("title", "")
    parts = [f"### title\n{title}"]
    n = 0
    plan, notes = [], []
    pages = doc.get("pages", [])
    for p_i, page in enumerate(pages):
        number = page.get("number", p_i + 1)
        first = n + 1
        for p in page.get("panels", []):
            n += 1
            parts.append(f"### panel{n}\n{panel_prompt(doc, p, manpu)}")
        head = "dialogue" if len(pages) == 1 else f"dialogue-p{p_i + 1}"
        plan.append(f"{head} = P{number}: panel{first}〜panel{n}（{n - first + 1} コマ）")
        notes += [f"P{number} " + x for x in page_notes(page)]
    if len(pages) == 1:
        parts.append(f"### dialogue\n{page_dialogue(pages[0])}")
    else:
        for p_i, page in enumerate(pages):
            parts.append(f"### dialogue-p{p_i + 1}\n{page_dialogue(page)}")
    parts.append("### pages\n" + "\n".join(plan))
    parts.append("### notes\n" + ("\n".join(notes) if notes else "（なし）"))
    return "\n\n".join(parts) + "\n"


def to_compose(doc, width=1600):
    """ページを組む道具（PIL など）に渡す、ページごとの画素の配置（JSON）.

    pages[i]: index（ファイルの中の順、1 から）、number、side、width、height、
    panels（id・rect・order＝読む順・inset・z＝重ねる順。z の小さい順に描くと、はめ込みが地のコマの上に乗る）、balloons（panel＝コマの添字・type・rect・lines・tail＝話者の中心）、
    texts（sfx / aside / sign の rect と lines）、characters・marks・effects（ネーム画像用の目安の箱）。
    座標はすべて左上原点の画素。配置はネームの段組みと推定（place_panel）による。
    """
    manpu = mn.manpu_index()
    fmt = mn.name_format(doc)
    cast = doc["_cast"]
    out = {"title": doc.get("meta", {}).get("title", ""), "format": fmt, "pages": []}
    for p_i, page in enumerate(doc.get("pages", [])):
        number = page.get("number", p_i + 1)
        mn.layout_page(page, fmt)
        aspect = mn.page_aspect(doc, page)
        height = int(round(width * aspect))
        sx, sy = width / 1000.0, height / 1000.0

        def px(b):
            return [round(b[0] * sx, 1), round(b[1] * sy, 1), round(b[2] * sx, 1), round(b[3] * sy, 1)]
        panels = page.get("panels", [])
        order = {id(p): k for k, p in enumerate(mn.geometric_order(panels, fmt))}
        pg = {"index": p_i + 1, "number": number, "side": mn.page_side(doc, number) if fmt == "page" else None,
              "width": width, "height": height, "panels": [], "balloons": [], "texts": [],
              "characters": [], "marks": [], "effects": []}
        for k, p in enumerate(panels):
            if "_bbox" not in p:
                continue
            shot, angle = mn.normalize_shot(p.get("shot")), mn.normalize_angle(p.get("angle"))
            cam = "・".join(x for x in (mn.SHOTS.get(shot, (None,))[0], mn.ANGLES.get(angle, (None,))[0]) if x)
            pg["panels"].append({"id": p.get("id") or f"P{number}-{k + 1}", "panel": k, "order": order.get(id(p), k),
                                 "inset": bool(p.get("inset")), "z": 1 if p.get("inset") else 0,
                                 "rect": px(p["_bbox"]), "camera": cam, "beat": p.get("beat") or p.get("visual") or ""})
            pl = mn.place_panel(p, height / width, manpu)
            centers = {}
            for o in pl["characters"]:
                cid = o["src"].get("id")
                r = px(o["bbox"])
                centers[cid] = [(r[0] + r[2]) / 2, r[1] + (r[3] - r[1]) * 0.25]
                pg["characters"].append({"panel": k, "rect": r, "name": cast.get(cid, {}).get("name", cid),
                                         "expression": o["src"].get("expression", "")})
            for bl in pl["balloons"]:
                t = bl["src"]
                pg["balloons"].append({"panel": k, "type": t.get("type", "dialogue"), "rect": px(bl["bbox"]), "lines": bl["lines"],
                                       "tail": None if t.get("offscreen") else centers.get(t.get("speaker"))})
            for tx in pl["texts"]:
                pg["texts"].append({"panel": k, "type": tx["src"].get("type"), "rect": px(tx["bbox"]), "lines": tx["lines"]})
            for m in pl["marks"]:
                pg["marks"].append({"panel": k, "rect": px(m["bbox"]), "label": m["def"].get("ja", m["name"]).split("（")[0]})
            for e in pl["effects"]:
                pg["effects"].append({"panel": k, "rect": px(e["bbox"]), "label": e["def"].get("ja", e["name"]).split("（")[0]})
        out["pages"].append(pg)
    return out


def to_lettering_csv(doc):
    cast = doc["_cast"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["page", "panel", "order", "type", "speaker", "text", "balloon"])
    for p_i, page in enumerate(doc.get("pages", [])):
        number = page.get("number", p_i + 1)
        for k, p in enumerate(page.get("panels", [])):
            for o, t in enumerate(p.get("text", []) or [], 1):
                ttype = t.get("type", "dialogue")
                w.writerow([number, k + 1, o, mn.TEXT_TYPES.get(ttype, (ttype,))[0],
                            cast.get(t.get("speaker"), {}).get("name", t.get("speaker") or ""),
                            "\n".join(mn.text_body(t)), t.get("balloon", "")])
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--to", required=True, choices=["krea", "panels", "page-prompt", "lettering", "sections", "compose", "anima", "layers", "reference"])
    ap.add_argument("--prefix", default="", help="anima: プロンプトの先頭に付ける文（品質タグなど）")
    ap.add_argument("--out", required=True, help="krea はディレクトリ、ほかはファイル（- で標準出力）")
    ap.add_argument("--width", type=int, default=None, help="画像の幅（krea の既定 864、compose の既定 1600）")
    ap.add_argument("--height", type=int, default=None, help="省略時は width × ページの縦横比（16 の倍数に丸める）")
    ap.add_argument("--text", choices=["in-image", "blank"], default=None,
                    help="文字を画像に描かせるか（in-image）、吹き出しだけ描かせて後で写植するか（blank）。既定は krea で in-image、ほかは blank")
    ap.add_argument("--layout-prefix", choices=["krea", "ph3"], default="krea",
                    help="krea の書き出しのプロンプトの形。krea=krea_layout_v1（既定）、ph3=PH3DaraskMix の manga lora（ph3_layout_v1、整数の座標、character_id なし）")
    ap.add_argument("--krea-repo", help="krea2_layout パッケージのあるディレクトリ（その validate_layout でも検査）")
    a = ap.parse_args(argv)
    try:
        doc = mn.load_name(a.name)
    except mn.NameError_ as e:
        mn.eprint(f"エラー: {e}")
        return 2
    struct = mn.precheck(doc)
    if struct:
        for i in struct:
            mn.eprint(f"エラー: {i.code} {i.where}: {i.message}")
        mn.eprint("ネームの構造に誤りがあるため、何も書き出していない（name_lint.py で確かめる）")
        return 2
    # 文字の方針: --text ＞ ネーム・作品の設定（meta.text / manga.toml の [render].text）＞ 既定（krea は in-image、ほかは blank）
    text_mode = a.text or doc.get("meta", {}).get("text") or ("in-image" if a.to == "krea" else "blank")
    if text_mode not in ("in-image", "blank"):
        mn.eprint(f"エラー: text={text_mode!r} は in-image か blank")
        return 2
    if a.to == "krea":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        ext_validate = None
        if a.krea_repo:
            sys.path.insert(0, str(Path(a.krea_repo).resolve()))
            try:
                from krea2_layout.layout import validate_layout as ext_validate  # type: ignore
            except Exception as e:  # noqa: BLE001
                mn.eprint(f"警告: krea2_layout を読み込めない（{e}）。内蔵の検査だけ行う")
        bad = 0
        prompts = []
        for i, page in enumerate(doc.get("pages", [])):
            number = page.get("number", i + 1)
            aspect = mn.page_aspect(doc, page)
            height = a.height or max(16, int(round((a.width or 864) * aspect / 16)) * 16)
            layout = to_krea(doc, page, a.width or 864, height, text_mode)
            errs, warns = validate_krea(layout)
            if ext_validate:
                e2, w2 = ext_validate(layout)
                errs += [f"krea2_layout: {x}" for x in e2]
                warns += [f"krea2_layout: {x}" for x in w2]
            for x in errs:
                mn.eprint(f"P{number} ERROR {x}")
            for x in warns:
                mn.eprint(f"P{number} warn  {x}")
            bad += bool(errs)
            stem = f"p{number:03d}"
            (out / f"{stem}.layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            prompt = to_prompt(layout, a.layout_prefix)
            (out / f"{stem}.prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            prompts.append({"page": number, "width": layout["width"], "height": layout["height"], "prompt": prompt,
                            "negative": negative_line(doc)})
        with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
            for r in prompts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # krea_layout_v1 / ph3_layout_v1 の JSON には negative の欄が無い。サンプラーの negative prompt に渡す
        (out / "negative.txt").write_text(negative_line(doc) + "\n", encoding="utf-8")
        ad = adapters_template(doc)
        if ad and not (out / "character_adapters.json").exists():
            (out / "character_adapters.json").write_text(json.dumps(ad, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{len(prompts)} ページを書き出した: {out}（検査エラーのあるページ {bad}）")
        return 1 if bad else 0
    if a.to in ("anima", "layers", "reference"):
        import name_layers as nlay
        pages = nlay.panels(doc)
        if a.to == "reference":
            # 参照で描く: 三面図は人物表の全員、場所の背景は places の key ごとに 1 枚ある前提の指示文
            style = doc.get("meta", {}).get("style") or style_line(doc)
            cast = doc["_cast"]
            rows = [{"page": pg["number"], "panel": p["id"], "aspect": p["aspect"],
                     "sheets": [c["id"] for c in p["scene"]["characters"]],
                     "place": p["scene"]["place_key"] if p["scene"]["plate"] else None,
                     "prompt": nlay.reference_prompt(p["scene"], style, [c["id"] for c in p["scene"]["characters"]],
                                                     p["scene"]["plate"] and bool(p["scene"]["place_key"]),
                                                     {k: v.get("appearance", "") for k, v in cast.items()})}
                    for pg in pages for p in pg["panels"]]
            text = json.dumps({"places": nlay.places(doc), "panels": rows}, ensure_ascii=False, indent=1) + "\n"
            if a.out == "-":
                sys.stdout.write(text)
            else:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(text, encoding="utf-8")
            return 0
        if a.to == "layers":
            text = json.dumps({"pages": [{k: v for k, v in pg.items() if k != "panels"} | {"panels": [
                {k: p[k] for k in ("id", "bbox", "aspect", "layers")} for p in pg["panels"]]} for pg in pages]},
                ensure_ascii=False, indent=1) + "\n"
        else:
            rows = []
            for pg in pages:
                for p in pg["panels"]:
                    w, hgt = nlay.panel_size(p["aspect"])
                    rows.append({"page": pg["number"], "panel": p["id"], "width": w, "height": hgt,
                                 "prompt": nlay.fit_prompt(a.prefix, p["anima"]),
                                 "negative": ", ".join(x for x in (p["avoid"], nlay.NO_TEXT) if x)})
            text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if a.out == "-":
            sys.stdout.write(text)
        else:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text, encoding="utf-8")
        return 0
    text = {"panels": lambda: to_panels_md(doc, text_mode),
            "page-prompt": lambda: to_page_prompt_md(doc, text_mode),
            "lettering": lambda: to_lettering_csv(doc),
            "sections": lambda: to_sections(doc),
            "compose": lambda: json.dumps(to_compose(doc, a.width or 1600), ensure_ascii=False, indent=1) + "\n"}[a.to]()
    if a.out == "-":
        sys.stdout.write(text)
    else:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
