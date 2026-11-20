#!/usr/bin/env python3
"""絵と、写植・演出の層を分ける. ネームから (1) 絵に描かせる分だけのプロンプトと (2) 層の叩き台を作る.

絵（Anima などの画像モデル）に描かせるのは人物と背景だけ。吹き出し・台詞・擬音・作中の文字と、層で描ける漫符・効果
（data/overlay.json）は絵から外し、typeset.py の層として後から載せる。顔・体そのものの表現（赤面・涙・震え）は絵に残す。
理由と層の書式は references/lettering-layers.md。

モジュールとして使う関数:
  picture_only(panel, overlay)        絵に描かせる分だけのコマ（文字・層で描く漫符と効果を外す）
  anima_prompt(doc, panel, manpu)     T5 512 トークンに収めた (正の文, avoid の文)。avoid はネガティブへ
  fit_prompt(prefix, prompt, extras)  足す文（検品の直しなど）を丸ごと残し、本文を後ろの文から削って収める
  auto_layers(panel, place, overlay, aspect)  ネームと推定配置からの層の叩き台（コマの中の割合）
  scene(doc, panel, place) / places(doc)       参照で描く（三面図・場所の背景 → 編集モデル）ための場面の情報と場所の一覧
  reference_prompt(scene, style, sheets, plate, cast)  参照で描くときの指示文（<Picture k> の割り当てつき）

コマンドとしては name_export.py --to anima / --to layers から使う。
"""
import json
from pathlib import Path

import manga_name as mn
import name_export as ne

HERE = Path(__file__).resolve().parent
ANIMA_CHARS = 1500    # 本文の目安。T5 は英語でおよそ 3.5 字 / トークン、上限 512 トークン
ANIMA_BUDGET = 1650   # 先頭の固定文・足す文を含めた全体の目安
NO_TEXT = "text, letters, speech bubble, sound effect lettering, signature"
STYLE = {"dialogue": "speech", "shout": "shout", "whisper": "whisper", "thought": "thought",
         "monologue": "monologue", "narration": "narration"}
BACKGROUND_MARKS = {"gloom_lines", "sparkle_background", "lightning_shock", "rivalry_sparks", "rage_zigzag"}


def overlay_names():
    d = json.loads((HERE / "data" / "overlay.json").read_text(encoding="utf-8"))
    return set(d["effects"]) | set(d["marks"])


def picture_only(p, overlay):
    """絵に描かせる分だけのコマ: 文字はすべて外し、層で描く漫符・効果も外す."""
    chars = [dict(c, manpu=[m for m in c.get("manpu", []) or [] if m not in overlay]) for c in p.get("characters", []) or []]
    return dict(p, text=[], characters=chars, effects=[e for e in p.get("effects", []) or [] if e not in overlay])


def anima_prompt(doc, p, manpu=None):
    """(正の文, avoid の文). avoid はネガティブに回す。長ければ人物の外見を要点（先頭の項目）に縮め、それでも長ければ文の切れ目で切る."""
    def split(text):
        head, _, avoid = text.partition(" avoid: ")
        return head.strip(), avoid.strip().rstrip(".")
    pos, avoid = split(ne.panel_prompt(doc, p, manpu))
    keep = 6
    while len(pos) > ANIMA_CHARS and keep >= 2:
        short = dict(doc)
        short["_cast"] = {k: dict(v, appearance=", ".join(str(v.get("appearance", "")).split(", ")[:keep]))
                          for k, v in doc["_cast"].items()}
        pos, avoid = split(ne.panel_prompt(short, p, manpu))
        keep -= 2
    if len(pos) > ANIMA_CHARS:
        cut = pos[:ANIMA_CHARS]
        pos = cut[:cut.rfind(". ") + 1] if ". " in cut else cut
    return pos, avoid


def fit_prompt(prefix, prompt, extras=()):
    """足す文を丸ごと残し、本文（コマのプロンプト）の後ろの文から削って ANIMA_BUDGET に収める."""
    tail = ", ".join(x for x in extras if x)
    body = str(prompt or "")
    room = ANIMA_BUDGET - len(prefix or "") - len(tail) - 4
    while len(body) > room and ". " in body:
        body = body[:body.rfind(". ", 0, len(body) - 2) + 1]
    if len(body) > room:
        body = body[:max(0, room)]
    return ", ".join(x for x in (prefix, body, tail) if x)


def auto_layers(p, place, overlay, aspect):
    """ネームと推定の配置（manga_name.place_panel）から、層の叩き台（コマの中の割合）を作る."""
    px1, py1, px2, py2 = p["_bbox"]
    pw, ph = max(px2 - px1, 1), max(py2 - py1, 1)

    def lx(v):
        return round(min(max((v - px1) / pw, 0), 1), 3)

    def ly(v):
        return round(min(max((v - py1) / ph, 0), 1), 3)
    heads = {}
    for c in place["characters"]:
        x1, y1, x2, y2 = c["bbox"]
        heads[c["src"].get("id")] = (lx((x1 + x2) / 2), ly(y1 + (y2 - y1) * 0.12), lx(x2), ly(y1))
    out = []
    for b in place["balloons"]:
        x1, y1, x2, y2 = b["bbox"]
        item = {"type": "balloon", "style": STYLE.get(b["src"].get("type", "dialogue"), "speech"), "lines": b["lines"],
                "x": lx(x1), "y": ly(y1), "w": round((x2 - x1) / pw, 3), "h": round((y2 - y1) / ph, 3)}
        head = heads.get(b["src"].get("speaker"))
        if head and item["style"] not in ("narration", "monologue"):
            item["tail"] = [head[0], head[1]]
        out.append(item)
    for t in place["texts"]:
        x1, y1, x2, y2 = t["bbox"]
        kind = t["src"].get("type")
        if kind == "sfx":
            out.append({"type": "sfx", "text": "".join(t["lines"]), "x": lx(x1), "y": ly(y1),
                        "size": round((x2 - x1) / pw * 0.85, 3), "vertical": True, "rotate": 0})
        else:
            out.append({"type": "text", "text": "／".join(t["lines"]), "x": lx(x1), "y": ly(y1),
                        "size": round(min(0.06, (y2 - y1) * aspect / pw * 0.6), 3), "vertical": kind == "aside"})
    centre = next(iter(heads.values()), (0.5, 0.45, 0.5, 0.2))
    for e in p.get("effects", []) or []:
        if e in overlay:
            out.append({"type": "effect", "name": e, "cx": centre[0], "cy": centre[1], "r": 0.28, "angle": 0, "density": 1})
    for c in p.get("characters", []) or []:
        head = heads.get(c.get("id"))
        for k, m in enumerate(c.get("manpu", []) or []):
            if m not in overlay:
                continue
            if m in BACKGROUND_MARKS:
                out.append({"type": "effect", "name": m, "cx": head[0] if head else 0.5, "cy": 0.3, "r": 0.3, "angle": 0, "density": 1})
            elif head:
                out.append({"type": "mark", "name": m, "x": round(min(head[2] + 0.02 * k, 0.92), 3),
                            "y": round(max(head[3] - 0.04, 0.02), 3), "size": 0.08, "rotate": 0})
    return out


# 背景の画を参照に渡さないコマ（背景を描かない指定）と、その代わりの指示
NO_PLATE = {"white": "plain white background", "tone": "plain background with flat screentone only",
            "black": "solid black background", "effect": "plain white background (effect lines are added later)"}


def place_key(text):
    """場所の名前を、背景の画を引くための key にする（大文字小文字と空白の違いを吸収）."""
    return " ".join(str(text or "").lower().split())[:120]


def scene(doc, p, place):
    """参照で描く（三面図・背景の画を渡す編集モデル）ための、コマの場面の情報（英語）."""
    cast = doc["_cast"]
    bg = ne.BG_ALIASES.get(str(p.get("background", "")).strip(), str(p.get("background", "")).strip())
    chars = []
    for c in p.get("characters", []) or []:
        slot = mn.POSITIONS.get(str(c.get("pos", "")))
        chars.append({"id": c.get("id"), "name": cast.get(c.get("id"), {}).get("name") or c.get("id"),
                      "position": ne.POS_EN.get(slot, ""), "pose": c.get("pose", ""), "expression": c.get("expression", ""),
                      "outfit": c.get("outfit_en", "")})
    return {"camera": ne._cam(p), "visual": p.get("visual", ""), "place": place.get("text", ""), "place_key": place.get("key", ""),
            "time": p.get("time_en") or "", "plate": bg not in NO_PLATE, "background": NO_PLATE.get(bg, ""),
            "detail": mn.normalize_detail(p.get("detail")) or "normal", "characters": chars}


def places(doc):
    """ネームに出てくる場所（背景の画を描く単位）: [{key, text, panels}]。場所はその場面の後のコマへ引き継ぐ."""
    out = {}
    for _, p, place in _walk_places(doc):
        if place.get("key"):
            out.setdefault(place["key"], {"key": place["key"], "text": place["text"], "panels": []})["panels"].append(p.get("id"))
    return list(out.values())


def _walk_places(doc):
    current, last_scene = {}, None
    for page in doc.get("pages", []):
        for p in page.get("panels", []):
            if p.get("scene") is not None and p.get("scene") != last_scene and not (p.get("place_en") or p.get("place")):
                current = {}   # 場面が変わったのに場所が書かれていない: 引き継がない
            if p.get("place_en") or p.get("place"):
                text = str(p.get("place_en") or p.get("place"))
                current = {"key": place_key(text), "text": text}
            last_scene = p.get("scene", last_scene)
            yield page, p, current


def reference_prompt(sc, style, sheets=(), plate=False, appearance=None, extras=()):
    """参照で描く（P-H3 などの参照生成）ときの指示文. sheets は三面図のある人物の id（この順に <Picture 1>〜）、
    plate は場所の背景の画を最後の <Picture> として渡すか。三面図の無い人物は appearance（id → 外見の英文）で書く."""
    appearance = appearance or {}
    lines, k = [], 0
    for ch in sc["characters"]:
        if ch["id"] in sheets:
            k += 1
            lines.append(f"<Picture {k}> is the character sheet of {ch['name']} (one person, possibly shown from several angles; "
                         f"draw {ch['name']} only once).")
        elif appearance.get(ch["id"]):
            lines.append(f"{ch['name']}: {appearance[ch['id']]}.")
    if plate and sc["plate"]:
        k += 1
        lines.append(f"<Picture {k}> is the place ({sc['place']}); keep its layout, furniture and objects, seen from this panel's camera.")
    elif sc["background"]:
        lines.append(sc["background"][0].upper() + sc["background"][1:] + ".")
    elif sc["place"]:
        lines.append(f"Place: {sc['place']}.")
    people = sc["characters"]
    out = [f"Draw one manga panel in exactly the same black-and-white ink style as the reference pictures: {style}.", *lines,
           f"Camera: {sc['camera']}." if sc["camera"] else "",
           f"Exactly {len(people)} {'people' if len(people) > 1 else 'person'}." if people else "No people."]
    for ch in people:
        bits = ", ".join(x for x in (ch["pose"], ch["outfit"] and "wearing " + ch["outfit"]) if x)
        out.append(f"{ch['name']}{' ' + ch['position'] if ch['position'] else ''}: {bits}"
                   + (f"; facial expression clearly readable and strong: {ch['expression']}" if ch["expression"] else "") + ".")
    out += [f"Scene: {sc['visual']}." if sc["visual"] else "",
            {"key": "This is a key panel: draw the characters and the place with full care and line work.",
             "light": "Keep the drawing simple with few lines."}.get(sc["detail"], ""),
            *extras, "No text, no letters, no speech bubbles, no sound effects, no effect lines, no color."]
    return " ".join(x for x in out if x)


def panels(doc, overlay=None):
    """ページごとのコマ: id、bbox（ページの norm_1000）、aspect（横/縦の実寸）、絵のプロンプト、Anima 用、
    層の叩き台、場面（参照で描くとき）."""
    overlay = overlay_names() if overlay is None else overlay
    manpu = mn.manpu_index()
    fmt = mn.name_format(doc)
    pages = []
    place_of = {id(p): pl for _, p, pl in _walk_places(doc)}
    for i, page in enumerate(doc.get("pages", [])):
        number = page.get("number", i + 1)
        aspect = mn.page_aspect(doc, page)
        mn.layout_page(page, fmt)
        out = []
        for p in page.get("panels", []):
            if "_bbox" not in p:
                continue
            place = mn.place_panel(p, aspect, manpu)
            pic = picture_only(p, overlay)
            anima, avoid = anima_prompt(doc, pic, manpu)
            x1, y1, x2, y2 = p["_bbox"]
            out.append({"id": p.get("id"), "bbox": [round(float(v), 1) for v in p["_bbox"]],
                        "aspect": round(max(x2 - x1, 1) / max((y2 - y1) * aspect, 1), 4),
                        "prompt": ne.panel_prompt(doc, pic, manpu), "anima": anima, "avoid": avoid,
                        "layers": auto_layers(p, place, overlay, aspect),
                        "scene": scene(doc, p, place_of.get(id(p), {}))})
        pages.append({"number": number, "side": mn.page_side(doc, number), "aspect": aspect, "panels": out})
    return pages


def panel_size(aspect, side=1024, step=16, lo=512, hi=2048):
    """コマの縦横比で、面積がおよそ side² の画像の大きさ（step の倍数）."""
    ratio = min(2.5, max(0.4, aspect))
    r = lambda v: int(min(hi, max(lo, round(v / step) * step)))  # noqa: E731
    return r(side * ratio ** 0.5), r(side / ratio ** 0.5)
