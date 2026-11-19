#!/usr/bin/env python3
"""マンガタブの下請け: ネーム JSON を ja-manga-writing のスクリプトで割り付け・検査し、コマごとのプロンプトと
写植・演出の層の叩き台を作る.

  python manga.py analyze NAME.json      → 標準出力に JSON
    {pages: [{number, side, aspect, panels: [{id, bbox, aspect, prompt, anima, avoid, layers}]}],
     issues: [{code, level, where, message}], stats, error?}
  python manga.py crop SRC DST ASPECT    コマの縦横比（横/縦）で中央を切り抜いた確認用の絵（長辺 1024px）

bbox はページの norm_1000。layers の座標はコマの中の割合（x は幅、y は高さに対して 0〜1）。
吹き出し・台詞・擬音・作中の文字と、層で描く漫符・効果（meta.overlay に名前）は絵に描かせない。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "skill" / "scripts"))
import manga_name as mn  # noqa: E402
import name_export as ne  # noqa: E402
import name_lint as nl  # noqa: E402

ANIMA_CHARS = 1500  # Anima の T5 上限 512 トークンに、先頭の固定文と余裕を残す目安（英語でおよそ 3.5 字 / トークン）
STYLE = {"dialogue": "speech", "shout": "shout", "whisper": "whisper", "thought": "thought",
         "monologue": "monologue", "narration": "narration"}
BACKGROUND_MARKS = {"gloom_lines", "sparkle_background", "lightning_shock", "rivalry_sparks", "rage_zigzag"}


def anima_prompt(doc, p, manpu):
    """Anima 用に詰めたプロンプト: (正の文, avoid の文). avoid はネガティブに回し、長ければ人物の外見を要点に縮める."""
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


def picture_only(p, overlay):
    """絵に描かせる分だけのコマ: 文字はすべて外し、層で描く漫符・効果も外す."""
    chars = [dict(c, manpu=[m for m in c.get("manpu", []) or [] if m not in overlay]) for c in p.get("characters", []) or []]
    return dict(p, text=[], characters=chars, effects=[e for e in p.get("effects", []) or [] if e not in overlay])


def r(box):
    return [round(float(v), 1) for v in box]


def layers(p, place, overlay, aspect):
    """ネームと推定の配置から、写植・演出の層の叩き台（コマの中の割合）を作る."""
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


def analyze(path):
    try:
        doc = mn.load_name(path)
    except mn.NameError_ as e:
        return {"error": str(e), "pages": [], "issues": []}
    struct = mn.precheck(doc)
    if struct:
        return {"error": "ネームの構造に誤りがあります", "pages": [], "issues": [i.as_dict() for i in struct]}
    overlay = set(doc["meta"].pop("overlay", []) or [])
    th, opts = nl.load_config(doc, path, None)
    issues, stats = nl.lint(doc, th, opts)
    fmt = mn.name_format(doc)
    manpu = mn.manpu_index()
    pages = []
    for i, page in enumerate(doc.get("pages", [])):
        number = page.get("number", i + 1)
        aspect = mn.page_aspect(doc, page)
        mn.layout_page(page, fmt)
        panels = []
        for p in page.get("panels", []):
            if "_bbox" not in p:
                continue
            place = mn.place_panel(p, aspect, manpu)
            pic = picture_only(p, overlay)
            anima, avoid = anima_prompt(doc, pic, manpu)
            x1, y1, x2, y2 = p["_bbox"]
            panels.append({
                "id": p.get("id"),
                "bbox": r(p["_bbox"]),
                "aspect": round(max(x2 - x1, 1) / max((y2 - y1) * aspect, 1), 4),   # 横/縦（実寸）
                "prompt": ne.panel_prompt(doc, pic, manpu),
                "anima": anima,
                "avoid": avoid,
                "layers": layers(p, place, overlay, aspect),
            })
        pages.append({"number": number, "side": mn.page_side(doc, number), "aspect": aspect, "panels": panels})
    return {"pages": pages, "issues": [i.as_dict() for i in issues], "stats": stats}


def crop(src, dst, aspect):
    from PIL import Image
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if w / h > aspect:
        nw = int(h * aspect)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / aspect)
        img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    img.thumbnail((1024, 1024))
    img.save(dst)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "analyze":
        print(json.dumps(analyze(sys.argv[2]), ensure_ascii=False, default=str))
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "crop":
        crop(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        return 0
    print("使い方: manga.py analyze NAME.json | crop SRC DST ASPECT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
