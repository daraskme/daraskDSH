#!/usr/bin/env python3
"""マンガタブの下請け: ネーム JSON を ja-manga-writing のスクリプトで割り付け・検査し、コマごとのプロンプトを作る.

  python manga.py analyze NAME.json   → 標準出力に JSON
    {pages: [{number, side, aspect, panels: [{id, bbox, prompt, balloons, texts, characters}]}],
     issues: [{code, level, where, message}], stats, error?}

bbox・balloons などの座標はページの norm_1000（横 0〜1000、縦 0〜1000）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "skill" / "scripts"))
import manga_name as mn  # noqa: E402
import name_export as ne  # noqa: E402
import name_lint as nl  # noqa: E402


ANIMA_CHARS = 1500  # Anima の T5 上限 512 トークンに、先頭の固定文と余裕を残す目安（英語でおよそ 3.5 字 / トークン）


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


def r(box):
    return [round(float(v), 1) for v in box]


def analyze(path):
    try:
        doc = mn.load_name(path)
    except mn.NameError_ as e:
        return {"error": str(e), "pages": [], "issues": []}
    struct = mn.precheck(doc)
    if struct:
        return {"error": "ネームの構造に誤りがあります", "pages": [], "issues": [i.as_dict() for i in struct]}
    # 擬音を写植で載せるとき（letter_sfx）は、絵のプロンプトに擬音を入れない（作中の文字は絵に残す）
    letter_sfx = bool(doc["meta"].pop("letter_sfx", False))
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
            pp = dict(p, text=[t for t in p.get("text", []) or [] if t.get("type") != "sfx"]) if letter_sfx else p
            anima, avoid = anima_prompt(doc, pp, manpu)
            panels.append({
                "id": p.get("id"),
                "bbox": r(p["_bbox"]),
                "prompt": ne.panel_prompt(doc, pp, manpu),
                "anima": anima,
                "avoid": avoid,
                "balloons": [{"bbox": r(b["bbox"]), "type": b["src"].get("type", "dialogue"),
                              "speaker": b["src"].get("speaker"), "lines": b["lines"], "cs": round(b["char_size"], 2)}
                             for b in place["balloons"]],
                "texts": [{"bbox": r(t["bbox"]), "type": t["src"].get("type"), "lines": t["lines"]}
                          for t in place["texts"]],
                "characters": [{"id": c["src"].get("id"), "bbox": r(c["bbox"])} for c in place["characters"]],
            })
        pages.append({"number": number, "side": mn.page_side(doc, number), "aspect": aspect, "panels": panels})
    return {"pages": pages, "issues": [i.as_dict() for i in issues], "stats": stats}


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "analyze":
        print("使い方: manga.py analyze NAME.json", file=sys.stderr)
        return 2
    print(json.dumps(analyze(sys.argv[2]), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
