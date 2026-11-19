#!/usr/bin/env python3
"""コマ割りの型（scripts/data/layouts.json）を、1 型ずつの SVG・PNG と解説のテキストに書き出す.

  python layout_assets.py [OUT_DIR]     # 既定は assets/layouts

書き出すもの（OUT_DIR の下）:
  svg/<ID>.svg     コマ枠・読む順の番号・shot の目安（右綴じ。薄い青ははめ込み）
  png/<ID>.png     同じ図の PNG（Pillow があるとき。無ければ飛ばす）
  text/<ID>.md     解説（目的・場面・実際の使われ方・コマごとの中身・避けるとき）とネーム TOML の雛形
  README.md        場面から引く索引と、全型の一覧（図と要約）

layouts.json を変えたら、このスクリプトを流し直す。SVG と解説は標準ライブラリだけで作る。
"""
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import layout_library as ll
import manga_name as mn

ROOT = Path(__file__).resolve().parent.parent
PAGE_W, PAGE_H = 500, 707          # B5 判に近い縦横比（1:1.414）
HEAD = 64                          # 図の上の見出しの高さ
FX1, FY1, FX2, FY2 = mn._frame({}, "page")   # 基本枠（0〜1000）
PNG_FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
             "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", "C:/Windows/Fonts/meiryo.ttc"]
KIND = {"common": "よく使う型", "rare": "珍しいが狙いのはっきりした型", "user": "作者の型"}


def shot_ja(code):
    return mn.SHOTS.get(code, (code, ""))[0]


def panels(lay):
    """図の座標（px）のコマ: [(読む順 1〜, x1, y1, x2, y2, はめ込みか, shot)]、大きい順（はめ込みが上に重なる）."""
    boxes = lay["boxes"]
    slots = lay.get("slots") or [None] * len(boxes)

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def inset(i):
        a = boxes[i]
        for j, b in enumerate(boxes):
            o = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if j != i and o > 0.15 * min(area(a), area(b)) and area(a) < area(b):
                return True
        return False

    def px(v, lo, hi, size):
        return (lo + v * (hi - lo) / 1000) * size / 1000

    out = []
    for i, b in enumerate(boxes):
        out.append((i + 1, px(b[0], FX1, FX2, PAGE_W), HEAD + px(b[1], FY1, FY2, PAGE_H),
                    px(b[2], FX1, FX2, PAGE_W), HEAD + px(b[3], FY1, FY2, PAGE_H), inset(i),
                    shot_ja(slots[i]["shot"]) if slots[i] else ""))
    return sorted(out, key=lambda p: (p[5], -(p[3] - p[1]) * (p[4] - p[2])))


R = 15  # 読む順の丸の半径


def badges(lay):
    """panels() に読む順の丸の中心を足す。右綴じなので右上に置き、上に重なるはめ込みに隠れるときは左上へ."""
    ps = panels(lay)
    out = []
    for k, (n, x1, y1, x2, y2, ins, shot) in enumerate(ps):
        cx, cy = x2 - R - 7, y1 + R + 7
        if any(q[1] - R <= cx <= q[3] + R and q[2] - R <= cy <= q[4] + R for q in ps[k + 1:]):
            cx = x1 + R + 7
        out.append((n, x1, y1, x2, y2, ins, shot, cx, cy))
    return out


def subtitle(lay):
    ev = lay.get("evidence") or {}
    bits = [KIND.get(lay["kind"], lay["kind"]), f"{len(lay['boxes'])} コマ"]
    if ev.get("pages"):
        bits.append(f"実例 {ev['pages']} ページ・{ev['books']} 冊（根拠: {ev['strength']}）")
    scenes = "・".join(ll.SCENES.get(s, s) for s in lay.get("scenes", []))
    if scenes:
        bits.append(scenes)
    return "　".join(bits)


def svg(lay):
    w, h = PAGE_W, HEAD + PAGE_H
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           'font-family="\'Noto Sans CJK JP\',\'Hiragino Sans\',\'Yu Gothic\',Meiryo,sans-serif">',
           f'<title>{escape(lay["id"] + " " + lay["name"])}</title>',
           f'<rect width="{w}" height="{h}" fill="#fff"/>',
           f'<text x="12" y="26" font-size="17" font-weight="700" fill="#111">{escape(lay["id"] + "　" + lay["name"])}</text>',
           f'<text x="12" y="50" font-size="11" fill="#555">{escape(subtitle(lay))}</text>',
           f'<rect x="0.5" y="{HEAD + 0.5}" width="{w - 1}" height="{PAGE_H - 1}" fill="#fafafa" stroke="#ccc"/>']
    for n, x1, y1, x2, y2, ins, shot, cx, cy in badges(lay):
        fill = "#dbeafe" if ins else "#fff"
        out.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}" fill="{fill}" '
                   'stroke="#111" stroke-width="3"/>')
        r = R
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="#111"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy + 6:.1f}" font-size="17" font-weight="700" fill="#fff" '
                   f'text-anchor="middle">{n}</text>')
        if shot and y2 - y1 > 60:
            out.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 + 6:.1f}" font-size="15" fill="#888" '
                       f'text-anchor="middle">{escape(shot)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def png(lay, path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    s = 2  # 2 倍で描く
    font_path = next((f for f in PNG_FONTS if Path(f).exists()), None)

    def font(size):
        return ImageFont.truetype(font_path, size * s) if font_path else ImageFont.load_default()
    img = Image.new("RGB", (PAGE_W * s, (HEAD + PAGE_H) * s), "white")
    d = ImageDraw.Draw(img)
    d.text((12 * s, 10 * s), lay["id"] + "　" + lay["name"], fill="#111", font=font(17))
    d.text((12 * s, 38 * s), subtitle(lay), fill="#555", font=font(11))
    d.rectangle([0, HEAD * s, PAGE_W * s - 1, (HEAD + PAGE_H) * s - 1], fill="#fafafa", outline="#ccc")
    for n, x1, y1, x2, y2, ins, shot, cx, cy in badges(lay):
        d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], fill="#dbeafe" if ins else "white", outline="#111", width=3 * s)
        r = R
        d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s], fill="#111")
        d.text((cx * s, cy * s), str(n), fill="white", font=font(17), anchor="mm")
        if shot and y2 - y1 > 60:
            d.text(((x1 + x2) / 2 * s, (y1 + y2) / 2 * s), shot, fill="#888", font=font(15), anchor="mm")
    img.save(path, optimize=True)
    return True


def text(lay):
    ev = lay.get("evidence") or {}
    lines = [f"# {lay['id']} {lay['name']}", "", f"![{lay['id']}](../svg/{lay['id']}.svg)", "",
             f"- 種類: {KIND.get(lay['kind'], lay['kind'])}、{len(lay['boxes'])} コマ"]
    if ev.get("pages"):
        lines.append(f"- 根拠: 実例 {ev['pages']} ページ・{ev['books']} 冊（強さ: {ev['strength']}、"
                     f"全ページの {ev.get('share', 0) * 100:.1f}%、最多の 1 冊が占める割合 {round(ev.get('top_book_share', 0) * 100)}%）")
    elif ev:
        lines.append("- 根拠: 実例ほぼなし（構図・目的から考えた型）")
    basis = lay.get("scene_basis") or {s: "" for s in lay.get("scenes", [])}
    if basis:
        lines.append("- 向く場面: " + "、".join(f"{ll.SCENES.get(k, k)}" + (f"（{v}）" if v else "") for k, v in basis.items()))
    if lay.get("purpose"):
        lines += ["", "## 目的", "", lay["purpose"]]
    if lay.get("when"):
        lines += ["", "## 実際の漫画での使われ方", ""] + [f"- {w}" for w in lay["when"]]
    if lay.get("slots"):
        lines += ["", "## コマごとの中身（読む順）", ""]
        lines += [f"{i}. **{shot_ja(sl['shot'])}**（`{sl['shot']}`）: {sl['role']}" for i, sl in enumerate(lay["slots"], 1)]
    if lay.get("avoid"):
        lines += ["", "## 避けるとき", "", lay["avoid"]]
    if ev.get("pages", 0) >= 10 and ev.get("top_book_share", 0) >= 0.3:
        lines += ["", f"注意: 実例の {round(ev['top_book_share'] * 100)}% が 1 冊の作品。その作品の癖の可能性がある。"]
    lines += ["", "## ネーム TOML の雛形", "", "```toml", ll.page_toml(lay).rstrip(), "```", ""]
    return "\n".join(lines)


def readme(layouts, has_png):
    img = "png" if has_png else "svg"
    out = ["# コマ割りの型（図と解説）", "",
           "`scripts/data/layouts.json` の型を 1 枚ずつ図にしたもの。右綴じで、丸数字は読む順、コマの中の語は"
           "そのコマで多い shot の目安、薄い青ははめ込み（上に重ねるコマ）。",
           "根拠の数字は Manga109-s 全 87 冊のストーリー 15,458 ページを型に当てはめた集計"
           "（references/layout-library.md）。元の絵・台詞・作品名は持たない。", "",
           "- `svg/<ID>.svg`・`png/<ID>.png`: 図", "- `text/<ID>.md`: 解説とネーム TOML の雛形",
           "- 作り直し: `python scripts/layout_assets.py`（layouts.json を変えたら流す）", "",
           "## 場面から引く", ""]
    for key, label in ll.SCENES.items():
        hits = [x for x in layouts if key in x.get("scenes", [])]
        hits.sort(key=lambda x: -(x.get("evidence") or {}).get("pages", 0))
        out.append(f"- **{label}**: " + "、".join(f"[{x['id']}](text/{x['id']}.md)（{len(x['boxes'])} コマ）" for x in hits))
    out.append("")
    for kind, label in KIND.items():
        group = [x for x in layouts if x["kind"] == kind]
        if not group:
            continue
        out += [f"## {label}", ""]
        for x in group:
            out += [f"### [{x['id']} {x['name']}](text/{x['id']}.md)", "",
                    f'<img src="{img}/{x["id"]}.{img}" width="220" alt="{x["id"]}">', "", subtitle(x), ""]
            if x.get("purpose"):
                out += [x["purpose"], ""]
            if x.get("avoid"):
                out += [f"避けるとき: {x['avoid']}", ""]
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else ROOT / "assets" / "layouts"
    layouts = ll.load()["layouts"]
    for sub in ("svg", "png", "text"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    has_png = True
    for lay in layouts:
        (out / "svg" / f"{lay['id']}.svg").write_text(svg(lay), encoding="utf-8")
        (out / "text" / f"{lay['id']}.md").write_text(text(lay), encoding="utf-8")
        has_png = png(lay, out / "png" / f"{lay['id']}.png") and has_png
    if not has_png:
        mn.eprint("Pillow が無いので PNG は飛ばした（SVG と解説は書き出した）")
    (out / "README.md").write_text(readme(layouts, has_png), encoding="utf-8")
    print(f"{len(layouts)} 型を書き出した: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
