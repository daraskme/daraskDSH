#!/usr/bin/env python3
"""ネーム（name TOML）を、読める台本（Markdown）と、コマ割りの見える下描き（SVG / HTML）にする.

  python name_render.py name/ep001.toml --md renders/ep001.md
  python name_render.py name/ep001.toml --md renders/ep001.reader.md --reader   # 読者役に渡す版
  python name_render.py name/ep001.toml --html renders/ep001.html              # 見開きで並べたネーム
  python name_render.py name/ep001.toml --svg renders/ep001/                   # ページごとの SVG

--reader は、紙面に載るもの（絵の中身・人物・文字）だけを書き、ページの狙い（purpose）、めくりの設計（turn）、
メモ（note）、場面 ID のような設計情報を落とす。読者役が意図を知らずに読むため。
配置は推定の叩き台で、絵の出来ではない。人物の箱・漫符・効果の位置は name_export.py と同じ計算。
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga_name as mn  # noqa: E402

TYPE_LABEL = {"dialogue": "", "shout": "（叫び）", "whisper": "（ささやき）", "thought": "（心の声）"}


def cast_name(cast, cid):
    if cid is None:
        return "？"
    return cast.get(cid, {}).get("name", cid)


def render_md(doc, reader=False):
    cast = doc["_cast"]
    meta = doc.get("meta", {})
    fmt = mn.name_format(doc)
    manpu = mn.manpu_index()
    out = []
    title = meta.get("title", "（無題）")
    out.append(f"# {title}" + (f" 第{meta['episode']}話" if meta.get("episode") else ""))
    out.append("")
    if not reader:
        kind = {"page": "横開き（右綴じ）", "yonkoma": "4コマ", "webtoon": "縦読み"}.get(fmt, fmt)
        out.append(f"- 形式: {kind} / {len(doc.get('pages', []))} ページ")
        if cast:
            out.append("- 人物: " + "、".join(f"{c.get('name', cid)}（{cid}）" for cid, c in cast.items()))
        out.append("")
    for p_i, page in enumerate(doc.get("pages", [])):
        number = page.get("number", p_i + 1)
        side = ""
        if fmt == "page":
            side = "（右ページ）" if mn.page_side(doc, number) == "right" else "（左ページ）"
        out.append(f"## P{number}{side}")
        if not reader:
            if page.get("purpose"):
                out.append(f"> 狙い: {page['purpose']}")
            if page.get("turn"):
                out.append(f"> めくり: {page['turn']}")
            if page.get("note"):
                out.append(f"> メモ: {page['note']}")
        out.append("")
        for k, p in enumerate(page.get("panels", [])):
            head = [f"コマ{k + 1}"]
            shot = mn.normalize_shot(p.get("shot"))
            angle = mn.normalize_angle(p.get("angle"))
            cam = "・".join(x for x in (mn.SHOTS.get(shot, (shot,))[0] if shot else None,
                                        mn.ANGLES.get(angle, (angle,))[0] if angle else None) if x)
            if cam:
                head.append(f"［{cam}］")
            if p.get("size"):
                head.append(f"［{p['size']}］")
            if not reader and p.get("id"):
                head.append(f"`{p['id']}`")
            out.append("### " + " ".join(head))
            where = "、".join(x for x in (p.get("place"), p.get("time")) if x)
            if where:
                out.append(f"- 場所: {where}")
            if p.get("beat"):
                out.append(f"- 画: {p['beat']}")
            elif p.get("visual"):
                out.append(f"- 画: {p['visual']}")
            if not reader and p.get("transition"):
                out.append(f"- 前のコマから: {mn.TRANSITIONS.get(p['transition'], p['transition'])}")
            for c in p.get("characters", []) or []:
                bits = [x for x in (c.get("expression"), c.get("pose_ja") or (None if reader else c.get("pose"))) if x]
                marks = [manpu.get(m, {}).get("ja", m) for m in c.get("manpu", []) or []]
                if marks:
                    bits.append("漫符: " + "・".join(marks))
                out.append(f"- {cast_name(cast, c.get('id'))}" + (f"（{' / '.join(bits)}）" if bits else ""))
            effects = [manpu.get(e, {}).get("ja", e) for e in p.get("effects", []) or []]
            if effects:
                out.append("- 効果: " + "・".join(effects))
            for t in p.get("text", []) or []:
                ttype = t.get("type", "dialogue")
                body = "／".join(mn.text_body(t))
                if ttype in TYPE_LABEL:
                    off = "（オフ）" if t.get("offscreen") else ""
                    out.append(f"- {cast_name(cast, t.get('speaker'))}{TYPE_LABEL[ttype]}{off}「{body}」")
                elif ttype in ("narration", "monologue"):
                    who = "" if ttype == "narration" else cast_name(cast, t.get("speaker"))
                    out.append(f"- {'ナレ' if ttype == 'narration' else 'モノローグ'}{('・' + who) if who else ''}［{body}］")
                elif ttype == "sfx":
                    out.append(f"- ――{body}")
                elif ttype == "sign":
                    out.append(f"- 作中の文字: 『{body}』")
                else:
                    out.append(f"- 小文字: {body}")
            if not reader and p.get("note"):
                out.append(f"- メモ: {p['note']}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- SVG

def _y(v, aspect):
    return round(v * aspect, 1)


def _jagged(x1, y1, x2, y2, n=14):
    import math
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    pts = []
    for i in range(n * 2):
        a = math.pi * i / n
        r = 1.0 if i % 2 == 0 else 0.78
        pts.append(f"{cx + rx * r * math.cos(a):.1f},{cy + ry * r * math.sin(a):.1f}")
    return " ".join(pts)


def render_svg(doc, page, show_notes=True):
    fmt = mn.name_format(doc)
    aspect = mn.page_aspect(doc, page)
    cast = doc["_cast"]
    manpu = mn.manpu_index()
    mn.layout_page(page, fmt)
    W, H = 1000, round(1000 * aspect)
    e = []
    e.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W // 2}" height="{H // 2}" '
             f'font-family="Noto Sans CJK JP, Hiragino Sans, sans-serif">')
    e.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff" stroke="#bbb"/>')
    if fmt == "page":
        fx1, fy1, fx2, fy2 = mn.MARGIN_X, mn.MARGIN_TOP, 1000 - mn.MARGIN_X, 1000 - mn.MARGIN_BOTTOM
        e.append(f'<rect x="{fx1}" y="{_y(fy1, aspect)}" width="{fx2 - fx1}" height="{_y(fy2 - fy1, aspect)}" '
                 f'fill="none" stroke="#9cf" stroke-dasharray="6 6"/>')
    # 重ねる順: 地のコマを先に、はめ込み（inset）を後に描く。番号は書いた順のまま
    draw = sorted(enumerate(page.get("panels", [])), key=lambda kp: bool(kp[1].get("inset")))
    for k, p in draw:
        if "_bbox" not in p:
            continue
        x1, y1, x2, y2 = p["_bbox"]
        e.append(f'<rect x="{x1}" y="{_y(y1, aspect)}" width="{x2 - x1:.1f}" height="{_y(y2 - y1, aspect)}" '
                 f'fill="#fff" stroke="#111" stroke-width="3"/>')
        pl = mn.place_panel(p, aspect, manpu)
        for fx in pl["effects"]:
            b = fx["bbox"]
            e.append(f'<rect x="{b[0]}" y="{_y(b[1], aspect)}" width="{b[2] - b[0]:.1f}" height="{_y(b[3] - b[1], aspect)}" '
                     f'fill="url(#fx)" opacity="0.25"/>')
            e.append(_label(b[0] + 6, _y(b[3], aspect) - 8, fx["def"].get("ja", fx["name"]), 14, "#888"))
        for o in pl["characters"]:
            b = o["bbox"]
            name = cast_name(cast, o["src"].get("id"))
            e.append(f'<rect x="{b[0]}" y="{_y(b[1], aspect)}" width="{b[2] - b[0]:.1f}" height="{_y(b[3] - b[1], aspect)}" '
                     f'rx="18" fill="#f3f3f3" stroke="#777" stroke-dasharray="8 5"/>')
            e.append(_label((b[0] + b[2]) / 2, _y(b[1], aspect) + 26, name, 20, "#444", anchor="middle"))
            if o["src"].get("expression"):
                e.append(_label((b[0] + b[2]) / 2, _y(b[1], aspect) + 48, o["src"]["expression"], 13, "#777", anchor="middle"))
        for m in pl["marks"]:
            b = m["bbox"]
            e.append(f'<rect x="{b[0]}" y="{_y(b[1], aspect)}" width="{b[2] - b[0]:.1f}" height="{_y(b[3] - b[1], aspect)}" '
                     f'fill="none" stroke="#d33" stroke-width="1.5"/>')
            e.append(_label(b[0] + 2, _y(b[1], aspect) - 3, m["def"].get("ja", m["name"]).split("（")[0], 12, "#d33"))
        for bl in pl["balloons"]:
            e.append(_balloon(bl, aspect))
        for tx in pl["texts"]:
            e.append(_free_text(tx, aspect))
        # コマ番号とカメラ
        shot = mn.normalize_shot(p.get("shot"))
        angle = mn.normalize_angle(p.get("angle"))
        cam = "・".join(x for x in (mn.SHOTS.get(shot, (shot,))[0] if shot else None,
                                    mn.ANGLES.get(angle, (angle,))[0] if angle else None) if x)
        e.append(_label(x1 + 8, _y(y1, aspect) + 22, f"{k + 1}  {cam}", 17, "#06c"))
        if show_notes and p.get("beat"):
            e.append(_label(x1 + 8, _y(y2, aspect) - 8, p["beat"][:40], 13, "#999"))
    e.insert(1, '<defs><pattern id="fx" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(30)">'
                '<line x1="0" y1="0" x2="0" y2="12" stroke="#000" stroke-width="2"/></pattern></defs>')
    e.append("</svg>")
    return "\n".join(e)


def _label(x, y, s, size, color, anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">'
            f'{html.escape(str(s))}</text>')


def _balloon(bl, aspect):
    t = bl["src"]
    ttype = t.get("type", "dialogue")
    x1, y1, x2, y2 = bl["bbox"]
    Y1, Y2 = _y(y1, aspect), _y(y2, aspect)
    parts = []
    if ttype == "shout":
        parts.append(f'<polygon points="{_jagged(x1, Y1, x2, Y2)}" fill="#fff" stroke="#000" stroke-width="2.5"/>')
    elif ttype in ("narration", "monologue"):
        parts.append(f'<rect x="{x1}" y="{Y1}" width="{x2 - x1:.1f}" height="{Y2 - Y1:.1f}" fill="#fff" stroke="#000" stroke-width="2"/>')
    else:
        dash = ' stroke-dasharray="5 4"' if ttype in ("whisper", "thought") else ""
        parts.append(f'<ellipse cx="{(x1 + x2) / 2:.1f}" cy="{(Y1 + Y2) / 2:.1f}" rx="{(x2 - x1) / 2:.1f}" ry="{(Y2 - Y1) / 2:.1f}" '
                     f'fill="#fff" stroke="#000" stroke-width="2.5"{dash}/>')
    cs = bl["char_size"]
    lines = bl["lines"]
    total_w = len(lines) * cs * 1.25
    x = (x1 + x2) / 2 + total_w / 2 - cs * 0.62
    top = (Y1 + Y2) / 2 - max(mn.count_chars(ln) for ln in lines) * cs * 1.05 / 2
    for ln in lines:
        parts.append(f'<text x="{x:.1f}" y="{top:.1f}" font-size="{cs:.1f}" style="writing-mode: vertical-rl" '
                     f'fill="#000">{html.escape(ln)}</text>')
        x -= cs * 1.25
    return "\n".join(parts)


def _free_text(tx, aspect):
    t = tx["src"]
    x1, y1, x2, y2 = tx["bbox"]
    body = "".join(tx["lines"])
    if t.get("type") == "sfx":
        size = min((x2 - x1) / 1.2, 60)
        return (f'<text x="{(x1 + x2) / 2:.1f}" y="{_y(y1, aspect):.1f}" font-size="{size:.1f}" font-weight="900" '
                f'style="writing-mode: vertical-rl" fill="#222" stroke="#fff" stroke-width="1.5">{html.escape(body)}</text>')
    return _label(x1, _y(y2, aspect) - 4, body, max(tx["char_size"] * 0.9, 10), "#333")


def render_html(doc):
    fmt = mn.name_format(doc)
    title = html.escape(doc.get("meta", {}).get("title", "ネーム"))
    pages = doc.get("pages", [])
    svgs = [(page.get("number", i + 1), render_svg(doc, page)) for i, page in enumerate(pages)]
    rows = []
    if fmt == "page":
        # 右綴じ: 見開きは「右ページ（偶数）｜左ページ（奇数）」を右から左に並べる
        spreads, cur = [], {}
        for number, svg in svgs:
            side = mn.page_side(doc, number)
            if side in cur:                      # 同じ側が続いた（ページが飛んでいる）: いまの見開きを閉じる
                spreads.append(cur)
                cur = {}
            cur[side] = (number, svg)
            if side == "left":
                spreads.append(cur)
                cur = {}
        if cur:
            spreads.append(cur)
        for sp in spreads:
            cells = []
            for side in ("left", "right"):
                if side in sp:
                    n, svg = sp[side]
                    cells.append(f'<figure>{svg}<figcaption>P{n}</figcaption></figure>')
                else:
                    cells.append('<figure class="blank"></figure>')
            rows.append('<div class="spread">' + "".join(cells) + "</div>")
    else:
        for n, svg in svgs:
            rows.append(f'<div class="spread single"><figure>{svg}<figcaption>{n}</figcaption></figure></div>')
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#eceae6; --fg:#222; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1d1d1f; --fg:#ddd; }} }}
body {{ background:var(--bg); color:var(--fg); margin:0; padding:16px; font-family:sans-serif; }}
h1 {{ font-size:18px; }}
.spread {{ display:flex; gap:4px; justify-content:center; margin:0 auto 28px; max-width:1040px; }}
.spread figure {{ margin:0; flex:1 1 0; max-width:500px; }}
.spread.single figure {{ max-width:520px; }}
.spread figure.blank {{ visibility:hidden; }}
svg {{ width:100%; height:auto; display:block; box-shadow:0 1px 4px rgba(0,0,0,.25); }}
figcaption {{ text-align:center; font-size:12px; opacity:.7; padding-top:4px; }}
</style></head><body>
<h1>{title}（ネーム・配置は推定の叩き台）</h1>
{chr(10).join(rows)}
</body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--md", help="台本の Markdown を書き出す先（- で標準出力）")
    ap.add_argument("--reader", action="store_true", help="設計情報を落とした読者用の台本にする")
    ap.add_argument("--html", help="見開きで並べたネームの HTML")
    ap.add_argument("--svg", help="ページごとの SVG を書き出すディレクトリ")
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
        mn.eprint("ネームの構造に誤りがあるため描画していない（name_lint.py で確かめる）")
        return 2
    if not (a.md or a.html or a.svg):
        a.md = "-"
    if a.md:
        text = render_md(doc, reader=a.reader)
        if a.md == "-":
            sys.stdout.write(text)
        else:
            Path(a.md).parent.mkdir(parents=True, exist_ok=True)
            Path(a.md).write_text(text, encoding="utf-8")
    if a.html:
        Path(a.html).parent.mkdir(parents=True, exist_ok=True)
        Path(a.html).write_text(render_html(doc), encoding="utf-8")
    if a.svg:
        d = Path(a.svg)
        d.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(doc.get("pages", [])):
            (d / f"p{page.get('number', i + 1):03d}.svg").write_text(render_svg(doc, page), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
