#!/usr/bin/env python3
"""写植・演出の層をコマごとの SVG（文字もすべてパス）に組む. 層の書式と置き方は references/lettering-layers.md、
書き文字のスタイルは references/sfx-styles.md.

  python typeset.py panel LAYERS.json --aspect 1.6 --out panel.svg   1 コマの層を SVG に（LAYERS.json は層の配列）
  python typeset.py serve          1 行 1 JSON の依頼 {id, panels: [{aspect, layers, seed}]} を読み、
                                   {id, panels: [{svg, boxes}]} を 1 行で返す（アプリが常駐させる）
  python typeset.py styles         書き文字のスタイルと、層で描ける効果・漫符の一覧（JSON）
  python typeset.py sample OUT     全スタイルの見本（OUT.svg）

要るもの: fonttools・uharfbuzz（pip install fonttools uharfbuzz）と、fetch_fonts.py で取るフォント。

座標: コマの幅を 1000 とし、高さは 1000 / aspect（aspect = 横/縦）。層の x・y・w・h はコマの中の割合。
SVG の viewBox はコマの外に PAD ずつ広げてある（吹き出しの枠越え）。効果はコマの中で切る。
文字はフォントの輪郭をパスにする（HarfBuzz で縦書きの字形 vert を選ぶ）ので、どの端末でも同じに見える。
フォントは MANGA_FONT_DIR（既定はこのリポジトリの .fonts/）。一覧は data/fonts.json（すべて SIL OFL）。
"""
import hashlib
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

import os

HERE = Path(__file__).resolve().parent
FONT_DIR = Path(os.environ.get("MANGA_FONT_DIR") or HERE.parent / ".fonts")
FONTS = {f["key"]: f for f in json.loads((HERE / "data" / "fonts.json").read_text(encoding="utf-8"))["fonts"]}
PAD = 0.15
FALLBACK = ["dela", "antique", "mochiy"]

# 書き文字のスタイル。CLIP STUDIO ASSETS の無料の描き文字ブラシの傾向（マーカー・べちょ筆・SE 文字ペン・横太・
# ふちペン / 二重フチ・ぽてぽて・手書きセリフ・丸筆）を参考に、フリーフォント＋加工で作った。
#   outline: 外から順の [色, 太さ（字の大きさに対する比）]。fill: 字の色。skew: 斜体（度）。squash: 横の倍率。
#   jitter: 字ごとの [回転（度）, 大きさ, ずれ] の揺れ。grow: 最後の字の大きさ（1 字目に対する倍率）。
#   bounce: 字ごとに上下（横書きは左右）へ互い違いにずらす量。tremble: 震えの残像の数。extrude: 立体の影 [dx, dy]。
#   echo: 外に広がる輪郭の残響の数。tails: 字の後ろに引く速度の線の数。spacing: 字間（字の大きさに対する比）。
STYLES = {
    "impact":   {"ja": "迫力", "use": "衝撃・打撃・爆発の決め（ドン、バーン、ガッ、ドゴォ）", "font": "dela",
                 "fill": "#000", "outline": [["#000", 0.30], ["#fff", 0.22]], "skew": -8, "jitter": [7, 0.12, 0.06]},
    "menace":   {"ja": "威圧", "use": "迫る気配・不穏・強者の登場（ゴゴゴ、ドドド、ズズズ）", "font": "dela",
                 "fill": "#000", "outline": [["#fff", 0.16]], "grow": 1.45, "tremble": 2, "jitter": [4, 0.05, 0.04]},
    "sharp":    {"ja": "鋭い SE", "use": "速さ・斬撃・素早い動き（シュッ、ザッ、ズバッ、ヒュン）", "font": "reggae",
                 "fill": "#000", "outline": [["#fff", 0.14]], "skew": -20, "squash": 0.86, "tails": 5, "jitter": [3, 0.05, 0.02]},
    "wide":     {"ja": "横太・二重フチ", "use": "大声・ドタバタ・重い衝突（ドカッ、ボンッ、ギャー）", "font": "rocknroll",
                 "fill": "#000", "outline": [["#000", 0.34], ["#fff", 0.24]], "squash": 1.28, "jitter": [8, 0.1, 0.05]},
    "pop":      {"ja": "ポップ・ふち", "use": "ギャグ・明るい驚き・軽い動き（ポンッ、ぴょん、わーい、パッ）", "font": "mochiy",
                 "fill": "#fff", "outline": [["#fff", 0.34], ["#000", 0.16]], "jitter": [12, 0.12, 0.06], "bounce": 0.08},
    "pote":     {"ja": "ぽてぽて", "use": "柔らかい・かわいい・弾む（ぷにっ、ぽよん、もちっ、ころん）", "font": "daruma",
                 "fill": "#fff", "outline": [["#000", 0.18]], "jitter": [10, 0.1, 0.04], "bounce": 0.12},
    "marker":   {"ja": "描き文字マーカー", "use": "汎用の叫び・ガヤ・勢いのある手書き（ギャー、うおお、ガヤガヤ）", "font": "yusei",
                 "fill": "#000", "outline": [["#fff", 0.2]], "jitter": [9, 0.14, 0.07], "bold": 0.07},
    "brush":    {"ja": "べちょ筆・墨", "use": "重い一撃・静かな決め・和風の迫力（ズン、斬、ドッ）", "font": "boku",
                 "fill": "#000", "outline": [["#fff", 0.14]], "skew": -4, "jitter": [5, 0.1, 0.05], "bold": 0.05},
    "syuku":    {"ja": "草書", "use": "時代劇・余韻・情緒（さらさら、ひらり、ゆらり）", "font": "syuku",
                 "fill": "#000", "outline": [["#fff", 0.12]], "jitter": [3, 0.06, 0.03], "bold": 0.05},
    "pen":      {"ja": "手書きペン", "use": "小さな物音・つぶやき・欄外（カサッ、そっ、ぽつり、コト）", "font": "kurenaido",
                 "fill": "#000", "outline": [["#fff", 0.16]], "jitter": [6, 0.08, 0.05], "bold": 0.06},
    "cute":     {"ja": "丸文字", "use": "ときめき・照れ・ラブコメ（きゅん、ドキッ、ぽっ、えへへ）", "font": "hachimaru",
                 "fill": "#000", "outline": [["#fff", 0.22]], "jitter": [10, 0.1, 0.05], "bounce": 0.06, "bold": 0.06},
    "soft":     {"ja": "やわらか手書き", "use": "ほのぼの・安心・ゆるい空気（ほわっ、ふふ、のんびり）", "font": "yomogi",
                 "fill": "#000", "outline": [["#fff", 0.18]], "jitter": [6, 0.08, 0.05], "bold": 0.06},
    "digital":  {"ja": "電子音", "use": "機械・電子音・ゲーム（ピッ、ピピピ、ブーッ、ガガッ）", "font": "dot",
                 "fill": "#000", "outline": [["#fff", 0.16]], "bold": 0.02},
    "solid":    {"ja": "立体", "use": "看板的な強調・タイトル的な一発（ガーン、ゴン、ジャーン）", "font": "rampart",
                 "fill": "#000", "outline": [["#fff", 0.12]], "jitter": [4, 0.06, 0.03], "bold": 0.02},
    "block":    {"ja": "袋文字の立体", "use": "派手な爆発・大きな着地・コミカルな衝撃（ドーン、どかーん、ズドーン）", "font": "potta",
                 "fill": "#fff", "outline": [["#000", 0.16]], "extrude": [0.07, 0.09], "jitter": [6, 0.1, 0.05]},
    "tremble":  {"ja": "震え・不気味", "use": "恐怖・寒気・きしみ（ゾワッ、ギギギ、ヒタ…、ぞくっ）", "font": "kurenaido",
                 "fill": "#000", "outline": [["#fff", 0.12]], "tremble": 3, "jitter": [8, 0.08, 0.08], "bold": 0.05},
    "quiet":    {"ja": "静寂", "use": "無音・間・気まずさ（しーん、シン…、……）", "font": "mincho",
                 "fill": "#000", "outline": [["#fff", 0.1]], "spacing": 0.35},
    "pulse":    {"ja": "鼓動・残響", "use": "心臓の音・緊張の高まり・反響（ドクン、バクバク、ドッ…）", "font": "dela",
                 "fill": "#000", "outline": [["#fff", 0.16]], "echo": 2, "jitter": [3, 0.06, 0.02]},
    "train":    {"ja": "抜き文字", "use": "白抜きの軽い強調・ベタの上の擬音（ザワ、キーン、ゴォォ）", "font": "train",
                 "fill": "#000", "outline": [["#fff", 0.1]], "jitter": [5, 0.08, 0.04], "bold": 0.03},
}
TEXT_FONT = {"speech": "antique", "whisper": "antique", "thought": "antique", "shout": "dela",
             "narration": "mincho", "monologue": "mincho"}
ROTATE = set("ー―－～〜…‥（）()「」『』【】〈〉《》＝=")


# ---------- fonts ----------

class Face:
    def __init__(self, key):
        path = FONT_DIR / FONTS[key]["file"]
        self.tt = TTFont(path, lazy=True)
        self.cmap = self.tt.getBestCmap()
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.upm = self.tt["head"].unitsPerEm
        blob = hb.Blob.from_file_path(str(path))
        self.hb = hb.Font(hb.Face(blob))

    def has(self, ch):
        return ord(ch) in self.cmap

    @lru_cache(maxsize=4096)
    def glyph(self, ch, vertical):
        """1 字の (グリフ名, 送り, x オフセット, y オフセット)（フォント単位）。縦は vert の字形と縦の送り."""
        buf = hb.Buffer()
        buf.add_str(ch)
        buf.guess_segment_properties()
        if vertical:
            buf.direction = "ttb"
        hb.shape(self.hb, buf, {"vert": vertical, "vrt2": vertical})
        info, pos = buf.glyph_infos[0], buf.glyph_positions[0]
        adv = -pos.y_advance if vertical else pos.x_advance
        return self.order[info.codepoint], adv or self.upm, pos.x_offset, pos.y_offset

    @lru_cache(maxsize=1)
    def middle(self):
        """横書きの字の箱の縦の中心（フォント単位）."""
        os2 = self.tt["OS/2"] if "OS/2" in self.tt else None
        if os2 is not None and os2.sTypoAscender:
            return (os2.sTypoAscender + os2.sTypoDescender) / 2
        return self.upm * 0.38


@lru_cache(maxsize=None)
def face(key):
    return Face(key)


def face_for(key, ch):
    fc = face(key)
    if fc.has(ch):
        return fc
    for k in FALLBACK:
        if face(k).has(ch):
            return face(k)
    return fc


def glyph_path(key, ch, size, vertical, x, y, rot=0.0, sx=1.0, skew=0.0):
    """字 1 つを、字の箱の中心を (x, y) に置いたパス（SVG の d）と送り（px）.

    rot は回転（度）、sx は横の倍率、skew は CSS の skewX と同じ向きの斜体（負で右に傾く）。"""
    fc = face_for(key, ch)
    name, adv, xo, yo = fc.glyph(ch, vertical)
    s = size / fc.upm
    if vertical:   # HarfBuzz の縦: (xo, yo) は縦の原点（送りの上端・字の中央）から横の原点へのずれ
        bcx, bcy = -xo, -yo - adv / 2
    else:
        bcx, bcy = adv / 2, fc.middle()
    a1, a2 = s * sx, -s
    b1, b2 = -s * sx * bcx, s * bcy
    k = math.tan(math.radians(skew))
    c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    T = (c * a1, sn * a1, (c * k - sn) * a2, (sn * k + c) * a2,
         x + c * (b1 + k * b2) - sn * b2, y + sn * (b1 + k * b2) + c * b2)
    pen = SVGPathPen(fc.glyphs)
    fc.glyphs[name].draw(TransformPen(pen, T))
    return pen.getCommands(), adv * s


# ---------- helpers ----------

def rng(*keys):
    return random.Random(int(hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12], 16))


def f(v):
    return f"{v:.1f}"


def num(v, default, lo=-10.0, hi=10.0):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return min(max(v, lo), hi) if math.isfinite(v) else default


def lines_of(layer):
    if isinstance(layer.get("lines"), list) and layer["lines"]:
        return [str(x) for x in layer["lines"]][:12]
    return [x for x in str(layer.get("text", "")).replace("/", "／").split("／") if x][:12] or [""]


def stroked(paths, fill, outline, extra="", bold=0.0):
    """パス（d）を、外のフチから順に重ねて描く. bold は字と同じ色の線で太らせる幅（px）."""
    d = " ".join(p for p in paths if p)
    if not d:
        return ""
    out = [f'<path d="{d}" fill="{c}" stroke="{c}" stroke-width="{f(w + bold)}" stroke-linejoin="round"{extra}/>' for c, w in outline]
    if bold:
        out.append(f'<path d="{d}" fill="{fill}" stroke="{fill}" stroke-width="{f(bold)}" stroke-linejoin="round"{extra}/>')
    out.append(f'<path d="{d}" fill="{fill}"{extra}/>')
    return "".join(out)


# ---------- text blocks ----------

def vertical_block(key, lines, size, cx, cy, spacing=1.06, colgap=1.18):
    """縦書きの段（右の列から）. 列の上をそろえ、全体の中心を (cx, cy) に置く."""
    cols = len(lines)
    rows = max((len(l) for l in lines), default=1)
    paths = []
    x0 = cx + (cols - 1) * size * colgap / 2
    top = cy - rows * size * spacing / 2
    for c, line in enumerate(lines):
        x = x0 - c * size * colgap
        y = top
        for ch in line:
            d, adv = glyph_path(key, ch, size, True, x, y + size * spacing / 2)
            paths.append(d)
            y += max(adv, size * 0.5) * spacing if ch not in "、。，．" else size * spacing
    w, h = cols * size * colgap, rows * size * spacing
    return paths, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def horizontal_block(key, lines, size, cx, cy, spacing=1.0, linegap=1.3):
    paths = []
    widths = []
    for line in lines:
        widths.append(sum(face_for(key, ch).glyph(ch, False)[1] * size / face_for(key, ch).upm for ch in line) * spacing)
    top = cy - len(lines) * size * linegap / 2
    for i, line in enumerate(lines):
        x = cx - widths[i] / 2
        y = top + size * linegap * (i + 0.5)
        for ch in line:
            fc = face_for(key, ch)
            adv = fc.glyph(ch, False)[1] * size / fc.upm
            d, _ = glyph_path(key, ch, size, False, x + adv * spacing / 2, y)
            paths.append(d)
            x += adv * spacing
    w, h = max(widths, default=size), len(lines) * size * linegap
    return paths, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------- balloons ----------

def ellipse_pts(cx, cy, rx, ry, n=72):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def poly(pts):
    return "M" + " L".join(f"{f(x)},{f(y)}" for x, y in pts) + " Z"


def tail_pts(cx, cy, rx, ry, tip, width=0.2):
    tx, ty = tip
    if ((tx - cx) / max(rx, 1)) ** 2 + ((ty - cy) / max(ry, 1)) ** 2 < 1.25:   # 口元が吹き出しの中・すぐ際: しっぽなし
        return None
    a = math.atan2((ty - cy) / max(ry, 1), (tx - cx) / max(rx, 1))
    ex, ey = cx + rx * math.cos(a), cy + ry * math.sin(a)
    dist = math.hypot(tx - ex, ty - ey)
    length = min(dist * 0.8, max(rx, ry) * 0.75)
    if length < 4:
        return None
    ux, uy = (tx - ex) / max(dist, 1e-6), (ty - ey) / max(dist, 1e-6)
    p1 = (cx + rx * 0.92 * math.cos(a - width), cy + ry * 0.92 * math.sin(a - width))
    p2 = (cx + rx * 0.92 * math.cos(a + width), cy + ry * 0.92 * math.sin(a + width))
    return [p1, (ex + ux * length, ey + uy * length), p2]


def balloon(layer, W, H, seed):
    x, y = num(layer.get("x"), 0.5, -PAD, 1) * W, num(layer.get("y"), 0.1, -PAD, 1) * H
    w, h = max(num(layer.get("w"), 0.2, 0.03, 1.2) * W, 20), max(num(layer.get("h"), 0.3, 0.03, 1.2) * H, 20)
    style = layer.get("style", "speech")
    cx, cy, rx, ry = x + w / 2, y + h / 2, w / 2, h / 2
    lines = lines_of(layer)
    sw = max(2.0, min(w, h) * 0.018) * (1.6 if style == "shout" else 1)
    tail = layer.get("tail")
    tip = (num(tail[0], 0.5, -0.5, 1.5) * W, num(tail[1], 0.5, -0.5, 1.5) * H) if isinstance(tail, list) and len(tail) == 2 else None
    r = rng(seed, "balloon", layer.get("x"), layer.get("y"))
    out = []
    if style in ("narration", "monologue"):
        if style == "narration":
            out.append(f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="#fff" stroke="#000" stroke-width="{f(sw)}"/>')
        else:
            out.append(f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="#fff" fill-opacity="0.85"/>')
        inner = (w * 0.84, h * 0.88)
    elif style == "thought":
        bumps = max(9, int((w + h) / 38))
        circles = []
        for i in range(bumps):
            a = 2 * math.pi * i / bumps
            br = min(rx, ry) * (0.3 + 0.08 * r.random())
            circles.append((cx + rx * 0.86 * math.cos(a), cy + ry * 0.86 * math.sin(a), br))
        trail = []
        if tip:
            for k, t in enumerate((0.35, 0.62, 0.85)):
                trail.append((cx + (tip[0] - cx) * (0.9 + 0.25 * t), cy + (tip[1] - cy) * (0.9 + 0.25 * t), min(rx, ry) * (0.16 - 0.04 * k)))
        shapes = [f'<circle cx="{f(a)}" cy="{f(b)}" r="{f(c)}"/>' for a, b, c in circles]
        out.append(f'<g fill="#000" stroke="#000" stroke-width="{f(sw * 2)}">' + "".join(shapes) +
                   f'<ellipse cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx * 0.9)}" ry="{f(ry * 0.9)}"/></g>')
        out.append('<g fill="#fff">' + "".join(shapes) + f'<ellipse cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx * 0.9)}" ry="{f(ry * 0.9)}"/></g>')
        out += [f'<circle cx="{f(a)}" cy="{f(b)}" r="{f(c)}" fill="#fff" stroke="#000" stroke-width="{f(sw)}"/>' for a, b, c in trail]
        inner = (w * 0.68, h * 0.72)
    else:
        if style == "shout":
            n = max(16, int((w + h) / 26))
            pts = []
            for i in range(n * 2):
                a = math.pi * i / n
                k = 1.0 if i % 2 else 1.18 + 0.12 * r.random()
                pts.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
            body = poly(pts)
        else:
            body = poly(ellipse_pts(cx, cy, rx, ry))
        tp = tail_pts(cx, cy, rx, ry, tip, 0.16 if style != "shout" else 0.12) if tip else None
        shapes = body + (" " + poly(tp) if tp else "")
        dash = f' stroke-dasharray="{f(sw * 3)} {f(sw * 2.2)}"' if style == "whisper" else ""
        if dash:
            out.append(f'<path d="{shapes}" fill="#fff" stroke="#000" stroke-width="{f(sw)}"{dash} stroke-linejoin="round"/>')
            out.append(f'<path d="{body}" fill="#fff"/>')
        else:
            out.append(f'<path d="{shapes}" fill="#000" stroke="#000" stroke-width="{f(sw * 2)}" stroke-linejoin="round"/>')
            out.append(f'<path d="{shapes}" fill="#fff"/>')
        inner = (w * (0.64 if style == "shout" else 0.7), h * (0.66 if style == "shout" else 0.76))
    key = TEXT_FONT.get(style, "antique")
    cols, rows = len(lines), max((len(l) for l in lines), default=1)
    size = min(inner[0] / (cols * 1.18), inner[1] / (rows * 1.06)) * num(layer.get("size"), 1, 0.3, 2)
    size = min(size, W * 0.09)
    paths, _ = vertical_block(key, lines, size, cx, cy)
    out.append(f'<path d="{" ".join(p for p in paths if p)}" fill="#000"/>')
    return "".join(out), (x, y, x + w, y + h)


# ---------- sound effects (書き文字) ----------

def sfx(layer, W, H, seed):
    st = STYLES.get(layer.get("style"), STYLES["impact"])
    text = str(layer.get("text", ""))[:24]
    size = num(layer.get("size"), 0.14, 0.02, 0.8) * W
    vertical = layer.get("vertical", True) is not False
    x0, y0 = num(layer.get("x"), 0.1, -PAD, 1) * W, num(layer.get("y"), 0.1, -PAD, 1) * H
    rot = num(layer.get("rotate"), 0, -90, 90)
    r = rng(seed, "sfx", text, layer.get("x"), layer.get("y"))
    jr, js, jo = (st.get("jitter") or [0, 0, 0])
    grow, squash, skew = st.get("grow", 1), st.get("squash", 1), st.get("skew", 0)
    spacing = 1 + st.get("spacing", 0) - (0.08 if not st.get("spacing") else 0)
    chars = [c for c in text if not c.isspace()]
    n = max(len(chars), 1)
    # 字ごとの位置（回転前、(x0, y0) を 1 字目の箱の左上として並べる）
    items = []
    pos = 0.0
    for i, ch in enumerate(chars):
        sz = size * (1 + (grow - 1) * (i / max(n - 1, 1))) * (1 + js * (r.random() - 0.5) * 2)
        b = st.get("bounce", 0) * size * (1 if i % 2 else -1)
        off = (jo * size * (r.random() - 0.5) * 2, jo * size * (r.random() - 0.5) * 2)
        if vertical:
            cx, cy = x0 + size * squash / 2 + b + off[0], y0 + pos + sz / 2 + off[1]
        else:
            cx, cy = x0 + pos + sz * squash / 2 + off[0], y0 + size / 2 + b + off[1]
        items.append((ch, sz, cx, cy, jr * (r.random() - 0.5) * 2))
        pos += sz * spacing * (squash if not vertical else 1) * (0.9 if ch in "ッっ" else 1)
    # 全体の回転（1 字目の箱の左上を中心に）
    a = math.radians(rot)
    def turn(px, py):
        dx, dy = px - x0, py - y0
        return x0 + dx * math.cos(a) - dy * math.sin(a), y0 + dx * math.sin(a) + dy * math.cos(a)
    paths, xs, ys = [], [], []
    for ch, sz, cx, cy, jrot in items:
        tx, ty = turn(cx, cy)
        d, _ = glyph_path(st["font"], ch, sz, vertical, tx, ty, rot + jrot, squash, skew)
        paths.append(d)
        xs += [tx - sz * 0.6, tx + sz * 0.6]
        ys += [ty - sz * 0.6, ty + sz * 0.6]
    if not paths:
        return "", (x0, y0, x0 + size, y0 + size)
    unit = size
    outline = [[c, w * unit] for c, w in st.get("outline", [])]
    out = []
    if st.get("tails"):
        for k in range(st["tails"]):
            ch, sz, cx, cy, _ = items[r.randrange(len(items))]
            tx, ty = turn(cx, cy)
            ang = a + (math.pi if not vertical else -math.pi / 2) + (r.random() - 0.5) * 0.7
            length = sz * (0.7 + 0.9 * r.random())
            ox, oy = tx + math.cos(ang) * sz * 0.4, ty + math.sin(ang) * sz * 0.4
            out.append(f'<path d="M{f(ox)},{f(oy)} L{f(ox + math.cos(ang) * length)},{f(oy + math.sin(ang) * length)}" '
                       f'stroke="#000" stroke-width="{f(sz * 0.05)}" stroke-linecap="round"/>')
    if st.get("extrude"):
        ex, ey = st["extrude"][0] * unit, st["extrude"][1] * unit
        d = " ".join(paths)
        for k in range(6, 0, -1):
            out.append(f'<path d="{d}" transform="translate({f(ex * k / 6)},{f(ey * k / 6)})" fill="#000" stroke="#000" '
                       f'stroke-width="{f(unit * 0.16)}" stroke-linejoin="round"/>')
    if st.get("echo"):
        d = " ".join(paths)
        for k in range(st["echo"], 0, -1):
            out.append(f'<path d="{d}" fill="none" stroke="#000" stroke-opacity="{0.55 / k:.2f}" stroke-width="{f(unit * (0.1 + 0.32 * k))}" '
                       f'stroke-linejoin="round"/>')
            out.append(f'<path d="{d}" fill="none" stroke="#fff" stroke-width="{f(unit * (0.04 + 0.32 * k) )}" stroke-linejoin="round"/>')
    for k in range(st.get("tremble", 0)):
        dx, dy = (r.random() - 0.5) * unit * 0.12, (r.random() - 0.5) * unit * 0.12
        out.append(f'<path d="{" ".join(paths)}" transform="translate({f(dx)},{f(dy)})" fill="#000" fill-opacity="0.35"/>')
    color = layer.get("color")
    fill = "#fff" if color == "white" else "#000" if color == "black" else st["fill"]
    out.append(stroked(paths, fill, outline, bold=st.get("bold", 0) * unit))
    return "".join(out), (min(xs), min(ys), max(xs), max(ys))


def text_layer(layer, W, H, seed):
    lines = lines_of(layer)
    size = num(layer.get("size"), 0.04, 0.01, 0.3) * W
    vertical = bool(layer.get("vertical"))
    key = STYLES.get(layer.get("style"), {}).get("font") or ("antique" if vertical else "mochiy")
    x, y = num(layer.get("x"), 0.1, -PAD, 1) * W, num(layer.get("y"), 0.1, -PAD, 1) * H
    if vertical:
        cols, rows = len(lines), max(len(l) for l in lines)
        w, h = cols * size * 1.18, rows * size * 1.06
        paths, _ = vertical_block(key, lines, size, x + w / 2, y + h / 2)
    else:
        paths, box = horizontal_block(key, lines, size, 0, 0)
        w, h = box[2] - box[0], box[3] - box[1]
        paths, _ = horizontal_block(key, lines, size, x + w / 2, y + h / 2)
    return stroked(paths, "#000", [["#fff", size * 0.22]]), (x, y, x + w, y + h)


# ---------- effects ----------

def wedge(x1, y1, x2, y2, w1, w2):
    a = math.atan2(y2 - y1, x2 - x1)
    nx, ny = -math.sin(a), math.cos(a)
    return poly([(x1 + nx * w1, y1 + ny * w1), (x2 + nx * w2, y2 + ny * w2), (x2 - nx * w2, y2 - ny * w2), (x1 - nx * w1, y1 - ny * w1)])


def star(cx, cy, ro, ri, n, rot=0.0, jitter=0.0, r=None):
    pts = []
    for i in range(n * 2):
        a = rot + math.pi * i / n
        rr = (ro if i % 2 == 0 else ri) * (1 + (jitter * (r.random() - 0.5) * 2 if r else 0))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return poly(pts)


def effect(layer, W, H, seed):
    name = layer.get("name")
    cx, cy = num(layer.get("cx"), 0.5, -0.5, 1.5) * W, num(layer.get("cy"), 0.5, -0.5, 1.5) * H
    rad = num(layer.get("r"), 0.28, 0.0, 1.5) * W
    density = num(layer.get("density"), 1, 0.2, 3)
    angle = math.radians(num(layer.get("angle"), 0, -360, 360))
    r = rng(seed, "effect", name, layer.get("cx"), layer.get("cy"))
    far = math.hypot(W, H) * 1.2
    shapes = []
    box = (0, 0, W, H)
    if name in ("focus_lines", "beta_flash"):
        n = int((150 if name == "focus_lines" else 300) * density)
        for i in range(n):
            a = 2 * math.pi * i / n + r.random() * 0.03
            inner = rad * (1 + r.random() * (0.9 if name == "focus_lines" else 0.35))
            w = W * (0.0022 if name == "focus_lines" else 0.007) * (0.3 + r.random() ** 2 * 1.7)
            shapes.append(wedge(cx + math.cos(a) * inner, cy + math.sin(a) * inner, cx + math.cos(a) * far, cy + math.sin(a) * far, 0.15, w * 4))
        box = (cx - rad, cy - rad, cx + rad, cy + rad)
    elif name == "uni_flash":
        n = int(120 * density)
        for i in range(n):
            a = 2 * math.pi * i / n + r.random() * 0.04
            r1 = rad * (0.95 + 0.1 * r.random())
            r2 = rad * (1.5 + 0.9 * r.random())
            shapes.append(wedge(cx + math.cos(a) * r1, cy + math.sin(a) * r1, cx + math.cos(a) * r2, cy + math.sin(a) * r2, W * 0.003, 0.3))
        box = (cx - rad * 2, cy - rad * 2, cx + rad * 2, cy + rad * 2)
    elif name in ("speed_lines", "wind_lines"):
        n = int((90 if name == "speed_lines" else 14) * density)
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        for i in range(n):
            o = (r.random() - 0.5) * far
            s0 = (r.random() - 0.5) * far
            length = far * (0.25 + 0.6 * r.random())
            px, py = W / 2 + nx * o + ux * s0, H / 2 + ny * o + uy * s0
            if rad and math.hypot(px - cx, py - cy) < rad:
                continue
            w = W * (0.002 + 0.004 * r.random()) * (0.5 if name == "wind_lines" else 1)
            shapes.append(wedge(px, py, px + ux * length, py + uy * length, 0.2, w))
    elif name == "impact_burst":
        shapes.append(star(cx, cy, rad, rad * 0.45, 12, r.random(), 0.25, r))
        out = f'<path d="{shapes[0]}" fill="#fff" stroke="#000" stroke-width="{f(W * 0.006)}" stroke-linejoin="round"/>'
        return out, (cx - rad, cy - rad, cx + rad, cy + rad)
    elif name == "emphasis_lines":
        n = int(9 * density)
        for i in range(n):
            a = angle + math.pi + (r.random() - 0.5) * 1.6
            r1, r2 = rad * (1 + 0.2 * r.random()), rad * (1.8 + 0.8 * r.random())
            shapes.append(wedge(cx + math.cos(a) * r1, cy + math.sin(a) * r1, cx + math.cos(a) * r2, cy + math.sin(a) * r2, W * 0.002, W * 0.006))
        box = (cx - rad * 2, cy - rad * 2, cx + rad * 2, cy + rad * 2)
    elif name in ("gloom_lines", "falling_lines"):
        n = int((70 if name == "gloom_lines" else 50) * density)
        x1, x2 = (cx - rad, cx + rad) if name == "gloom_lines" else (0, W)
        for i in range(n):
            x = x1 + (x2 - x1) * r.random()
            length = (cy if name == "gloom_lines" else H) * (0.6 + 0.8 * r.random())
            top = 0 if name == "gloom_lines" else H * (r.random() - 0.3)
            shapes.append(wedge(x, top, x, top + length, W * 0.0025, 0.2))
        box = (x1, 0, x2, cy if name == "gloom_lines" else H)
    elif name == "rain_lines":
        for i in range(int(120 * density)):
            x, y = W * 1.2 * r.random() - W * 0.1, H * 1.1 * r.random() - H * 0.05
            l = H * (0.04 + 0.05 * r.random())
            shapes.append(wedge(x, y, x - l * 0.25, y + l, W * 0.0015, W * 0.0015))
    elif name in ("lightning_shock", "rivalry_sparks", "electric_arc"):
        bolts = 3 if name == "lightning_shock" else 2
        for b in range(bolts):
            x = cx + (r.random() - 0.5) * rad * 2
            y = 0 if name == "lightning_shock" else cy - rad * 0.5
            pts = [(x, y)]
            while y < (H * 0.7 if name == "lightning_shock" else cy + rad * 0.5):
                y += H * (0.05 + 0.06 * r.random())
                x += (r.random() - 0.5) * W * 0.12
                pts.append((x, y))
            back = [(px + W * 0.012, py) for px, py in reversed(pts)]
            shapes.append(poly(pts + back))
    elif name in ("sparkle_background", "sparkles", "impact_stars", "glint"):
        n = int((14 if name == "sparkle_background" else 6) * density)
        out = []
        for i in range(n):
            a, d = 2 * math.pi * r.random(), rad * (0.3 + 1.0 * r.random())
            sx, sy = cx + math.cos(a) * d, cy + math.sin(a) * d
            s = W * (0.015 + 0.03 * r.random())
            pts = 5 if name == "impact_stars" else 4
            out.append(f'<path d="{star(sx, sy, s, s * (0.42 if pts == 5 else 0.18), pts, -math.pi / 2)}" fill="#fff" stroke="#000" stroke-width="{f(W * 0.003)}"/>')
        return "".join(out), (cx - rad * 1.3, cy - rad * 1.3, cx + rad * 1.3, cy + rad * 1.3)
    elif name == "rage_zigzag":
        out = []
        for i in range(int(6 * density)):
            a = 2 * math.pi * r.random()
            d = rad * (0.9 + 0.5 * r.random())
            x, y = cx + math.cos(a) * d, cy + math.sin(a) * d
            pts = [(x + k * W * 0.02 * math.cos(a + 1.57), y + k * W * 0.02 * math.sin(a + 1.57) + (W * 0.015 if k % 2 else 0)) for k in range(6)]
            out.append(f'<path d="M{" L".join(f"{f(px)},{f(py)}" for px, py in pts)}" fill="none" stroke="#000" stroke-width="{f(W * 0.006)}"/>')
        return "".join(out), (cx - rad * 1.4, cy - rad * 1.4, cx + rad * 1.4, cy + rad * 1.4)
    elif name in ("falling_petals", "snowfall", "confetti", "embers_sparks", "falling_leaves"):
        out = []
        for i in range(int(30 * density)):
            x, y, s = W * r.random(), H * r.random(), W * (0.008 + 0.012 * r.random())
            if name == "snowfall":
                out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(s * 0.6)}" fill="#fff" stroke="#000" stroke-width="{f(W * 0.0015)}"/>')
            elif name == "confetti":
                out.append(f'<rect x="{f(x)}" y="{f(y)}" width="{f(s)}" height="{f(s * 0.5)}" transform="rotate({r.randrange(180)} {f(x)} {f(y)})" fill="#fff" stroke="#000" stroke-width="{f(W * 0.0015)}"/>')
            elif name == "embers_sparks":
                out.append(f'<path d="{star(x, y, s * 0.7, s * 0.15, 4)}" fill="#000"/>')
            else:
                out.append(f'<ellipse cx="{f(x)}" cy="{f(y)}" rx="{f(s)}" ry="{f(s * 0.5)}" transform="rotate({r.randrange(180)} {f(x)} {f(y)})" fill="#fff" stroke="#000" stroke-width="{f(W * 0.0018)}"/>')
        return "".join(out), (0, 0, W, H)
    elif name == "light_rays":
        for i in range(int(7 * density)):
            a = angle + math.pi / 4 + (r.random() - 0.5) * 0.5
            x = W * r.random()
            shapes.append(wedge(x, 0, x + math.cos(a) * far, math.sin(a) * far, W * 0.004, W * 0.04))
        return f'<path d="{" ".join(shapes)}" fill="#fff" fill-opacity="0.55"/>', (0, 0, W, H)
    else:
        return "", (cx - 10, cy - 10, cx + 10, cy + 10)
    return f'<path d="{" ".join(shapes)}" fill="#000"/>', box


# ---------- marks (漫符) ----------

DROP = [(0, -1), (0.55, -0.15), (0.62, 0.3), (0.32, 0.62), (0, 0.68), (-0.32, 0.62), (-0.62, 0.3), (-0.55, -0.15)]
HEART = "M0,0.35 C-0.9,-0.25 -0.45,-0.95 0,-0.45 C0.45,-0.95 0.9,-0.25 0,0.35 Z"
GLYPH_MARKS = {"surprise_mark": "！", "question_mark": "？", "surprise_question": "！？", "multiple_questions": "？？？",
               "silence_dots": "…", "music_notes": "♪", "sleep_zzz": "Zzz", "cross_marks": "＃"}


def mark(layer, W, H, seed):
    name = layer.get("name")
    x, y = num(layer.get("x"), 0.5, -PAD, 1.1) * W, num(layer.get("y"), 0.1, -PAD, 1.1) * H
    s = num(layer.get("size"), 0.08, 0.02, 0.5) * W
    rot = num(layer.get("rotate"), 0, -180, 180)
    r = rng(seed, "mark", name, layer.get("x"), layer.get("y"))
    cx, cy = x + s / 2, y + s / 2
    sw = max(1.5, s * 0.06)
    t = f' transform="rotate({f(rot)} {f(cx)} {f(cy)})"' if rot else ""
    box = (x, y, x + s, y + s)

    def scaled(d, ox, oy, k):
        import re
        nums = iter(re.findall(r"-?[\d.]+", d))
        return re.sub(r"-?[\d.]+,-?[\d.]+", lambda m: (lambda a, b: f"{f(ox + float(a) * k)},{f(oy + float(b) * k)}")(*m.group().split(",")), d)
    if name in GLYPH_MARKS:
        txt = GLYPH_MARKS[name]
        paths, _ = horizontal_block("dela" if name != "music_notes" else "antique", [txt], s * (0.9 if len(txt) == 1 else 0.6), cx, cy)
        return f'<g{t}>' + stroked(paths, "#000", [["#fff", s * 0.16]]) + "</g>", box
    out = []
    if name in ("sweat_drops", "single_sweat", "droplets_splash", "sweat_spray"):
        k = {"single_sweat": 1, "sweat_drops": 3, "droplets_splash": 5, "sweat_spray": 6}[name]
        for i in range(k):
            ss = s * (0.5 if k == 1 else 0.28) * (0.7 + 0.5 * r.random())
            px, py = cx + (0 if k == 1 else (r.random() - 0.5) * s), cy + (0 if k == 1 else (r.random() - 0.5) * s)
            out.append(f'<path d="{poly([(px + a * ss, py + b * ss) for a, b in DROP])}" fill="#fff" stroke="#000" stroke-width="{f(sw)}" stroke-linejoin="round"/>')
    elif name == "anger_vein":
        k = s * 0.45
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            out.append(f'<path d="M{f(cx + sx * k * 0.18)},{f(cy + sy * k)} Q{f(cx + sx * k * 0.18)},{f(cy + sy * k * 0.18)} {f(cx + sx * k)},{f(cy + sy * k * 0.18)}" '
                       f'fill="none" stroke="#000" stroke-width="{f(s * 0.11)}" stroke-linecap="round"/>')
    elif name in ("floating_hearts", "broken_heart", "pulse_marks"):
        k = 1 if name != "floating_hearts" else 2
        for i in range(k):
            ss = s * (0.5 if k == 1 else 0.32)
            px, py = cx + (i - (k - 1) / 2) * s * 0.45, cy - i * s * 0.2
            out.append(f'<path d="{scaled(HEART, px, py, ss)}" fill="#fff" stroke="#000" stroke-width="{f(sw)}"/>')
        if name == "broken_heart":
            out.append(f'<path d="M{f(cx)},{f(cy - s * 0.25)} L{f(cx - s * 0.08)},{f(cy - s * 0.05)} L{f(cx + s * 0.06)},{f(cy + s * 0.05)} L{f(cx)},{f(cy + s * 0.17)}" fill="none" stroke="#000" stroke-width="{f(sw)}"/>')
        if name == "pulse_marks":
            for a in (-2.4, -1.57, -0.7):
                out.append(f'<path d="M{f(cx + math.cos(a) * s * 0.42)},{f(cy + math.sin(a) * s * 0.42)} L{f(cx + math.cos(a) * s * 0.62)},{f(cy + math.sin(a) * s * 0.62)}" stroke="#000" stroke-width="{f(sw)}" stroke-linecap="round"/>')
    elif name in ("surprise_ticks", "joy_burst", "flinch_spike", "shock_spikes"):
        if name == "surprise_ticks":
            angles = [-2.2, -1.57, -0.95]
        elif name == "joy_burst":
            angles = [2 * math.pi * i / 10 for i in range(10)]
        else:
            angles = []
            pts = [(cx - s * 0.45 + i * s * 0.15, cy + (s * 0.15 if i % 2 else -s * 0.15)) for i in range(7)]
            out.append(f'<path d="M{" L".join(f"{f(a)},{f(b)}" for a, b in pts)}" fill="none" stroke="#000" stroke-width="{f(sw * 1.4)}" stroke-linejoin="round"/>')
        for a in angles:
            out.append(f'<path d="M{f(cx + math.cos(a) * s * 0.2)},{f(cy + math.sin(a) * s * 0.2)} L{f(cx + math.cos(a) * s * 0.5)},{f(cy + math.sin(a) * s * 0.5)}" stroke="#000" stroke-width="{f(sw * 1.3)}" stroke-linecap="round"/>')
    elif name in ("squiggle_cloud", "dizzy_swirl"):
        pts = []
        turns = 3 if name == "dizzy_swirl" else 7
        for i in range(120):
            tt = i / 119
            if name == "dizzy_swirl":
                rr, a = s * 0.45 * tt, tt * turns * 2 * math.pi
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.6))
            else:
                a = tt * turns * 2 * math.pi
                pts.append((cx + s * 0.35 * math.cos(a) + (tt - 0.5) * s * 0.5 + (r.random() - 0.5) * s * 0.05, cy + s * 0.22 * math.sin(a * 1.3)))
        out.append(f'<path d="M{" L".join(f"{f(a)},{f(b)}" for a, b in pts)}" fill="none" stroke="#000" stroke-width="{f(sw)}" stroke-linejoin="round"/>')
    elif name in ("daze_bubbles", "thought_dots"):
        for i in range(3):
            rr = s * (0.12 - 0.03 * i)
            out.append(f'<circle cx="{f(cx + i * s * 0.12)}" cy="{f(cy - i * s * 0.3 + s * 0.3)}" r="{f(rr)}" fill="#fff" stroke="#000" stroke-width="{f(sw)}"/>')
    elif name in ("steam_breath", "sigh_puff", "breath_puff"):
        for i in range(3 if name == "steam_breath" else 1):
            px, py = cx + (i - 1) * s * 0.3, cy + (r.random() - 0.5) * s * 0.3
            for j in range(4):
                a = 2 * math.pi * j / 4
                out.append(f'<circle cx="{f(px + math.cos(a) * s * 0.1)}" cy="{f(py + math.sin(a) * s * 0.08)}" r="{f(s * 0.12)}" fill="#fff" stroke="#000" stroke-width="{f(sw * 0.8)}"/>')
    elif name == "fluster_arcs":
        for i in range(3):
            a = -2.6 + i * 0.9
            px, py = cx + math.cos(a) * s * 0.4, cy + math.sin(a) * s * 0.4
            out.append(f'<path d="M{f(px)},{f(py)} a{f(s * 0.12)},{f(s * 0.12)} 0 1 1 {f(s * 0.12)},{f(s * 0.12)}" fill="none" stroke="#000" stroke-width="{f(sw)}"/>')
    elif name == "lightbulb":
        out.append(f'<circle cx="{f(cx)}" cy="{f(cy - s * 0.08)}" r="{f(s * 0.25)}" fill="#fff" stroke="#000" stroke-width="{f(sw)}"/>')
        out.append(f'<rect x="{f(cx - s * 0.1)}" y="{f(cy + s * 0.15)}" width="{f(s * 0.2)}" height="{f(s * 0.14)}" fill="#000"/>')
        for a in (-2.6, -2.0, -1.57, -1.1, -0.5):
            out.append(f'<path d="M{f(cx + math.cos(a) * s * 0.34)},{f(cy - s * 0.08 + math.sin(a) * s * 0.34)} L{f(cx + math.cos(a) * s * 0.5)},{f(cy - s * 0.08 + math.sin(a) * s * 0.5)}" stroke="#000" stroke-width="{f(sw)}" stroke-linecap="round"/>')
    elif name in ("sparkles", "status_sigil"):
        for i in range(3):
            ss = s * (0.3 if i == 0 else 0.16)
            px, py = cx + (0 if i == 0 else (r.random() - 0.5) * s * 0.8), cy + (0 if i == 0 else (r.random() - 0.5) * s * 0.8)
            out.append(f'<path d="{star(px, py, ss, ss * 0.2, 4, -math.pi / 2)}" fill="#fff" stroke="#000" stroke-width="{f(sw * 0.8)}"/>')
    else:
        return "", box
    return f'<g{t}>' + "".join(out) + "</g>", box


# ---------- panel ----------

ORDER = {"effect": 0, "mark": 1, "sfx": 2, "text": 3, "balloon": 4}


def render(panel):
    aspect = num(panel.get("aspect"), 1, 0.1, 10)
    W, H = 1000.0, 1000.0 / aspect
    seed = panel.get("seed", "")
    layers = panel.get("layers") or []
    parts, boxes = [], [None] * len(layers)
    cid = "c" + hashlib.sha1(str(seed).encode()).hexdigest()[:8]   # ページに並べても衝突しない id
    clip = f'<clipPath id="{cid}"><rect x="0" y="0" width="{f(W)}" height="{f(H)}"/></clipPath>'
    for i in sorted(range(len(layers)), key=lambda i: ORDER.get(layers[i].get("type"), 5)):
        layer = layers[i]
        kind = layer.get("type")
        try:
            svg, box = {"balloon": balloon, "sfx": sfx, "text": text_layer, "effect": effect, "mark": mark}[kind](layer, W, H, seed)
        except KeyError:
            continue
        except Exception as e:  # noqa: BLE001  1 つの層の失敗で全体を落とさない
            print(f"typeset: {kind}: {e}", file=sys.stderr)
            continue
        if kind == "effect":
            svg = f'<g clip-path="url(#{cid})">{svg}</g>'
        parts.append(svg)
        boxes[i] = [round(box[0] / W, 4), round(box[1] / H, 4), round(box[2] / W, 4), round(box[3] / H, 4)]
    vb = f"{f(-PAD * W)} {f(-PAD * H)} {f(W * (1 + 2 * PAD))} {f(H * (1 + 2 * PAD))}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{f(W * (1 + 2 * PAD))}" height="{f(H * (1 + 2 * PAD))}">'
           f'<defs>{clip}</defs>{"".join(parts)}</svg>')
    return {"svg": svg, "boxes": boxes}


_OVERLAY = json.loads((HERE / "data" / "overlay.json").read_text(encoding="utf-8"))
EFFECTS, MARKS = _OVERLAY["effects"], _OVERLAY["marks"]


def styles():
    """書き文字のスタイルと、層で描ける効果・漫符の名前（manpu.json の key）."""
    return {"styles": [{"key": k, "ja": v["ja"], "use": v["use"], "font": FONTS[v["font"]]["name"]} for k, v in STYLES.items()],
            "effects": EFFECTS, "marks": MARKS}


def sample(out):
    import io
    from PIL import Image
    rows = []
    words = {"impact": "ドンッ", "menace": "ゴゴゴゴ", "sharp": "シュッ", "wide": "ドカッ", "pop": "ポンッ", "pote": "ぽよん",
             "marker": "ギャー", "brush": "ズンッ", "syuku": "ひらり", "pen": "カサッ", "cute": "きゅん", "soft": "ほわっ",
             "digital": "ピピピッ", "solid": "ガーン", "block": "ドーン", "tremble": "ゾワッ", "quiet": "しーん", "pulse": "ドクン", "train": "ザワッ"}
    layers = []
    keys = list(STYLES)
    for i, k in enumerate(keys):
        col, row = i % 5, i // 5
        layers.append({"type": "sfx", "style": k, "text": words.get(k, "ドン"), "x": 0.03 + col * 0.195, "y": 0.04 + row * 0.24,
                       "size": 0.045 if k != "menace" else 0.028, "vertical": False})
        layers.append({"type": "text", "text": STYLES[k]["ja"], "x": 0.03 + col * 0.195, "y": 0.16 + row * 0.24, "size": 0.022})
    res = render({"aspect": 1.25, "layers": layers, "seed": "sample"})
    bg = ('<defs><pattern id="tone" width="9" height="9" patternUnits="userSpaceOnUse"><rect width="9" height="9" fill="#fff"/>'
          '<circle cx="4.5" cy="4.5" r="2.6" fill="#555"/></pattern></defs><rect x="-150" y="-150" width="1300" height="1100" fill="url(#tone)"/>')
    svg = res["svg"].replace("<defs>", bg + "<defs>", 1)
    Path(out).with_suffix(".svg").write_text(svg, encoding="utf-8")
    print(out)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "panel" and len(sys.argv) >= 3:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("layers")
        ap.add_argument("--aspect", type=float, required=True, help="コマの横/縦（実寸）")
        ap.add_argument("--out", required=True)
        ap.add_argument("--seed", default="")
        a = ap.parse_args()
        layers = json.loads(Path(a.layers).read_text(encoding="utf-8"))
        res = render({"aspect": a.aspect, "layers": layers, "seed": a.seed})
        Path(a.out).write_text(res["svg"], encoding="utf-8")
        print(json.dumps(res["boxes"]))
        return 0
    if cmd == "serve":
        for line in sys.stdin:
            try:
                req = json.loads(line)
                res = {"id": req.get("id"), "panels": [render(p) for p in req.get("panels", [])]}
            except Exception as e:  # noqa: BLE001
                res = {"id": None, "error": str(e)}
            sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        return 0
    if cmd == "styles":
        print(json.dumps(styles(), ensure_ascii=False))
        return 0
    if cmd == "sample" and len(sys.argv) == 3:
        sample(sys.argv[2])
        return 0
    print("使い方: typeset.py panel LAYERS.json --aspect A --out OUT.svg | serve | styles | sample OUT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
