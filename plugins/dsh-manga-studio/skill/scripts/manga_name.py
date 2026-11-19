"""ネーム TOML の読み込み・正規化・コマの配置計算（他のスクリプトの共通部品）.

標準ライブラリのみ。ネームの書式は references/name-format.md。
座標は内部ではページ全体を 0-1000 に正規化した xyxy（norm_1000）で持つ。
"""
from __future__ import annotations

import json
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

SHOTS = {
    "extreme_long": ("超ロング", "extreme long shot, tiny figures in a wide landscape"),
    "long": ("ロング", "long shot, full body with surroundings"),
    "full": ("フル", "full shot, whole body"),
    "medium": ("ミドル", "medium shot, from the knees or waist up"),
    "bust": ("バストアップ", "bust shot, chest up"),
    "closeup": ("アップ", "close-up of the face"),
    "extreme_closeup": ("超アップ", "extreme close-up on the eyes or a detail"),
    "insert": ("インサート", "insert shot of an object, no faces"),
}
SHOT_ALIASES = {
    "超ロング": "extreme_long", "遠景": "extreme_long", "ロング": "long", "引き": "long",
    "フル": "full", "全身": "full", "ミドル": "medium", "ニー": "medium", "ウエスト": "medium",
    "バスト": "bust", "バストアップ": "bust", "アップ": "closeup", "顔アップ": "closeup",
    "超アップ": "extreme_closeup", "クローズアップ": "closeup", "インサート": "insert", "物": "insert",
}
ANGLES = {
    "eye": ("アイレベル", "eye-level angle"),
    "high": ("俯瞰", "high angle looking down"),
    "low": ("煽り", "low angle looking up"),
    "bird": ("真俯瞰", "bird's-eye view from directly above"),
    "worm": ("真煽り", "worm's-eye view from directly below"),
    "dutch": ("斜め", "dutch angle, tilted frame"),
    "pov": ("主観", "first-person point of view"),
    "over_shoulder": ("肩越し", "over-the-shoulder shot"),
    "back": ("背後", "view from behind the character"),
    "profile": ("横顔", "side view, profile"),
}
ANGLE_ALIASES = {
    "アイレベル": "eye", "水平": "eye", "俯瞰": "high", "見下ろし": "high", "煽り": "low", "あおり": "low",
    "見上げ": "low", "真俯瞰": "bird", "真上": "bird", "真煽り": "worm", "真下": "worm", "斜め": "dutch",
    "ダッチ": "dutch", "主観": "pov", "肩越し": "over_shoulder", "背後": "back", "後ろ姿": "back", "横顔": "profile", "横": "profile",
}
TRANSITIONS = {
    "moment": "瞬間（ほぼ同じ時点のわずかな変化）",
    "action": "動作（一つの動作の段階）",
    "subject": "対象（同じ場面の別の人や物へ）",
    "scene": "場面（時間・場所が飛ぶ）",
    "aspect": "様相（時間を止めて場所や気分の断片を並べる）",
    "non_sequitur": "飛躍（意味のつながりを外す）",
}
# コマの描き込みの段階（`detail`）。決めコマだけ人物も背景もきちんと描き、ほかは軽く描く。
DETAILS = {
    "key": "決め（人物を丁寧に。background が無ければ背景もきちんと。1 ページに 1 つまで）",
    "normal": "通常（人物は線とベタ、背景は場所が分かる最低限）",
    "light": "簡略（線を減らす。顔は簡略、背景は省くか白・トーン）",
}
DETAIL_ALIASES = {"決め": "key", "決めコマ": "key", "見せ": "key", "通常": "normal", "簡略": "light", "つなぎ": "light"}


def normalize_detail(v):
    if v is None:
        return None
    v = str(v).strip()
    return v if v in DETAILS else DETAIL_ALIASES.get(v, v)


TEXT_TYPES = {
    # type: (日本語, krea の吹き出し kind または None, krea の text kind または None)
    "dialogue": ("台詞", "speech_balloon", "dialogue"),
    "shout": ("叫び", "shout_balloon", "dialogue"),
    "whisper": ("ささやき", "speech_balloon", "dialogue"),
    "thought": ("心の声", "thought_balloon", "dialogue"),
    "monologue": ("モノローグ", "narration_box", "narration_box"),
    "narration": ("ナレーション", "narration_box", "narration_box"),
    "sfx": ("擬音・描き文字", None, "sound_effect"),
    "aside": ("欄外の小文字・手書きのつぶやき", None, None),
    "sign": ("作中の文字（看板・画面・手紙）", None, None),
}
BALLOON_TYPES = {"dialogue", "shout", "whisper", "thought", "monologue", "narration"}
POSITIONS = {"right": 0, "center": 1, "left": 2, "右": 0, "中央": 1, "中": 1, "左": 2}
FORMATS = {"page", "yonkoma", "webtoon"}

# ページの体裁（norm_1000 に対する比率）
MARGIN_X = 70      # 左右の余白（基本枠の外）
MARGIN_TOP = 60
MARGIN_BOTTOM = 70
GUTTER_X = 13      # 段の中のコマの間（左右）。Manga109-s の中央値 12（ページ幅 1000 に対して）
GUTTER_Y = 18      # 段と段の間（上下）。同 16（ページ高 1000 に対して）。左右より上下を広く（corpus-norms.md §7）


class NameError_(Exception):
    """ネームの読み込みで続行できない誤り."""


@dataclass
class Issue:
    code: str
    level: str          # FAIL / WARN / INFO
    where: str
    message: str

    def as_dict(self):
        return {"code": self.code, "level": self.level, "where": self.where, "message": self.message}


@dataclass
class Panel:
    id: str
    page: int
    index: int                    # ファイル内の順（作者の意図した読み順）
    raw: dict
    bbox: list = field(default_factory=list)   # norm_1000 xyxy
    geo_order: int = -1           # 配置から計算した読み順

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def shot(self):
        return normalize_shot(self.raw.get("shot"))

    @property
    def angle(self):
        return normalize_angle(self.raw.get("angle"))

    @property
    def characters(self):
        return self.raw.get("characters", []) or []

    @property
    def texts(self):
        return self.raw.get("text", []) or []


def normalize_shot(v):
    if v is None:
        return None
    v = str(v).strip()
    return v if v in SHOTS else SHOT_ALIASES.get(v, v)


def normalize_angle(v):
    if v is None:
        return None
    v = str(v).strip()
    return v if v in ANGLES else ANGLE_ALIASES.get(v, v)


def load_json(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def manpu_index():
    """名前と別名 → (key, 定義). marks と effects を一つの索引にする."""
    data = load_json("manpu.json")
    idx = {}
    for group in ("marks", "effects"):
        for key, d in data[group].items():
            d = dict(d, group=group, key=key)
            idx[key] = d
            for a in d.get("aliases", []):
                idx.setdefault(a, d)
    return idx


def load_toml(path):
    """TOML を読む. 拡張子が .json なら JSON として読む（LLM に JSON で書かせたネームを、そのまま検査するため）."""
    try:
        if str(path).endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise NameError_(f"JSON の最上位がオブジェクトではありません: {path}")
            return data
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise NameError_(f"ファイルがありません: {path}")
    except tomllib.TOMLDecodeError as e:
        raise NameError_(f"TOML として読めません: {path}: {e}")
    except json.JSONDecodeError as e:
        raise NameError_(f"JSON として読めません: {path}: {e}")


def load_name(path):
    """ネーム TOML を読み、人物表を解決して dict を返す.

    人物は、ネーム内の [[characters]] と、meta.cast（ネームからの相対パス）の [[characters]] を合わせる。
    同じ id はネーム内の定義が勝つ。
    """
    path = Path(path)
    doc = load_toml(path)
    if doc.get("meta") is None:
        doc["meta"] = {}
    if not isinstance(doc["meta"], dict):
        raise NameError_(f"meta は表（[meta]）が必要です: {doc['meta']!r}")
    for key in ("characters", "pages"):
        if key in doc and doc[key] is not None and not isinstance(doc[key], list):
            raise NameError_(f"{key} は表の配列（[[{key}]]）が必要です")
    for i, c in enumerate(doc.get("characters") or []):
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"].strip():
            raise NameError_(f"characters[{i}] は id（文字列）を持つ表が必要です: {c!r}")
    ids = [c["id"] for c in doc.get("characters") or []]
    if len(ids) != len(set(ids)):
        raise NameError_(f"characters の id が重複しています: {sorted({x for x in ids if ids.count(x) > 1})}")
    meta = doc["meta"]
    mt = find_manga_toml(path)
    if mt:
        work = load_toml(mt)
        for sec, keys in (("work", ("format", "first_page_side", "rating", "title")),
                          ("render", ("style", "negative", "page_aspect", "text"))):
            for k in keys:
                v = work.get(sec, {}).get(k)
                if v not in (None, "") and k not in meta:
                    meta[k] = v
        doc["_manga_toml"] = str(mt)
    cast = {}
    cast_path = meta.get("cast")
    if cast_path:
        cp = (path.parent / cast_path).resolve()
        for c in load_toml(cp).get("characters", []) or []:
            if isinstance(c, dict) and isinstance(c.get("id"), str):
                cast[c["id"]] = c
    for c in doc.get("characters") or []:
        cast[c["id"]] = c
    doc["_cast"] = cast
    doc["_path"] = str(path)
    return doc


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _box4(v, lo, hi):
    return (isinstance(v, list) and len(v) == 4 and all(_finite(x) for x in v)
            and v[0] < v[2] and v[1] < v[3] and min(v) >= lo and max(v) <= hi)


BLEED_SIDES = {"top", "bottom", "left", "right"}


def validate_structure(doc):
    """配置計算より前に、型・配列の長さ・数値の範囲・ページ番号を確かめる（E 群の FAIL）.

    ここで誤りのあるページは、配置を計算しない（計算すると例外で落ちるか、壊れた矩形になるため）。
    戻り値: (issues, 壊れたページ番号の集合)
    """
    issues, bad = [], set()

    def err(code, where, msg, page=None):
        issues.append(Issue(code, "FAIL", where, msg))
        if page is not None:
            bad.add(page)

    pa = doc.get("meta", {}).get("page_aspect")
    if pa is not None and not (_finite(pa) and 0.2 <= pa <= 50):
        err("E11", "meta", f"page_aspect（縦/横）は 0.2〜50 の数: {pa!r}")
        return issues, {"*"}
    pages = doc.get("pages", [])
    if not isinstance(pages, list) or not all(isinstance(pg, dict) for pg in pages):
        err("E11", "全体", "pages は表（[[pages]]）の配列")
        return issues, {"*"}
    seen = {}
    for p_i, page in enumerate(pages):
        number = page.get("number", p_i + 1)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            err("E10", f"pages[{p_i}]", f"number は 1 以上の整数: {number!r}", p_i)
            continue
        if number in seen:
            err("E10", f"P{number}", f"ページ番号が重複（pages[{seen[number]}] と pages[{p_i}]）。書き出しで上書きされる", p_i)
        seen.setdefault(number, p_i)
        pw = f"P{number}"
        tiers = page.get("tiers")
        if tiers is not None and not (isinstance(tiers, list) and tiers and all(_finite(t) and t > 0 for t in tiers)):
            err("E11", pw, f"tiers は正の数の配列: {tiers!r}", p_i)
        panels = page.get("panels", [])
        if not isinstance(panels, list) or not all(isinstance(x, dict) for x in panels):
            err("E11", pw, "panels は表（[[pages.panels]]）の配列", p_i)
            continue
        for k, p in enumerate(panels):
            w = f"{pw} {p.get('id') or f'{pw}-{k + 1}'}"
            if "tier" in p and not (isinstance(p["tier"], int) and not isinstance(p["tier"], bool) and p["tier"] >= 0):
                err("E11", w, f"tier は 0 以上の整数: {p['tier']!r}", p_i)
            for key in ("width", "height", "gap"):
                if key in p and not (_finite(p[key]) and (p[key] > 0 or (key == "gap" and p[key] >= 0))):
                    err("E11", w, f"{key} は正の数: {p[key]!r}", p_i)
            if "column" in p and not isinstance(p["column"], (int, str)):
                err("E11", w, f"column は整数か文字列: {p['column']!r}", p_i)
            if "bbox" in p and not _box4(p["bbox"], 0, 1000):
                err("E08", w, f"bbox は 0〜1000 の数 4 つ [x1,y1,x2,y2]（x1<x2, y1<y2）: {p['bbox']!r}", p_i)
            bl = p.get("bleed")
            if bl is not None and not isinstance(bl, bool):
                if not isinstance(bl, str) or not {x.strip() for x in bl.split(",")} <= BLEED_SIDES:
                    err("E11", w, f"bleed は true/false か \"top,left\" のような辺の指定: {bl!r}", p_i)
            for key in ("characters", "text", "manpu", "effects"):
                if key in p and not isinstance(p[key], list):
                    err("E11", w, f"{key} は配列", p_i)
            if isinstance(p.get("effects"), list) and not all(isinstance(e, str) for e in p["effects"]):
                err("E11", w, "effects の要素は文字列", p_i)
            for key in ("characters", "text"):
                for j, it in enumerate(p.get(key, []) if isinstance(p.get(key), list) else []):
                    if not isinstance(it, dict):
                        err("E11", f"{w} {key}[{j}]", "表（{...}）が必要", p_i)
                        continue
                    if "box" in it and not _box4(it["box"], 0, 1):
                        err("E12", f"{w} {key}[{j}]", f"box はコマの中の 0〜1 の数 4 つ [x1,y1,x2,y2]: {it['box']!r}", p_i)
                    if key == "characters":
                        if "manpu" in it and not (isinstance(it["manpu"], list) and all(isinstance(m, str) for m in it["manpu"])):
                            err("E11", f"{w} {key}[{j}]", "manpu は文字列の配列", p_i)
                        if not isinstance(it.get("id"), str):
                            err("E11", f"{w} {key}[{j}]", "人物の id（文字列）が要る", p_i)
                    if key == "text":
                        if "lines" in it and not (isinstance(it["lines"], list) and all(isinstance(x, str) for x in it["lines"])):
                            err("E11", f"{w} {key}[{j}]", "lines は文字列の配列", p_i)
                        if "text" in it and not isinstance(it["text"], str):
                            err("E11", f"{w} {key}[{j}]", "text は文字列", p_i)
                        if it.get("type", "dialogue") not in TEXT_TYPES:
                            err("E06", f"{w} {key}[{j}]", f"type={it.get('type')!r} は {', '.join(TEXT_TYPES)} のどれか", p_i)
    return issues, bad


MIN_PANEL = 8.0   # 計算後のコマの最小の幅・高さ（norm_1000）


def precheck(doc):
    """描画・書き出しの前の共通検査. 構造（validate_structure）に加え、配置を計算して E07（段の不足）と
    E08（壊れた・小さすぎる矩形）も確かめる。lint の E 群のうち、出力を壊すものをすべて含む。戻り値は issues."""
    issues, bad = validate_structure(doc)
    if "*" in bad:
        return issues
    fmt = name_format(doc)
    if fmt not in FORMATS:
        issues.append(Issue("E01", "FAIL", "meta", f"format={fmt!r} は page / yonkoma / webtoon のどれか"))
        return issues
    for p_i, page in enumerate(doc.get("pages", [])):
        if p_i in bad:
            continue
        pw = f"P{page.get('number', p_i + 1)}"
        if not page.get("panels"):
            issues.append(Issue("E02", "FAIL", pw, "コマがない"))
            continue
        layout_page(page, fmt, issues, pw)
        for p in page.get("panels", []):
            b = p.get("_bbox")
            where = f"{pw} {p.get('id', '?')}"
            if b is None:
                issues.append(Issue("E08", "FAIL", where, "配置を計算できない（tier か bbox が要る）"))
            elif not (b[2] - b[0] >= MIN_PANEL and b[3] - b[1] >= MIN_PANEL and min(b) >= 0 and max(b) <= 1000):
                issues.append(Issue("E08", "FAIL", where, f"配置の結果が壊れた・小さすぎる矩形になった: {b}（tiers・width・height を見直す）"))
                p.pop("_bbox", None)
    return issues


def find_manga_toml(start):
    """ネームのある場所から上へたどって、最初に見つかる manga.toml."""
    p = Path(start).resolve()
    for d in [p.parent, *p.parents]:
        if (d / "manga.toml").exists():
            return d / "manga.toml"
    return None


def name_format(doc):
    return doc.get("meta", {}).get("format", "page")


def page_side(doc, number):
    """右綴じの本でのページの左右. first_page_side（既定 left = 奇数ページが左）から決める."""
    first = doc.get("meta", {}).get("first_page_side", "left")
    odd_side = first if first in ("left", "right") else "left"
    if number % 2 == 1:
        return odd_side
    return "right" if odd_side == "left" else "left"


def iter_panels(doc):
    out = []
    for p_i, page in enumerate(doc.get("pages", [])):
        number = page.get("number", p_i + 1)
        for i, raw in enumerate(page.get("panels", [])):
            pid = raw.get("id") or f"P{number}-{i + 1}"
            out.append(Panel(id=pid, page=number, index=i, raw=raw))
    return out


# ---------------------------------------------------------------- 配置計算

def _frame(page, fmt):
    """基本枠（コマを並べる範囲）. norm_1000."""
    if fmt == "webtoon":
        return [0, 0, 1000, 1000]
    return [MARGIN_X, MARGIN_TOP, 1000 - MARGIN_X, 1000 - MARGIN_BOTTOM]


def layout_page(page, fmt="page", issues=None, where="P?"):
    """1 ページのコマの bbox（norm_1000）を計算して panel dict の '_bbox' に入れる.

    明示の bbox があればそれを使う。無ければ tier（段）・width（段内の幅比）・column（段内で縦に積む組）から計算する。
    段の中は右から左へ並べる（右綴じ）。読み順は「段が上から下、段の中は右から左、縦に積んだ組は上から下」。
    """
    issues = issues if issues is not None else []
    panels = page.get("panels", [])
    fx1, fy1, fx2, fy2 = _frame(page, fmt)

    for k, p in enumerate(panels):
        p.pop("_auto_order", None)
        p.pop("_bbox", None)
        p.pop("_written_order", None)
        if page.get("order") == "written":
            # 重なり・はめ込みの型など、形から読む順を推定できないページ。書いた順を読む順として保つ
            p["_written_order"] = k
    if fmt == "yonkoma" and "tiers" not in page and not any("bbox" in p for p in panels):
        _layout_yonkoma(panels, (fx1, fy1, fx2, fy2))
        return panels
    if fmt == "webtoon" and not any("bbox" in p for p in panels):
        _layout_webtoon(panels)
        return panels

    auto = [p for p in panels if "bbox" not in p]
    if auto and fmt == "page" and not page.get("tiers") and not any(
            k in p for p in auto for k in ("tier", "column", "width", "height")):
        _auto_tiers(auto)
    n_tiers = max([p.get("tier", 0) for p in auto], default=-1) + 1
    tiers = page.get("tiers")
    if n_tiers:
        if not tiers:
            tiers = [1.0] * n_tiers
        if len(tiers) < n_tiers:
            issues.append(Issue("E07", "FAIL", where, f"tiers の数（{len(tiers)}）が、使われている段（{n_tiers}）より少ない"))
            tiers = list(tiers) + [1.0] * (n_tiers - len(tiers))
        total = sum(tiers) or 1.0
        avail_h = (fy2 - fy1) - GUTTER_Y * (len(tiers) - 1)
        y = fy1
        tier_y = []
        for t in tiers:
            h = avail_h * t / total
            tier_y.append((y, y + h))
            y += h + GUTTER_Y
        for t_i in range(n_tiers):
            members = [p for p in auto if p.get("tier", 0) == t_i]
            if not members:
                continue
            # column ごとにまとめる（column 指定が無いコマは単独の列）
            columns = []
            seen = {}
            for p in members:
                col = p.get("column")
                if col is None:
                    columns.append([p])
                elif col in seen:
                    seen[col].append(p)
                else:
                    seen[col] = [p]
                    columns.append(seen[col])
            widths = [c[0].get("width", 1.0) for c in columns]
            wtotal = sum(widths) or 1.0
            avail_w = (fx2 - fx1) - GUTTER_X * (len(columns) - 1)
            x_right = fx2
            ty1, ty2 = tier_y[t_i]
            for c_i, (col, w) in enumerate(zip(columns, widths)):
                cw = avail_w * w / wtotal
                x1, x2 = x_right - cw, x_right
                x_right = x1 - GUTTER_X
                hs = [p.get("height", 1.0) for p in col]
                htotal = sum(hs) or 1.0
                avail_ch = (ty2 - ty1) - GUTTER_Y * (len(col) - 1)
                cy = ty1
                for s_i, (p, hh) in enumerate(zip(col, hs)):
                    ch = avail_ch * hh / htotal
                    p["_bbox"] = [x1, cy, x2, cy + ch]
                    p["_auto_order"] = (t_i, c_i, s_i)
                    cy += ch + GUTTER_Y
    for p in panels:
        if "bbox" in p:
            p["_bbox"] = [float(v) for v in p["bbox"]]
        if p.get("bleed") and "_bbox" in p:
            p["_bbox"] = _bleed(p["_bbox"], (fx1, fy1, fx2, fy2), p.get("bleed"))
        if "_bbox" in p:
            p["_bbox"] = [round(v, 1) for v in p["_bbox"]]
    return panels


def _auto_tiers(panels):
    """段の指定が一つも無いページに、叩き台の段組みを入れる（書いた順を保つ）.

    1 段に 2 コマずつ、上から。奇数なら最上段を 1 コマの横長（場所を見せる引きに向く）。
    """
    n = len(panels)
    rows = [[0]] if n % 2 else []
    start = 1 if n % 2 else 0
    rows += [[i, i + 1] for i in range(start, n, 2)]
    for t, row in enumerate(rows):
        for i in row:
            panels[i]["tier"] = t
            panels[i].setdefault("width", 0.55 if len(row) == 2 and i == row[0] else 0.45 if len(row) == 2 else 1.0)


def _bleed(b, frame, how):
    """タチキリ: 基本枠に接している辺を紙の端まで伸ばす. how は true か 'top,right' のような辺の指定."""
    x1, y1, x2, y2 = b
    fx1, fy1, fx2, fy2 = frame
    eps = 1.0
    sides = {"top", "bottom", "left", "right"} if how is True else {s.strip() for s in str(how).split(",")}
    if "top" in sides and abs(y1 - fy1) < eps:
        y1 = 0
    if "bottom" in sides and abs(y2 - fy2) < eps:
        y2 = 1000
    if "left" in sides and abs(x1 - fx1) < eps:
        x1 = 0
    if "right" in sides and abs(x2 - fx2) < eps:
        x2 = 1000
    return [x1, y1, x2, y2]


def _layout_yonkoma(panels, frame):
    """4コマ: 4 コマずつ縦一列。8 コマなら右列→左列の 2 本."""
    fx1, fy1, fx2, fy2 = frame
    strips = [panels[i:i + 4] for i in range(0, len(panels), 4)]
    n = max(len(strips), 1)
    col_w = ((fx2 - fx1) - 60 * (n - 1)) / n
    col_w = min(col_w, 520)
    total_w = col_w * n + 60 * (n - 1)
    x_right = 500 + total_w / 2
    for s_i, strip in enumerate(strips):
        x1, x2 = x_right - col_w, x_right
        x_right = x1 - 60
        h = ((fy2 - fy1) - GUTTER_Y * 3) / 4
        for k, p in enumerate(strip):
            y1 = fy1 + k * (h + GUTTER_Y)
            p["_bbox"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y1 + h, 1)]
            p["_auto_order"] = (s_i, k)


def _layout_webtoon(panels):
    """縦読み: 1 ページ（1 区切り）の中でコマを縦に積む. 高さは height（幅に対する比、既定 1.0）、前の余白は gap（同）."""
    total = 0.0
    spans = []
    for p in panels:
        gap = float(p.get("gap", 0.25 if spans else 0.0))
        h = float(p.get("height", 1.0))
        total += gap
        spans.append((total, total + h))
        total += h
    total = total or 1.0
    for k, (p, (a, b)) in enumerate(zip(panels, spans)):
        x1, x2 = (0.0, 1000.0) if p.get("bleed", True) else (60.0, 940.0)
        p["_bbox"] = [x1, round(a / total * 1000, 1), x2, round(b / total * 1000, 1)]
        p["_auto_order"] = (k,)
    return total


def webtoon_aspect(page):
    """縦読みの区切り全体の 縦/横 比."""
    total = 0.0
    for k, p in enumerate(page.get("panels", [])):
        total += float(p.get("gap", 0.25 if k else 0.0)) + float(p.get("height", 1.0))
    return total or 1.0


def geometric_order(page_panels, fmt="page"):
    """読み順（右綴じ）. 戻り値は panel のリスト（読む順）.

    段組みで自動配置したページは、配置の構造（4コマの本→コマ、段→列（右から）→縦積み、縦読みの順）から決める。
    明示の bbox が混じるページは形から推定する: 上下に重ならない帯に分け、帯の中は右端の大きい順、同じ列なら上から。
    縦読みは上から。
    """
    items = [p for p in page_panels if "_bbox" in p]
    if items and all("_written_order" in p for p in items):
        return sorted(items, key=lambda p: p["_written_order"])
    if items and all("_auto_order" in p for p in items):
        return sorted(items, key=lambda p: p["_auto_order"])
    if fmt == "webtoon":
        return sorted(items, key=lambda p: p["_bbox"][1])
    rows = []
    for p in sorted(items, key=lambda p: p["_bbox"][1]):
        y1, y2 = p["_bbox"][1], p["_bbox"][3]
        for row in rows:
            ry1, ry2 = row["y"]
            overlap = min(y2, ry2) - max(y1, ry1)
            if overlap > 0.5 * min(y2 - y1, ry2 - ry1):
                row["items"].append(p)
                row["y"] = (min(ry1, y1), max(ry2, y2))
                break
        else:
            rows.append({"y": (y1, y2), "items": [p]})
    order = []
    for row in sorted(rows, key=lambda r: r["y"][0]):
        order.extend(sorted(row["items"], key=lambda p: (-round(p["_bbox"][2] / 25), p["_bbox"][1])))
    return order


# ---------------------------------------------------------------- 文字

def text_body(t):
    """吹き出しの文字. lines（配列）か text（文字列、改行または ／ で行を分ける）."""
    if isinstance(t.get("lines"), list):
        return [str(s) for s in t["lines"]]
    s = str(t.get("text", ""))
    parts = []
    for chunk in s.replace("／", "\n").split("\n"):
        if chunk.strip():
            parts.append(chunk.strip())
    return parts


def count_chars(s):
    """吹き出しの字数. 空白と改行を数えない. 三点リーダ・ダッシュ・記号は 1 字."""
    return sum(1 for ch in s if not ch.isspace())


def eprint(*a):
    print(*a, file=sys.stderr)


# ---------------------------------------------------------------- コマの中の配置

FIGURE = {
    # shot: (人物の高さ / コマの高さ, 1 人のときの幅 / コマの幅, 下端の位置 / コマの高さ)
    # long〜closeup は Manga109-s の 1 人のコマの体の枠（頭から見えている下端まで、腕や身振りを含む）の中央値
    # （2026-11-15、references/corpus-norms.md §7）。体の枠は多くがコマの下端で切れる
    "extreme_long": (0.22, 0.08, 0.88),
    "long": (0.37, 0.40, 0.84),
    "full": (0.86, 0.65, 0.99),
    "medium": (0.86, 0.71, 0.99),
    "bust": (0.92, 0.80, 1.0),
    "closeup": (0.98, 0.77, 1.0),
    "extreme_closeup": (1.0, 0.9, 1.0),
    "insert": (0.5, 0.3, 0.95),
}
SLOT_X = {0: 0.72, 1: 0.5, 2: 0.28}
# pos を書かないときの人物の横位置（右から）。Manga109-s の体の枠の中心の中央値（2 人 0.68/0.32、3 人 0.75/0.49/0.23）
AUTO_X = {1: [0.5], 2: [0.68, 0.32], 3: [0.75, 0.49, 0.23]}


def _rel(box, rb):
    """コマの bbox の中の相対座標 rb（0-1）を、ページの norm_1000 に直す."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [round(x1 + rb[0] * w, 1), round(y1 + rb[1] * h, 1), round(x1 + rb[2] * w, 1), round(y1 + rb[3] * h, 1)]


def _clamp(b, box):
    x1, y1, x2, y2 = box
    bw, bh = min(b[2] - b[0], x2 - x1), min(b[3] - b[1], y2 - y1)
    nx1 = min(max(b[0], x1), x2 - bw)
    ny1 = min(max(b[1], y1), y2 - bh)
    return [round(nx1, 1), round(ny1, 1), round(nx1 + bw, 1), round(ny1 + bh, 1)]


def balloon_shape(t, line_chars=10):
    """吹き出しの列数と 1 列の字数（縦書き）. 改行が無ければ line_chars で折る."""
    lines = text_body(t)
    if len(lines) == 1 and count_chars(lines[0]) > line_chars and not isinstance(t.get("lines"), list):
        s = lines[0].replace(" ", "").replace("　", "")
        lines = [s[i:i + line_chars] for i in range(0, len(s), line_chars)]
    cols = max(len(lines), 1)
    rows = max((count_chars(ln) for ln in lines), default=1)
    return lines, cols, max(rows, 1)


def place_panel(panel, aspect, manpu=None):
    """1 コマの中の人物・吹き出し・文字・漫符・効果の bbox（ページの norm_1000）を推定する.

    aspect はページの 縦/横 の実寸比。推定は叩き台で、各項目に box（コマ内の相対 0-1）を書けばそれが勝つ。
    戻り値: {"characters": [...], "balloons": [...], "texts": [...], "marks": [...], "effects": [...]}
    """
    manpu = manpu if manpu is not None else manpu_index()
    box = panel["_bbox"]
    px1, py1, px2, py2 = box
    pw, ph = px2 - px1, py2 - py1
    ph_real = ph * aspect                       # 横と同じ単位に直した高さ
    shot = normalize_shot(panel.get("shot")) or "medium"
    fig_h, fig_w, fig_bottom = FIGURE.get(shot, FIGURE["medium"])
    chars = panel.get("characters", []) or []
    texts = panel.get("text", []) or []
    out = {"characters": [], "balloons": [], "texts": [], "marks": [], "effects": []}
    has_balloons = any(t.get("type", "dialogue") in BALLOON_TYPES for t in texts)

    # 人物
    n = len(chars)
    centers = {}
    for k, c in enumerate(chars):
        if isinstance(c.get("box"), list) and len(c["box"]) == 4:
            centers[k] = (c["box"][0] + c["box"][2]) / 2
            continue
        slot = POSITIONS.get(str(c.get("pos", "")))
        if slot is not None:
            centers[k] = SLOT_X[slot]
        elif n in AUTO_X:
            centers[k] = AUTO_X[n][k]
        else:
            centers[k] = 1 - (k + 0.5) / n
    for k, c in enumerate(chars):
        if isinstance(c.get("box"), list):
            cb = _rel(box, c["box"])
        else:
            cx = centers[k]
            # 幅: 1 人のときの目安（FIGURE）を、隣の人物との中心の間隔の 95% までに縮め（人物の枠を重ねない）、
            # さらにコマの端で切れないよう中心を保ったまま左右対称に縮める（中心がずれると吹き出しの位置もずれる）
            # 左右の空き: 自動の人物とは中心の間の半分、box で固定した人物とはその枠の端まで
            left, right = 0.02, 0.98
            for j, cj in enumerate(chars):
                if j == k:
                    continue
                if isinstance(cj.get("box"), list) and len(cj["box"]) == 4:
                    bx1, bx2 = cj["box"][0], cj["box"][2]
                    if bx2 <= cx:
                        left = max(left, bx2 + 0.01)
                    elif bx1 >= cx:
                        right = min(right, bx1 - 0.01)
                    else:                       # 中心が固定の枠の中: 空きの広い側へずらす
                        if bx1 - 0.02 >= 0.98 - bx2:
                            cx = (0.02 + bx1) / 2
                            right = min(right, bx1 - 0.01)
                        else:
                            cx = (bx2 + 0.98) / 2
                            left = max(left, bx2 + 0.01)
                else:
                    if centers[j] < cx:
                        left = max(left, cx - (cx - centers[j]) * 0.475)
                    elif centers[j] > cx:
                        right = min(right, cx + (centers[j] - cx) * 0.475)
            w = max(min(fig_w, 2 * (cx - left), 2 * (right - cx)), 0.04)
            top = fig_bottom - fig_h
            if has_balloons and shot in ("full", "medium", "long", "extreme_long"):
                top = max(top, 0.2)
            cb = _rel(box, [cx - w / 2, max(top, 0.0), cx + w / 2, fig_bottom])
        out["characters"].append({"src": c, "bbox": cb})

    # 吹き出し（縦書き、右上から左へ）
    cs = min(pw, ph_real) * 0.06                # 1 字の大きさ（横の単位）
    cs = max(min(cs, 26.0), 9.0)
    gap = cs * 0.6
    cursor_x = px2 - gap
    row_top = py1 + gap / aspect
    row_bottom = row_top
    char_by_id = {o["src"].get("id"): o for o in out["characters"]}
    for t in texts:
        ttype = t.get("type", "dialogue")
        lines, cols, rows = balloon_shape(t)
        if isinstance(t.get("box"), list):
            b = _rel(box, t["box"])
            if ttype in BALLOON_TYPES:          # 固定した吹き出しより後の吹き出しは、その左か下へ
                cursor_x = min(cursor_x, b[0] - gap)
                row_top = max(row_top, b[1])
                row_bottom = max(row_bottom, b[3])
        elif ttype in BALLOON_TYPES:
            pad = cs * (1.4 if ttype not in ("narration", "monologue") else 0.7)
            bw = cols * cs * 1.25 + pad * 2
            bh_real = rows * cs * 1.05 + pad * 2
            bh = bh_real / aspect
            spk = char_by_id.get(t.get("speaker"))
            x2 = cursor_x
            if spk is not None and ttype not in ("narration", "monologue"):
                sc = (spk["bbox"][0] + spk["bbox"][2]) / 2
                x2 = min(cursor_x, sc + bw / 2)
            if x2 - bw < px1 + gap:            # 段を下げる（下げた段がコマに収まるときだけ。収まらなければ同じ段の左へ詰める。
                new_top = row_bottom + gap / aspect   # 下端で押し戻すと前の段に重なり、読む順が逆転する）
                if new_top + bh <= py2 - gap / aspect:
                    cursor_x = px2 - gap
                    row_top = new_top
                    x2 = cursor_x
            b = _clamp([x2 - bw, row_top, x2, row_top + bh], box)
            if b[1] < row_top:                  # 下にはみ出して押し上げられた: 前の段に重ねず、段の上端を保って縮める
                b = [b[0], row_top, b[2], max(min(row_top + bh, py2), row_top + 1)]
            cursor_x = b[0] - gap
            row_bottom = max(row_bottom, b[3])
        else:                                   # 擬音・作中文字
            size = cs * (2.6 if ttype == "sfx" else 1.0)
            n_ch = max(count_chars("".join(lines)), 1)
            at = str(t.get("at", {"sfx": "left", "aside": "bottom"}.get(ttype, "center")))
            w_ = size * 1.2 if ttype == "sfx" else min(n_ch * size, pw * 0.6)
            h_real = n_ch * size if ttype == "sfx" else size * 1.6
            h_ = min(h_real / aspect, ph * 0.6)
            cx = {"left": px1 + pw * 0.18, "right": px2 - pw * 0.18}.get(at, px1 + pw / 2)
            cy = {"top": py1 + ph * 0.25, "bottom": py2 - ph * 0.25}.get(at, py1 + ph * 0.5)
            b = _clamp([cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2], box)
        item = {"src": t, "bbox": b, "lines": lines, "char_size": cs}
        (out["balloons"] if ttype in BALLOON_TYPES else out["texts"]).append(item)

    # 漫符
    for o in out["characters"]:
        cx1, cy1, cx2, cy2 = o["bbox"]
        cw, chh = cx2 - cx1, cy2 - cy1
        head_h = chh * (0.35 if shot in ("full", "long", "extreme_long") else 0.45)
        for k, m in enumerate(o["src"].get("manpu", []) or []):
            d = manpu.get(m, {"place": "head", "key": m})
            place = d.get("place", "head")
            if place == "head":
                s = max(cw * 0.3, 14.0)
                bx = [cx2 - s * (0.4 + 0.6 * k), cy1 - s * 0.3 / aspect, cx2 + s * (0.6 - 0.6 * k), cy1 + s * 0.7 / aspect]
            elif place == "face":
                bx = [cx1 + cw * 0.2, cy1 + head_h * 0.25, cx2 - cw * 0.2, cy1 + head_h * 0.85]
            elif place == "background":
                bx = [cx1 - cw * 0.15, cy1 - chh * 0.05, cx2 + cw * 0.15, cy1 + chh * 0.6]
            elif place == "panel":
                bx = list(box)
            else:
                bx = [cx1 - cw * 0.08, cy1 + head_h, cx2 + cw * 0.08, cy2]
            out["marks"].append({"name": m, "def": d, "owner": o["src"].get("id"), "bbox": _clamp(bx, box)})

    # 効果
    for e in panel.get("effects", []) or []:
        d = manpu.get(e, {"place": "panel", "key": e})
        if d.get("place") in ("panel", "background") or not out["characters"]:
            bx = list(box)
        else:
            o = out["characters"][0]["bbox"]
            bx = [o[0] - pw * 0.1, o[1], o[2] + pw * 0.1, o[3]]
        out["effects"].append({"name": e, "def": d, "bbox": _clamp(bx, box)})
    return out


def page_aspect(doc, page, default=1.414):
    """ページの 縦/横 比. meta.page_aspect、縦読みは区切りの長さから."""
    fmt = name_format(doc)
    if fmt == "webtoon":
        return webtoon_aspect(page)
    return float(doc.get("meta", {}).get("page_aspect", default))
