#!/usr/bin/env python3
"""ネーム（name TOML）の機械検査.

構造の誤り（E）、表記（N）、数え上げ（C）、吹き出しと文字量（B）、コマ割りと演出（P）、漫符（M）、
人物の一貫性（X）を調べる。FAIL は構造の誤り・本文以外の混入・数え上げの不一致だけで、ほかは警報。
lint は「確認する箇所」を示す検知器で、修正命令ではない（SKILL.md「lint の読み方」）。

使い方:
  python name_lint.py name/ep001.toml [--stats] [--json] [--config lint.json] [--only EBPM]
終了コード: FAIL なし 0 / FAIL あり 1 / 入力エラー 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga_name as mn  # noqa: E402
from manga_name import Issue  # noqa: E402

KANJI_NUM = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
# 本文以外の混入: 括弧で囲んだ演出メモ、〔〕、TODO、「※」に続く演出・作者の注記（作中の注意書き「※関係者以外…」は含めない）
META_PAT = re.compile(r"（\s*(?:※|注[:：]|演出|ト書き|TODO|ここで|作者)|〔|\bTODO\b|※\s*(?:演出|ト書き|ここで|作者|メモ|注[:：])")
COUNT_PAT_A = re.compile(r"[「『]([^」』]{1,24})[」』]の([〇零一二三四五六七八九十百0-9０-９]+)(?:文字|字)")
COUNT_PAT_B = re.compile(r"([〇零一二三四五六七八九十百0-9０-９]+)(?:文字|字)の[「『]([^」』]{1,24})[」』]")
# 字数に数えない記号（長音符「ー」と波線は字として数える）
SKIP_CHARS = set("、。，．・…‥！？!?　 「」『』（）()")
LOW_BG = {None, "", "simple", "white", "tone", "none", "簡略", "白", "トーン", "なし"}
TALK_SHOTS = {"bust", "closeup", "extreme_closeup"}
WIDE_SHOTS = {"extreme_long", "long", "full"}


def parse_num(s):
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in KANJI_NUM:
            cur = KANJI_NUM[ch]
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch == "百":
            total += (cur or 1) * 100
            cur = 0
        else:
            return None
    return total + cur


def claimed_len(word):
    return sum(1 for ch in word if ch not in SKIP_CHARS and not ch.isspace())


def load_config(doc, path, extra):
    th_all = mn.load_json("thresholds.json")
    fmt = mn.name_format(doc)
    th = dict(th_all.get(fmt, th_all["page"]))
    opts = {"disable": set(), "balloon_period": False, "rating": doc.get("meta", {}).get("rating", "general")}
    mt = mn.find_manga_toml(path)
    sources = []
    if mt:
        try:
            with open(mt, "rb") as f:
                m = tomllib.load(f)
        except tomllib.TOMLDecodeError:
            m = {}
        c = m.get("contract", {})
        if c.get("manpu_level") == "low":
            th["manpu_per_character"] = 1
            th["manpu_same_run"] = max(2, th["manpu_same_run"] - 1)
        elif c.get("manpu_level") == "high":
            th["manpu_per_character"] += 1
            th["manpu_same_run"] += 2
        if c.get("narration") == "high":
            th["narration_run"] += 3
        sources.append(m.get("lint", {}))
    if extra:
        with open(extra, encoding="utf-8") as f:
            sources.append(json.load(f))
    for s in sources:
        for k, v in s.items():
            if k == "disable":
                opts["disable"] |= set(v)
            elif k in opts:
                opts[k] = v
            elif k in th:
                th[k] = v
    return th, opts


def lint(doc, th, opts):
    issues: list[Issue] = []
    add = issues.append
    fmt = mn.name_format(doc)
    cast = doc["_cast"]
    manpu = mn.manpu_index()

    if fmt not in mn.FORMATS:
        add(Issue("E01", "FAIL", "meta", f"format={fmt!r} は page / yonkoma / webtoon のどれか"))
        fmt = "page"
    pages = doc.get("pages", [])
    if not pages:
        add(Issue("E02", "FAIL", "全体", "ページがない（[[pages]] が空）"))
        return issues, {}

    struct, broken = mn.validate_structure(doc)
    issues.extend(struct)
    if "*" in broken:
        return issues, {}

    for cid, c in cast.items():
        if not str(c.get("appearance", "")).strip():
            add(Issue("X02", "WARN", f"人物 {cid}", "appearance（外見の固定記述）が空。画像生成で人物の見た目が揺れる"))

    stats = {"pages": [], "panels": 0, "balloons": 0, "chars": 0}
    seen_ids = set()
    prev_scene = None
    scene_panels = []           # (panel, scene)
    shot_run = []
    talk_run = []
    narr_run = 0
    last_outfit = {}            # (scene, char) -> outfit
    manpu_run = {}              # (人物, 記号の正規キー) -> 連続したコマ数
    page_numbers = [pg.get("number", i + 1) for i, pg in enumerate(pages)]
    # 見開き: spread = true の右ページと、その次のページ（左）が一組
    spread_pages = set()
    for p_i, page in enumerate(pages):
        if page.get("spread") and p_i not in spread_pages:
            spread_pages |= {p_i, p_i + 1}

    for p_i, page in enumerate(pages):
        number = page_numbers[p_i]
        pw = f"P{number}"
        panels = page.get("panels", [])
        if p_i in broken:
            continue
        if not panels:
            add(Issue("E02", "FAIL", pw, "コマがない"))
            continue
        if page.get("order") not in (None, "written"):
            add(Issue("P13", "WARN", pw, f"order={page.get('order')!r} は語彙にない（written だけ。書いた順を読む順にする）"))
        mn.layout_page(page, fmt, issues, pw)
        pstat = {"page": number, "side": mn.page_side(doc, number) if fmt == "page" else "-",
                 "panels": len(panels), "balloons": 0, "chars": 0, "shots": {}}

        if fmt == "yonkoma" and len(panels) % 4:
            add(Issue("E09", "FAIL", pw, f"4コマのページのコマ数が 4 の倍数でない（{len(panels)}）"))
        elif fmt != "yonkoma" and not (th["panels_min"] <= len(panels) <= th["panels_max"]) and not page.get("splash"):
            add(Issue("P01", "WARN", pw, f"コマ数 {len(panels)}（目安 {th['panels_min']}〜{th['panels_max']}）。"
                      "大ゴマ 1 枚の扉や見せ場なら splash = true"))

        # 配置の結果と読み順
        for p in panels:
            b = p.get("_bbox")
            if b is None:
                add(Issue("E08", "FAIL", f"{pw} {p.get('id', '?')}", "配置を計算できない（tier か bbox が要る）"))
            elif not (b[2] - b[0] >= mn.MIN_PANEL and b[3] - b[1] >= mn.MIN_PANEL and min(b) >= 0 and max(b) <= 1000):
                add(Issue("E08", "FAIL", f"{pw} {p.get('id', '?')}", f"配置の結果が壊れた・小さすぎる矩形になった: {b}（tiers・width・height を見直す）"))
                p.pop("_bbox", None)
        geo = mn.geometric_order(panels, fmt)
        if [id(p) for p in geo] != [id(p) for p in panels if "_bbox" in p]:
            names = " → ".join(p.get("id", "?") for p in geo)
            add(Issue("P02", "WARN", pw, f"書いた順と、配置から読む順が違う。配置どおりなら {names}"))
        if fmt != "webtoon":
            boxes = [(p.get("id", "?"), p["_bbox"], p.get("inset")) for p in panels if "_bbox" in p]
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i][1], boxes[j][1]
                    ov = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
                    if ov / small > 0.15 and not (boxes[i][2] or boxes[j][2]):
                        add(Issue("P10", "WARN", pw, f"{boxes[i][0]} と {boxes[j][0]} が重なっている。はめ込みなら inset = true"))

        # めくりと見開き
        if fmt == "page":
            side = mn.page_side(doc, number)
            if side == "left" and p_i < len(pages) - 1 and not page.get("turn"):
                add(Issue("P07", "INFO", pw, "左ページ（次はめくり）。最終コマで何を引き、めくった右ページの 1 コマ目で何を見せるかを turn に書く"))
            starts_spread = page.get("spread") and not (p_i > 0 and pages[p_i - 1].get("spread") and p_i in spread_pages
                                                        and p_i - 1 in spread_pages and mn.page_side(doc, page_numbers[p_i - 1]) == "right")
            if starts_spread and side != "right":
                add(Issue("P08", "WARN", pw, "見開きは右ページから始める（このページは左）。ページ数か first_page_side を見直す"))

        page_chars = 0
        page_shouts = 0
        nodo = {"right": "left", "left": "right"}.get(mn.page_side(doc, number)) if fmt == "page" else None
        for k, p in enumerate(panels):
            pid = p.get("id") or f"{pw}-{k + 1}"
            where = f"{pw} {pid}"
            if pid in seen_ids:
                add(Issue("E03", "FAIL", where, "コマの id が重複"))
            seen_ids.add(pid)
            stats["panels"] += 1
            b = p.get("_bbox")
            if nodo and b and p_i not in spread_pages and ((nodo == "left" and b[0] <= 0.5) or (nodo == "right" and b[2] >= 999.5)):
                add(Issue("P12", "WARN", where, f"ノド側（綴じる側＝{'左' if nodo == 'left' else '右'}）へのタチキリ。"
                          "綴じに絵が食われる。見開きでなければノド以外の辺に"))
            shot = mn.normalize_shot(p.get("shot"))
            angle = mn.normalize_angle(p.get("angle"))
            if shot and shot not in mn.SHOTS:
                add(Issue("P13", "WARN", where, f"shot={shot!r} は語彙にない（{', '.join(mn.SHOTS)}）"))
            if angle and angle not in mn.ANGLES:
                add(Issue("P13", "WARN", where, f"angle={angle!r} は語彙にない（{', '.join(mn.ANGLES)}）"))
            detail = mn.normalize_detail(p.get("detail"))
            if detail and detail not in mn.DETAILS:
                add(Issue("P13", "WARN", where, f"detail={p.get('detail')!r} は語彙にない（{', '.join(mn.DETAILS)}）"))
            if p.get("transition") and p["transition"] not in mn.TRANSITIONS:
                add(Issue("P13", "WARN", where, f"transition={p['transition']!r} は語彙にない（{', '.join(mn.TRANSITIONS)}）"))
            pstat["shots"][shot or "-"] = pstat["shots"].get(shot or "-", 0) + 1
            chars = p.get("characters", []) or []
            texts = p.get("text", []) or []
            present = {c.get("id") for c in chars}

            if not chars and not texts and not p.get("visual") and not p.get("beat"):
                add(Issue("P09", "WARN", where, "空のコマ（人物・文字・visual・beat がどれも無い）"))
            elif not p.get("visual") and not p.get("beat"):
                add(Issue("P06", "WARN", where, "visual / beat が無い。絵で何を見せるコマかが決まっていない"))

            # 場面の頭
            scene = p.get("scene", prev_scene if prev_scene is not None else "_s1")
            if scene != prev_scene:
                if scene_panels:
                    _check_establish(scene_panels, add)
                scene_panels = []
                if not p.get("place"):
                    add(Issue("P03", "WARN", where, f"場面 {scene} の最初のコマに place（場所）が無い"))
                prev_scene = scene
            scene_panels.append((where, shot, scene))

            # 同じショットの連続
            shot_run = shot_run + [shot] if shot and shot_run and shot_run[-1] == shot else ([shot] if shot else [])
            if shot and len(shot_run) == th["same_shot_run"]:
                add(Issue("P04", "WARN", where, f"{mn.SHOTS.get(shot, (shot,))[0]}が {len(shot_run)} コマ続いた"))
            has_dialogue = any(t.get("type", "dialogue") in {"dialogue", "shout", "whisper"} for t in texts)
            if shot in TALK_SHOTS and has_dialogue and p.get("background") in LOW_BG and not p.get("action"):
                talk_run.append(where)
                if len(talk_run) == th["talking_head_run"]:
                    add(Issue("P05", "WARN", f"{talk_run[0]}〜{pid}",
                              f"顔の寄りと台詞だけのコマが {len(talk_run)} 続いた（背景が簡略で action なし）"))
            else:
                talk_run = []

            # 人物（このコマにいない人物・付いていない記号は、漫符の連続を切る）
            now = {(c.get("id"), (manpu.get(m) or {}).get("key", m)) for c in chars for m in (c.get("manpu") or [])}
            for key in [k for k in manpu_run if k not in now]:
                manpu_run.pop(key)
            for c in chars:
                cid = c.get("id")
                if cid not in cast:
                    add(Issue("E04", "FAIL", where, f"人物 {cid!r} が characters に無い"))
                outfit = c.get("outfit")
                key = (scene, cid)
                if outfit:
                    if key in last_outfit and last_outfit[key] != outfit and not c.get("outfit_change"):
                        add(Issue("X01", "WARN", where, f"{cid} の服装が同じ場面の中で変わった（{last_outfit[key]} → {outfit}）。"
                                  "着替えなら outfit_change = true"))
                    last_outfit[key] = outfit
                marks = c.get("manpu", []) or []
                if len(marks) > th["manpu_per_character"]:
                    add(Issue("M02", "WARN", where, f"{cid} に漫符が {len(marks)} 個。感情の焦点が散る"))
                for m in marks:
                    d = manpu.get(m)
                    if d is None:
                        add(Issue("M01", "WARN", where, f"漫符 {m!r} は語彙表に無い（scripts/data/manpu.json）"))
                    elif d.get("rating") == "adult" and opts["rating"] not in ("adult", "r18"):
                        add(Issue("M04", "WARN", where, f"{m!r} は成人向けの定型。作品の rating が {opts['rating']}"))
                for key in dict.fromkeys((manpu.get(m) or {}).get("key", m) for m in marks):
                    cnt = manpu_run.get((cid, key), 0) + 1
                    manpu_run[(cid, key)] = cnt
                    if cnt == th["manpu_same_run"]:
                        add(Issue("M03", "WARN", where, f"{cid} の {key} が {cnt} コマ続いた。記号が感情の変化を運ばなくなる"))
            for e in p.get("effects", []) or []:
                d = manpu.get(e)
                if d is None:
                    add(Issue("M01", "WARN", where, f"効果 {e!r} は語彙表に無い（scripts/data/manpu.json）"))
                elif d.get("rating") == "adult" and opts["rating"] not in ("adult", "r18"):
                    add(Issue("M04", "WARN", where, f"{e!r} は成人向けの定型。作品の rating が {opts['rating']}"))

            # 文字
            balloons = [t for t in texts if t.get("type", "dialogue") in mn.BALLOON_TYPES]
            if len(balloons) > th["balloons_per_panel"]:
                add(Issue("B04", "WARN", where, f"吹き出しが {len(balloons)} 個（目安 {th['balloons_per_panel']} まで）"))
            narr = any(t.get("type") in {"narration", "monologue"} for t in texts)
            narr_run = narr_run + 1 if narr else 0
            if narr_run == th["narration_run"]:
                add(Issue("B08", "WARN", where, f"ナレーション・モノローグが {narr_run} コマ続いた。絵で見せられないか"))
            speakers_order = []
            for t_i, t in enumerate(texts):
                tw = f"{where} 文字{t_i + 1}"
                ttype = t.get("type", "dialogue")
                if ttype not in mn.TEXT_TYPES:
                    continue                      # E06 は構造検査（validate_structure）で出している
                lines = mn.text_body(t)
                body = "".join(lines)
                if not body.strip():
                    add(Issue("E05", "FAIL", tw, "文字が空"))
                    continue
                n = mn.count_chars(body)
                if ttype in mn.BALLOON_TYPES:
                    stats["balloons"] += 1
                    pstat["balloons"] += 1
                if ttype == "shout":
                    page_shouts += 1
                if ttype != "sfx":
                    page_chars += n
                spk = t.get("speaker")
                if spk is not None and spk not in cast:
                    add(Issue("E04", "FAIL", tw, f"話者 {spk!r} が characters に無い"))
                if ttype in {"dialogue", "shout", "whisper", "thought"}:
                    if spk is None and len(chars) >= 2:
                        add(Issue("B09", "WARN", tw, "話者が無い。人物が 2 人以上いるコマでは speaker を書く"))
                    if spk is not None and spk not in present and not t.get("offscreen"):
                        add(Issue("B07", "WARN", tw, f"話者 {spk} がこのコマに描かれていない。画面外の声なら offscreen = true"))
                    if spk is not None:
                        speakers_order.append(spk)
                if ttype in mn.BALLOON_TYPES and ttype not in {"narration", "monologue"}:
                    if n > th["balloon_chars"]:
                        add(Issue("B01", "WARN", tw, f"1 つの吹き出しに {n} 字（目安 {th['balloon_chars']} 字まで）。割るか削る"))
                    if len(lines) > th["balloon_lines"]:
                        add(Issue("B03", "WARN", tw, f"{len(lines)} 行（目安 {th['balloon_lines']} 行まで）"))
                    if len(lines) > 1 or isinstance(t.get("lines"), list):
                        for ln in lines:
                            if mn.count_chars(ln) > th["line_chars"]:
                                add(Issue("B02", "WARN", tw, f"1 行 {mn.count_chars(ln)} 字（目安 {th['line_chars']} 字まで）: {ln}"))
                _notation(body, tw, ttype, opts, add)
            # 吹き出しの順と人物の左右
            if len(speakers_order) >= 2:
                pos = {c.get("id"): mn.POSITIONS.get(str(c.get("pos", "")), None) for c in chars}
                a, b = speakers_order[0], next((s for s in speakers_order[1:] if s != speakers_order[0]), None)
                if b and pos.get(a) is not None and pos.get(b) is not None and pos[a] > pos[b]:
                    add(Issue("B06", "WARN", where, f"先に話す {a} が {b} より左にいる。右の人物から話すと、吹き出しの順と視線が揃う"))
            # 配置した吹き出しの順（box で固定した吹き出しが混じると逆転しうる）
            if len(balloons) >= 2 and "_bbox" in p:
                placed = mn.place_panel(p, mn.page_aspect(doc, page), manpu)["balloons"]
                order = balloon_reading_order(placed)
                if order != list(range(len(placed))):
                    first_wrong = next(i for i, k in enumerate(order) if k != i)
                    y = placed[order[first_wrong]]
                    add(Issue("B12", "WARN", where, f"配置した吹き出しを読む順（右→左、上→下）が、書いた順と違う。"
                              f"先に読まれるのは「{''.join(y['lines'])[:10]}」。box を見直す"))
        if fmt == "page":
            _check_capacity(panels, pw, add)
        if fmt == "page" and not page.get("splash"):
            _check_density(panels, pw, th, add)
        if pstat["balloons"] > th["page_balloons"]:
            add(Issue("B10", "WARN", pw, f"ページの吹き出しが {pstat['balloons']} 個（目安 {th['page_balloons']} 個まで）。文字が絵より主役になる"))
        if page_shouts > th["shout_per_page"]:
            add(Issue("B11", "WARN", pw, f"叫びの吹き出しが {page_shouts} 個（目安 {th['shout_per_page']} 個まで）。強調が強調でなくなる"))
        pstat["chars"] = page_chars
        stats["chars"] += page_chars
        if page_chars > th["page_chars"]:
            add(Issue("B05", "WARN", pw, f"ページの文字が {page_chars} 字（目安 {th['page_chars']} 字まで）"))
        stats["pages"].append(pstat)
    if scene_panels:
        _check_establish(scene_panels, add)
    issues = [i for i in issues if i.code not in opts["disable"]]
    return issues, stats


def balloon_reading_order(placed):
    """配置した吹き出しの読む順（右綴じ: 上下に重なる帯ごとに、右から左。帯は上から）. 戻り値は placed の添字の並び."""
    rows = []
    for i in sorted(range(len(placed)), key=lambda i: placed[i]["bbox"][1]):
        y1, y2 = placed[i]["bbox"][1], placed[i]["bbox"][3]
        for row in rows:
            if min(y2, row["y"][1]) - max(y1, row["y"][0]) > 0:
                row["items"].append(i)
                row["y"] = (min(row["y"][0], y1), max(row["y"][1], y2))
                break
        else:
            rows.append({"y": (y1, y2), "items": [i]})
    out = []
    for row in sorted(rows, key=lambda r: r["y"][0]):
        out += sorted(row["items"], key=lambda i: -placed[i]["bbox"][2])
    return out


# コマの大きさ（ページに対する面積）ごとの、台詞の字数の上位 1 割（Manga109-s 全 87 冊、corpus-norms.md §7）
CAPACITY = [(0.04, 15), (0.08, 33), (0.15, 44), (9.0, 54)]


def _check_capacity(panels, pw, add):
    """B13: コマの大きさに対して台詞が多い（実際の漫画の上位 1 割を超える）."""
    for k, p in enumerate(panels):
        b = p.get("_bbox")
        if not b:
            continue
        share = (b[2] - b[0]) * (b[3] - b[1]) / 1e6
        n = sum(mn.count_chars("".join(mn.text_body(t))) for t in (p.get("text") or [])
                if t.get("type", "dialogue") in mn.BALLOON_TYPES)
        i = next(i for i, (s, _) in enumerate(CAPACITY) if share < s)
        cap = CAPACITY[i][1]
        lo = CAPACITY[i - 1][0] if i else 0.0
        band = f"{CAPACITY[0][0]:.0%} 未満" if i == 0 else (f"{lo:.0%} 以上" if i == len(CAPACITY) - 1 else f"{lo:.0%} 以上 {CAPACITY[i][0]:.0%} 未満")
        if n > cap:
            add(Issue("B13", "INFO", f"{pw} {p.get('id') or k + 1}",
                      f"コマの大きさ（ページの {share * 100:.2f}%、区分「{band}」）に対して台詞が {n} 字。"
                      f"実際の漫画では、この区分のコマの 9 割が {cap} 字以下。コマを大きくするか、台詞を次のコマへ分ける"))


def _check_density(panels, pw, th, add):
    """描き込みと大きさの強弱（P14 突出した大ゴマが無い＝最大のコマが中くらいのコマの standout_ratio 倍未満、P15 描き込みの配分）。"""
    boxes = [p["_bbox"] for p in panels if p.get("_bbox") and not p.get("inset")]
    if len(boxes) >= 4:
        areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        mid = areas[len(areas) // 2] if len(areas) % 2 else (areas[len(areas) // 2 - 1] + areas[len(areas) // 2]) / 2
        ratio = areas[-1] / (mid or 1)
        if ratio < th.get("standout_ratio", 1.3):
            add(Issue("P14", "INFO", pw, f"突出した大ゴマが無い（いちばん大きいコマが、中くらいのコマの {ratio:.2f} 倍）。"
                      "実際の漫画では中央値 2.2 倍、1.3 倍未満は 1 割に満たない。めくり前の溜めや緊張の会話劇でなければ、主役のコマを大きく"))
    details = [mn.normalize_detail(p.get("detail")) for p in panels]
    keys = details.count("key")
    if keys > 1:
        add(Issue("P15", "INFO", pw, f"決めコマ（detail = \"key\"）が {keys} つ。1 ページに 1 つまでにすると、決めが決めになる"))
    rich = sum(1 for p, d in zip(panels, details) if p.get("background") in ("detailed", "描き込み") and d != "key")
    if len(panels) >= 4 and rich > len(panels) * th.get("detailed_bg_share", 0.5):
        add(Issue("P15", "INFO", pw, f"{len(panels)} コマ中 {rich} コマが背景 detailed（決めコマ以外）。"
                  "背景は場面の頭と決めコマで描き、ほかは simple・white・tone へ。全コマを一枚絵のように描くと漫画に見えない"))


def _check_establish(scene_panels, add):
    first3 = [s for _, s, _ in scene_panels[:3]]
    if first3 and not any(s in WIDE_SHOTS for s in first3) and any(s for s in first3):
        where, _, scene = scene_panels[0]
        add(Issue("P11", "INFO", where, f"場面 {scene} の最初の 3 コマに引き（ロング・フル）が無い。場所を伏せる意図なら keep"))


def _notation(body, where, ttype, opts, add):
    if (ttype in ("sign", "aside") and re.search(r"\bTODO\b|（\s*(?:演出|ト書き|ここで|作者)", body)) or \
            (ttype not in ("sign", "aside") and META_PAT.search(body)):
        add(Issue("N05", "FAIL", where, f"本文以外（注記・ト書き）が文字に混じっている: {body[:30]}"))
    if re.search(r"(?<![!?])[!?](?![!?])", body) or re.search(r"[!?]{2,}", body):
        add(Issue("N01", "WARN", where, "半角の ! ? がある。写植は全角（！？・!?合字）が普通"))
    if re.search(r"\.\.\.|・・・|(?<!…)…(?!…)", body):
        add(Issue("N02", "WARN", where, "三点リーダは「……」（2 つ続き）が普通"))
    if "~" in body:
        add(Issue("N04", "WARN", where, "半角チルダ ~ は「〜」に"))
    if ttype in mn.BALLOON_TYPES and ttype not in {"narration", "monologue"} and not opts["balloon_period"]:
        if body.rstrip().endswith("。"):
            add(Issue("N03", "INFO", where, "吹き出しの末尾に句点。漫画の写植では省くのが一般的（作品で付けるなら [lint] balloon_period = true）"))
    if re.search(r"[A-Za-z]{4,}", body):
        add(Issue("N06", "INFO", where, "英字の連なりがある。意図した表記か確かめる"))
    for m in COUNT_PAT_A.finditer(body):
        _count(m.group(1), m.group(2), where, add)
    for m in COUNT_PAT_B.finditer(body):
        _count(m.group(2), m.group(1), where, add)


def _count(word, num, where, add):
    claim = parse_num(num)
    real = claimed_len(word)
    if claim is not None and claim != real:
        add(Issue("C01", "FAIL", where, f"「{word}」は {real} 字なのに {num} 字と書いている"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="ネームの TOML")
    ap.add_argument("--stats", action="store_true", help="ページごとの統計を出す")
    ap.add_argument("--json", action="store_true", help="JSON で出す")
    ap.add_argument("--config", help="閾値の上書き（JSON）")
    ap.add_argument("--only", help="この群の頭文字だけ出す（例: EBP）")
    a = ap.parse_args(argv)
    try:
        doc = mn.load_name(a.name)
        th, opts = load_config(doc, a.name, a.config)
        issues, stats = lint(doc, th, opts)
    except mn.NameError_ as e:
        mn.eprint(f"エラー: {e}")
        return 2
    if a.only:
        issues = [i for i in issues if i.code[0] in a.only.upper()]
    fails = sum(1 for i in issues if i.level == "FAIL")
    if a.json:
        print(json.dumps({"stats": stats, "issues": [i.as_dict() for i in issues], "fail": fails,
                          "thresholds": th, "calibrated": False}, ensure_ascii=False, indent=2))
        return 1 if fails else 0
    print(f"ページ {len(stats.get('pages', []))} / コマ {stats.get('panels', 0)} / 吹き出し {stats.get('balloons', 0)}"
          f" / 文字 {stats.get('chars', 0)} 字（擬音を除く） / FAIL {fails} / WARN {sum(1 for i in issues if i.level == 'WARN')}"
          f" / INFO {sum(1 for i in issues if i.level == 'INFO')}")
    if a.stats:
        for ps in stats.get("pages", []):
            shots = " ".join(f"{mn.SHOTS.get(k, (k,))[0]}{v}" for k, v in ps["shots"].items())
            print(f"  P{ps['page']}（{ps['side']}）コマ{ps['panels']} 吹き出し{ps['balloons']} 文字{ps['chars']}  {shots}")
    for i in issues:
        print(f"{i.level:4} {i.code} {i.where}: {i.message}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
