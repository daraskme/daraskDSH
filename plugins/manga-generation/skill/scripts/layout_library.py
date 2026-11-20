#!/usr/bin/env python3
"""コマ割りの型の保存庫（scripts/data/layouts.json）を引く・見る・足す.

型は 3 種類ある。
  common  実際の漫画でよく使われる型（manga lora の注釈 6,681 ページから、同じ形のページを束ねて寸法の中央値を取った）
  rare    出現は少ないが、狙いのはっきりした型（同じ注釈から、コマ枠の形だけを見て選んだ）
  user    作者が足した型（自分の原稿、気に入ったネーム）

どの型も、コマ枠の座標（コマを並べる範囲を 0〜1000 にした [x1, y1, x2, y2]、読む順）だけを持つ。
元のページの絵・台詞・作品名は持たない。

使い方:
  python layout_library.py list [--kind common|rare|user] [--panels N] [--scene talk|action|gag|emotion|tense]
  python layout_library.py show ID              # 説明と、ネーム TOML の [[pages]] の雛形
  python layout_library.py preview OUT.html [ID ...]   # 型をネーム画像（見開き）で並べる
  python layout_library.py add ID --name 名前 --boxes "[[x1,y1,x2,y2], ...]" [--why 説明] [--scenes talk,emotion]
                                                # user の型を足す（boxes は読む順）

雛形は、段（tiers）・幅（width）・縦に積む組（column / height）で書けるときはそれで、
書けないとき（重なり・はめ込み・斜め）は bbox で出す。人物や台詞は入っていないので、コマごとに足す。
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import manga_name as mn

DATA = Path(__file__).resolve().parent / "data" / "layouts.json"
SCENES = {"talk": "会話・説明", "action": "動き・戦い・スポーツ", "gag": "ギャグ", "emotion": "感情・恋愛・静けさ", "tense": "恐怖・緊張"}
TOL = 12  # 0〜1000 の座標で、コマが切れ目をまたいでよい量（注釈の揺れ）
FIT = 40  # 段組みに直すとき、コマの端が段・列の端からずれてよい量。超えるずれは bbox で残す
GAP = 60  # 段組みに直すとき、コマの間の空きとして許す量（ふつうのガターは 13〜18）。超える空白は bbox で残す


def load():
    return json.loads(DATA.read_text(encoding="utf-8"))


def save(d):
    DATA.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---- boxes -> 段組み ----
def _groups(idx, boxes, axis):
    a0, a1 = (0, 2) if axis == 0 else (1, 3)
    idx = sorted(idx, key=lambda i: boxes[i][a0])
    out, cur, end = [], [], None
    for i in idx:
        if cur and boxes[i][a0] >= end - TOL:
            out.append(cur)
            cur, end = [], None
        cur.append(i)
        end = boxes[i][a1] if end is None else max(end, boxes[i][a1])
    if cur:
        out.append(cur)
    return out[::-1] if axis == 0 else out  # 列は右から


def _extent(idx, boxes, axis):
    a0, a1 = (0, 2) if axis == 0 else (1, 3)
    return min(boxes[i][a0] for i in idx), max(boxes[i][a1] for i in idx)


def to_tiers(boxes):
    """段 → 列 → 縦に積む組 の 3 層で書ければ [(段の高さ, [(列の幅, [(コマ, 高さ)...])...])...]。書けなければ None."""
    rows = _groups(range(len(boxes)), boxes, 1)
    # 段組みは基本枠いっぱいに、ふつうのガターで並べる。外側の余白やコマの間の大きな空白は表せない
    xs1, ys1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    xs2, ys2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if xs1 > FIT or ys1 > FIT or xs2 < 1000 - FIT or ys2 < 1000 - FIT:
        return None
    for a, b in zip(rows, rows[1:]):
        if _extent(b, boxes, 1)[0] - _extent(a, boxes, 1)[1] > GAP:
            return None
    out, order = [], []
    for r in rows:
        y1, y2 = _extent(r, boxes, 1)
        cols = []
        for c in _groups(r, boxes, 0):
            stack = _groups(c, boxes, 1)
            if any(len(s) > 1 for s in stack):
                return None
            x1, x2 = _extent(c, boxes, 0)
            # 段組みでは、列は段の上下いっぱい、縦積みのコマは列の左右いっぱいに伸びる。元の位置から大きくずれるなら書けない
            if abs(boxes[stack[0][0]][1] - y1) > FIT or abs(boxes[stack[-1][0]][3] - y2) > FIT:
                return None
            if any(abs(boxes[s[0]][0] - x1) > FIT or abs(boxes[s[0]][2] - x2) > FIT for s in stack):
                return None
            for s1, s2 in zip(stack, stack[1:]):
                if boxes[s2[0]][1] - boxes[s1[0]][3] > GAP:
                    return None
            cols.append((x2 - x1, [(s[0], boxes[s[0]][3] - boxes[s[0]][1]) for s in stack]))
            order += [s[0] for s in stack]
        cg = _groups(r, boxes, 0)
        for c1, c2 in zip(cg, cg[1:]):           # 右の列の左端と、左の列の右端の間
            if _extent(c1, boxes, 0)[0] - _extent(c2, boxes, 0)[1] > GAP:
                return None
        out.append((y2 - y1, cols))
    if order != list(range(len(boxes))):
        return None  # 段組みで読む順と、型の読む順が違う（4コマの縦 2 本など）
    return out


def _role(lay, k):
    """k 番目（0 始まり）のコマの役割の目安を、雛形のコメントにする."""
    sl = (lay.get("slots") or [None] * len(lay["boxes"]))[k]
    if not sl:
        return []
    why = ("このコマで全体より多い shot（いちばん多いとは限らない）" if "全体より多い（" in sl["role"].split("、")[0]
           else "このコマでいちばん多い shot")
    return [f"  # 役割の目安: {sl['role']}",
            f"  # shot = \"{sl['shot']}\"   # 目安: {why}。場面に合わせて決める"]


def page_toml(lay, number=1):
    boxes = lay["boxes"]
    lines = [f"[[pages]]", f"number = {number}", f"# 型 {lay['id']}: {lay['name']}"]
    if len(boxes) == 1:
        lines += ["splash = true", "", "  [[pages.panels]]", "  tier = 0"] + _role(lay, 0) + ['  beat = ""']
        return "\n".join(lines) + "\n"
    t = to_tiers(boxes)
    if t:
        tot = sum(h for h, _ in t)
        lines.append("tiers = [" + ", ".join(f"{h / tot:.2f}" for h, _ in t) + "]")
        k = 0
        for ti, (_, cols) in enumerate(t):
            wt = sum(w for w, _ in cols)
            for ci, (w, stack) in enumerate(cols):
                ht = sum(h for _, h in stack)
                for si, (_, h) in enumerate(stack):
                    k += 1
                    lines += ["", f"  [[pages.panels]]   # {k}", f"  tier = {ti}"]
                    if len(cols) > 1 and si == 0:
                        lines.append(f"  width = {w / wt:.2f}")
                    if len(stack) > 1:
                        lines += [f"  column = {ci + 1}", f"  height = {h / ht:.2f}"]
                    lines += _role(lay, k - 1)
                    lines.append('  beat = ""')
        return "\n".join(lines) + "\n"
    # 重なり・はめ込み・斜め: bbox（基本枠の中に写す）
    fx1, fy1, fx2, fy2 = mn._frame({}, "page")
    lines.append("# bbox で組む型（重なり・はめ込み・段からずれた位置）。order = \"written\" で、書いた順を読む順として書き出しまで保つ")
    lines.append('order = "written"')

    def ar(i):
        return (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])

    def overlaps(i, j):
        a, b = boxes[i], boxes[j]
        o = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return o > 0.15 * min(ar(i), ar(j))
    for k, b in enumerate(boxes, 1):
        bb = [round(fx1 + b[0] * (fx2 - fx1) / 1000), round(fy1 + b[1] * (fy2 - fy1) / 1000),
              round(fx1 + b[2] * (fx2 - fx1) / 1000), round(fy1 + b[3] * (fy2 - fy1) / 1000)]
        lines += ["", f"  [[pages.panels]]   # {k}", f"  bbox = {bb}"]
        # 重なる相手より小さいコマは、はめ込み（上に重ねる）として書く
        if any(j != k - 1 and overlaps(k - 1, j) and ar(k - 1) < ar(j) for j in range(len(boxes))):
            lines.append("  inset = true      # 重ねる（はめ込み・枠越え）")
        lines += _role(lay, k - 1)
        lines.append('  beat = ""')
    return "\n".join(lines) + "\n"


def parse_boxes(text):
    """--boxes の JSON を検査する。正しければ数のリストのリスト、だめなら None."""
    import math
    try:
        boxes = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(boxes, list) or not boxes:
        return None
    out = []
    for b in boxes:
        if not isinstance(b, list) or len(b) != 4:
            return None
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in b):
            return None
        if not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000):
            return None
        r = [round(v) for v in b]
        if not (r[0] < r[2] and r[1] < r[3]):    # 丸めて幅・高さが 0 になる枠は表せない
            return None
        out.append(r)
    return out


def describe(lay, full=True):
    ev = lay.get("evidence") or {}
    head = f"{lay['id']}  {lay['name']}  （{lay['kind']}、{len(lay['boxes'])} コマ"
    if ev.get("pages"):
        head += f"、実例 {ev['pages']} ページ・{ev['books']} 冊、根拠: {ev['strength']}"
    elif ev:
        head += "、実例ほぼなし"
    out = [head + "）"]
    if lay.get("purpose"):
        out.append("  目的: " + lay["purpose"])
    basis = lay.get("scene_basis") or {s: "" for s in lay.get("scenes", [])}
    if basis:
        out.append("  場面: " + "、".join(f"{SCENES.get(s, s)}（{b}）" if b else SCENES.get(s, s) for s, b in basis.items()))
    if not full:
        return "\n".join(out)
    if lay.get("when"):
        out.append("  使う場面（実際の漫画での使われ方）:")
        out += [f"    - {w}" for w in lay["when"]]
    if basis and all(b in ("構図から", "目的から", "") for b in basis.values()) and not lay.get("when"):
        out.append("  （場面は構図・目的から考えた候補で、実例の数字の裏付けは無い）")
    if lay.get("slots"):
        out.append("  コマごとの中身（読む順）:")
        out += [f"    {i + 1}. {sl['role']}" for i, sl in enumerate(lay["slots"])]
    if lay.get("avoid"):
        out.append("  避けるとき: " + lay["avoid"])
    if ev.get("pages", 0) >= 10 and ev.get("top_book_share", 0) >= 0.3:
        out.append(f"  注意: 実例の {round(ev['top_book_share'] * 100)}% が 1 冊の作品。その作品の癖の可能性がある")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_list = sub.add_parser("list")
    a_list.add_argument("--kind")
    a_list.add_argument("--panels", type=int)
    a_list.add_argument("--scene")
    a_show = sub.add_parser("show")
    a_show.add_argument("id")
    a_prev = sub.add_parser("preview")
    a_prev.add_argument("out")
    a_prev.add_argument("ids", nargs="*")
    a_add = sub.add_parser("add")
    a_add.add_argument("id")
    a_add.add_argument("--name", required=True)
    a_add.add_argument("--boxes", required=True)
    a_add.add_argument("--why", default="")
    a_add.add_argument("--scenes", default="")
    a = ap.parse_args(argv)
    d = load()
    by_id = {x["id"]: x for x in d["layouts"]}

    if a.cmd == "list":
        for x in d["layouts"]:
            if a.kind and x["kind"] != a.kind:
                continue
            if a.panels and len(x["boxes"]) != a.panels:
                continue
            if a.scene and a.scene not in x.get("scenes", []):
                continue
            print(describe(x, full=False))
        return 0
    if a.cmd == "show":
        x = by_id.get(a.id)
        if not x:
            mn.eprint(f"型 {a.id} が無い（list で一覧）")
            return 2
        print(describe(x))
        print()
        print(page_toml(x))
        return 0
    if a.cmd == "preview":
        ids = a.ids or [x["id"] for x in d["layouts"]]
        missing = [i for i in ids if i not in by_id]
        if missing:
            mn.eprint(f"型が無い: {', '.join(missing)}")
            return 2
        body = ['[meta]', 'title = "コマ割りの型"', 'format = "page"', 'first_page_side = "right"', ""]
        for n, i in enumerate(ids, 1):
            body.append(page_toml(by_id[i], n).replace('beat = ""', f'beat = "{i}"'))
        import name_render
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "layouts.toml"
            p.write_text("\n".join(body), encoding="utf-8")
            return name_render.main([str(p), "--html", a.out])
    if a.cmd == "add":
        if a.id in by_id:
            mn.eprint(f"型 {a.id} は既にある")
            return 2
        boxes = parse_boxes(a.boxes)
        if boxes is None:
            mn.eprint("boxes は [[x1,y1,x2,y2], ...]（0〜1000 の数、x1<x2、y1<y2）を読む順に")
            return 2
        if not a.id.strip():
            mn.eprint("ID が空")
            return 2
        d["layouts"].append({"id": a.id, "kind": "user", "name": a.name, "boxes": boxes, "purpose": a.why,
                             "scenes": [s for s in a.scenes.split(",") if s]})
        save(d)
        print(f"型 {a.id} を足した")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
