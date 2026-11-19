#!/usr/bin/env python3
"""連載の 1 話ぶんのネームを切るのに要る情報だけを 1 枚に束ねる（コンテキストパック）.

  python build_context.py --project DIR --episode 3 [--out renders/ctx-ep003.md] [--recent 2]

束ねるもの（この順）: manga.toml の契約、STATUS.md、人物表（名前・外見・声・欲しいもの）、演出シート、
canon/facts.md の confirmed 行、直前 --recent 話の要約（canon/summaries/epNNN.md のうち、1 行目が「状態: confirmed」のもの）、今回の話の筋（plot/episodes/epNNN.md）。
proposed の事実と、未確認の要約は前提にしないので入れない（件数・話数だけ示す）。
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


def read(p):
    try:
        return Path(p).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def summary_status(text):
    """要約の 1 行目「状態: confirmed」（または status: confirmed）を読む. 無ければ proposed 扱い."""
    first, _, rest = text.partition("\n")
    m = re.match(r"\s*(?:状態|status)\s*[:：]\s*(\w+)", first, re.I)
    if m:
        return m.group(1).lower(), rest.strip()
    return "proposed", text.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--recent", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    root = Path(a.project)
    if not (root / "manga.toml").exists():
        print(f"エラー: {root / 'manga.toml'} が無い（init_project.py で作る）", file=sys.stderr)
        return 2
    out = [f"# コンテキストパック 第{a.episode}話", ""]
    with open(root / "manga.toml", "rb") as f:
        m = tomllib.load(f)
    out.append("## 作品の契約（manga.toml）")
    for sec in ("work", "contract", "render"):
        for k, v in m.get(sec, {}).items():
            if v not in ("", [], 0):
                out.append(f"- {sec}.{k}: {v}")
    out.append("")
    st = read(root / "STATUS.md")
    if st:
        out += ["## STATUS", st, ""]
    cast_p = root / "characters" / "cast.toml"
    if cast_p.exists():
        with open(cast_p, "rb") as f:
            cast = tomllib.load(f).get("characters", [])
        out.append("## 人物")
        for c in cast:
            out.append(f"- {c.get('name') or c.get('id')}（{c.get('id')}）")
            for k in ("appearance", "silhouette", "voice", "wants", "lora", "reference"):
                if c.get(k):
                    out.append(f"  - {k}: {c[k]}")
        out.append("")
    d = read(root / "style" / "directing.md")
    if d:
        out += ["## 演出シート", d, ""]
    facts = read(root / "canon" / "facts.md")
    if facts:
        rows = [ln for ln in facts.splitlines() if ln.startswith("|")]
        conf = [ln for ln in rows if ln.split("|")[1].strip() == "confirmed"]
        prop = [ln for ln in rows if ln.split("|")[1].strip() == "proposed"]
        out.append("## 既出の事実（confirmed のみ）")
        out += conf or ["（なし）"]
        if prop:
            out.append(f"\n未確認の proposed が {len(prop)} 件ある（前提にしない。STATUS で作者に確認）")
        out.append("")
    sums, held = [], []
    for n in range(max(1, a.episode - a.recent), a.episode):
        s = read(root / "canon" / "summaries" / f"ep{n:03d}.md")
        if not s:
            continue
        status, body = summary_status(s)
        if status == "confirmed":
            sums.append(f"### 第{n}話\n{body}")
        else:
            held.append(n)
    if sums:
        out += ["## 直前の話の要約（confirmed のみ）"] + sums + [""]
    if held:
        out += [f"未確認の要約（第{'・'.join(map(str, held))}話）は前提にしない。作者が認めたら 1 行目を「状態: confirmed」に", ""]
    ep = read(root / "plot" / "episodes" / f"ep{a.episode:03d}.md")
    out += ["## 今回の話の筋", ep or "（plot/episodes にまだ無い。先に筋を決める）", ""]
    text = "\n".join(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
