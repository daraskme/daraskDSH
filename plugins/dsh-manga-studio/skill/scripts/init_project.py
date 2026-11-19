#!/usr/bin/env python3
"""漫画プロジェクトの雛形を作る.

  python init_project.py my-manga [--format page|yonkoma|webtoon] [--title 題名] [--rating general|adult]

作るもの:
  manga.toml            作品の契約（形式・ページ数・rating・約束・画風・lint の上書き）
  STATUS.md             現在地と次にやること
  decisions.md          決定ログ（却下した案も）
  characters/cast.toml  人物表（画像生成に貼る外見の固定文面を含む）
  plot/logline.md       ログライン
  plot/structure.md     ページ配分
  plot/episodes/ep001.md  第 1 話の筋
  name/ep001.toml       第 1 話のネーム
  style/directing.md    演出シート
  canon/facts.md        既出の事実（proposed / confirmed）
  canon/summaries/  reviews/  renders/
既存のファイルは上書きしない。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent.parent / "assets" / "templates"
FILES = {
    "manga.toml": "manga.toml",
    "STATUS.md": "STATUS.md",
    "decisions.md": "decisions.md",
    "characters/cast.toml": "cast.toml",
    "plot/logline.md": "logline.md",
    "plot/structure.md": "structure.md",
    "plot/episodes/ep001.md": "episode.md",
    "name/ep001.toml": "name-blank.toml",
    "style/directing.md": "directing.md",
    "canon/facts.md": "facts.md",
}
DIRS = ["canon/summaries", "reviews", "renders"]


def toml_str(v):
    """TOML の基本文字列にする（引用符・バックスラッシュ・制御文字をエスケープ）."""
    out = []
    for ch in str(v):
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--format", choices=["page", "yonkoma", "webtoon"], default="page")
    ap.add_argument("--title", default="")
    ap.add_argument("--rating", choices=["general", "adult"], default="general")
    a = ap.parse_args(argv)
    root = Path(a.project)
    root.mkdir(parents=True, exist_ok=True)
    made, kept = [], []
    for dst, src in FILES.items():
        p = root / dst
        if p.exists():
            kept.append(dst)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        text = (TEMPLATES / src).read_text(encoding="utf-8")
        title = toml_str(a.title)
        if dst == "manga.toml":
            text = re.sub(r'^title = ""', lambda m: f"title = {title}", text, count=1, flags=re.M)
            text = re.sub(r'^format = "page"', lambda m: f'format = "{a.format}"', text, count=1, flags=re.M)
            text = re.sub(r'^rating = "general"', lambda m: f'rating = "{a.rating}"', text, count=1, flags=re.M)
        if dst == "name/ep001.toml":
            text = text.replace('title = ""', f"title = {title}", 1)
            text = text.replace("[meta]\n", f'[meta]\nformat = "{a.format}"\nrating = "{a.rating}"\n', 1)
        p.write_text(text, encoding="utf-8")
        made.append(dst)
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    print(f"作成 {len(made)} / 既存のため残した {len(kept)}: {root}")
    for m in made:
        print(f"  + {m}")
    for k in kept:
        print(f"  = {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
