#!/usr/bin/env python3
"""ネームを JSON と TOML のあいだで変換する.

  python name_convert.py name.json --out name.toml     # LLM に JSON で書かせたネームを、人が直しやすい TOML に
  python name_convert.py name.toml --out name.json

TOML への書き出しは、このスキルのネームの形（meta / characters / pages / panels / characters / text）に限る。
コメントは残らない。JSON の null は「指定なし」として書かない（配列の中の null は誤り）。書いた TOML は読み直して確かめる。
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga_name as mn  # noqa: E402

ORDER_META = ["title", "episode", "format", "first_page_side", "rating", "cast", "style", "negative", "page_aspect"]


def _str(v):
    """TOML の基本文字列（引用符・バックスラッシュ・制御文字・DEL をエスケープ）."""
    out = []
    for ch in str(v):
        if ch in '"\\':
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"TOML に書けない数: {v!r}")
        return repr(v)
    if isinstance(v, list):
        if any(x is None for x in v):
            raise ValueError(f"配列に null が含まれている: {v!r}")
        return "[" + ", ".join(_val(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_key(k)} = {_val(x)}" for k, x in v.items() if x is not None) + "}"
    return _str(v)


def _key(k):
    return k if k.replace("_", "").replace("-", "").isalnum() and k.isascii() else _str(k)


def _scalars(d, indent, skip=()):
    out = []
    for k, v in d.items():
        if k in skip or k.startswith("_") or v is None:      # null は「指定なし」として書かない
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            continue
        out.append(f"{indent}{_key(k)} = {_val(v)}")
    return out


def to_toml(doc):
    lines = []
    meta = doc.get("meta", {})
    if meta:
        lines.append("[meta]")
        keys = [k for k in ORDER_META if k in meta] + [k for k in meta if k not in ORDER_META]
        lines += _scalars({k: meta[k] for k in keys}, "")
        lines.append("")
    for c in doc.get("characters", []):
        lines.append("[[characters]]")
        lines += _scalars(c, "")
        lines.append("")
    for page in doc.get("pages", []):
        lines.append("[[pages]]")
        lines += _scalars(page, "", skip=("panels",))
        for p in page.get("panels", []):
            lines.append("")
            lines.append("  [[pages.panels]]")
            lines += _scalars(p, "  ", skip=("characters", "text"))
            for c in p.get("characters", []) or []:
                lines.append("    [[pages.panels.characters]]")
                lines += _scalars(c, "    ")
            for t in p.get("text", []) or []:
                lines.append("    [[pages.panels.text]]")
                lines += _scalars(t, "    ")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--out", required=True, help=".toml か .json（- で標準出力に TOML）")
    a = ap.parse_args(argv)
    try:
        doc = mn.load_toml(a.name)
    except mn.NameError_ as e:
        mn.eprint(f"エラー: {e}")
        return 2
    if a.out.endswith(".json"):
        text = json.dumps({k: v for k, v in doc.items() if not k.startswith("_")}, ensure_ascii=False, indent=2) + "\n"
    else:
        try:
            text = to_toml(doc)
            tomllib.loads(text)                                  # 書いた TOML が読めることを確かめる
        except (ValueError, tomllib.TOMLDecodeError) as e:
            mn.eprint(f"エラー: TOML にできませんでした: {e}")
            return 2
    if a.out == "-":
        sys.stdout.write(text)
    else:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
