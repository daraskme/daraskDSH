#!/usr/bin/env python3
"""data/fonts.json のフォント（SIL OFL、Google Fonts）とライセンス文を取る。あるものは飛ばす.

  python fetch_fonts.py [--dest DIR]     既定は MANGA_FONT_DIR か、このリポジトリの .fonts/
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.environ.get("MANGA_FONT_DIR") or str(HERE.parent / ".fonts"))
    a = ap.parse_args(argv)
    DEST = Path(a.dest)
    DEST.mkdir(parents=True, exist_ok=True)
    for font in json.loads((HERE / "data" / "fonts.json").read_text(encoding="utf-8"))["fonts"]:
        for url, name in ((font["url"], font["file"]), (font["license_url"], Path(font["url"]).parent.name + ".OFL.txt")):
            path = DEST / name
            if path.exists() and path.stat().st_size:
                continue
            print("取得:", name, file=sys.stderr)
            with urllib.request.urlopen(url, timeout=120) as res:
                path.with_suffix(".part").write_bytes(res.read())
            path.with_suffix(".part").rename(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
