#!/usr/bin/env python3
"""fonts.json のフォント（SIL OFL）とライセンス文を .runtime/manga/fonts に取る。あるものは飛ばす."""
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE.parent.parent / ".runtime" / "manga" / "fonts"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for font in json.loads((HERE / "fonts.json").read_text(encoding="utf-8"))["fonts"]:
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
