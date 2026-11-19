#!/usr/bin/env python3
"""ページの絵を CBZ（漫画の読書アプリで読める zip）にまとめる。右綴じ・白黒を ComicInfo.xml に書く.

  python make_cbz.py OUT.cbz p001.png p002.png ... [--title 題名] [--summary あらすじ] [--ltr] [--color]

ページはファイル名の順。中では 001.png, 002.png … に付け直す。無圧縮（画像は圧縮済みなので）。
"""
import argparse
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def comic_info(title, summary, pages, rtl=True, bw=True):
    return ('<?xml version="1.0" encoding="utf-8"?>\n<ComicInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
            f"  <Title>{escape(title)}</Title>\n  <Summary>{escape(summary)}</Summary>\n  <PageCount>{pages}</PageCount>\n"
            f"  <LanguageISO>ja</LanguageISO>\n  <Manga>{'YesAndRightToLeft' if rtl else 'No'}</Manga>\n"
            f"  <BlackAndWhite>{'Yes' if bw else 'No'}</BlackAndWhite>\n</ComicInfo>\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--title", default="")
    ap.add_argument("--summary", default="")
    ap.add_argument("--ltr", action="store_true", help="左綴じ（横書きの本）")
    ap.add_argument("--color", action="store_true")
    a = ap.parse_args(argv)
    pages = sorted(Path(p) for p in a.pages)
    missing = [str(p) for p in pages if not p.exists()]
    if missing:
        print("見つからない: " + ", ".join(missing), file=sys.stderr)
        return 2
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        for i, p in enumerate(pages, 1):
            z.write(p, f"{i:03d}{p.suffix.lower()}")
        z.writestr("ComicInfo.xml", comic_info(a.title, a.summary, len(pages), not a.ltr, not a.color))
    print(f"{len(pages)} ページを {a.out} にまとめた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
