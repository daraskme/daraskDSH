#!/usr/bin/env python3
"""マンガタブの下請け: ネーム JSON を ja-manga-writing のスクリプト（skill/scripts）で割り付け・検査し、
コマごとのプロンプトと写植・演出の層の叩き台を作る（絵と層の分け方は skill/scripts/name_layers.py）.

  python manga.py analyze NAME.json      → 標準出力に JSON
    {pages: [{number, side, aspect, panels: [{id, bbox, aspect, prompt, anima, avoid, layers}]}],
     issues: [{code, level, where, message}], stats, error?}
  python manga.py crop SRC DST ASPECT    コマの縦横比（横/縦）で中央を切り抜いた確認用の絵（長辺 1024px）

bbox はページの norm_1000。layers の座標はコマの中の割合（x は幅、y は高さに対して 0〜1）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "skill" / "scripts"))
import manga_name as mn  # noqa: E402
import name_layers as nlay  # noqa: E402
import name_lint as nl  # noqa: E402


def analyze(path):
    try:
        doc = mn.load_name(path)
    except mn.NameError_ as e:
        return {"error": str(e), "pages": [], "issues": []}
    struct = mn.precheck(doc)
    if struct:
        return {"error": "ネームの構造に誤りがあります", "pages": [], "issues": [i.as_dict() for i in struct]}
    doc["meta"].pop("overlay", None)   # 層で描く漫符・効果は skill/scripts/data/overlay.json
    th, opts = nl.load_config(doc, path, None)
    issues, stats = nl.lint(doc, th, opts)
    return {"pages": nlay.panels(doc), "issues": [i.as_dict() for i in issues], "stats": stats}


def crop(src, dst, aspect):
    from PIL import Image
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if w / h > aspect:
        nw = int(h * aspect)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / aspect)
        img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    img.thumbnail((1024, 1024))
    img.save(dst)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "analyze":
        print(json.dumps(analyze(sys.argv[2]), ensure_ascii=False, default=str))
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "crop":
        crop(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        return 0
    print("使い方: manga.py analyze NAME.json | crop SRC DST ASPECT", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
