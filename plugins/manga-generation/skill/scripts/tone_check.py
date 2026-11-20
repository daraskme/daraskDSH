#!/usr/bin/env python3
"""生成した白黒ページの「灰色の塗り」を測る（Pillow が要る。他のスクリプトと違い標準ライブラリだけでは動かない）。

白黒の漫画は、線・ベタ（黒）・トーン（網点）・白でできている。画像生成モデルは、ここに
エアブラシのような「なめらかな灰色の面」を足しやすい（厚塗り・グラデーション・光沢）。
網点は元の解像度では白と黒の点なので、縮小しすぎると灰の面と区別できない。このスクリプトは
長辺 2000px まで（それより小さい画像は元のまま）で、次の 4 つを測る。

  white     白（明度 225 以上）の割合
  black     黒（明度 50 以下）の割合
  mid       その間の灰の割合（網点も縮小で灰に混ざるので、参考値。警報には使わない）
  flatgray  なめらかな灰の面の割合（5px 四方の明度のばらつきが小さく、明度が 60〜215 の場所）

測る単位は 1 ページ。見開きの画像は --spread で左右に割り、右（#右）と左（#左）を別の行で返す。1 コマだけの画像には使わない
（背景を飛ばしたコマや、回想のグラデーションのコマで高く出る）。

目安（2026-11-15、Manga109-s の 400 見開きを左右に割った 800 ページ。スキャンの JPEG）:
  flatgray  中央値 0.000、上位 1 割 0.001、上位 5% 0.003
  white     中央値 0.67、 下位 1 割 0.51
同じ AI で作った比較画像 1 組（各 1 ページ）では、厚塗りの版が flatgray 0.018・white 0.58、
描き込みを絞った版が 0.000・0.81。1600x2400 の 2px 周期の網点は flatgray 0.000、縦のグラデーションは 0.50。
閾値は実際の漫画の上位 5% の少し上に置いた目安で、作品ごとには較正していない。

使い方:
  python tone_check.py page1.png page2.png ...      # 1 行ずつ数値と判定
  python tone_check.py --spread spread.jpg          # 見開きの画像
  python tone_check.py --json DIR/*.png             # JSON で

判定は「目で確かめる場所」の印で、合否でも原因の断定でもない。カラーページ、意図した回想のグラデーション、
写真の背景、ぼかしたトーンは高く出る。警報だけを理由に画面を均さない。
終了コード: 0 = 警報なし、1 = 警報あり、2 = 読めない画像があった。
"""
import argparse
import json
import sys

try:
    from PIL import Image, ImageChops, ImageFilter, ImageStat
except ImportError:  # pragma: no cover
    sys.stderr.write("Pillow が要る（pip install Pillow）。無ければこの検査は飛ばし、visual-check.md の層 7 を目で見る\n")
    sys.exit(2)

LIMITS = {"flatgray": 0.005, "white_min": 0.48}  # 実際の漫画の上位 5% の少し上、白の下位 7% ほど


MAX_SIDE = 2000  # 網点を潰さないよう、縮小は長辺 2000px まで（それより小さい画像は元のまま）


def measure(path, spread=False):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg
    im = im.convert("L")
    if spread and im.width > im.height:
        # 見開きは右ページと左ページに割って、両方を返す（目安は 1 ページ単位で作った）
        return [("右", _measure(im.crop((im.width // 2, 0, im.width, im.height)))),
                ("左", _measure(im.crop((0, 0, im.width // 2, im.height))))]
    return [("", _measure(im))]


def _measure(im):
    im = im.copy()
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    n = im.width * im.height
    h = im.histogram()
    white = sum(h[225:]) / n
    black = sum(h[:51]) / n
    mid = 1 - white - black
    # なめらかな面: 5x5 の平均との差の平均（局所のばらつき）が小さく、明度が中間
    blur = im.filter(ImageFilter.BoxBlur(2))
    dev = ImageChops.difference(im, blur).filter(ImageFilter.BoxBlur(2))
    flat = dev.point(lambda v: 255 if v < 4 else 0)
    midband = blur.point(lambda v: 255 if 60 < v < 215 else 0)
    both = ImageChops.multiply(flat, midband)
    flatgray = ImageStat.Stat(both).mean[0] / 255
    return {"white": round(white, 3), "black": round(black, 3), "mid": round(mid, 3), "flatgray": round(flatgray, 3)}


def verdict(m):
    notes = []
    if m["flatgray"] > LIMITS["flatgray"]:
        notes.append("なめらかな灰の面が多い。厚塗り・エアブラシ・グラデーションの可能性があるので、目で確かめる")
    if m["white"] < LIMITS["white_min"]:
        notes.append("白が少ない。描き込みかトーンの貼りすぎの可能性があるので、目で確かめる")
    return notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--spread", action="store_true", help="横長の画像を見開きとして左右に割って測る")
    a = ap.parse_args(argv)
    rows, errors = [], []
    for p in a.images:
        try:
            parts = measure(p, spread=a.spread)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"{p}: 読めない（{e}）\n")
            errors.append({"image": p, "error": str(e)})
            continue
        for side, m in parts:
            rows.append({"image": f"{p}#{side}" if side else p, **m, "warn": verdict(m)})
    if a.json:
        print(json.dumps({"results": rows, "errors": errors}, ensure_ascii=False, indent=1))
    else:
        for r in rows:
            flag = "WARN " + "／".join(r["warn"]) if r["warn"] else "ok"
            print(f"{r['image']}\twhite {r['white']:.3f}\tblack {r['black']:.3f}\tmid {r['mid']:.3f}\tflatgray {r['flatgray']:.3f}\t{flag}")
    if errors:
        return 2
    return 1 if any(r["warn"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
