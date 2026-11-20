#!/usr/bin/env python3
"""ネーム・コマの絵・写植と演出の層から、ページの絵（SVG。あれば PNG も）を組む.

  python page_compose.py name.toml --images DIR [--layers LAYERS.json] --out DIR [--png] [--width 1654]

  --images   コマの絵のフォルダ。ファイル名はコマの id（P1-1.png など。png / jpg / webp）
  --layers   層。name_export.py --to layers の出力か、{コマの id: [層, ...]}。省略するとネームからの自動配置
  --png      SVG を PNG にもする（rsvg-convert か Chrome / Chromium が要る）

コマの絵はコマ枠いっぱいに中央で切り抜いて置き（cover）、枠線を引き、層（typeset.py）を上に重ねる。
吹き出しはコマの外へ PAD まではみ出せる（枠越え）。文字はすべてパスなので、どの環境でも同じに見える。
要るもの: typeset.py と同じ（fonttools・uharfbuzz・フォント）。
"""
import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import manga_name as mn
import name_layers as nlay
import typeset

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def find_image(folder, panel_id):
    if not folder:
        return None
    for ext in MIME:
        p = Path(folder) / f"{panel_id}{ext}"
        if p.exists():
            return p
    return None


def load_layers(path):
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and "pages" in d:
        return {p["id"]: p["layers"] for pg in d["pages"] for p in pg["panels"]}
    return d


def compose_page(page, images, layers, width=1654, border=4):
    W, H = width, round(width * page["aspect"])
    sx, sy = W / 1000, H / 1000
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>']
    for k, p in enumerate(page["panels"]):
        x1, y1, x2, y2 = p["bbox"]
        x, y, w, h = x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy
        out.append(f'<clipPath id="pc{k}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/></clipPath>')
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#fff"/>')
        img = find_image(images, p["id"])
        if img:
            data = base64.b64encode(img.read_bytes()).decode()
            out.append(f'<image clip-path="url(#pc{k})" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="xMidYMid slice" '
                       f'href="data:{MIME[img.suffix.lower()]};base64,{data}"/>')
    for p in page["panels"]:
        x1, y1, x2, y2 = p["bbox"]
        out.append(f'<rect x="{x1 * sx:.1f}" y="{y1 * sy:.1f}" width="{(x2 - x1) * sx:.1f}" height="{(y2 - y1) * sy:.1f}" fill="none" stroke="#000" stroke-width="{border}"/>')
    pad = typeset.PAD
    for p in page["panels"]:
        x1, y1, x2, y2 = p["bbox"]
        w, h = (x2 - x1) * sx, (y2 - y1) * sy
        svg = typeset.render({"aspect": p["aspect"], "layers": layers.get(p["id"], p["layers"]), "seed": p["id"]})["svg"]
        # 層の SVG を、コマの外へ PAD 広げた位置に入れ子で置く
        inner = re.sub(r'^<svg [^>]*?(viewBox="[^"]+")[^>]*>',
                       lambda m: f'<svg x="{x1 * sx - pad * w:.1f}" y="{y1 * sy - pad * h:.1f}" width="{w * (1 + 2 * pad):.1f}" '
                                 f'height="{h * (1 + 2 * pad):.1f}" {m.group(1)} overflow="visible">', svg, count=1)
        out.append(inner)
    out.append("</svg>")
    return "\n".join(out)


def to_png(svg_path, png_path, width, height):
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(width), "-o", str(png_path), str(svg_path)], check=True)
        return True
    chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser", "chrome") if shutil.which(c)), None)
    if not chrome:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "page.html"
        html.write_text(f'<html><body style="margin:0"><img src="{Path(svg_path).resolve().as_uri()}" width="{width}" height="{height}"></body></html>')
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars", "--allow-file-access-from-files",
                        f"--window-size={width},{height}", f"--screenshot={Path(png_path).resolve()}", html.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--images")
    ap.add_argument("--layers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--width", type=int, default=1654)
    a = ap.parse_args(argv)
    try:
        doc = mn.load_name(a.name)
    except mn.NameError_ as e:
        mn.eprint(f"エラー: {e}")
        return 2
    if mn.precheck(doc):
        mn.eprint("ネームの構造に誤りがある（name_lint.py で確かめる）")
        return 2
    layers = load_layers(a.layers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for page in nlay.panels(doc):
        svg = out / f"p{page['number']:03d}.svg"
        svg.write_text(compose_page(page, a.images, layers, a.width), encoding="utf-8")
        made.append(svg)
        if a.png and not to_png(svg, svg.with_suffix(".png"), a.width, round(a.width * page["aspect"])):
            mn.eprint("PNG にする道具（rsvg-convert か Chrome）が無いので SVG だけ書き出した")
            a.png = False
    print(f"{len(made)} ページを書き出した: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
