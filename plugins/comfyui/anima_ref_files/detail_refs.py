"""Prepare crops and reference sheets from local Gemma detections."""
import math
from pathlib import Path

from PIL import Image, ImageOps

OUTFIT_PARTS = ("hair_fastening", "upper_body", "lower_body", "clothing_detail")
DETAILS = ("face", "eyes", "hair_ornament", "flowers", "earring", "top", "neck_trim", "gloves", "skirt", "hem", "cape_trim", "shoes", "bottoms", "waist", "chest_mark", "headphones", *OUTFIT_PARTS)
VIEWS = ("front", "left", "right", "back")
PAPER = (237, 235, 232)


def place(canvas, image, box):
    x, y, w, h = box
    im = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
    canvas.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def crops_from_regions(records, input_root, edits):
    """Overrides use [left, top, right, bottom] in the original image pixels."""
    images = {}
    best = {}
    for record in records:
        file = record["file"]
        root = Path(input_root).resolve()
        path = (root / file).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source files must be inside the selected input folder.")
        with Image.open(path) as im:
            images[file] = ImageOps.exif_transpose(im).convert("RGB")
        w, h = images[file].size
        if [w, h] != record["size"]:
            raise ValueError(f"Image size changed for {file}. Run Gemma again.")
        for part, data in record["parts"].items():
            y, x, yy, xx = data["box"]
            box = [round(x*w/1000), round(y*h/1000), round(xx*w/1000), round(yy*h/1000)]
            area = (box[2]-box[0]) * (box[3]-box[1])
            if area <= 0:
                continue
            score = (data["visibility"] == "clear", area)
            if part not in best or score > best[part][0]:
                best[part] = (score, {"file":file, "box":box, "description":data["description"]})
    selected = {key: val[1] for key, val in best.items()}
    for part, change in edits.items():
        if change is None:
            selected.pop(part, None)
        else:
            selected[part] = change
    if "figure" not in selected:
        raise ValueError("Select a figure crop before building the reference.")
    images_out = {}
    for name, selection in selected.items():
        image = images[selection["file"]]
        x, y, xx, yy = selection["box"]
        if not (0 <= x < xx <= image.width and 0 <= y < yy <= image.height):
            raise ValueError(f"Crop outside {selection['file']}: {name}")
        images_out[name] = image.crop((x,y,xx,yy))
    return images_out, selected


def detail_sheet(crops):
    names = [key for key in DETAILS if key in crops]
    cols = 3
    rows = max(1, math.ceil(len(names) / cols))
    sheet = Image.new("RGB", (cols*300, rows*236), PAPER)
    for i, key in enumerate(names):
        place(sheet, crops[key], (i % cols*300+4, i//cols*236+4, 292,228))
    return sheet


def edit_sheet(crops):
    board = Image.new("RGB", (832, 1088), PAPER)
    place(board, crops["figure"], (6,6,537,1076))
    regions = [("face",245),("eyes",99),("hair_ornament",173),("top",200),("hem",139),("shoes",130)]
    y = 4
    for key, h in regions:
        if key in crops:
            place(board, crops[key], (550,y,276,h))
        y += h+12
    return board


def overview_sheet(crops, views, closeups=None):
    board = Image.new("RGB", (1792, 1408), PAPER)
    place(board, views.get("front", crops["figure"]), (8,8,568,904))
    if closeups is None:
        place(board, detail_sheet(crops), (590,8,1194,904))
    else:
        place(board, detail_sheet(crops), (585,8,600,904))
        place(board, closeups, (1190,8,594,904))
    supplied = [views[k] for k in VIEWS if k in views]
    for i, im in enumerate(supplied):
        place(board, im, (i*448+3,924,442,476))
    return board


def appearance_text(selected):
    return ". ".join(str(selected[k]["description"]).strip().rstrip(".")
                     for k in DETAILS if k in selected and selected[k].get("description"))
