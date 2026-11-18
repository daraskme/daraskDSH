"""Locate visible costume details using the local Folder LoRA vision runtime."""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

PARTS = ("figure", "face", "eyes", "hair_ornament", "earring", "top", "neck_trim", "gloves",
         "skirt", "hem", "bottoms", "waist", "chest_mark", "shoes", "headphones",
         "hair_fastening", "upper_body", "lower_body", "clothing_detail", "flowers", "cape_trim")
REQUEST = """Locate visible garment and character-design references on the main subject in this image.
Return a sparse list. A close-up portrait often has NO visible lower body. Omit absent or completely
hidden parts. Never turn background motifs, UI, decorative frames, printed cartoons or text into body
parts. A hair fastening may be a plain ribbon or a visible twist of hair; only report flowers when
flowers are visible. Give one upper_body crop showing the visible outfit between shoulders and waist,
and one lower_body crop of the visible outfit below the waist if present. Also look separately for
hair_fastening, eyes, hair_ornament, and clothing_detail, e.g. embroidery, piping, fasteners, emblems,
neckline or stripe. Include useful alternatives (up to two per part) when two distinct details occur.
For each candidate, describe what is visible, why that detail can be lost at full-body scale, and
whether it is clear, occluded or small. Box = [top, left, bottom, right], each normalized to 0..1000.
Do not invent unobserved parts or make guesses about the reverse side. Output JSON only."""
REGION = {"type": "object", "properties": {
    "part": {"type": "string", "enum": list(PARTS)},
    "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
    "description": {"type": "string"}, "reason": {"type": "string"},
    "visibility": {"type": "string", "enum": ["clear", "occluded", "small"]}},
    "required": ["part", "box", "description", "reason", "visibility"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {"regions": {"type": "array", "items": REGION}},
          "required": ["regions"], "additionalProperties": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.caption_pack.resolve()))
    from vision import image_item, server
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    cfg = Path(os.environ.get("COMFY_FOLDER_LORA_CONFIG") or config_dir / "comfy-folder-lora/config.json")
    vision_cfg = json.loads(cfg.read_text())["vision"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with server(vision_cfg, args.out.parent) as (session, base):
        for path in args.images:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                size = im.size
                content = [image_item(im), {"type": "text", "text": REQUEST}]
            response = session.post(base + "/v1/chat/completions", json={
                "model": "ortenzya", "temperature": 0, "seed": 42, "max_tokens": 5000,
                "response_format": {"type": "json_schema", "json_schema": {"name":"regions", "schema":SCHEMA}},
                "messages": [{"role":"user","content":content}]}, timeout=(10,360))
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice["finish_reason"] == "length":
                raise RuntimeError(f"Incomplete region list for {path.name}")
            regions = json.loads(choice["message"]["content"])["regions"]
            candidates = []
            for entry in regions:
                y1, x1, y2, x2 = entry["box"]
                if 0 <= y1 < y2 <= 1000 and 0 <= x1 < x2 <= 1000:
                    candidates.append(entry)
            parts = {}
            for entry in candidates:
                name = entry["part"]
                if name not in parts or (parts[name]["visibility"] != "clear" and entry["visibility"] == "clear"):
                    parts[name] = entry
            records.append({"file": path.name, "size": size, "parts": parts, "candidates": candidates})
            print(path.name, len(candidates), "candidates", flush=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
