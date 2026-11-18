import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps

from .detail_refs import (DETAILS, OUTFIT_PARTS, PAPER, VIEWS, appearance_text, crops_from_regions,
                          detail_sheet, edit_sheet, overview_sheet, place)

CATEGORY = "Anima Ref Adapter/Character details"


def image_tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255).unsqueeze(0)


def pil_image(tensor):
    if len(tensor) != 1:
        raise ValueError("Choose a single image for each view.")
    return Image.fromarray((tensor[0, ..., :3].detach().cpu().clamp(0, 1) * 255).round().byte().numpy())


def source_paths(files):
    root = Path(folder_paths.get_input_directory()).resolve()
    names = [name.strip() for name in files.splitlines() if name.strip()]
    if not names:
        raise ValueError("Enter one image path per line, relative to the ComfyUI input folder.")
    paths = [(root / name).resolve() for name in names]
    if any(not p.is_relative_to(root) or not p.is_file() for p in paths):
        raise ValueError("Select existing images within the ComfyUI input folder.")
    return root, paths


class GemmaDetailRegions:
    CATEGORY = CATEGORY
    FUNCTION = "locate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("regions_json",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"files": ("STRING", {"multiline": True, "default": "reference.jpg"})}}

    @classmethod
    def IS_CHANGED(cls, files):
        _, paths = source_paths(files)
        return [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

    def locate(self, files):
        root, paths = source_paths(files)
        caption = next((Path(p) / "ComfyUI-Folder-LoRA" for p in folder_paths.get_folder_paths("custom_nodes")
                        if (Path(p) / "ComfyUI-Folder-LoRA/vision.py").is_file()), None)
        if caption is None:
            raise ValueError("Install and configure the local ComfyUI-Folder-LoRA caption node.")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "regions.json"
            cmd = [sys.executable, str(Path(__file__).with_name("locate_details.py")), "--caption-pack",
                   str(caption), "--out", str(dest), *map(str, paths)]
            from comfy import model_management
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            log_path = Path(tmp) / "detection.log"
            with log_path.open("w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
                try:
                    while proc.poll() is None:
                        model_management.throw_exception_if_processing_interrupted()
                        try:
                            proc.wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
            if proc.returncode:
                raise RuntimeError("Gemma detection failed: " + log_path.read_text()[-1600:])
            records = json.loads(dest.read_text())
        for rec, path in zip(records, paths, strict=True):
            rec["file"] = path.relative_to(root).as_posix()
        return (json.dumps(records, ensure_ascii=False),)


class AnimaPrepareDetailRefs:
    CATEGORY = CATEGORY
    FUNCTION = "prepare"
    RETURN_TYPES = ("ANIMA_REF_PARTS", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("parts", "edit_reference", "detail_sheet", "appearance")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"regions_json": ("STRING", {"forceInput": True}),
                             "overrides_json": ("STRING", {"multiline": True, "default": "{}"})}}

    def prepare(self, regions_json, overrides_json):
        records = json.loads(regions_json)
        overrides = json.loads(overrides_json.strip() or "{}")
        crops, selected = crops_from_regions(records, folder_paths.get_input_directory(), overrides)
        return ((crops, selected), image_tensor(edit_sheet(crops)), image_tensor(detail_sheet(crops)),
                appearance_text(selected))


class AnimaSaveDetailPack:
    CATEGORY = CATEGORY
    FUNCTION = "save"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("saved_path", "overview")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"parts": ("ANIMA_REF_PARTS",), "vae": ("VAE",),
                             "character_name": ("STRING", {"default":"character_details"}),
                             "view_origin": (["Qwen Image 2.1", "source images"],),
                             "macro_parts": ("STRING", {"default": ""})},
                "optional": {key: ("IMAGE",) for key in (*VIEWS, "qwen_sheet", "outfit_sheet")}}

    def save(self, parts, vae, character_name, view_origin="Qwen Image 2.1", macro_parts="", qwen_sheet=None, outfit_sheet=None, **images):
        from nodes import NODE_CLASS_MAPPINGS
        from . import write_reference
        encoder = NODE_CLASS_MAPPINGS["AnimaRefLatent"]()
        crops, selected = parts
        views = {key: pil_image(t) for key, t in images.items() if t is not None}
        closeups = pil_image(qwen_sheet) if qwen_sheet is not None else None
        outfit = pil_image(outfit_sheet) if outfit_sheet is not None else None
        board = closeups
        if outfit is not None and closeups is not None:
            board = Image.new("RGB", (960, 1488), PAPER)
            place(board, closeups, (0, 0, 960, 740))
            place(board, outfit, (0, 744, 960, 740))
        elif outfit is not None:
            board = outfit
        overview = overview_sheet(crops, views, board)
        front_details = Image.new("RGB", (1280, 1152), PAPER)
        place(front_details, views.get("front", crops["figure"]), (4, 4, 684, 1144))
        place(front_details, board if board is not None else detail_sheet(crops), (694, 4, 582, 1144))
        sections = {"overview": overview, "front_details": front_details, **views, **crops}
        labels = [label.strip() for label in macro_parts.split(",") if label.strip()]
        if labels and closeups is None:
            raise ValueError("Connect the Qwen close-up sheet that matches macro_parts.")
        for sheet_name, sheet, names in (("qwen_details", closeups, labels), ("qwen_outfit", outfit, OUTFIT_PARTS)):
            if sheet is None:
                continue
            sections[sheet_name] = sheet
            if names:
                if len(names) % 2 or len(set(names)) != len(names) or set(names) - set(DETAILS):
                    raise ValueError("Enter distinct supported part names in row order for a two-row sheet.")
                cols = len(names) // 2
                for i, label in enumerate(names):
                    col, row = i % cols, i // cols
                    sections["qwen_" + label] = sheet.crop((
                        col * sheet.width // cols, row * sheet.height // 2,
                        (col + 1) * sheet.width // cols, (row + 1) * sheet.height // 2))
        tensors = {}
        for key, im in sections.items():
            scale = min(1.0, 2048 / max(im.size))
            width, height = [max(32, round(side * scale / 16) * 16) for side in im.size]
            latent, _, _ = encoder.encode(image_tensor(im), vae, width, height, 235)
            prefix = "" if key == "overview" else key + "."
            for field in ("samples", "samples_half"):
                tensors[prefix + field] = latent[field].detach().cpu().contiguous()
        metadata = {"sections": json.dumps(list(sections)), "original_crops": json.dumps(selected, ensure_ascii=False),
                    "view_origins": json.dumps({key: view_origin for key in views}),
                    "synthesized_views": json.dumps(list(views) if view_origin != "source images" else []),
                    "qwen_macro_parts": json.dumps(labels + (list(OUTFIT_PARTS) if outfit is not None else []))}
        path = write_reference(character_name, tensors, metadata)
        return (str(path), image_tensor(overview))


class QwenDetailInputs:
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("costume_reference", "macro_layout")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"parts": ("ANIMA_REF_PARTS",), "original": ("IMAGE",),
                             "macro_parts": ("STRING", {"default": "eyes,hair_ornament,top,hem"})}}

    def build(self, parts, original, macro_parts):
        crops, _ = parts
        labels = [key.strip() for key in macro_parts.split(",")]
        if not labels or len(labels) % 2 or set(labels) - crops.keys():
            raise ValueError("Enter an even number of crop names from the reviewed regions.")
        cols = len(labels) // 2
        tile = 576 if tuple(labels) == OUTFIT_PARTS else 480
        canvas = Image.new("RGB", (cols*tile, 2*tile), "white")
        for i, label in enumerate(labels):
            place(canvas, crops[label], (i%cols*tile+4, i//cols*tile+4, tile-8, tile-8))
        costume = ImageOps.contain(pil_image(original), (672,1056), Image.Resampling.LANCZOS)
        return (image_tensor(costume), image_tensor(canvas))


class SplitQwenTurnaround:
    CATEGORY = CATEGORY
    FUNCTION = "split"
    RETURN_TYPES = ("IMAGE",) * 4
    RETURN_NAMES = VIEWS

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sheet": ("IMAGE",)}}

    def split(self, sheet):
        w = sheet.shape[2]
        return tuple(sheet[:, :, i*w//4:(i+1)*w//4, :] for i in range(4))


class SaveGemmaDetailCandidates:
    CATEGORY = CATEGORY
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "folder")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"regions_json": ("STRING", {"forceInput": True}),
                             "outfit": ("STRING", {"default": "character"})}}

    def save(self, regions_json, outfit):
        from datetime import datetime
        from PIL import ImageDraw, ImageOps
        from . import check_filename
        records = json.loads(regions_json)
        source_root = Path(folder_paths.get_input_directory()).resolve()
        root = Path(folder_paths.get_output_directory()) / "gemma_candidates" / check_filename(outfit.strip())
        root.mkdir(parents=True, exist_ok=True)
        listing = []
        groups = {}
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            staged = Path(tmp) / "crops"
            staged.mkdir()
            for number, record in enumerate(records, 1):
                source = (source_root / record["file"]).resolve()
                if not source.is_relative_to(source_root):
                    raise ValueError("Source is outside the input folder.")
                with Image.open(source) as file:
                    im = ImageOps.exif_transpose(file).convert("RGB")
                if list(im.size) != record["size"]:
                    raise ValueError(f"Source size changed: {source.name}")
                for i, entry in enumerate(record.get("candidates", []), 1):
                    part = check_filename(entry["part"])
                    y, x, yy, xx = entry["box"]
                    box = [round(x*im.width/1000), round(y*im.height/1000),
                           round(xx*im.width/1000), round(yy*im.height/1000)]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    name = f"{part}_{number:02d}_{i:02d}.png"
                    crop = im.crop(box)
                    crop.save(staged / name)
                    listing.append({**entry, "source":record["file"], "pixel_box_xyxy":box, "crop":name})
                    score = (entry["visibility"] == "clear", crop.width * crop.height)
                    if part not in groups or score > groups[part][0]:
                        groups[part] = score, name
            if not listing:
                raise ValueError("Gemma found no usable candidates in these images.")
            priorities = ("hair_fastening", "hair_ornament", "upper_body", "lower_body", "clothing_detail", "eyes")
            keys = sorted(groups, key=lambda k: (priorities.index(k) if k in priorities else len(priorities), k))
            sheet = Image.new("RGB", (800, 155*((len(keys)+3)//4)), PAPER)
            draw = ImageDraw.Draw(sheet)
            for i, key in enumerate(keys):
                left, top = i%4*200, i//4*155
                with Image.open(staged / groups[key][1]) as crop:
                    place(sheet, crop, (left+3,top+3,194,127))
                draw.text((left+6,top+134), key, fill="black")
            sheet.save(staged / "preview.jpg", quality=94)
            (staged / "candidates.json").write_text(json.dumps(listing, ensure_ascii=False, indent=2), encoding="utf-8")
            destination = root / (datetime.now().strftime("%Y%m%d-%H%M%S-") + Path(tmp).name)
            staged.rename(destination)
        return image_tensor(sheet), str(destination)
