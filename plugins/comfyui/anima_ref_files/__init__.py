import os
import tempfile
from itertools import count
from pathlib import Path, PureWindowsPath

import folder_paths
from safetensors import safe_open
from safetensors.torch import save_file

from .detail_nodes import (AnimaPrepareDetailRefs, AnimaSaveDetailPack, GemmaDetailRegions,
                           QwenDetailInputs, SaveGemmaDetailCandidates, SplitQwenTurnaround)
from .detail_refs import DETAILS, VIEWS


FORMAT = "anima-ref-latent/1"
TENSOR_KEYS = ("samples", "samples_half")


def reference_dir():
    return Path(folder_paths.get_output_directory()) / "anima_refs"


def check_filename(name):
    if (not name or name.endswith((" ", ".")) or PureWindowsPath(name).is_reserved()
            or any(c in '/\\<>:"|?*' or ord(c) < 32 for c in name)):
        raise ValueError("Use a filename without a directory or reserved characters.")
    return name


def write_reference(character_name, tensors, metadata=None):
    name = check_filename(character_name.strip().removesuffix(".safetensors"))
    directory = reference_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        staged = Path(tmp) / "reference"
        save_file(tensors, str(staged), metadata={"format": FORMAT, **(metadata or {})})
        # Publish a complete file without overwriting another export of the same character.
        for version in count(1):
            suffix = "" if version == 1 else f"_{version:03d}"
            destination = directory / f"{name}{suffix}.safetensors"
            try:
                os.link(staged, destination)
                break
            except FileExistsError:
                continue
    return destination


class SaveAnimaRefSafetensors:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ref_latent": ("LATENT",),
            "character_name": ("STRING", {"default": "character"}),
        }}

    CATEGORY = "Anima Ref Adapter"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)

    def save(self, ref_latent, character_name):
        if any(key not in ref_latent for key in TENSOR_KEYS):
            raise ValueError("Connect the ref_latent output of Anima Ref Latent (full and half resolution).")
        tensors = {key: ref_latent[key].detach().cpu().contiguous() for key in TENSOR_KEYS}
        if any(t.ndim not in (4, 5) or t.shape[:2] != (1, 16) or
               (t.ndim == 5 and t.shape[2] != 1) for t in tensors.values()):
            raise ValueError("Use a single reference image and the Anima Qwen-Image VAE.")

        return (str(write_reference(character_name, tensors)),)


class LoadAnimaRefSafetensors:
    @classmethod
    def INPUT_TYPES(cls):
        names = sorted(p.name for p in reference_dir().glob("*.safetensors") if p.is_file())
        return {"required": {"filename": (names,)},
                "optional": {"reference": (["overview", "front_details", "figure", *VIEWS, *DETAILS, "qwen_details", "qwen_outfit",
                                            *("qwen_" + part for part in DETAILS)],)}}

    CATEGORY = "Anima Ref Adapter"
    FUNCTION = "load"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("ref_latent",)

    @classmethod
    def IS_CHANGED(cls, filename, reference="overview"):
        info = (reference_dir() / check_filename(filename)).stat()
        return (info.st_mtime_ns, info.st_size)

    def load(self, filename, reference="overview"):
        path = reference_dir() / check_filename(filename)
        with safe_open(str(path), framework="pt", device="cpu") as file:
            if (file.metadata() or {}).get("format") != FORMAT:
                raise ValueError(f"{filename} is not an Anima reference exported by this workflow.")
            prefix = "" if reference == "overview" else reference + "."
            if prefix + "samples" not in file.keys():
                raise ValueError(f"{filename} has no {reference} reference. Select overview for a single reference.")
            return ({key: file.get_tensor(prefix + key) for key in TENSOR_KEYS},)


NODE_CLASS_MAPPINGS = {
    "SaveGemmaDetailCandidates": SaveGemmaDetailCandidates,
    "QwenDetailInputs": QwenDetailInputs,
    "SplitQwenTurnaround": SplitQwenTurnaround,
    "GemmaDetailRegions": GemmaDetailRegions,
    "AnimaPrepareDetailRefs": AnimaPrepareDetailRefs,
    "AnimaSaveDetailPack": AnimaSaveDetailPack,
    "SaveAnimaRefSafetensors": SaveAnimaRefSafetensors,
    "LoadAnimaRefSafetensors": LoadAnimaRefSafetensors,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveGemmaDetailCandidates": "Gemma · Save Detail Candidates",
    "QwenDetailInputs": "Qwen 2.1 · Outfit and Detail References",
    "SplitQwenTurnaround": "Qwen 2.1 · Split Four Views",
    "GemmaDetailRegions": "Gemma · Locate Character Details",
    "AnimaPrepareDetailRefs": "Anima · Review and Crop Details",
    "AnimaSaveDetailPack": "Anima · Save Character Reference Pack",
    "SaveAnimaRefSafetensors": "Save Anima Ref (.safetensors)",
    "LoadAnimaRefSafetensors": "Load Anima Ref (.safetensors)",
}
