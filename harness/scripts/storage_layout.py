"""Keep the current storage layout and the paths saved in earlier jobs compatible."""
import json
import os
from pathlib import Path


def link_directory(link, target):
    relative = os.path.relpath(target, link.parent)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(relative, target_is_directory=True)


def relocate(old, new, aliases):
    if old.is_symlink():
        if old.resolve() == new.resolve():
            return
        if old.exists():
            raise FileExistsError(f"{old} already refers to {old.resolve()}")
        old.unlink()
    if old.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            for child in old.iterdir():
                if (new / child.name).exists() or (new / child.name).is_symlink():
                    raise FileExistsError(new / child.name)
            for child in old.iterdir():
                child.rename(new / child.name)
            old.rmdir()
        else:
            old.rename(new)
    if aliases:
        new.mkdir(parents=True, exist_ok=True)
        link_directory(old, new)


def job_output(root, job, model):
    if job.name.endswith("-h3"):
        return root / "outputs/videos" / job.name
    request = root / ".runtime/anima-native/jobs" / job.name / "request.json"
    body = json.loads(request.read_text()) if request.exists() else {}
    group = "edits" if body.get("image") or body.get("mode") == "edit" else "images"
    return root / "outputs" / group / model / job.name


def organize_storage(root, *, aliases=True):
    runtime = root / ".runtime"
    for legacy, target in (
        ("plugins/dsh-comfyui/models", "models"),
        ("plugins/dsh-comfyui/characters", "datasets/references/characters"),
        ("plugins/dsh-comfyui/refmods", "datasets/references/refmods"),
        (".runtime/vocab-pack", "models/vocab-pack"),
        (".runtime/ComfyUI/input", "datasets/comfyui-input"),
        (".runtime/anima-native/uploads", "datasets/media-uploads"),
        (".runtime/manga/projects", "outputs/manga/projects"),
        (".runtime/manga/trash", "outputs/manga/trash"),
        (".runtime/exports", "outputs/exports"),
        (".runtime/ComfyUI/output", "outputs/comfyui"),
        (".runtime/anima-native/datasets", "datasets/uploads"),
    ):
        relocate(root / legacy, root / target, aliases)
    for legacy, model in (("anima-native", "anima"), ("h3-native", "p-h3")):
        previous = runtime / legacy / "generated"
        if previous.is_dir():
            for job in previous.iterdir():
                if job.is_dir() and not job.is_symlink():
                    relocate(job, job_output(root, job, model), aliases)
        if aliases:
            groups = [f"images/{model}", f"edits/{model}"]
            if model == "p-h3":
                groups.append("videos")
            for group in groups:
                for job in (root / "outputs" / group).glob("*/"):
                    link_directory(previous / job.name, job)
    if aliases:
        for name in ("images/anima", "images/p-h3", "edits/anima", "edits/p-h3", "videos"):
            (root / "outputs" / name).mkdir(parents=True, exist_ok=True)
        (root / "models/loras/trained").mkdir(parents=True, exist_ok=True)
        link_directory(root / "datasets/shared", root.parent / "dataset")
        link_directory(root / "datasets/references/gemma_candidates", root / "outputs/comfyui/gemma_candidates")


if __name__ == "__main__":
    organize_storage(Path(__file__).resolve().parents[2])
