#!/usr/bin/env python3
"""Restore the pinned DSH backup into this checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"
SUPPORT = ROOT / "scripts/restore"
MODEL_ROOT = ROOT / "plugins/dsh-comfyui/models"
CACHE = RUNTIME / "restore-downloads"
SOURCES = (
    ("ComfyUI", "https://github.com/Comfy-Org/ComfyUI.git",
     "989d93d0d968e704c589ce0146bb18c77f6b50cc", False),
    ("diffusion-pipe", "https://github.com/tdrussell/diffusion-pipe.git",
     "c1159b4893e06a7a77446b5a3f72aa969836b3f9", True),
    ("h3-native/cutlass", "https://github.com/NVIDIA/cutlass.git",
     "d4b4b494c3c51bf6507e7ab09fbafd1e9fa94f39", False),
    ("ComfyUI/custom_nodes/ComfyUI-Anima-Ref-Adapter",
     "https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter.git",
     "f02d54dd34593b4d4ce27b38de2a996049aeac58", False),
)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_file(repo, revision, repo_type, entry, directory):
    path = directory / entry["path"]
    if path.is_symlink():
        raise ValueError(f"Restore destination is a symlink: {path}")
    if not path.exists():
        hf_hub_download(
            repo_id=repo, revision=revision, repo_type=repo_type, token=True,
            filename=entry["path"], local_dir=directory,
        )
    if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
        raise ValueError(f"Backup checksum mismatch: {path}")
    return path


def merge_directory(source, destination):
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=source.stat().st_mode & 0o777)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_symlink():
            if target.is_symlink() and target.readlink() == child.readlink():
                continue
            target.symlink_to(child.readlink())
        elif child.is_dir():
            merge_directory(child, target)
        elif target.is_symlink():
            raise FileExistsError(target)
        elif target.exists():
            if not target.is_file() or digest(child) != digest(target):
                raise FileExistsError(f"Existing data differs from the backup: {target}")
        else:
            shutil.copy2(child, target)


def link_directory(link, target):
    relative = Path(os.path.relpath(target, link.parent))
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        raise FileExistsError(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(relative, target_is_directory=True)


def restore_sources():
    for name, url, revision, submodules in SOURCES:
        destination = RUNTIME / name
        if not (destination / ".git").exists():
            print(f"Restoring {name} at {revision[:12]}", flush=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "--quiet", destination], check=True)
        git = ["git", "-C", destination]
        head = subprocess.run(
            [*git, "rev-parse", "--verify", "HEAD"], capture_output=True, text=True,
        ).stdout.strip()
        if head and head != revision:
            raise ValueError(f"{destination} is at {head}, expected {revision}")
        if not head:
            remote = subprocess.check_output([*git, "remote"], text=True).splitlines()
            if "origin" not in remote:
                subprocess.run([*git, "remote", "add", "origin", url], check=True)
            subprocess.run([*git, "fetch", "--depth", "1", "origin", revision], check=True)
            subprocess.run([*git, "checkout", "--quiet", "--detach", revision], check=True)
        if submodules:
            subprocess.run(
                [*git, "submodule", "update", "--init", "--recursive", "--depth", "1"],
                check=True,
            )


def restore_layout():
    comfy = RUNTIME / "ComfyUI"
    plugin = MODEL_ROOT.parent
    merge_directory(SUPPORT / "legacy/dsh-comfyui", plugin)
    merge_directory(SUPPORT / "legacy/anima_edit_reference", comfy / "custom_nodes/anima_edit_reference")
    # ComfyUI keeps its upstream model placeholders and loads the DSH library under the same model categories.
    config = comfy / "extra_model_paths.yaml"
    if not config.exists():
        shutil.copyfile(SUPPORT / "extra_model_paths.yaml", config)
    link_directory(plugin / "loras", MODEL_ROOT / "loras")
    link_directory(plugin / "workflows", comfy / "user/default/workflows")
    link_directory(plugin / "gemma_candidates", comfy / "output/gemma_candidates")
    link_directory(plugin / "custom_nodes/anima_ref_files", ROOT / "plugins/comfyui/anima_ref_files")
    link_directory(comfy / "custom_nodes/anima_ref_files", ROOT / "plugins/comfyui/anima_ref_files")
    link_directory(RUNTIME / "gguf-caption/models", MODEL_ROOT / "gguf")


def restore_archive(path, restore_home):
    with TemporaryDirectory(prefix="unpack-", dir=CACHE) as temp:
        staging = Path(temp)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(staging, filter="data")
        if (staging / "workspace").exists():
            merge_directory(staging / "workspace", ROOT)
        if restore_home and (staging / "home").exists():
            config_path = staging / "home/.config/comfy-folder-lora/config.json"
            if config_path.is_file():
                configuration = config_path.read_text(encoding="utf-8")
                old_root = json.dumps("/mnt/6tb/daraskDSH")[1:-1]
                new_root = json.dumps(str(ROOT))[1:-1]
                config_path.write_text(configuration.replace(old_root, new_root), encoding="utf-8")
            merge_directory(staging / "home", Path.home())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true", help="Restore data and source dependencies only")
    parser.add_argument("--with-user-state", action="store_true", help="Restore saved DSH settings and history to the home directory")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or later is required")
    snapshot = json.loads((SUPPORT / "snapshot.json").read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    restore_sources()
    restore_layout()
    datasets = snapshot["data"]
    for entry in datasets["files"]:
        if entry["path"] == "user-state.tar.gz" and not args.with_user_state:
            continue
        print(f"Restoring {entry['path']}", flush=True)
        path = download_file(datasets["repo"], datasets["revision"], "dataset", entry, CACHE)
        restore_archive(path, args.with_user_state)
    if not args.skip_models:
        models = snapshot["models"]
        print(f"Downloading and verifying {len(models['files'])} model files", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(download_file, models["repo"], models["revision"], "model", entry, MODEL_ROOT)
                for entry in models["files"]
            ]
            for future in futures:
                print(f"Verified {future.result().relative_to(ROOT)}", flush=True)
    print("DSH backup restored. Run scripts/setup-restored.sh to install and build the environment.")


if __name__ == "__main__":
    main()
