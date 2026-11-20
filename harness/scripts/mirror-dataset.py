#!/usr/bin/env python3
"""Back up or restore the shared dataset on Hugging Face."""
import argparse
from collections import Counter
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from restore import ROOT, SUPPORT, digest, download_file, merge_directory
from storage_layout import LEGACY_DATASET, LEGACY_OUTPUTS, rebase_file, rebased_link


REPO = "darask0/darask-dataset"
LARGE = 128 * 1024**2
SHARD_SIZE = 2 * 1024**3
OMIT_NAMES = {
    ".git", ".debris", "cache", ".cache", "uvcache", "__pycache__", "node_modules", ".aws", ".secrets",
    ".codex", ".agents", ".ssh", ".gnupg", ".netrc", ".npmrc", ".pypirc", ".credentials.yaml",
    "auth.json", "credentials.json", "token", "secrets.json",
}
TEXT_SUFFIXES = {
    "", ".txt", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini",
    ".conf", ".cfg", ".csv", ".log", ".py", ".js", ".sh", ".xml", ".ipynb",
}
CONFIG_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg", ".py", ".sh", ".mjs", ".js", ".service"}
SECRET = re.compile(
    rb"(sk-(?:proj-|ant-)[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}"
    rb"|hf_[A-Za-z0-9]{30,}|AIza[A-Za-z0-9_-]{35}|AKIA[A-Z0-9]{16}"
    rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    rb"|eyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})"
)


def excluded(path):
    return (
        path.name in OMIT_NAMES or path.name.startswith((".env", ".venv"))
        or path.suffix in {".pyc", ".pem", ".key", ".sock", ".p12"}
    )


def contains_secret(path):
    with path.open("rb") as file:
        tail = b""
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            if SECRET.search(tail + chunk):
                return True
            tail = chunk[-512:]
    return False


def walk(source, omitted):
    def raise_error(error):
        raise error

    for folder, dirs, files in os.walk(source, onerror=raise_error):
        root = Path(folder)
        recurse = []
        for name in sorted(dirs + files):
            path = root / name
            relative = path.relative_to(source)
            if excluded(relative):
                omitted["cache_or_credentials"] += 1
            elif path.is_symlink():
                yield "link", path, relative
            elif path.is_dir():
                recurse.append(name)
                yield "directory", path, relative
            elif path.is_file():
                if path.suffix.lower() in TEXT_SUFFIXES and contains_secret(path):
                    omitted["credential_pattern"] += 1
                else:
                    yield "file", path, relative
            else:
                omitted["special_file"] += 1
        dirs[:] = recurse


def file_entry(path, name, checksum=None):
    return {"path": name, "size": path.stat().st_size, "sha256": checksum or digest(path)}


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM):
            raise
        shutil.copy2(source, destination)


def tar_metadata(entry):
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = ""
    return entry


def pack(source, stage):
    source = source.resolve(strict=True)
    if not source.is_dir() or stage.resolve().is_relative_to(source):
        raise ValueError("Choose a source directory and a staging directory outside it")
    stage.mkdir(parents=True)
    (stage / "objects").mkdir()
    (stage / "archives").mkdir()
    models = json.loads((SUPPORT / "snapshot.json").read_text())["models"]
    model_by_hash = {file["sha256"]: file for file in models["files"]}
    omitted = Counter()
    manifest = {
        "version": 1, "files": [], "large": [], "links": [], "directories": [],
        "source_root": str(source),
        "model_source": {"repo": models["repo"], "revision": models["revision"]},
    }

    def finish_shard(archive, path):
        archive.close()
        checksum = digest(path)
        destination = path.with_name(f"{checksum}.tar")
        path.replace(destination)
        manifest["files"].append(file_entry(destination, destination.relative_to(stage).as_posix(), checksum))
        print(f"Packed archive {shard_index}: {destination.stat().st_size / 1024**3:.2f} GiB", flush=True)

    archive = None
    shard_index = 0
    file_count = 0
    try:
        for kind, path, relative in walk(source, omitted):
            if kind == "link":
                target = path.readlink()
                if target.is_absolute() and target.is_relative_to(source):
                    target = Path(os.path.relpath(target, path.parent))
                manifest["links"].append([relative.as_posix(), str(target)])
                continue
            if kind == "directory":
                manifest["directories"].append(relative.as_posix())
                continue
            initial = path.stat()
            size = initial.st_size
            if size >= LARGE:
                checksum = digest(path)
                existing_model = model_by_hash.get(checksum)
                if existing_model:
                    entry = dict(existing_model, source="model")
                else:
                    name = f"objects/{checksum}.bin"
                    dest = stage / name
                    if not dest.exists():
                        link_or_copy(path, dest)
                        manifest["files"].append(file_entry(dest, name, checksum))
                    entry = {"path": name, "sha256": checksum, "size": size, "source": "dataset"}
                manifest["large"].append(dict(entry, destination=relative.as_posix()))
                print(f"Indexed large object {len(manifest['large'])}: {size / 1024**3:.2f} GiB", flush=True)
            else:
                if archive and archive.offset + size > SHARD_SIZE:
                    finish_shard(archive, shard_path)
                    archive = None
                if archive is None:
                    shard_index += 1
                    shard_path = stage / f"archives/{shard_index:04d}.tar"
                    archive = tarfile.open(shard_path, "w")
                archive.add(path, arcname=relative.as_posix(), recursive=False, filter=tar_metadata)
            current = path.stat()
            if (initial.st_ino, initial.st_mtime_ns, size) != (current.st_ino, current.st_mtime_ns, current.st_size):
                raise RuntimeError(f"Source changed while being read: {relative}")
            file_count += 1
        if archive:
            finish_shard(archive, shard_path)
    finally:
        if archive and not archive.closed:
            archive.close()
    manifest["omitted"] = dict(omitted)
    manifest["file_count"] = file_count
    (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    amount = sum(item["size"] for item in manifest["files"])
    borrowed = sum(item["size"] for item in manifest["large"] if item["source"] == "model")
    print(f"Surveyed {file_count} files; {amount / 1024**3:.2f} GiB to upload, {borrowed / 1024**3:.2f} GiB from the model backup")
    print(f"Omitted: {dict(omitted)}", flush=True)


def require_private(api, repo, revision=None):
    info = api.repo_info(repo, repo_type="dataset", revision=revision)
    if not info.private:
        raise ValueError(f"Expected a private dataset: {repo}")
    return info


def verify_uploaded(api, repo, stage, revision, entries):
    remote = {
        item.path: item for item in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
        if isinstance(item, RepoFile)
    }
    for entry in entries:
        item = remote[entry["path"]]
        if item.lfs:
            checksum = item.lfs.sha256
            expected = entry["sha256"]
        else:
            checksum = item.blob_id
            git_hash = hashlib.sha1(f"blob {entry['size']}\0".encode())
            with (stage / entry["path"]).open("rb") as file:
                expected = hashlib.file_digest(file, lambda: git_hash).hexdigest()
        if item.size != entry["size"] or checksum != expected:
            raise ValueError(f"Uploaded file differs: {entry['path']}")


def push(stage, repo):
    api = HfApi()
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    require_private(api, repo)
    drift = [
        item["path"] for item in manifest["files"]
        if (stage / item["path"]).stat().st_size != item["size"]
    ]
    if drift:
        raise ValueError(f"Staged files changed: {drift[:10]}")
    api.upload_large_folder(
        repo_id=repo, repo_type="dataset", folder_path=stage,
        allow_patterns=["archives/*.tar", "objects/*.bin"], num_workers=6,
        print_report=True, print_report_every=60,
    )
    info = require_private(api, repo)
    verify_uploaded(api, repo, stage, info.sha, manifest["files"])
    commit = api.upload_file(
        repo_id=repo, repo_type="dataset", path_in_repo="manifest.json",
        path_or_fileobj=manifest_path, parent_commit=info.sha,
        commit_message="Publish verified dataset backup manifest",
    )
    verify_uploaded(api, repo, stage, commit.oid, [file_entry(manifest_path, "manifest.json")])
    print(f"Verified {len(manifest['files'])} objects at {commit.commit_url}", flush=True)


def target_path(root, name):
    relative = Path(name)
    destination = root / relative
    if relative.is_absolute() or ".." in relative.parts or not destination.parent.resolve().is_relative_to(root):
        raise ValueError(f"Invalid backup path: {name}")
    return destination


def restore_dataset(repo, revision, destination, cache):
    api = HfApi()
    info = require_private(api, repo, revision=revision)

    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    pointer = hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=info.sha, local_dir=cache, token=True)
    manifest = json.loads(Path(pointer).read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise ValueError("Unsupported dataset backup version")
    source = Path(manifest.get("source_root", LEGACY_DATASET))
    outputs = ROOT / "outputs/comfyui"
    for entry in manifest["files"]:
        if not entry["path"].startswith("archives/"):
            continue
        path = download_file(repo, info.sha, "dataset", entry, cache)
        with TemporaryDirectory(dir=cache) as extracted:
            with tarfile.open(path, "r:") as archive:
                archive.extractall(extracted, filter="data")
            tree = Path(extracted)
            for file in tree.rglob("*"):
                if file.suffix in CONFIG_SUFFIXES and file.is_file() and not file.is_symlink():
                    rebase_file(file, [(source, root), (LEGACY_OUTPUTS, outputs)])
            merge_directory(tree, root)
        path.unlink()
    model_source = manifest["model_source"]
    uses = Counter((entry["source"], entry["path"]) for entry in manifest["large"])
    for entry in manifest["large"]:
        key = (entry["source"], entry["path"])
        uses[key] -= 1
        target = target_path(root, entry["destination"])
        if target.is_symlink():
            raise FileExistsError(target)
        if target.exists():
            if target.stat().st_size != entry["size"] or digest(target) != entry["sha256"]:
                raise FileExistsError(f"Existing data differs from the backup: {target}")
            continue
        if entry["source"] == "model":
            path = download_file(model_source["repo"], model_source["revision"], "model", entry, cache / "models")
        else:
            path = download_file(repo, info.sha, "dataset", entry, cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        if uses[key]:
            shutil.copy2(path, target)
        else:
            link_or_copy(path, target)
            path.unlink()
    for name in manifest["directories"]:
        target_path(root, name).mkdir(parents=True, exist_ok=True)
    for name, link in manifest["links"]:
        target = target_path(root, name)
        link = rebased_link(link, Path(name), source, root, outputs)
        if not (target.is_symlink() and target.readlink() == Path(link)):
            target.symlink_to(link)
    print(f"Restored dataset revision {info.sha} to {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("pack")
    build.add_argument("source", type=Path)
    build.add_argument("stage", type=Path)
    send = commands.add_parser("push")
    send.add_argument("stage", type=Path)
    send.add_argument("--repo", default=REPO)
    restore = commands.add_parser("restore")
    restore.add_argument("destination", type=Path)
    restore.add_argument("--cache", type=Path, required=True)
    restore.add_argument("--revision")
    restore.add_argument("--repo", default=REPO)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.source, args.stage)
    elif args.command == "push":
        push(args.stage, args.repo)
    else:
        restore_dataset(args.repo, args.revision, args.destination, args.cache)


if __name__ == "__main__":
    os.umask(0o077)
    main()
