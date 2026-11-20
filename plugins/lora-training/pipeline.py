"""Caption a folder with the local Gemma GGUF, then train an Anima LoRA natively.

Usage: pipeline.py JOB_DIR   (JOB_DIR/request.json is written by jobs.mjs)

Progress goes to JOB_DIR/status.json; llama-server output to JOB_DIR/llama.log and
trainer output to JOB_DIR/train.log. The caller cancels by killing the process group.
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
# Concurrent caption requests. Decoding a 31B model is memory-bandwidth bound, so
# llama-server batches the slots' tokens almost for free (docs/gemma-caption-native-plan.md).
PARALLEL = 16
SLOT_CONTEXT = 1024  # image (<=512 tokens) + instruction + up to 256 caption tokens
JOB = REQUEST = None
status = {"stage": "starting", "done": 0, "total": 0, "started": time.time()}


def update(**changes):
    status.update(changes, updated=time.time())
    with tempfile.NamedTemporaryFile("w", dir=JOB, delete=False, encoding="utf-8") as temporary:
        json.dump(status, temporary, ensure_ascii=False)
    os.replace(temporary.name, JOB / "status.json")


def vision_config():
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = Path(os.environ.get("COMFY_FOLDER_LORA_CONFIG") or base / "comfy-folder-lora/config.json")
    vision = json.loads(path.read_text(encoding="utf-8")).get("vision") or {}
    for key in ("server", "model", "projector"):
        if not vision.get(key) or not Path(vision[key]).is_file():
            raise RuntimeError(f"Gemma のキャプション設定がありません: {path} の vision.{key}")
    return vision


def call(base, key, path, body=None, timeout=10):
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def start_llama(vision, log_dir, parallel=PARALLEL, extra=()):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = os.environ.copy()
    if vision.get("libraryPath"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [vision["libraryPath"], env.get("LD_LIBRARY_PATH")]))
    command = [vision["server"], "-m", vision["model"], "--mmproj", vision["projector"], "--host", "127.0.0.1",
               "--port", str(port), "--api-key", key, "--alias", "ortenzya", "-c", str(SLOT_CONTEXT * parallel), "-np", str(parallel),
               # Every image differs, so a prompt cache only costs time swapping ~670 MiB entries.
               "-ngl", "all", "--reasoning", "off", "--cache-ram", "0", *extra]
    log = (Path(log_dir) / "llama.log").open("ab")
    process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Gemma の起動に失敗しました。llama.log を確認してください。")
        try:
            call(base, key, "/health", timeout=2)
            return process, base, key
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    process.terminate()
    raise RuntimeError("Gemma の起動がタイムアウトしました。")


VIDEOS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def video_frame(path):
    """The middle frame of a video (H3 training clips are captioned from one still)."""
    import av
    with av.open(str(path)) as container:
        middle = (container.streams.video[0].frames or 1) // 2
        for index, frame in enumerate(container.decode(video=0)):
            if index >= middle:
                return frame.to_image()
    return frame.to_image()


def image_data_url(path):
    with (video_frame(path) if Path(path).suffix.lower() in VIDEOS else Image.open(path)) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        data = BytesIO()
        image.convert("RGB").save(data, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def caption(base, key, path, instruction):
    prompt = ("Write only a short, factual English diffusion-training caption (one or two sentences). "
              + instruction)
    body = {"model": "ortenzya", "temperature": 0.2, "max_tokens": 256, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_data_url(path)}}, {"type": "text", "text": prompt}]}]}
    choice = call(base, key, "/v1/chat/completions", body, timeout=240)["choices"][0]
    text = " ".join((choice["message"].get("content") or "").split())
    if not text or choice.get("finish_reason") == "length":
        raise RuntimeError(f"キャプションが不完全です: {path.name}")
    return text


def with_trigger(text, trigger):
    if not trigger or text == trigger or re.match(re.escape(trigger) + r"[,\s.:]", text):
        return text
    return f"{trigger}, {text}"


def write_caption(path, text):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".caption-", delete=False, encoding="utf-8") as tmp:
        tmp.write(text.strip() + "\n")
    os.replace(tmp.name, path)


def caption_all(base, key, paths, instruction, trigger, on_caption, parallel=PARALLEL):
    """Caption `paths` concurrently; `on_caption(path, text)` runs in this thread, in completion order."""
    with ThreadPoolExecutor(parallel) as pool:
        futures = {pool.submit(caption, base, key, path, instruction): path for path in paths}
        for future in as_completed(futures):
            on_caption(futures[future], with_trigger(future.result(), trigger))


def caption_folder():
    images = [Path(p) for p in REQUEST["images"]]
    trigger = REQUEST["trigger"]
    pending = [p for p in images if REQUEST["overwrite"] or not p.with_suffix(".txt").is_file()
               or not p.with_suffix(".txt").read_text(encoding="utf-8").strip()]
    existing = [p.with_suffix(".txt") for p in pending if p.with_suffix(".txt").is_file()]
    if existing:
        backup = Path(REQUEST["folder"]) / f".caption-backup-{time.strftime('%Y%m%d-%H%M%S')}"
        backup.mkdir()
        for path in existing:
            shutil.copy2(path, backup / path.name)
        status["backup"] = str(backup)
    update(stage="captioning", done=0, total=len(pending), message="Gemma を起動しています")
    if pending:
        process, base, key = start_llama(vision_config(), JOB)
        try:
            update(message="キャプションを作成中")
            done = 0

            def save(path, text):
                nonlocal done
                write_caption(path.with_suffix(".txt"), text)
                done += 1
                update(done=done, last_caption=text[:300])

            caption_all(base, key, pending, REQUEST["instruction"], trigger, save)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    # Images that kept their own caption still need the trigger word.
    if trigger:
        for path in images:
            sidecar = path.with_suffix(".txt")
            text = sidecar.read_text(encoding="utf-8").strip()
            if text and with_trigger(text, trigger) != text:
                write_caption(sidecar, with_trigger(text, trigger))


def train():
    metrics = JOB / "metrics.json"
    model = REQUEST.get("model", "anima")
    env = None
    if model == "anima":
        command = [str(ROOT / "dsh"), "anima", "train", "--folder", REQUEST["folder"], "--output", REQUEST["output"],
                   "--metrics", str(metrics), "--summary-only", "--log-every", "5",
                   "--cache-dir", str(ROOT / ".runtime/anima-native/cache")]
    else:  # MiniMax H3 / P-H3: plugins/media-shared/native/train_lora.py in ComfyUI's venv (ninja from this venv)
        command = [str(ROOT / ".runtime/comfy-venv/bin/python"), str(ROOT / "plugins/media-shared/native/train_lora.py"),
                   "--model", model, "--folder", REQUEST["folder"], "--output", REQUEST["output"],
                   "--metrics", str(metrics), "--summary-only", "--log-every", "1"]
        env = {**os.environ, "PATH": str(Path(sys.executable).parent) + ":" + os.environ.get("PATH", "")}
    for flag, value in REQUEST["train_args"].items():
        command += [f"--{flag}", str(value)]
    update(stage="training", done=0, total=REQUEST["steps"], message="学習を準備中（素材の変換）")
    with (JOB / "train.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", env=env)
        for line in process.stdout:
            log.write(line)
            log.flush()
            match = re.match(r"step (\d+)/(\d+) loss=(\S+) elapsed_ms=(\S+)", line)
            if match:
                update(done=int(match[1]), total=int(match[2]), loss=float(match[3]),
                       step_ms=float(match[4]), message="学習中")
            elif line.startswith("Training "):
                update(message=line.strip())
        if process.wait():
            raise RuntimeError("学習に失敗しました: " + (JOB / "train.log").read_text(encoding="utf-8")[-600:])
    summary = json.loads(metrics.read_text(encoding="utf-8"))
    return {k: summary.get(k) for k in ("steps", "images", "training_seconds", "median_step_ms_after_first",
                                        "first_loss", "last_loss", "training_peak_vram_bytes")}


if __name__ == "__main__":
    JOB = Path(sys.argv[1])
    REQUEST = json.loads((JOB / "request.json").read_text(encoding="utf-8"))
    try:
        caption_folder()
        result = train()
        update(stage="done", message="完了", result=result, output=REQUEST["output"], finished=time.time())
    except Exception as error:  # report every failure to the panel
        update(stage="error", message=str(error), finished=time.time())
        sys.exit(1)
