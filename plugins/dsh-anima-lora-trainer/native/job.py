"""Job harness shared by the native generation scripts (Anima generate/generate.py, MiniMax H3 generate.py, P-H3 ph3.py).

jobs.mjs starts each as `python -I SCRIPT JOB_DIR` with JOB_DIR/request.json; progress goes to JOB_DIR/status.json, which
the panel reads (stage, message, step / steps, done / total, images [{file, seed, ...}]). A script calls
run(main, initial_status); main(job) does the model work. Any exception is printed to stderr (kept by jobs.mjs) and
reported as stage "error" with exit code 1.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback


class Job:
    def __init__(self, directory, status):
        self.dir = Path(directory).resolve()
        self.status = {"stage": "starting", "images": [], "started": time.time(), **status}
        self.request = None
        self._lock = threading.RLock()  # a saver thread and the sampler both report

    def update(self, **changes):
        """Merge `changes` into the status and replace status.json atomically, one writer at a time."""
        with self._lock:
            self.status.update(changes, updated=time.time())
            with tempfile.NamedTemporaryFile("w", dir=self.dir, delete=False, encoding="utf-8") as temporary:
                json.dump(self.status, temporary, ensure_ascii=False)
            os.replace(temporary.name, self.dir / "status.json")

    def add_image(self, record):
        """A finished output file ({file, seed, ...}); done counts them."""
        with self._lock:
            self.status["images"].append(record)
            self.update(done=len(self.status["images"]))


def run(main, status=None):
    job = Job(sys.argv[1], status or {})
    try:
        job.request = json.loads((job.dir / "request.json").read_text(encoding="utf-8"))
        main(job)
    except Exception as error:  # reported to the panel; the traceback goes to stderr for jobs.mjs
        traceback.print_exc()
        job.update(stage="error", message=str(error), finished=time.time())
        sys.exit(1)
