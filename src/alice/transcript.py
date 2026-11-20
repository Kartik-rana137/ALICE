"""Session transcripts as JSONL (local only, gitignored). Raw material for evaluation and the thesis."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class Transcript:
    def __init__(self, directory: Path, **session_info):
        self._lock = threading.Lock()             # compaction / memory events come from a background thread
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / time.strftime("%Y-%m-%d_%H%M%S.jsonl")
        n = 1
        while self.path.exists():                      # two sessions in the same second
            self.path = directory / time.strftime(f"%Y-%m-%d_%H%M%S_{n}.jsonl")
            n += 1
        self._write({"type": "session", "started": _now(), **session_info})

    def _write(self, record: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def user(self, text: str) -> None:
        self._write({"type": "user", "ts": _now(), "content": text})

    def assistant(self, text: str, **meta) -> None:
        self._write({"type": "assistant", "ts": _now(), "content": text, **meta})

    def event(self, name: str, **meta) -> None:
        self._write({"type": "event", "ts": _now(), "name": name, **meta})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
