"""llama-server process management and a streaming OpenAI-compatible chat client (stdlib only)."""
from __future__ import annotations

import atexit
import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Iterable, Iterator

from . import config
from .config import Profile

BASE = f"http://{config.HOST}:{config.PORT}"


def server_healthy(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


class LlamaServer:
    """Starts llama-server for a profile, or reuses one that is already running."""

    def __init__(self, profile: Profile):
        self.profile = profile
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        p = self.profile
        cmd = [str(config.SERVER_EXE), "-m", str(p.path), "-ngl", str(p.ngl), "-c", str(p.ctx),
               "--host", config.HOST, "--port", str(config.PORT), "--jinja"]
        if p.n_cpu_moe:
            cmd += ["--n-cpu-moe", str(p.n_cpu_moe)]
        return cmd

    def start(self, wait_s: int = 600) -> bool:
        """Return True if we started a server, False if one was already running."""
        if server_healthy():
            return False
        if not config.SERVER_EXE.exists():
            raise FileNotFoundError(f"llama-server not found at {config.SERVER_EXE}")
        if not self.profile.path.exists():
            raise FileNotFoundError(f"model not found: {self.profile.path}")
        config.LOG_DIR.mkdir(exist_ok=True)
        log = open(config.LOG_DIR / "llama-server.log", "w")
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.stop)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited (code {self.proc.returncode}); see logs/llama-server.log")
            if server_healthy():
                return True
            time.sleep(1.5)
        raise TimeoutError("llama-server did not become ready in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def iter_sse_content(lines: Iterable[bytes | str]) -> Iterator[str]:
    """Yield text deltas from an OpenAI-style server-sent-events stream."""
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                yield delta


def chat_stream(messages: list[dict], profile: Profile) -> Iterator[str]:
    """Stream the assistant reply for `messages` as text deltas."""
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": profile.max_tokens,
        "temperature": profile.temperature,
        "chat_template_kwargs": {"enable_thinking": False},   # latency; Phase 0 ran with thinking off
    }
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        yield from iter_sse_content(resp)
