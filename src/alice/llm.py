"""llama-server process management and a streaming OpenAI-compatible chat client (stdlib only)."""
from __future__ import annotations

import atexit
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Iterable, Iterator, Literal

from . import config
from .config import Profile

BASE = f"http://{config.HOST}:{config.PORT}"


class LLMError(RuntimeError):
    """The server could not be reached or rejected a request (message is user-readable)."""


def _get_json(path: str, timeout: float = 5.0, base: str = BASE) -> dict:
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as r:
        return json.load(r)


def _post(path: str, payload: dict, timeout: float = 600.0, base: str = BASE):
    req = urllib.request.Request(f"{base}{path}", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        raise LLMError(f"server returned {e.code}: {msg or e.reason}") from None
    except (urllib.error.URLError, OSError) as e:
        raise LLMError(f"cannot reach llama-server at {base}: {e}") from None


def server_healthy(timeout: float = 2.0, base: str = BASE) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def loaded_model_name(base: str = BASE) -> str | None:
    """File name of the model the running server has loaded, if it reports one."""
    try:
        path = _get_json("/props", base=base).get("model_path")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return PurePath(path.replace("\\", "/")).name if path else None


class ServerProcess:
    """A llama-server child process on one port; reuses a running server if it has the same model."""

    def __init__(self, command: list[str], port: int, model_path, log_name: str, env: dict | None = None):
        self._command = command
        self.port = port
        self.base = f"http://{config.HOST}:{port}"
        self.model_path = model_path
        self.log_name = log_name
        self.env = env
        self.proc: subprocess.Popen | None = None
        self._log = None

    def command(self) -> list[str]:
        return list(self._command)

    def start(self, wait_s: int = 600) -> bool:
        """Return True if we started a server, False if a matching one was already running."""
        if server_healthy(base=self.base):
            running = loaded_model_name(self.base)
            if running and running != self.model_path.name:
                raise LLMError(f"port {self.port} is already serving {running}, not {self.model_path.name}. "
                               "Stop that server or change the port (ALICE_PORT / ALICE_EMBED_PORT).")
            return False
        if not config.SERVER_EXE.exists():
            raise LLMError(f"llama-server not found at {config.SERVER_EXE}")
        if not self.model_path.exists():
            raise LLMError(f"model not found: {self.model_path} (see scripts/download_models.sh)")
        config.LOG_DIR.mkdir(exist_ok=True)
        self._log = open(config.LOG_DIR / self.log_name, "w")
        self.proc = subprocess.Popen(self.command(), stdout=self._log, stderr=subprocess.STDOUT, env=self.env)
        atexit.register(self.stop)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise LLMError(f"llama-server exited (code {self.proc.returncode}); see logs/{self.log_name}")
            if server_healthy(base=self.base):
                return True
            time.sleep(1.0)
        raise LLMError("llama-server did not become ready in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log and not self._log.closed:
            self._log.close()


class LlamaServer(ServerProcess):
    """The chat model server for a profile."""

    def __init__(self, profile: Profile):
        self.profile = profile
        super().__init__([], config.PORT, profile.path, "llama-server.log")

    def command(self) -> list[str]:
        p = self.profile
        cmd = [str(config.SERVER_EXE), "-m", str(p.path), "-ngl", str(p.ngl), "-c", str(p.ctx),
               "-np", "1", "--host", config.HOST, "--port", str(config.PORT), "--jinja",
               "--reasoning-budget", str(p.reasoning_budget)]
        if p.n_cpu_moe:
            cmd += ["--n-cpu-moe", str(p.n_cpu_moe)]
        cmd += list(p.extra_args)
        return cmd


def count_tokens(text: str) -> int:
    """Exact token count from the running server's tokenizer."""
    with _post("/tokenize", {"content": text}, timeout=30) as r:
        return len(json.load(r)["tokens"])


@dataclass
class Event:
    kind: Literal["reasoning", "content", "tool_calls", "finish"]
    text: str = ""                                   # delta text, or finish reason for "finish"
    timings: dict = field(default_factory=dict)      # only on "finish"
    calls: list = field(default_factory=list)        # only on "tool_calls": [{id, name, arguments}]


def iter_sse_events(lines: Iterable[bytes | str]) -> Iterator[Event]:
    """Parse an OpenAI-style server-sent-events stream into reasoning/content/tool_calls/finish events.

    Streamed tool calls arrive as fragments keyed by index; they are assembled and emitted as one
    "tool_calls" event just before the "finish" event.
    """
    pending: dict[int, dict] = {}
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
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("reasoning_content"):
            yield Event("reasoning", delta["reasoning_content"])
        if delta.get("content"):
            yield Event("content", delta["content"])
        for tc in delta.get("tool_calls") or []:
            slot = pending.setdefault(tc.get("index", len(pending)), {"id": "", "name": "", "arguments": ""})
            slot["id"] = tc.get("id") or slot["id"]
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""
            slot["arguments"] += fn.get("arguments") or ""
        if choices[0].get("finish_reason"):
            if pending:
                calls = [pending[i] for i in sorted(pending)]
                for n, c in enumerate(calls):
                    c["id"] = c["id"] or f"call_{n}"
                yield Event("tool_calls", calls=calls)
                pending = {}
            yield Event("finish", choices[0]["finish_reason"], chunk.get("timings") or {})


def chat_stream(messages: list[dict], profile: Profile, thinking: bool = False,
                tools: list[dict] | None = None) -> Iterator[Event]:
    """Stream the assistant reply for `messages` as Events. `tools` are OpenAI-style function specs."""
    sampling = profile.thinking_sampling if thinking else profile.sampling
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": profile.thinking_max_tokens if thinking else profile.max_tokens,
        **sampling.as_dict(),
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if profile.logit_bias:
        payload["logit_bias"] = [[s, b] for s, b in profile.logit_bias]
    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = True
    with _post("/v1/chat/completions", payload) as resp:
        yield from iter_sse_events(resp)
