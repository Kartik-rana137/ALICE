"""llama-server process management and a streaming OpenAI-compatible chat client (stdlib only)."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePath
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
        if p.template_path:
            cmd += ["--chat-template-file", str(p.template_path)]
        if config.PROMPT_CACHE:
            config.PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cmd += ["--slot-save-path", str(config.PROMPT_CACHE_DIR)]
        cmd += list(p.extra_args)
        return cmd

    def start(self, wait_s: int = 600) -> bool:
        tpl = self.profile.template_path
        if tpl and not tpl.exists():
            raise LLMError(f"chat template not found: {tpl} (profile '{self.profile.name}', chat_template)")
        return super().start(wait_s)


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
                tools: list[dict] | None = None, max_tokens: int | None = None,
                extra: dict | None = None) -> Iterator[Event]:
    """Stream the assistant reply for `messages` as Events. `tools` are OpenAI-style function specs.
    `extra` is merged into the request (e.g. response_format, temperature overrides)."""
    sampling = profile.thinking_sampling if thinking else profile.sampling
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens or (profile.thinking_max_tokens if thinking else profile.max_tokens),
        **sampling.as_dict(),
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if profile.logit_bias:
        payload["logit_bias"] = [[s, b] for s, b in profile.logit_bias]
    payload.update(extra or {})
    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = True
    with _post("/v1/chat/completions", payload) as resp:
        yield from iter_sse_events(resp)


def complete_json(messages: list[dict], profile: Profile, schema: dict, max_tokens: int = 1024,
                  temperature: float = 0.2) -> dict:
    """One non-thinking completion constrained to a JSON schema (llama-server grammar), parsed."""
    text = "".join(ev.text for ev in chat_stream(
        messages, profile, max_tokens=max_tokens,
        extra={"temperature": temperature, "presence_penalty": 0.0,
               "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}})
        if ev.kind == "content")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned invalid JSON ({e.msg})") from None


def _post_json(path: str, payload: dict, timeout: float = 600.0) -> dict:
    with _post(path, payload, timeout=timeout) as r:
        return json.load(r)


PREFIX_FILES_KEPT = 2          # saved prefix states are ~250 MB each


def prime_prefix(messages: list[dict], profile: Profile, tools: list[dict] | None, marker: str,
                 cache_dir: Path, post=_post_json) -> dict:
    """Restore the processed state of the prompt's fixed part from disk, or compute and save it.

    The fixed part is everything the chat template renders before `marker` (the first text in the system
    prompt that can change between sessions, e.g. today's date): tool schemas, persona, tool guidance -
    ~3k tokens, ~19 s to process on the dev GPU, ~0.3 s to restore (more from a hard disk). The file name is a
    hash of the model and the exact tokens, so any change to the persona, tools, template or model gets a
    fresh file. Ends with the server's cache holding exactly those tokens; returns what happened."""
    if not any(m["role"] == "user" for m in messages):          # Qwen's template refuses a prompt without one;
        messages = [*messages, {"role": "user", "content": "hi"}]  # it comes after the marker, so no effect
    rendered = post("/apply-template", {"messages": messages, **({"tools": tools} if tools else {}),
                                        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    cut = rendered.find(marker)
    if cut <= 0:
        return {"status": "no-marker"}
    tokens = post("/tokenize", {"content": rendered[:cut], "add_special": False})["tokens"]
    key = hashlib.sha256(json.dumps([profile.gguf, profile.ctx, tokens]).encode()).hexdigest()[:20]
    name = f"prefix-{key}.bin"
    path = cache_dir / name
    t0 = time.monotonic()
    if path.exists():
        try:
            r = post("/slots/0?action=restore", {"filename": name})
            if r.get("n_restored") == len(tokens):
                os.utime(path)                               # most recently used: kept by the pruning below
                return {"status": "restored", "tokens": len(tokens), "seconds": round(time.monotonic() - t0, 2)}
        except LLMError:
            pass                                             # unreadable or mismatched file: compute again
    post("/completion", {"prompt": tokens, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
    post("/slots/0?action=save", {"filename": name})
    for old in sorted(cache_dir.glob("prefix-*.bin"), key=lambda f: f.stat().st_mtime, reverse=True)[PREFIX_FILES_KEPT:]:
        old.unlink(missing_ok=True)
    return {"status": "saved", "tokens": len(tokens), "seconds": round(time.monotonic() - t0, 2)}


def warm_up(messages: list[dict], profile: Profile, tools: list[dict] | None = None) -> dict:
    """Evaluate a prompt prefix (tools + system prompt) ahead of the first question, so llama-server has
    it cached when the student asks. Generates one token. Returns the server timings."""
    timings: dict = {}
    for ev in chat_stream([*messages, {"role": "user", "content": "hi"}], profile, tools=tools, max_tokens=1):
        if ev.kind == "finish":
            timings = ev.timings
    return timings
