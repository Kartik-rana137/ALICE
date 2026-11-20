"""The GUI's back end: a Starlette app on 127.0.0.1 that drives one ChatSession over a WebSocket.

The page (static/) sends what the student types; replies stream back as events (deltas, tool activity,
notices, run confirmations, the /claude preview). Slow work - a turn, starting the model, preparing or
sending an escalation, running a code block - happens on worker threads and reports through `Hub.send`.

Security: ALICE can run code (with confirmation), so the server listens on 127.0.0.1 only, every request
must carry the random token created at launch (it is in the window's URL), and WebSocket connections must
come from the app's own origin - other programs or web pages on this computer can't drive her.
"""
from __future__ import annotations

import asyncio
import itertools
import re
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .. import config
from ..cli import tool_summary, turn_stats
from ..llm import LLMError
from ..session import HELP, StreamFilter

STATIC = Path(__file__).parent / "static"
UPLOADS = config.ROOT / "data" / "uploads"              # gitignored, like the rest of data/
REASONING_EVERY_S = 0.3


class Hub:
    """The open window's WebSocket, usable from any thread. Messages to a closed window are dropped."""

    def __init__(self):
        self.ws: WebSocket | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def send(self, msg: dict) -> None:
        ws, loop = self.ws, self.loop
        if ws is None or loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(msg), loop)
        except RuntimeError:                              # the loop has shut down
            pass


class Controller:
    """Everything the window can ask for, on top of one ChatSession."""

    def __init__(self, session, autostart: bool = True):
        self.s = session
        self.hub = Hub()
        self.ready = False
        self.start_error: str | None = None
        self.started_at = time.monotonic()
        self.busy = False
        self.phase = ""
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._waits: dict[int, tuple[threading.Event, dict]] = {}
        self._escalations: dict[int, object] = {}
        if autostart:
            threading.Thread(target=self._start, name="alice-start", daemon=True).start()

    # ----- state
    def _start(self) -> None:
        try:
            self.s.start()
            self.ready = True
        except (LLMError, ValueError) as e:
            self.start_error = str(e)
        self.hub.send(self.status())

    def status(self) -> dict:
        s = self.s
        return {"type": "status", "ready": self.ready, "error": self.start_error, "busy": self.busy,
                "phase": self.phase, "elapsed": round(time.monotonic() - self.started_at, 1),
                "model": s.profile.gguf.removesuffix(".gguf"), "thinking": s.thinking,
                "tools": list(s.tool_names), "library": s.library_size if s.library else None,
                "memory": s.memory_size if getattr(s, "memory", None) else None,
                "claude": bool(getattr(s, "claude", None)),
                "document": s.document.title if getattr(s, "document", None) is not None else None,
                "background": getattr(s, "background", None)}

    # ----- messages from the window
    def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self.hub.send({**self.status(), "help": HELP})
        elif kind == "send":
            self._send(str(msg.get("text", "")).strip())
        elif kind == "stop":
            self.cancel.set()
        elif kind in ("confirm_reply", "claude_reply"):
            wait = self._waits.get(int(msg.get("id", 0)))
            if wait:
                wait[1]["ok"] = bool(msg.get("ok"))
                wait[0].set()
        elif kind == "thinking":
            self._info(self.s.command("/think on" if msg.get("on") else "/think off").message)
        elif kind == "reset":
            if not self.busy and self.ready:
                self._info(self.s.command("/reset").message, clear=True)
        elif kind == "run":
            threading.Thread(target=self._run_block, args=(msg,), daemon=True).start()
        self.hub.send(self.status())

    def _info(self, text: str, clear: bool = False) -> None:
        self.hub.send({"type": "info", "text": text, "clear": clear})

    def _send(self, text: str) -> None:
        if not text:
            return
        if not self.ready:
            self._info("ALICE is still loading the model; one moment.")
            return
        if self.busy:
            self._info("ALICE is still answering. Press Stop first.")
            return
        cmd = self.s.command(text)
        if cmd is not None:
            if cmd.quit:
                self._info("(close the window to quit)")
            elif cmd.escalate:
                self._begin("preparing the request for Claude")
                threading.Thread(target=self._escalate, args=(cmd.escalate_question,), daemon=True).start()
            else:
                self._info(cmd.message)
                if cmd.ask:
                    self.hub.send({"type": "user", "text": cmd.ask})
                    self._begin("reading the document")
                    threading.Thread(target=self._turn, args=(cmd.ask,), daemon=True).start()
            return
        self.hub.send({"type": "user", "text": text})
        self._begin("reading your question")
        threading.Thread(target=self._turn, args=(text,), daemon=True).start()

    def _begin(self, phase: str) -> None:
        self.busy, self.phase = True, phase
        self.cancel.clear()
        self.started_at = time.monotonic()
        self.hub.send(self.status())

    def _end(self) -> None:
        self.busy, self.phase = False, ""
        self.hub.send(self.status())

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.hub.send({"type": "phase", "text": phase})

    def _wait_for_reply(self, event: dict) -> bool:
        """Show a dialog in the window and wait for its answer (or Stop / a closed window)."""
        wid = next(self._ids)
        done, answer = threading.Event(), {}
        self._waits[wid] = (done, answer)
        self.hub.send({**event, "id": wid})
        try:
            while not done.wait(0.25):
                if self.cancel.is_set():
                    return False
            return answer.get("ok", False)
        finally:
            self._waits.pop(wid, None)

    # ----- one turn
    def _turn(self, text: str) -> None:
        r = _Relay(self, "alice")
        try:
            if getattr(self.s, "background", None):
                self._set_phase(f"finishing {self.s.background}")
            out = self.s.ask(text, on_event=r.event, on_tool_start=r.tool_start, on_tool=r.tool_done,
                             confirm=self._confirm, on_library=r.library, on_arxiv=r.arxiv, cancel=self.cancel)
            r.flush()
            self.hub.send({"type": "turn_end", "kept": out.kept, "error": out.error,
                           "notices": [{"kind": n.kind, "text": n.text} for n in out.notices],
                           "stats": turn_stats(out.result) if out.result and out.kept else "",
                           "reply": out.result.reply if out.result and out.kept else ""})
        except Exception as e:                            # never leave the window stuck in "busy"
            self.hub.send({"type": "turn_end", "kept": False, "error": f"{type(e).__name__}: {e}", "notices": []})
        finally:
            self._end()

    def _confirm(self, tool, args: dict) -> bool:
        return self._wait_for_reply({"type": "confirm", "tool": tool.name, "language": args.get("language", "python"),
                                     "code": str(args.get("code", "")), "stdin": str(args.get("stdin", "") or "")})

    # ----- escalation to Claude
    def _escalate(self, question: str | None) -> None:
        from ..claude import ClaudeError
        try:
            try:
                req = self.s.prepare_escalation(question)
            except ClaudeError as e:
                self.hub.send({"type": "notice", "kind": "warning", "text": f"(claude: {e})"})
                return
            c = self.s.claude
            month = (f"This month so far: ${c.ledger.month_total():.2f} of ${c.settings.monthly_budget_usd:.2f} "
                     f"(model {c.settings.model}, effort {c.settings.effort}).")
            self._set_phase("waiting for you to confirm")
            if not self._wait_for_reply({"type": "claude_preview", "summary": req.summary(), "month": month,
                                         "preview": req.preview()}):
                self._info("(not sent to Claude)")
                return
            if question:
                self.hub.send({"type": "user", "text": question})
            self._set_phase("waiting for Claude")
            r = _Relay(self, "claude")
            out = self.s.escalate(req, on_text=r.claude_text, on_status=r.claude_status, cancel=self.cancel)
            r.flush()
            self.hub.send({"type": "turn_end", "kept": bool(out.reply and not out.reply.interrupted),
                           "error": out.error, "notices": [{"kind": n.kind, "text": n.text} for n in out.notices],
                           "stats": "", "reply": out.reply.text if out.reply else ""})
        except Exception as e:
            self.hub.send({"type": "notice", "kind": "warning", "text": f"(claude: {type(e).__name__}: {e})"})
        finally:
            self._end()

    # ----- running a code block the student clicked "Run" on
    def _run_block(self, msg: dict) -> None:
        from ..tools.coderun import run_code
        out = run_code(str(msg.get("language", "python")), str(msg.get("code", "")), stdin=str(msg.get("stdin", "")))
        self.hub.send({"type": "run_result", "block": msg.get("block"), "output": out})

    # ----- files dropped on the window
    def save_upload(self, filename: str, data: bytes) -> Path:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w.\- ]", "_", Path(filename).name)[:120] or "document"
        dest = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
        dest.write_bytes(data)
        return dest


class _Relay:
    """Session callbacks (worker thread) -> window events."""

    def __init__(self, ctl: Controller, target: str):
        self.ctl, self.target = ctl, target
        self.filter = StreamFilter()
        self.reasoning, self.last = 0, 0.0
        self.ctl.hub.send({"type": "segment", "target": target})

    def _send(self, msg: dict) -> None:
        self.ctl.hub.send(msg)

    def library(self, places) -> None:
        # only the phase: the sources notice after the reply lists the same passages
        self.ctl._set_phase(f"reading {len(places)} passage{'s' * (len(places) != 1)} from your library")

    def arxiv(self, query: str) -> None:
        self.ctl._set_phase("searching arXiv")
        self._send({"type": "notice", "kind": "sources", "text": f"arXiv: searching for '{query}' (sent to arxiv.org)"})

    def event(self, ev) -> None:
        if ev.kind == "reasoning":
            self.reasoning += len(ev.text)
            if time.monotonic() - self.last > REASONING_EVERY_S:
                self.last = time.monotonic()
                self.ctl._set_phase("thinking")
                self._send({"type": "reasoning", "chars": self.reasoning})
        elif ev.kind == "content":
            if self.ctl.phase != "writing":
                self.ctl._set_phase("writing")
            text = self.filter.feed(ev.text)
            if text:
                self._send({"type": "delta", "text": text, "target": self.target})
        elif ev.kind in ("tool_calls", "finish"):
            self.flush()

    def flush(self) -> None:
        rest = self.filter.flush()
        if rest:
            self._send({"type": "delta", "text": rest, "target": self.target})

    def tool_start(self, name: str, raw_args: str) -> None:
        self.flush()
        self.filter = StreamFilter()
        self.ctl._set_phase(f"using {name}")
        self._send({"type": "tool_start", "name": name, "summary": tool_summary(name, raw_args)})
        self._send({"type": "segment", "target": self.target})            # text after the tool: a new bubble

    def tool_done(self, tr) -> None:
        mark = "declined" if tr.declined else ("ok" if tr.ok else "failed")
        self.ctl._set_phase("reading the result")
        self._send({"type": "tool_done", "name": tr.name, "mark": mark, "output": tr.output[:3000]})

    def claude_status(self, what: str) -> None:
        self.ctl._set_phase("Claude is thinking" if what == "thinking" else what)

    def claude_text(self, text: str) -> None:
        if self.ctl.phase != "Claude is writing":
            self.ctl._set_phase("Claude is writing")
        self._send({"type": "delta", "text": text, "target": self.target})


def create_app(controller: Controller, token: str) -> Starlette:
    def authorised(request) -> bool:
        return request.query_params.get("token") == token

    async def index(request: Request):
        if not authorised(request):
            return Response("ALICE: missing or wrong token. Start the window with `alice gui`.", status_code=403)
        return FileResponse(STATIC / "index.html")

    async def upload(request: Request):
        if not authorised(request):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        form = await request.form()
        f = form.get("file")
        if f is None:
            return JSONResponse({"error": "no file"}, status_code=400)
        path = controller.save_upload(f.filename, await f.read())
        return JSONResponse({"path": str(path)})

    async def ws_endpoint(ws: WebSocket):
        origin = ws.headers.get("origin", "")
        host = ws.headers.get("host", "")
        if ws.query_params.get("token") != token or (origin and origin != f"http://{host}"):
            await ws.close(code=4403)
            return
        await ws.accept()
        controller.hub.ws, controller.hub.loop = ws, asyncio.get_running_loop()
        try:
            while True:
                msg = await ws.receive_json()
                await asyncio.get_running_loop().run_in_executor(None, controller.handle, msg)
        except WebSocketDisconnect:
            pass
        finally:
            if controller.hub.ws is ws:
                controller.hub.ws = None
            controller.cancel.set()                       # a closed window stops a running reply

    return Starlette(routes=[
        Route("/", index),
        Route("/upload", upload, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ])
