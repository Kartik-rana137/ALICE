"""Chat session logic shared by the line-based CLI (cli.py) and the chat-box UI (tui.py).

Owns the model server, tools, paper library, conversation memory and transcript, and runs one turn
end to end: automatic library retrieval, the unverified-reference guard, the agent loop, and the
post-checks (citations, citation support, calculator grounding). It never prints: front-ends get
callbacks while a reply streams and a TurnOutcome with user-facing notices afterwards.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from . import config, persona
from .agent import TurnResult, run_turn
from .conversation import Conversation
from .llm import Event, LLMError, LlamaServer, count_tokens
from .tools import Tool, ToolBox, ToolResult, calc, default_toolbox
from .transcript import Transcript

HELP = ("Commands: /think on|off  step-by-step reasoning (slower, better at maths/logic)\n"
        "          /tools        list available tools\n"
        "          /reset  clear the conversation   /help   /quit")


@dataclass
class Notice:
    kind: Literal["sources", "warning", "info"]
    text: str


@dataclass
class TurnOutcome:
    result: TurnResult | None = None
    notices: list[Notice] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None

    @property
    def kept(self) -> bool:
        return bool(self.result and self.result.reply and not self.interrupted and not self.error)


@dataclass
class CommandResult:
    message: str = ""
    quit: bool = False
    reset: bool = False


class StreamFilter:
    """Holds back text that could be the start of leaked "<tool_call>" markup and drops everything after it
    (the agent strips it from the stored reply). Feed deltas in, get displayable text out."""

    TAG = "<tool_call>"

    def __init__(self):
        self.pending = ""
        self.suppressed = False

    def feed(self, text: str) -> str:
        if self.suppressed:
            return ""
        buf = self.pending + text
        i = buf.find(self.TAG)
        if i >= 0:
            self.pending, self.suppressed = "", True
            return buf[:i]
        hold = next((k for k in range(min(len(self.TAG) - 1, len(buf)), 0, -1) if self.TAG.startswith(buf[-k:])), 0)
        self.pending = buf[len(buf) - hold:]
        return buf[:len(buf) - hold]

    def flush(self) -> str:
        out = "" if self.suppressed else self.pending
        self.pending = ""
        return out


class ChatSession:
    def __init__(self, profile: config.Profile, workspace: str | None = ".", tools: bool = True,
                 offline: bool = False, transcript: bool = True, thinking: bool = False):
        self.profile = profile
        self.thinking = thinking
        self.startup_notes: list[str] = []
        self._library_size: int | None = None
        self.toolbox: ToolBox | None = None
        self.library = None                      # LibrarySession or None
        if tools:
            self.toolbox, self.library = self._build_tools(workspace, offline)
        self.server = LlamaServer(profile)
        self.convo: Conversation | None = None
        self.transcript: Transcript | None = None
        self._want_transcript = transcript
        self.started_server = False

    # ----- setup
    def _build_tools(self, workspace, offline):
        box = default_toolbox(workspace=Path(workspace) if workspace else None)
        if offline:
            for name in [n for n, t in box.tools.items() if t.external]:
                del box.tools[name]
        from .library import Library, LibrarySession, LlamaEmbedder
        session = None
        if not config.EMBED_GGUF.exists():
            if config.LIBRARY_DIR.joinpath("library.db").exists() and not Library().is_empty():
                self.startup_notes.append(f"Library disabled: embedding model not found at {config.EMBED_GGUF} "
                                          "(see scripts/download_models.sh or [embedding] in alice.toml)")
        else:
            lib = Library()
            problem = lib.embedding_problem()
            if problem:
                self.startup_notes.append(f"Library disabled: {problem}")
            elif not lib.is_empty():
                session = LibrarySession(lib, LlamaEmbedder())          # device from alice.toml
                session.embedder.start_in_background()                   # ~4 s on CPU, hidden behind typing
                box.add(*session.tools())
        return box, session

    @property
    def tool_names(self) -> list[str]:
        return list(self.toolbox.tools) if self.toolbox else []

    @property
    def library_size(self) -> int:
        if self.library and self._library_size is None:      # counted once; the UI asks often
            self._library_size = len(self.library.library.documents())
        return self._library_size or 0

    def start(self) -> bool:
        """Start (or reuse) the model server and set up memory. Raises LLMError / ValueError."""
        self.started_server = self.server.start()
        self.convo = Conversation(persona.system_prompt(tools=self.tool_names), self.profile.ctx,
                                  self.profile.thinking_max_tokens, counter=count_tokens)
        if self._want_transcript:
            self.transcript = Transcript(
                config.TRANSCRIPT_DIR, model=self.profile.gguf, profile=self.profile.name,
                persona_version=persona.PERSONA_VERSION, ctx=self.profile.ctx, thinking=self.thinking,
                tools=sorted(self.tool_names))
        return self.started_server

    def stop(self) -> None:
        self.server.stop()
        if self.library:
            self.library.embedder.stop()

    # ----- commands
    def command(self, text: str) -> CommandResult | None:
        """Handle a /command. Returns None if `text` is not a command."""
        if not text.startswith("/"):
            return None
        if text in ("/quit", "/exit"):
            return CommandResult(quit=True)
        if text == "/help":
            return CommandResult(HELP)
        if text == "/tools":
            if not self.toolbox:
                return CommandResult("(tools are off)")
            return CommandResult("\n".join(f"  {t.name}{' (asks first)' if t.needs_confirmation else ''}"
                                           f"{' (internet)' if t.external else ''}" for t in self.toolbox.tools.values()))
        if text == "/reset":
            self.convo.reset()
            if self.transcript:
                self.transcript.event("reset")
            return CommandResult("(conversation cleared)", reset=True)
        if text.startswith("/think"):
            arg = text.split()[1:] or ["on" if not self.thinking else "off"]
            self.thinking = arg[0].lower() in ("on", "1", "true", "yes")
            if self.transcript:
                self.transcript.event("thinking", on=self.thinking)
            return CommandResult(f"(thinking {'on' if self.thinking else 'off'})")
        return CommandResult("unknown command. " + HELP)

    # ----- one turn
    def ask(self, text: str, *, on_event: Callable[[Event], None] | None = None,
            on_tool_start: Callable[[str, str], None] | None = None,
            on_tool: Callable[[ToolResult], None] | None = None,
            confirm: Callable[[Tool, dict], bool] | None = None,
            on_library: Callable[[list[str]], None] | None = None,
            cancel: threading.Event | None = None) -> TurnOutcome:
        convo = self.convo
        if convo is None:
            raise RuntimeError("ChatSession.ask() called before start()")
        convo.add_user(text)
        if self.transcript:
            self.transcript.user(text)
        prefetched = []
        if self.library:
            prefetched, hits = self.library.prefetch(text)
            if prefetched:
                convo.add(*prefetched)
                if on_library:
                    on_library(sorted({h.where for h in hits}))
        style = persona.pick_opener_style()
        guard = not prefetched and persona.needs_reference_guard(text)
        system = persona.system_prompt(style, tools=self.tool_names, guard=guard)
        out = TurnOutcome()
        try:
            out.result = run_turn(convo.messages(system), self.profile, self.toolbox, thinking=self.thinking,
                                  on_event=on_event, on_tool=on_tool, on_tool_start=on_tool_start,
                                  confirm=confirm, cancel=cancel)
            out.interrupted = out.result.finish == "interrupted"
        except KeyboardInterrupt:
            out.interrupted = True
        except LLMError as e:
            out.error = str(e)
        if not out.kept:
            convo.pop_turn()                         # nothing usable came back: forget the question
            if out.interrupted:
                out.notices.append(Notice("info", "(stopped; this exchange was not kept)"))
            return out

        result = out.result
        citations = self._check_citations(result.reply, out.notices) if self.library else None
        numbers_ok = self._check_calculations(result, out.notices)
        if result.finish == "length":
            out.notices.append(Notice("info", "(reply hit the length limit and was cut off; type 'continue' "
                                              "for the rest)"))
        convo.add(*result.messages)
        if self.transcript:
            self.transcript.assistant(
                result.reply, thinking=self.thinking, opener_style=style, finish=result.finish,
                seconds=result.seconds, gen_tokens=result.gen_tokens, rounds=result.rounds,
                reasoning_chars=result.reasoning_chars,
                tools=[{"name": t.name, "args": t.args, "ok": t.ok, "declined": t.declined,
                        "output": t.output[:2000]} for t in result.tool_results],
                citations=citations, prefetched=bool(prefetched), calculation_grounded=numbers_ok,
                reference_guard=guard)
        return out

    def _check_citations(self, reply: str, notices: list[Notice]) -> dict:
        rep = self.library.check(reply)
        if rep.cited:
            notices.append(Notice("sources", "Sources: " + "; ".join(f"[{l}] {h.where}" for l, h in rep.cited)))
        if rep.unknown:
            notices.append(Notice("warning", f"WARNING: cites {', '.join('[' + l + ']' for l in rep.unknown)} "
                                             "which were never retrieved from your library. Treat those claims "
                                             "as unverified."))
        if rep.unsupported:
            notices.append(Notice("warning", "WARNING: these numbers are not in the passages they cite: "
                                  + "; ".join(f"{num} {labels}" for num, labels in rep.unsupported)
                                  + ". They may come from the model's memory, not your papers."))
        return {"cited": [{"label": l, "doc": h.title, "page": h.page, "chunk": h.chunk_id} for l, h in rep.cited],
                "unknown": rep.unknown, "unsupported": [list(u) for u in rep.unsupported]}

    @staticmethod
    def _check_calculations(result: TurnResult, notices: list[Notice]) -> bool | None:
        """Warn when the final calculator result doesn't appear in the answer (small models mis-copy numbers)."""
        calcs = [t for t in result.tool_results if t.name == "calculate" and t.ok]
        if not calcs:
            return None
        value = calc.last_result_value(calcs[-1].output)
        if value is None or calc.reply_mentions(result.reply, value):
            return True
        notices.append(Notice("warning", f"WARNING: the calculator's final result "
                                         f"({calcs[-1].output.rsplit(' = ', 1)[1]}) does not appear in this "
                                         "answer. Double-check the numbers above."))
        return False
