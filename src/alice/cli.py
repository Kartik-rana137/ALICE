"""ALICE command line.

  alice tui [--model NAME] [--think] [--workspace DIR] [--no-tools] [--offline]   chat box (full-screen UI)
  alice [--model NAME] [--think] [--workspace DIR] [--no-tools] [--offline]       line-based chat
  alice ingest PATH [PATH ...] [--title T]                                         add papers/notes to the library
  alice library [--remove ID | --retitle ID TITLE | --rebuild]                     manage the library
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .llm import Event, LLMError
from .session import HELP, ChatSession, StreamFilter, TurnOutcome
from .tools import Tool, ToolResult


def _reconfigure_utf8() -> None:
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _short(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def tool_summary(name: str, raw_args: str) -> str:
    """One-line description of a tool call for display."""
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        args = raw_args
    shown = next(iter(args.values()), "") if isinstance(args, dict) and len(args) == 1 else args
    return _short(shown if name != "run_python" else "(code below)")


def chat_arguments(prog: str = "alice") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="ALICE: local CS/AI mentor "
                                 "(subcommands: tui, ingest, library)")
    ap.add_argument("--model", choices=sorted(config.PROFILES), default=config.DEFAULT_PROFILE,
                    help="model profile (alice.toml [models.*])")
    ap.add_argument("--think", action="store_true", help="start with thinking mode on")
    ap.add_argument("--show-thinking", action="store_true", help="print the model's reasoning text")
    ap.add_argument("--workspace", default=".", help="folder the file tools may read (default: current dir)")
    ap.add_argument("--no-tools", action="store_true", help="disable all tools")
    ap.add_argument("--offline", action="store_true", help="disable tools that use the internet (arXiv)")
    ap.add_argument("--no-transcript", action="store_true", help="don't save this session to logs/transcripts")
    return ap


def session_from_args(args) -> ChatSession:
    return ChatSession(config.PROFILES[args.model], workspace=args.workspace, tools=not args.no_tools,
                       offline=args.offline, transcript=not args.no_transcript, thinking=args.think)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("ingest", "library"):
        _reconfigure_utf8()
        from . import library_cli
        return library_cli.main(argv)
    if argv and argv[0] == "tui":
        from . import tui
        return tui.main(argv[1:])
    args = chat_arguments().parse_args(argv)
    _reconfigure_utf8()

    session = session_from_args(args)
    for note in session.startup_notes:
        print(note)
    profile = session.profile
    print(f"Loading {profile.gguf} ({profile.note}) ...", flush=True)
    try:
        started = session.start()
    except (LLMError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        session.stop()
        return 1
    print("ALICE is ready." + ("" if started else " (reusing running server)")
          + f"  thinking: {'on' if session.thinking else 'off'}"
          + f"  tools: {', '.join(session.tool_names) or 'off'}")
    if session.library:
        print(f"Library: {session.library_size} documents (search runs locally)")
    print(HELP + "   (Ctrl+C stops a reply)\n")

    try:
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            cmd = session.command(text)
            if cmd is not None:
                if cmd.quit:
                    break
                print(cmd.message)
                continue
            try:
                _respond(session, text, args)
            except KeyboardInterrupt:                # e.g. during library retrieval, before streaming
                session.convo.pop_turn()
                print("\n(stopped; this exchange was not kept)\n")
    finally:
        session.stop()
        if session.transcript:
            print(f"(transcript saved to {session.transcript.path.relative_to(config.ROOT)})")
    return 0


class _Printer:
    """Renders streamed events: a thinking indicator, the answer text, tool activity.
    Leaked "<tool_call>" markup is held back and dropped (see session.StreamFilter)."""

    def __init__(self, show_thinking: bool, prompt: str = ""):
        self.show_thinking = show_thinking
        self.in_reasoning = False
        self.reasoning_chars = 0
        self.at_line_start = False
        self.filter = StreamFilter()
        self.prompt = prompt
        self.begun = False

    def begin(self) -> None:
        """Print the 'alice> ' prefix once, just before the first output."""
        if not self.begun:
            self.begun = True
            if self.prompt:
                print(self.prompt, end="", flush=True)

    def _out(self, text: str) -> None:
        if text:
            self.begin()
            print(text, end="", flush=True)
            self.at_line_start = text.endswith("\n")

    def flush(self) -> None:
        self._out(self.filter.flush())

    def event(self, ev: Event) -> None:
        if ev.kind == "reasoning":
            self.begin()
            if not self.in_reasoning:
                self.in_reasoning = True
                print("(thinking" + (":\n" if self.show_thinking else ""), end="", flush=True)
            before = self.reasoning_chars
            self.reasoning_chars += len(ev.text)
            if self.show_thinking:
                print(ev.text, end="", flush=True)
            elif self.reasoning_chars // 400 > before // 400:
                print(".", end="", flush=True)
        elif ev.kind == "content":
            self._end_reasoning()
            self._out(self.filter.feed(ev.text))
        elif ev.kind in ("tool_calls", "finish"):
            self._end_reasoning()
            self.flush()

    def _end_reasoning(self) -> None:
        if self.in_reasoning:
            self.in_reasoning = False
            print(")\n", end="", flush=True)

    def newline(self) -> None:
        if not self.at_line_start:
            print()
            self.at_line_start = True

    def tool_start(self, name: str, raw_args: str) -> None:
        self.begin()
        self.newline()
        print(f"  [{name}] {tool_summary(name, raw_args)}", flush=True)

    def tool_done(self, tr: ToolResult) -> None:
        mark = "declined" if tr.declined else ("ok" if tr.ok else "failed")
        print(f"  [{tr.name} {mark}] {_short(tr.output, 160)}", flush=True)


def _confirm(tool: Tool, args: dict) -> bool:
    print(f"\n  ALICE wants to run {tool.name}:")
    code = args.get("code", json.dumps(args, indent=2))
    print("  " + "\n  ".join(str(code).splitlines()))
    try:
        return input("  Run it? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _respond(session: ChatSession, text: str, args) -> TurnOutcome:
    printer = _Printer(args.show_thinking, prompt="alice> ")
    outcome = session.ask(text, on_event=printer.event, on_tool=printer.tool_done,
                          on_tool_start=printer.tool_start, confirm=_confirm,
                          on_library=lambda places: print("  [library] " + "; ".join(places)))
    printer.begin()
    if outcome.error:
        printer.newline()
        print(f"(error: {outcome.error})\n")
        return outcome
    if outcome.interrupted or not outcome.result or not outcome.result.reply:
        printer.newline()
        print("\n".join(n.text for n in outcome.notices) or "(no reply)")
        print()
        return outcome
    result = outcome.result
    printer.flush()
    if result.finish == "tool_loop":                 # nothing usable was printed: show the fallback
        printer.newline()
        print(result.reply, end="")
    printer.newline()
    print(turn_stats(result))
    if result.finish == "length" and session.thinking and not result.reply:
        print("(the reasoning used the whole budget; try again or turn /think off)")
    for n in outcome.notices:
        print(n.text)
    print()
    return outcome


def turn_stats(result) -> str:
    tps = result.timings.get("predicted_per_second")
    tools_used = f" · tools: {', '.join(t.name for t in result.tool_results)}" if result.tool_results else ""
    return (f"[{result.seconds:.1f}s · {result.gen_tokens} tok" + (f" · {tps:.0f} tok/s" if tps else "")
            + tools_used + "]")


if __name__ == "__main__":
    raise SystemExit(main())
