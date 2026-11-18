"""ALICE command line.

  alice [--model NAME] [--think] [--workspace DIR] [--no-tools] [--offline]   chat
  alice ingest PATH [PATH ...] [--title T]                                     add papers/notes to the library
  alice library [--remove ID]                                                  list (or remove) library documents
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, persona
from .agent import TurnResult, run_turn
from .conversation import Conversation
from .llm import Event, LLMError, LlamaServer, count_tokens
from .tools import Tool, ToolBox, ToolResult, calc, default_toolbox
from .transcript import Transcript

HELP = ("Commands: /think on|off  step-by-step reasoning (slower, better at maths/logic)\n"
        "          /tools        list available tools\n"
        "          /reset  clear the conversation   /help   /quit   (Ctrl+C stops a reply)")


def _reconfigure_utf8() -> None:
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _short(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def build_toolbox(args):
    """(toolbox or None, library session or None)."""
    if args.no_tools:
        return None, None
    box = default_toolbox(workspace=Path(args.workspace) if args.workspace else None)
    if args.offline:
        for name in [n for n, t in box.tools.items() if t.external]:
            del box.tools[name]
    session = None
    from .library import Library, LibrarySession, LlamaEmbedder
    if not config.EMBED_GGUF.exists():
        if config.LIBRARY_DIR.joinpath("library.db").exists() and not Library().is_empty():
            print(f"Library disabled: embedding model not found at {config.EMBED_GGUF} "
                  "(see scripts/download_models.sh or [embedding] in alice.toml)")
    else:
        lib = Library()
        problem = lib.embedding_problem()
        if problem:
            print(f"Library disabled: {problem}")
        elif not lib.is_empty():
            session = LibrarySession(lib, LlamaEmbedder(gpu=False))
            session.embedder.start_in_background()                   # ~4 s on CPU, hidden behind typing
            box.add(*session.tools())
    return box, session


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("ingest", "library"):
        _reconfigure_utf8()
        from . import library_cli
        return library_cli.main(argv)
    ap = argparse.ArgumentParser(prog="alice", description="ALICE: local CS/AI mentor "
                                 "(subcommands: ingest, library)")
    ap.add_argument("--model", choices=sorted(config.PROFILES), default=config.DEFAULT_PROFILE,
                    help="model profile (see alice/config.py)")
    ap.add_argument("--think", action="store_true", help="start with thinking mode on")
    ap.add_argument("--show-thinking", action="store_true", help="print the model's reasoning text")
    ap.add_argument("--workspace", default=".", help="folder the file tools may read (default: current dir)")
    ap.add_argument("--no-tools", action="store_true", help="disable all tools")
    ap.add_argument("--offline", action="store_true", help="disable tools that use the internet (arXiv)")
    ap.add_argument("--no-transcript", action="store_true", help="don't save this session to logs/transcripts")
    args = ap.parse_args(argv)
    _reconfigure_utf8()

    profile = config.PROFILES[args.model]
    toolbox, library = build_toolbox(args)
    tool_names = list(toolbox.tools) if toolbox else []
    server = LlamaServer(profile)
    print(f"Loading {profile.gguf} ({profile.note}) ...", flush=True)
    try:
        started = server.start()
        convo = Conversation(persona.system_prompt(tools=tool_names), profile.ctx,
                             profile.thinking_max_tokens, counter=count_tokens)
    except (LLMError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        server.stop()
        return 1

    thinking = args.think
    transcript = None if args.no_transcript else Transcript(
        config.TRANSCRIPT_DIR, model=profile.gguf, profile=profile.name,
        persona_version=persona.PERSONA_VERSION, ctx=profile.ctx, thinking=thinking,
        tools=sorted(toolbox.tools) if toolbox else [])
    print("ALICE is ready." + ("" if started else " (reusing running server)")
          + f"  thinking: {'on' if thinking else 'off'}"
          + f"  tools: {', '.join(tool_names) or 'off'}")
    if library:
        print(f"Library: {len(library.library.documents())} documents (search runs locally)")
    print(HELP + "\n")

    try:
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/help":
                print(HELP)
                continue
            if text == "/tools":
                print("\n".join(f"  {t.name}{' (asks first)' if t.needs_confirmation else ''}"
                                f"{' (internet)' if t.external else ''}" for t in toolbox.tools.values())
                      if toolbox else "(tools are off)")
                continue
            if text == "/reset":
                convo.reset()
                if transcript:
                    transcript.event("reset")
                print("(conversation cleared)")
                continue
            if text.startswith("/think"):
                arg = text.split()[1:] or ["on" if not thinking else "off"]
                thinking = arg[0].lower() in ("on", "1", "true", "yes")
                if transcript:
                    transcript.event("thinking", on=thinking)
                print(f"(thinking {'on' if thinking else 'off'})")
                continue
            if text.startswith("/"):
                print("unknown command. " + HELP)
                continue

            convo.add_user(text)
            if transcript:
                transcript.user(text)
            prefetched = []
            if library:
                prefetched, hits = library.prefetch(text)
                if prefetched:
                    convo.add(*prefetched)
                    print("  [library] " + "; ".join(sorted({h.where for h in hits})))
            style = persona.pick_opener_style()
            guard = not prefetched and persona.needs_reference_guard(text)
            system = persona.system_prompt(style, tools=tool_names, guard=guard)
            result = _respond(convo.messages(system), profile, toolbox, thinking, args)
            citations = _report_citations(library, result.reply) if (library and result and result.reply) else None
            numbers_ok = _check_calculations(result) if (result and result.reply) else None
            if result and result.reply:
                convo.add(*result.messages)
                if transcript:
                    transcript.assistant(
                        result.reply, thinking=thinking, opener_style=style, finish=result.finish,
                        seconds=result.seconds, gen_tokens=result.gen_tokens, rounds=result.rounds,
                        reasoning_chars=result.reasoning_chars,
                        tools=[{"name": t.name, "args": t.args, "ok": t.ok, "declined": t.declined,
                                "output": t.output[:2000]} for t in result.tool_results],
                        citations=citations, prefetched=bool(prefetched), calculation_grounded=numbers_ok,
                        reference_guard=guard)
            else:
                convo.pop_turn()                      # nothing usable came back: forget the question
    finally:
        server.stop()
        if library:
            library.embedder.stop()
        if transcript:
            print(f"(transcript saved to {transcript.path.relative_to(config.ROOT)})")
    return 0


def _check_calculations(result: TurnResult) -> bool | None:
    """Warn when the final calculator result doesn't appear in the answer (small models mis-copy numbers)."""
    calcs = [t for t in result.tool_results if t.name == "calculate" and t.ok]
    if not calcs:
        return None
    value = calc.last_result_value(calcs[-1].output)
    if value is None or calc.reply_mentions(result.reply, value):
        return True
    print(f"WARNING: the calculator's final result ({calcs[-1].output.rsplit(' = ', 1)[1]}) does not appear "
          "in this answer. Double-check the numbers above.\n")
    return False


def _report_citations(library, reply: str) -> dict:
    """Print the sources a reply cited and flag labels that were never retrieved."""
    rep = library.check(reply)
    if rep.cited:
        print("Sources: " + "; ".join(f"[{label}] {hit.where}" for label, hit in rep.cited))
    if rep.unknown:
        print(f"WARNING: cites {', '.join('[' + l + ']' for l in rep.unknown)} which were never retrieved "
              "from your library. Treat those claims as unverified.")
    if rep.unsupported:
        print("WARNING: these numbers are not in the passages they cite: "
              + "; ".join(f"{num} {labels}" for num, labels in rep.unsupported)
              + ". They may come from the model's memory, not your papers.")
    if rep.cited or rep.unknown:
        print()
    return {"cited": [{"label": l, "doc": h.title, "page": h.page, "chunk": h.chunk_id} for l, h in rep.cited],
            "unknown": rep.unknown, "unsupported": [list(u) for u in rep.unsupported]}


class _Printer:
    """Renders streamed events: a thinking indicator, the answer text, tool activity.

    Holds back any text that could be the start of leaked "<tool_call>" markup, and prints nothing
    after it (the agent strips it from the stored reply)."""

    TAG = "<tool_call>"

    def __init__(self, show_thinking: bool):
        self.show_thinking = show_thinking
        self.in_reasoning = False
        self.reasoning_chars = 0
        self.at_line_start = False
        self.pending = ""
        self.suppressed = False

    def _content(self, text: str) -> None:
        if self.suppressed:
            return
        buf = self.pending + text
        i = buf.find(self.TAG)
        if i >= 0:
            self._out(buf[:i])
            self.pending, self.suppressed = "", True
            return
        hold = next((k for k in range(min(len(self.TAG) - 1, len(buf)), 0, -1)
                     if self.TAG.startswith(buf[-k:])), 0)
        self._out(buf[:len(buf) - hold])
        self.pending = buf[len(buf) - hold:]

    def flush(self) -> None:
        if self.pending and not self.suppressed:
            self._out(self.pending)
        self.pending = ""

    def _out(self, text: str) -> None:
        if text:
            print(text, end="", flush=True)
            self.at_line_start = text.endswith("\n")

    def event(self, ev: Event) -> None:
        if ev.kind == "reasoning":
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
            self._content(ev.text)
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
        self.newline()
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            args = raw_args
        shown = next(iter(args.values()), "") if isinstance(args, dict) and len(args) == 1 else args
        print(f"  [{name}] {_short(shown if name != 'run_python' else '(code below)')}", flush=True)

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


def _respond(messages: list[dict], profile, toolbox, thinking: bool, args) -> TurnResult | None:
    printer = _Printer(args.show_thinking)
    print("alice> ", end="", flush=True)
    try:
        result = run_turn(messages, profile, toolbox, thinking=thinking, on_event=printer.event,
                          on_tool=printer.tool_done, on_tool_start=printer.tool_start, confirm=_confirm)
    except KeyboardInterrupt:
        print("\n(stopped; this exchange was not kept)\n")
        return None
    except LLMError as e:
        print(f"\n(error: {e})\n")
        return None
    printer.flush()
    if result.finish == "tool_loop":                 # nothing usable was printed: show the fallback
        printer.newline()
        print(result.reply, end="")
    printer.newline()
    tps = result.timings.get("predicted_per_second")
    tools_used = f" · tools: {', '.join(t.name for t in result.tool_results)}" if result.tool_results else ""
    print(f"[{result.seconds:.1f}s · {result.gen_tokens} tok" + (f" · {tps:.0f} tok/s" if tps else "")
          + tools_used + "]")
    if result.finish == "length":
        print("(reply hit the length limit and was cut off; type 'continue' for the rest)")
        if thinking and not result.reply:
            print("(the reasoning used the whole budget; try again or turn /think off)")
    print()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
