"""One assistant turn with tool use: stream, run requested tools, feed results back, repeat."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import Profile
from .llm import Event, chat_stream
from .tools import Tool, ToolBox, ToolResult

MAX_TOOL_ROUNDS = 6
FINAL_ROUND_NOTE = ("[Note from the system, not the student: no more tool calls are possible in this turn. "
                    "Answer the student's question now in plain text, using what you already have.]")
STUCK_REPLY = ("I got stuck trying to use my tools on that one and couldn't finish an answer. "
               "Could you rephrase it, or ask me to answer without tools?")
# Qwen-style tool-call markup that sometimes leaks into plain text when the model has no tools left.
_LEAKED_CALL = re.compile(r"<tool_call>.*?(</tool_call>|$)", re.S)


@dataclass
class TurnResult:
    reply: str                                      # final visible answer
    messages: list[dict]                            # new messages to append to history (tool rounds + answer)
    tool_results: list[ToolResult] = field(default_factory=list)
    finish: str | None = None
    rounds: int = 0
    reasoning_chars: int = 0
    gen_tokens: int = 0
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)
    repeated_calls: int = 0                          # identical failed calls the model tried again
    leaked_markup: bool = False                      # tool-call text had to be stripped from the reply


def _signature(name: str, raw_args: str) -> str:
    try:
        return name + json.dumps(json.loads(raw_args or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        return name + (raw_args or "")


def run_turn(messages: list[dict], profile: Profile, toolbox: ToolBox | None = None, thinking: bool = False,
             on_event: Callable[[Event], None] | None = None,
             on_tool: Callable[[ToolResult], None] | None = None,
             on_tool_start: Callable[[str, str], None] | None = None,
             confirm: Callable[[Tool, dict], bool] | None = None,
             max_rounds: int = MAX_TOOL_ROUNDS) -> TurnResult:
    """`messages` is the full request context (system + history + new user message).

    Streams every model round through `on_event`; executes tool calls via `toolbox` and reports
    them through `on_tool`. Returns the new messages so the caller can store them in history.
    Guards against small-model tool loops: identical failed calls are not re-run, and the last round
    offers no tools plus a note asking for a plain-text answer.
    """
    t0 = time.time()
    specs = toolbox.specs() if toolbox and toolbox.tools else None
    new: list[dict] = []
    failed: set[str] = set()
    result = TurnResult(reply="", messages=new)
    force_final = False
    for round_no in range(1, max_rounds + 1):
        result.rounds = round_no
        final = force_final or round_no == max_rounds or not specs
        request = messages + new
        if final and new:                       # only needed once tools have been in play
            request = request + [{"role": "user", "content": FINAL_ROUND_NOTE}]
        content: list[str] = []
        calls: list[dict] = []
        for ev in chat_stream(request, profile, thinking=thinking, tools=None if final else specs):
            if on_event:
                on_event(ev)
            if ev.kind == "content":
                content.append(ev.text)
            elif ev.kind == "reasoning":
                result.reasoning_chars += len(ev.text)
            elif ev.kind == "tool_calls":
                calls = ev.calls
            elif ev.kind == "finish":
                result.finish, result.timings = ev.text, ev.timings
                result.gen_tokens += ev.timings.get("predicted_n") or 0
        text = "".join(content)
        if not calls or final:
            cleaned = _LEAKED_CALL.sub("", text).strip()
            if cleaned != text.strip():
                result.leaked_markup = True
            result.reply = cleaned or (STUCK_REPLY if result.leaked_markup or result.tool_results else "")
            if result.leaked_markup and not cleaned:
                result.finish = "tool_loop"
            new.append({"role": "assistant", "content": result.reply})
            break
        new.append({"role": "assistant", "content": text or None,
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                   for c in calls]})
        repeats = 0
        for c in calls:
            sig = _signature(c["name"], c["arguments"])
            if on_tool_start:
                on_tool_start(c["name"], c["arguments"])
            if sig in failed:
                repeats += 1
                result.repeated_calls += 1
                tr = ToolResult(c["name"], {}, "error: this exact call already failed. Do not repeat it; change "
                                               "the arguments, use another tool, or answer without it.", False)
            else:
                tr = toolbox.call(c["name"], c["arguments"], confirm=confirm)
                if not tr.ok:
                    failed.add(sig)
            result.tool_results.append(tr)
            if on_tool:
                on_tool(tr)
            new.append({"role": "tool", "tool_call_id": c["id"], "content": tr.output})
        if repeats == len(calls):               # the model is looping: make the next round the last
            force_final = True
    result.seconds = round(time.time() - t0, 1)
    return result
