"""Tools ALICE can call (OpenAI-style function calling through llama-server).

Each tool returns a plain string for the model. Errors are returned as "error: ..." strings rather
than raised, so the model can see them and recover.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

MAX_RESULT_CHARS = 6000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                      # JSON schema for the arguments
    run: Callable[..., str]
    needs_confirmation: bool = False      # ask the user before running (e.g. executing code)
    external: bool = False                # sends data off this machine (shown to the user)

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class ToolResult:
    name: str
    args: dict
    output: str
    ok: bool
    declined: bool = False


@dataclass
class ToolBox:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, *tools: Tool) -> "ToolBox":
        for t in tools:
            self.tools[t.name] = t
        return self

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, raw_args: str | dict | None,
             confirm: Callable[[Tool, dict], bool] | None = None) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(name, {}, f"error: unknown tool '{name}'. Available: {', '.join(self.tools)}", False)
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return ToolResult(name, {}, f"error: could not parse arguments as JSON ({e})", False)
        if tool.needs_confirmation and not (confirm and confirm(tool, args)):
            return ToolResult(name, args, "error: the user declined to run this. Do not retry; "
                                          "answer without it or ask the user.", False, declined=True)
        try:
            out = tool.run(**args)
        except TypeError as e:
            return ToolResult(name, args, f"error: bad arguments for {name}: {e}", False)
        except Exception as e:                       # tool bugs must not crash the chat
            return ToolResult(name, args, f"error: {type(e).__name__}: {e}", False)
        out = str(out)
        if len(out) > MAX_RESULT_CHARS:
            out = out[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(out) - MAX_RESULT_CHARS} more characters]"
        return ToolResult(name, args, out, not out.startswith("error:"))


def default_toolbox(workspace=None, python_exe=None) -> ToolBox:
    from . import arxiv, calc, files, pyexec
    box = ToolBox().add(calc.TOOL, pyexec.make_tool(python_exe), arxiv.SEARCH, arxiv.PAPER)
    if workspace is not None:
        box.add(*files.make_tools(workspace))
    return box
