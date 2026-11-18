"""In-session conversation memory with context-window trimming."""
from __future__ import annotations

import json
from typing import Callable

CHARS_PER_TOKEN = 3.0     # conservative fallback when no tokenizer is available
MESSAGE_OVERHEAD = 8      # chat-template tokens per message (role markers etc.), rounded up


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _message_text(m: dict) -> str:
    """Everything in a message that costs tokens (content plus any tool-call payload)."""
    text = m.get("content") or ""
    if m.get("tool_calls"):
        text += json.dumps(m["tool_calls"], ensure_ascii=False)
    return text


class Conversation:
    """Keeps the system prompt plus as many recent turns as fit the context budget.

    A turn is a user message plus everything after it until the next user message (assistant
    replies, tool calls, tool results), so trimming never splits a tool call from its result.
    `counter` should return exact token counts (e.g. the server's /tokenize); results are cached
    per text. If it fails, the conservative character estimate is used instead.
    """

    def __init__(self, system_prompt: str, ctx: int, reply_budget: int, margin: int = 384,
                 counter: Callable[[str], int] | None = None):
        self.system = {"role": "system", "content": system_prompt}
        self.history: list[dict] = []
        self._counter = counter
        self._cache: dict[str, int] = {}
        self.budget = ctx - reply_budget - margin - self.tokens(system_prompt)
        if self.budget <= 0:
            raise ValueError("context too small for the system prompt plus reply budget")

    def tokens(self, text_or_message: str | dict) -> int:
        text = text_or_message if isinstance(text_or_message, str) else _message_text(text_or_message)
        if text not in self._cache:
            n = None
            if self._counter is not None:
                try:
                    n = self._counter(text)
                except Exception:
                    n = None
            self._cache[text] = (n if n is not None else estimate_tokens(text)) + MESSAGE_OVERHEAD
        return self._cache[text]

    # ----- editing -----
    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def add(self, *messages: dict) -> None:
        self.history.extend(messages)

    def pop_last(self) -> None:
        if self.history:
            self.history.pop()

    def pop_turn(self) -> None:
        """Remove the latest turn (last user message and everything after it)."""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "user":
                del self.history[i:]
                return
        self.history.clear()

    def reset(self) -> None:
        self.history.clear()

    # ----- reading -----
    def turns(self) -> list[list[dict]]:
        groups: list[list[dict]] = []
        for m in self.history:
            if m["role"] == "user" or not groups:
                groups.append([])
            groups[-1].append(m)
        # a leading group that doesn't start with a user message is unusable context
        if groups and groups[0][0]["role"] != "user":
            groups.pop(0)
        return groups

    def messages(self, system_prompt: str | None = None) -> list[dict]:
        """System prompt + the newest whole turns that fit. The latest turn is always kept.

        `system_prompt` overrides the stored one for this request (e.g. with a per-turn opener style);
        keep overrides small, the `margin` absorbs the difference.
        """
        kept: list[list[dict]] = []
        used = 0
        for turn in reversed(self.turns()):
            cost = sum(self.tokens(m) for m in turn)
            if kept and used + cost > self.budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()
        system = {"role": "system", "content": system_prompt} if system_prompt else self.system
        return [system, *(m for turn in kept for m in turn)]

    def used_tokens(self) -> int:
        return sum(self.tokens(m) for m in self.messages())
