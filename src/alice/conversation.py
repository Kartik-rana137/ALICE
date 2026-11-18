"""In-session conversation memory with context-window trimming."""
from __future__ import annotations

CHARS_PER_TOKEN = 3.0     # deliberately conservative estimate; replace with /tokenize if it matters


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 4


class Conversation:
    """Keeps the system prompt plus as many recent messages as fit the context budget."""

    def __init__(self, system_prompt: str, ctx: int, reply_budget: int, margin: int = 256):
        self.system = {"role": "system", "content": system_prompt}
        self.history: list[dict] = []
        self.budget = ctx - reply_budget - margin - estimate_tokens(system_prompt)
        if self.budget <= 0:
            raise ValueError("context too small for the system prompt plus reply budget")

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        self.history.clear()

    def messages(self) -> list[dict]:
        """System prompt + the newest messages that fit. The latest message is always kept."""
        kept: list[dict] = []
        used = 0
        for m in reversed(self.history):
            cost = estimate_tokens(m["content"])
            if kept and used + cost > self.budget:
                break
            kept.append(m)
            used += cost
        kept.reverse()
        # never start the window on an assistant turn
        while len(kept) > 1 and kept[0]["role"] == "assistant":
            kept.pop(0)
        return [self.system, *kept]
