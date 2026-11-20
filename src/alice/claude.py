"""Escalation to Claude for questions the local model is too weak for.

Opt-in, and never silent:

* Off unless `[claude] enabled = true` in alice.toml (or `--claude`, or ALICE_CLAUDE=1). Needs the `anthropic`
  package (`pip install -e ".[claude]"`) and an API key in ANTHROPIC_API_KEY (or an `ant auth login` profile).
* After each local answer, `suggestion()` looks at the automatic checks (citations that were never retrieved,
  numbers that aren't in the cited passages, a calculator mismatch, a claimed search that never happened, a
  tool loop, the model admitting it doesn't know a named paper) and ALICE offers /claude when one fired.
* /claude prepares the request and shows exactly what would be sent - redacted: workspace and home paths,
  e-mail addresses, anything that looks like a key - with a cost estimate, and sends it only after a yes.
  `/claude <question>` asks a new question directly. Library passages and tool results of the escalated turn
  go along (they are what the answer should rest on); long-term memory notes never do.
* Claude answers in ALICE's voice; the answer joins the local conversation, so follow-ups continue locally.
* Every call is logged to logs/claude_usage.jsonl (tokens, cost, reasons; never message text) and the
  monthly budget is enforced from that log.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import persona
from .config import ClaudeSettings

# USD per million tokens: (input, output, cache read). Anthropic list prices as of 2026-11-03; thinking
# tokens are billed as output. Unknown models are priced like the most expensive one (conservative).
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CHARS_PER_TOKEN = 3.3            # rough, for the estimate shown before sending
TYPICAL_OUTPUT_TOKENS = 3000     # answer + adaptive thinking on a hard question

REASONS = {
    "unknown_citations": "it cites library passages that were never retrieved",
    "unsupported_numbers": "it states numbers that are not in the passages it cites",
    "calculation_mismatch": "its answer doesn't match the calculator's result",
    "claimed_tools": "it claims a search or run that never happened",
    "tool_loop": "it got stuck using its tools",
    "doesnt_know": "it doesn't know the paper or method you named",
}

ESCALATION_NOTE = """\

You are the stronger model behind ALICE. Her local model (a small model on the student's own computer) \
handles most questions; this one was passed to you because it is too hard for it, or because automatic \
checks flagged its answer. Answer the student directly, in ALICE's voice and format. If a local answer is \
attached and it was wrong, say briefly what was wrong so the student is not misled; otherwise don't mention \
it. Passages from the student's library are their own copies of the papers: cite them by label (e.g. [L2]) \
when you use them. If you are not sure of something, say so; never invent papers, citations or numbers."""


class ClaudeError(RuntimeError):
    """Escalation failed or is not possible; the message is user-readable."""


def suggestion(flags) -> str | None:
    reasons = [REASONS[f] for f in dict.fromkeys(flags) if f in REASONS]
    if not reasons:
        return None
    return "This answer may be unreliable: " + "; ".join(reasons) + "."


# ----- redaction
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                     r"xox[abpr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,})")
_WIN_USER = re.compile(r"(?i)\b([A-Z]:\\Users\\)[^\\\s\"']+")
_NIX_USER = re.compile(r"(/(?:home|Users)/)[^/\s\"']+")


def redact(text: str, workspace: Path | None = None) -> tuple[str, int]:
    """Remove things that identify the student or their machine. Returns (text, number of replacements)."""
    count = 0

    def sub(pattern, repl, s):
        nonlocal count
        s, n = (pattern.subn(repl, s) if hasattr(pattern, "subn") else (s.replace(pattern, repl), s.count(pattern)))
        count += n
        return s

    for root in [workspace, Path.home()]:
        if root:
            for form in {str(root), str(root).replace("\\", "/")}:
                if len(form) > 3:
                    text = sub(form, "<workspace>" if root == workspace else "~", text)
    text = sub(_SECRET, "<secret>", text)
    text = sub(_EMAIL, "<email>", text)
    text = sub(_WIN_USER, r"\1<user>", text)
    text = sub(_NIX_USER, r"\1<user>", text)
    return text, count


# ----- building the request
@dataclass
class EscalationRequest:
    system: str
    messages: list[dict]
    question: str
    reasons: list[str]
    replaces_last: bool          # the answer replaces the flagged local answer in the history
    n_context_turns: int
    n_passages: int
    n_tool_results: int
    redactions: int
    est_input_tokens: int
    est_cost: float              # with a typical answer length
    max_cost: float              # if the answer uses all of max_tokens
    model: str

    def preview(self) -> str:
        """Exactly what is sent, as text."""
        parts = [f"=== system ({len(self.system):,} characters: ALICE's persona and escalation note)"]
        for m in self.messages:
            parts.append(f"=== {m['role']}\n{m['content']}")
        return "\n\n".join(parts)

    def summary(self) -> str:
        what = ["your question"]
        if self.n_context_turns:
            what.append(f"{self.n_context_turns} earlier turn{'s' * (self.n_context_turns != 1)}")
        if self.n_passages:
            what.append(f"{self.n_passages} library passage{'s' * (self.n_passages != 1)}")
        if self.n_tool_results:
            what.append(f"{self.n_tool_results} tool result{'s' * (self.n_tool_results != 1)}")
        if self.replaces_last:
            what.append("ALICE's flagged answer")
        s = (f"Sends to Anthropic ({self.model}): {', '.join(what)}. ~{self.est_input_tokens:,} tokens in; "
             f"about ${self.est_cost:.2f}, at most ${self.max_cost:.2f}.")
        if self.redactions:
            s += f" {self.redactions} path/e-mail/key mention{'s' * (self.redactions != 1)} redacted."
        return s


def price(model: str) -> tuple[float, float, float]:
    return PRICES.get(model, max(PRICES.values()))


def _turn_parts(turn: list[dict]) -> tuple[str, list[tuple[str, str]], str]:
    """(question, [(tool name, output)], final reply) of one stored turn."""
    names: dict[str, str] = {}
    tools: list[tuple[str, str]] = []
    question, reply = "", ""
    for m in turn:
        if m["role"] == "user" and not question:
            question = persona.strip_note(m.get("content") or "")
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                names[c["id"]] = c["function"]["name"]
        elif m["role"] == "tool":
            tools.append((names.get(m.get("tool_call_id", ""), "tool"), m.get("content") or ""))
        elif m["role"] == "assistant" and m.get("content"):
            reply = m["content"]
    return question, tools, reply


def build_request(settings: ClaudeSettings, turns: list[list[dict]], *, question: str | None = None,
                  flags=(), workspace: Path | None = None, today: datetime.date | None = None,
                  max_tool_chars: int = 6000) -> EscalationRequest:
    """`turns`: the stored conversation (Conversation.turns()). Without `question`, the last turn is
    escalated (its question, the material its tools returned and the flagged local answer); with one, it is
    a new question and the last turns are context."""
    replaces_last = question is None
    if replaces_last:
        if not turns:
            raise ClaudeError("there is no question to pass on yet; ask something first, or use /claude <question>")
        *context, last = turns
        question, tool_results, local_reply = _turn_parts(last)
    else:
        context, tool_results, local_reply = list(turns), [], ""
    context = context[-settings.context_turns:] if settings.context_turns > 0 else []

    messages: list[dict] = []
    for turn in context:
        q, _, r = _turn_parts(turn)
        if q:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": r or "(no answer)"}]
    final = [question.strip()]
    passages = [o for n, o in tool_results if n == "search_library"]
    others = [(n, o) for n, o in tool_results if n != "search_library"]
    if tool_results:
        blocks = []
        for name, out in tool_results:
            out = out if len(out) <= max_tool_chars else out[:max_tool_chars] + "\n[...truncated]"
            blocks.append(f"<tool name=\"{name}\">\n{out}\n</tool>")
        final.append("Material ALICE's tools returned for this question (attached by ALICE's software, not "
                      "written by the student):\n" + "\n".join(blocks))
    if local_reply:
        why = "; ".join(REASONS[f] for f in dict.fromkeys(flags) if f in REASONS) or "the student asked for a second opinion"
        final.append(f"ALICE's local model answered as follows; automatic checks flagged it ({why}):\n"
                     f"<local_answer>\n{local_reply}\n</local_answer>")
    messages.append({"role": "user", "content": "\n\n".join(final)})

    redactions = 0
    for m in messages:
        m["content"], n = redact(m["content"], workspace)
        redactions += n
    system = (persona.SYSTEM_PROMPT + ESCALATION_NOTE
              + f"\n\nToday's date: {(today or datetime.date.today()).isoformat()}.\n")
    chars = len(system) + sum(len(m["content"]) for m in messages)
    est_in = int(chars / CHARS_PER_TOKEN) + 20
    p_in, p_out, _ = price(settings.model)
    return EscalationRequest(
        system=system, messages=messages, question=question, reasons=list(dict.fromkeys(flags)),
        replaces_last=replaces_last, n_context_turns=len(context),
        n_passages=sum(o.count("\n[L") + o.startswith("[L") for o in passages),
        n_tool_results=len(others), redactions=redactions, est_input_tokens=est_in,
        est_cost=(est_in * p_in + TYPICAL_OUTPUT_TOKENS * p_out) / 1e6,
        max_cost=(est_in * p_in + settings.max_tokens * p_out) / 1e6, model=settings.model)


# ----- usage log and budget
class UsageLedger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def add(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def month_total(self, now: datetime.datetime | None = None) -> float:
        month = (now or datetime.datetime.now()).strftime("%Y-%m")
        return sum(float(r.get("cost_usd") or 0) for r in self.records() if str(r.get("time", "")).startswith(month))


@dataclass
class ClaudeReply:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    estimated: bool = False          # usage unknown (stopped early): cost estimated from text length
    seconds: float = 0.0
    refusal_category: str | None = None
    month_total: float = 0.0

    @property
    def interrupted(self) -> bool:
        return self.stop_reason == "interrupted"


def cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    p_in, p_out, p_cache = price(model)
    return (input_tokens * p_in + output_tokens * p_out + cache_read * p_cache + cache_write * p_in * 1.25) / 1e6


class ClaudeClient:
    """Sends an EscalationRequest with the official SDK (streaming) and records usage."""

    def __init__(self, settings: ClaudeSettings, ledger: UsageLedger, client=None):
        self.settings = settings
        self.ledger = ledger
        self._client = client            # injected in tests

    @staticmethod
    def sdk_installed() -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _sdk_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ClaudeError("the 'anthropic' package is not installed: pip install -e \".[claude]\"") from None
            try:
                self._client = anthropic.Anthropic(max_retries=2)
            except Exception as e:                       # e.g. no credentials could be resolved
                raise ClaudeError(f"cannot set up the Anthropic client ({e}); set ANTHROPIC_API_KEY") from None
        return self._client

    def budget_left(self) -> float:
        return self.settings.monthly_budget_usd - self.ledger.month_total()

    def check_budget(self, request: EscalationRequest) -> None:
        left = self.budget_left()
        if request.est_cost > left:
            raise ClaudeError(f"this month's Claude budget is used up (${self.settings.monthly_budget_usd:.2f}; "
                              f"${max(0.0, left):.2f} left, this would cost about ${request.est_cost:.2f}). "
                              "Raise [claude] monthly_budget_usd in alice.toml if you want to spend more.")

    def ask(self, request: EscalationRequest, on_text: Callable[[str], None] | None = None,
            on_status: Callable[[str], None] | None = None, cancel: threading.Event | None = None) -> ClaudeReply:
        self.check_budget(request)
        client = self._sdk_client()
        s = self.settings
        params = dict(model=s.model, max_tokens=s.max_tokens, system=request.system, messages=request.messages,
                      thinking={"type": "adaptive"}, output_config={"effort": s.effort},
                      # If the model declines (safety classifiers), the API re-runs the request on a fallback
                      # model instead of returning nothing.
                      betas=[FALLBACK_BETA], fallbacks="default")
        parts: list[str] = []
        final = None
        stopped = False
        t0 = time.monotonic()
        try:
            with client.beta.messages.stream(**params) as stream:
                for event in stream:
                    if cancel is not None and cancel.is_set():
                        stopped = True
                        break                            # leaving the block closes the connection
                    if event.type == "content_block_start" and getattr(event.content_block, "type", "") == "thinking":
                        if on_status:
                            on_status("thinking")
                    elif event.type == "text":
                        parts.append(event.text)
                        if on_text:
                            on_text(event.text)
                if not stopped:
                    final = stream.get_final_message()
        except Exception as e:                           # mapped to readable messages below
            raise ClaudeError(_describe_error(e, s.model)) from None
        text = "".join(parts)
        if final is not None:
            u = final.usage
            reply = ClaudeReply(
                text=text, model=getattr(final, "model", s.model) or s.model, stop_reason=final.stop_reason,
                input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
            if final.stop_reason == "refusal":
                details = getattr(final, "stop_details", None)
                reply.refusal_category = getattr(details, "category", None) or "unspecified"
        else:                                            # stopped early: no usage reported
            reply = ClaudeReply(text=text, model=s.model, stop_reason="interrupted",
                                input_tokens=request.est_input_tokens,
                                output_tokens=int(len(text) / CHARS_PER_TOKEN), estimated=True)
        reply.cost_usd = cost(reply.model, reply.input_tokens, reply.output_tokens,
                              reply.cache_read_tokens, reply.cache_write_tokens)
        reply.seconds = round(time.monotonic() - t0, 1)
        self.ledger.add({"time": datetime.datetime.now().isoformat(timespec="seconds"), "model": s.model,
                         "served_model": reply.model, "input_tokens": reply.input_tokens,
                         "output_tokens": reply.output_tokens, "cache_read_tokens": reply.cache_read_tokens,
                         "cache_write_tokens": reply.cache_write_tokens, "cost_usd": round(reply.cost_usd, 6),
                         "estimated": reply.estimated, "reasons": request.reasons,
                         "new_question": not request.replaces_last, "stop_reason": reply.stop_reason,
                         "seconds": reply.seconds})
        reply.month_total = self.ledger.month_total()
        return reply


def _describe_error(e: Exception, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        return str(e)
    # most specific first
    if isinstance(e, anthropic.AuthenticationError):
        return "Anthropic rejected the credentials: set a valid ANTHROPIC_API_KEY (or run `ant auth login`)"
    if isinstance(e, anthropic.PermissionDeniedError):
        return f"this API key is not allowed to use {model}"
    if isinstance(e, anthropic.NotFoundError):
        return f"model '{model}' was not found; check [claude] model in alice.toml"
    if isinstance(e, anthropic.RateLimitError):
        return "Anthropic is rate-limiting this key; try again in a minute"
    if isinstance(e, anthropic.BadRequestError):
        return f"Anthropic rejected the request: {getattr(e, 'message', e)}"
    if isinstance(e, anthropic.APIStatusError):
        return f"Anthropic server error ({e.status_code}); try again later"
    if isinstance(e, anthropic.APIConnectionError):
        return "cannot reach the Anthropic API (offline?)"
    if isinstance(e, ClaudeError):
        return str(e)
    return f"{type(e).__name__}: {e}"


def default_ledger(log_dir: Path) -> UsageLedger:
    return UsageLedger(Path(os.environ.get("ALICE_CLAUDE_LEDGER", log_dir / "claude_usage.jsonl")))


def record_dict(reply: ClaudeReply) -> dict:
    return {k: v for k, v in asdict(reply).items() if k != "text"}
