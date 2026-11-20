"""Chat session logic shared by the line-based CLI (cli.py) and the chat-box UI (tui.py).

Owns the model server, tools, paper library, conversation, long-term memory, transcript and the optional
Claude escalation, and runs one turn end to end: automatic library retrieval, the unverified-reference
guard, the agent loop, and the post-checks (citations, citation support, calculator grounding, claimed tool
use) whose results also decide whether escalating to Claude is suggested. Slow housekeeping - evaluating the
system prompt ahead of the first question, compacting a long conversation, saving memory notes - runs in
a background thread between turns; ask() waits for it (the model server has one slot). It never prints:
front-ends get callbacks while a reply streams and outcomes with user-facing notices afterwards.
"""
from __future__ import annotations

import datetime
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from . import claude, config, memory, persona
from .agent import STUCK_REPLY, TurnResult, run_turn
from .conversation import Conversation, turns_text
from .llm import Event, LLMError, LlamaServer, count_tokens, prime_prefix, warm_up
from .render import ascii_emoji
from .tools import Tool, ToolBox, ToolResult, calc, coderun, default_toolbox
from .transcript import Transcript

# Claims of having searched or run something. Deliberately excludes "checked"/"verified", which models
# also use for mental work ("I checked the formula").
CLAIMED_TOOL_USE = re.compile(
    r"\bI(?:'ve| have| just|'ve just)?\s+(?:also\s+)?(?:searched|looked (?:it |this |that )?up|queried|ran|executed|tested)\b"
    r"|\bmy (?:search|searches|lookup)\b|\b(?:arxiv|library|web) search(?:es)?(?: results?)?\b|\bsearch results?\b",
    re.I)

HELP = ("Commands: /think on|off       step-by-step reasoning (slower, better at maths/logic)\n"
        "          /arxiv auto on|off  look up named papers/methods on arXiv automatically\n"
        "          /claude [question]  ask Claude: re-check the last answer, or a new question (shows what is sent first)\n"
        "          /read PATH [question]  read a PDF, Word or PowerPoint file (or paste / drag its path in)\n"
        "          /memory  what ALICE remembers about you   /remember TEXT   /forget ID|all\n"
        "          /tools   /reset (new conversation)   /help   /quit")

# Documents the student hands over with /read (or by pasting / dragging a file path into the chat). Short ones
# go into the conversation whole (~4k tokens of a 16k context); longer ones stay "open" and every question
# gets its most relevant pages / slides (keyword search, documents.Excerpts) until /read off.
ATTACH_FULL_CHARS = 14000
DOC_EXCERPTS = 4
DEFAULT_DOC_QUESTION = "Give me a short overview of this document."
_PATH_MESSAGE = re.compile(r'^\s*(?:"([^"]+)"|\'([^\']+)\'|(\S+))\s*(.*)$', re.S)


def file_in_message(text: str) -> tuple[str, str] | None:
    """(path, rest) when a message starts with the path of a readable document - what dragging a file into
    the terminal or pasting its path produces - else None."""
    from .documents import LEGACY, SUPPORTED
    m = _PATH_MESSAGE.match(text)
    if not m:
        return None
    path = next(g for g in m.groups()[:3] if g)
    if Path(path).suffix.lower() not in SUPPORTED | set(LEGACY) or not Path(path).expanduser().is_file():
        return None
    return path, m.group(4).strip()


# Full abstracts (arXiv caps them at ~1,920 characters) for the top 2: with 3 results cut at 1,200
# characters, the answer to "BQ-LoRA's WikiText-2 perplexity" was in the part that got cut off.
ARXIV_AUTO_RESULTS = 2
ARXIV_AUTO_SUMMARY_CHARS = 2000


def arxiv_lookup_query(text: str) -> str | None:
    """What to send to arXiv for a question: the paper/method it names, never the whole question.
    "the Gemma 4 MATLAB fine-tuning paper" -> the longest such phrase; "BQ-LoRA variants" -> "BQ-LoRA"."""
    from .library import paper_references
    refs = [p for p, _ in paper_references(text)]
    if refs:
        return max(refs, key=len)
    terms = persona.named_terms(text)
    return " ".join(terms[:3]) or None


def arxiv_prefetch(text: str) -> tuple[list[dict], str | None]:
    """Search arXiv for the work a question names and return a synthetic arxiv_search call + result to
    insert into the turn (like library prefetch). Measured on 2026 papers the model could not know: with
    the reference guard alone it searched arXiv for 8/13 questions; the rest were answered from nothing."""
    from .tools import arxiv
    query = arxiv_lookup_query(text)
    if not query:
        return [], None
    result = arxiv.search(query, max_results=ARXIV_AUTO_RESULTS, summary_chars=ARXIV_AUTO_SUMMARY_CHARS)
    call_id = "auto_arxiv_1"
    note = ("arXiv search results for the paper or method the student asked about. If one of them is the "
            "work in question, answer from its abstract and cite its arXiv id. If none matches, say you "
            "could not find it.\n\n")
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "arxiv_search", "arguments": json.dumps({"query": query})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": note + result}], query


@dataclass
class Notice:
    kind: Literal["sources", "warning", "info", "escalate", "memory"]
    text: str


@dataclass
class TurnOutcome:
    result: TurnResult | None = None
    notices: list[Notice] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None
    flags: list[str] = field(default_factory=list)      # failed checks (claude.REASONS keys)

    @property
    def kept(self) -> bool:
        return bool(self.result and self.result.reply and not self.interrupted and not self.error)


@dataclass
class EscalationOutcome:
    reply: claude.ClaudeReply | None = None
    notices: list[Notice] = field(default_factory=list)
    error: str | None = None


@dataclass
class CommandResult:
    message: str = ""
    quit: bool = False
    reset: bool = False
    escalate: bool = False                 # front-end: prepare_escalation(), confirm, escalate()
    escalate_question: str | None = None   # /claude <question>
    ask: str | None = None                 # front-end: send this as the student's message now (/read)


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


def _new_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class ChatSession:
    def __init__(self, profile: config.Profile, workspace: str | None = ".", tools: bool = True,
                 offline: bool = False, transcript: bool = True, thinking: bool = False,
                 auto_arxiv: bool | None = None, memory_notes: bool | None = None,
                 claude_escalation: bool | None = None):
        self.profile = profile
        self.thinking = thinking
        self.auto_arxiv = config.ARXIV_AUTO if auto_arxiv is None else auto_arxiv
        self.workspace = Path(workspace).resolve() if workspace else None
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
        self._today: datetime.date | None = None
        # background work between turns (warm-up, compaction, saving memory)
        self._bg_thread: threading.Thread | None = None
        self._bg_label = ""
        self.warm_timings: dict = {}
        self.prefix_cache: dict = {}                # what prime_prefix did at startup (restored / saved / ...)
        self.prefix_primed = False
        self.pending_notices: list[Notice] = []     # from background work; shown with the next answer
        # long-term memory
        self.memory: memory.MemoryStore | None = None
        self.session_id = _new_session_id()
        self._memory_section = ""
        self._pending_extra: list[str] = []         # one-off lines for the next turn note (/remember)
        # documents (/read)
        self.document = None                        # an open long document (documents.Document) or None
        self._excerpts = None
        self._doc_turns = 0
        self._pending_document: str | None = None   # a short document's full text, sent with the next message
        if config.MEMORY_ENABLED if memory_notes is None else memory_notes:
            try:
                self.memory = memory.MemoryStore(config.MEMORY_DIR)
            except Exception as e:                  # e.g. a read-only data folder: chat still works
                self.startup_notes.append(f"Memory disabled: {e}")
        # escalation to Claude
        self.claude: claude.ClaudeClient | None = None
        self.last_flags: list[str] = []
        if config.CLAUDE.enabled if claude_escalation is None else claude_escalation:
            if claude.ClaudeClient.sdk_installed():
                settings = config.CLAUDE if config.CLAUDE.enabled else claude.ClaudeSettings(
                    **{**config.CLAUDE.__dict__, "enabled": True})
                self.claude = claude.ClaudeClient(settings, claude.default_ledger(config.LOG_DIR))
            else:
                self.startup_notes.append("Claude escalation is on, but the 'anthropic' package is missing: "
                                          "pip install -e \".[claude]\"")

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

    @property
    def memory_size(self) -> int:
        return self.memory.count("fact") if self.memory else 0

    def _system_prompt(self) -> str:
        return persona.system_prompt(tools=self.tool_names, today=self._today, memory=self._memory_section)

    def start(self, warm: bool = True) -> bool:
        """Start (or reuse) the model server and set up the conversation. Raises LLMError / ValueError.

        With `warm`, the tools + system prompt are evaluated in the background right away, so the first
        question doesn't wait for them (14.9 s -> 0.7 s to the first token in the latency benchmark)."""
        self.started_server = self.server.start()
        self._today = datetime.date.today()
        if self.memory:
            self._memory_section = self.memory.prompt_section(config.MEMORY_PROMPT_FACTS,
                                                              config.MEMORY_PROMPT_SESSIONS)
        self.convo = Conversation(self._system_prompt(), self.profile.ctx, self.profile.thinking_max_tokens,
                                  counter=count_tokens)
        if self._want_transcript:
            self.transcript = Transcript(
                config.TRANSCRIPT_DIR, model=self.profile.gguf, profile=self.profile.name,
                persona_version=persona.PERSONA_VERSION, ctx=self.profile.ctx, thinking=self.thinking,
                tools=sorted(self.tool_names), memory_notes=self.memory_size,
                claude=self.claude.settings.model if self.claude else None)
        if warm:
            self._run_background("warming up", self._warm_up)
        return self.started_server

    def stop(self) -> None:
        self.wait_background(timeout=30)
        self.server.stop()
        if self.library:
            self.library.embedder.stop()
        if self.memory:
            self.memory.close()

    # ----- background work
    def _run_background(self, label: str, fn: Callable[[], None]) -> None:
        self.wait_background()

        def run():
            try:
                fn()
            except Exception as e:                  # never kill the session over housekeeping
                self.pending_notices.append(Notice("warning", f"({label} failed: {e})"))
        self._bg_label = label
        self._bg_thread = threading.Thread(target=run, name=f"alice-{label.replace(' ', '-')}", daemon=True)
        self._bg_thread.start()

    @property
    def background(self) -> str | None:
        """What the session is doing between turns ("warming up", "compacting", ...), or None."""
        t = self._bg_thread
        return self._bg_label if t is not None and t.is_alive() else None

    @property
    def warming(self) -> bool:
        return self.background == "warming up"

    def wait_background(self, timeout: float | None = None) -> None:
        t = self._bg_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)

    wait_warm = wait_background                     # older name (scripts/latency_bench.py)

    def _warm_up(self) -> None:
        specs = self.toolbox.specs() if self.toolbox and self.toolbox.tools else None
        if config.PROMPT_CACHE and self.started_server and not self.prefix_primed:
            try:                                      # fixed part of the prompt: restore from disk, or compute + save
                self.prefix_cache = prime_prefix(self.convo.messages(), self.profile, specs,
                                                 persona.VARIABLE_PART_MARKER, config.PROMPT_CACHE_DIR)
                self.prefix_primed = True
            except LLMError as e:
                self.prefix_cache = {"status": f"unavailable ({e})"}
        try:
            self.warm_timings = warm_up(self.convo.messages(), self.profile, tools=specs)
        except LLMError:
            self.warm_timings = {}                    # not fatal: the next question just takes longer

    def _take_pending_notices(self) -> list[Notice]:
        out, self.pending_notices = self.pending_notices, []
        return out

    # ----- memory
    def _compact(self) -> None:
        """Summarise all but the newest turns into the conversation summary (and memory notes), then
        re-evaluate the new, shorter prompt so the next question doesn't wait for it."""
        convo = self.convo
        turns = convo.oldest_turns(config.COMPACT_KEEP_TURNS)
        if not turns:
            return
        known = [n.text for n in self.memory.facts(60)] if self.memory else []
        res = memory.summarize(turns_text(turns), self.profile, known, convo.summary)
        summary = memory.join_summaries(convo.summary, str(res.get("summary", "")))
        if not summary:
            raise LLMError("the model returned an empty summary")
        before = convo.history_tokens()
        convo.compact_oldest(len(turns), summary)
        kept: list[str] = []
        if self.memory:
            kept, _ = memory.filter_facts(res.get("facts", []), memory.student_words(turns), known)
            for f in kept:
                self.memory.add("fact", f, session=self.session_id)
            self.memory.replace_session_summary(self.session_id, summary)       # interim; final at the end
        if self.transcript:
            self.transcript.event("compacted", turns=len(turns), tokens_before=before,
                                  tokens_after=convo.history_tokens(), summary=summary, facts=kept)
        msg = f"(memory: compacted {len(turns)} older turns into a summary to keep the conversation in context"
        self.pending_notices.append(Notice("memory", msg + (f"; noted: {'; '.join(kept)})" if kept else ")")))
        self._warm_up()

    def save_memory(self, turns: list[list[dict]] | None = None, earlier_summary: str | None = None,
                    session_id: str | None = None) -> memory.MemoryUpdate | None:
        """Write a summary of the conversation (and new facts about the student) to long-term memory.
        Blocking (one model call, ~5-15 s). Defaults to the current conversation."""
        if not self.memory or self.convo is None:
            return None
        turns = self.convo.turns() if turns is None else turns
        earlier = self.convo.summary if earlier_summary is None else earlier_summary
        said = memory.student_words(turns)
        if not earlier and len(said.strip()) < 40:            # nothing worth remembering
            return None
        known = [n.text for n in self.memory.facts(60)]
        res = memory.summarize(turns_text(turns), self.profile, known, earlier) if turns else {"facts": []}
        summary = memory.join_summaries(earlier, str(res.get("summary", "")))
        kept, rejected = memory.filter_facts(res.get("facts", []), said, known)
        sid = session_id or self.session_id
        for f in kept:
            self.memory.add("fact", f, session=sid)
        if summary:
            self.memory.replace_session_summary(sid, summary)
        if self.transcript:
            self.transcript.event("memory_saved", summary=summary, facts=kept, rejected=rejected)
        return memory.MemoryUpdate(summary, kept, rejected)

    def save_memory_on_exit(self) -> memory.MemoryUpdate | None:
        if not (self.memory and config.MEMORY_SAVE_ON_EXIT and self.convo is not None):
            return None
        self.wait_background()
        try:
            return self.save_memory()
        except LLMError:
            return None

    def _memory_command(self, text: str) -> CommandResult:
        if not self.memory:
            return CommandResult("(memory is off: started with --no-memory, or [memory] enabled = false)")
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        if cmd == "/remember":
            if not arg:
                return CommandResult("usage: /remember something about you, e.g. /remember I'm taking "
                                     "Distributed Systems this term")
            note = self.memory.add("fact", arg, source="user", session=self.session_id)
            if note is None:
                return CommandResult("(I already have a note like that; /memory shows them)")
            self._pending_extra.append(f"The student asked you to remember: {arg}")
            return CommandResult(f"(noted #{note.id}; I'll keep it for future sessions)")
        if cmd == "/forget":
            if arg == "all":
                return CommandResult(f"(forgot everything: {self.memory.clear()} notes deleted)")
            if arg.lstrip("#").isdigit():
                ok = self.memory.delete(int(arg.lstrip("#")))
                return CommandResult(f"(note #{arg.lstrip('#')} deleted)" if ok else f"(no note #{arg.lstrip('#')})")
            return CommandResult("usage: /forget ID   or   /forget all   (/memory lists the IDs)")
        facts, sessions = self.memory.facts(), self.memory.sessions(5)
        if not facts and not sessions:
            return CommandResult("(no notes yet. ALICE writes them when a long conversation is compacted, on "
                                 "/reset and when the chat closes; /remember adds one now)")
        lines = [f"What ALICE remembers (stored only on this computer, {self.memory.path}):"]
        lines += [f"  #{n.id} {n.text}" + ("  (you told me)" if n.source == "user" else "") for n in facts]
        if sessions:
            lines.append("Recent conversations:")
            lines += [f"  #{n.id} {n.day}: {n.text}" for n in sessions]
        lines.append("/forget ID deletes a note, /forget all deletes everything. Changes apply from the next start.")
        return CommandResult("\n".join(lines))

    # ----- commands
    def command(self, text: str) -> CommandResult | None:
        """Handle a /command. Returns None if `text` is not a command."""
        if not text.startswith("/"):
            found = file_in_message(text)                 # a dragged / pasted document path
            return self._read_command(*found) if found else None
        word = text.split()[0]
        if word in ("/quit", "/exit"):
            return CommandResult(quit=True)
        if word == "/help":
            return CommandResult(HELP)
        if word == "/tools":
            if not self.toolbox:
                return CommandResult("(tools are off)")
            return CommandResult("\n".join(f"  {t.name}{' (asks first)' if t.needs_confirmation else ''}"
                                           f"{' (internet)' if t.external else ''}" for t in self.toolbox.tools.values()))
        if word == "/reset":
            self.wait_background()
            turns, summary, sid = self.convo.turns(), self.convo.summary, self.session_id
            self.convo.reset()
            self.session_id, self.last_flags = _new_session_id(), []
            if self.transcript:
                self.transcript.event("reset")
            msg = "(conversation cleared)"
            if self.memory and (turns or summary):
                self._run_background("saving memory", lambda: self._save_after_reset(turns, summary, sid))
                msg = "(conversation cleared; notes from it are being saved to memory in the background)"
            return CommandResult(msg, reset=True)
        if word in ("/memory", "/remember", "/forget"):
            return self._memory_command(text)
        if word == "/read":
            arg = text[len("/read"):].strip()
            if not arg:
                return CommandResult(f"(open document: {self.document.describe()}; /read off closes it)"
                                     if self.document else
                                     "usage: /read PATH [question] - or paste / drag a file path into the chat. "
                                     "PDF, Word (.docx), PowerPoint (.pptx), .txt and .md")
            if arg.lower() in ("off", "close", "none"):
                closed, self.document, self._excerpts = self.document, None, None
                return CommandResult(f"(closed {closed.title})" if closed else "(no document is open)")
            found = file_in_message(arg)
            if found:
                return self._read_command(*found)
            m = _PATH_MESSAGE.match(arg)                  # not a readable file: let read() say why
            return self._read_command(next(g for g in m.groups()[:3] if g), m.group(4).strip())
        if word == "/claude":
            if not self.claude:
                return CommandResult("(Claude escalation is off: set [claude] enabled = true in alice.toml, or "
                                     "start with --claude; it needs an Anthropic API key)")
            q = text[len("/claude"):].strip() or None
            return CommandResult(escalate=True, escalate_question=q)
        if word == "/arxiv":
            arg = text.split()[2:] if text.split()[1:2] == ["auto"] else []
            if "arxiv_search" not in self.tool_names:
                return CommandResult("(arXiv tools are off: started with --offline or --no-tools)")
            if arg:
                self.auto_arxiv = arg[0].lower() in ("on", "1", "true", "yes")
                if self.transcript:
                    self.transcript.event("auto_arxiv", on=self.auto_arxiv)
            return CommandResult(f"(automatic arXiv lookup {'on' if self.auto_arxiv else 'off'}"
                                 + ("; named papers/methods are sent to arxiv.org)" if self.auto_arxiv else ")"))
        if word == "/think":
            arg = text.split()[1:] or ["on" if not self.thinking else "off"]
            self.thinking = arg[0].lower() in ("on", "1", "true", "yes")
            if self.transcript:
                self.transcript.event("thinking", on=self.thinking)
            return CommandResult(f"(thinking {'on' if self.thinking else 'off'})")
        return CommandResult("unknown command. " + HELP)

    def _read_command(self, path: str, question: str = "") -> CommandResult:
        from . import documents
        try:
            doc = documents.read(path)
        except documents.DocumentError as e:
            return CommandResult(f"(can't read that: {e})")
        if self.transcript:
            self.transcript.event("document", title=doc.title, kind=doc.kind, units=len(doc.units), chars=doc.chars)
        ask = question or DEFAULT_DOC_QUESTION
        if doc.chars <= ATTACH_FULL_CHARS:
            self.document, self._excerpts = None, None
            self._pending_document = (f"{persona.ATTACH_MARKER.strip()} shared by the student: {doc.describe()}. Its "
                                      f"full text follows; answer from it and say which {doc.unit_name} you mean.]\n"
                                      + doc.text())
            return CommandResult(f"(read {doc.describe()}; the whole text goes along with your question)", ask=ask)
        self.document, self._excerpts, self._doc_turns = doc, documents.Excerpts(doc), 0
        return CommandResult(f"(opened {doc.describe()}: too long to send whole, so each question gets its most "
                             f"relevant {doc.unit_name}s. /read off closes it)", ask=ask)

    def _document_context(self, text: str) -> str:
        """Text from the open long document for this message: the best-matching excerpts, plus an outline and
        the beginning on the first question (usually "give me an overview")."""
        doc = self.document
        pieces = self._excerpts.top(text, DOC_EXCERPTS)
        first = self._doc_turns == 0
        self._doc_turns += 1
        if not pieces and not first:
            return ""
        parts = []
        if first:
            outline = [u.label for u in doc.units if u.heading]
            if outline:
                parts.append("Outline: " + "; ".join(outline)[:1500])
            opening = doc.text(1, 2)[:3000]
            parts.append(opening)
        parts += [f"--- {label} ---\n{t}" for label, t in pieces if t not in "\n".join(parts)]
        return (f"{persona.ATTACH_MARKER.strip()} excerpts from {doc.describe()} - the parts most relevant to this "
                f"message; answer from them, say which {doc.unit_name} you used, and say so if they don't cover "
                f"the question.]\n" + "\n\n".join(parts))

    def _save_after_reset(self, turns, summary, sid) -> None:
        update = self.save_memory(turns, summary, session_id=sid)
        if update and update.facts:
            self.pending_notices.append(Notice("memory", "(memory: noted " + "; ".join(update.facts) + ")"))

    # ----- one turn
    def ask(self, text: str, *, on_event: Callable[[Event], None] | None = None,
            on_tool_start: Callable[[str, str], None] | None = None,
            on_tool: Callable[[ToolResult], None] | None = None,
            confirm: Callable[[Tool, dict], bool] | None = None,
            on_library: Callable[[list[str]], None] | None = None,
            on_arxiv: Callable[[str], None] | None = None,
            on_program: Callable[[dict], None] | None = None,
            cancel: threading.Event | None = None) -> TurnOutcome:
        """`on_program` gets the live events of programs ALICE runs (start, output, end; see coderun)."""
        convo = self.convo
        if convo is None:
            raise RuntimeError("ChatSession.ask() called before start()")
        self.wait_background()                           # one server slot: housekeeping finishes first anyway
        if datetime.date.today() != self._today:         # keep "today's date" right in long sessions
            self._today = datetime.date.today()
            convo.system["content"] = self._system_prompt()
        if self.transcript:
            self.transcript.user(text)
        attachment, from_pending = "", False                 # a document from /read: it is the context
        if self._pending_document:
            attachment, self._pending_document, from_pending = self._pending_document, None, True
        elif self.document is not None and not persona.is_casual(text):
            attachment = self._document_context(text)
        prefetched, hits = [], []
        if self.library and not attachment:
            prefetched, hits = self.library.prefetch(text)
            if prefetched and on_library:
                on_library(sorted({h.where for h in hits}))
        guard = not prefetched and not attachment and persona.needs_reference_guard(text)
        arxiv_query, looked_up = None, []
        if guard and self.auto_arxiv and "arxiv_search" in self.tool_names:
            if on_arxiv and (q := arxiv_lookup_query(text)):
                on_arxiv(q)                              # show what is sent before sending it
            looked_up, arxiv_query = arxiv_prefetch(text)
        style = persona.pick_sass()
        extra, self._pending_extra = self._pending_extra, []
        casual = not prefetched and persona.is_casual(text)
        notes = [*extra, persona.CASUAL_NOTE] if casual else extra       # `extra` is restored if the turn fails
        # Per-reply instructions travel with the question (and stay in the history), so the system prompt and
        # everything before this turn are byte-identical to the last request and come from the server's cache.
        convo.add_user(text + (f"\n\n{attachment}" if attachment else "")
                       + persona.turn_note(style, guard, arxiv_available="arxiv_search" in self.tool_names,
                                           extra=notes, now=datetime.datetime.now().astimezone()))
        convo.add(*prefetched, *looked_up)
        out = TurnOutcome(notices=self._take_pending_notices())
        try:
            with coderun.live_output(on_program, cancel):
                out.result = run_turn(convo.messages(), self.profile, self.toolbox, thinking=self.thinking,
                                      on_event=on_event, on_tool=on_tool, on_tool_start=on_tool_start,
                                      confirm=confirm, cancel=cancel)
            out.interrupted = out.result.finish == "interrupted"
        except KeyboardInterrupt:
            out.interrupted = True
        except LLMError as e:
            out.error = str(e)
        if not out.kept:
            convo.pop_turn()                         # nothing usable came back: forget the question
            self._pending_extra = extra + self._pending_extra
            if from_pending:
                self._pending_document = attachment  # the shared document goes with the next try
            if out.interrupted:
                out.notices.append(Notice("info", "(stopped; this exchange was not kept)"))
            return out

        result = out.result
        ascii_reply = ascii_emoji(result.reply)          # history only ever shows her ASCII emoticons
        if ascii_reply != result.reply:
            for m in reversed(result.messages):
                if m["role"] == "assistant" and m.get("content") == result.reply:
                    m["content"] = ascii_reply
                    break
            result.reply = ascii_reply
        citations = self._check_citations(result.reply, out.notices, hits) if self.library else None
        numbers_ok = self._check_calculations(result, out.notices)
        fake_tool_claim = self._check_claimed_tools(result, bool(prefetched or arxiv_query), out.notices)
        if citations and citations["unknown"]:
            out.flags.append("unknown_citations")
        if citations and citations["unsupported"]:
            out.flags.append("unsupported_numbers")
        if numbers_ok is False:
            out.flags.append("calculation_mismatch")
        if fake_tool_claim:
            out.flags.append("claimed_tools")
        if result.finish == "tool_loop" or result.reply == STUCK_REPLY:
            out.flags.append("tool_loop")
        if guard and persona.ADMITS_UNKNOWN.search(result.reply):
            out.flags.append("doesnt_know")
        self.last_flags = list(out.flags)
        if out.flags and self.claude and self.claude.settings.suggest:
            out.notices.append(Notice("escalate", claude.suggestion(out.flags)
                                      + " /claude asks Claude (you'll see what is sent first)."))
        if result.finish == "length":
            out.notices.append(Notice("info", "(reply hit the length limit and was cut off; type 'continue' "
                                              "for the rest)"))
        convo.add(*result.messages)
        if self.transcript:
            self.transcript.assistant(
                result.reply, thinking=self.thinking, opener_style=style, finish=result.finish,
                seconds=result.seconds, gen_tokens=result.gen_tokens, rounds=result.rounds,
                reasoning_chars=result.reasoning_chars, timings=result.timings,
                tools=[{"name": t.name, "args": t.args, "ok": t.ok, "declined": t.declined,
                        "output": t.output[:2000]} for t in result.tool_results],
                citations=citations, prefetched=bool(prefetched), calculation_grounded=numbers_ok,
                reference_guard=guard, claimed_tool_use_without_tools=fake_tool_claim, auto_arxiv_query=arxiv_query,
                flags=out.flags)
        if convo.needs_compaction(config.COMPACT_AT, config.COMPACT_KEEP_TURNS):
            self._run_background("compacting", self._compact)
        return out

    # ----- escalation to Claude
    def prepare_escalation(self, question: str | None = None) -> claude.EscalationRequest:
        """Build (but don't send) the request for /claude. Raises claude.ClaudeError with a readable reason."""
        if not self.claude:
            raise claude.ClaudeError("Claude escalation is off")
        self.wait_background()
        req = claude.build_request(self.claude.settings, self.convo.turns(), question=question,
                                   flags=self.last_flags if question is None else (),
                                   workspace=self.workspace, today=self._today)
        self.claude.check_budget(req)
        return req

    def escalate(self, req: claude.EscalationRequest, on_text: Callable[[str], None] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 cancel: threading.Event | None = None) -> EscalationOutcome:
        out = EscalationOutcome()
        try:
            reply = self.claude.ask(req, on_text=on_text, on_status=on_status, cancel=cancel)
        except claude.ClaudeError as e:
            out.error = str(e)
            return out
        out.reply = reply
        if self.transcript:
            self.transcript.event("claude", question=req.question, reasons=req.reasons, reply=reply.text,
                                  **claude.record_dict(reply))
        budget = self.claude.settings.monthly_budget_usd
        cost_line = (f"(Claude · {reply.model} · {reply.input_tokens:,} in / {reply.output_tokens:,} out · "
                     f"${reply.cost_usd:.3f}{' est.' if reply.estimated else ''}; ${reply.month_total:.2f} of "
                     f"${budget:.2f} this month)")
        if reply.interrupted:
            out.notices += [Notice("info", "(stopped; Claude's partial answer was not kept)"), Notice("info", cost_line)]
            return out
        if reply.refusal_category and not reply.text.strip():
            out.notices += [Notice("warning", f"Claude declined to answer this (reason: {reply.refusal_category})."),
                            Notice("info", cost_line)]
            return out
        if req.replaces_last:              # the flagged local answer is replaced, so ALICE builds on the right one
            for m in reversed(self.convo.history):
                if m["role"] == "assistant" and not m.get("tool_calls"):
                    m["content"] = reply.text
                    break
        else:
            self.convo.add_user(req.question)
            self.convo.add_assistant(reply.text)
        self.last_flags = []
        if self.library:
            self._check_citations(reply.text, out.notices)
        if reply.stop_reason == "max_tokens":
            out.notices.append(Notice("info", "(Claude's answer hit the length limit and was cut off)"))
        out.notices.append(Notice("info", cost_line))
        return out

    # ----- post-checks
    def _check_citations(self, reply: str, notices: list[Notice], prefetched_hits=()) -> dict:
        rep = self.library.check(reply)
        if rep.cited:
            notices.append(Notice("sources", "Sources: " + "; ".join(f"[{l}] {h.where}" for l, h in rep.cited)))
        elif prefetched_hits:
            # The 9B sometimes answers from the passages without citing them (or copies the paper's own
            # reference numbers, e.g. "Adam [20]"). Still show where the information came from.
            notices.append(Notice("sources", "Retrieved from your library (not cited inline): "
                                  + "; ".join(sorted({h.where for h in prefetched_hits}))))
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
    def _check_claimed_tools(result: TurnResult, prefetched: bool, notices: list[Notice]) -> bool:
        """Warn when the reply claims a search / code run that never happened this turn.

        Survey of 120 eval replies: 3 claimed tool use; 2 were fabricated ("I've searched for this ...",
        "... doesn't appear in arXiv search results" with no tool call), 1 was genuine. Returns True if flagged."""
        if result.tool_results or prefetched:
            return False
        m = CLAIMED_TOOL_USE.search(result.reply)
        if not m:
            return False
        notices.append(Notice("warning", f'WARNING: ALICE says "{m.group(0)}", but no search or tool was used in '
                                         "this answer. Treat that part as unverified."))
        return True

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
