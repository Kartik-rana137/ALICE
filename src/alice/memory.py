"""Long-term memory: what ALICE remembers about the student between sessions.

Two mechanisms, both run by the local model and stored on this machine only:

* Compaction (within a session). When the conversation fills most of the context window, the oldest turns
  are summarised and the summary replaces them, instead of those turns silently falling out of the window.
* Notes (across sessions). A small SQLite database (data/memory/memory.db, gitignored) holds two kinds of
  notes: "fact" - something the student said about themselves (courses, projects, deadlines, tools,
  how they like explanations, what they find hard) - and "session" - a few sentences on what a past
  conversation covered. Notes are written when a conversation is compacted, on /reset and when the chat
  closes; the newest ones go into the system prompt at the next start (a few hundred tokens, fixed for the
  whole session so the prompt cache stays valid). /memory shows them, /remember adds one, /forget deletes.

A 9B model will sometimes "remember" things nobody said, so candidate facts are kept only if most of their
content words occur in the student's own messages (`grounded`), and near-duplicates are skipped.
"""
from __future__ import annotations

import datetime
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import Profile
from .llm import complete_json

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": ["summary", "facts"],
    "additionalProperties": False,
}

SUMMARIZER_SYSTEM = """\
You keep notes for ALICE, an AI mentor, about her conversations with one student. Read the conversation \
and return JSON with two fields.

"summary": 2-5 sentences on what was discussed and worked out in the conversation below: topics, papers, \
key results and answers (keep exact numbers), code that was written, and anything left open. Write it so \
ALICE can pick up the thread later. An earlier summary may be given as context: don't repeat it, cover only \
the conversation below.

"facts": up to 6 short facts about the STUDENT that will still matter in future sessions: what they \
study or work on, their projects and deadlines, the tools and languages they use, how they like \
explanations, what they find hard. Only things the student said or clearly showed about themselves - never \
facts about the topic, the papers or ALICE's answers. Write each one as "The student ...". Leave out \
anything already among the known facts. Use an empty list when there is nothing new (that is common).

Never add details that are not in the conversation."""

# Words that carry no evidence when checking a fact against the student's messages.
_GENERIC = set("""student students they them their theirs the and for with that this these those has have had
is are was were be been being wants want wanted likes like liked prefers prefer preferred uses use using used
working works work worked studying studies study studied currently about from into will would should can could
also often usually asked asks ask says said learning learn learns learned trying tries try tried needs need
needed does did doing done get gets got getting make makes made some very more most much many well other
which what when where who how why there here than then just still already find finds found hard easy
explanation explanations explain explained alice mentor""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9+#]*")


def _stem(word: str) -> str:
    """Crude suffix stripping so "taking" / "takes" / "take" and "systems" / "system" compare equal."""
    for suffix in ("ing", "ed", "es", "e", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _content_words(text: str) -> list[str]:
    return [_stem(w) for w in _words(text) if len(w) >= 3 and w not in _GENERIC]


def grounded(fact: str, student_text: str, min_share: float = 0.5) -> bool:
    """Is a candidate fact backed by the student's own words? At least `min_share` of its content words
    must occur (after crude stemming) in what the student wrote."""
    content = _content_words(fact)
    if not content:
        return False
    have = {_stem(w) for w in _words(student_text)}
    return sum(1 for w in content if w in have) / len(content) >= min_share


def similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Near-duplicate facts (Jaccard overlap of content words)."""
    wa, wb = set(_content_words(a)), set(_content_words(b))
    if not wa or not wb:
        return a.strip().lower() == b.strip().lower()
    return len(wa & wb) / len(wa | wb) >= threshold


@dataclass
class Note:
    id: int
    kind: str              # "fact" | "session"
    text: str
    created: str           # ISO date-time
    source: str            # "extracted" | "user"
    session: str = ""

    @property
    def day(self) -> str:
        return self.created[:10]


@dataclass
class MemoryUpdate:
    summary: str
    facts: list[str] = field(default_factory=list)       # kept after grounding/dedup
    rejected: list[str] = field(default_factory=list)    # dropped as ungrounded or duplicate


class MemoryStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "memory.db"
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}          # cached; the chat box's status line asks often
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL, created TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'extracted', session TEXT NOT NULL DEFAULT '')""")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ----- reading
    def notes(self, kind: str | None = None, limit: int | None = None) -> list[Note]:
        sql = "SELECT id, kind, text, created, source, session FROM notes"
        args: list = []
        if kind:
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " ORDER BY created DESC, id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [Note(*row) for row in self.db.execute(sql, args)]

    def facts(self, limit: int | None = None) -> list[Note]:
        return self.notes("fact", limit)

    def count(self, kind: str = "fact") -> int:
        if kind not in self._counts:
            with self._lock:
                self._counts[kind] = self.db.execute("SELECT COUNT(*) FROM notes WHERE kind = ?", (kind,)).fetchone()[0]
        return self._counts[kind]

    def _changed(self) -> None:
        self._counts.clear()

    def sessions(self, limit: int | None = None) -> list[Note]:
        return self.notes("session", limit)

    # ----- writing
    def add(self, kind: str, text: str, source: str = "extracted", session: str = "",
            when: datetime.datetime | None = None) -> Note | None:
        """Add a note; returns None for a fact that duplicates an existing one."""
        text = " ".join(text.split())
        if not text:
            return None
        if kind == "fact" and any(similar(text, n.text) for n in self.facts()):
            return None
        created = (when or datetime.datetime.now()).isoformat(timespec="seconds")
        with self._lock:
            cur = self.db.execute("INSERT INTO notes (kind, text, created, source, session) VALUES (?,?,?,?,?)",
                                  (kind, text, created, source, session))
            self.db.commit()
        self._changed()
        return Note(cur.lastrowid, kind, text, created, source, session)

    def replace_session_summary(self, session: str, text: str) -> Note | None:
        """One summary per session: a final summary replaces the interim ones written at compaction."""
        with self._lock:
            self.db.execute("DELETE FROM notes WHERE kind = 'session' AND session = ?", (session,))
            self.db.commit()
        self._changed()
        return self.add("session", text, session=session)

    def delete(self, note_id: int) -> bool:
        with self._lock:
            cur = self.db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self.db.commit()
        self._changed()
        return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            cur = self.db.execute("DELETE FROM notes")
            self.db.commit()
        self._changed()
        return cur.rowcount

    # ----- prompt
    def prompt_section(self, max_facts: int = 25, max_sessions: int = 3, max_chars: int = 3000) -> str:
        """The notes that go into the system prompt; facts the student gave with /remember first."""
        facts = self.facts()
        facts = [n for n in facts if n.source == "user"] + [n for n in facts if n.source != "user"]
        facts = facts[:max_facts]
        sessions = self.sessions(max_sessions)
        if not facts and not sessions:
            return ""
        lines = ["What you remember about the student from earlier sessions (your own notes; they can be out "
                 "of date and what the student says now wins; use them when relevant, don't recite them):"]
        lines += [f"- {n.text}" for n in facts]
        if sessions:
            lines.append("Recent sessions:")
            lines += [f"- {n.day}: {_first_sentences(n.text, 350)}" for n in reversed(sessions)]
        out = "\n".join(lines)
        return out if len(out) <= max_chars else out[:max_chars].rsplit("\n", 1)[0]


def _first_sentences(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > max_chars // 3 else cut.rstrip() + "..."


def summarize(conversation_text: str, profile: Profile, known_facts: list[str] = (),
              earlier_summary: str = "", max_tokens: int = 700) -> dict:
    """Ask the local model for {"summary", "facts"} about a stretch of conversation."""
    parts = []
    if known_facts:
        parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in known_facts))
    if earlier_summary:
        parts.append("Earlier summary:\n" + earlier_summary)
    parts.append("Conversation:\n" + conversation_text)
    return complete_json([{"role": "system", "content": SUMMARIZER_SYSTEM},
                          {"role": "user", "content": "\n\n".join(parts)}],
                         profile, SUMMARY_SCHEMA, max_tokens=max_tokens)


def join_summaries(earlier: str, new: str, max_chars: int = 1800) -> str:
    """Earlier summary + summary of what came after it. Asked to merge the two itself, the 9B dropped most of
    the earlier part (memory eval, 2026-11-20), so they are joined here; past `max_chars` the oldest
    sentences go first."""
    earlier, new = earlier.strip(), new.strip()
    if not earlier:
        return new[-max_chars:] if len(new) > max_chars else new
    sentences = re.split(r"(?<=[.!?])\s+", earlier)
    while sentences and len(" ".join(sentences)) + 1 + len(new) > max_chars:
        sentences.pop(0)
    return (" ".join(sentences) + " " + new).strip() if sentences else new[-max_chars:]


def filter_facts(candidates, student_text: str, existing: list[str]) -> tuple[list[str], list[str]]:
    """Keep candidate facts that are grounded in the student's words and new; return (kept, rejected)."""
    kept, rejected = [], []
    for fact in candidates:
        fact = " ".join(str(fact).split())
        if not fact:
            continue
        if not grounded(fact, student_text) or any(similar(fact, e) for e in [*existing, *kept]):
            rejected.append(fact)
        else:
            kept.append(fact)
    return kept, rejected


def student_words(turns: list[list[dict]]) -> str:
    """Everything the student wrote in these turns (without ALICE's turn notes)."""
    from .persona import strip_note
    return "\n".join(strip_note(m.get("content") or "") for t in turns for m in t if m["role"] == "user")
