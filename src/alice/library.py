"""Paper library (Phase 3): PDF/notes ingestion, hybrid search, and verifiable citations.

Storage: one SQLite file (chunks + float32 embeddings + an FTS5 keyword index).
Search:  dense cosine similarity (Qwen3-Embedding via llama-server) fused with BM25 keyword
         search using reciprocal-rank fusion. Keyword search catches exact terms (names, symbols,
         numbers) that small embedding models blur; dense search catches paraphrases.
Citations: every passage shown to the model gets a session label [L1], [L2], ...; after a reply,
         labels it cites are checked against what was actually retrieved.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config
from .llm import LLMError, ServerProcess, _post
from .tools import Tool
from .vectors import VectorIndex

QUERY_INSTRUCTION = ("Given a question from a computer-science student, retrieve passages from "
                     "research papers or notes that answer it")
CHUNK_CHARS = 900
MAX_PASSAGE_CHARS = 1000
RRF_K = 60
CANDIDATES = 30
# Top-1 cosine similarity needed before passages are inserted automatically (configurable per embedding
# model in alice.toml; recalibrate with scripts/calibrate_retrieval.py). For Qwen3-Embedding-0.6B: paper
# questions 0.68-0.83, unrelated CS questions 0.22-0.58.
AUTO_RETRIEVE_MIN_SIM = config.RETRIEVE_MIN_SIM
# Similarity alone can't tell "a question about my papers" from "a general question on a topic my papers
# cover" (a paraphrased Transformer-paper question scored 0.605, generic ML questions up to 0.58). When the
# message itself points at papers/notes, a lower bar applies.
AUTO_RETRIEVE_MIN_SIM_EXPLICIT = config.RETRIEVE_MIN_SIM_EXPLICIT
LEGACY_EMBED_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf"     # libraries built before the model was recorded


class LibraryError(RuntimeError):
    """The library can't be used as configured (message tells the user what to do)."""
REFERS_TO_LIBRARY = re.compile(r"\b(papers?|articles?|library|notes|according to|authors?|my (pdfs?|reading))\b",
                               re.I)
SUPPORTED = {".pdf", ".md", ".txt"}
STOPWORDS = set("""a an and are as at be by does do for from how in is it of on or that the this to was
what when where which who why with did can you your about paper papers""".split())


# ---------------------------------------------------------------- embeddings

class Embedder(Protocol):
    def embed(self, texts: list[str], query: bool = False) -> np.ndarray: ...


class LlamaEmbedder:
    """The embedding model in its own llama-server. Device comes from alice.toml ([embedding] device):
    "cpu" by default so it never competes with the chat model for a small GPU."""

    def __init__(self, gpu: bool | None = None, gguf: Path = config.EMBED_GGUF, port: int = config.EMBED_PORT):
        if gpu is None:
            gpu = config.EMBED_DEVICE == "gpu"
        self.gpu = gpu
        cmd = [str(config.SERVER_EXE), "-m", str(gguf), "--embedding", "--pooling", "last",
               "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "1",
               "--host", config.HOST, "--port", str(port), "-ngl", "99" if gpu else "0"]
        env = dict(os.environ)
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"           # keep the CUDA build entirely off the GPU
        self.server = ServerProcess(cmd, port, gguf, "embed-server.log", env=env)
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:                         # the CLI warms it up in a background thread
            if not self._started:
                self.server.start(wait_s=180)
                self._started = True

    def start_in_background(self) -> None:
        threading.Thread(target=self._warm_up, daemon=True).start()

    def _warm_up(self) -> None:
        try:
            self.start()
        except LLMError:
            pass                                 # the first real search will report the problem

    def stop(self) -> None:
        self.server.stop()

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        self.start()
        if query:
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
        out = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            with _post("/v1/embeddings", {"input": batch}, timeout=600, base=self.server.base) as r:
                data = sorted(json.load(r)["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return _normalize(np.asarray(out, dtype=np.float32))


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------- text extraction and chunking

def clean_text(text: str) -> str:
    # NFKC turns PDF ligatures ("ﬁne-tuning" with U+FB01) and math letters (𝑊) into plain text,
    # otherwise keyword search for "fine-tuning" never matches.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                                  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                       # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text) if s.strip()]


def chunk_text(text: str, target: int = CHUNK_CHARS) -> list[str]:
    """Sentence-aligned chunks of ~target chars; consecutive chunks share one sentence of overlap."""
    sents = split_sentences(text)
    chunks, cur = [], []
    for s in sents:
        while len(s) > target * 1.5:                                   # pathological long "sentence"
            head, s = s[:target], s[target:]
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            chunks.append(head)
        if cur and sum(len(x) + 1 for x in cur) + len(s) > target:
            chunks.append(" ".join(cur))
            cur = [cur[-1]] if len(cur[-1]) < target / 3 else []
        cur.append(s)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c) > 40]


def _pdf_title(reader) -> str | None:
    meta = getattr(reader, "metadata", None)
    title = (meta.title if meta else None) or ""
    if 3 <= len(title.split()) <= 30 and not re.search(r"(untitled|microsoft word|\.dvi|\.tex)", title, re.I):
        return title.strip()
    frags: list[tuple[float, float, str]] = []          # (font size, y position, text)

    def visit(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        if abs(tm[1]) > 1e-6 or abs(tm[2]) > 1e-6:          # rotated text, e.g. the arXiv margin stamp
            return
        if re.match(r"\s*arXiv:\d{4}\.\d{4,5}", text):        # arXiv stamp even when not rotated
            return
        size = round(abs(font_size * (tm[0] or 1) * (cm[0] or 1)), 1)
        y = tm[5] * (cm[3] or 1) + cm[5]
        frags.append((size, y, text.strip()))
    try:
        page = reader.pages[0]
        page.extract_text(visitor_text=visit)
        height = float(page.mediabox.height)
    except Exception:
        return None
    # titles sit near the top of page 1; figures with big labels usually don't
    candidates = [f for f in frags if f[1] > 0.55 * height] or frags
    if not candidates:
        return None
    top = max(s for s, _, _ in candidates)
    title = " ".join(t for s, _, t in candidates if s == top)
    title = re.sub(r"\s+", " ", title).strip()
    return None if _looks_broken(title) else title


def _looks_broken(title: str) -> bool:
    """Small-caps or symbol-heavy titles come out of font analysis as letter soup."""
    words = title.split()
    if not 2 <= len(words) <= 30 or len(title) < 8:
        return True
    short = sum(len(w.strip(":-,.")) <= 1 for w in words)
    return short / len(words) > 0.25 or sum(ch.isalpha() for ch in title) / len(title) < 0.6


def _first_line_title(first_page_text: str) -> str | None:
    lines = [re.sub(r"\s+", " ", l).strip() for l in first_page_text.splitlines()[:16]]
    for i, line in enumerate(lines[:15]):
        if line.endswith("-") and lines[i + 1]:                 # title wrapped with a hyphen
            line = line[:-1] + lines[i + 1]
        if 15 <= len(line) <= 150 and not re.search(r"@|arXiv:|https?://|permission|copyright", line, re.I) \
                and not _looks_broken(line):
            return line
    return None


# Numbered section headings on their own line: "3.1 Pre-training BERT", "5 Experiments", "A.2 Details".
_PDF_HEADING = re.compile(r"^\s*((?:\d{1,2}(?:\.\d{1,2}){0,3})|(?:[A-H](?:\.\d{1,2}){1,3}))\.?\s+"
                          r"([A-Z][^\n]{1,80}?)\s*$")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def heading_of(line: str, markdown: bool = False) -> str | None:
    """The section heading on this line, if it is one."""
    if markdown:
        m = _MD_HEADING.match(line)
        return m.group(1).strip() if m else None
    m = _PDF_HEADING.match(line)
    if not m:
        return None
    name = m.group(2).strip()
    words = name.split()
    if (len(words) > 10 or name[-1] in ".,;:=" or "=" in name or "@" in name
            or sum(c.isdigit() for c in name) > len(name) * 0.3):
        return None                                   # table rows, equations, sentences, footnotes
    return f"{m.group(1)} {name}"


def split_sections(raw: str, section: str | None, markdown: bool = False
                   ) -> tuple[list[tuple[str | None, str]], str | None]:
    """Split raw page text at headings. Returns ([(section, raw_segment)], section in effect at the end).
    The section carries over from the previous page until a new heading appears."""
    segments: list[tuple[str | None, list[str]]] = [(section, [])]
    for line in raw.splitlines():
        h = heading_of(line, markdown)
        if h:
            section = h
            segments.append((section, []))
        else:
            segments[-1][1].append(line)
    return [(s, "\n".join(lines)) for s, lines in segments if "".join(lines).strip()], section


def extract(path: Path) -> tuple[str, list[tuple[int | None, str | None, str]], int]:
    """(title, [(page_number, section, cleaned_text), ...], n_pages). Page is None for text notes."""
    segments: list[tuple[int | None, str | None, str]] = []
    if path.suffix.lower() == ".pdf":
        import logging
        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)    # font-encoding chatter, harmless for text
        reader = PdfReader(str(path))
        first_raw, section = "", None
        for i, page in enumerate(reader.pages, 1):
            try:
                raw = unicodedata.normalize("NFKC", page.extract_text() or "")
            except Exception:
                raw = ""
            if i == 1:
                first_raw = raw
            parts, section = split_sections(raw, section)
            segments += [(i, s, clean_text(t)) for s, t in parts]
        title = _pdf_title(reader) or _first_line_title(first_raw) or path.stem
        return title, segments, len(reader.pages)
    text = path.read_text(encoding="utf-8", errors="replace")
    first = next((l.lstrip("# ").strip() for l in text.splitlines() if l.strip()), path.stem)
    parts, _ = split_sections(text, None, markdown=path.suffix.lower() == ".md")
    return first[:120], [(None, s, clean_text(t)) for s, t in parts], 0


# ---------------------------------------------------------------- store and search

@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    title: str
    page: int | None
    text: str
    section: str | None = None
    score: float = 0.0                      # fused (RRF) score
    similarity: float | None = None         # dense cosine similarity, if the chunk was a dense candidate

    @property
    def where(self) -> str:
        place = f"{self.title}, p. {self.page}" if self.page else self.title
        return f"{place} ({self.section})" if self.section else place


# "restrict": search only the named paper; "boost": also fuse in the global search. Measured 2026-11-18:
# restrict found 29/29 evidence passages (alias 6/6), boost 28/29 (it missed "the Transformer paper's
# optimizer settings", the case scoping exists for); mean MRR tied at 0.79.
SCOPE_MODE = os.environ.get("ALICE_SCOPE_MODE", "restrict")
SCOPE_MIN_RATIO = 1.5      # "Transformer": Attention paper 0.31 of chunks vs LoRA 0.20 (1.6x); "LoRA" in a
SCOPE_MIN_CHUNKS = 2       # library of two LoRA papers: 1.0x -> ambiguous, not scoped (correct)

_QUOTED = re.compile(r"[\"“]([^\"”]{4,120})[\"”]")                                             # "quoted title"
_PAPER_WORD = re.compile(r"\b(?:paper|study|article|preprint)\b", re.I)                         # the X paper
_PAPER_ON = re.compile(r"\b(?:paper|study|article|preprint)\s+(?:on|about|called|titled|named)\s+(.{3,90}?)"
                       r"(?=\s*[,?;:]|\s*$|\s+(?:how|what|which|did|does|is|was|were|use|uses|used|report|reports)\b)",
                       re.I)                                                                    # the paper on X, ...
_IN_X = re.compile(r"^\s*(?:in|for|according to)\s+([A-Z][\w+\-.’' ]{1,60}?),", re.I)          # In Ambient SFT, ...
_DETERMINER = {"the", "this", "that", "a", "an", "your", "my"}


@dataclass
class PaperScope:
    doc_id: int
    title: str
    phrase: str          # how the question referred to the paper
    rest: str            # the question without that reference
    how: str             # "title" or "density"


def paper_references(query: str) -> list[tuple[str, tuple[int, int]]]:
    """Candidate paper references in a question: (phrase, span to remove), in the order to try.

    For "... the X Y Z paper" the candidates run from the nearest word outwards ("Z", "Y Z", "X Y Z"),
    stopping at the determiner, so "the deepest network the ResNet paper" yields "ResNet" rather than
    "deepest network the ResNet"; resolve_reference() takes the first candidate that identifies a paper."""
    out = [(m.group(1).strip(), m.span()) for m in _QUOTED.finditer(query)]
    for m in _PAPER_WORD.finditer(query):
        words = list(re.finditer(r"[\w+\-.'’]+", query[:m.start()]))
        stop = next((i for i in range(len(words) - 1, -1, -1) if words[i].group(0).lower() in _DETERMINER), None)
        if stop is None:
            continue
        name = words[stop + 1:][-7:]
        for n in range(1, len(name) + 1):
            part = name[-n:]
            out.append((" ".join(w.group(0) for w in part), (words[stop].start(), m.end())))
    out += [(_bare_name(m.group(1)), m.span()) for m in _PAPER_ON.finditer(query)]
    out += [(_bare_name(m.group(1)), m.span()) for m in _IN_X.finditer(query)]
    return [(p, s) for p, s in out if p]


def _bare_name(phrase: str) -> str:
    """'the Kernel Reality Check paper' -> 'Kernel Reality Check'."""
    phrase = re.sub(r"^(?:the|this|that|a|an)\s+", "", phrase.strip(), flags=re.I)
    return re.sub(r"\s+(?:paper|study|article|preprint)$", "", phrase, flags=re.I).strip()


def _strip_span(query: str, span: tuple[int, int]) -> str:
    rest = query[:span[0]] + " " + query[span[1]:]
    rest = re.sub(r"^\s*(?:in|according to|for)(?:\s+the)?\s*,\s*", "", rest, flags=re.I)   # "In , what ..."
    rest = re.sub(r"\b(?:in|of|from|by)\s+(?:the\s+)?,", ",", rest, flags=re.I)
    return re.sub(r"\s+", " ", rest).strip(" ,")


def fts_query(query: str) -> str | None:
    words = [w for w in re.findall(r"[\w]+", query.lower()) if w not in STOPWORDS and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words)) or None


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Library:
    """One SQLite file (documents, chunks, embeddings, FTS5 keyword index, metadata) plus a memory-mapped
    vector cache next to it. The library records which embedding model built it and refuses to mix
    models: vectors from different embedders are not comparable (and may differ in dimension)."""

    def __init__(self, directory: Path = config.LIBRARY_DIR, embed_model: str = config.EMBED_MODEL):
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self.embed_model = embed_model
        # The chat box opens the library on the UI thread and searches it on a worker thread. SQLite here
        # is built "serialized" (sqlite3.threadsafety == 3), so one connection may be shared; Python's
        # same-thread check is only a conservative default. Chat-time use is read-only.
        self.db = sqlite3.connect(directory / "library.db", check_same_thread=sqlite3.threadsafety != 3)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, path TEXT, sha256 TEXT UNIQUE,
                title TEXT, pages INTEGER, added TEXT);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER,
                ord INTEGER, text TEXT, embedding BLOB);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
        """)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(chunks)")}
        if "section" not in cols:                       # libraries created before section headers
            self.db.execute("ALTER TABLE chunks ADD COLUMN section TEXT")
        if self.meta("embed_model") is None and not self.is_empty():
            self.set_meta("embed_model", LEGACY_EMBED_MODEL)   # built before the model was recorded
        self.db.commit()
        self.index = VectorIndex(
            directory,
            rows=lambda: self.db.execute("SELECT id, embedding FROM chunks ORDER BY id"),
            count=lambda: self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            stamp=lambda: f"{self.meta('version') or 0}:{self.meta('embed_model')}")

    # ----- metadata
    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))

    def _bump_version(self) -> None:
        """Every change to the stored vectors bumps this counter; vector caches (also in other processes,
        e.g. a running chat while `alice ingest` adds papers) compare it before each search."""
        self.set_meta("version", str(int(self.meta("version") or 0) + 1))

    def embedding_problem(self) -> str | None:
        """A user-facing message if this library was built with a different embedding model."""
        built_with = self.meta("embed_model")
        if built_with and built_with != self.embed_model and not self.is_empty():
            return (f"the library at {self.dir} was built with {built_with}, but the configured embedding "
                    f"model is {self.embed_model}. Run `alice library --rebuild` to re-embed it.")
        return None

    def _check_compatible(self) -> None:
        problem = self.embedding_problem()
        if problem:
            raise LibraryError(problem)

    # ----- ingestion
    def add(self, path: Path, embedder: Embedder, title: str | None = None,
            section_headers: bool = True) -> tuple[str, int, str]:
        """Ingest one file. Returns (status, n_chunks, title); status is 'added', 'skipped' or 'empty'.

        With `section_headers`, each chunk is indexed and embedded together with its section heading
        ("contextual chunk headers"): a passage under "3.1 Pre-training BERT" that never repeats
        "pre-training" can still be found by a question about BERT's pre-training tasks."""
        self._check_compatible()
        path = Path(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        row = self.db.execute("SELECT title FROM docs WHERE sha256=?", (sha,)).fetchone()
        if row:
            return "skipped", 0, row[0]
        auto_title, segments, n_pages = extract(path)
        title = title or auto_title
        items = [(page, section if section_headers else None, c)
                 for page, section, text in segments for c in chunk_text(text)]
        if not items:
            return "empty", 0, title
        vectors = embedder.embed([f"{title} | {s}\n{c}" if s else f"{title}\n{c}" for _, s, c in items])
        cur = self.db.execute("INSERT INTO docs(path, sha256, title, pages, added) VALUES (?,?,?,?,?)",
                              (str(path), sha, title, n_pages, time.strftime("%Y-%m-%d")))
        doc_id = cur.lastrowid
        for n, ((page, section, text), vec) in enumerate(zip(items, vectors)):
            c = self.db.execute("INSERT INTO chunks(doc_id, page, ord, text, embedding, section) "
                                "VALUES (?,?,?,?,?,?)",
                                (doc_id, page, n, text, vec.astype(np.float32).tobytes(), section))
            self.db.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                            (c.lastrowid, f"{section}\n{text}" if section else text))
        self.set_meta("embed_model", self.embed_model)
        self.set_meta("embed_dim", str(len(vectors[0])))
        self._bump_version()
        self.db.commit()
        self.index.invalidate()
        return "added", len(items), title

    def rebuild(self, embedder: Embedder, section_headers: bool = True) -> list[tuple[str, str, int]]:
        """Re-ingest every document from its original file with the current pipeline and embedding
        model (keeps titles). [(title, status, chunks)]."""
        docs = self.documents()
        self.set_meta("embed_model", self.embed_model)      # switching models: the old vectors all go
        self._bump_version()
        self.db.commit()
        out = []
        for doc_id, title, _, _, _, path in docs:
            if not Path(path).exists():
                out.append((title, "missing file", 0))
                continue
            self.remove(doc_id)
            status, n, _ = self.add(Path(path), embedder, title=title, section_headers=section_headers)
            out.append((title, status, n))
        return out

    def remove(self, doc_id: int) -> bool:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        n = self.db.execute("DELETE FROM docs WHERE id=?", (doc_id,)).rowcount
        self._bump_version()
        self.db.commit()
        self.index.invalidate()
        return n > 0

    def set_title(self, doc_id: int, title: str) -> bool:
        n = self.db.execute("UPDATE docs SET title=? WHERE id=?", (title, doc_id)).rowcount
        self.db.commit()
        return n > 0

    def documents(self) -> list[tuple]:
        return self.db.execute("""SELECT d.id, d.title, d.pages, d.added, COUNT(c.id), d.path FROM docs d
                                  LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.id""").fetchall()

    def is_empty(self) -> bool:
        return self.db.execute("SELECT 1 FROM docs LIMIT 1").fetchone() is None

    # ----- search
    def dense_ranking(self, query: str, embedder: Embedder, n: int = CANDIDATES,
                      doc_id: int | None = None) -> list[tuple[int, float]]:
        """[(chunk_id, cosine similarity)] best first; optionally only within one document."""
        self._check_compatible()
        if len(self.index) == 0:
            return []
        q = embedder.embed([query], query=True)[0]
        try:
            if doc_id is None:
                return self.index.search(q, n)
            ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=? ORDER BY id", (doc_id,))]
            sims = self.index.score(q, ids)
            return [(ids[i], float(sims[i])) for i in np.argsort(-sims)[:n]]
        except ValueError as e:                          # dimension mismatch: different embedding model
            raise LibraryError(str(e)) from None

    def keyword_ranking(self, query: str, n: int = CANDIDATES, doc_id: int | None = None) -> list[int]:
        q = fts_query(query)
        if not q:
            return []
        try:
            if doc_id is None:
                rows = self.db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                                       "ORDER BY bm25(chunks_fts) LIMIT ?", (q, n)).fetchall()
            else:
                rows = self.db.execute("SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                                       "WHERE chunks_fts MATCH ? AND c.doc_id = ? ORDER BY bm25(chunks_fts) LIMIT ?",
                                       (q, doc_id, n)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]

    # ----- "the X paper": scope the search to the paper the question names
    def resolve_reference(self, query: str) -> PaperScope | None:
        """Which document does the question name ("the Transformer paper", "the paper on X", a quoted
        title, "In Ambient SFT, ...")? Title match first, else the document where the phrase is clearly
        densest (>= SCOPE_MIN_RATIO x the runner-up). None when nothing is named or it's ambiguous."""
        docs = self.db.execute("SELECT d.id, d.title, COUNT(c.id) FROM docs d JOIN chunks c ON c.doc_id = d.id "
                               "GROUP BY d.id").fetchall()
        if len(docs) < 1:
            return None
        for phrase, span in paper_references(query):
            words = [w for w in re.findall(r"[\w]+", phrase.lower()) if w not in STOPWORDS and len(w) > 1]
            if not words:
                continue
            in_title = [d for d in docs if all(w in re.findall(r"[\w]+", d[1].lower()) for w in words)]
            if len(in_title) == 1:
                return PaperScope(in_title[0][0], in_title[0][1], phrase, _strip_span(query, span), "title")
            try:
                counts = dict(self.db.execute(
                    "SELECT c.doc_id, COUNT(*) FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                    "WHERE chunks_fts MATCH ? GROUP BY c.doc_id", (" ".join(f'"{w}"' for w in words),)).fetchall())
            except sqlite3.OperationalError:
                continue
            dens = sorted(((counts.get(i, 0) / n, i, t) for i, t, n in docs if n), reverse=True)
            if not dens or counts.get(dens[0][1], 0) < SCOPE_MIN_CHUNKS:
                continue
            second = dens[1][0] if len(dens) > 1 else 0.0
            if second == 0 or dens[0][0] / second >= SCOPE_MIN_RATIO:
                return PaperScope(dens[0][1], dens[0][2], phrase, _strip_span(query, span), "density")
        return None

    def search_with_scope(self, query: str, embedder: Embedder, k: int = 5, mode: str = "hybrid",
                          scope: bool = True) -> tuple[list[Hit], PaperScope | None]:
        """mode: hybrid (dense + keyword fused with RRF; the default), dense, or keyword. With `scope`, a
        question that names one paper is searched within that paper, using the question minus the
        reference (dense search otherwise gets pulled toward every passage that mentions the nickname)."""
        target = self.resolve_reference(query) if scope else None
        rankings, sims = [], {}
        # Global rankings for the full question, plus (if a paper was named) rankings inside that paper for
        # the question without the reference. SCOPE_MODE "restrict" uses only the latter; "boost" fuses both.
        plans = [(query, None)] if (not target or SCOPE_MODE == "boost") else []
        if target:
            plans.append((target.rest or query, target.doc_id))
        for q, doc_id in plans:
            if mode in ("hybrid", "dense"):
                dense = self.dense_ranking(q, embedder, doc_id=doc_id)
                for cid, s in dense:
                    sims[cid] = max(s, sims.get(cid, s))
                rankings.append([cid for cid, _ in dense])
            if mode in ("hybrid", "keyword"):
                rankings.append(self.keyword_ranking(q, doc_id=doc_id))
        scores = rrf([r for r in rankings if r])
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [h for cid in best if (h := self.hit(cid, scores[cid], sims.get(cid)))], target

    def search(self, query: str, embedder: Embedder, k: int = 5, mode: str = "hybrid",
               scope: bool = True) -> list[Hit]:
        return self.search_with_scope(query, embedder, k, mode, scope)[0]

    def hit(self, chunk_id: int, score: float = 0.0, similarity: float | None = None) -> Hit | None:
        row = self.db.execute("""SELECT c.id, c.doc_id, d.title, c.page, c.text, c.section FROM chunks c
                                 JOIN docs d ON d.id = c.doc_id WHERE c.id=?""", (chunk_id,)).fetchone()
        return Hit(*row, score=score, similarity=similarity) if row else None


# ---------------------------------------------------------------- tools + citation checking

@dataclass
class CitationReport:
    cited: list[tuple[str, Hit]]           # labels the reply used that were really retrieved
    unknown: list[str]                     # labels that were never retrieved (fabricated)
    unsupported: list[tuple[str, str]] = field(default_factory=list)   # (number, labels) not in the cited text


# Numbers worth checking: decimals, percentages, and integers >= 10 (small integers are often written
# as words in papers: "two tasks").
_CLAIM_NUM = re.compile(r"(?<![\w.\[])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{2,})(\s*%)?(?![\w\]])")


def _norm_numbers(text: str) -> set[str]:
    return {m.group(1).replace(",", "") for m in _CLAIM_NUM.finditer(text)}


def unsupported_numbers(reply: str, passages: dict[str, str]) -> list[tuple[str, str]]:
    """For each sentence that cites [Lk] labels, numbers that appear in none of the cited passages.

    Found in the live test: the model answered from memory ("learning rate 0.0001") and attached real but
    irrelevant labels ([L4] = the LoRA introduction). Checking that a label was retrieved is not enough;
    this checks that the cited text actually contains the numbers it is cited for."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", reply):
        labels = [l for l in LibrarySession.LABEL.findall(sentence) if l in passages]
        if not labels:
            continue
        source = _norm_numbers(" ".join(passages[l] for l in labels))
        for num in sorted(_norm_numbers(LibrarySession.LABEL.sub("", sentence))):
            if num not in source:
                out.append((num, ", ".join(f"[{l}]" for l in labels)))
    return out


class LibrarySession:
    """Per-chat state: labels passages shown to the model and checks the labels it cites."""

    # [L1] as instructed; the 9B sometimes writes (L1). Only labels that were retrieved count as cited.
    LABEL = re.compile(r"[\[(](L\d+)[\])]")

    def __init__(self, library: Library, embedder: Embedder):
        self.library = library
        self.embedder = embedder
        self.labels: dict[str, Hit] = {}
        self._by_chunk: dict[int, str] = {}

    def _label(self, hit: Hit) -> str:
        if hit.chunk_id not in self._by_chunk:
            label = f"L{len(self.labels) + 1}"
            self.labels[label] = hit
            self._by_chunk[hit.chunk_id] = label
        return self._by_chunk[hit.chunk_id]

    def search_library(self, query: str, k: int = 5) -> str:
        k = max(1, min(int(k), 8))
        try:
            hits = self.library.search(query, self.embedder, k=k)
        except LLMError as e:
            return f"error: embedding server problem: {e}"
        except LibraryError as e:
            return f"error: {e}"
        if not hits:
            return "no matching passages in the library"
        parts = []
        for h in hits:
            text = h.text if len(h.text) <= MAX_PASSAGE_CHARS else h.text[:MAX_PASSAGE_CHARS] + "..."
            parts.append(f"[{self._label(h)}] {h.where}\n{text}")
        return ("Cite these passages inline by their label, e.g. [L1]. Use only what they say.\n\n"
                + "\n\n".join(parts))

    def list_library(self) -> str:
        docs = self.library.documents()
        if not docs:
            return "the library is empty"
        return "\n".join(f"- {title} ({pages} pages)" if pages else f"- {title} (notes)"
                         for _, title, pages, *_ in docs)

    def tools(self) -> list[Tool]:
        return [
            Tool("search_library",
                 "Search the student's own library of papers and notes (local). Returns passages labelled "
                 "[L1], [L2], ... with title and page. Use it for any question about their papers or notes, "
                 "and cite the labels inline in your answer.",
                 {"type": "object", "properties": {"query": {"type": "string", "description": "what to look for"},
                                                   "k": {"type": "integer", "description": "passages, 1-8, default 5"}},
                  "required": ["query"]},
                 self.search_library),
            Tool("list_library", "List the titles of the papers and notes in the student's library.",
                 {"type": "object", "properties": {}}, self.list_library),
        ]

    def prefetch(self, user_text: str, k: int = config.PREFETCH_K, min_similarity: float | None = None
                 ) -> tuple[list[dict], list[Hit]]:
        """Retrieval-first RAG for small models: search the library for the user's message and, if the
        best passage is relevant enough, return a synthetic search_library call + result to insert into
        the turn. Phase 3 eval: left to itself, the 9B model answered famous-paper questions from memory
        and never searched (0/10)."""
        if min_similarity is None:
            min_similarity = (AUTO_RETRIEVE_MIN_SIM_EXPLICIT if REFERS_TO_LIBRARY.search(user_text)
                              else AUTO_RETRIEVE_MIN_SIM)
        try:
            hits, scope = self.library.search_with_scope(user_text, self.embedder, k=k)
        except (LLMError, LibraryError):
            return [], []
        best = max((h.similarity for h in hits if h.similarity is not None), default=0.0)
        # A question that names a paper in the library is about the library, whatever the similarity.
        if not hits or (scope is None and best < min_similarity):
            return [], []
        call_id = f"prefetch_{len(self.labels) + 1}"
        body = "\n\n".join(f"[{self._label(h)}] {h.where}\n"
                           + (h.text if len(h.text) <= MAX_PASSAGE_CHARS else h.text[:MAX_PASSAGE_CHARS] + "...")
                           for h in hits)
        result = ("Passages from the student's own library (their copies of the papers; no need to look these "
                  "up elsewhere). Answer from them if they cover the question and cite them inline by label, "
                  "e.g. [L1]. If they do not answer it, say so and answer normally.\n\n"
                  + body)
        messages = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "search_library", "arguments": json.dumps({"query": user_text})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": result},
        ]
        return messages, hits

    def check(self, reply: str) -> CitationReport:
        cited, unknown = [], []
        for label in dict.fromkeys(self.LABEL.findall(reply)):
            if label in self.labels:
                cited.append((label, self.labels[label]))
            else:
                unknown.append(label)
        passages = {label: f"{hit.section or ''} {hit.text}" for label, hit in self.labels.items()}
        return CitationReport(cited, unknown, unsupported_numbers(reply, passages))
