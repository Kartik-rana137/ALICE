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
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config
from .llm import LLMError, ServerProcess, _post
from .tools import Tool

QUERY_INSTRUCTION = ("Given a question from a computer-science student, retrieve passages from "
                     "research papers or notes that answer it")
CHUNK_CHARS = 900
MAX_PASSAGE_CHARS = 1000
RRF_K = 60
CANDIDATES = 30
# Top-1 cosine similarity needed before passages are inserted automatically. Calibrated on the eval
# library (Qwen3-Embedding-0.6B): paper questions 0.68-0.83, unrelated CS questions 0.22-0.58.
AUTO_RETRIEVE_MIN_SIM = float(os.environ.get("ALICE_RETRIEVE_MIN_SIM", "0.65"))
# Similarity alone can't tell "a question about my papers" from "a general question on a topic my papers
# cover" (a paraphrased Transformer-paper question scored 0.605, generic ML questions up to 0.58). When the
# message itself points at papers/notes, a lower bar applies.
AUTO_RETRIEVE_MIN_SIM_EXPLICIT = 0.50
REFERS_TO_LIBRARY = re.compile(r"\b(papers?|articles?|library|notes|according to|authors?|my (pdfs?|reading))\b",
                               re.I)
SUPPORTED = {".pdf", ".md", ".txt"}
STOPWORDS = set("""a an and are as at be by does do for from how in is it of on or that the this to was
what when where which who why with did can you your about paper papers""".split())


# ---------------------------------------------------------------- embeddings

class Embedder(Protocol):
    def embed(self, texts: list[str], query: bool = False) -> np.ndarray: ...


class LlamaEmbedder:
    """Qwen3-Embedding in its own llama-server (CPU by default so it never competes for VRAM)."""

    def __init__(self, gpu: bool = False, gguf: Path = config.EMBED_GGUF, port: int = config.EMBED_PORT):
        cmd = [str(config.SERVER_EXE), "-m", str(gguf), "--embedding", "--pooling", "last",
               "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "1",
               "--host", config.HOST, "--port", str(port), "-ngl", "99" if gpu else "0"]
        env = dict(os.environ)
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"           # keep the CUDA build entirely off the GPU
        self.server = ServerProcess(cmd, port, gguf, "embed-server.log", env=env)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.server.start(wait_s=180)
            self._started = True

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


def extract(path: Path) -> tuple[str, list[tuple[int | None, str]]]:
    """(title, [(page_number, cleaned_text), ...]). Page is None for text notes."""
    if path.suffix.lower() == ".pdf":
        import logging
        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)    # font-encoding chatter, harmless for text
        reader = PdfReader(str(path))
        pages, first_raw = [], ""
        for i, page in enumerate(reader.pages, 1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            if i == 1:
                first_raw = raw
            pages.append((i, clean_text(raw)))
        return (_pdf_title(reader) or _first_line_title(first_raw) or path.stem), pages
    text = path.read_text(encoding="utf-8", errors="replace")
    first = next((l.lstrip("# ").strip() for l in text.splitlines() if l.strip()), path.stem)
    return first[:120], [(None, clean_text(text))]


# ---------------------------------------------------------------- store and search

@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    title: str
    page: int | None
    text: str
    score: float = 0.0                      # fused (RRF) score
    similarity: float | None = None         # dense cosine similarity, if the chunk was a dense candidate

    @property
    def where(self) -> str:
        return f"{self.title}, p. {self.page}" if self.page else self.title


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
    def __init__(self, directory: Path = config.LIBRARY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "library.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, path TEXT, sha256 TEXT UNIQUE,
                title TEXT, pages INTEGER, added TEXT);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER,
                ord INTEGER, text TEXT, embedding BLOB);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
        """)
        self._matrix: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    # ----- ingestion
    def add(self, path: Path, embedder: Embedder, title: str | None = None) -> tuple[str, int, str]:
        """Ingest one file. Returns (status, n_chunks, title); status is 'added', 'skipped' or 'empty'."""
        path = Path(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        row = self.db.execute("SELECT title FROM docs WHERE sha256=?", (sha,)).fetchone()
        if row:
            return "skipped", 0, row[0]
        auto_title, pages = extract(path)
        title = title or auto_title
        items = [(page, c) for page, text in pages for c in chunk_text(text)]
        if not items:
            return "empty", 0, title
        vectors = embedder.embed([f"{title}\n{c}" for _, c in items])
        cur = self.db.execute("INSERT INTO docs(path, sha256, title, pages, added) VALUES (?,?,?,?,?)",
                              (str(path), sha, title, len(pages), time.strftime("%Y-%m-%d")))
        doc_id = cur.lastrowid
        for n, ((page, text), vec) in enumerate(zip(items, vectors)):
            c = self.db.execute("INSERT INTO chunks(doc_id, page, ord, text, embedding) VALUES (?,?,?,?,?)",
                                (doc_id, page, n, text, vec.astype(np.float32).tobytes()))
            self.db.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (c.lastrowid, text))
        self.db.commit()
        self._matrix = None
        return "added", len(items), title

    def remove(self, doc_id: int) -> bool:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        n = self.db.execute("DELETE FROM docs WHERE id=?", (doc_id,)).rowcount
        self.db.commit()
        self._matrix = None
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
    def _load_matrix(self) -> None:
        rows = self.db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        self._ids = np.array([r[0] for r in rows], dtype=np.int64)
        self._matrix = (np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                        if rows else np.zeros((0, 1), dtype=np.float32))

    def dense_ranking(self, query: str, embedder: Embedder, n: int = CANDIDATES) -> list[tuple[int, float]]:
        """[(chunk_id, cosine similarity)] best first."""
        if self._matrix is None:
            self._load_matrix()
        if len(self._ids) == 0:
            return []
        q = embedder.embed([query], query=True)[0]
        sims = self._matrix @ q
        top = np.argsort(-sims)[:n]
        return [(int(self._ids[i]), float(sims[i])) for i in top]

    def keyword_ranking(self, query: str, n: int = CANDIDATES) -> list[int]:
        q = fts_query(query)
        if not q:
            return []
        try:
            rows = self.db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) "
                                   "LIMIT ?", (q, n)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]

    def search(self, query: str, embedder: Embedder, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        rankings, sims = [], {}
        if mode in ("hybrid", "dense"):
            dense = self.dense_ranking(query, embedder)
            sims = dict(dense)
            rankings.append([cid for cid, _ in dense])
        if mode in ("hybrid", "keyword"):
            rankings.append(self.keyword_ranking(query))
        scores = rrf([r for r in rankings if r])
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [h for cid in best if (h := self.hit(cid, scores[cid], sims.get(cid)))]

    def hit(self, chunk_id: int, score: float = 0.0, similarity: float | None = None) -> Hit | None:
        row = self.db.execute("""SELECT c.id, c.doc_id, d.title, c.page, c.text FROM chunks c
                                 JOIN docs d ON d.id = c.doc_id WHERE c.id=?""", (chunk_id,)).fetchone()
        return Hit(*row, score=score, similarity=similarity) if row else None


# ---------------------------------------------------------------- tools + citation checking

@dataclass
class CitationReport:
    cited: list[tuple[str, Hit]]           # labels the reply used that were really retrieved
    unknown: list[str]                     # labels that were never retrieved (fabricated)


class LibrarySession:
    """Per-chat state: labels passages shown to the model and checks the labels it cites."""

    LABEL = re.compile(r"\[(L\d+)\]")

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

    def prefetch(self, user_text: str, k: int = 4, min_similarity: float | None = None
                 ) -> tuple[list[dict], list[Hit]]:
        """Retrieval-first RAG for small models: search the library for the user's message and, if the
        best passage is relevant enough, return a synthetic search_library call + result to insert into
        the turn. Phase 3 eval: left to itself, the 9B model answered famous-paper questions from memory
        and never searched (0/10)."""
        if min_similarity is None:
            min_similarity = (AUTO_RETRIEVE_MIN_SIM_EXPLICIT if REFERS_TO_LIBRARY.search(user_text)
                              else AUTO_RETRIEVE_MIN_SIM)
        try:
            hits = self.library.search(user_text, self.embedder, k=k)
        except LLMError:
            return [], []
        best = max((h.similarity for h in hits if h.similarity is not None), default=0.0)
        if not hits or best < min_similarity:
            return [], []
        call_id = f"prefetch_{len(self.labels) + 1}"
        body = "\n\n".join(f"[{self._label(h)}] {h.where}\n"
                           + (h.text if len(h.text) <= MAX_PASSAGE_CHARS else h.text[:MAX_PASSAGE_CHARS] + "...")
                           for h in hits)
        result = ("Library passages relevant to the question. Answer from these if they cover it and cite them "
                  "inline by label, e.g. [L1]. If they do not answer the question, say so and answer normally.\n\n"
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
        return CitationReport(cited, unknown)
