"""Reading documents the student hands ALICE: PDF, Word (.docx), PowerPoint (.pptx), plain text and Markdown.

Word and PowerPoint files are zip archives of XML (Office Open XML), so they are read with the standard library:
paragraphs, headings, list items and tables from .docx; every slide's text in presentation order plus its
speaker notes from .pptx. Old binary .doc / .ppt files need converting first (Save As .docx/.pptx or PDF).

A Document is a list of units - PDF pages, slides, or ~3,000-character parts of a Word file - each with a label
("page 3", "slide 2: Gradient descent", "part 1") so answers can point at where something is.
"""
from __future__ import annotations

import math
import re
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}
LEGACY = {".doc": "Word", ".ppt": "PowerPoint", ".odt": "OpenDocument", ".odp": "OpenDocument"}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
DOCX_PART_CHARS = 3000


class DocumentError(ValueError):
    """The file can't be read; the message is user-readable."""


@dataclass
class Unit:
    label: str            # "page 3", "slide 2: Title", "part 1"
    text: str
    number: int           # page / slide / part number
    heading: str | None = None


@dataclass
class Document:
    path: Path
    title: str
    kind: str             # "pdf", "docx", "pptx", "text"
    units: list[Unit] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(u.text) for u in self.units)

    @property
    def unit_name(self) -> str:
        return {"pdf": "page", "pptx": "slide"}.get(self.kind, "part")

    def describe(self) -> str:
        n = len(self.units)
        return f"{self.title} ({n} {self.unit_name}{'s' * (n != 1)}, {self.path.suffix.lower()[1:].upper()})"

    def text(self, start: int = 1, end: int | None = None) -> str:
        """Units start..end (1-based, inclusive) with their labels."""
        chosen = self.units[max(0, start - 1):end]
        return "\n\n".join(f"--- {u.label} ---\n{u.text}" for u in chosen if u.text.strip())


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read(path: str | Path) -> Document:
    """Read a document into units. Raises DocumentError with a readable message."""
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if not p.is_file():
        raise DocumentError(f"no such file: {p}")
    if suffix in LEGACY:
        raise DocumentError(f"{p.name} is an old {LEGACY[suffix]} file; save it as "
                            f"{'.docx' if suffix in ('.doc', '.odt') else '.pptx'} or PDF and try again")
    if suffix not in SUPPORTED:
        raise DocumentError(f"can't read {suffix or 'files without an extension'} files "
                            f"(supported: {', '.join(sorted(SUPPORTED))})")
    try:
        if suffix == ".pdf":
            return _read_pdf(p)
        if suffix == ".docx":
            return _read_docx(p)
        if suffix == ".pptx":
            return _read_pptx(p)
        text = _clean(p.read_text(encoding="utf-8", errors="replace"))
        return Document(p, _first_line(text) or p.stem, "text", _split_parts(text, "part"))
    except DocumentError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as e:
        raise DocumentError(f"{p.name} looks damaged or isn't really a {suffix} file ({type(e).__name__})") from None
    except Exception as e:                                  # pypdf and friends: report, don't crash the chat
        raise DocumentError(f"couldn't read {p.name}: {type(e).__name__}: {e}") from None


def _first_line(text: str) -> str:
    return next((l.lstrip("# ").strip() for l in text.splitlines() if l.strip()), "")[:120]


def _split_parts(text: str, name: str, size: int = DOCX_PART_CHARS) -> list[Unit]:
    """~size-character parts, split at paragraph boundaries; each keeps the last heading seen."""
    units, buf, heading, part_heading = [], [], None, None
    for para in text.split("\n"):
        if para.startswith("#"):
            heading = para.lstrip("# ").strip() or heading
        if buf and sum(len(b) + 1 for b in buf) + len(para) > size:
            units.append(Unit(f"{name} {len(units) + 1}", "\n".join(buf).strip(), len(units) + 1, part_heading))
            buf, part_heading = [], heading
        if not buf:
            part_heading = heading
        buf.append(para)
    if "".join(buf).strip():
        units.append(Unit(f"{name} {len(units) + 1}", "\n".join(buf).strip(), len(units) + 1, part_heading))
    for u in units:
        if u.heading:
            u.label = f"{u.label} ({u.heading[:60]})"
    return units


# ----- PDF
def _read_pdf(path: Path) -> Document:
    import logging
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise DocumentError(f"{path.name} is password-protected") from None
    units = []
    for i, page in enumerate(reader.pages, 1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        units.append(Unit(f"page {i}", _clean(raw), i))
    if units and not any(u.text for u in units):
        raise DocumentError(f"{path.name} has no text layer (it's probably scanned images); OCR it first")
    # the library's title detection skips licence lines and arXiv stamps ("Provided proper attribution ...")
    from .library import _first_line_title, _pdf_title
    try:
        title = _pdf_title(reader) or _first_line_title(reader.pages[0].extract_text() or "")
    except Exception:
        title = None
    return Document(path, title or _first_line(units[0].text if units else "") or path.stem, "pdf", units)


# ----- Word
def _core_title(z: zipfile.ZipFile) -> str:
    try:
        root = ET.fromstring(z.read("docProps/core.xml"))
    except KeyError:
        return ""
    el = root.find("{http://purl.org/dc/elements/1.1/}title")
    return (el.text or "").strip() if el is not None else ""


def _para_text(p: ET.Element) -> str:
    out = []
    for el in p.iter():
        if el.tag == W + "t" and el.text:
            out.append(el.text)
        elif el.tag == W + "tab":
            out.append("\t")
        elif el.tag in (W + "br", W + "cr"):
            out.append("\n")
    return "".join(out)


def _read_docx(path: Path) -> Document:
    with zipfile.ZipFile(path) as z:
        body = ET.fromstring(z.read("word/document.xml")).find(W + "body")
        title = _core_title(z)
    lines: list[str] = []
    for block in (body if body is not None else []):
        if block.tag == W + "p":
            text = _para_text(block).strip()
            if not text:
                continue
            style = block.find(f"{W}pPr/{W}pStyle")
            sval = (style.get(W + "val") or "") if style is not None else ""
            level = re.match(r"(?i)heading\s*(\d)", sval) or re.match(r"(?i)title", sval)
            if level:
                depth = int(level.group(1)) if level.groups() else 1
                lines.append("#" * depth + " " + text)
            elif block.find(f"{W}pPr/{W}numPr") is not None:
                lines.append("- " + text)
            else:
                lines.append(text)
        elif block.tag == W + "tbl":
            for row in block.iter(W + "tr"):
                cells = [" ".join(_para_text(p).strip() for p in tc.iter(W + "p")).strip()
                         for tc in row.iter(W + "tc")]
                if any(cells):
                    lines.append("| " + " | ".join(cells) + " |")
    text = _clean("\n".join(lines))
    if not text:
        raise DocumentError(f"{path.name} contains no text")
    heading = next((l.lstrip("# ").strip() for l in lines if l.startswith("#")), "")
    return Document(path, title or heading or _first_line(text) or path.stem, "docx", _split_parts(text, "part"))


# ----- PowerPoint
def _rels(z: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """rId -> (type, resolved part path) for a part's relationships."""
    p = PurePosixPath(part)
    rels_path = str(p.parent / "_rels" / (p.name + ".rels"))
    try:
        root = ET.fromstring(z.read(rels_path))
    except KeyError:
        return {}
    out = {}
    for rel in root.iter(PKG_REL + "Relationship"):
        target = rel.get("Target", "")
        resolved = target.lstrip("/") if target.startswith("/") else str(_norm(p.parent / target))
        out[rel.get("Id")] = (rel.get("Type", "").rsplit("/", 1)[-1], resolved)
    return out


def _norm(p: PurePosixPath) -> PurePosixPath:
    parts: list[str] = []
    for seg in p.parts:
        if seg == "..":
            if parts:
                parts.pop()
        elif seg != ".":
            parts.append(seg)
    return PurePosixPath(*parts)


def _shape_paragraphs(root: ET.Element) -> list[tuple[bool, str]]:
    """(is_title, text) for each text paragraph, in document order."""
    out = []
    for sp in root.iter(P + "sp"):
        ph = sp.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
        is_title = ph is not None and ph.get("type") in ("title", "ctrTitle")
        for para in sp.iter(A + "p"):
            text = "".join(t.text or "" for t in para.iter(A + "t")).strip()
            if text:
                out.append((is_title, text))
    for frame in root.iter(P + "graphicFrame"):                    # tables
        for row in frame.iter(A + "tr"):
            cells = [" ".join((t.text or "").strip() for t in tc.iter(A + "t")).strip() for tc in row.iter(A + "tc")]
            if any(cells):
                out.append((False, "| " + " | ".join(cells) + " |"))
    return out


def _read_pptx(path: Path) -> Document:
    with zipfile.ZipFile(path) as z:
        pres = ET.fromstring(z.read("ppt/presentation.xml"))
        pres_rels = _rels(z, "ppt/presentation.xml")
        order = [pres_rels[s.get(R + "id")][1] for s in pres.iter(P + "sldId") if s.get(R + "id") in pres_rels]
        if not order:                                              # fall back to file names
            order = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                           key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
        title = _core_title(z)
        units = []
        for i, part in enumerate(order, 1):
            paras = _shape_paragraphs(ET.fromstring(z.read(part)))
            slide_title = next((t for is_t, t in paras if is_t), "")
            body = [t for is_t, t in paras if not is_t]
            notes = []
            for kind, target in _rels(z, part).values():
                if kind == "notesSlide" and target in z.namelist():
                    notes = [t for _, t in _shape_paragraphs(ET.fromstring(z.read(target)))
                             if not t.isdigit()]                   # notes pages repeat the slide number
            text = "\n".join(body)
            if notes:
                text += "\nSpeaker notes: " + " ".join(notes)
            label = f"slide {i}" + (f": {slide_title[:60]}" if slide_title else "")
            units.append(Unit(label, _clean(text), i, slide_title or None))
    if not any(u.text or u.heading for u in units):
        raise DocumentError(f"{path.name} contains no text")
    first_title = next((u.heading for u in units if u.heading), "")
    return Document(path, title or first_title or path.stem, "pptx", units)


# ----- finding the relevant parts of a long document
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by for from has have how i in is it its of on or that the this to was what
when where which who why will with you your do does did can could should would about into than then them they
their there these those not no yes we our us me my more most also just one two""".split())


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class Excerpts:
    """BM25 keyword search over a document's units (split into ~1,500-character pieces); no embedding model
    needed, so it works the moment a document is opened, even while the chat model holds the GPU."""

    def __init__(self, doc: Document, piece_chars: int = 1500):
        self.pieces: list[tuple[str, str]] = []
        for u in doc.units:
            text = u.text
            while text.strip():
                if len(text) <= piece_chars:
                    cut = len(text)
                else:                                       # break at a line end, unless that leaves a tiny piece
                    nl = text.rfind("\n", 0, piece_chars)
                    cut = nl + 1 if nl > piece_chars // 2 else piece_chars
                self.pieces.append((u.label, text[:cut].strip()))
                text = text[cut:]
        self.terms = [Counter(_terms(t) + _terms(label)) for label, t in self.pieces]
        self.avg = sum(sum(c.values()) for c in self.terms) / max(1, len(self.terms))
        df = Counter(w for c in self.terms for w in c)
        n = len(self.terms)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def top(self, query: str, k: int = 4) -> list[tuple[str, str]]:
        q = set(_terms(query))
        scored = []
        for i, c in enumerate(self.terms):
            length = sum(c.values()) or 1
            s = sum(self.idf.get(w, 0) * c[w] * 2.2 / (c[w] + 1.2 * (0.25 + 0.75 * length / self.avg))
                    for w in q if w in c)
            if s > 0:
                scored.append((s, i))
        best = sorted(scored, reverse=True)[:k]
        return [self.pieces[i] for _, i in sorted(best, key=lambda x: x[1])]      # in document order
