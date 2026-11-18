"""arXiv search via the public export API (no key). Sends the query text to arxiv.org."""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import Tool

API = "https://export.arxiv.org/api/query"
USER_AGENT = "ALICE-local-study-assistant/0.1"      # generic: no personal details
MIN_INTERVAL_S = 3.0                                 # arXiv API etiquette
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_last_call = 0.0


def _fetch(params: dict) -> str:
    global _last_call
    wait = MIN_INTERVAL_S - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode("utf-8")
    finally:
        _last_call = time.time()


def parse_feed(xml_text: str, summary_chars: int = 500) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", "", NS) or "").strip()
        if "/abs/" not in raw_id:                     # error entries have no abs link
            continue
        arxiv_id = raw_id.split("/abs/")[-1]
        clean = lambda s: re.sub(r"\s+", " ", s or "").strip()
        summary = clean(e.findtext("a:summary", "", NS))
        papers.append({
            "id": arxiv_id,
            "title": clean(e.findtext("a:title", "", NS)),
            "authors": [clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)],
            "published": (e.findtext("a:published", "", NS) or "")[:10],
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "summary": summary if len(summary) <= summary_chars else summary[:summary_chars] + "...",
        })
    return papers


def _format(papers: list[dict]) -> str:
    if not papers:
        return "no results on arXiv"
    out = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:4]) + (" et al." if len(p["authors"]) > 4 else "")
        out.append(f"[{i}] {p['title']}\n    {authors} ({p['published']}) arXiv:{p['id']}  {p['url']}\n"
                   f"    {p['summary']}")
    return "\n".join(out)


STOPWORDS = set("""a an and are as at be by for from how in is it of on or that the this to was what
with paper papers about using via""".split())


def query_strategies(query: str) -> list[tuple[str, str]]:
    """arXiv search_query strings to try in order, with a label for what a hit means.

    ANDing every word fails: arXiv drops stopwords, so 'all:Is AND all:All ...' matches nothing
    (found when searching 'Attention Is All You Need' returned no results)."""
    words = re.findall(r"[\w\-]+", query)
    content = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    out = []
    if 2 <= len(words) <= 15:
        phrase = " ".join(words).replace('"', "")
        out.append((f'ti:"{phrase}"', "exact title match"))
    if content:
        out.append((" AND ".join(f"all:{w}" for w in content[:8]), "keyword match"))
    if len(content) > 3:
        out.append((" AND ".join(f"all:{w}" for w in content[:3]), "partial keyword match"))
    return out


def search(query: str, max_results: int = 5, sort: str = "relevance") -> str:
    max_results = max(1, min(int(max_results), 10))
    sort_by = {"relevance": "relevance", "newest": "submittedDate"}.get(sort, "relevance")
    strategies = query_strategies(query)
    if not strategies:
        return "error: empty query"
    tried = []
    for search_query, label in strategies:
        try:
            papers = parse_feed(_fetch({"search_query": search_query, "start": 0, "max_results": max_results,
                                        "sortBy": sort_by, "sortOrder": "descending"}))
        except OSError as e:
            return f"error: arXiv request failed ({e})"
        if papers:
            note = f"({label}" + (f"; no {', no '.join(tried)}" if tried else "") + ")"
            return note + "\n" + _format(papers)
        tried.append(label)
    return "no results on arXiv (tried: " + ", ".join(tried) + ")"


def paper(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip().removeprefix("arXiv:").removeprefix("https://arxiv.org/abs/")
    if not re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?", arxiv_id):
        return f"error: '{arxiv_id}' is not a valid arXiv id"
    try:
        papers = parse_feed(_fetch({"id_list": arxiv_id}), summary_chars=2000)
    except OSError as e:
        return f"error: arXiv request failed ({e})"
    return _format(papers) if papers else f"no arXiv paper with id {arxiv_id}"


SEARCH = Tool(
    name="arxiv_search",
    description=("Search arXiv for papers (title, authors, date, id, abstract). Use it to find papers and "
                 "to verify that a paper or citation really exists before citing it. Keywords work best, "
                 "e.g. 'attention is all you need' or 'LoRA low-rank adaptation'."),
    parameters={"type": "object",
                "properties": {"query": {"type": "string"},
                               "max_results": {"type": "integer", "description": "1-10, default 5"},
                               "sort": {"type": "string", "enum": ["relevance", "newest"]}},
                "required": ["query"]},
    run=search, external=True,
)

PAPER = Tool(
    name="arxiv_paper",
    description="Fetch one arXiv paper's metadata and full abstract by id (e.g. '1706.03762').",
    parameters={"type": "object", "properties": {"arxiv_id": {"type": "string"}}, "required": ["arxiv_id"]},
    run=paper, external=True,
)
