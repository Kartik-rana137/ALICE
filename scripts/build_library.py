"""Build ALICE's starter library: AI/CS papers from arXiv plus open, bachelor-level science textbooks.

Nothing here is invented: every arXiv id is checked against arXiv's own metadata (the title must contain the
expected words) before it is downloaded, and every textbook URL comes from its publisher's catalogue
(OpenStax: CC BY-NC-SA; MIT "Mathematics for Computer Science": CC BY-SA; Nature open access: CC BY) - fine
for personal study, not for redistribution, which is why the PDFs stay in data/papers/ (gitignored) and only
this list is in the repo. Research papers in maths, physics, chemistry and biology are mostly far beyond a
bachelor's level, so those subjects get textbooks plus a few landmark open papers.

    python scripts/build_library.py --dry-run          # verify the list against arXiv, download nothing
    python scripts/build_library.py                    # download + ingest everything (re-runs skip what exists)
    python scripts/build_library.py --only ai physics  # some subjects
    python scripts/build_library.py --no-books         # papers only (the textbooks are 50-220 MB each)

Ingestion embeds on the GPU when the chat model is not running (stop ALICE first): papers take seconds,
a 1,000-page textbook several minutes.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PAPERS_DIR = ROOT / "data" / "papers"
USER_AGENT = "ALICE-local-study-assistant/0.1"           # generic: no personal details

# subject -> [(arXiv id, words the real title must contain)]
ARXIV = {
    "ai": [
        # optimisation, normalisation, regularisation, tutorials
        ("1412.6980", "adam"), ("1609.04747", "gradient descent optimization"),
        ("1502.03167", "batch normalization"), ("1607.06450", "layer normalization"),
        ("1207.0580", "co-adaptation"), ("1503.02531", "distilling the knowledge"),
        ("1802.01528", "matrix calculus"), ("1603.07285", "convolution arithmetic"),
        ("1404.1100", "principal component analysis"), ("1606.05908", "variational autoencoders"),
        # vision
        ("1409.1556", "very deep convolutional"), ("1512.03385", "deep residual learning"),
        ("1505.04597", "u-net"), ("1506.02640", "you only look once"), ("1704.04861", "mobilenets"),
        ("1905.11946", "efficientnet"), ("2010.11929", "16x16 words"),
        # language and sequences
        ("1301.3781", "efficient estimation of word representations"), ("1409.3215", "sequence to sequence"),
        ("1409.0473", "jointly learning to align and translate"), ("1412.3555", "gated recurrent"),
        ("1508.07909", "subword units"), ("1706.03762", "attention is all you need"), ("1810.04805", "bert"),
        ("2005.14165", "few-shot learners"), ("2001.08361", "scaling laws"), ("2201.11903", "chain-of-thought"),
        ("2005.11401", "retrieval-augmented generation"), ("2203.02155", "human feedback"),
        ("2302.13971", "llama"), ("2106.09685", "lora"),
        # generative models
        ("1406.2661", "generative adversarial"), ("1312.6114", "auto-encoding variational bayes"),
        ("1511.06434", "deep convolutional generative adversarial"), ("2006.11239", "denoising diffusion"),
        ("2103.00020", "natural language supervision"),
        # reinforcement learning, graphs, classical ML
        ("1312.5602", "playing atari"), ("1707.06347", "proximal policy optimization"),
        ("1609.02907", "graph convolutional networks"), ("1603.02754", "xgboost"),
        # interpretability, safety, fairness
        ("1602.04938", "why should i trust you"), ("1705.07874", "interpreting model predictions"),
        ("1606.06565", "concrete problems in ai safety"), ("1908.09635", "bias and fairness"),
    ],
    # Left out after checking: 1910.11333 is only the supplementary information of the quantum-supremacy
    # paper, and the ATLAS Higgs paper (1207.7214) is mostly a multi-page author list.
    "physics": [("1602.03837", "gravitational waves")],
    "chemistry": [("1704.01212", "neural message passing for quantum chemistry")],
    "math": [("math/0702396", "what is good mathematics")],
}

OPENSTAX = "https://assets.openstax.org/oscms-prodcms/media/documents/"
# (subject, title, url, licence)
BOOKS = [
    ("math", "Mathematics for Computer Science (Lehman, Leighton, Meyer; MIT)",
     "https://courses.csail.mit.edu/6.042/spring18/mcs.pdf", "CC BY-SA 3.0"),
    ("math", "Calculus Volume 1 (OpenStax)", OPENSTAX + "calculus-volume-1_-_WEB.pdf", "CC BY-NC-SA"),
    ("math", "Calculus Volume 2 (OpenStax)", OPENSTAX + "calculus-volume-2_-_WEB.pdf", "CC BY-NC-SA"),
    ("math", "Calculus Volume 3 (OpenStax)", OPENSTAX + "calculus-volume-3_-_WEB.pdf", "CC BY-NC-SA"),
    ("math", "Introductory Statistics 2e (OpenStax)", OPENSTAX + "introductory-statistics-2e_-_WEB.pdf", "CC BY-NC-SA"),
    ("cs", "Introduction to Computer Science (OpenStax)", OPENSTAX + "Introduction_To_Computer_Science_-_WEB.pdf",
     "CC BY-NC-SA"),
    ("cs", "Introduction to Python Programming (OpenStax)", OPENSTAX + "Introduction_to_Python_Programming-WEB.pdf",
     "CC BY-NC-SA"),
    ("cs", "Principles of Data Science (OpenStax)", OPENSTAX + "Principles-of-Data-Science-WEB.pdf", "CC BY-NC-SA"),
    ("physics", "University Physics Volume 1 (OpenStax)", OPENSTAX + "university-physics-volume-1_-_WEB.pdf",
     "CC BY-NC-SA"),
    ("physics", "University Physics Volume 2 (OpenStax)", OPENSTAX + "university-physics-volume-2_-_WEB.pdf",
     "CC BY-NC-SA"),
    ("physics", "University Physics Volume 3 (OpenStax)", OPENSTAX + "university-physics-volume-3_-_WEB.pdf",
     "CC BY-NC-SA"),
    ("chemistry", "Chemistry 2e (OpenStax)", OPENSTAX + "chemistry-2e_-_WEB.pdf", "CC BY-NC-SA"),
    ("biology", "Biology 2e (OpenStax)", OPENSTAX + "Biology-2e_-_WEB.pdf", "CC BY-NC-SA"),
    # First-year engineering syllabus (added 2026-11-20). Already covered by the books above: multivariable and
    # vector calculus, sequences/series and power series (Calculus 2-3), engineering physics, mechanics and
    # thermodynamics (University Physics 1-3), electrochemistry/batteries/corrosion (Chemistry 2e), cell
    # biology and ecology (Biology 2e), Python basics (Introduction to Python Programming).
    ("math", "Interactive Linear Algebra (Margalit, Rabinoff)", "https://textbooks.math.gatech.edu/ila/ila.pdf",
     "GFDL"),
    ("math", "Linear Algebra (Cherney, Denton, Waldron; UC Davis)",
     "https://www.math.ucdavis.edu/~linear/linear-guest.pdf", "free textbook"),
    ("math", "Notes on Diffy Qs: Differential Equations for Engineers (Lebl)", "https://www.jirka.org/diffyqs/diffyqs.pdf",
     "CC BY-SA"),
    ("math", "Contemporary Mathematics (OpenStax)", OPENSTAX + "contemporary-mathematics_-_WEB.pdf", "CC BY-NC-SA"),
    ("cs", "Think Python 2e (Downey)", "https://greenteapress.com/thinkpython2/thinkpython2.pdf", "CC BY-NC"),
    ("cs", "Beej's Guide to C Programming", "https://beej.us/guide/bgc/pdf/bgc_usl_c_1.pdf", "CC BY-NC-ND"),
    ("engineering", "Lessons in Electric Circuits, Vol. I: DC (Kuphaldt)",
     "https://www.ibiblio.org/kuphaldt/electricCircuits/DC/DC.pdf", "Design Science License"),
    ("engineering", "Lessons in Electric Circuits, Vol. II: AC (Kuphaldt)",
     "https://www.ibiblio.org/kuphaldt/electricCircuits/AC/AC.pdf", "Design Science License"),
    ("engineering", "Lessons in Electric Circuits, Vol. III: Semiconductors (Kuphaldt)",
     "https://www.ibiblio.org/kuphaldt/electricCircuits/Semi/SEMI.pdf", "Design Science License"),
    ("engineering", "Lessons in Electric Circuits, Vol. IV: Digital (Kuphaldt)",
     "https://www.ibiblio.org/kuphaldt/electricCircuits/Digital/DIGI.pdf", "Design Science License"),
    ("engineering", "Engineering Statics: Open and Interactive (Baker, Haynes)",
     "https://engineeringstatics.org/pdf/statics.pdf", "CC BY-NC-SA"),
    ("engineering", "Additive Manufacturing Essentials (OpenStax)",
     OPENSTAX + "Additive_Manufacturing_Essentials_-_WEB.pdf", "CC BY-NC-SA"),
    ("chemistry", "Organic Chemistry (OpenStax)", OPENSTAX + "organic-chemistry_-_WEB.pdf", "CC BY-NC-SA"),
    ("english", "Writing Guide with Handbook (OpenStax)", OPENSTAX + "WritingGuide-WEB.pdf", "CC BY-NC-SA"),
    ("ethics", "Business Ethics (OpenStax)", OPENSTAX + "business-ethics_-_WEB.pdf", "CC BY-NC-SA"),
    # Not downloadable by a script (their hosts answer automated requests with 403): Introduction to
    # Environmental Science (USG), Open Technical Communication (Kennesaw), Pressbooks technical-writing texts,
    # the Constitution of India (legislative.gov.in). Download them in a browser and use /read or alice ingest.
]
# open-access papers outside arXiv: (subject, title, url, licence)
OPEN_PAPERS = [
    ("biology", "Highly accurate protein structure prediction with AlphaFold (Jumper et al., Nature 2021)",
     "https://www.nature.com/articles/s41586-021-03819-2.pdf", "CC BY 4.0"),
]


def arxiv_meta(arxiv_id: str) -> dict | None:
    """Title, first author and year from the paper's arXiv abstract page (its citation_* meta tags).
    The export API currently answers multi-id queries with 429 and single ones slowly (~30 s); the
    abstract pages answer in well under a second."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    for attempt in range(4):
        time.sleep(3)                                       # arXiv etiquette: one request every 3 s
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                        timeout=60) as r:
                page = r.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(15 * 2 ** attempt)
    meta = lambda name: [html.unescape(m) for m in re.findall(rf'<meta name="{name}" content="([^"]*)"', page)]  # noqa: E731
    titles, authors, dates = meta("citation_title"), meta("citation_author"), meta("citation_date")
    if not titles:
        return None
    surnames = [a.split(",")[0].strip() for a in authors]
    who = (surnames[0] if len(surnames) == 1 else " and ".join(surnames) if len(surnames) == 2
           else f"{surnames[0]} et al." if surnames else "")
    return {"title": " ".join(titles[0].split()), "who": who, "year": dates[0][:4] if dates else ""}


def verify_arxiv(subjects: list[str]) -> list[tuple[str, str, str]]:
    """[(subject, id, library title)] for ids whose real arXiv title contains the expected words."""
    wanted = [(s, i, w) for s in subjects for i, w in ARXIV.get(s, [])]
    ok = []
    for s, i, words in wanted:
        m = arxiv_meta(i)
        if m is None:
            print(f"  MISSING  {i}: no such arXiv paper - left out", flush=True)
        elif words.lower() not in m["title"].lower():
            print(f"  MISMATCH {i}: expected '{words}', arXiv says '{m['title']}' - left out", flush=True)
        else:
            ok.append((s, i, f"{m['title']} ({m['who']}, {m['year']})"))
    print(f"arXiv: {len(ok)}/{len(wanted)} ids verified", flush=True)
    return ok


def download(url: str, dest: Path, tries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(part, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            with open(part, "rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("not a PDF")
            part.replace(dest)
            return True
        except Exception as e:                           # network hiccup: resume from where it stopped
            print(f"    download attempt {attempt} failed: {type(e).__name__}: {e}")
            if isinstance(e, ValueError):                # got a web page instead of the PDF: retry with curl
                part.unlink(missing_ok=True)
                break
            time.sleep(5 * attempt)
    # Some publishers (nature.com) answer Python's HTTP client with an HTML page but curl with the PDF.
    curl = shutil.which("curl")
    if curl:
        subprocess.run([curl, "-s", "-L", "-A", USER_AGENT, "--max-time", "600", "-o", str(part), url],
                       check=False, stdin=subprocess.DEVNULL)
        if part.exists():
            with open(part, "rb") as f:
                is_pdf = f.read(5) == b"%PDF-"
            if is_pdf:
                part.replace(dest)
                print("    downloaded with curl")
                return True
            part.unlink(missing_ok=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="subjects: ai cs math physics chemistry biology")
    ap.add_argument("--no-books", action="store_true")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--reingest-books", action="store_true",
                    help="afterwards, re-ingest the books already in the library (section names from their bookmarks)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    subjects = args.only or ["ai", "cs", "math", "physics", "chemistry", "biology", "engineering", "english", "ethics"]

    items: list[tuple[str, str, str, Path]] = []      # (subject, title, url, file)
    for s, i, title in verify_arxiv(subjects):
        items.append((s, title, f"https://export.arxiv.org/pdf/{i}", PAPERS_DIR / s / f"{i.replace('/', '_')}.pdf"))
    extra = OPEN_PAPERS + ([] if args.no_books else BOOKS)
    for s, title, url, _lic in extra:
        if s in subjects:
            items.append((s, title, url, PAPERS_DIR / s / url.rsplit("/", 1)[1]))
    print(f"{len(items)} documents: " + ", ".join(f"{s} {sum(1 for it in items if it[0] == s)}" for s in subjects))
    if args.dry_run:
        for s, title, url, _ in items:
            print(f"  [{s}] {title}  <- {url}")
        return 0

    ready = []
    for s, title, url, path in items:
        fresh = not path.exists()
        if fresh and "arxiv.org" in url:
            time.sleep(3)                                  # arXiv etiquette: one request every 3 s
        print(f"  {'download' if fresh else 'have    '} [{s}] {title}", flush=True)
        if download(url, path):
            ready.append((title, path))
        else:
            print(f"    giving up on {url}")
    if args.no_ingest:
        return 0

    from alice import config
    from alice.library import BOOK_MIN_PAGES, Library, LlamaEmbedder
    from alice.llm import server_healthy
    gpu = config.EMBED_DEVICE == "gpu" or not server_healthy()
    lib, emb = Library(), LlamaEmbedder(gpu=gpu)
    print(f"ingesting into {config.LIBRARY_DIR} (embedding on {'GPU' if gpu else 'CPU'}) ...", flush=True)
    t0 = time.time()
    # Books ingested before this run (the ones added now already get their sections from the bookmarks).
    old_books = {d[0] for d in lib.documents() if (d[2] or 0) >= BOOK_MIN_PAGES}
    try:
        for title, path in ready:
            t = time.time()
            try:
                status, n, _ = lib.add(path, emb, title=title)
            except Exception as e:                        # one bad PDF must not stop the rest
                print(f"  failed  {path.name}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"  {status:7} {title} ({n} chunks, {time.time() - t:.0f} s)", flush=True)
        if args.reingest_books and old_books:
            print(f"re-ingesting {len(old_books)} books already in the library ...", flush=True)
            last = [time.time()]

            def done(title, status, n):
                print(f"  {status:7} {title} ({n} chunks, {time.time() - last[0]:.0f} s)", flush=True)
                last[0] = time.time()
            lib.rebuild(emb, doc_ids=old_books, on_done=done)
    finally:
        emb.stop()
    print(f"done in {(time.time() - t0) / 60:.1f} min; the library has {len(lib.documents())} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
