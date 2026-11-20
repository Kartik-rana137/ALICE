"""Download the browser libraries the GUI uses into src/alice/gui/static/vendor/ (pinned versions, so ALICE's
window works offline and never loads code from the internet at run time). Run once; the files are committed.

    python scripts/vendor_gui_assets.py
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src" / "alice" / "gui" / "static" / "vendor"
CDN = "https://cdn.jsdelivr.net/npm/"
FILES = {
    "marked.min.js": "marked@12.0.2/marked.min.js",                                   # markdown (MIT)
    "purify.min.js": "dompurify@3.1.6/dist/purify.min.js",                            # HTML sanitiser (Apache/MPL)
    "katex/katex.min.js": "katex@0.16.11/dist/katex.min.js",                          # maths (MIT)
    "katex/katex.min.css": "katex@0.16.11/dist/katex.min.css",
    "katex/auto-render.min.js": "katex@0.16.11/dist/contrib/auto-render.min.js",
    "hljs/highlight.min.js": "@highlightjs/cdn-assets@11.10.0/highlight.min.js",      # code (BSD-3)
    "hljs/github.min.css": "@highlightjs/cdn-assets@11.10.0/styles/github.min.css",
    "hljs/github-dark.min.css": "@highlightjs/cdn-assets@11.10.0/styles/github-dark.min.css",
}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ALICE-local-study-assistant/0.1"}),
                                timeout=60) as r:
        return r.read()


def main() -> int:
    for name, path in FILES.items():
        dest = VENDOR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(CDN + path))
        print(f"{name}: {dest.stat().st_size // 1024} KB")
    css = (VENDOR / "katex" / "katex.min.css").read_text(encoding="utf-8")
    fonts = sorted(set(re.findall(r"url\((fonts/[^)]+\.woff2)\)", css)))
    for f in fonts:                                    # woff2 only: every current browser engine uses it
        dest = VENDOR / "katex" / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(get(CDN + "katex@0.16.11/dist/" + f))
    print(f"katex fonts: {len(fonts)} woff2 files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
