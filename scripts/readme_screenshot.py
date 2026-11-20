"""Screenshot of the real chat box for the README: the actual TUI driving the actual model (headless).

Uses the 5-paper eval library (data/eval_library), no memory notes and no transcript, asks two questions and
exports what is on screen as SVG. Nothing is staged: if an answer comes out wrong, run it again.

    python scripts/readme_screenshot.py [--out docs/assets/alice-tui.svg]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "Which optimizer did the Transformer paper use, and with which settings?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/alice-tui.svg")
    ap.add_argument("--size", default="118x46", help="terminal columns x rows")
    args = ap.parse_args()
    os.environ["ALICE_LIBRARY"] = str(ROOT / "data" / "eval_library")
    sys.path.insert(0, str(ROOT / "src"))
    from textual.widgets import TextArea

    from alice import config
    from alice.session import ChatSession
    from alice.tui import AliceApp

    cols, rows = (int(x) for x in args.size.split("x"))
    session = ChatSession(config.PROFILES[config.DEFAULT_PROFILE], workspace=str(ROOT), offline=True,
                          transcript=False, memory_notes=False)
    app = AliceApp(session)

    async def run() -> str:
        async with app.run_test(size=(cols, rows)) as pilot:
            while not app.ready:
                await pilot.pause(0.5)
            session.wait_background()
            for q in QUESTIONS:
                app.query_one("#prompt", TextArea).text = q
                await pilot.press("enter")
                await pilot.pause(0.5)
                while app.busy:
                    await pilot.pause(0.5)
            app.query_one("#prompt", TextArea).text = ""
            await pilot.pause(1.0)
            app.log_view.scroll_end(animate=False)
            await pilot.pause(0.5)
            texts = [getattr(w, "source", None) or str(getattr(w, "content", "")) for w in app.log_view.children]
            return app.export_screenshot(title="ALICE"), texts

    try:
        svg, texts = asyncio.run(run())
    finally:
        session.stop()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")
    for t in texts:
        print("---", t[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
