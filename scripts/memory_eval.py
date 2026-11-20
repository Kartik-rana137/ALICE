"""Long-term memory evaluation with the real model (temporary memory folder, your notes are not touched).

Session 1: a scripted student mentions six facts about themselves in passing over six turns; the conversation
is compacted half-way (low threshold) and saved at the end. Scored: which planted facts were stored (recall),
which stored facts were never said (should be none), and what the grounding filter rejected.
Session 2: a fresh session with the same memory answers two questions that should use the notes.

    python scripts/memory_eval.py [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TURNS = [
    "I'm in my second year of computer science and taking an operating systems course this term. "
    "What does a page table actually do?",
    "Our OS project is written in Rust, so Rust examples help me most. How is a TLB miss handled?",
    "Honestly I always mix up paging and segmentation. What's the cleanest way to tell them apart?",
    "My OS exam is on December 12th, so I want to be solid on virtual memory by then. What should I review?",
    "What's the difference between a process and a thread?",
    "By the way, short explanations with a bit of code work best for me.",
]
PLANTED = {                       # fact -> pattern a stored note must match
    "second year of CS": r"second[- ]year",
    "operating systems course": r"operating systems|\bOS\b",
    "project in Rust": r"\bRust\b",
    "mixes up paging/segmentation": r"paging|segmentation",
    "exam on December 12": r"December 12|12(th)? December|Dec(ember)?\.? 12",
    "likes short explanations with code": r"short|concise|brief",
}
SESSION2 = [
    "Can you give me one quick practice question for my exam?",
    "What do you remember about me?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="alice-memory-eval-"))
    os.environ["ALICE_MEMORY_DIR"] = str(tmp / "memory")
    os.environ["ALICE_LIBRARY"] = str(tmp / "library")           # no library: keep the turns about memory
    sys.path.insert(0, str(ROOT / "src"))
    sys.stdout.reconfigure(encoding="utf-8")
    from alice import config
    from alice.llm import server_healthy
    from alice.session import ChatSession

    if server_healthy():
        print("a llama-server is already running on the chat port; stop it first")
        return 1
    config.COMPACT_AT = 0.12            # compact after a few turns (normally 0.75 of the context budget)
    profile = config.PROFILES[config.DEFAULT_PROFILE]
    out: dict = {"turns": [], "session2": []}

    s = ChatSession(profile, workspace=str(tmp), offline=True, transcript=False, memory_notes=True)
    s.start()
    try:
        for q in TURNS:
            t0 = time.perf_counter()
            r = s.ask(q)
            out["turns"].append({"q": q, "reply": r.result.reply if r.result else r.error,
                                 "seconds": round(time.perf_counter() - t0, 1),
                                 "notices": [n.text for n in r.notices]})
            print(f"turn {len(out['turns'])}: {out['turns'][-1]['seconds']} s"
                  + (f"  [background: {s.background}]" if s.background else ""), flush=True)
            t0 = time.perf_counter()
            s.wait_background()
            if time.perf_counter() - t0 > 0.5:
                print(f"  background work took {time.perf_counter() - t0:.1f} s more", flush=True)
        out["compaction_summary"] = s.convo.summary
        out["turns_kept_verbatim"] = len(s.convo.turns())
        t0 = time.perf_counter()
        update = s.save_memory()
        out["save_seconds"] = round(time.perf_counter() - t0, 1)
        out["final_summary"] = update.summary if update else None
        out["rejected_by_filter"] = update.rejected if update else []
        facts = [n.text for n in s.memory.facts()]
        out["stored_facts"] = facts
    finally:
        s.stop()

    found = {name: [f for f in facts if re.search(rx, f, re.I)] for name, rx in PLANTED.items()}
    out["recall"] = {name: bool(v) for name, v in found.items()}
    out["unmatched_facts"] = [f for f in facts if not any(f in v for v in found.values())]

    s2 = ChatSession(profile, workspace=str(tmp), offline=True, transcript=False, memory_notes=True)
    s2.start()
    try:
        out["memory_prompt_section"] = s2._memory_section
        for q in SESSION2:
            r = s2.ask(q)
            reply = r.result.reply if r.result else r.error
            out["session2"].append({"q": q, "reply": reply,
                                    "mentions": [n for n, rx in PLANTED.items() if re.search(rx, reply or "", re.I)]})
    finally:
        s2.stop()

    print("\nstored facts:\n  " + "\n  ".join(facts))
    print("recall:", sum(out["recall"].values()), "/", len(PLANTED), {k: v for k, v in out["recall"].items() if not v})
    print("facts matching nothing planted (check by hand):", out["unmatched_facts"])
    print("rejected by the grounding filter:", out["rejected_by_filter"])
    for t in out["session2"]:
        print(f"\nsession 2: {t['q']}\n  uses: {t['mentions']}\n  {t['reply'][:400]}")
    path = ROOT / "benchmarks" / "results" / f"memory_eval{args.tag}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
