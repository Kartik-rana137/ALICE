"""Response-time benchmark: a fixed multi-turn conversation through the real chat pipeline.

Per turn it records the time to the first visible token (what the user waits for), the total time, how
many prompt tokens llama-server had to evaluate vs. reuse from its prompt cache, and generation speed.
Replies are saved too, so a speed change that breaks answers is visible.

    python scripts/latency_bench.py --tag baseline
    python scripts/latency_bench.py --tag ngram --extra-args "--spec-type ngram-mod"
    python scripts/latency_bench.py --tag x --repeat 2       # the conversation twice (reset in between)

Stop any running ALICE / llama-server first: a running server is reused and --extra-args would be ignored.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shlex
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Turn 2-4 need the 5 famous eval papers (data/eval_library); turn 4 needs the calculator.
CONVERSATION = [
    "What is the difference between a process and a thread? Keep it short.",
    "What learning rate schedule does the Transformer paper use?",
    "And how many warmup steps did they use?",
    "With d_model = 512 and that many warmup steps, what is the learning rate at step 8000?",
    "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.",
    "Now make it ignore punctuation too.",
]
# lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5) = 512^-0.5 * 8000^-0.5
EXPECTED = {3: 0.000494}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--library", default="data/eval_library")
    ap.add_argument("--extra-args", default="", help="extra llama-server flags for this run")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    os.environ["ALICE_LIBRARY"] = str(ROOT / args.library)
    os.environ.setdefault("ALICE_ARXIV_AUTO", "0")
    sys.path.insert(0, str(ROOT / "src"))
    from alice import config
    from alice.llm import server_healthy
    from alice.session import ChatSession

    if server_healthy():
        print("a llama-server is already running on the chat port; stop it first (flags would be ignored)")
        return 1
    profile = config.PROFILES[args.model or config.DEFAULT_PROFILE]
    if args.extra_args:
        profile = dataclasses.replace(profile, extra_args=[*profile.extra_args, *shlex.split(args.extra_args)])
    random.seed(0)                                   # same opener styles every run

    session = ChatSession(profile, workspace=str(ROOT), offline=True, transcript=False, auto_arxiv=False,
                          memory_notes=False, claude_escalation=False)   # reproducible: no personal notes
    t = time.perf_counter()
    session.start()
    start_s = time.perf_counter() - t
    warm = getattr(session, "wait_warm", None)
    warm_s = 0.0
    if warm:
        t = time.perf_counter()
        warm()
        warm_s = time.perf_counter() - t
    print(f"server start {start_s:.1f} s, warm-up wait {warm_s:.1f} s")

    rows = []
    try:
        for rep in range(args.repeat):
            if rep:
                session.command("/reset")
            for i, question in enumerate(CONVERSATION):
                rows.append(run_turn(session, rep, i, question))
                r = rows[-1]
                print(f"[{rep}.{i}] first token {r['ttft_s']:5.2f} s  total {r['total_s']:5.1f} s  "
                      f"prompt eval {r['prompt_n']:5d} tok ({r['prompt_ms'] / 1000:4.1f} s)  "
                      f"cached {r['cache_n']:5d}  gen {r['gen_n']:4d} @ {r['gen_tps']:5.1f} tok/s"
                      + (f"  prefetch {r['prefetch_s']:.2f} s" if r["prefetch_s"] else ""))
    finally:
        session.stop()

    def med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(vals), 2) if vals else None
    summary = {k: med(k) for k in ("ttft_s", "total_s", "prompt_n", "prompt_ms", "cache_n", "gen_tps")}
    summary.update(server_start_s=round(start_s, 1), warmup_wait_s=round(warm_s, 1),
                   ttft_total_s=round(sum(r["ttft_s"] for r in rows), 1),
                   prompt_eval_total_s=round(sum(r["prompt_ms"] for r in rows) / 1000, 1))
    out = ROOT / "benchmarks" / "results" / f"latency_{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "profile": profile.name, "model": profile.gguf,
                               "extra_args": list(profile.extra_args), "summary": summary, "turns": rows},
                              indent=1), encoding="utf-8")
    print(json.dumps(summary))
    print(f"saved {out.relative_to(ROOT)}")
    return 0


def run_turn(session, rep: int, i: int, question: str) -> dict:
    from alice.tools import calc
    stats = {"prompt_n": 0, "prompt_ms": 0.0, "cache_n": 0, "gen_n": 0, "gen_ms": 0.0, "rounds": 0}
    first: list[float] = []
    prefetch_s = [0.0]
    if session.library and not hasattr(session.library, "_timed"):
        orig = session.library.prefetch

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                session.library._last_prefetch_s = time.perf_counter() - t0
        session.library.prefetch, session.library._timed = timed, True
    if session.library:
        session.library._last_prefetch_s = 0.0

    def on_event(ev):
        if ev.kind in ("content", "reasoning") and not first:
            first.append(time.perf_counter())
        if ev.kind == "finish":
            tm = ev.timings or {}
            stats["rounds"] += 1
            stats["prompt_n"] += int(tm.get("prompt_n") or 0)
            stats["prompt_ms"] += float(tm.get("prompt_ms") or 0)
            stats["cache_n"] += int(tm.get("cache_n") or 0)
            stats["gen_n"] += int(tm.get("predicted_n") or 0)
            stats["gen_ms"] += float(tm.get("predicted_ms") or 0)

    t0 = time.perf_counter()
    outcome = session.ask(question, on_event=on_event, confirm=lambda tool, a: False)
    total = time.perf_counter() - t0
    if session.library:
        prefetch_s[0] = getattr(session.library, "_last_prefetch_s", 0.0)
    reply = outcome.result.reply if outcome.result else (outcome.error or "")
    row = {"rep": rep, "turn": i, "question": question,
           "ttft_s": round((first[0] - t0) if first else total, 2), "total_s": round(total, 1),
           "prompt_n": stats["prompt_n"], "prompt_ms": round(stats["prompt_ms"]), "cache_n": stats["cache_n"],
           "gen_n": stats["gen_n"], "gen_tps": round(stats["gen_n"] / (stats["gen_ms"] / 1000), 1) if stats["gen_ms"] else 0.0,
           "rounds": stats["rounds"], "prefetch_s": round(prefetch_s[0], 2),
           "tools": [t.name for t in outcome.result.tool_results] if outcome.result else [],
           "notices": [n.text for n in outcome.notices], "reply": reply[:1500]}
    if i in EXPECTED:
        row["expected"] = EXPECTED[i]
        row["correct"] = calc.reply_mentions(reply, EXPECTED[i])
    return row


if __name__ == "__main__":
    raise SystemExit(main())
