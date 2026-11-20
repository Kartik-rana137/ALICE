"""A/B check for the Phase 1 fixes (numeric correctness + opener variety).

Usage (repo root, venv):  .venv\\Scripts\\python scripts\\phase1_check.py [--samples 5]
Compares: old persona v2.1 @ temp 0.3 (as committed in 4705353), persona v3 with Qwen's
recommended non-thinking sampling, and persona v3 with thinking on.
Writes benchmarks/results/phase1_check.json and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import time

from alice import config, persona
from alice.config import Sampling
from alice.agent import run_turn
from alice.llm import LlamaServer, chat_stream
from alice.tools import ToolBox, calc

OLD_PERSONA_COMMIT = "4705353"
OLD_SAMPLING = Sampling(temperature=0.3, top_p=0.95, top_k=40, min_p=0.05)   # llama-server defaults + temp 0.3

NUMERIC = [
    ("bayes", "A test for a disease is 99% sensitive and 95% specific. The disease affects 1 in 1000 "
              "people. If someone tests positive, what is the probability they have the disease?",
     lambda x: abs(x - 1.94) < 0.015 or abs(x - 0.0194) < 0.00015 or abs(x - 1.9) < 1e-9),
    ("params", "How many trainable parameters does nn.Sequential(nn.Linear(784, 128), nn.ReLU(), "
               "nn.Linear(128, 10)) have?",
     lambda x: x == 101770),
    ("batches", "I have 1,250 samples, use an 80/10/10 train/val/test split and a DataLoader with "
                "batch_size=32 and drop_last=False. How many training batches are there per epoch?",
     lambda x: x == 32),
]
WRONG_BAYES = re.compile(r"1\.9[5-9]\s*%|0\.019[5-9]")

OPENERS = [
    "What's the difference between a process and a thread?",
    "Explain gradient descent in simple terms.",
    "Why is my Python loop so much slower than the numpy version?",
    "What does the softmax function do?",
    "How do I reverse a singly linked list in C?",
    "What is overfitting and how do I detect it?",
    "What's the time complexity of quicksort?",
    "How does a hash table work?",
]

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def solution_part(text: str) -> str:
    """The reply minus the walkthrough and the closing check-your-understanding question."""
    m = re.search(r"\n[#*\s]*(walk-?\s?through|step[- ]by[- ]step|why (this|it) works)", text, re.I)
    if m:
        text = text[:m.start()]
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    while len(paras) > 1 and "?" in paras[-1]:
        paras.pop()
    return "\n\n".join(paras)


def final_number(text: str) -> float | None:
    """Last bold number in the solution part (excludes walkthrough and follow-up question),
    else the last number there."""
    sol = solution_part(text)
    bolds = re.findall(r"\*\*(.+?)\*\*|\\(?:mathbf|boxed)\{([^}]*)\}", sol, re.S)
    bolds = [x or y for x, y in bolds if NUM.search(x or y)]
    src = bolds[-1] if bolds else sol
    nums = NUM.findall(src)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def opener(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s|\n", text.strip(), maxsplit=1)[0]
    return first.strip()


def old_persona() -> str:
    src = subprocess.run(["git", "show", f"{OLD_PERSONA_COMMIT}:src/alice/persona.py"],
                         capture_output=True, text=True, encoding="utf-8", cwd=config.ROOT, check=True).stdout
    ns: dict = {}
    exec(src, ns)
    return ns["SYSTEM_PROMPT"]


def ask(system: str, prompt: str, sampling: Sampling, thinking: bool, tools: bool = False) -> dict:
    profile = config.PROFILES["default"]
    if thinking:
        profile = config.Profile(**{**profile.__dict__, "thinking_sampling": sampling})
    else:
        profile = config.Profile(**{**profile.__dict__, "sampling": sampling})
    t0 = time.time()
    if tools:                                  # full agent turn with the calculator (Phase 2)
        res = run_turn([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                       profile, ToolBox().add(calc.TOOL), thinking=thinking)
        return {"reply": res.reply, "finish": res.finish, "seconds": round(time.time() - t0, 1),
                "gen_tokens": res.gen_tokens, "reasoning_chars": res.reasoning_chars,
                "tool_calls": [{"name": t.name, "args": t.args, "output": t.output} for t in res.tool_results]}
    content, reasoning, finish, timings = [], 0, None, {}
    for ev in chat_stream([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          profile, thinking=thinking):
        if ev.kind == "content":
            content.append(ev.text)
        elif ev.kind == "reasoning":
            reasoning += len(ev.text)
        else:
            finish, timings = ev.text, ev.timings
    return {"reply": "".join(content), "finish": finish, "seconds": round(time.time() - t0, 1),
            "gen_tokens": timings.get("predicted_n"), "reasoning_chars": reasoning}


CHECKS = {pid: ok for pid, _, ok in NUMERIC}


def score(r: dict) -> None:
    r["final"] = final_number(r["reply"])
    r["correct"] = r["final"] is not None and CHECKS[r["id"]](r["final"])
    r["wrong_value_mentioned"] = bool(r["id"] == "bayes" and WRONG_BAYES.search(r["reply"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--only", nargs="*", help="run only these setups")
    ap.add_argument("--openers-only", action="store_true")
    ap.add_argument("--opener-repeats", type=int, default=1, help="run the opener prompts N times")
    ap.add_argument("--rescore", action="store_true", help="re-score the saved JSON, no model calls")
    ap.add_argument("--out", default="phase1_check.json")
    ap.add_argument("--layout", choices=["note", "system"], default="note",
                    help="where per-reply instructions go: appended to the question (current) or the system prompt (old)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = config.ROOT / "benchmarks" / "results" / a.out

    if a.rescore:
        results = json.load(open(out, encoding="utf-8"))
        for res in results.values():
            for r in res["numeric"]:
                score(r)
            for r in res["openers"]:
                r["opener"] = opener(r["reply"])
        json.dump(results, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        summarize(results)
        return 0

    setups = {
        "v2.1@t0.3": lambda: (old_persona(), OLD_SAMPLING, False),
        "v3-direct": lambda: (persona.SYSTEM_PROMPT, config.QWEN_DIRECT, False),
        "v3-think": lambda: (persona.SYSTEM_PROMPT, config.QWEN_THINKING, True),
        # v3.1: random opener style per request, as the CLI does
        "v3.1-direct": lambda: (None, config.QWEN_DIRECT, False),
        # v3.1 + calculator tool, no thinking (Phase 2)
        "v3.1-tools": lambda: (None, config.QWEN_DIRECT, False),
    }
    use_tools = {"v3.1-tools"}
    chosen = a.only or ["v2.1@t0.3", "v3-direct", "v3-think"]
    server = LlamaServer(config.PROFILES["default"])
    server.start()
    results: dict = {name: {"numeric": [], "openers": []} for name in chosen}

    def system_for(base, tools):
        """(system prompt, opener style, note appended to the question)"""
        if base is not None:
            return base, None, ""
        names = ["calculate"] if tools else ()
        if a.layout == "system":                    # pre-2026-11-19: per-reply note inside the system prompt
            style = persona.pick_opener_style()
            return persona.system_prompt(style, tools=names), style, ""
        style = persona.pick_sass()
        return persona.system_prompt(tools=names), style, persona.turn_note(style)

    try:
        for name in chosen:
            base, sampling, thinking = setups[name]()
            tools = name in use_tools
            if not a.openers_only:
                for pid, prompt, _ in NUMERIC:
                    for i in range(a.samples):
                        system, style, note = system_for(base, tools)
                        r = ask(system, prompt + note, sampling, thinking, tools)
                        r.update(id=pid, sample=i, opener_style=style)
                        score(r)
                        results[name]["numeric"].append(r)
                        print(f"{name:11} {pid:8} #{i} final={r['final']} correct={r['correct']} "
                              f"{r['seconds']}s [{r['finish']}]", flush=True)
            for prompt in OPENERS * a.opener_repeats:
                system, style, note = system_for(base, tools)
                r = ask(system, prompt + note, sampling, thinking, tools)
                r.update(prompt=prompt, opener=opener(r["reply"]), opener_style=style)
                results[name]["openers"].append(r)
                print(f"{name:11} opener: {r['opener'][:110]}", flush=True)
    finally:
        server.stop()

    json.dump(results, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    summarize(results)
    return 0


def summarize(results: dict) -> None:
    print("\n=== summary")
    for name, res in results.items():
        by = collections.defaultdict(list)
        for r in res["numeric"]:
            by[r["id"]].append(r)
        acc = "  ".join(f"{k} {sum(r['correct'] for r in v)}/{len(v)}" for k, v in by.items()) or "(no numeric)"
        wrong = sum(r["wrong_value_mentioned"] for r in res["numeric"])
        secs = [r["seconds"] for r in res["numeric"]] or [0.0]
        cut = sum(r["finish"] == "length" for r in res["numeric"] + res["openers"])
        firsts = collections.Counter((r["opener"].lower().split() or [""])[0].strip(",.!") for r in res["openers"])
        repeated_first = sum(c for c in firsts.values() if c > 1)
        another = sum(r["opener"].lower().startswith("another") for r in res["openers"])
        print(f"{name:11} {acc}  bayes-wrong-mentioned {wrong}  mean {sum(secs)/len(secs):.1f}s  cut-off {cut}  "
              f"openers: same-first-word {repeated_first}/{len(res['openers'])}, 'Another...' {another}")


if __name__ == "__main__":
    raise SystemExit(main())
