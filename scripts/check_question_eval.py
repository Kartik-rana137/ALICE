"""Does ALICE end with a check-your-understanding question only when the student is learning something?

12 casual messages (should get no check question) and 8 study questions (should get exactly one), each in a
fresh conversation through the real chat pipeline (library on, memory off). A reply "has a check question"
if its last paragraph asks one (explicit markers like "Check your understanding", or a question that sets
the student a task). Replies are saved for reading by hand.

    python scripts/check_question_eval.py --tag NAME
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASUAL = [
    "hi", "Good morning! How are you today?", "thanks, that helped a lot", "What time is it?",
    "I'm so tired today, long week.", "Can you suggest a fun name for my study group chat?",
    "lol that was a good one", "What's your favourite programming language?", "I just finished my exams!",
    "Do you like music?", "ok, see you tomorrow", "How many days until 25 December?",
]
STUDY = [
    "Can you explain what a hash table is?",
    "I don't get why quicksort is O(n log n) on average. Can you clarify?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain Bayes' theorem with an example.",
    "Why does gradient descent need a learning rate?",
    "What is entropy in thermodynamics?",
    "Write a Python function that reverses a linked list.",
    "How does photosynthesis produce oxygen?",
]
_MARKER = re.compile(r"check your understanding|quick check|test (your|yourself)|pop quiz|quiz:|your turn|try this|"
                     r"exercise|challenge", re.I)
_TASK_Q = re.compile(r"\b(what (would|will|happens|is the|does)|how (would|many|much|does)|why (would|does|is)|"
                     r"can you|could you|which|if you|predict|calculate|compute|explain)\b[^?]*\?", re.I)


# Bringing up the day or time when nobody asked (the local time travels in every turn note).
TIME_MENTION = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\bafternoon\b|"
                          r"\b[0-2]?\d:\d\d\b|past lunch|\bweekend\b", re.I)
# Offers and small talk ("Want me to show more?", "What's next on your list?") are not check questions.
_OFFER = re.compile(r"\b(do you want|want me to|would you like|shall i|should i|what would you like|what'?s next|"
                    r"what are we|what are you|what can i|anything (else|specific)|let me know|what'?s on your|"
                    r"what confuses you|which part|what part)\b", re.I)


def has_check_question(reply: str) -> bool:
    paras = [p.strip() for p in re.split(r"\n\s*\n", reply.strip()) if p.strip()]
    if not paras:
        return False
    tail = " ".join(paras[-2:]) if len(paras[-1]) < 40 else paras[-1]
    if _MARKER.search(tail):
        return True
    questions = [q for q in re.findall(r"[^.?!\n]*\?", tail) if not _OFFER.search(q)]
    return any(_TASK_Q.search(q) for q in questions)


def rescore(tag: str) -> dict:
    path = ROOT / "benchmarks" / "results" / f"check_question_{tag}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for r in data["rows"]:
        r["check_question"] = has_check_question(r["reply"])
    rows = data["rows"]
    data["summary"] = {k: f"{sum(r['check_question'] for r in rows if r['kind'] == kind)}/"
                          f"{sum(r['kind'] == kind for r in rows)}"
                       for k, kind in (("casual_with_check", "casual"), ("study_with_check", "study"))}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return data["summary"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rescore", action="store_true", help="re-score a saved run, no model calls")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    sys.stdout.reconfigure(encoding="utf-8")
    if args.rescore:
        print(rescore(args.tag))
        return 0
    from alice import config
    from alice.session import ChatSession

    s = ChatSession(config.PROFILES[config.DEFAULT_PROFILE], workspace=str(ROOT), offline=True,
                    transcript=False, memory_notes=False, claude_escalation=False)
    s.start()
    rows = []
    try:
        for kind, prompts in (("casual", CASUAL), ("study", STUDY)):
            for q in prompts:
                s.convo.reset()
                out = s.ask(q, confirm=lambda tool, a: False)
                reply = out.result.reply if out.result else (out.error or "")
                rows.append({"kind": kind, "q": q, "check_question": has_check_question(reply), "reply": reply})
                print(f"[{kind:6}] check={rows[-1]['check_question']!s:5} {q}", flush=True)
    finally:
        s.stop()
    casual = [r for r in rows if r["kind"] == "casual"]
    study = [r for r in rows if r["kind"] == "study"]
    not_about_time = [r for r in rows if r["q"] not in ("What time is it?", "How many days until 25 December?")]
    summary = {"casual_with_check": f"{sum(r['check_question'] for r in casual)}/{len(casual)}",
               "study_with_check": f"{sum(r['check_question'] for r in study)}/{len(study)}",
               "unprompted_day_or_time": f"{sum(bool(TIME_MENTION.search(r['reply'])) for r in not_about_time)}"
                                         f"/{len(not_about_time)}"}
    print(summary)
    out = ROOT / "benchmarks" / "results" / f"check_question_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
