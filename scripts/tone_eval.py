"""Tone check over one multi-turn conversation: emoji, excitement, and topics ALICE keeps dragging back in.

An 8-turn conversation that jumps between unrelated topics, with two planted memory notes (an exam date and a
Rust project; temporary memory folder - your notes are not touched). Counted per run:
  - Unicode emoji vs. ASCII emoticons (":)", ";)", "-_-", ...)
  - exclamation marks per reply
  - unprompted memory: replies that bring up the exam / the Rust project when the message isn't about them
  - repeated themes: words that are not in the message but show up in 3+ replies (e.g. "GPU", "coffee")
Replies are saved for reading.

    python scripts/tone_eval.py --tag NAME
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TURNS = [
    "hey",
    "Can you explain how binary search works?",
    "nice. any tips for staying focused while studying?",
    "What's the difference between TCP and UDP?",
    "lol ok. what should I cook for dinner tonight?",
    "Explain what a p-value means.",
    "thanks! I'm heading out for a walk.",
    "How does a CPU cache work?",
]
MEMORY = ["The student has an operating systems exam on December 12th.",
          "The student is working on an OS project written in Rust."]
MEMORY_RX = re.compile(r"\bexams?\b|\brust\b|december|os project", re.I)      # not "example"
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2705\u274C\u203C\u2049\u2122\u2139\u2194-\u21AA"
                   "\u231A-\u23FF\u25AA-\u25FE\u2934\u2935\u3030\u303D\u3297\u3299\uFE0F]")
EMOTICON = re.compile(r"(?<![\w/])(?:[:;=8xX][-o']?[)(DPpOo/\\|\]\[*]|\^[_.-]?\^|-_-|o_O|O_o|>_<|T_T|<3|"
                      r"\\o/|¯\\_\(ツ\)_/¯)(?![\w])")
STOP = set("""about above after again against also another anything around because been before being below between
both could does doing down during each even every from further have having here into itself just like make more
most much must need only other over really same should since some such than that their them then there these they
this those through under until very want well were what when where which while will with without would your yours
yourself thing things something someone time first second next back good great right still sure know think going
want okay just also student alice reply question answer let's here's that's it's you're don't can't won't i'm""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{3,}", text.lower()) if w not in STOP}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="alice-tone-"))
    os.environ["ALICE_MEMORY_DIR"] = str(tmp / "memory")
    sys.path.insert(0, str(ROOT / "src"))
    sys.stdout.reconfigure(encoding="utf-8")
    from alice import config
    from alice.memory import MemoryStore
    from alice.session import ChatSession

    store = MemoryStore(tmp / "memory")
    for fact in MEMORY:
        store.add("fact", fact, source="user")
    store.close()
    s = ChatSession(config.PROFILES[config.DEFAULT_PROFILE], workspace=str(ROOT), offline=True,
                    transcript=False, memory_notes=True, claude_escalation=False)
    s.start()
    rows = []
    try:
        for q in TURNS:
            out = s.ask(q, confirm=lambda tool, a: False)
            reply = out.result.reply if out.result else (out.error or "")
            rows.append({"q": q, "reply": reply})
            print(f"- {q}\n  {' '.join(reply.split())[:160]}", flush=True)
    finally:
        s.memory.clear()                                   # temporary store; leave nothing behind
        s.stop()
    n = len(rows)
    off_topic = Counter()
    for r in rows:
        off_topic.update(content_words(r["reply"]) - content_words(r["q"]))
    repeated = sorted([(w, c) for w, c in off_topic.items() if c >= 3], key=lambda x: -x[1])
    summary = {
        "unicode_emoji": sum(len(EMOJI.findall(r["reply"])) for r in rows),
        "ascii_emoticons": sum(len(EMOTICON.findall(r["reply"])) for r in rows),
        "exclamations_per_reply": round(sum(r["reply"].count("!") for r in rows) / n, 2),
        "unprompted_memory_mentions": f"{sum(bool(MEMORY_RX.search(r['reply'])) for r in rows)}/{n}",
        "words_in_3plus_replies": repeated[:15],
    }
    print(json.dumps(summary, ensure_ascii=False))
    out = ROOT / "benchmarks" / "results" / f"tone_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
