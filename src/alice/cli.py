"""Text chat CLI:  alice [--model default|heavy]"""
from __future__ import annotations

import argparse
import sys
import time

from . import config, persona
from .conversation import Conversation
from .llm import LlamaServer, chat_stream

HELP = "Commands: /reset  clear the conversation | /help | /quit"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="alice", description="ALICE: local CS/AI mentor")
    ap.add_argument("--model", choices=sorted(config.PROFILES), default="default",
                    help="model profile (see alice/config.py)")
    ap.add_argument("--no-stream", action="store_true", help="print each reply only when complete")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    profile = config.PROFILES[args.model]
    server = LlamaServer(profile)
    print(f"Loading {profile.gguf} ({profile.note}) ...", flush=True)
    try:
        started = server.start()
    except (FileNotFoundError, RuntimeError, TimeoutError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("ALICE is ready." + ("" if started else " (reusing running server)"))
    print(HELP + "\n")

    convo = Conversation(persona.SYSTEM_PROMPT, profile.ctx, profile.max_tokens)
    try:
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/reset":
                convo.reset()
                print("(conversation cleared)")
                continue
            if text == "/help":
                print(HELP)
                continue

            convo.add_user(text)
            print("alice> ", end="", flush=True)
            parts: list[str] = []
            t0 = time.time()
            try:
                if args.no_stream:
                    parts.append("".join(chat_stream(convo.messages(), profile)))
                    print(parts[0], end="")
                else:
                    for delta in chat_stream(convo.messages(), profile):
                        parts.append(delta)
                        print(delta, end="", flush=True)
            except KeyboardInterrupt:
                print("\n(interrupted)")
            except OSError as e:
                print(f"\n(error talking to the model: {e})")
                convo.history.pop()          # drop the unanswered user turn
                continue
            reply = "".join(parts)
            print(f"\n[{time.time() - t0:.1f}s]\n")
            if reply:
                convo.add_assistant(reply)
            else:
                convo.history.pop()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
