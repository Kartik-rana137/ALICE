import json

import pytest

from alice import config
from alice.config import PROFILES, Profile
from alice.conversation import MESSAGE_OVERHEAD, Conversation, estimate_tokens
from alice.llm import LlamaServer, iter_sse_events
from alice.transcript import Transcript


# ---------- SSE parsing ----------

def chunk(delta=None, finish=None, timings=None):
    d = {"choices": [{"delta": delta or {}, "finish_reason": finish}]}
    if timings:
        d["timings"] = timings
    return "data: " + json.dumps(d)


def kinds(events):
    return [(e.kind, e.text) for e in events]


def test_sse_content_in_order_and_finish():
    lines = [chunk({"content": "Hel"}), chunk({"content": "lo"}),
             chunk(finish="stop", timings={"predicted_n": 2}), "data: [DONE]"]
    evs = list(iter_sse_events(lines))
    assert kinds(evs) == [("content", "Hel"), ("content", "lo"), ("finish", "stop")]
    assert evs[-1].timings == {"predicted_n": 2}


def test_sse_separates_reasoning_from_content():
    lines = [chunk({"role": "assistant", "content": None}), chunk({"reasoning_content": "think"}),
             chunk({"content": "391"}), chunk(finish="length")]
    assert kinds(iter_sse_events(lines)) == [("reasoning", "think"), ("content", "391"), ("finish", "length")]


def test_sse_ignores_noise():
    lines = [": keepalive", "", "data: not json", 'data: {"choices": []}', chunk({"content": "ok"})]
    assert kinds(iter_sse_events(lines)) == [("content", "ok")]


def test_sse_stops_at_done_and_accepts_bytes():
    lines = [chunk({"content": "a"}).encode(), b"data: [DONE]", chunk({"content": "b"}).encode()]
    assert kinds(iter_sse_events(lines)) == [("content", "a")]


# ---------- conversation memory ----------

def test_conversation_keeps_system_first_and_order():
    c = Conversation("sys", ctx=4096, reply_budget=500)
    c.add_user("q1"); c.add_assistant("a1"); c.add_user("q2")
    msgs = c.messages()
    assert msgs[0]["role"] == "system"
    assert [m["content"] for m in msgs[1:]] == ["q1", "a1", "q2"]


def test_conversation_trims_oldest_but_keeps_latest():
    c = Conversation("sys", ctx=1200, reply_budget=300)
    for i in range(30):
        c.add_user(f"question {i} " + "x" * 200)
        c.add_assistant(f"answer {i} " + "y" * 200)
    c.add_user("final question")
    msgs = c.messages()
    assert msgs[-1]["content"] == "final question"
    assert len(msgs) < 20
    assert msgs[1]["role"] == "user"               # window never starts on an assistant turn
    assert sum(c.tokens(m["content"]) for m in msgs[1:]) <= c.budget


def test_conversation_uses_exact_counter_and_caches():
    calls = []
    def counter(text):
        calls.append(text)
        return 5
    c = Conversation("sys", ctx=4096, reply_budget=500, counter=counter)
    c.add_user("hello")
    c.messages(); c.messages()
    assert calls.count("hello") == 1
    assert c.tokens("hello") == 5 + MESSAGE_OVERHEAD


def test_conversation_falls_back_when_counter_fails():
    def broken(text):
        raise OSError("server gone")
    c = Conversation("sys", ctx=4096, reply_budget=500, counter=broken)
    assert c.tokens("abcdef") == estimate_tokens("abcdef") + MESSAGE_OVERHEAD


def test_conversation_oversized_latest_message_still_kept():
    c = Conversation("sys", ctx=1200, reply_budget=300)
    c.add_user("z" * 20000)
    assert len(c.messages()) == 2


def test_conversation_reset_and_pop():
    c = Conversation("sys", ctx=4096, reply_budget=500)
    c.add_user("hi"); c.pop_last()
    assert c.messages() == [c.system]
    c.add_user("hi"); c.reset()
    assert c.messages() == [c.system]


def test_conversation_rejects_tiny_context():
    with pytest.raises(ValueError):
        Conversation("s" * 5000, ctx=1000, reply_budget=500)


# ---------- config / server command ----------

def test_profiles_build_server_commands():
    cmd = LlamaServer(PROFILES["default"]).command()
    assert "-ngl" in cmd and "--n-cpu-moe" not in cmd
    assert cmd[cmd.index("-np") + 1] == "1"
    assert cmd[cmd.index("-c") + 1] == str(PROFILES["default"].ctx)
    moe = LlamaServer(Profile("moe", "x.gguf", n_cpu_moe=36)).command()
    assert moe[moe.index("--n-cpu-moe") + 1] == "36"


def test_thinking_budget_fits_inside_context():
    for p in PROFILES.values():
        assert p.reasoning_budget < p.thinking_max_tokens < p.ctx
        assert p.gguf.endswith(".gguf")


def test_conversation_system_override_is_per_request():
    c = Conversation("base", ctx=4096, reply_budget=500)
    c.add_user("q")
    assert c.messages("base + style")[0]["content"] == "base + style"
    assert c.messages()[0]["content"] == "base"


# ---------- persona ----------

def test_persona_opener_style_is_appended_and_random_choice_is_valid():
    import random
    from alice import persona
    style = persona.pick_opener_style(random.Random(0))
    assert style in persona.OPENER_STYLES
    assert persona.system_prompt(style).startswith(persona.SYSTEM_PROMPT)
    assert style in persona.system_prompt(style)
    assert persona.system_prompt(None) == persona.SYSTEM_PROMPT
    assert "Another day" not in persona.SYSTEM_PROMPT      # don't prime the stock opener


# ---------- transcripts ----------

def test_transcript_writes_jsonl(tmp_path):
    t = Transcript(tmp_path, model="m.gguf", persona_version="v3")
    t.user("hi"); t.assistant("hello", finish="stop"); t.event("reset")
    t2 = Transcript(tmp_path, model="m.gguf")
    assert t2.path != t.path
    rows = [json.loads(l) for l in t.path.read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in rows] == ["session", "user", "assistant", "event"]
    assert rows[0]["persona_version"] == "v3" and rows[2]["finish"] == "stop"


def test_transcripts_live_in_gitignored_logs_dir():
    assert config.TRANSCRIPT_DIR.parent == config.LOG_DIR
    assert "logs/" in (config.ROOT / ".gitignore").read_text()
