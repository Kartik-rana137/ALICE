import json

import pytest

from alice.config import PROFILES, Profile
from alice.conversation import Conversation, estimate_tokens
from alice.llm import LlamaServer, iter_sse_content


def sse(*contents, done=True):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in contents]
    if done:
        lines.append("data: [DONE]")
    return lines


def test_sse_yields_deltas_in_order():
    assert list(iter_sse_content(sse("Hel", "lo", "!"))) == ["Hel", "lo", "!"]


def test_sse_ignores_noise_and_empty_deltas():
    lines = [": keepalive", "", "data: not json",
             'data: {"choices": [{"delta": {"role": "assistant"}}]}',
             'data: {"choices": []}', *sse("ok")]
    assert list(iter_sse_content(lines)) == ["ok"]


def test_sse_stops_at_done():
    lines = sse("a", done=True) + sse("b", done=False)
    assert list(iter_sse_content(lines)) == ["a"]


def test_sse_accepts_bytes():
    assert list(iter_sse_content([l.encode() for l in sse("x")])) == ["x"]


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
    total = sum(estimate_tokens(m["content"]) for m in msgs[1:])
    assert total <= c.budget + estimate_tokens(msgs[-1]["content"])


def test_conversation_oversized_latest_message_still_kept():
    c = Conversation("sys", ctx=1200, reply_budget=300)
    c.add_user("z" * 20000)
    assert len(c.messages()) == 2


def test_conversation_reset():
    c = Conversation("sys", ctx=4096, reply_budget=500)
    c.add_user("hi"); c.reset()
    assert c.messages() == [c.system]


def test_conversation_rejects_tiny_context():
    with pytest.raises(ValueError):
        Conversation("s" * 5000, ctx=1000, reply_budget=500)


def test_profiles_build_server_commands():
    cmd = LlamaServer(PROFILES["default"]).command()
    assert "-ngl" in cmd and "--n-cpu-moe" not in cmd
    moe = LlamaServer(Profile("moe", "x.gguf", n_cpu_moe=36)).command()
    assert moe[moe.index("--n-cpu-moe") + 1] == "36"


def test_every_profile_points_at_a_model_file_name():
    assert all(p.gguf.endswith(".gguf") for p in PROFILES.values())
