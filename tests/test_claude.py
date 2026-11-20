"""Escalation to Claude without network access: a fake SDK client stands in for anthropic.Anthropic."""
import datetime
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import alice.session as session_mod
from alice import claude, persona
from alice.claude import ClaudeClient, ClaudeError, UsageLedger, build_request, redact, suggestion
from alice.config import ClaudeSettings, Profile
from alice.conversation import Conversation

SETTINGS = ClaudeSettings(enabled=True, model="claude-opus-5-5", effort="high", max_tokens=16000,
                          monthly_budget_usd=5.0, context_turns=2)


def library_turn(question="What learning rate did the Transformer paper use at step 8000?",
                 answer="It was 0.0156 [L1].") -> list[dict]:
    return [
        {"role": "user", "content": question + persona.turn_note("a pun")},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "p1", "type": "function", "function": {"name": "search_library", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "p1",
         "content": "Passages from the student's own library ...\n\n[L1] Attention Is All You Need, p. 7\n"
                    "lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)\n\n[L2] Other, p. 2\ntext"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "calculate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "512**-0.5 * (8000/4000)**-1.5 = 0.015625"},
        {"role": "assistant", "content": answer},
    ]


def plain_turn(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


# ----- router and redaction
def test_suggestion_lists_the_failed_checks_once():
    assert suggestion([]) is None and suggestion(["something_else"]) is None
    s = suggestion(["unsupported_numbers", "calculation_mismatch", "unsupported_numbers"])
    assert s.startswith("This answer may be unreliable:") and s.count("passages it cites") == 1
    assert "calculator" in s


def test_redact_paths_emails_and_keys(tmp_path):
    ws = tmp_path / "proj"
    text = (f"open {ws}\\src\\main.py and {Path.home()}\\notes.txt; mail me at jane.doe@uni.example.edu; "
            "key sk-ant-REDACTED-0123456789abcdef and C:\\Users\\someone\\x.py and /home/bob/y.py")
    out, n = redact(text, ws)
    assert "<workspace>\\src\\main.py" in out and "~\\notes.txt" in out
    assert "<email>" in out and "jane.doe" not in out
    assert "<secret>" in out and "sk-ant" not in out
    assert "C:\\Users\\<user>\\x.py" in out and "/home/<user>/y.py" in out
    assert n >= 6


# ----- building the request
def test_escalating_the_last_turn_sends_question_material_and_flagged_answer(tmp_path):
    turns = [plain_turn("old q1", "old a1"), plain_turn("old q2", "old a2"), plain_turn("old q3", "old a3"),
             library_turn()]
    req = build_request(SETTINGS, turns, flags=["calculation_mismatch"], workspace=tmp_path,
                        today=datetime.date(2026, 11, 20))
    assert req.replaces_last and req.question.startswith("What learning rate")
    assert req.n_context_turns == 2                              # context_turns=2: q2 and q3 only
    assert [m["role"] for m in req.messages] == ["user", "assistant", "user", "assistant", "user"]
    assert req.messages[0]["content"] == "old q2"
    final = req.messages[-1]["content"]
    assert "Note from ALICE" not in final                        # turn notes are stripped
    assert '<tool name="search_library">' in final and "warmup_steps^-1.5" in final
    assert '<tool name="calculate">' in final
    assert "<local_answer>\nIt was 0.0156 [L1].\n</local_answer>" in final
    assert "doesn't match the calculator" in final
    assert req.n_passages == 2 and req.n_tool_results == 1
    assert "Today's date: 2026-11-20." in req.system and persona.SYSTEM_PROMPT in req.system
    assert 0 < req.est_cost < req.max_cost
    assert "ALICE's flagged answer" in req.summary() and "=== user" in req.preview()


def test_a_new_question_goes_with_recent_context_only():
    req = build_request(SETTINGS, [plain_turn("q1", "a1"), plain_turn("q2", "a2"), plain_turn("q3", "a3")],
                        question="Prove that the halting problem is undecidable.")
    assert not req.replaces_last and req.n_context_turns == 2
    assert req.messages[-1] == {"role": "user", "content": "Prove that the halting problem is undecidable."}
    assert "local_answer" not in json.dumps(req.messages)


def test_nothing_to_escalate_yet():
    with pytest.raises(ClaudeError, match="no question"):
        build_request(SETTINGS, [])


# ----- budget and usage log
def test_budget_is_enforced_from_the_usage_log(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    now = datetime.datetime.now()
    ledger.add({"time": now.isoformat(), "cost_usd": 4.99})
    ledger.add({"time": "2020-01-01T00:00:00", "cost_usd": 100.0})          # another month
    assert abs(ledger.month_total() - 4.99) < 1e-9
    client = ClaudeClient(SETTINGS, ledger, client=FakeAnthropic())
    req = build_request(SETTINGS, [library_turn()])
    with pytest.raises(ClaudeError, match="budget"):
        client.check_budget(req)


# ----- the client, with a fake SDK
class FakeStream:
    def __init__(self, texts, final, thinking=True):
        self.texts, self.final, self.thinking = texts, final, thinking

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if self.thinking:
            yield SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking"))
        for t in self.texts:
            yield SimpleNamespace(type="text", text=t)

    def get_final_message(self):
        return self.final


class FakeAnthropic:
    def __init__(self, texts=("The rate is ", "0.000494 [L1]."), stop_reason="end_turn", category=None):
        self.calls = []
        usage = SimpleNamespace(input_tokens=2000, output_tokens=1000, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        final = SimpleNamespace(model="claude-opus-5-5", stop_reason=stop_reason, usage=usage,
                                stop_details=SimpleNamespace(category=category) if category else None)
        self.texts, self.final = list(texts), final
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **params):
        self.calls.append(params)
        return FakeStream(self.texts, self.final)


def test_client_streams_records_cost_and_uses_the_right_parameters(tmp_path):
    fake = FakeAnthropic()
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    client = ClaudeClient(SETTINGS, ledger, client=fake)
    req = build_request(SETTINGS, [library_turn()], flags=["calculation_mismatch"])
    got, status = [], []
    reply = client.ask(req, on_text=got.append, on_status=status.append)
    assert "".join(got) == reply.text == "The rate is 0.000494 [L1]."
    assert status == ["thinking"]
    p = fake.calls[0]
    assert p["model"] == "claude-opus-5-5" and p["output_config"] == {"effort": "high"}
    assert p["thinking"] == {"type": "adaptive"} and p["fallbacks"] == "default"
    assert p["betas"] == [claude.FALLBACK_BETA] and p["system"] == req.system and p["max_tokens"] == 16000
    assert reply.cost_usd == pytest.approx((2000 * 4.0 + 1000 * 20.0) / 1e6)
    rec = ledger.records()[-1]
    assert rec["cost_usd"] == pytest.approx(0.028) and rec["reasons"] == ["calculation_mismatch"]
    assert "text" not in rec and "The rate" not in json.dumps(rec)          # no message text in the log
    assert reply.month_total == pytest.approx(0.028)


def test_refusal_and_cancel(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    refused = ClaudeClient(SETTINGS, ledger, client=FakeAnthropic(texts=(), stop_reason="refusal", category="cyber"))
    reply = refused.ask(build_request(SETTINGS, [library_turn()]))
    assert reply.refusal_category == "cyber" and reply.text == ""
    cancel = threading.Event()
    cancel.set()
    stopped = ClaudeClient(SETTINGS, ledger, client=FakeAnthropic()).ask(build_request(SETTINGS, [library_turn()]),
                                                                       cancel=cancel)
    assert stopped.interrupted and stopped.estimated


def test_sdk_errors_become_readable_messages(tmp_path):
    class Boom:
        beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **p: (_ for _ in ()).throw(ValueError("bad"))))
    client = ClaudeClient(SETTINGS, UsageLedger(tmp_path / "u.jsonl"), client=Boom())
    with pytest.raises(ClaudeError, match="ValueError: bad"):
        client.ask(build_request(SETTINGS, [library_turn()]))


# ----- through the chat session
@pytest.fixture
def esc_session(tmp_path):
    s = session_mod.ChatSession(Profile("default", "fake.gguf"), workspace=str(tmp_path), tools=False,
                                transcript=False, memory_notes=False, claude_escalation=False)
    s.convo = Conversation(s._system_prompt(), ctx=16384, reply_budget=1000)
    s._today = datetime.date(2026, 11, 20)
    return s


def test_claude_command_when_escalation_is_off(esc_session):
    assert "escalation is off" in esc_session.command("/claude").message


def test_session_escalation_replaces_the_flagged_answer(esc_session, tmp_path):
    s = esc_session
    fake = FakeAnthropic()
    s.claude = ClaudeClient(SETTINGS, UsageLedger(tmp_path / "usage.jsonl"), client=fake)
    s.convo.add(*library_turn())
    s.last_flags = ["calculation_mismatch"]
    cmd = s.command("/claude")
    assert cmd.escalate and cmd.escalate_question is None
    req = s.prepare_escalation()
    out = s.escalate(req)
    assert out.error is None and out.reply.text == "The rate is 0.000494 [L1]."
    assert s.convo.history[-1]["content"] == "The rate is 0.000494 [L1]."       # flagged answer replaced
    assert s.last_flags == []
    assert any("$0.028" in n.text for n in out.notices)


def test_session_escalation_of_a_new_question_adds_a_turn(esc_session, tmp_path):
    s = esc_session
    s.claude = ClaudeClient(SETTINGS, UsageLedger(tmp_path / "usage.jsonl"), client=FakeAnthropic(texts=("Proof: ...",)))
    cmd = s.command("/claude Prove the pumping lemma")
    assert cmd.escalate_question == "Prove the pumping lemma"
    out = s.escalate(s.prepare_escalation(cmd.escalate_question))
    assert [m["role"] for m in s.convo.history] == ["user", "assistant"]
    assert s.convo.history[1]["content"] == "Proof: ..." and out.error is None


def test_failed_checks_suggest_claude_only_when_it_is_on(esc_session, tmp_path, monkeypatch):
    s = esc_session
    from alice.agent import TurnResult
    from alice.tools import ToolResult

    def fake_run_turn(messages, profile, toolbox, **kw):
        return TurnResult(reply="It is 0.0156.", messages=[{"role": "assistant", "content": "It is 0.0156."}],
                          finish="stop", tool_results=[ToolResult("calculate", {}, "2**-6 = 0.000494", True)])
    monkeypatch.setattr(session_mod, "run_turn", fake_run_turn)
    out = s.ask("what is the learning rate?")
    assert out.flags == ["calculation_mismatch"] and not any(n.kind == "escalate" for n in out.notices)
    s.claude = ClaudeClient(SETTINGS, UsageLedger(tmp_path / "usage.jsonl"), client=FakeAnthropic())
    out = s.ask("and again?")
    assert any(n.kind == "escalate" and "/claude" in n.text for n in out.notices)
