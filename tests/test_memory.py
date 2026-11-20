"""Long-term memory and compaction, without a model: the summariser and the server are faked."""
import datetime

import pytest

import alice.session as session_mod
from alice import config, memory, persona
from alice.agent import TurnResult
from alice.config import Profile
from alice.conversation import Conversation, turns_text
from alice.memory import MemoryStore, filter_facts, grounded, similar


# ----- fact grounding and de-duplication
def test_grounded_needs_the_students_own_words():
    said = "I'm taking Distributed Systems this term and my project uses Rust. Exams are in January."
    assert grounded("The student is taking Distributed Systems this term.", said)
    assert grounded("The student uses Rust for their project.", said)
    assert not grounded("The student is preparing for a PhD in quantum computing.", said)
    assert not grounded("The student likes it.", said)                    # nothing checkable


def test_similar_catches_rephrased_duplicates():
    assert similar("The student uses Rust for their project.", "The student's project uses Rust.")
    assert not similar("The student uses Rust.", "The student is taking Distributed Systems.")


def test_filter_facts_keeps_only_new_grounded_facts():
    said = "my thesis is on retrieval augmented generation and I write everything in python"
    kept, rejected = filter_facts(
        ["The student's thesis is on retrieval augmented generation.",
         "The student writes everything in python.",                           # already known
         "The student works at Google.",                                       # never said
         "The student's thesis is about retrieval-augmented generation."],     # duplicate of the first
        said, existing=["The student writes everything in Python."])
    assert kept == ["The student's thesis is on retrieval augmented generation."]
    assert len(rejected) == 3 and "The student works at Google." in rejected


# ----- the store
def test_store_add_dedupe_delete_and_counts(tmp_path):
    store = MemoryStore(tmp_path)
    a = store.add("fact", "The student is taking Distributed Systems.", source="user")
    assert a and store.count("fact") == 1
    assert store.add("fact", "The student takes Distributed Systems.") is None      # near-duplicate
    b = store.add("fact", "The student prefers short explanations with code.")
    assert store.count("fact") == 2
    assert store.delete(b.id) and not store.delete(9999)
    assert store.count("fact") == 1
    store.replace_session_summary("s1", "Worked on Raft leader election.")
    store.replace_session_summary("s1", "Worked on Raft leader election and log replication.")
    assert [n.text for n in store.sessions()] == ["Worked on Raft leader election and log replication."]
    assert store.clear() == 2 and store.count("fact") == 0 and store.count("session") == 0


def test_prompt_section_puts_user_notes_first_and_stays_bounded(tmp_path):
    store = MemoryStore(tmp_path)
    t0 = datetime.datetime(2026, 11, 1, 12, 0)
    for i in range(30):
        store.add("fact", f"The student finished assignment number {i} about topic{i}.",
                  when=t0 + datetime.timedelta(minutes=i))
    store.add("fact", "The student's exam is on 2026-12-15.", source="user", when=t0)
    store.add("session", "Debugged a segfault in their C allocator.", session="a", when=t0)
    section = store.prompt_section(max_facts=5, max_sessions=3)
    lines = section.splitlines()
    assert lines[1] == "- The student's exam is on 2026-12-15."          # user-given note first, though oldest
    assert "assignment number 29" in section and "assignment number 10" not in section
    assert "Recent sessions:" in section and "2026-11-01: Debugged a segfault" in section
    assert len(store.prompt_section(max_facts=30, max_chars=400)) <= 400
    assert MemoryStore(tmp_path / "empty").prompt_section() == ""


def test_memory_section_goes_into_the_system_prompt():
    p = persona.system_prompt(tools=[], today=datetime.date(2026, 11, 20), memory="What you remember:\n- x")
    assert p.rstrip().endswith("- x") and "Today's date: 2026-11-20." in p


# ----- compaction in the conversation
def _turn(c: Conversation, q: str, a: str, note: str = "") -> None:
    c.add_user(q + note)
    c.add({"role": "assistant", "content": a})


def test_compaction_replaces_oldest_turns_with_a_summary():
    c = Conversation("SYSTEM", ctx=4000, reply_budget=1000, margin=0)
    for i in range(5):
        _turn(c, f"question {i} " + "x" * 600, f"answer {i} " + "y" * 600, note=persona.turn_note("a pun"))
    assert c.needs_compaction(fraction=0.5, keep_turns=2)
    old = c.oldest_turns(keep_turns=2)
    assert len(old) == 3
    text = turns_text(old)
    assert "Student: question 0" in text and "ALICE: answer 0" in text and "Note from ALICE" not in text
    c.compact_oldest(len(old), "We covered questions 0-2.")
    assert [m["content"][:10] for m in c.history if m["role"] == "user"] == ["question 3", "question 4"]
    system = c.messages()[0]["content"]
    assert system.startswith("SYSTEM") and "We covered questions 0-2." in system
    assert not c.needs_compaction(fraction=0.5, keep_turns=2)
    c.reset()
    assert c.summary == "" and c.messages()[0]["content"] == "SYSTEM"


def test_summaries_are_joined_not_merged_by_the_model():
    assert memory.join_summaries("", "B happened.") == "B happened."
    assert memory.join_summaries("A happened. Then C.", "B happened.") == "A happened. Then C. B happened."
    long = " ".join(f"Sentence {i}." for i in range(100))
    joined = memory.join_summaries(long, "The newest part.", max_chars=120)
    assert len(joined) <= 120 and joined.endswith("The newest part.") and "Sentence 99." in joined
    assert "Sentence 0." not in joined                                   # oldest sentences dropped first


def test_turns_text_shows_tool_use_and_shortens_replies():
    turn = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "calculate", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "2+2 = 4"},
            {"role": "assistant", "content": "z" * 50}]
    assert turns_text([turn], max_reply_chars=10) == ("Student: q\n(ALICE used calculate)\n"
                                                      "ALICE: zzzzzzzzzz [rest of the reply omitted from these notes]")


def test_compaction_waits_until_it_frees_real_space():
    c = Conversation("S", ctx=4000, reply_budget=1000, margin=0)
    c.summary = "s" * 3000                                      # a long summary alone exceeds the threshold
    for i in range(3):
        _turn(c, f"q{i}", f"a{i}")
    assert c.history_tokens() > 0.3 * c.budget
    assert not c.needs_compaction(fraction=0.3, keep_turns=2)    # compacting one tiny turn frees nothing


# ----- the session: commands, saving, background compaction (model calls faked)
@pytest.fixture
def mem_session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "memory")
    s = session_mod.ChatSession(Profile("default", "fake.gguf"), workspace=str(tmp_path), tools=False,
                                transcript=False, memory_notes=True, claude_escalation=False)
    s.convo = Conversation(s._system_prompt(), ctx=6000, reply_budget=1000, margin=0)
    s._today = datetime.date.today()
    calls = []

    def fake_summarize(text, profile, known=(), earlier=""):
        calls.append({"text": text, "known": list(known), "earlier": earlier})
        return {"summary": f"Summary #{len(calls)} of: {text[:40]}",
                "facts": ["The student is writing a compiler in OCaml.", "The student lives in Berlin."]}
    monkeypatch.setattr(memory, "summarize", fake_summarize)
    monkeypatch.setattr(session_mod, "warm_up", lambda *a, **k: {})
    yield s, calls
    s.memory.close()


def test_remember_memory_and_forget_commands(mem_session):
    s, _ = mem_session
    assert "usage" in s.command("/remember").message
    msg = s.command("/remember My exam is on 15 December").message
    assert msg.startswith("(noted #1")
    assert "already have a note" in s.command("/remember my exam is on 15 December").message
    assert s._pending_extra == ["The student asked you to remember: My exam is on 15 December"]
    listing = s.command("/memory").message
    assert "#1 My exam is on 15 December  (you told me)" in listing
    assert s.command("/forget 1").message == "(note #1 deleted)"
    assert s.command("/forget 1").message == "(no note #1)"
    assert "usage" in s.command("/forget").message
    assert "no notes yet" in s.command("/memory").message


def test_save_memory_keeps_only_grounded_facts(mem_session):
    s, calls = mem_session
    _turn(s.convo, "I'm writing a compiler in OCaml for my course, how do I parse expressions?", "Use Pratt parsing.")
    update = s.save_memory()
    assert update.facts == ["The student is writing a compiler in OCaml."]
    assert update.rejected == ["The student lives in Berlin."]                      # never said
    assert [n.text for n in s.memory.facts()] == ["The student is writing a compiler in OCaml."]
    assert s.memory.sessions()[0].text.startswith("Summary #1")
    assert "Student: I'm writing a compiler" in calls[0]["text"]


def test_tiny_conversations_are_not_saved(mem_session):
    s, calls = mem_session
    _turn(s.convo, "hi", "Hello!")
    assert s.save_memory() is None and not calls


def test_reset_saves_in_the_background_and_starts_a_new_session(mem_session):
    s, calls = mem_session
    _turn(s.convo, "I'm writing a compiler in OCaml, explain lexers please", "A lexer splits text into tokens.")
    old_id = s.session_id
    res = s.command("/reset")
    assert res.reset and "saved to memory in the background" in res.message
    s.wait_background(5)
    assert s.convo.turns() == [] and s.session_id != old_id
    assert s.memory.sessions()[0].session == old_id
    assert any("noted" in n.text for n in s.pending_notices)


def test_long_conversation_is_compacted_after_a_turn(mem_session, monkeypatch):
    s, calls = mem_session
    for i in range(6):
        _turn(s.convo, f"I'm writing a compiler in OCaml, question {i} " + "x" * 900, f"answer {i} " + "y" * 900)

    def fake_run_turn(messages, profile, toolbox, **kw):
        return TurnResult(reply="short answer", messages=[{"role": "assistant", "content": "short answer"}],
                          finish="stop")
    monkeypatch.setattr(session_mod, "run_turn", fake_run_turn)
    monkeypatch.setattr(config, "COMPACT_AT", 0.3)
    out = s.ask("one more question")
    assert out.kept
    s.wait_background(5)
    turns = s.convo.turns()
    assert len(turns) == config.COMPACT_KEEP_TURNS and s.convo.summary.startswith("Summary #1")
    assert "question 0" in calls[0]["text"] and "one more question" not in calls[0]["text"]
    assert [n.text for n in s.memory.facts()] == ["The student is writing a compiler in OCaml."]
    notices = s.ask("next").notices                         # background notices arrive with the next answer
    assert any(n.kind == "memory" and "compacted" in n.text for n in notices)


def test_remembered_note_rides_along_with_the_next_question_once(mem_session, monkeypatch):
    s, _ = mem_session
    seen = []

    def fake_run_turn(messages, profile, toolbox, **kw):
        seen.append(messages[-1]["content"])
        return TurnResult(reply="ok", messages=[{"role": "assistant", "content": "ok"}], finish="stop")
    monkeypatch.setattr(session_mod, "run_turn", fake_run_turn)
    s.command("/remember I use Neovim")
    s.ask("first")
    s.ask("second")
    assert "The student asked you to remember: I use Neovim" in seen[0]
    assert "remember" not in seen[1]
