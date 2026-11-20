"""Date arithmetic, system information and read-only git tools, plus the local time in the turn note."""
import datetime
import shutil
import subprocess

import pytest

from alice import persona
from alice.tools import default_toolbox, system


@pytest.fixture
def fixed_today(monkeypatch):
    monkeypatch.setattr(system, "_today", lambda: datetime.date(2024, 2, 28))      # a Wednesday


def test_date_calc_weekday_spans_and_offsets(fixed_today):
    out = system.date_calc("2024-02-29")
    assert "2024-02-29 (Thursday, ISO week 9)" in out
    out = system.date_calc("today", "2024-03-01")
    assert "2 days (0 weeks and 2 days)" in out and "2 of them weekdays" in out           # leap day counted
    assert "today is 2024-02-28 (Wednesday" in out
    out = system.date_calc("today", "2024-03-11")
    assert "12 days (1 week and 5 days)" in out and "8 of them weekdays" in out
    assert "2024-02-28 + 30 days = 2024-03-29 (Friday" in system.date_calc(add_days=30)
    assert "2024-02-28 - 1 day = 2024-02-27" in system.date_calc(add_days=-1)
    assert "2024-12-12 (Thursday" in system.date_calc("12th December 2024")
    assert "2024-08-01 (Thursday" in system.date_calc("August 1st 2024")                 # "st" inside "August" kept
    assert system.date_calc("next blue moon").startswith("error: can't read the date")


def test_system_info_reports_the_machine_and_only_checks_known_programs(tmp_path):
    out = system.system_info(["git", "rm", "definitely-not-installed"], tmp_path)
    assert out.startswith("OS: ") and "Python running ALICE:" in out and "CPU:" in out and "GPU:" in out
    assert "Disk (workspace drive):" in out
    assert "rm: not checked" in out                                       # not on the allowlist: never run
    assert "definitely-not-installed: not checked" in out
    if shutil.which("git"):
        assert "git: git version" in out


@pytest.mark.skipif(not shutil.which("git"), reason="git not installed")
def test_git_tool_is_read_only_and_stays_in_the_workspace(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "first commit")
    (tmp_path / "a.py").write_text("x = 2\n")
    reader = system.GitReader(tmp_path)
    assert " M a.py" in reader.run("status")
    assert "-x = 1" in reader.run("diff") and "+x = 2" in reader.run("diff", path="a.py")
    assert "first commit" in reader.run("log", n=5) and "first commit" in reader.run("show")
    assert reader.run("log", ref="--output=/tmp/x").startswith("error:")          # no option injection
    assert reader.run("diff", path="../elsewhere").startswith("error:")
    assert reader.run("push").startswith("error: action must be")
    plain = tmp_path.parent / (tmp_path.name + "_not_a_repo")
    plain.mkdir()
    assert "is not a git repository" in system.GitReader(plain).run("status")


def test_new_tools_are_in_the_default_toolbox(tmp_path):
    names = default_toolbox(workspace=tmp_path).tools
    assert {"date_calc", "system_info", "git"} <= set(names)
    assert "git" not in default_toolbox(workspace=None).tools          # no workspace, no git
    guide = persona.tools_guide(names)
    assert "date_calc:" in guide and "system_info:" in guide and "- git:" in guide


def test_dates_do_not_trigger_the_unknown_paper_guard():
    """Live run: "How many days until 12 December 2026?" matched the "Gemma 4" name pattern, so the reply ended
    with an offer to look up "a specific paper or method tied to that date"."""
    assert not persona.needs_reference_guard("How many days are left until 12 December 2026?")
    assert not persona.needs_reference_guard("Is my deadline on Friday 12 a weekend?")
    assert persona.needs_reference_guard("What changed in Gemma 4?")
    assert persona.named_terms("Compare Gemma 4 with Qwen 3.5 by December 2026") == ["Gemma 4", "Qwen 3.5"]


def test_casual_messages_are_recognised_and_study_questions_are_not():
    casual = ["hi", "Good morning! How are you today?", "thanks, that helped a lot", "What time is it?",
              "lol that was a good one", "Do you like music?", "ok, see you tomorrow", "I just finished my exams!",
              "How many days until 25 December?", "What's your favourite programming language?"]
    study = ["Can you explain what a hash table is?", "Why does gradient descent need a learning rate?",
             "Explain what a closure is, thanks", "hi, can you help me debug this function?",
             "How are hash maps implemented?", "What is entropy in thermodynamics?", "What is recursion?",
             "Write a Python function that reverses a linked list."]
    assert [q for q in casual if not persona.is_casual(q)] == []
    assert [q for q in study if persona.is_casual(q)] == []


def test_casual_note_rides_with_casual_messages_only(tmp_path, monkeypatch):
    import alice.session as session_mod
    from alice.agent import TurnResult
    from alice.config import Profile
    from alice.conversation import Conversation
    s = session_mod.ChatSession(Profile("default", "fake.gguf"), workspace=str(tmp_path), tools=False,
                                transcript=False, memory_notes=False, claude_escalation=False)
    s.convo, s._today = Conversation("SYS", ctx=16384, reply_budget=1000), datetime.date.today()
    seen = []

    def fake_run_turn(messages, profile, toolbox, **kw):
        seen.append(messages[-1]["content"])
        return TurnResult(reply="ok", messages=[{"role": "assistant", "content": "ok"}], finish="stop")
    monkeypatch.setattr(session_mod, "run_turn", fake_run_turn)
    s.ask("thanks!")
    s.ask("Explain how a B-tree splits a node.")
    assert persona.CASUAL_NOTE in seen[0] and persona.CASUAL_NOTE not in seen[1]


def test_turn_note_carries_the_local_time():
    now = datetime.datetime(2026, 11, 20, 14, 32, tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    note = persona.turn_note("a pun", now=now)
    assert "Local time now: Fri 20 Nov 2026, 14:32 (UTC+05:30)" in note
    assert "Local time" not in persona.turn_note("a pun")                 # evals pass no time: unchanged
