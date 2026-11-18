import json

import pytest

import alice.agent as agent
from alice.config import PROFILES
from alice.conversation import Conversation
from alice.llm import Event, iter_sse_events
from alice.tools import Tool, ToolBox, arxiv, calc, files, pyexec


# ---------- calculator ----------

@pytest.mark.parametrize("expr,expected", [
    ("2+3*4", "14"),
    ("0.99*0.001/(0.99*0.001+0.05*0.999)", "0.01943462898"),
    ("784*128+128+128*10+10", "101770"),
    ("ceil(1250*0.8/32)", "32"),
    ("2^10", "1024"),
    ("sqrt(16) + log(e)", "5"),
    ("comb(5, 2)", "10"),
    ("max(3, 7, 2)", "7"),
])
def test_calculate_values(expr, expected):
    assert calc.calculate(expr).rsplit(" = ", 1)[1].split(" (")[0] == expected


def test_calculate_adds_copyable_forms():
    assert calc.calculate("784*128+128+1290").endswith("= 101770 (101,770)")
    assert calc.calculate("0.99*0.001/(0.99*0.001+0.05*0.999)").endswith("(= 1.943%)")
    assert calc.calculate("6*7").endswith("= 42")


@pytest.mark.parametrize("reply,output,ok", [
    ("The total is **101,770**.", "x = 101770 (101,770)", True),
    ("The tool confirms the total is **10,177**.", "x = 101770 (101,770)", False),   # real Phase 2 failure
    ("About **1.94%** have the disease.", "x = 0.01943462898 (= 1.943%)", True),
    ("Roughly 0.0194.", "x = 0.01943462898 (= 1.943%)", True),
    ("Approximately **2%**.", "x = 0.01943462898 (= 1.943%)", False),                # too imprecise
    ("There are 32 batches.", "ceil(1000/32) = 32", True),
])
def test_calculation_grounding_check(reply, output, ok):
    assert calc.reply_mentions(reply, calc.last_result_value(output)) is ok


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "open('x')", "(1).__class__", "a + 1", "[x for x in range(3)]",
    "lambda: 1", "9**99999", "factorial(100000)",
])
def test_calculate_rejects_unsafe_or_huge(expr):
    assert calc.calculate(expr).startswith("error:")


def test_calculate_division_by_zero():
    assert calc.calculate("1/0") == "error: division by zero"


# ---------- toolbox ----------

def _box():
    return ToolBox().add(
        Tool("echo", "echo", {"type": "object"}, lambda text="": f"echo {text}"),
        Tool("boom", "fails", {"type": "object"}, lambda: 1 / 0),
        Tool("big", "big output", {"type": "object"}, lambda: "x" * 10000),
        Tool("danger", "needs ok", {"type": "object"}, lambda: "ran", needs_confirmation=True))


def test_toolbox_call_ok_and_specs():
    box = _box()
    assert box.call("echo", '{"text": "hi"}').output == "echo hi"
    assert {s["function"]["name"] for s in box.specs()} == {"echo", "boom", "big", "danger"}


@pytest.mark.parametrize("name,args,fragment", [
    ("nope", "{}", "unknown tool"),
    ("echo", "{not json", "could not parse"),
    ("echo", "[1, 2]", "JSON object"),
    ("echo", '{"wrong": 1}', "bad arguments"),
    ("boom", "{}", "ZeroDivisionError"),
])
def test_toolbox_errors_are_returned_not_raised(name, args, fragment):
    r = _box().call(name, args)
    assert not r.ok and fragment in r.output


def test_toolbox_truncates_and_confirms():
    box = _box()
    assert "truncated" in box.call("big", "{}").output
    assert box.call("danger", "{}").declined                              # no confirm callback -> declined
    assert box.call("danger", "{}", confirm=lambda t, a: False).declined
    assert box.call("danger", "{}", confirm=lambda t, a: True).output == "ran"


# ---------- file tools ----------

@pytest.fixture
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("\n".join(f"line {i}" for i in range(1, 501)) + "\ndef target(): pass\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret = target")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01binary")
    return files.Workspace(tmp_path)


def test_workspace_blocks_escape(ws, tmp_path):
    assert ws.read_file("../outside.txt").startswith("error:")
    assert ws.list_files("..").startswith("error:")
    assert ws.search_text("x", path="../..").startswith("error:")


def test_list_and_search_skip_git(ws):
    listing = ws.list_files()
    assert "src/main.py" in listing and ".git" not in listing
    hits = ws.search_text(r"def target")
    assert "src/main.py:501" in hits and ".git" not in hits


def test_read_file_pages_and_rejects_binary(ws):
    first = ws.read_file("src/main.py")
    assert "lines 1-400 of 501" in first and "start_line=401" in first
    assert "  501| def target(): pass" in ws.read_file("src/main.py", start_line=490)
    assert ws.read_file("blob.bin") == "error: binary file"


# ---------- python runner (subprocess) ----------

def test_run_python_output_error_and_timeout():
    assert "exit code 0\n42" == pyexec.run_python("print(6*7)")
    err = pyexec.run_python("raise ValueError('nope')")
    assert "exit code 1" in err and "ValueError" in err
    assert "timed out" in pyexec.run_python("while True: pass", timeout=1)


# ---------- arXiv (offline parts) ----------

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <summary>  The dominant sequence transduction models are based on complex recurrent networks. </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_feed():
    [p] = arxiv.parse_feed(FEED)
    assert p["id"] == "1706.03762v7" and p["title"] == "Attention Is All You Need"
    assert p["authors"] == ["Ashish Vaswani", "Noam Shazeer"] and p["published"] == "2017-06-12"
    assert "arXiv:1706.03762v7" in arxiv._format([p])


def test_paper_rejects_bad_ids_without_network():
    assert arxiv.paper("not-an-id; drop table").startswith("error:")


# ---------- streamed tool calls ----------

def _chunk(delta=None, finish=None):
    return "data: " + json.dumps({"choices": [{"delta": delta or {}, "finish_reason": finish}]})


def test_sse_assembles_fragmented_tool_calls():
    lines = [
        _chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                "function": {"name": "calculate", "arguments": '{"expr'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ession": "1+1"}'}}]}),
        _chunk({"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "arxiv_search", "arguments": "{}"}}]}),
        _chunk(finish="tool_calls"),
    ]
    evs = list(iter_sse_events(lines))
    assert [e.kind for e in evs] == ["tool_calls", "finish"]
    assert evs[0].calls == [{"id": "c1", "name": "calculate", "arguments": '{"expression": "1+1"}'},
                            {"id": "c2", "name": "arxiv_search", "arguments": "{}"}]


# ---------- agent loop (fake model) ----------

def test_run_turn_executes_tools_and_feeds_results_back(monkeypatch):
    seen_requests = []

    def fake_stream(messages, profile, thinking=False, tools=None):
        seen_requests.append((list(messages), tools))
        if len(seen_requests) == 1:
            yield Event("tool_calls", calls=[{"id": "c1", "name": "calculate", "arguments": '{"expression": "6*7"}'}])
            yield Event("finish", "tool_calls")
        else:
            yield Event("content", "It is **42**.")
            yield Event("finish", "stop", {"predicted_n": 5})

    monkeypatch.setattr(agent, "chat_stream", fake_stream)
    box = ToolBox().add(calc.TOOL)
    res = agent.run_turn([{"role": "user", "content": "6*7?"}], PROFILES["default"], box)
    assert res.reply == "It is **42**." and res.rounds == 2
    assert [m["role"] for m in res.messages] == ["assistant", "tool", "assistant"]
    assert res.messages[1]["content"] == "6*7 = 42" and res.messages[1]["tool_call_id"] == "c1"
    assert seen_requests[1][0][-1]["role"] == "tool"            # result was sent back to the model
    assert seen_requests[0][1] is not None


def test_run_turn_last_round_has_no_tools(monkeypatch):
    calls = []

    def always_tools(messages, profile, thinking=False, tools=None):
        calls.append(tools)
        if tools:
            yield Event("tool_calls", calls=[{"id": "x", "name": "calculate", "arguments": '{"expression": "1"}'}])
            yield Event("finish", "tool_calls")
        else:
            yield Event("content", "done")
            yield Event("finish", "stop")

    monkeypatch.setattr(agent, "chat_stream", always_tools)
    res = agent.run_turn([{"role": "user", "content": "loop"}], PROFILES["default"],
                         ToolBox().add(calc.TOOL), max_rounds=3)
    assert res.reply == "done" and calls[-1] is None and len(calls) == 3


# ---------- memory keeps tool exchanges together ----------

def test_conversation_trims_whole_turns_with_tool_messages():
    c = Conversation("sys", ctx=1500, reply_budget=300, margin=100)
    for i in range(10):
        c.add_user(f"q{i} " + "x" * 300)
        c.add({"role": "assistant", "content": None,
               "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "calculate", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": f"t{i}", "content": "r" * 300},
              {"role": "assistant", "content": f"a{i}"})
    msgs = c.messages()[1:]
    assert msgs[0]["role"] == "user"
    ids_called = {m["tool_calls"][0]["id"] for m in msgs if m.get("tool_calls")}
    ids_answered = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
    assert ids_called == ids_answered and msgs[-1]["content"] == "a9"
    c.pop_turn()
    assert c.history[-1]["content"] == "a8"


def test_arxiv_query_strategies_drop_stopwords_and_try_title_first():
    s = arxiv.query_strategies("Attention Is All You Need")
    assert s[0] == ('ti:"Attention Is All You Need"', "exact title match")
    assert s[1][0] == "all:Attention AND all:All AND all:You AND all:Need"
    assert all("all:Is" not in q for q, _ in s)
    assert arxiv.query_strategies("") == []


def test_calculate_explains_that_code_belongs_in_run_python():
    out = calc.calculate("import math; z = [2.0, 1.0]; z")
    assert out.startswith("error:") and "run_python" in out
    assert calc.calculate("exp(2)/(exp(2)+exp(1)+exp(-1))").startswith("exp(2)")


def test_run_turn_breaks_identical_failing_tool_loops_and_strips_leaked_markup(monkeypatch):
    """Reproduces the Phase 2 failure: same invalid calculate call 5x, then raw <tool_call> text."""
    requests = []
    bad = '{"expression": "import math; z = [2.0]; z"}'

    def looping(messages, profile, thinking=False, tools=None):
        requests.append((messages, tools))
        if tools:
            yield Event("tool_calls", calls=[{"id": f"c{len(requests)}", "name": "calculate", "arguments": bad}])
            yield Event("finish", "tool_calls")
        else:
            yield Event("content", "<tool_call>\n<function=calculate>\n<parameter=expression>\nimport math\n"
                                   "</parameter>\n</function>\n</tool_call>")
            yield Event("finish", "stop")

    executed = []
    box = ToolBox().add(Tool("calculate", "calc", {"type": "object"},
                             lambda expression: executed.append(expression) or calc.calculate(expression)))
    monkeypatch.setattr(agent, "chat_stream", looping)
    res = agent.run_turn([{"role": "user", "content": "softmax?"}], PROFILES["default"], box)
    assert len(executed) == 1                         # the identical failing call ran only once
    assert res.repeated_calls == 1 and res.rounds == 3
    assert requests[-1][1] is None and requests[-1][0][-1]["content"] == agent.FINAL_ROUND_NOTE
    assert res.leaked_markup and "<tool_call>" not in res.reply and res.reply == agent.STUCK_REPLY
    assert res.messages[-1] == {"role": "assistant", "content": agent.STUCK_REPLY}
    assert all(m.get("content") != agent.FINAL_ROUND_NOTE for m in res.messages)   # note not saved


def test_run_turn_keeps_text_around_leaked_markup(monkeypatch):
    def leaky(messages, profile, thinking=False, tools=None):
        yield Event("content", "Softmax turns scores into probabilities. <tool_call>junk</tool_call>")
        yield Event("finish", "stop")
    monkeypatch.setattr(agent, "chat_stream", leaky)
    res = agent.run_turn([{"role": "user", "content": "x"}], PROFILES["default"], ToolBox().add(calc.TOOL))
    assert res.reply == "Softmax turns scores into probabilities." and res.leaked_markup


def test_cli_printer_never_shows_leaked_tool_markup(capsys):
    from alice.cli import _Printer
    p = _Printer(show_thinking=False)
    for piece in ["Softmax maps scores ", "to probabilities. <", "tool", "_call>\n<function=calc", "ulate>"]:
        p.event(Event("content", piece))
    p.event(Event("finish", "stop"))
    assert capsys.readouterr().out == "Softmax maps scores to probabilities. "


def test_cli_printer_releases_held_back_text_that_was_not_markup(capsys):
    from alice.cli import _Printer
    p = _Printer(show_thinking=False)
    for piece in ["if a <", "b then"]:
        p.event(Event("content", piece))
    p.event(Event("finish", "stop"))
    assert capsys.readouterr().out == "if a <b then"
