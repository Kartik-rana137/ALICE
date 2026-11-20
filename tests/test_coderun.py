"""run_code: Python always; C, C++ and Java when their toolchains are present (scripts/setup_toolchains.py)."""
import json
import subprocess
import sys
import threading
import time

import pytest

from alice.tools import coderun
from alice.tools.coderun import available, live_output, run_code

HAVE = available()


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append((time.monotonic(), event))

    def kinds(self):
        return [e["event"] for _, e in self.events]

    def text(self, stream=None):
        return "".join(e["text"] for _, e in self.events if e["event"] == "out" and stream in (None, e["stream"]))


def test_the_student_sees_the_program_run_live():
    """Output arrives while the program is still running, not only at the end."""
    rec = Recorder()
    code = "import sys, time\nprint('first')\ntime.sleep(1.0)\nprint('second')\nprint('oops', file=sys.stderr)"
    result = run_code("python", code, emit=rec)
    assert result.startswith("exit code 0") and "first\nsecond" in result
    kinds = rec.kinds()
    assert kinds[0] == "start" and kinds[-1] == "end" and "out" in kinds
    start, end = rec.events[0][1], rec.events[-1][1]
    assert start["title"] == "Python" and start["code"] == code and end["ok"] and end["status"].startswith("exit code 0")
    first = next(t for t, e in rec.events if e["event"] == "out" and "first" in e["text"])
    assert rec.events[-1][0] - first > 0.7
    assert rec.text("stdout") == "first\nsecond\n" and "oops" in rec.text("stderr")


def test_live_output_reaches_the_tool_and_stop_kills_the_program():
    rec, stop = Recorder(), threading.Event()
    tool = coderun.make_tool()
    threading.Timer(0.8, stop.set).start()
    with live_output(rec, cancel=stop):
        t0 = time.monotonic()
        out = tool.run(language="python", code="import time\nprint('tick')\ntime.sleep(30)")
    assert time.monotonic() - t0 < 10 and out.startswith("error: the student stopped the program")
    assert rec.events[-1][1]["status"] == "stopped by the student" and "tick" in rec.text()
    assert run_code("python", "print(1)") == "exit code 0\n1"          # outside the block: nothing recorded
    assert len([k for k in rec.kinds() if k == "start"]) == 1


def test_the_tool_line_names_the_program():
    from alice.cli import tool_summary
    assert tool_summary("run_code", json.dumps({"language": "cpp", "code": "int main() {\n}\n"})) == "a C++ program, 2 lines"
    assert tool_summary("run_code", json.dumps({"language": "py", "code": "print(1)"})) == "a Python program, 1 line"


def test_errors_are_shown_too():
    rec = Recorder()
    run_code("rust", "fn main() {}", emit=rec)
    assert rec.kinds() == ["start", "out", "end"] and "unsupported language" in rec.text("stderr")


@pytest.mark.skipif(sys.platform != "win32", reason="console windows are a Windows feature")
def test_open_in_console_starts_a_new_console_window(monkeypatch, tmp_path):
    started = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            started.update(cmd=cmd, **kw)

    monkeypatch.setattr(coderun, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(coderun.subprocess, "Popen", FakePopen)
    msg = coderun.open_in_console("python", "print(input('name? '))")
    assert msg == "opened the Python program in a console window"
    assert started["creationflags"] == subprocess.CREATE_NEW_CONSOLE and started["cmd"][-2].endswith("console_runner.py")
    plan = json.loads(open(started["cmd"][-1], encoding="utf-8").read())
    assert plan["title"] == "ALICE - Python program" and plan["build"] is None and plan["run"][-1] == "main.py"


def test_the_console_runner_runs_the_plan_and_waits_for_enter(tmp_path):
    plan = coderun.prepare("python", "print('hi ' + input())", tmp_path)
    (tmp_path / "plan.json").write_text(json.dumps({"title": "t", "build": None, "run": plan.run, "cwd": str(tmp_path)}))
    p = subprocess.run([sys.executable, "-I", str(coderun.CONSOLE_RUNNER), str(tmp_path / "plan.json")],
                       input="ALICE\n\n", capture_output=True, text=True, timeout=60)
    assert "hi ALICE" in p.stdout and "exit code 0" in p.stdout and "Press Enter" in p.stdout


def test_python_output_input_errors_and_unicode():
    assert run_code("python", "print(6*7)") == "exit code 0\n42"
    out = run_code("python", "n = int(input())\nprint(sum(range(n + 1)))", stdin="100\n")
    assert out.endswith("5050")
    err = run_code("python", "raise ValueError('nope')")
    assert err.startswith("exit code 1") and "ValueError: nope" in err
    assert "β₁ = 0.9 ✓" in run_code("python", "print('β₁ = 0.9 ✓')")             # no cp1252 crash on Windows


def test_input_without_stdin_ends_instead_of_hanging():
    out = run_code("python", "name = input('name? ')\nprint(name)", timeout=5)
    assert "EOFError" in out and "timed out" not in out


def test_timeouts_and_limits():
    assert "timed out after 1 s while running" in run_code("python", "while True: pass", timeout=1)
    assert run_code("python", "print('x' * 10000)").endswith("...[stdout truncated]")
    assert run_code("rust", "fn main() {}").startswith("error: unsupported language")


def test_secrets_are_not_passed_to_the_program(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    out = run_code("python", "import os\nprint(os.environ.get('ANTHROPIC_API_KEY'))")
    assert out.endswith("None")


def test_the_tool_asks_first_and_lists_its_languages():
    tool = coderun.make_tool()
    assert tool.name == "run_code" and tool.needs_confirmation
    assert tool.parameters["properties"]["language"]["enum"] == ["python", "c", "cpp", "java"]
    assert tool.run(language="python", code="print(1+1)") == "exit code 0\n2"


@pytest.mark.skipif(not HAVE["c"], reason="no C compiler (run scripts/setup_toolchains.py)")
def test_c_with_scanf_and_compile_errors():
    code = '#include <stdio.h>\nint main(void){int a,b; scanf("%d %d",&a,&b); printf("%d\\n", a*b); return 0;}'
    assert run_code("c", code, stdin="6 7\n").endswith("42")
    bad = run_code("c", "int main(void){ return x; }")
    assert bad.startswith("compile error:") and "x" in bad
    warn = run_code("c", "#include <stdio.h>\nint main(void){int unused; printf(\"hi\\n\"); return 0;}")
    assert "compiler warnings:" in warn and warn.endswith("hi")


@pytest.mark.skipif(not HAVE["c"], reason="no C compiler (run scripts/setup_toolchains.py)")
def test_c_output_is_live_too():
    """C's stdout to a pipe is fully buffered; ALICE links in setvbuf(stdout, NULL, _IONBF, 0)."""
    rec = Recorder()
    code = ('#include <stdio.h>\n#include <time.h>\nint main(void){ printf("first\\n"); clock_t t = clock();\n'
            '  while (clock() - t < CLOCKS_PER_SEC) {}\n  printf("second\\n"); return 0; }')
    assert run_code("c", code, emit=rec).endswith("first\nsecond")
    assert "compiling" in [e.get("text") for _, e in rec.events if e["event"] == "phase"]
    first = next(t for t, e in rec.events if e["event"] == "out" and "first" in e["text"])
    assert rec.events[-1][0] - first > 0.7


@pytest.mark.skipif(not HAVE["cpp"], reason="no C++ compiler (run scripts/setup_toolchains.py)")
def test_cpp_with_cin_and_the_standard_library():
    code = ("#include <iostream>\n#include <vector>\n#include <algorithm>\nint main(){int n; std::cin>>n; "
            "std::vector<int> v(n); for(auto&x:v) std::cin>>x; std::sort(v.begin(), v.end()); "
            "for(int x: v) std::cout<<x<<' '; std::cout<<'\\n';}")
    assert run_code("cpp", code, stdin="5\n3 1 4 1 5\n").endswith("1 1 3 4 5")


@pytest.mark.skipif(not HAVE["java"], reason="no JDK (run scripts/setup_toolchains.py)")
def test_java_finds_the_class_and_reads_input():
    code = ("import java.util.Scanner;\npublic class Greeter {\n  public static void main(String[] a) {\n"
            "    Scanner s = new Scanner(System.in);\n    System.out.println(\"Hello, \" + s.nextLine() + \"!\");\n  }\n}")
    assert run_code("java", code, stdin="ALICE\n").endswith("Hello, ALICE!")
    assert run_code("java", "public class A { void f() { int x = \"s\"; } }").startswith("compile error:")
