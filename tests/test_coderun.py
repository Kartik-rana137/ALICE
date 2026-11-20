"""run_code: Python always; C, C++ and Java when their toolchains are present (scripts/setup_toolchains.py)."""
import pytest

from alice.tools import coderun
from alice.tools.coderun import available, run_code

HAVE = available()


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
