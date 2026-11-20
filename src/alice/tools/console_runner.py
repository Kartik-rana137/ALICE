"""Runs one program in its own console window, started by `coderun.open_in_console`:

    python -I console_runner.py PLAN.json

The plan holds the build command (C, C++, Java), the run command and the folder. The student watches the
compiler and the program in the window, types into it, and closes it with Enter. Standard library only:
it runs isolated (-I), outside the alice package.
"""
import json
import subprocess
import sys
import time


def main() -> int:
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(plan["title"])
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)     # UTF-8, so C programs' accents show
        ctypes.windll.kernel32.SetConsoleCP(65001)
    print(f"--- {plan['title']} ---", flush=True)
    code = None
    try:
        if plan.get("build"):
            print("compiling ...", flush=True)
            if subprocess.run(plan["build"], cwd=plan["cwd"]).returncode != 0:
                print("\n--- compile error (see above) ---")
                return 1
            print("running ...\n", flush=True)
        t0 = time.monotonic()
        code = subprocess.run(plan["run"], cwd=plan["cwd"]).returncode
        print(f"\n--- exit code {code} ({time.monotonic() - t0:.1f} s) ---")
    except KeyboardInterrupt:
        print("\n--- stopped (Ctrl+C) ---")
    except OSError as e:
        print(f"\n--- could not start it: {e} ---")
    finally:
        try:
            input("Press Enter to close this window.")
        except (EOFError, KeyboardInterrupt):
            pass
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
