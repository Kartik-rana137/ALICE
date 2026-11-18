"""Run the shared prompt set against one model via llama-server and save outputs.

Usage (from repo root):
  python scripts/run_prompts.py Qwen3.5-4B-Q4_K_M
  python scripts/run_prompts.py gemma-4-26B-A4B-it-UD-IQ4_XS --n-cpu-moe 20

Starts llama-server at 8k context, sends each prompt in benchmarks/prompts.json,
records the reply + llama.cpp timings to benchmarks/results/quality/<model>.json.
Replies are then read/graded by hand (or by a stronger model) -- see DEVLOG.
Thinking mode is disabled for fairness/latency; the persona system prompt is a
short v0 mentor-with-sass prompt (also being tested).
"""
import argparse, json, subprocess, sys, time, urllib.request, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SYSTEM = ("You are ALICE, a mentor for a computer-science (AI specialisation) student. "
          "Teach, don't just answer: guide with questions when appropriate, explain your reasoning, "
          "and be honest about uncertainty. Never invent papers, authors or citations. "
          "You may be slightly sassy, but stay kind and accurate. Be concise.")
PORT = 8089


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-cpu-moe", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1000)
    a = ap.parse_args()

    cmd = [str(ROOT / "tools/llama.cpp/llama-server.exe"), "-m", str(ROOT / f"models/{a.model}.gguf"),
           "-ngl", str(a.ngl), "-c", str(a.ctx), "--port", str(PORT), "--jinja"]
    if a.n_cpu_moe:
        cmd += ["--n-cpu-moe", str(a.n_cpu_moe)]
    log = open(ROOT / "benchmarks/results/server.log", "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if srv.poll() is not None:
                sys.exit(f"llama-server exited early (see benchmarks/results/server.log), code {srv.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            sys.exit("server did not become healthy")

        prompts = json.load(open(ROOT / "benchmarks/prompts.json", encoding="utf-8"))
        results = []
        for p in prompts:
            t0 = time.time()
            r = post("/v1/chat/completions", {
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["prompt"]}],
                "max_tokens": a.max_tokens, "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False}})
            tm = r.get("timings", {})
            results.append({"id": p["id"], "category": p["category"], "reply": r["choices"][0]["message"]["content"],
                            "finish": r["choices"][0]["finish_reason"], "wall_s": round(time.time() - t0, 1),
                            "prompt_tps": tm.get("prompt_per_second"), "gen_tps": tm.get("predicted_per_second"),
                            "gen_tokens": tm.get("predicted_n")})
            print(f'{p["id"]:20} {results[-1]["wall_s"]:6.1f}s  gen {results[-1]["gen_tps"] or 0:5.1f} t/s  [{results[-1]["finish"]}]')
        out = ROOT / "benchmarks/results/quality"
        out.mkdir(parents=True, exist_ok=True)
        json.dump({"model": a.model, "ngl": a.ngl, "n_cpu_moe": a.n_cpu_moe, "ctx": a.ctx, "results": results},
                  open(out / f"{a.model}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
