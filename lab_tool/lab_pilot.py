"""LAB episodes, judging and reporting (2026-10-05/06; the user: "can we do a tiny bit of the LAB benchmark?").
LAB's own harness and podman sandbox (harvey-labs @ a2b429e, our opt-in patches: harvey_labs_patches.diff):
  plain  LAB's six tools as they are
  tool   the same + contract_tool (server: contract_tool/server.py on 127.0.0.1:18090) and one line about it
Agents (PILOT_MODEL): deepseek = DeepSeek V4.1 Flash on Fireworks (default; set PILOT_EFFORT=low: at default effort it
overthinks or loops past the output limit), gemma-e4b (vLLM, tunnel 18080), gemma-26b (llama.cpp, tunnel 18081, one
episode at a time). All runs: temperature 0.6, an emulated context window (262,144 tokens for DeepSeek, like the
recorded runs), a 32k output cap, a looping response redrawn twice at most, usage logs (runs/pilot/<run>.usage.jsonl).
Judge: Kimi K3 on Fireworks, one pass per criterion, LAB's rubric prompt; `pandoc` on PATH for the scorer.
TASKS = the first 6 (round 1, the 2 run first used the default effort); EXTRA = the next ones of the 7 smallest
held-out paper-review / first-turn-redline tasks (round 2). Recorded base / Ivo Sage outputs can be re-judged (judge
without --ours). Results: ../RESULTS.md.
usage: dspy_venv/bin/python lab_pilot.py run [task slug ...] | judge [task slug ...] [--ours] | report
       (resumable: finished runs / scores are skipped)"""
import concurrent.futures as cf, json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = f"{HERE}/harvey-labs"; LOGS = f"{HERE}/runs/pilot"
MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"; JUDGE = "accounts/fireworks/models/kimi-k3"
TASKS = ["contracts/commercial-channel-partnerships/teaming-agreement-counterparty-paper-review",  # recorded base: overflow
         "contracts/healthcare/quality-agreement-first-turn-redline",                              # overflow
         "contracts/real-estate/easement-agreement-first-turn-redline",                            # overflow
         "contracts/commercial-vendor-customer/non-disclosure-agreement-first-turn-redline",      # finished
         "contracts/commercial-vendor-customer/statement-of-work-first-turn-redline",             # finished
         "contracts/corporate-ma/stock-purchase-agreement-first-turn-redline"]                    # finished (Ivo 0.00)
EXTRA = ["contracts/commercial-vendor-customer/non-disclosure-agreement-counterparty-paper-review",  # the next 3 smallest held-out
         "contracts/real-estate/easement-agreement-counterparty-paper-review",                    # paper reviews (42-59k tokens of
         "contracts/healthcare/quality-agreement-counterparty-paper-review",                      # documents), the user: "do another 3"
         "contracts/banking/isda-confirmation-counterparty-paper-review",                         # then "Let's do 6" (73k / 73k)
         "contracts/data-privacy-security/cyber-insurance-policy-counterparty-paper-review"]
REC = {"base": "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace/art/un/train/deepseek-v4-flash/base_model/episodes/eval_base",
       "ivo": "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace/art/un/train/deepseek-v4-flash/run_16x8_v12/episodes/eval_round_005"}
FW = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("FIREWORKS_API_KEY="))


EFFORT = os.environ.get("PILOT_EFFORT")  # reasoning effort (low / none): V4.1 Flash overthinks or loops at the drafting step
# PILOT_MODEL: which agent (the user, 2026-10-06: "also in parallel with the Gemma model on the 4090"). Our own servers are
# reached through LAB's Fireworks adapter (same patches) with FIREWORKS_API_BASE; served under Fireworks-style names.
AGENTS = {"deepseek": {"model": MODEL, "cap": 262144, "out": 32768},
          "gemma-e4b": {"model": "accounts/fireworks/models/gemma-4-e4b", "base": "http://127.0.0.1:18080/v1", "cap": 131072 - 16384, "out": 16384},
          "gemma-26b": {"model": "accounts/fireworks/models/gemma-4-26b", "base": "http://127.0.0.1:18081/v1", "cap": 131072 - 32768, "out": 32768,
                        "parallel": 1}}  # llama.cpp's 131k KV is shared by all requests: one episode at a time
AGENT = os.environ.get("PILOT_MODEL", "deepseek")


def run_id(arm, task):
    tag = ("" if AGENT == "deepseek" else f"-{AGENT}") + (f"-e{EFFORT}" if EFFORT else "")
    return f"pilot-{arm}-{task.split('/')[-1]}" + (tag if arm in ("plain", "tool") else "")


def env(arm):
    e = dict(os.environ, FIREWORKS_API_KEY=FW, LAB_CONTEXT_CAP="262144", LAB_MAX_OUTPUT="32768", LAB_CALL_TIMEOUT="1200", LAB_RESAMPLE_LENGTH="2")
    e["PATH"] = "/root/zadumai_nli_proto/dspy_venv/lib/python3.12/site-packages/pypandoc/files:" + e["PATH"]  # LAB's scorer runs `pandoc`
    e.pop("LAB_CONTRACT_TOOL", None)
    if arm == "tool": e["LAB_CONTRACT_TOOL"] = "http://127.0.0.1:18090"
    return e


def run_one(arm, task):
    rid = run_id(arm, task); A = AGENTS[AGENT]
    if os.path.exists(f"{LAB}/results/{rid}/metrics.json"): return rid, "done before"
    shutil.rmtree(f"{LAB}/results/{rid}", ignore_errors=True)
    e = dict(env(arm), LAB_DEBUG_DIR=f"{LOGS}/length/{rid}", LAB_USAGE_LOG=f"{LOGS}/{rid}.usage.jsonl", LAB_CONTEXT_CAP=str(A["cap"]), LAB_MAX_OUTPUT=str(A["out"]))
    if A.get("base"): e.update(FIREWORKS_API_BASE=A["base"], FIREWORKS_API_KEY="local")  # the agent only; the judge stays on Fireworks
    t0 = time.time()
    with open(f"{LOGS}/{rid}.log", "w") as log:
        p = subprocess.run([f"{LAB}/.venv/bin/python", "-m", "harness.run", "--model", A["model"], "--task", task, "--run-id", rid, "--temperature", "0.6"] + (["--reasoning-effort", EFFORT] if EFFORT else []),
                           cwd=LAB, env=e, stdout=log, stderr=subprocess.STDOUT)
    return rid, f"exit {p.returncode}, {(time.time() - t0) / 60:.0f} min"


def recorded(arm, task):
    """copies a recorded episode (output/ + metrics.json) into LAB's results/ so the same judge can score it."""
    rid = run_id(arm, task); dst = f"{LAB}/results/{rid}"
    if not os.path.exists(dst):
        src = f"{REC[arm]}/{task.replace('/', '__')}/rollout_0"
        os.makedirs(dst); shutil.copytree(f"{src}/output", f"{dst}/output") if os.path.isdir(f"{src}/output") else os.makedirs(f"{dst}/output")
        shutil.copy(f"{src}/metrics.json", dst)
    return rid


def judge_one(arm, task):
    rid = run_id(arm, task)
    if os.path.exists(f"{LAB}/results/{rid}/scores.json"): return rid, "scored before"
    with open(f"{LOGS}/{rid}.judge.log", "w") as log:
        p = subprocess.run([f"{LAB}/.venv/bin/python", "-m", "evaluation.run_eval", "--run-id", rid, "--task", task, "--judges", JUDGE, "--parallel", "2"],
                           cwd=LAB, env=dict(env("plain"), LAB_JUDGE_USAGE=f"{LOGS}/judge_usage_{AGENT}{'-e' + EFFORT if EFFORT else ''}.jsonl"), stdout=log, stderr=subprocess.STDOUT)
    return rid, f"judge exit {p.returncode}"


def report():
    rows = []
    for task in TASKS:
        r = {"task": task.split("/")[-1]}
        for arm in ("base", "ivo", "plain", "tool"):
            d = f"{LAB}/results/{run_id(arm, task)}"
            s = json.load(open(f"{d}/scores.json")) if os.path.exists(f"{d}/scores.json") else None
            m = json.load(open(f"{d}/metrics.json")) if os.path.exists(f"{d}/metrics.json") else {}
            r[arm] = {"crit": round(s["n_passed"] / max(1, s["n_criteria"]), 3) if s else None, "all_pass": s and s["all_pass"],
                      "input_tok": m.get("input_tokens"), "turns": m.get("turn_count"), "overflow": m.get("context_overflow"),
                      "wall_min": round(m.get("wall_clock_seconds", 0) / 60, 1)}
        if any(r[a]["crit"] is not None for a in ("base", "ivo", "plain", "tool")): rows.append(r)
    json.dump(rows, open(f"{HERE}/data/lab_pilot_results.json", "w"), indent=1)
    for r in rows:
        print(r["task"][:48].ljust(48), " | ".join(f"{a} {r[a]['crit']} ({'OVF' if r[a]['overflow'] else 'ok'}, {((r[a]['input_tok'] or 0) / 1e6):.1f}M tok)" for a in ("base", "ivo", "plain", "tool")))
    for a in ("base", "ivo", "plain", "tool"):
        v = [r[a]["crit"] for r in rows if r[a]["crit"] is not None]
        print(f"{a}: mean criterion pass {sum(v) / max(1, len(v)):.3f} over {len(v)} tasks; mean input tokens {sum((r[a]['input_tok'] or 0) for r in rows) / max(1, len(rows)) / 1e6:.1f}M")


if __name__ == "__main__":
    os.makedirs(LOGS, exist_ok=True); what = sys.argv[1]
    if what == "run":  # run [task slug ...]
        jobs = [(arm, t) for t in TASKS + EXTRA if (not sys.argv[2:] and t in TASKS) or t.split("/")[-1] in sys.argv[2:] for arm in ("tool", "plain")]
        with cf.ThreadPoolExecutor(AGENTS[AGENT].get("parallel", 4)) as ex:  # 4 episodes in flight: one call each at a time, inside the shared-account guard
            for rid, msg in ex.map(lambda j: run_one(*j), jobs): print(time.strftime("%H:%M"), rid, msg, flush=True)
    elif what == "judge":  # judge [task slug ...]: only those tasks (the user: "maybe an even smaller test first?")
        slugs = [a for a in sys.argv[2:] if not a.startswith("--")]; ours = "--ours" in sys.argv  # --ours: only our two arms
        tasks = [t for t in TASKS + EXTRA if (not slugs and t in TASKS) or t.split("/")[-1] in slugs]
        if not ours:
            for t in tasks:
                for arm in ("base", "ivo"): recorded(arm, t)
        jobs = [(arm, t) for t in tasks for arm in (("plain", "tool") if ours else ("plain", "tool", "base", "ivo"))]
        with cf.ThreadPoolExecutor(3) as ex:  # 3 runs x 2 criteria in flight = 6 judge calls
            for rid, msg in ex.map(lambda j: judge_one(*j), jobs): print(time.strftime("%H:%M"), rid, msg, flush=True)
    elif what == "report":
        report()
