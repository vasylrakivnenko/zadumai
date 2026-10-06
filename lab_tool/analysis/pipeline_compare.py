"""Pipeline vs agent loop on our 5-task working set (2026-10-06). Every run of ours graded by the same grader
(jev_judge/grade.py: Jev + Kimi K3, scores_jev.json); base / Ivo Sage = Ivo's recorded scores (original judge).
Also time and cost per run (Luna from the usage log by run tag; Gemma: local GPU time).
usage (from lab_tool/): dspy_venv/bin/python analysis/pipeline_compare.py -> prints, data/pipeline_compare.json"""
import gzip, json, os, sys
sys.path.insert(0, "jev_judge")
from luna import cost as luna_cost
S = "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace"
R = "harvey-labs/results"
ep = json.load(gzip.open(f"{S}/space/data/episodes.json.gz"))["tasks"]
cpr = lambda e: sum(x for x in e["judge_cpr"] if x is not None) / max(1, len([x for x in e["judge_cpr"] if x is not None]))
T = open("runs/pilot/set5_tasks.txt").read().split()
ARMS = [("DeepSeek agent, LAB tools", "pilot-plain-{t}-elow"), ("DeepSeek agent + tool", "pilot-tool-{t}-elow"),
        ("Luna agent, LAB tools", "pilot-plain-{t}-luna-emax"), ("Luna agent + tool", "pilot-tool-{t}-luna-emax"),
        ("Gemma 26B agent, LAB tools", "pilot-plain-{t}-gemma-26b"), ("Gemma 26B agent + tool", "pilot-tool-{t}-gemma-26b"),
        ("Luna PIPELINE", "pipe-luna-{t}"), ("Gemma 26B PIPELINE", "pipe-gemma26-{t}")]


def score(rid):
    p = f"{R}/{rid}/scores_jev.json"
    return json.load(open(p))["criterion_pass_rate"] if os.path.exists(p) else None


if __name__ == "__main__":
    usage = [json.loads(l) for l in open("runs/luna_usage.jsonl")]
    rows, cols = [], {"base (recorded)": [], "Ivo Sage (recorded)": []} | {a: [] for a, _ in ARMS}
    for t in T:
        full = next(k for k in ep if k.endswith("/" + t)); r = {"task": t, "base (recorded)": cpr(ep[full]["base"]), "Ivo Sage (recorded)": cpr(ep[full]["sage"])}
        for a, pat in ARMS: r[a] = score(pat.format(t=t))
        rows.append(r)
        for k in cols: cols[k].append(r[k])
    print("task".ljust(46) + "".join(k[:14].rjust(15) for k in cols))
    for r in rows: print(r["task"][:45].ljust(46) + "".join(("-" if r[k] is None else f"{r[k]:.3f}").rjust(15) for k in cols))
    means = {k: (sum(v for v in vs if v is not None) / len([v for v in vs if v is not None]) if any(v is not None for v in vs) else None) for k, vs in cols.items()}
    print("mean".ljust(46) + "".join(("-" if m is None else f"{m:.3f}").rjust(15) for m in means.values()))
    print("\nper pipeline run: calls, failed calls, minutes, Luna cost")
    for t in T:
        for m in ("luna", "gemma26"):
            rid = f"pipe-{m}-{t}"
            if not os.path.exists(f"{R}/{rid}/metrics.json"): continue
            x = json.load(open(f"{R}/{rid}/metrics.json")); c = sum(luna_cost(u) for u in usage if u["tag"] == f"pipe:{rid}")
            print(f"  {rid[:62]:62s} {x['llm_calls']:4d} calls, {x['llm_errors']} failed, {x['wall_clock_seconds'] / 60:5.1f} min" + (f", ${c:.2f}" if m == "luna" else ""))
    json.dump({"rows": rows, "means": means}, open("data/pipeline_compare.json", "w"), indent=1)
