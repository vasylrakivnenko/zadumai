"""LAB pilot summary (2026-10-06): criterion pass rates per task and arm (Kimi K3 judge, one pass) next to the recorded base
and Ivo Sage scores (original judge, mean of 3 passes), plus exact token costs (usage logs) and judge costs."""
import glob, gzip, json, os
L = "/root/zadumai_nli_proto/lab_tool"; R = f"{L}/harvey-labs/results"; P = f"{L}/runs/pilot"
S = "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace"
ep = json.load(gzip.open(f"{S}/space/data/episodes.json.gz"))["tasks"]
cpr = lambda e: sum(x for x in e["judge_cpr"] if x is not None) / max(1, len([x for x in e["judge_cpr"] if x is not None]))
TASKS = ["non-disclosure-agreement-first-turn-redline", "non-disclosure-agreement-counterparty-paper-review", "easement-agreement-counterparty-paper-review",
         "quality-agreement-counterparty-paper-review", "isda-confirmation-counterparty-paper-review", "cyber-insurance-policy-counterparty-paper-review",
         "easement-agreement-first-turn-redline"]
ARMS = [("DeepSeek LAB tools", "plain", "-elow"), ("DeepSeek + tool", "tool", "-elow"), ("Gemma26B LAB tools", "plain", "-gemma-26b"), ("Gemma26B + tool", "tool", "-gemma-26b")]
DS = (0.30, 0.006, 1.20)
def score(rid):
    f = f"{R}/{rid}/scores.json"
    if not os.path.exists(f): return None
    s = json.load(open(f)); return s["n_passed"] / s["n_criteria"]
rows, cost = [], 0.0
for t in TASKS:
    full = next(k for k in ep if k.endswith("/" + t)); r = {"task": t, "base": round(cpr(ep[full]["base"]), 3), "ivo": round(cpr(ep[full]["sage"]), 3)}
    for name, arm, suf in ARMS:
        rid = f"pilot-{arm}-{t}{suf}"; v = score(rid); r[name] = None if v is None else round(v, 3)
        u = f"{P}/{rid}.usage.jsonl"
        if os.path.exists(u) and "elow" in suf:
            cost += sum(((x["prompt"] - x["cached"]) * DS[0] + x["cached"] * DS[1] + x["completion"] * DS[2]) / 1e6 for x in map(json.loads, open(u)))
    rows.append(r)
cols = ["base", "ivo"] + [a[0] for a in ARMS]
print("task".ljust(52) + "".join(c[:18].rjust(20) for c in cols))
for r in rows: print(r["task"][:50].ljust(52) + "".join(("-" if r[c] is None else f"{r[c]:.3f}").rjust(20) for c in cols))
for c in cols:
    v = [r[c] for r in rows if r[c] is not None]; print(f"mean {c}: {sum(v) / max(1, len(v)):.3f} over {len(v)} tasks")
J = [json.loads(l) for f in glob.glob(f"{P}/judge_usage_deepseek-elow.jsonl") + glob.glob(f"{P}/judge_usage_gemma-26b.jsonl") for l in open(f)]
jc = sum(((u["prompt"] - u["cached"]) * 3 + u["cached"] * 0.3 + u["completion"] * 15) / 1e6 for u in J)
print(f"DeepSeek agent tokens (low effort, these runs): ${cost:.2f}; judge for these runs: {len(J)} calls ${jc:.2f}")
json.dump(rows, open(f"{L}/data/lab_pilot_summary.json", "w"), indent=1)
