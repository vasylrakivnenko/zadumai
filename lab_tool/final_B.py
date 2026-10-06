"""Part B test scoring (2026-10-05), after optimize_dspy_b.py's own final step crashed on a truncated answer: the GEPA
instruction (data/gepa_B_instruction.txt, candidate 5, best on val from round 8 on) vs the starting one, on the 40 test
tasks; answers up to 3,000 tokens; a failed / unparseable plan counts as no queries. Also the deployment format
(run_replay outputs, TAG=test_plain / test_gepa) scored with the same coverage function.
usage: dspy_venv/bin/python final_B.py"""
import concurrent.futures as cf, json, os, random, sys, zlib
import numpy as np
import dspy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import optimize_dspy_b as B
lm = dspy.LM("openai/gemma-4-e4b", api_base="http://127.0.0.1:18080/v1", api_key="local", temperature=0.0, max_tokens=3000)


def plans(prog, rows):
    def run(r):
        try:
            with dspy.context(lm=lm): return r["task"], B.queries(prog(task=f"{r['title']}\n{r['instructions']}", documents=", ".join(r["documents"])))
        except Exception:
            return r["task"], None
    with cf.ThreadPoolExecutor(24) as ex: return dict(ex.map(run, rows))


def score(P, rows):
    rng = random.Random(0); tasks = [r["task"] for r in rows]; by = {r["task"]: r for r in rows}
    q = {t: P.get(t) or [] for t in tasks}
    own = [B.coverage(by[t], q[t])[0] for t in tasks]
    shuf = [B.coverage(by[t], q[rng.choice([u for u in tasks if u != t])])[0] for t in tasks]
    n = sorted(len(q[t]) for t in tasks)
    return {"coverage": round(float(np.mean(own)), 4), "floor": round(float(np.mean(shuf)), 4), "lift": round(float(np.mean(own) - np.mean(shuf)), 4),
            "queries_median": n[len(n) // 2], "failed_plans": sum(P.get(t) is None for t in tasks), "n": len(tasks)}


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(f"{HERE}/data/replay_B.jsonl")]
    test = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 >= 7][:40]
    out = {}
    out["dspy_start"] = score(plans(B.Planner(), test), test); print("DSPy format, start:", out["dspy_start"], flush=True)
    g = B.Planner(); g.plan.signature = g.plan.signature.with_instructions(open(f"{HERE}/data/gepa_B_instruction.txt").read())
    out["dspy_gepa"] = score(plans(g, test), test); print("DSPy format, GEPA:", out["dspy_gepa"], flush=True)
    for tag in ("test_plain", "test_gepa"):
        p = f"{HERE}/data/out_B_gemma-4-e4b_guided_{tag}.jsonl"
        if not os.path.exists(p): continue
        P = {}
        for l in open(p):
            r = json.loads(l)
            if "error" in r: continue
            qs = []
            for c in r["calls"]:
                if c["name"] != "contract_tool": continue
                try: a = json.loads(c["arguments"])
                except ValueError: continue
                qs.append(" ".join(str(a.get(k) or "") for k in ("action", "query", "document")).strip())
            P[r["task"]] = qs
        out[f"deploy_{tag}"] = score(P, test); print(f"deployment format, {tag}:", out[f"deploy_{tag}"], flush=True)
    json.dump(out, open(f"{HERE}/data/dspy_B_results.json", "w"), indent=1)
