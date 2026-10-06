"""Scores the held-out test runs of any caller label (2026-10-05): part A balanced accuracy (native calls, and decisions
counting JSON-text calls) and part B rubric coverage / floor / lift (bge-small cosine 0.65, as final_B.py).
usage: dspy_venv/bin/python muse/score_muse.py LABEL [LABEL ...]"""
import glob, json, os, random, re, sys, zlib
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import optimize_dspy_b as B
rowsB = [json.loads(l) for l in open(f"{HERE}/data/replay_B.jsonl")]
test = [r for r in rowsB if zlib.crc32(r["task"].encode()) % 10 >= 7][:40]; by = {r["task"]: r for r in test}
out = {}
for label in sys.argv[1:]:
    for tag in ("test_plain", "test_gepa_native"):
        p = f"{HERE}/data/out_A_{label}_guided_{tag}.jsonl"
        if not os.path.exists(p): continue
        rows = list({json.loads(l)["key"]: json.loads(l) for l in open(p)}.values()); ok = [r for r in rows if "error" not in r]
        native = lambda r: any(c["name"] == "contract_tool" for c in r["calls"])
        decided = lambda r: native(r) or bool(re.search(r'"(tool_name|name)"\s*:\s*"contract_tool"', r.get("text") or ""))
        res = {}
        for k, f in (("native", native), ("decision", decided)):
            rc = np.mean([f(r) for r in ok if r["label"] == "call"]); rn = np.mean([not f(r) for r in ok if r["label"] == "no_call"])
            res[k] = (round(float(rc), 3), round(float(rn), 3), round(float((rc + rn) / 2), 3))
        out[f"A {label} {tag}"] = res; print(f"A {label:14s} {tag:17s} n {len(ok)} err {len(rows) - len(ok)} | native: uses {res['native'][0]:.1%} holds off {res['native'][1]:.1%} balanced {res['native'][2]:.1%} | decisions balanced {res['decision'][2]:.1%}")
    for tag in ("test_plain", "test_gepa"):
        p = f"{HERE}/data/out_B_{label}_guided_{tag}.jsonl"
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
        rng = random.Random(0); tasks = [r["task"] for r in test]; q = {t: P.get(t) or [] for t in tasks}
        own = np.mean([B.coverage(by[t], q[t])[0] for t in tasks]); shuf = np.mean([B.coverage(by[t], q[rng.choice([u for u in tasks if u != t])])[0] for t in tasks])
        n = sorted(len(q[t]) for t in tasks)
        out[f"B {label} {tag}"] = (round(float(own), 4), round(float(shuf), 4))
        print(f"B {label:14s} {tag:17s} coverage {own:.1%} floor {shuf:.1%} lift {own - shuf:+.1%} | queries median {n[len(n) // 2]} | plans {len(P)}/40")
json.dump(out, open(f"{HERE}/muse/scores_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
