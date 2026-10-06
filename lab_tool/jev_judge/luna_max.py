"""GPT-6 Luna as LAB's judge at MAX reasoning effort (2026-10-06; the user: "try Luna on max effort and report"). Same
prompt and texts as luna_judge.py (default effort, which failed: 82.8% vs 96.1% on Jev's uncertain criteria), re-run on
a paired subset of its items: every uncertain criterion, 300 of the Kimi-graded ones, 200 random recorded ones.
usage: dspy_venv/bin/python jev_judge/luna_max.py [effort]   (default max) -> data/jev_judge/luna_verdicts_<effort>.jsonl"""
import concurrent.futures as cf, json, os, random, sys, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data/jev_judge"; sys.path.insert(0, HERE)
from luna import ask, spent
EFFORT = sys.argv[1] if len(sys.argv) > 1 else "max"
TEMPLATE = open(f"{HERE}/../harvey-labs/evaluation/prompts/rubric_criterion.txt").read()
OUT = f"{D}/luna_verdicts_{EFFORT}.jsonl"; _LOCK = threading.Lock(); TAG = f"luna_judge_{EFFORT}"


def grade(r):
    cfg = json.load(open(f"{HERE}/../harvey-labs/tasks/{r['task']}/task.json")); c = next(x for x in cfg["criteria"] if x["id"] == r["cid"])
    prompt = TEMPLATE.format(task_description=cfg["title"], agent_output=open(f"{D}/texts/{r['text']}.txt").read(),
                             criterion_title=c["title"], match_criteria=c["match_criteria"])
    try:
        txt = ask(prompt, json_mode=True, effort=EFFORT, tag=TAG); v = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]).get("verdict", "").lower()
    except Exception as e:
        v = f"error: {type(e).__name__}"
    row = dict(r, luna_default=r["luna"], luna=v)
    with _LOCK, open(OUT, "a") as f: f.write(json.dumps(row) + "\n")


def kappa(a, b):
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)); return (np.mean(a == b) - pe) / max(1e-9, 1 - pe)


def report():
    R = [r for r in map(json.loads, open(OUT)) if r["luna"] in ("pass", "fail") and r["luna_default"] in ("pass", "fail")]
    out = {"effort": EFFORT, "cost_usd": round(spent(TAG), 2)}
    K = [r for r in R if r["why"] == "kimi"]
    if K:
        k = np.array([r["verdicts"][0] == "pass" for r in K])
        for name in ("luna_default", "luna"):
            l = np.array([r[name] == "pass" for r in K])
            out[f"vs_kimi_{name}"] = {"n": len(K), "agreement": round(float(np.mean(l == k)), 4), "kappa": round(float(kappa(l, k)), 3), "pass_rate": round(float(l.mean()), 3)}
        out["vs_kimi_kimi_pass_rate"] = round(float(k.mean()), 3)
    for why in ("uncertain", "random"):
        S = [r for r in R if r["why"] == why and len(r["verdicts"]) == 3 and r["verdicts"][1] == r["verdicts"][2]]
        if not S: continue
        y = np.array([r["verdicts"][1] == "pass" for r in S]); one = np.array([r["verdicts"][0] == "pass" for r in S])
        d = {"n": len(S), "one_recorded_pass": round(float(np.mean(one == y)), 4)}
        for name in ("luna_default", "luna"):
            l = np.array([r[name] == "pass" for r in S]); d[name] = round(float(np.mean(l == y)), 4); d[name + "_kappa"] = round(float(kappa(l, y)), 3)
        out[f"recorded_{why}"] = d
    out["errors"] = sum(1 for r in map(json.loads, open(OUT)) if r["luna"].startswith("error"))
    print(json.dumps(out, indent=1)); json.dump(out, open(f"{D}/luna_judge_eval_{EFFORT}.json", "w"), indent=1)


if __name__ == "__main__":
    base = [r for r in map(json.loads, open(f"{D}/luna_verdicts.jsonl")) if r["luna"] in ("pass", "fail")]
    rng = random.Random(0)
    pick = [r for r in base if r["why"] == "uncertain"]
    pick += rng.sample([r for r in base if r["why"] == "kimi"], 300) + rng.sample([r for r in base if r["why"] == "random"], 200)
    done = {(r["text"], r["task"], r["cid"], r["why"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    todo = [r for r in pick if (r["text"], r["task"], r["cid"], r["why"]) not in done]
    print(len(todo), "criteria at effort", EFFORT, flush=True)
    with cf.ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(grade, todo), 1):
            if i % 100 == 0: print(i, f"${spent(TAG):.2f}", flush=True)
    report()
