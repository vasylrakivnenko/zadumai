"""Is GPT-6 Luna (Azure, ~30x cheaper than Kimi K3) good enough as LAB's LLM judge? (2026-10-06; the user: "check Luna
agreement with Kimi so if Luna is validated as a judge we would switch to Luna"). Luna grades criteria with LAB's own
judge prompt (evaluation/prompts/rubric_criterion.txt) on the exact text LAB's judge reads:
  - every non-empty criterion Kimi K3 graded in our pilot runs (Luna vs Kimi, same deliverables)
  - Ivo Sage's recorded criteria on the test tasks: those inside Jev's uncertain band (where an LLM is needed) and a
    random 600 more (Luna vs the recorded judge passes; fair label = passes 2 and 3 where they agree, vs pass 1)
usage: dspy_venv/bin/python jev_judge/luna_judge.py run | report"""
import concurrent.futures as cf, json, os, random, re, sys, threading, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data/jev_judge"; sys.path.insert(0, HERE)
from luna import ask, spent
TEMPLATE = open(f"{HERE}/../harvey-labs/evaluation/prompts/rubric_criterion.txt").read()
OUT = f"{D}/luna_verdicts.jsonl"; _LOCK = threading.Lock()


def sample():
    import evaluate as E
    from sklearn.linear_model import LogisticRegression
    items = [json.loads(l) for l in open(f"{D}/items.jsonl")]
    rows = E.load(); cal = [r for r in rows if r["src"] == "rec" and zlib.crc32(r["task"].encode()) % 2 == 0]
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.array([r["x"] for r in cal]), np.array([r["y"] for r in cal]))
    A = {(r["text"], r["task"], r["cid"]): r for r in map(json.loads, open(f"{D}/jev_answers.jsonl"))}
    pick, rng = [], random.Random(0)
    test = [it for it in items if not it["empty"] and it["src"] == "rec" and zlib.crc32(it["task"].encode()) % 2 == 1]
    for it in test:
        a = A.get((it["text"], it["task"], it["cid"]))
        if a is None: continue
        p = clf.predict_proba(np.array([E.feats(a)]))[0, 1]
        it["jev_p"] = float(p); it["why"] = "uncertain" if 0.10 < p < 0.94 else "random"
    unc = [it for it in test if it.get("why") == "uncertain"]; rnd = [it for it in test if it.get("why") == "random"]
    pick += unc + rng.sample(rnd, min(600, len(rnd)))
    pick += [dict(it, why="kimi") for it in items if not it["empty"] and it["src"] == "kimi"]
    return pick


def grade(it):
    cfg = json.load(open(f"{HERE}/../harvey-labs/tasks/{it['task']}/task.json"))
    prompt = TEMPLATE.format(task_description=cfg["title"], agent_output=open(f"{D}/texts/{it['text']}.txt").read(),
                             criterion_title=it["title"], match_criteria=it["match"])
    try:
        txt = ask(prompt, json_mode=True, tag="luna_judge")
        v = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]).get("verdict", "").lower()
    except Exception as e:
        v = f"error: {type(e).__name__}"
    row = {k: it[k] for k in ("src", "episode", "task", "cid", "text", "verdicts", "why")}; row["luna"] = v; row["jev_p"] = it.get("jev_p")
    with _LOCK, open(OUT, "a") as f: f.write(json.dumps(row) + "\n")
    return v


def report():
    R = [json.loads(l) for l in open(OUT)]; R = [r for r in R if r["luna"] in ("pass", "fail")]
    kappa = lambda a, b: (np.mean(a == b) - (np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))) / max(1e-9, 1 - (np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))))
    out = {}
    K = [r for r in R if r["why"] == "kimi"]
    if K:
        l = np.array([r["luna"] == "pass" for r in K]); k = np.array([r["verdicts"][0] == "pass" for r in K])
        out["luna_vs_kimi"] = {"n": len(K), "agreement": round(float(np.mean(l == k)), 4), "kappa": round(float(kappa(l, k)), 3),
                               "luna_pass_rate": round(float(l.mean()), 3), "kimi_pass_rate": round(float(k.mean()), 3)}
    for why in ("random", "uncertain"):
        S = [r for r in R if r["why"] == why and len(r["verdicts"]) == 3 and r["verdicts"][1] == r["verdicts"][2]]
        if not S: continue
        y = np.array([r["verdicts"][1] == "pass" for r in S]); l = np.array([r["luna"] == "pass" for r in S]); one = np.array([r["verdicts"][0] == "pass" for r in S])
        out[f"recorded_{why}"] = {"n": len(S), "luna_agreement": round(float(np.mean(l == y)), 4), "luna_kappa": round(float(kappa(l, y)), 3),
                                  "one_recorded_pass_agreement": round(float(np.mean(one == y)), 4), "one_recorded_pass_kappa": round(float(kappa(one, y)), 3)}
    out["luna_cost_usd"] = round(spent("luna_judge"), 2); out["errors"] = sum(1 for l in open(OUT) if json.loads(l)["luna"].startswith("error"))
    print(json.dumps(out, indent=1)); json.dump(out, open(f"{D}/luna_judge_eval.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        done = {(r["text"], r["task"], r["cid"], r["why"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
        todo = [it for it in sample() if (it["text"], it["task"], it["cid"], it["why"]) not in done]
        print(len(todo), "criteria to grade", flush=True)
        with cf.ThreadPoolExecutor(32) as ex:
            for i, v in enumerate(ex.map(grade, todo), 1):
                if i % 100 == 0: print(i, f"${spent('luna_judge'):.2f}", flush=True)
    report()
