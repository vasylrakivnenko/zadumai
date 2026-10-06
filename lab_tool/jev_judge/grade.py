"""Jev judge for new LAB runs (2026-10-06): grade a run's deliverables with Jev + an LLM for the criteria Jev is unsure
of (the setup measured in evaluate.py: 99.15% agreement vs 99.32% for one LLM pass, at a fraction of the cost).
  texts   (harvey venv)  the text LAB's judge reads per criterion, for each run -> data/jev_judge/new_items.jsonl
  grade   (dspy venv)    Jev on every check (cached decompositions; a task without one is decomposed by Luna first),
                         the combiner fitted on all of Ivo's recorded criteria, band p <= 0.10 fail / >= 0.94 pass,
                         the rest graded by Kimi K3 (Fireworks) with LAB's judge prompt -> results/<run>/scores_jev.json
usage: harvey-labs/.venv/bin/python jev_judge/grade.py texts RUN_ID ...
       dspy_venv/bin/python jev_judge/grade.py grade RUN_ID ..."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data/jev_judge"; LAB = f"{HERE}/../harvey-labs"
NEW = f"{D}/new_items.jsonl"; LO, HI = 0.10, 0.94
JUDGE = "accounts/fireworks/models/kimi-k3"; KIMI = (3.00, 0.30, 15.00)


def texts(runs):
    sys.path.insert(0, HERE)
    import contextlib, io
    import texts as T
    # a rerun under the same run id must be re-read: drop that run's old rows first (stale texts graded the old output)
    old = [l for l in open(NEW) if json.loads(l)["episode"] not in runs] if os.path.exists(NEW) else []
    open(NEW, "w").writelines(old); have = set()
    with open(NEW, "a") as f:
        for rid in runs:
            cfg = json.load(open(f"{LAB}/results/{rid}/config.json")); task = cfg.get("task") or cfg.get("task_name")
            with contextlib.redirect_stdout(io.StringIO()): criteria, X = T.texts_for(task, f"{LAB}/results/{rid}/output")
            empty = not T.has_files(f"{LAB}/results/{rid}/output")
            for c in criteria:
                if (rid, c["id"]) in have: continue
                f.write(json.dumps({"src": "new", "episode": rid, "task": task, "cid": c["id"], "title": c["title"], "match": c["match_criteria"],
                                    "text": T.store(X[c["id"]]), "verdicts": [], "empty": empty}) + "\n")
            print(rid, task, len(criteria), "criteria", "(no deliverables)" if empty else "", flush=True)


def grade(runs):
    import concurrent.futures as cf, threading
    import numpy as np, openai
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, HERE)
    import check as C, evaluate as E
    import decompose
    items = [r for r in map(json.loads, open(NEW)) if r["episode"] in runs]
    for task in sorted({r["task"] for r in items}):  # decompositions: cached; Luna for a task not seen before
        if not os.path.exists(f"{D}/decomp/{task.replace('/', '__')}.json"): print("decompose", decompose.one(task))
    # the combiner, on all of Ivo's recorded criteria
    rec = [r for r in E.load() if r["src"] == "rec"]
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.array([r["x"] for r in rec]), np.array([r["y"] for r in rec]))
    # Jev answers (cached in jev_answers.jsonl, same key as the evaluation)
    have = {(r["text"], r["task"], r["cid"]) for r in map(json.loads, open(C.OUT))}
    groups = {}
    for it in items:
        if not it["empty"] and (it["text"], it["task"], it["cid"]) not in have: groups.setdefault((it["text"], it["task"]), {})[it["cid"]] = it
    decomps = {t: {c["id"]: c for c in json.load(open(f"{D}/decomp/{t.replace('/', '__')}.json"))["criteria"]} for t in {r["task"] for r in items}}
    jobs = [(t, task, list(cs.values()), decomps[task]) for (t, task), cs in groups.items()]
    if jobs:
        C.model()
        with cf.ThreadPoolExecutor(8) as ex: list(ex.map(C.run_group, jobs))
    A = {(r["text"], r["task"], r["cid"]): r for r in map(json.loads, open(C.OUT))}
    # decide, escalate
    client = openai.OpenAI(api_key=next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("FIREWORKS_API_KEY=")),
                           base_url="https://api.fireworks.ai/inference/v1", timeout=900, max_retries=4)
    template = open(f"{LAB}/evaluation/prompts/rubric_criterion.txt").read(); usage = []; lock = threading.Lock()
    def llm(it):
        cfg = json.load(open(f"{LAB}/tasks/{it['task']}/task.json"))
        prompt = template.format(task_description=cfg["title"], agent_output=open(f"{D}/texts/{it['text']}.txt").read(), criterion_title=it["title"], match_criteria=it["match"])
        for attempt in range(3):
            try:
                r = client.chat.completions.create(model=JUDGE, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=16384, response_format={"type": "json_object"})
                u = r.usage; cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                with lock:
                    usage.append(((u.prompt_tokens - cached) * KIMI[0] + cached * KIMI[1] + u.completion_tokens * KIMI[2]) / 1e6)
                    with open(f"{HERE}/../runs/pilot/judge_usage_grade.jsonl", "a") as f:  # counted by analysis/costs.py
                        f.write(json.dumps({"model": JUDGE, "prompt": u.prompt_tokens, "cached": cached, "completion": u.completion_tokens}) + "\n")
                txt = r.choices[0].message.content or ""; return json.loads(txt[txt.index("{"):txt.rindex("}") + 1]).get("verdict", "fail").lower() == "pass"
            except Exception:
                continue
        return False
    out = {}
    for it in items:
        if it["empty"]: out[(it["episode"], it["cid"])] = (False, "empty", None); continue
        p = float(clf.predict_proba(np.array([E.feats(A[(it["text"], it["task"], it["cid"])])]))[0, 1])
        out[(it["episode"], it["cid"])] = (p >= HI, "jev", p) if (p <= LO or p >= HI) else (None, "llm", p)
    esc = [it for it in items if out[(it["episode"], it["cid"])][1] == "llm"]
    with cf.ThreadPoolExecutor(6) as ex:  # Fireworks is shared with the live router: 6 in flight
        for it, v in zip(esc, ex.map(llm, esc)): out[(it["episode"], it["cid"])] = (v, "llm", out[(it["episode"], it["cid"])][2])
    for rid in runs:
        rows = [(it, out[(it["episode"], it["cid"])]) for it in items if it["episode"] == rid]
        if not rows: continue
        res = {"run_id": rid, "task": rows[0][0]["task"], "n_criteria": len(rows), "n_passed": sum(bool(v) for _, (v, _, _) in rows),
               "decided_by_jev": sum(s == "jev" for _, (_, s, _) in rows), "llm": sum(s == "llm" for _, (_, s, _) in rows),
               "criteria_results": [{"id": it["cid"], "title": it["title"], "verdict": "pass" if v else "fail", "by": s, "p_jev": p} for it, (v, s, p) in rows]}
        res["criterion_pass_rate"] = round(res["n_passed"] / res["n_criteria"], 4)
        json.dump(res, open(f"{LAB}/results/{rid}/scores_jev.json", "w"), indent=1)
        print(f"{rid}: {res['n_passed']}/{res['n_criteria']} = {res['criterion_pass_rate']:.3f} (Jev {res['decided_by_jev']}, LLM {res['llm']})", flush=True)
    C.log_jev_usage("grade")
    print(f"LLM judge (Kimi K3) cost: ${sum(usage):.2f} for {len(esc)} criteria; Jev tokens this run: {C.JEV.input_tokens:,} (${C.JEV.input_tokens * 0.042 / 1e6:.2f})")


if __name__ == "__main__":
    {"texts": texts, "grade": grade}[sys.argv[1]](sys.argv[2:])
