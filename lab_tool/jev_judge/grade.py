"""Jev judge for new LAB runs (2026-10-06): grade a run's deliverables with Jev + an LLM for the criteria Jev is unsure
of (the setup measured in evaluate.py: 99.15% agreement vs 99.32% for one LLM pass, at a fraction of the cost).
  texts   (harvey venv)  the text LAB's judge reads per criterion, for each run -> data/jev_judge/new_items.jsonl
  grade   (dspy venv)    Jev on every check (cached decompositions; a task without one is decomposed by Luna first),
                         the combiner fitted on all of Ivo's recorded criteria, band p <= 0.10 fail / >= 0.94 pass,
                         the rest graded by Kimi K2.6 (Azure; Kimi K3 on Fireworks until 2026-10-06, kept as scores_jev_k3.json) with LAB's
                         judge prompt -> results/<run>/scores_jev.json
usage: harvey-labs/.venv/bin/python jev_judge/grade.py texts RUN_ID ...
       dspy_venv/bin/python jev_judge/grade.py grade RUN_ID ..."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data/jev_judge"; LAB = f"{HERE}/../harvey-labs"
NEW = f"{D}/new_items.jsonl"; LO, HI = 0.10, 0.94
JUDGE = "Kimi-K2.6"; PRICE = (0.95, 0.16, 4.00)  # Azure, per 1M: input, cached, output
MAX_OUT = 32768  # asked for, but Azure clamps K2.6 at 4,096 output tokens, reasoning included: a call that runs out has no verdict and is retried
# so LAB's judge prompt + this line (2026-10-06; on 30 hard validation items: 12% of calls at the cap instead of 54%,
# 25/29 vs 22/29 agreement with the recorded judges); scores say "Kimi-K2.6 (brief)"
BRIEF = "\n\nKeep your reasoning brief: decide from the decisive passage, in a few sentences, then give the JSON."
JUDGE_TAG = "Kimi-K2.6 (brief)"


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
        if (it["text"], it["task"], it["cid"]) not in have: groups.setdefault((it["text"], it["task"]), {})[it["cid"]] = it
    decomps = {t: {c["id"]: c for c in json.load(open(f"{D}/decomp/{t.replace('/', '__')}.json"))["criteria"]} for t in {r["task"] for r in items}}
    jobs = [(t, task, list(cs.values()), decomps[task]) for (t, task), cs in groups.items()]
    if jobs:
        C.model()
        with cf.ThreadPoolExecutor(8) as ex: list(ex.map(C.run_group, jobs))
    A = {(r["text"], r["task"], r["cid"]): r for r in map(json.loads, open(C.OUT))}
    # decide, escalate
    # the LLM for criteria Jev is unsure of: Kimi K2.6 Thinking on Azure (the user, 2026-10-06: "Forget about Fireworks", so no
    # Kimi K3); LAB's own judge prompt; usage per call with run id (analysis/costs.py)
    akey = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("AZURE_API_KEY="))
    client = openai.OpenAI(base_url="https://ai-vasyl-0670.services.ai.azure.com/openai/v1", api_key=akey, timeout=900, max_retries=0)
    template = open(f"{LAB}/evaluation/prompts/rubric_criterion.txt").read(); usage = []; lock = threading.Lock()
    def llm(it):
        cfg = json.load(open(f"{LAB}/tasks/{it['task']}/task.json"))
        prompt = template.format(task_description=cfg["title"], agent_output=open(f"{D}/texts/{it['text']}.txt").read(), criterion_title=it["title"], match_criteria=it["match"]) + BRIEF
        for attempt in range(6):
            try:
                r = client.responses.create(model=JUDGE, input=prompt, text={"format": {"type": "json_object"}}, max_output_tokens=MAX_OUT)
                u = r.usage; cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
                with lock:
                    usage.append(((u.input_tokens - cached) * PRICE[0] + cached * PRICE[1] + u.output_tokens * PRICE[2]) / 1e6)
                    with open(f"{HERE}/../runs/judge_usage_grade_k26.jsonl", "a") as f:  # counted by analysis/costs.py
                        f.write(json.dumps({"model": JUDGE, "brief": True, "run": it["episode"], "input": u.input_tokens, "cached": cached, "output": u.output_tokens, "t": time.time()}) + "\n")
                txt = r.output_text or ""; return json.loads(txt[txt.index("{"):txt.rindex("}") + 1]).get("verdict", "fail").lower() == "pass"
            except Exception:
                time.sleep(20 * (attempt + 1)); continue  # Azure's token-per-minute cap answers 429: back off
        return None  # no silent fail: the caller falls back to Jev's decision and counts it
    out = {}
    for it in items:  # v4.1 (review): runs without deliverables are judged on what LAB's judge reads, not hard-coded to fail
        p = float(clf.predict_proba(np.array([E.feats(A[(it["text"], it["task"], it["cid"])])]))[0, 1])
        out[(it["episode"], it["cid"])] = (p >= HI, "jev", p) if (p <= LO or p >= HI) else (None, "llm", p)
    esc = [it for it in items if out[(it["episode"], it["cid"])][1] == "llm"]
    with cf.ThreadPoolExecutor(6) as ex:  # Azure's token-per-minute cap: few in flight
        for it, v in zip(esc, ex.map(llm, esc)):
            p = out[(it["episode"], it["cid"])][2]
            out[(it["episode"], it["cid"])] = (v, "llm", p) if v is not None else (p >= 0.5, "jev-fallback (LLM failed)", p)
    for rid in runs:
        rows = [(it, out[(it["episode"], it["cid"])]) for it in items if it["episode"] == rid]
        if not rows: continue
        res = {"run_id": rid, "task": rows[0][0]["task"], "n_criteria": len(rows), "n_passed": sum(bool(v) for _, (v, _, _) in rows),
               "decided_by_jev": sum(s == "jev" for _, (_, s, _) in rows), "llm": sum(s == "llm" for _, (_, s, _) in rows),
               "llm_failed": sum(s.startswith("jev-fallback") for _, (_, s, _) in rows),
               "criteria_results": [{"id": it["cid"], "title": it["title"], "verdict": "pass" if v else "fail", "by": s, "p_jev": p} for it, (v, s, p) in rows]}
        res["criterion_pass_rate"] = round(res["n_passed"] / res["n_criteria"], 4)
        res["escalation_judge"] = JUDGE_TAG; f = f"{LAB}/results/{rid}/scores_jev.json"
        if os.path.exists(f):  # keep earlier graders' versions: Kimi K3 (Fireworks), K2.6 with LAB's plain prompt
            old = json.load(open(f)).get("escalation_judge")
            if old is None: os.replace(f, f"{LAB}/results/{rid}/scores_jev_k3.json")
            elif old == "Kimi-K2.6": os.replace(f, f"{LAB}/results/{rid}/scores_jev_k26plain.json")
        json.dump(res, open(f, "w"), indent=1)
        print(f"{rid}: {res['n_passed']}/{res['n_criteria']} = {res['criterion_pass_rate']:.3f} (Jev {res['decided_by_jev']}, LLM {res['llm']})", flush=True)
    C.log_jev_usage("grade")
    print(f"LLM judge ({JUDGE}) cost: ${sum(usage):.2f} for {len(esc)} criteria; Jev tokens this run: {C.JEV.input_tokens:,} (${C.JEV.input_tokens * 0.042 / 1e6:.2f})")


if __name__ == "__main__":
    {"texts": texts, "grade": grade}[sys.argv[1]](sys.argv[2:])
