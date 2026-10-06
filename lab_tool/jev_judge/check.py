"""Jev judge, step 3 (2026-10-06): Jev's probabilities for every check of every criterion, plus the whole criterion as one
more statement. The state is the text LAB's judge reads; Jev takes ~32k tokens, so a text over 100k characters is cut
to the criterion's most relevant passages (BM25 + bge-small, reciprocal-rank fusion, up to 60k characters, in document
order). Texts that fit: one state per (text, task), all its questions batched (40 per request).
usage: dspy_venv/bin/python jev_judge/check.py -> data/jev_judge/jev_answers.jsonl"""
import concurrent.futures as cf, json, os, re, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data/jev_judge"; sys.path.insert(0, f"{HERE}/..")
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed/dev")
from router.systemone import SystemOne
from contract_tool.tool import BM25, model, toks
KEY = next(l.split("=", 1)[1].strip() for l in open("/root/projects/zadumai/.env") if l.startswith("JEV_API="))
JEV = SystemOne("https://api.typesafe.ai/v1/systemone", "jev-latest", api_key=KEY, timeout=180)
FIT, BUDGET, PER_REQ = 100_000, 60_000, 40
OUT = f"{D}/jev_answers.jsonl"; _LOCK = threading.Lock()
TRUE = "the deliverable clearly does or says this"
FALSE = "the deliverable does not do or say this, says the opposite, or does not address it"


def q_check(s):
    return {"type": "noul", "instructions": {"statement": s, "question": "Is this statement true of the deliverable?"}, "criteria": {"true": TRUE, "false": FALSE}}


def q_whole(c):
    return {"type": "noul", "instructions": {"criterion": f"{c['title']}: {c['match']}",
            "question": "Does the deliverable meet this grading criterion's PASS condition?"},
            "criteria": {"true": "the deliverable meets the PASS condition", "false": "it meets the FAIL condition, or does not address the criterion"}}


def passages(text):
    out, cur = [], ""
    for b in re.split(r"\n\s*\n", text):
        cur = (cur + "\n\n" + b) if cur else b
        if len(cur) >= 400: out.append(cur); cur = ""
    if cur: out.append(cur)
    return out


_EMB = {}


def retrieve(text, query):
    """the criterion's most relevant passages, up to BUDGET characters, in document order (file headers kept)."""
    if text not in _EMB:
        P = passages(text); _EMB[text] = (P, BM25([toks(p) for p in P]), model().encode(P, batch_size=64, normalize_embeddings=True))
    P, bm, E = _EMB[text]
    s1 = bm.scores(toks(query)); s2 = E @ model().encode(["Represent this sentence for searching relevant passages: " + query], normalize_embeddings=True)[0]
    rr = np.zeros(len(P))
    for s in (s1, s2):
        o = np.argsort(-s); r = np.empty(len(P)); r[o] = np.arange(len(P)); rr += 1 / (60 + r)
    keep, n = set(i for i, p in enumerate(P) if p.lstrip().startswith("## Agent Output")), 0
    for i in np.argsort(-rr):
        if n + len(P[i]) > BUDGET: continue
        keep.add(int(i)); n += len(P[i])
    return "\n\n".join(P[i] for i in sorted(keep))


def ask(state, questions):
    out = {}
    keys = list(questions)
    for k in range(0, len(keys), PER_REQ):
        a = JEV.ask({"deliverable": state}, {q: questions[q] for q in keys[k:k + PER_REQ]})
        out.update({q: float(a[q]["noul"]) for q in a})
    return out


def run_group(job):
    text_id, task, crits, decomp = job
    text = open(f"{D}/texts/{text_id}.txt").read(); res = []
    def qs(c):
        d = decomp[c["cid"]]; q = {f"{c['cid']}|w": q_whole(c)}
        q.update({f"{c['cid']}|{i}": q_check(ch["s"]) for i, ch in enumerate(d["checks"])}); return q
    if len(text) <= FIT:
        Q = {}
        for c in crits: Q.update(qs(c))
        A = ask(text, Q); mode = "whole"
        res = [(c, A) for c in crits]
    else:
        mode = "retrieved"
        for c in crits:
            d = decomp[c["cid"]]
            query = " ".join([c["title"], c["match"][:400]] + [ch["s"] for ch in d["checks"]])
            res.append((c, ask(retrieve(text, query), qs(c))))
    rows = []
    for c, A in res:
        d = decomp[c["cid"]]
        rows.append({"text": text_id, "task": task, "cid": c["cid"], "mode": mode, "whole": A[f"{c['cid']}|w"],
                     "checks": [A[f"{c['cid']}|{i}"] for i in range(len(d["checks"]))], "pass_if": [bool(ch.get("pass_if", True)) for ch in d["checks"]],
                     "logic": d.get("logic", "all"), "whole_document": bool(d.get("whole_document", False))})
    with _LOCK, open(OUT, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return text_id, task, len(rows), mode


if __name__ == "__main__":
    items = [json.loads(l) for l in open(f"{D}/items.jsonl")]
    done = {(r["text"], r["task"], r["cid"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    groups = {}
    for it in items:
        if it["empty"] or (it["text"], it["task"], it["cid"]) in done: continue
        groups.setdefault((it["text"], it["task"]), {})[it["cid"]] = it
    decomps = {}
    jobs = []
    for (t, task), cs in groups.items():
        if task not in decomps:
            decomps[task] = {c["id"]: c for c in json.load(open(f"{D}/decomp/{task.replace('/', '__')}.json"))["criteria"]}
        jobs.append((t, task, list(cs.values()), decomps[task]))
    print(len(jobs), "groups,", sum(len(j[2]) for j in jobs), "criteria", flush=True)
    model(); t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for i, (t, task, n, mode) in enumerate(ex.map(run_group, jobs), 1):
            print(f"{i}/{len(jobs)} {task.split('/')[-1][:45]} {n} criteria ({mode}); Jev calls {JEV.calls}, tokens in {JEV.input_tokens:,}, {time.time() - t0:.0f}s", flush=True)
