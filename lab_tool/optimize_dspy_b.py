"""DSPy prompt optimization, part B (WHAT to ask; 2026-10-05): Gemma 4 E4B plans its contract_tool queries for a LAB
Contracts task (training split only, split BY TASK: crc32 % 10 < 7 optimization, >= 7 test). Metric: rubric coverage =
share of criteria (title + pass condition) with a query at bge-small cosine >= 0.65, minus 0.01 per query beyond 12 (no
flooding). GEPA feedback names a few uncovered rubric points (training tasks only). Test: coverage of the test tasks'
rubrics, own vs shuffled (another task's queries: the floor of generic wording).
usage: dspy_venv/bin/python optimize_dspy_b.py -> data/dspy_B_results.json"""
import json, os, random, sys, zlib
import numpy as np
import dspy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import optimize_dspy as O  # LMs, adapter, the contract_tool definition
from sentence_transformers import SentenceTransformer
EMB = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
TOOL = [t for t in O.TOOLS if t.name == "contract_tool"]
T, FLOOD = 0.65, 12


class PlanQueries(dspy.Signature):
    __doc__ = ("You are a lawyer working as an agent. " + O.R.GUIDE + " Before drafting anything, gather every fact, clause, "
               "position and number you will need from the documents: make all the contract_tool calls you need now, in parallel.")
    task: str = dspy.InputField(desc="the task title and instructions")
    documents: str = dspy.InputField(desc="the matter's source documents in documents/")
    tools: list[dspy.Tool] = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


class Planner(dspy.Module):
    def __init__(self):
        super().__init__(); self.plan = dspy.Predict(PlanQueries)

    def forward(self, task, documents):
        return self.plan(task=task, documents=documents, tools=TOOL)


_CE = {}


def crit_emb(r):
    if r["task"] not in _CE:
        _CE[r["task"]] = EMB.encode([f"{c['title']}. {c['match']}"[:400] for c in r["criteria"]], normalize_embeddings=True)
    return _CE[r["task"]]


def queries(pred):
    out = []
    for c in getattr(getattr(pred, "tool_calls", None), "tool_calls", None) or []:
        if c.name != "contract_tool": continue
        a = c.args or {}
        out.append(" ".join(str(a.get(k) or "") for k in ("action", "query", "document")).strip())
    return [q for q in out if q]


def coverage(r, qs):
    if not qs: return 0.0, np.zeros(len(r["criteria"]), bool)
    best = (crit_emb(r) @ EMB.encode(qs, normalize_embeddings=True).T).max(1)
    hit = best >= T
    return float(hit.mean()), hit


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    r = gold.row; qs = queries(pred); cov, hit = coverage(r, qs)
    score = cov - 0.01 * max(0, len(qs) - FLOOD)
    if pred_name is None and pred_trace is None: return score
    missed = [c["title"] for c, h in zip(r["criteria"], hit) if not h][:6]
    fb = (f"Your {len(qs)} queries covered {hit.sum()} of {len(hit)} points the reviewer checks ({cov:.0%})."
          + (f" Over {FLOOD} queries is penalized." if len(qs) > FLOOD else "")
          + (" Points no query reached, for example: " + "; ".join(missed) + "." if missed else "")
          + " Good queries name the specific clause, position, party, deadline or number the deliverable must address.")
    return dspy.Prediction(score=score, feedback=fb)


def example(r):
    return dspy.Example(task=f"{r['title']}\n{r['instructions']}", documents=", ".join(r["documents"]), row=r).with_inputs("task", "documents")


def test_scores(prog, rows):
    preds = {}
    def run(r):
        return r["task"], queries(prog(task=f"{r['title']}\n{r['instructions']}", documents=", ".join(r["documents"])))
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(O.THREADS) as ex:
        for t, qs in ex.map(run, rows): preds[t] = qs
    rng = random.Random(0); tasks = [r["task"] for r in rows]; by = {r["task"]: r for r in rows}
    own = [coverage(by[t], preds[t])[0] for t in tasks]
    shuf = [coverage(by[t], preds[rng.choice([u for u in tasks if u != t])])[0] for t in tasks]
    n = [len(preds[t]) for t in tasks]
    return {"coverage": float(np.mean(own)), "shuffled": float(np.mean(shuf)), "lift": float(np.mean(own) - np.mean(shuf)),
            "queries_median": sorted(n)[len(n) // 2], "n": len(tasks)}


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(f"{HERE}/data/replay_B.jsonl")]
    opt = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 < 7]; test = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 >= 7][:40]
    rng = random.Random(0); rng.shuffle(opt)
    train, val = [example(r) for r in opt[:50]], [example(r) for r in opt[50:100]]
    print(f"B: train {len(train)}, val {len(val)}, test {len(test)} tasks", flush=True)
    out = {"start": test_scores(Planner(), test)}; print("start instruction:", out["start"], flush=True)
    gepa = dspy.GEPA(metric=metric, auto="light", reflection_lm=O.kimi, num_threads=O.THREADS, track_stats=True, seed=0)
    best = gepa.compile(Planner(), trainset=train, valset=val)
    out["gepa"] = test_scores(best, test); print("GEPA:", out["gepa"], flush=True)
    out["gepa_instruction"] = best.plan.signature.instructions; best.save(f"{HERE}/data/dspy_B_gepa.json")
    print("GEPA instruction:\n" + out["gepa_instruction"], flush=True)
    json.dump(out, open(f"{HERE}/data/dspy_B_results.json", "w"), indent=1)
