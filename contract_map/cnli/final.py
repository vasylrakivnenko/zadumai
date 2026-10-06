"""The one sealed run (2026-10-03): engine v2 configured on all 484 tuning NDAs, per question, then applied once to the
123 test NDAs. Only the questions that passed >= 98% in cross-validation answer (`--questions`); the others abstain.
Logged in sealed_runs.log; refuses to run twice unless --again (which is also logged).
Compiler queries (queries.py, `--compiler`) answer first for their questions; engine v2 for the rest.
usage: final.py --questions nda-1,nda-3,... [--compiler nda-3,nda-4,...] [--target 0.99]"""
import argparse, collections, json, os, sys, time, zipfile
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from baseline import QUESTIONS, ZIP
from cv_mentioned import load, wilson_lo
from cv_engine2 import configure, features, decide, right
HERE = os.path.dirname(os.path.abspath(__file__))


def test_docs():
    import re
    out = []
    for d in json.loads(zipfile.ZipFile(ZIP).read("contract-nli/test.json"))["documents"]:
        ann = d["annotation_sets"][0]["annotations"]
        out.append({"id": d["id"], "text": d["text"], "spans": [re.sub(r"\s+", " ", d["text"][a:b]).strip() for a, b in d["spans"]],
                    "gold": {q: ann[q]["choice"] for q in QUESTIONS}, "ev": {q: set(ann[q]["spans"]) for q in QUESTIONS}})
    return out


def run(q, tune, test, target, p0_tune, p0_test):
    cfg = configure(tune, q, target)
    pa = [(p0_tune[(d["id"], q)], d["gold"][q]) for d in tune if (d["id"], q) in p0_tune]
    use_pt0 = bool(len(pa) >= 10 and np.mean([right(x, g, q) for x, g in pa]) >= target)
    feats, _ = features(tune, test, q); out = []
    for d, ft in zip(test, feats):
        ans, why = decide(ft, d, q, cfg)
        if ans is None and use_pt0 and (d["id"], q) in p0_test: ans, why = p0_test[(d["id"], q)], "Pre-Tier 0"
        out.append({"doc": d["id"], "q": q, "gold": d["gold"][q], "answer": ans, "why": why})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--questions", required=True); ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--again", action="store_true"); ap.add_argument("--compiler", default=""); a = ap.parse_args()
    log = f"{HERE}/sealed_runs.log"
    if os.path.exists(log) and "final.py" in open(log).read() and not a.again:
        raise SystemExit("the sealed test was already run once; --again to run it again (logged)")
    open(log, "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
    qs = a.questions.split(","); tune = load(); test = test_docs()
    from router.pretier0 import check
    p0 = lambda docs: {(d["id"], q): p.answer for d in docs for q in qs if (p := check(QUESTIONS[q], d["text"])).fired}
    p0_tune, p0_test = p0(tune), p0(test)
    res = [r for rs in Parallel(n_jobs=8)(delayed(run)(q, tune, test, a.target, p0_tune, p0_test) for q in qs) for r in rs]
    cq = [q for q in a.compiler.split(",") if q]
    if cq:
        from compiler import compile_doc
        from queries import QUERIES
        progs = {d["id"]: compile_doc(d["text"]) for d in test}; gold = {d["id"]: d["gold"] for d in test}
        have = {(r["doc"], r["q"]) for r in res}
        for q in cq:
            for d in test:
                if (d["id"], q) not in have: res.append({"doc": d["id"], "q": q, "gold": d["gold"][q], "answer": None, "why": ""})
        for r in res:
            if r["q"] in cq and (c := QUERIES[r["q"]](progs[r["doc"]])):
                r["answer"], r["why"] = c[0], "compiler"
    json.dump(res, open(f"{HERE}/runs/final_test.json", "w"))
    st = collections.defaultdict(lambda: [0, 0, 0])
    for r in res:
        c = st[r["q"]]; c[0] += 1
        if r["answer"]: c[1] += 1; c[2] += right(r["answer"], r["gold"], r["q"])
    print(f"== sealed test: {len(test)} NDAs x {len(qs)} questions")
    for q in sorted({r["q"] for r in res}, key=lambda x: int(x.split("-")[1])):
        n, k, ok = st[q]; print(f"  {q:6} answered {k:3}/{n} = {k / n:4.0%}, right {ok}/{k} = {ok / max(k, 1):6.1%}")
    n = sum(v[0] for v in st.values()); k = sum(v[1] for v in st.values()); ok = sum(v[2] for v in st.values())
    print(f"  all: answered {k}/{n} = {k / n:.1%}, right {ok}/{k} = {ok / max(k, 1):.1%} [lo {wilson_lo(ok, k):.3f}]")


if __name__ == "__main__":
    main()
