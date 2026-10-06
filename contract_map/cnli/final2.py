"""The one sealed run of the full NDA system (2026-10-03, overnight; configuration recorded in RESULTS.md before running).
Sources on the 123 test NDAs: Pre-Tier 0; engine v2 configured on all 484 tuning NDAs; the compiler queries; the IR
queries; the graded variants. Which source answers which question (and which answer type) is decided on the 484 tuning
NDAs exactly as select_cv.py does (trusted if >= TARGET right with >= MIN_N answers; v2 judged on its held-out answers;
agreeing pairs as extra sources; disagreement -> no answer). Logged in sealed_runs.log; refuses a second run.
usage: final2.py [--target 0.98] [--min-n 15]"""
import argparse, collections, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from baseline import QUESTIONS, POLARITY, DEFAULT
from cv_mentioned import load, wilson_lo
from final import test_docs

Q = list(QUESTIONS); HERE = os.path.dirname(os.path.abspath(__file__))


def right(ans, gold, q):
    if ans == "not mentioned": return gold == "NotMentioned"
    return POLARITY.get(q, DEFAULT).get(gold) == ans


def v2_test(q, tune, test):
    from cv_engine2 import configure, features, decide
    cfg = configure(tune, q, 0.99); feats, _ = features(tune, test, q)
    return q, {d["id"]: decide(ft, d, q, cfg)[0] for d, ft in zip(test, feats)}


def answer_types(src, names):
    out = collections.defaultdict(dict)
    for s0 in names:
        for k, v in src[s0].items(): out[f"{s0}:{v}"][k] = v
    base = sorted(names)
    for i, s1 in enumerate(base):
        for s2 in base[i + 1:]:
            for v in ("yes", "no", "not mentioned"):
                for k, x in out[f"{s1}:{v}"].items():
                    if out[f"{s2}:{v}"].get(k) == x: out[f"{s1}+{s2}:{v}"][k] = x
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.98); ap.add_argument("--min-n", type=int, default=15)
    a = ap.parse_args()
    log = f"{HERE}/sealed_runs.log"
    if os.path.exists(log) and "final" in open(log).read():
        raise SystemExit("the sealed test was already run; refusing a second run")
    open(log, "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
    tune = load(); test = test_docs(); names = ["pt0", "v2", "comp", "ir", "var_c3", "var_c4", "var_oral"]
    # --- tuning-side answers (as select_cv.py) ---
    tsrc = collections.defaultdict(dict)
    for r in json.load(open(f"{HERE}/runs/cv_engine2.json")):
        if r["answer"] and r["why"] != "Pre-Tier 0": tsrc["v2"][(r["doc"], r["q"])] = r["answer"]
    for s, name in (("comp", "compiler"), ("ir", "ir")):
        for split in ("train", "dev"):
            for r in json.load(open(f"{HERE}/runs/{name}_{split}.json")):
                if r["answer"]: tsrc[s][(r["doc"], r["q"])] = r["answer"]
    for split in ("train", "dev"):
        for r in json.load(open(f"{HERE}/runs/variants_{split}.json")):
            if r["answer"]: tsrc[r["source"]][(r["doc"], r["q"])] = r["answer"]
    from router.pretier0 import check
    for d in tune:
        for q in Q:
            p = check(QUESTIONS[q], d["text"])
            if p.fired: tsrc["pt0"][(d["id"], q)] = p.answer
    tgold = {(d["id"], q): d["gold"][q] for d in tune for q in Q}
    tt = answer_types(tsrc, names)
    trusted = {}
    for q in Q:
        trusted[q] = []
        for s, m in tt.items():
            ans = [(v, tgold[k]) for k, v in m.items() if k[1] == q]
            if len(ans) >= a.min_n and sum(right(v, g, q) for v, g in ans) / len(ans) >= a.target: trusted[q].append(s)
    # --- test-side answers ---
    xsrc = collections.defaultdict(dict)
    for q, ans in Parallel(n_jobs=8)(delayed(v2_test)(q, tune, test) for q in Q):
        for i, v in ans.items():
            if v: xsrc["v2"][(i, q)] = v
    from compiler import compile_doc
    from queries import QUERIES as CQ
    from queries_ir import QUERIES as IQ
    from ir import facts
    from variants import VARIANTS
    for d in test:
        p = compile_doc(d["text"]); f = facts(d["text"])
        for q in Q:
            pc = check(QUESTIONS[q], d["text"])
            if pc.fired: xsrc["pt0"][(d["id"], q)] = pc.answer
            if q in CQ and (r := CQ[q](p)): xsrc["comp"][(d["id"], q)] = r[0]
            if q in IQ and (r := IQ[q](f)): xsrc["ir"][(d["id"], q)] = r[0]
        for name, (q, fn) in VARIANTS.items():
            if (v := fn(p)): xsrc[name][(d["id"], q)] = v
    xt = answer_types(xsrc, names)
    out = []
    for d in test:
        for q in Q:
            vals = {xt[s][(d["id"], q)] for s in trusted[q] if (d["id"], q) in xt[s]}
            ans = vals.pop() if len(vals) == 1 else None
            out.append({"doc": d["id"], "q": q, "gold": d["gold"][q], "answer": ans, "conflict": len(vals) > 0})
    json.dump({"rows": out, "trusted": trusted}, open(f"{HERE}/runs/final2_test.json", "w"))
    def summ(rows):
        n = len(rows); ans = [r for r in rows if r["answer"]]; ok = sum(right(r["answer"], r["gold"], r["q"]) for r in ans)
        return f"{len(ans):4}/{n} = {len(ans) / n:5.1%} answered, right {ok}/{len(ans)} = {ok / max(len(ans), 1):6.1%} [lo {wilson_lo(ok, len(ans)):.3f}]"
    print(f"== SEALED TEST: {len(test)} NDAs x {len(Q)} questions (target {a.target}, min_n {a.min_n})")
    for q in Q:
        print(f"  {q:6} {summ([r for r in out if r['q'] == q])}")
    print(f"  all    {summ(out)}")


if __name__ == "__main__":
    main()
