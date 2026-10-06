"""Stacking: one small model per NDA question combines every signal, nested 5-fold CV over the 484 tuning NDAs
(2026-10-03, overnight). Features per (NDA, question): the span model's best score and polarity p_no (out-of-fold:
computed by a model that never saw the NDA), topic words present, and each rule source's answer (Pre-Tier 0, engine v2's
held-out answer, compiler queries, IR queries incl. "not mentioned", graded variants). A multinomial logistic regression
predicts the gold label (Entailment / Contradiction / NotMentioned); an answer is given when its probability is above the
class threshold picked inside the training folds (inner 4-fold) for >= TARGET precision.
usage: stack_cv.py [--target 0.98]"""
import argparse, collections, json, os, random, sys
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from baseline import QUESTIONS, POLARITY, DEFAULT
from cv_mentioned import load, wilson_lo
from mentioned import RX
from cv_engine2 import features as span_features

Q = list(QUESTIONS); LABELS = ["Entailment", "Contradiction", "NotMentioned"]


def answer_of(label, q):
    return "not mentioned" if label == "NotMentioned" else POLARITY.get(q, DEFAULT)[label]


def oof_span(q, docs, fold):
    out = {}
    for f in range(5):
        tr = [d for d in docs if fold[d["id"]] != f]; te = [d for d in docs if fold[d["id"]] == f]
        feats, _ = span_features(tr, te, q)
        for d, (s, i, pn) in zip(te, feats): out[d["id"]] = (s, pn)
    return q, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.98); a = ap.parse_args()
    docs = load(); idx = list(range(len(docs))); random.Random(0).shuffle(idx); fold = {}
    for r, i in enumerate(idx): fold[docs[i]["id"]] = r % 5
    dev_ids = {r["doc"] for r in json.load(open("runs/compiler_dev.json"))}
    cache = "runs/oof_span.json"
    if os.path.exists(cache): span = json.load(open(cache))
    else:
        span = dict(Parallel(n_jobs=8)(delayed(oof_span)(q, docs, fold) for q in Q)); json.dump(span, open(cache, "w"))
    src = collections.defaultdict(dict)
    for r in json.load(open("runs/cv_engine2.json")):
        if r["answer"] and r["why"] != "Pre-Tier 0": src["v2"][(r["doc"], r["q"])] = r["answer"]
    for s, name in (("comp", "compiler"), ("ir", "ir"), ("var", "variants")):
        for split in ("train", "dev"):
            for r in json.load(open(f"runs/{name}_{split}.json")):
                if r["answer"]: src[s if s != "var" else r["source"]][(r["doc"], r["q"])] = r["answer"]
    from router.pretier0 import check
    from baseline import QUESTIONS as QT
    for d in docs:
        for q in Q:
            p = check(QT[q], d["text"])
            if p.fired: src["pt0"][(d["id"], q)] = p.answer
    names = ["pt0", "v2", "comp", "ir", "var_c3", "var_c4", "var_oral"]
    def vec(d, q):
        s, pn = span[q][d["id"]]
        x = [s, s * s, (pn if pn is not None else 0.5), float(pn is not None), float(bool(RX[q].search(d["text"])))]
        for n in names:
            v = src[n].get((d["id"], q))
            x += [float(v == "yes"), float(v == "no"), float(v == "not mentioned")]
        return x
    out = []
    for q in Q:
        X = np.array([vec(d, q) for d in docs]); y = np.array([d["gold"][q] for d in docs]); ids = [d["id"] for d in docs]
        fo = np.array([fold[i] for i in ids])
        for f in range(5):
            tr = fo != f; te = fo == f
            # inner out-of-fold probabilities on the training folds, for thresholds
            tri = np.where(tr)[0]; inner = np.arange(len(tri)) % 4; P_in = np.zeros((len(tri), 3))
            for g in range(4):
                a_, b_ = tri[inner != g], tri[inner == g]
                if len(set(y[a_])) < 2: continue
                m = LogisticRegression(C=1.0, max_iter=3000).fit(X[a_], y[a_])
                P_in[inner == g] = [[p[list(m.classes_).index(L)] if L in m.classes_ else 0 for L in LABELS] for p in m.predict_proba(X[b_])]
            ths = {}
            for k, L in enumerate(LABELS):
                sc = P_in[:, k]; good = y[tri] == L; best = None
                for t in sorted(set(np.round(sc, 3)), reverse=True):
                    sel = sc >= t
                    if sel.sum() >= 15 and good[sel].mean() >= a.target: best = t
                    elif sel.sum() >= 15: break
                ths[L] = best
            m = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
            P = m.predict_proba(X[te])
            for i, p in zip(np.where(te)[0], P):
                probs = {L: (p[list(m.classes_).index(L)] if L in m.classes_ else 0) for L in LABELS}
                cands = [L for L in LABELS if ths[L] is not None and probs[L] >= ths[L]]
                lab = max(cands, key=lambda L: probs[L]) if cands else None
                out.append({"doc": ids[i], "q": q, "gold": y[i], "answer": answer_of(lab, q) if lab else None, "dev": ids[i] in dev_ids})
    json.dump(out, open("runs/stack_cv.json", "w"))
    def right(r): return r["answer"] == answer_of(r["gold"], r["q"]) if r["gold"] != "NotMentioned" else r["answer"] == "not mentioned"
    def summ(rows):
        n = len(rows); ans = [r for r in rows if r["answer"]]; ok = sum(right(r) for r in ans)
        return f"{len(ans):5}/{n} = {len(ans) / n:5.1%} answered, right {ok / max(len(ans), 1):6.1%} [lo {wilson_lo(ok, len(ans)):.3f}]"
    print(f"== stacking, nested 5-fold CV, target {a.target}")
    for q in Q:
        rows = [r for r in out if r["q"] == q]; print(f"  {q:6} {summ(rows)}")
    print(f"  all    {summ(out)}\n  dev    {summ([r for r in out if r['dev']])}")


if __name__ == "__main__":
    main()
