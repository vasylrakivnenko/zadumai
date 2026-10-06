"""The NDA question bank's learned part, scored with nested 5-fold cross-validation over the 484 tuning NDAs
(train + dev; the 123 test NDAs untouched) (2026-10-03).
Per question q, a span model (TF-IDF + LR: q's gold evidence spans vs all other spans) scores each NDA by its best
span. Thresholds are picked inside the 4 training folds (inner 4-fold out-of-fold scores), each the loosest one
whose answers are >= TARGET right there (and at least MIN_CALLS of them), then applied once to the held-out fold:
  yes            best span >= t_yes, for questions whose answer when addressed is almost always "yes"
                 (Contradiction < 5% of addressed NDAs in the training folds); the best span is the evidence
  not mentioned  best span < t_nm (model), or no topic word in the NDA (keywords), or either, whichever variant
                 the inner folds pick
Pre-Tier 0's own answers are added where the bank has none (it reads the same NDA; its "no" answers count too).
Writes runs/cv_engine.json (every held-out answer) for reading errors.
usage: cv_engine.py [--target 0.99]"""
import argparse, collections, json, os, random, sys
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from mentioned import RX
from baseline import QUESTIONS, POLARITY, DEFAULT
from cv_mentioned import load, wilson_lo

Q = list(QUESTIONS)
MIN_CALLS = 20
HERE = os.path.dirname(os.path.abspath(__file__))


def model_scores(train_docs, test_docs, q):
    X, y = [], []
    for d in train_docs:
        for i, s in enumerate(d["spans"]): X.append(s); y.append(int(i in d["ev"][q]))
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000)
    m = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(v.fit_transform(X), y)
    out = []
    for d in test_docs:
        p = m.predict_proba(v.transform(d["spans"]))[:, 1] if d["spans"] else np.zeros(1)
        out.append((float(p.max()), int(p.argmax())))
    return out


def loosest(scores, good, target, above):
    """The loosest threshold whose calls (score >= t if above else score < t) are >= target good, >= MIN_CALLS."""
    ts = sorted(set(np.round(scores, 4)), reverse=above)  # from strict to loose (bug before 2026-10-03 13:00 PT: loose first)
    best = None
    for t in ts:
        k = scores >= t if above else scores < t
        if k.sum() >= MIN_CALLS and good[k].mean() >= target: best = t
        elif k.sum() >= MIN_CALLS: break
    return best


def one_question(q, docs, folds, target, p0=None):
    out = []
    for f in range(5):
        tr = [d for d, k in zip(docs, folds) if k != f]; te = [d for d, k in zip(docs, folds) if k == f]
        g_tr = np.array([d["gold"][q] for d in tr])
        addressed = g_tr != "NotMentioned"
        inner = np.array([i % 4 for i in range(len(tr))]); oof = np.zeros(len(tr))
        for g in range(4):
            a = [d for d, k in zip(tr, inner) if k != g]; b = [i for i, k in enumerate(inner) if k == g]
            oof[b] = [s for s, _ in model_scores(a, [tr[i] for i in b], q)]
        nm = g_tr == "NotMentioned"; kw = np.array([bool(RX[q].search(d["text"])) for d in tr])
        t_yes = loosest(oof, g_tr == "Entailment", target, above=True)
        t_no = loosest(oof, g_tr == "Contradiction", target, above=True)  # the top is mostly "no" (e.g. "technical only")
        if t_yes is not None and t_no is not None:  # both can't hold; keep the one answering more
            t_yes, t_no = (t_yes, None) if (oof >= t_yes).sum() >= (oof >= t_no).sum() else (None, t_no)
        t_nm = loosest(oof, nm, target, above=False)
        # not-mentioned variants, picked on the inner out-of-fold calls: the one answering most at >= target
        variants = {"model": (oof < t_nm) if t_nm is not None else np.zeros(len(tr), bool), "keywords": ~kw}
        variants["either"] = variants["model"] | variants["keywords"]
        ok = {n: (c.sum() >= MIN_CALLS and nm[c].mean() >= target) for n, c in variants.items()}
        pick = max((n for n in variants if ok[n]), key=lambda n: variants[n].sum(), default=None)
        # Pre-Tier 0 counts for q only if its answers on the training folds are >= target right
        pa = [(p0[(d["id"], q)], d["gold"][q]) for d in tr if (d["id"], q) in (p0 or {})]
        use_pt0 = bool(len(pa) >= 10 and np.mean([right(x, g, q) for x, g in pa]) >= target)
        sc = model_scores(tr, te, q)
        for d, (s, i) in zip(te, sc):
            ans, why = None, ""
            want_yes = POLARITY.get(q, DEFAULT)["Entailment"]; want_no = POLARITY.get(q, DEFAULT)["Contradiction"]
            if t_yes is not None and s >= t_yes: ans, why = want_yes, f"model: span {i} scores {s:.2f} >= {t_yes} (gold Entailment)"
            elif t_no is not None and s >= t_no: ans, why = want_no, f"model: span {i} scores {s:.2f} >= {t_no} (gold Contradiction)"
            else:
                kwd = bool(RX[q].search(d["text"]))
                call = {"model": t_nm is not None and s < t_nm, "keywords": not kwd,
                        "either": (t_nm is not None and s < t_nm) or not kwd}.get(pick, False)
                if call: ans, why = "not mentioned", f"{pick}: best span {s:.2f}"
            out.append({"doc": d["id"], "q": q, "gold": d["gold"][q], "answer": ans, "why": why, "span": i,
                        "evidence_ok": i in d["ev"][q], "fold": f, "t_yes": t_yes, "t_no": t_no, "t_nm": t_nm, "nm_variant": pick,
                        "use_pt0": use_pt0})
    return out


def right(ans, gold, q):
    if ans == "not mentioned": return gold == "NotMentioned"
    return POLARITY.get(q, DEFAULT).get(gold) == ans


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.99); a = ap.parse_args()
    docs = load(); idx = list(range(len(docs))); random.Random(0).shuffle(idx); folds = [0] * len(docs)
    for r, i in enumerate(idx): folds[i] = r % 5
    from router.pretier0 import check
    p0 = {}
    for d in docs:
        for q in Q:
            p = check(QUESTIONS[q], d["text"])
            if p.fired: p0[(d["id"], q)] = p.answer
    print(f"== nested 5-fold CV, {len(docs)} NDAs x 17 questions, target {a.target}; Pre-Tier 0 answers {len(p0)}", flush=True)
    res = [r for rs in Parallel(n_jobs=8)(delayed(one_question)(q, docs, folds, a.target, p0) for q in Q) for r in rs]
    for r in res:
        r["pt0"] = p0.get((r["doc"], r["q"]))
        if r["answer"] is None and r["pt0"] and r["use_pt0"]: r["answer"], r["why"] = r["pt0"], "Pre-Tier 0"
    json.dump(res, open(f"{HERE}/runs/cv_engine.json", "w"))
    st = collections.defaultdict(collections.Counter)
    for r in res:
        c = st[r["q"]]; c["n"] += 1; c[r["gold"]] += 1
        if r["answer"]:
            kind = "nm" if r["answer"] == "not mentioned" else ("pt0" if r["why"] == "Pre-Tier 0" else "model")
            c[kind] += 1; c[kind + "_ok"] += right(r["answer"], r["gold"], r["q"])
    print(f"{'question':7} {'E/C/N':>12} | {'model yes/no':>13} {'not mentioned':>15} {'Pre-Tier 0':>12} | answered      right")
    tot = collections.Counter()
    for q in Q:
        c = st[q]; tot.update(c)
        ans = c["model"] + c["nm"] + c["pt0"]; ok = c["model_ok"] + c["nm_ok"] + c["pt0_ok"]
        print(f"{q:7} {c['Entailment']:4}/{c['Contradiction']:3}/{c['NotMentioned']:3} | {c['model_ok']:4}/{c['model']:<4}     "
              f"{c['nm_ok']:4}/{c['nm']:<4}      {c['pt0_ok']:4}/{c['pt0']:<4} | {ans:4} = {ans / c['n']:4.0%}  {ok / max(ans, 1):6.1%} [lo {wilson_lo(ok, ans):.3f}]")
    c = tot; ans = c["model"] + c["nm"] + c["pt0"]; ok = c["model_ok"] + c["nm_ok"] + c["pt0_ok"]
    print(f"{'all':7} {c['Entailment']:4}/{c['Contradiction']:3}/{c['NotMentioned']:4} | model {c['model_ok']}/{c['model']}, not mentioned {c['nm_ok']}/{c['nm']}, "
          f"Pre-Tier 0 {c['pt0_ok']}/{c['pt0']} | answered {ans} = {ans / c['n']:.1%}, right {ok / ans:.1%} [lo {wilson_lo(ok, ans):.3f}]")


if __name__ == "__main__":
    main()
