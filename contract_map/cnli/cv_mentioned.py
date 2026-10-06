"""Learned "does this NDA address question q at all?" detector, with nested 5-fold cross-validation over the 484 tuning
NDAs (train + dev), split by document (2026-10-03; the user asked for cross-validation).
Model: TF-IDF + logistic regression per question over the NDA's spans (ContractNLI's own segmentation): positives =
the gold evidence spans of q (Entailment or Contradiction), negatives = every other span. A document's score =
its highest span score. "Not mentioned" is answered when the score is below t_q (optionally: and no topic word).
t_q is picked inside the 4 training folds (inner 4-fold out-of-fold scores) as the highest threshold whose calls
are >= TARGET precision there; then applied once to the held-out fold. The test NDAs are not used.
usage: cv_mentioned.py [--target 0.99]"""
import argparse, collections, json, math, os, random, re, sys, zipfile
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mentioned import RX
from baseline import QUESTIONS, ZIP

Q = list(QUESTIONS)


def wilson_lo(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def load():
    z = zipfile.ZipFile(ZIP); docs = []
    for s in ("train", "dev"):
        for d in json.loads(z.read(f"contract-nli/{s}.json"))["documents"]:
            spans = [re.sub(r"\s+", " ", d["text"][a:b]).strip() for a, b in d["spans"]]
            ann = d["annotation_sets"][0]["annotations"]
            docs.append({"id": d["id"], "text": d["text"], "spans": spans,
                         "gold": {q: ann[q]["choice"] for q in Q}, "ev": {q: set(ann[q]["spans"]) for q in Q}})
    return docs


def fit_score(train_docs, test_docs, q):
    """Train the span classifier for q on train_docs; return each test doc's max span score."""
    X, y = [], []
    for d in train_docs:
        for i, s in enumerate(d["spans"]): X.append(s); y.append(int(i in d["ev"][q]))
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000)
    m = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(v.fit_transform(X), y)
    out = []
    for d in test_docs:
        out.append(float(m.predict_proba(v.transform(d["spans"]))[:, 1].max()) if d["spans"] else 0.0)
    return np.array(out)


def pick_threshold(scores, notmentioned, target):
    """Highest t such that the docs scoring below t are >= target not-mentioned (and at least 20 calls)."""
    best = 0.0
    for t in sorted(set(np.round(scores, 4))):
        k = scores < t
        if k.sum() >= 20 and notmentioned[k].mean() >= target: best = t
    return best


def one_question(q, docs, folds, target):
    res = []
    for f in range(5):
        tr = [d for d, k in zip(docs, folds) if k != f]; te = [d for d, k in zip(docs, folds) if k == f]
        inner = np.array([i % 4 for i in range(len(tr))]); oof = np.zeros(len(tr))
        for g in range(4):
            a = [d for d, k in zip(tr, inner) if k != g]; b = [i for i, k in enumerate(inner) if k == g]
            oof[b] = fit_score(a, [tr[i] for i in b], q)
        t = pick_threshold(oof, np.array([d["gold"][q] == "NotMentioned" for d in tr]), target)
        s = fit_score(tr, te, q)
        for d, x in zip(te, s):
            res.append((d["gold"][q], x < t, bool(RX[q].search(d["text"])), t))
    return q, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.99); a = ap.parse_args()
    docs = load(); idx = list(range(len(docs))); random.Random(0).shuffle(idx)
    folds = [0] * len(docs)
    for r, i in enumerate(idx): folds[i] = r % 5
    print(f"== nested 5-fold CV over {len(docs)} NDAs (train + dev), target precision {a.target}", flush=True)
    out = Parallel(n_jobs=8)(delayed(one_question)(q, docs, folds, a.target) for q in Q)
    tot = collections.Counter()
    for q, res in out:
        n_nm = sum(g == "NotMentioned" for g, *_ in res)
        for variant, call in (("model", lambda c, kw: c), ("model+no topic word", lambda c, kw: c and not kw)):
            calls = [(g, c) for g, c0, kw, t in res if (c := call(c0, kw))]
            ok = sum(g == "NotMentioned" for g, _ in calls)
            if variant == "model": tot["n_nm"] += n_nm; tot["ok"] += ok; tot["calls"] += len(calls)
            else: tot["ok2"] += ok; tot["calls2"] += len(calls)
            print(f"  {q:6} {variant:20} answers {ok:3}/{n_nm:3} not-mentioned ({ok / max(n_nm, 1):4.0%}) | wrong {len(calls) - ok:2} | "
                  f"precision {ok / max(len(calls), 1):6.1%} [lo {wilson_lo(ok, len(calls)):.3f}] | thresholds {sorted({round(t, 3) for *_, t in res})}", flush=True)
    print(f"  all model: {tot['ok']}/{tot['n_nm']} not-mentioned answered ({tot['ok'] / tot['n_nm']:.0%}), precision {tot['ok'] / max(tot['calls'], 1):.1%} "
          f"[lo {wilson_lo(tot['ok'], tot['calls']):.3f}] | model+no topic word: {tot['ok2']} answered, precision {tot['ok2'] / max(tot['calls2'], 1):.1%}")


if __name__ == "__main__":
    main()
