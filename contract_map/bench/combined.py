"""Train one clause classifier on all clause datasets vs one per dataset (2026-10-02). TF-IDF + one-vs-rest
logistic regression in both, so only the training data differs.

Labels are mapped into one taxonomy (UNIFIED below):
  eq  - the dataset's label means the same as the unified class: its rows are positives AND negatives for it,
        and the dataset is scored through the unified classifier;
  sub - the dataset's label is a narrower kind (ToS marks only *unfair* clauses; CUAD's "termination for
        convenience" is one kind of termination): its positive rows are added to the unified class, while the
        dataset keeps and is scored through its own classifier.
Everything else stays the dataset's own class. A class trains on the rows of the datasets that mark it
completely, plus the positive rows of datasets that map to it with `sub`.
Scoring: each dataset on its own test rows and label space (LEDGAR with near-synonym labels merged), threshold
picked on its own validation rows. usage: combined.py --stage N"""
import argparse, collections, sys, time, os
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench

DATASETS = ["ledgar", "cuad_clause", "tos", "opp115", "contractnli", "lease"]
U = lambda name: "U:" + name
UNIFIED = {
    "ledgar": {**{l: (U(g), "eq") for l, g in bench.MERGE.items()},
               "Arbitration": (U("arbitration"), "eq"), "Insurances": (U("insurance"), "eq"),
               "Change In Control": (U("change of control"), "eq"), "Non-Disparagement": (U("non-disparagement"), "eq"),
               "Terminations": (U("termination"), "eq"), "Survival": (U("survival"), "eq"), "Payments": (U("payment terms"), "eq"),
               "Taxes": (U("taxes"), "eq"), "Warranties": (U("warranties"), "eq"),
               "Intellectual Property": (U("intellectual property"), "eq"), "Terms": (U("term of agreement"), "eq")},
    "cuad_clause": {"Governing Law": (U("governing law"), "eq"), "Anti-Assignment": (U("assignment/successors"), "eq"),
                    "Insurance": (U("insurance"), "eq"), "Change Of Control": (U("change of control"), "eq"),
                    "Non-Disparagement": (U("non-disparagement"), "eq"), "Expiration Date": (U("term of agreement"), "eq"),
                    "Renewal Term": (U("renewal/extension"), "eq"), "Cap On Liability": (U("limitation of liability"), "eq"),
                    "No-Solicit Of Employees": (U("non-solicitation of employees"), "eq"),
                    "Termination For Convenience": (U("termination"), "sub"), "Warranty Duration": (U("warranties"), "sub"),
                    "Ip Ownership Assignment": (U("intellectual property"), "sub"), "Joint Ip Ownership": (U("intellectual property"), "sub")},
    "tos": {"Choice of law": (U("governing law"), "sub"), "Jurisdiction": (U("jurisdiction/venue"), "sub"),
            "Arbitration": (U("arbitration"), "sub"), "Limitation of liability": (U("limitation of liability"), "sub"),
            "Unilateral termination": (U("termination"), "sub"), "Unilateral change": (U("changes by the provider"), "sub")},
    "opp115": {"Policy Change": (U("changes by the provider"), "eq")},
    "contractnli": {"Survival of obligations": (U("survival"), "eq"), "No solicitation": (U("non-solicitation of employees"), "eq")},
    "lease": {"term_of_payment": (U("payment terms"), "eq"), "extension_period": (U("renewal/extension"), "eq"),
              "vat": (U("taxes"), "sub")},
}


def own(ds, l): return f"{ds}:{l}"


def score_class(ds, l):
    u, kind = UNIFIED[ds].get(l, (None, None))
    return u if kind == "eq" else own(ds, l)


def row_classes(ds, labels):
    """The classes a row is positive for: its scoring classes + unified classes of its `sub` labels."""
    out = {score_class(ds, l) for l in labels}
    out |= {UNIFIED[ds][l][0] for l in labels if UNIFIED[ds].get(l, (0, ""))[1] == "sub"}
    return out


def fit_one(X, y, w):
    if y.min() == y.max(): return None
    m = LogisticRegression(C=10, max_iter=2000, class_weight="balanced"); m.fit(X, y); return m


def train(rows_by_ds, full_by_ds, vec):
    """rows_by_ds: ds -> training rows; full_by_ds: ds -> classes the ds marks completely. Returns class -> model."""
    allrows = [(ds, r) for ds, rs in rows_by_ds.items() for r in rs]
    X = vec.transform([r["text"] for _, r in allrows])
    pos = [row_classes(ds, r["labels"]) for ds, r in allrows]
    classes = sorted(set().union(*full_by_ds.values()))
    jobs = []
    for c in classes:
        idx = [i for i, (ds, _) in enumerate(allrows) if c in full_by_ds[ds] or c in pos[i]]
        y = np.array([c in pos[i] for i in idx], dtype=int)
        jobs.append((c, idx, y))
    models = Parallel(n_jobs=int(os.environ.get("BENCH_JOBS", "8")))(delayed(fit_one)(X[idx], y, None) for c, idx, y in jobs)
    return {c: m for (c, _, _), m in zip(jobs, models) if m is not None}


def predict(models, vec, rows, ds, labels):
    X = vec.transform([r["text"] for r in rows])
    cls = sorted({score_class(ds, l) for l in labels})
    S = np.zeros((len(rows), len(cls)))
    for j, c in enumerate(cls):
        if c in models: S[:, j] = models[c].predict_proba(X)[:, 1]
    return cls, S


def evaluate(ds, S_, cls, rows, val_pack, all_classes=False):
    gold = [{score_class(ds, l) for l in r["labels"]} for r in rows]
    if ds == "ledgar":  # one label per provision: argmax, near-synonyms merged
        pred = [cls[i] for i in S_.argmax(1)]
        return {"acc": np.mean([p in g for p, g in zip(pred, gold)])}
    vcls, VS, vrows = val_pack
    vgold = [{score_class(ds, l) for l in r["labels"]} for r in vrows]
    def f1(G, P):
        tp = sum(len(g & p) for g, p in zip(G, P)); fp = sum(len(p - g) for g, p in zip(G, P)); fn = sum(len(g - p) for g, p in zip(G, P))
        return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    th = max((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: f1(vgold, [{vcls[j] for j in np.where(s >= t)[0]} for s in VS])[0])
    P = [{cls[j] for j in np.where(s >= th)[0]} for s in S_]
    F, pr, rc = f1(gold, P)
    per = {}
    for c in cls:
        if c.startswith("U:") or all_classes:
            g = [c in x for x in gold]; p = [c in x for x in P]
            tp = sum(a and b for a, b in zip(g, p)); per[c] = (2 * tp / max(sum(g) + sum(p), 1), sum(g))
    return {"f1": F, "p": pr, "r": rc, "th": th, "per": per}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=1); a = ap.parse_args()
    n_eval, n_train = bench.STAGES[a.stage]
    S = {ds: bench.SETS[ds]() for ds in DATASETS}
    tr = {ds: bench.sample(S[ds]["train"], n_train, seed=1, multi=S[ds]["multi"]) for ds in DATASETS}
    ev = {ds: bench.sample(S[ds]["test"], n_eval, multi=S[ds]["multi"]) for ds in DATASETS}
    va = {ds: bench.sample(S[ds]["val"], 2000, multi=True) for ds in DATASETS}
    full = {ds: {score_class(ds, l) for l in S[ds]["labels"]} for ds in DATASETS}
    print(f"== combined vs separate, stage {a.stage}: train " + ", ".join(f"{ds} {len(tr[ds])}" for ds in DATASETS)
          + " | eval " + ", ".join(f"{ds} {len(ev[ds])}" for ds in DATASETS), flush=True)
    print(f"   unified classes shared by 2+ datasets: {sorted({u for m in UNIFIED.values() for u, _ in m.values()})}")
    res = {}
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=400_000, sublinear_tf=True)
    vec.fit([r["text"] for ds in DATASETS for r in tr[ds]])
    models = train(tr, full, vec)
    print(f"   combined: {len(models)} classes trained in {time.time() - t0:.0f}s", flush=True)
    for ds in DATASETS:
        cls, Sc = predict(models, vec, ev[ds], ds, S[ds]["labels"])
        vp = predict(models, vec, va[ds], ds, S[ds]["labels"]) + (va[ds],) if va[ds] else None
        res[(ds, "combined")] = evaluate(ds, Sc, cls, ev[ds], vp)
    for ds in DATASETS:  # separate: the same code with only this dataset's rows (its sub positives stay in it)
        t1 = time.time()
        v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=400_000, sublinear_tf=True).fit([r["text"] for r in tr[ds]])
        m1 = train({ds: tr[ds]}, {ds: full[ds] | {UNIFIED[ds][l][0] for l in S[ds]["labels"] if UNIFIED[ds].get(l, (0, ""))[1] == "sub"}}, v1)
        cls, Sc = predict(m1, v1, ev[ds], ds, S[ds]["labels"])
        vp = predict(m1, v1, va[ds], ds, S[ds]["labels"]) + (va[ds],) if va[ds] else None
        res[(ds, "separate")] = evaluate(ds, Sc, cls, ev[ds], vp)
        print(f"   separate {ds}: {time.time() - t1:.0f}s", flush=True)
    print(f"\n   {'dataset':12} {'separate':>22} {'combined':>22}   change")
    for ds in DATASETS:
        s, c = res[(ds, "separate")], res[(ds, "combined")]
        if ds == "ledgar":
            print(f"   {ds:12} {'acc %.1f%%' % (100 * s['acc']):>22} {'acc %.1f%%' % (100 * c['acc']):>22}   {100 * (c['acc'] - s['acc']):+.1f} pts")
        else:
            fmt = lambda x: f"F1 {x['f1']:.3f} (P {100 * x['p']:.0f} R {100 * x['r']:.0f})"
            print(f"   {ds:12} {fmt(s):>22} {fmt(c):>22}   {c['f1'] - s['f1']:+.3f}")
            for u in sorted(c["per"]):
                if u in s["per"] and c["per"][u][1]:
                    print(f"        {u:38} n {c['per'][u][1]:4}  F1 {s['per'][u][0]:.2f} -> {c['per'][u][0]:.2f}")


if __name__ == "__main__":
    main()
