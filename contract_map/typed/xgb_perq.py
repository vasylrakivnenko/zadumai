"""Separate models per NDA question (2026-10-04; the user: "Should we build separate models for each clause within NDA
maybe?") vs one model with the question's identity as a feature. 100% coverage, the likeliest of yes / no / not_stated.
Training NDAs: nda_tune (and, with --more, the 284 other NDAs: installed-network features only). Held-out: nda_ho.
usage: venv_xgb python xgb_perq.py [--tag ft] [--more]"""
import argparse, collections, json, os
import numpy as np
import xgboost as xgb
import xgb_stack as S
import xgb_full as F

HERE = os.path.dirname(os.path.abspath(__file__))


def small(n):  # fewer, shallower trees for small per-question sets
    return xgb.XGBClassifier(n_estimators=150 if n < 300 else 300, max_depth=3, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                             n_jobs=6, verbosity=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--more", action="store_true"); a = ap.parse_args()
    ex = F.extra()
    train = S.load("nda_tune", a.tag)
    if a.more:
        for k in range(3):
            p = f"{HERE}/feats/nda_more_more{k}.jsonl"
            if os.path.exists(p): train += [dict(json.loads(l), set="nda_more") for l in open(p)]
    test = S.load("nda_ho", a.tag)
    qids = sorted({o["qid"] for o in train})
    Xtr = F.features(train, qids, ex, True, True); ytr = np.array([S.target(o) for o in train])
    Xte = F.features(test, qids, ex, True, True)
    pooled = S.fit(Xtr, ytr).predict_proba(Xte)
    per = np.zeros_like(pooled); stats = {}
    for q in qids:
        i_tr = [i for i, o in enumerate(train) if o["qid"] == q]; i_te = [i for i, o in enumerate(test) if o["qid"] == q]
        ys = ytr[i_tr]
        if len(set(ys)) < 2:
            per[i_te] = np.eye(3)[ys[0]]; continue
        m = small(len(i_tr))
        present = sorted(set(ys)); remap = {c: k for k, c in enumerate(present)}
        m.set_params(num_class=len(present)) if len(present) > 2 else m.set_params(objective="binary:logistic", num_class=None)
        m.fit(Xtr[i_tr], np.array([remap[c] for c in ys]))
        p = m.predict_proba(Xte[i_te])
        full = np.zeros((len(i_te), 3))
        for k, c in enumerate(present): full[:, c] = p[:, k]
        per[i_te] = full
    res = collections.defaultdict(lambda: [0, 0, 0])
    for o, pp, pq in zip(test, pooled, per):
        r = res[o["qid"]]; r[0] += 1
        r[1] += S.right(o, S.CLASSES[int(np.argmax(pp))]); r[2] += S.right(o, S.CLASSES[int(np.argmax(pq))])
    n = len(test); tp = sum(v[1] for v in res.values()); tq = sum(v[2] for v in res.values())
    print(f"training NDAs {len({o['id'].split('/')[0] for o in train})} ({len(train)} rows), network {'retrained' if a.tag else 'installed'}")
    print(f"  ONE model + question identity: {tp / n:6.1%}   SEPARATE model per question: {tq / n:6.1%}   (held-out {n} questions)")
    print("  per question (one model / separate):", ", ".join(f"{q} {v[1]}/{v[2]} of {v[0]}" for q, v in sorted(res.items(), key=lambda kv: int(kv[0].split('-')[1]))))


if __name__ == "__main__":
    main()
