"""The NDA stacker with the NDA statement encoder's features (ndaenc/aggregate.py -> feats/ndaenc_{train,ho}.jsonl;
out-of-fold for the training NDAs) and/or the symbolic detectors (nda_rules.py). 5-fold CV on the 384 training NDAs
(folds by NDA), then held-out (nda_ho, 60 NDAs).
usage: venv_xgb python nda_eval2.py [config ...]   configs: base, enc, enc_only, enc_rules (default: all four)"""
import collections, json, os, sys, zlib
import numpy as np
import xgb_stack as S, xgb_full as F, nda_joint as J
HERE = os.path.dirname(os.path.abspath(__file__))
rules = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/nda_rules.jsonl")}
rkeys = sorted({k for r in rules.values() for k in r if k != "id"})
enc = {}
for s in ("train", "ho"):
    p = f"{HERE}/feats/ndaenc_{s}.jsonl"
    if os.path.exists(p): enc.update({json.loads(l)["id"]: json.loads(l) for l in open(p)})
ekeys = sorted({k for r in enc.values() for k in r if k != "id"})


def feats(rows, qids, ex, cfg):
    X = F.features(rows, qids, ex, True, True)
    if cfg == "enc_only": X = np.array([[float(o["qid"] == q) for q in qids] for o in rows])
    if cfg in ("enc", "enc_only", "enc_rules"):
        X = np.hstack([X, np.array([[float(enc.get(o["id"], {}).get(k, np.nan)) for k in ekeys] for o in rows])])
    if cfg == "enc_rules":
        R = [[float(bool(rules.get(o["id"], {}).get(k))) if o["qid"] == "nda-" + k.split("_")[0][1:] else np.nan for k in rkeys] for o in rows]
        X = np.hstack([X, np.array(R)])
    return X


ex = F.extra(); train = J.load_train(); test = S.load("nda_ho", ""); qids = sorted({o["qid"] for o in train})
y = np.array([S.target(o) for o in train]); folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
print(f"encoder features for {sum(o['id'] in enc for o in train)}/{len(train)} training rows, {sum(o['id'] in enc for o in test)}/{len(test)} held-out rows")
for cfg in sys.argv[1:] or ["base", "enc", "enc_only", "enc_rules"]:
    X = feats(train, qids, ex, cfg); oof = np.zeros((len(train), 3))
    for f in range(5): oof[folds == f] = S.fit(X[folds != f], y[folds != f]).predict_proba(X[folds == f])
    per = collections.Counter(); tot = collections.Counter()
    for o, p in zip(train, oof):
        tot[o["qid"]] += 1; per[o["qid"]] += S.right(o, S.CLASSES[int(np.argmax(p))])
    P = S.fit(X, y).predict_proba(feats(test, qids, ex, cfg))
    hper = collections.Counter(); htot = collections.Counter()
    for o, p in zip(test, P):
        htot[o["qid"]] += 1; hper[o["qid"]] += S.right(o, S.CLASSES[int(np.argmax(p))])
    print(f"{cfg:10s}: training CV {sum(per.values()) / len(train):.2%} | held-out {sum(hper.values()) / len(test):.2%}")
    print("   per question CV:", ", ".join(f"{q} {per[q] / tot[q]:.0%}" for q in qids))
    print("   per question HO:", ", ".join(f"{q} {hper[q] / htot[q]:.0%}" for q in qids))
