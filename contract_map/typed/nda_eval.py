"""The NDA stacker with and without the symbolic detectors (nda_rules.py): 5-fold CV on the 384 training NDAs (folds by
NDA) per question, and held-out (60 NDAs). Installed-network features (the retrained network learned from the extra NDAs).
usage: venv_xgb python nda_eval.py [RULE_PREFIX ...]   (e.g. r16 r20; default: all detector flags)"""
import collections, json, os, sys, zlib
import numpy as np
import xgb_stack as S, xgb_full as F, nda_joint as J
HERE = os.path.dirname(os.path.abspath(__file__))
rules = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/nda_rules.jsonl")}
pref = sys.argv[1:] or None
keys = sorted({k for r in rules.values() for k in r if k != "id" and (pref is None or any(k.startswith(p) for p in pref))})


def feats(rows, qids, ex, use_rules):
    X = F.features(rows, qids, ex, True, True)
    if not use_rules: return X
    # a detector speaks only on its own question (r7_* -> nda-7); elsewhere missing (GATE=0: everywhere)
    gate = os.environ.get("GATE", "1") == "1"
    R = np.array([[float(bool(rules.get(o["id"], {}).get(k))) if not gate or o["qid"] == "nda-" + k.split("_")[0][1:] else np.nan
                   for k in keys] for o in rows])
    return np.hstack([X, R])


ex = F.extra(); train = J.load_train(); test = S.load("nda_ho", ""); qids = sorted({o["qid"] for o in train})
y = np.array([S.target(o) for o in train]); folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
for use in (False, True):
    X = feats(train, qids, ex, use); oof = np.zeros((len(train), 3))
    for f in range(5): oof[folds == f] = S.fit(X[folds != f], y[folds != f]).predict_proba(X[folds == f])
    per = collections.Counter(); tot = collections.Counter()
    for o, p in zip(train, oof):
        tot[o["qid"]] += 1; per[o["qid"]] += S.right(o, S.CLASSES[int(np.argmax(p))])
    P = S.fit(X, y).predict_proba(feats(test, qids, ex, use))
    ho = sum(S.right(o, S.CLASSES[int(np.argmax(p))]) for o, p in zip(test, P)) / len(test)
    print(f"{'with' if use else 'without'} detectors {keys if use else ''}: training CV {sum(per.values()) / len(train):.2%} | held-out {ho:.2%}")
    print("   per question CV:", ", ".join(f"{q} {per[q] / tot[q]:.1%}" for q in ("nda-1", "nda-2", "nda-7", "nda-16", "nda-17", "nda-20")))
