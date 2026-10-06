"""Hyperparameters for the NDA stacker by 5-fold CV on the tuning NDAs only (folds by document)."""
import zlib, numpy as np, xgboost as xgb
import xgb_stack as S, xgb_full as F
train = S.load("nda_tune", "ft"); ex = F.extra()
qids = sorted({o["qid"] for o in train})
X = F.features(train, qids, ex, True, True); y = np.array([S.target(o) for o in train])
folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
for depth in (3, 4, 6):
    for n in (150, 300, 600):
        for lr in (0.03, 0.08):
            ok = 0
            for f in range(5):
                m = xgb.XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.9, colsample_bytree=0.8,
                                      objective="multi:softprob", num_class=3, n_jobs=4, verbosity=0, min_child_weight=2)
                m.fit(X[folds != f], y[folds != f]); P = m.predict_proba(X[folds == f])
                ok += sum(S.right(o, S.CLASSES[int(np.argmax(p))]) for o, p in zip([o for o, k in zip(train, folds) if k == f], P))
            print(f"depth {depth} trees {n} lr {lr}: CV {ok / len(train):.1%}", flush=True)
