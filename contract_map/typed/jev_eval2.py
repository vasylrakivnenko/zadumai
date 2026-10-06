"""Jev v1 (jev_nda.py: the user-style question; noul + a 3-way choice) vs v2 (jev_nda2.py: ContractNLI's statement,
two Nouls "says yes" / "says no") on the same 100 NDAs x 17 questions, alone and as features of our stacker (2026-10-04).
  v2 rule: not stated if max(p_yes, p_no) < T, else the larger; T per question, chosen by 5-fold CV over the NDAs
  stacker + Jev: 5-fold CV over the 384 training NDAs (Jev missing for 284 of them), scored on these 100
usage: venv_xgb python jev_eval2.py"""
import collections, json, os, zlib
import numpy as np
import xgb_stack as S, xgb_full as F, nda_joint as J
HERE = os.path.dirname(os.path.abspath(__file__))
v1 = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/jev_nda_tune.jsonl")}
v2 = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/jev2_nda_tune.jsonl")}
train = J.load_train(); qids = sorted({o["qid"] for o in train}); ex = F.extra()
rows = [o for o in train if o["id"] in v1 and o["id"] in v2]
dfold = {o["id"]: zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in rows}
CH = {"yes": "yes", "no": "no", "not stated": "not_stated"}
acc = lambda pred: sum(pred[o["id"]] == o["gold"] for o in rows) / len(rows)
print(f"{len(rows)} questions, {len({o['id'].split('/')[0] for o in rows})} NDAs")

yn = [o for o in rows if o["gold"] != "not_stated"]
print(f"binary on gold yes/no ({len(yn)}): v1 p(true)>=.5 {sum((v1[o['id']]['p_true'] >= .5) == (o['gold'] == 'yes') for o in yn) / len(yn):.1%} | "
      f"v2 p_yes>=p_no {sum((v2[o['id']]['p_yes'] >= v2[o['id']]['p_no']) == (o['gold'] == 'yes') for o in yn) / len(yn):.1%}")


def rule(o, t):
    j = v2[o["id"]]
    return "not_stated" if max(j["p_yes"], j["p_no"]) < t else "yes" if j["p_yes"] >= j["p_no"] else "no"


TS = [x / 100 for x in range(10, 91, 5)]
pred_rule, pred_g = {}, {}
for f in range(5):
    tr = [o for o in rows if dfold[o["id"]] != f]
    tg = max(TS, key=lambda t: sum(rule(o, t) == o["gold"] for o in tr))
    for q in qids:
        tq = max(TS, key=lambda t: sum(rule(o, t) == o["gold"] for o in tr if o["qid"] == q))
        for o in rows:
            if dfold[o["id"]] == f and o["qid"] == q: pred_rule[o["id"]] = rule(o, tq)
    for o in rows:
        if dfold[o["id"]] == f: pred_g[o["id"]] = rule(o, tg)
res = {"v1 choice": {o["id"]: CH.get(v1[o["id"]]["choice"]) for o in rows}, "v2 rule, one T": pred_g, "v2 rule, T per question": pred_rule}

y = np.array([S.target(o) for o in train]); folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
f1 = lambda o: [v1[o["id"]]["p_true"]] + [v1[o["id"]]["probs"].get(k, np.nan) for k in ("yes", "no", "not stated")] if o["id"] in v1 else [np.nan] * 4
f2 = lambda o: [v2[o["id"]]["p_yes"], v2[o["id"]]["p_no"]] if o["id"] in v2 else [np.nan] * 2
base = F.features(train, qids, ex, True, True)
for name, add in (("stacker", []), ("stacker + v1", [f1]), ("stacker + v2", [f2]), ("stacker + v1 + v2", [f1, f2])):
    X = np.hstack([base] + [np.array([g(o) for o in train]) for g in add]) if add else base
    oof = np.zeros((len(train), 3))
    for f in range(5): oof[folds == f] = S.fit(X[folds != f], y[folds != f]).predict_proba(X[folds == f])
    res[name] = {o["id"]: S.CLASSES[int(np.argmax(p))] for o, p in zip(train, oof) if o["id"] in v2}
for name, g in (("v2 numbers + question id", [f2]), ("v1 + v2 numbers + question id", [f1, f2])):
    X = np.array([[float(o["qid"] == q) for q in qids] + sum((h(o) for h in g), []) for o in rows]); yy = np.array([S.target(o) for o in rows])
    ff = np.array([dfold[o["id"]] for o in rows]); oof = np.zeros((len(rows), 3))
    for f in range(5): oof[ff == f] = S.fit(X[ff != f], yy[ff != f]).predict_proba(X[ff == f])
    res[name] = {o["id"]: S.CLASSES[int(np.argmax(p))] for o, p in zip(rows, oof)}
print("\n3-way accuracy, all 1,700 questions:")
for k, v in res.items(): print(f"  {k:32s} {acc(v):.1%}")
print("\nper question:", " | ".join(("v1 choice", "v2 rule", "stacker", "stk+v1", "stk+v2", "stk+v1+v2")))
for q in qids:
    rq = [o for o in rows if o["qid"] == q]
    print(f"  {q:7s}", " ".join(f"{sum(res[k][o['id']] == o['gold'] for o in rq) / len(rq):6.0%}"
                                 for k in ("v1 choice", "v2 rule, T per question", "stacker", "stacker + v1", "stacker + v2", "stacker + v1 + v2")))
for k in ("v1 choice", "v2 rule, T per question"):
    print(f"\n{k} confusion (gold rows yes/no/not_stated; columns yes/no/not_stated):")
    for g in ("yes", "no", "not_stated"):
        c = collections.Counter(res[k][o["id"]] for o in rows if o["gold"] == g); print(f"  {g:10s}", [c.get(x, 0) for x in ("yes", "no", "not_stated")])
