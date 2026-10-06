"""Jev on 100 NDAs x 17 questions (jev_nda.py) against ContractNLI's gold and against our stacker's out-of-fold answers on
the same 1,700 questions (2026-10-04).
  binary   p(true) >= 0.5 on the questions whose gold is yes / no (what a true / false read can get right)
  3-way    (a) Jev's choice yes / no / not stated; (b) p(true) with two cut-offs (>= hi yes, <= lo no, else not
           stated), cut-offs chosen by 5-fold CV over the NDAs; (c) our stacker; (d) our stacker + Jev's numbers as
           features (5-fold CV over the 384 training NDAs, Jev missing for the other 284; scored on the 100)
usage: venv_xgb python jev_eval.py"""
import collections, json, os, zlib
import numpy as np
import xgb_stack as S, xgb_full as F, nda_joint as J
HERE = os.path.dirname(os.path.abspath(__file__))
jev = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/jev_nda_tune.jsonl")}
train = J.load_train(); qids = sorted({o["qid"] for o in train}); ex = F.extra()
rows = [o for o in train if o["id"] in jev]
print(f"{len(rows)} questions on {len({o['id'].split('/')[0] for o in rows})} NDAs; gold {dict(collections.Counter(o['gold'] for o in rows))}")
CH = {"yes": "yes", "no": "no", "not stated": "not_stated"}

yn = [o for o in rows if o["gold"] != "not_stated"]
print(f"binary (gold yes/no only, {len(yn)}): p(true) >= 0.5 right {sum((jev[o['id']]['p_true'] >= 0.5) == (o['gold'] == 'yes') for o in yn) / len(yn):.1%}")
print(f"3-way, Jev's choice: {sum(CH.get(jev[o['id']]['choice']) == o['gold'] for o in rows) / len(rows):.1%}")


def cut(p, lo, hi):
    return "yes" if p >= hi else "no" if p <= lo else "not_stated"


grid = [(lo / 100, hi / 100) for lo in range(5, 55, 5) for hi in range(50, 100, 5) if lo < hi]
dfold = {o["id"]: zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in rows}
right = 0
for f in range(5):
    tr = [o for o in rows if dfold[o["id"]] != f]; te = [o for o in rows if dfold[o["id"]] == f]
    lo, hi = max(grid, key=lambda g: sum(cut(jev[o["id"]]["p_true"], *g) == o["gold"] for o in tr))
    right += sum(cut(jev[o["id"]]["p_true"], lo, hi) == o["gold"] for o in te)
print(f"3-way, p(true) with cut-offs (CV): {right / len(rows):.1%}")

y = np.array([S.target(o) for o in train]); folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
on = np.array([o["id"] in jev for o in train])


def jf(o):
    j = jev.get(o["id"])
    if not j: return [np.nan] * 4
    return [j["p_true"]] + [j["probs"].get(k, np.nan) for k in ("yes", "no", "not stated")]


res = {}
for name, use in (("stacker", False), ("stacker + Jev", True)):
    X = F.features(train, qids, ex, True, True)
    if use: X = np.hstack([X, np.array([jf(o) for o in train])])
    oof = np.zeros((len(train), 3))
    for f in range(5): oof[folds == f] = S.fit(X[folds != f], y[folds != f]).predict_proba(X[folds == f])
    res[name] = {o["id"]: S.CLASSES[int(np.argmax(p))] for o, p in zip(train, oof) if o["id"] in jev}
    print(f"3-way, {name} (out-of-fold, these 100 NDAs): {sum(res[name][o['id']] == o['gold'] for o in rows) / len(rows):.1%}")
X = np.array([[float(o["qid"] == q) for q in qids] + jf(o) for o in rows]); yy = np.array([S.target(o) for o in rows])
ff = np.array([dfold[o["id"]] for o in rows]); oof = np.zeros((len(rows), 3))
for f in range(5): oof[ff == f] = S.fit(X[ff != f], yy[ff != f]).predict_proba(X[ff == f])
res["Jev numbers + question id"] = {o["id"]: S.CLASSES[int(np.argmax(p))] for o, p in zip(rows, oof)}
print(f"3-way, a model on Jev's numbers + the question id only (CV over these 100): "
      f"{sum(res['Jev numbers + question id'][o['id']] == o['gold'] for o in rows) / len(rows):.1%}")

print("\nper question (3-way): Jev choice | stacker | stacker + Jev")
for q in qids:
    rq = [o for o in rows if o["qid"] == q]
    print(f"  {q:7s} {sum(CH.get(jev[o['id']]['choice']) == o['gold'] for o in rq) / len(rq):5.0%} | "
          f"{sum(res['stacker'][o['id']] == o['gold'] for o in rq) / len(rq):5.0%} | {sum(res['stacker + Jev'][o['id']] == o['gold'] for o in rq) / len(rq):5.0%}")
print("\nJev choice confusion (gold rows yes/no/not_stated; columns yes/no/not stated):")
for g in ("yes", "no", "not_stated"):
    c = collections.Counter(CH.get(jev[o["id"]]["choice"]) for o in rows if o["gold"] == g)
    print(f"  {g:10s}", [c.get(k, 0) for k in ("yes", "no", "not_stated")])
both = sum(CH.get(jev[o["id"]]["choice"]) == o["gold"] == res["stacker"][o["id"]] for o in rows)
print(f"\nboth right {both}, only Jev {sum(CH.get(jev[o['id']]['choice']) == o['gold'] != res['stacker'][o['id']] for o in rows)}, "
      f"only stacker {sum(res['stacker'][o['id']] == o['gold'] != CH.get(jev[o['id']]['choice']) for o in rows)}")
