"""Held-out check (2026-10-04): the NDA stacker with and without Jev v1's numbers, trained on the 384 training NDAs
(Jev asked on 100 of them, missing for the rest), scored once on nda_ho (60 NDAs, Jev asked on all).
usage: venv_xgb python jev_ho.py"""
import collections, json, os
import numpy as np
import xgb_stack as S, xgb_full as F, nda_joint as J
HERE = os.path.dirname(os.path.abspath(__file__))
jev = {}
for s in ("nda_tune", "nda_ho"): jev.update({json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/jev_{s}.jsonl")})
train = J.load_train(); test = S.load("nda_ho", ""); qids = sorted({o["qid"] for o in train}); ex = F.extra()
CH = {"yes": "yes", "no": "no", "not stated": "not_stated"}
jf = lambda o: [jev[o["id"]]["p_true"]] + [jev[o["id"]]["probs"].get(k, np.nan) for k in ("yes", "no", "not stated")] if o["id"] in jev else [np.nan] * 4
y = np.array([S.target(o) for o in train])
print(f"held-out: {len(test)} questions, Jev on {sum(o['id'] in jev for o in test)}; gold {dict(collections.Counter(o['gold'] for o in test))}")
print(f"Jev v1 choice alone: {sum(CH.get(jev[o['id']]['choice']) == o['gold'] for o in test) / len(test):.2%}")
res = {}
for name, use in (("stacker", False), ("stacker + Jev", True)):
    Xtr, Xte = F.features(train, qids, ex, True, True), F.features(test, qids, ex, True, True)
    if use: Xtr, Xte = np.hstack([Xtr, np.array([jf(o) for o in train])]), np.hstack([Xte, np.array([jf(o) for o in test])])
    P = S.fit(Xtr, y).predict_proba(Xte); res[name] = [S.CLASSES[int(np.argmax(p))] for p in P]
    r = [S.right(o, a) for o, a in zip(test, res[name])]
    docs = collections.defaultdict(list)
    for o, k in zip(test, r): docs[o["id"].split("/")[0]].append(k)
    m = np.array([np.mean(v) for v in docs.values()]); rng = np.random.default_rng(0)
    bs = sorted(np.mean(rng.choice(m, len(m))) for _ in range(2000))
    print(f"{name:14s}: held-out {np.mean(r):.2%}  (95% bootstrap over NDAs {bs[50]:.1%}-{bs[1949]:.1%})")
a, b = res["stacker"], res["stacker + Jev"]
print(f"changed {sum(x != z for x, z in zip(a, b))}: fixed {sum(z == o['gold'] != x for o, x, z in zip(test, a, b))}, broke {sum(x == o['gold'] != z for o, x, z in zip(test, a, b))}")
print("per question:", ", ".join(f"{q} {np.mean([S.right(o, x) for o, x in zip(test, a) if o['qid'] == q]):.0%}->{np.mean([S.right(o, z) for o, z in zip(test, b) if o['qid'] == q]):.0%}" for q in qids))
