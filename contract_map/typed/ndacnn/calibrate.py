"""Per-question calibration of the CNN's answers (2026-10-05; idea #4): for each question, weights on p(yes) and
p(not stated) against p(no) (grid 0.4-2.5) chosen to maximise accuracy, then argmax. Honest by nesting: the weights for
fold F are fitted on the other four folds' out-of-fold predictions; for held-out, on all five.
usage: python calibrate.py PRED_NAME"""
import itertools, json, pickle, sys
import numpy as np
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
Q = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
     "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]
gold = {(d["doc"], Q[k]): y for s in ("train", "ho") for d in pickle.load(open(f"{HERE}/data/{s}.pkl", "rb")) for k, y in enumerate(d["y"].tolist())}
name = sys.argv[1]
P = {f: {tuple(o["id"].split("/")): np.array(o["p"]) for o in map(json.loads, open(f"{HERE}/preds/{name}_f{f}.jsonl"))} for f in (0, 1, 2, 3, 4, -1)}


def fit(rows):  # rows: [(q, p, y)] -> {q: w}
    out = {}
    for q in Q:
        rq = [(p, y) for qq, p, y in rows if qq == q]
        out[q] = max(((1.0, wy, wn) for wy, wn in itertools.product(GRID, GRID)),
                     key=lambda w: (sum(int(np.argmax(p * w)) == y for p, y in rq), -abs(np.log(w[1])) - abs(np.log(w[2]))))
    return out


def acc(rows, W):
    return np.mean([int(np.argmax(p * (W[q] if W else 1))) == y for q, p, y in rows])


rows = {f: [(k[1], p, gold[k]) for k, p in P[f].items()] for f in P}
raw, cal = [], []
for f in range(5):
    W = fit([r for g in range(5) if g != f for r in rows[g]])
    raw += [int(np.argmax(p)) == y for _, p, y in rows[f]]; cal += [int(np.argmax(p * W[q])) == y for q, p, y in rows[f]]
Wall = fit([r for g in range(5) for r in rows[g]])
print(f"{name}: out-of-fold {np.mean(raw):.2%} -> calibrated {np.mean(cal):.2%} | held-out {acc(rows[-1], None):.2%} -> {acc(rows[-1], Wall):.2%}")
print("weights (yes, not stated vs no) that moved:", {q: (w[1], w[2]) for q, w in Wall.items() if w != (1.0, 1.0, 1.0)})
