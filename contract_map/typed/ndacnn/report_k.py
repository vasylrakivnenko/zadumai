"""Kernel widths + learning curve (sched_k.py): each CNN's own answers, out-of-fold on the 384 training NDAs (5 folds) and
held-out (the model on all of them) where it was trained. usage: python report_k.py"""
import json, os, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Q = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
     "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
gold = {(d["doc"], Q[k]): y for s in ("train", "ho") for d in pickle.load(open(f"{HERE}/data/{s}.pkl", "rb")) for k, y in enumerate(d["y"].tolist())}


def acc(name, folds):
    r = []
    for f in folds:
        p = f"{HERE}/preds/{name}_f{f}.jsonl"
        if not os.path.exists(p): return None
        for l in open(p):
            o = json.loads(l); r.append(int(np.argmax(o["p"])) == gold[tuple(o["id"].split("/"))])
    return np.mean(r)


fmt = lambda x: f"{x:.2%}" if x is not None else "  -   "
print("CNN variants (out-of-fold | held-out):")
for n in ("plain", "plain+k13", "plain3", "rising+k3", "rising+k5", "rising", "rising+k13", "rising+d128+k5", "rising+d128+k3", "rising+jevhead+k5", "rising+jevhead+d128+k5", "pyramid", "rising+sep+k5"):
    print(f"  {n:12s} {fmt(acc(n, range(5)))} | {fmt(acc(n, [-1]))}")
print("learning curve, rising (share of the ~307 training NDAs per fold -> out-of-fold accuracy):")
for fr, n in ((0.25, "rising+frac0.25"), (0.5, "rising+frac0.5"), (0.75, "rising+frac0.75"), (1.0, "rising")):
    print(f"  {fr:4.2f} (~{int(307 * fr)} NDAs) {fmt(acc(n, range(5)))}")
