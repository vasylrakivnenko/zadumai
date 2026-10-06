"""Early per-clause round (sched_e.py): out-of-fold accuracy on folds 0 + 1 only (~148 NDAs), next to the whole-stream
CNNs on the same NDAs. usage: python report_e.py"""
import json, os, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Q = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
     "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
gold = {(d["doc"], Q[k]): y for d in pickle.load(open(f"{HERE}/data/train.pkl", "rb")) for k, y in enumerate(d["y"].tolist())}
for n in ("plain", "rising", "rising+k5", "updown", "clause+k5", "clause+across4+k5", "clausetok+k5"):
    per = []
    for f in (0, 1):
        p = f"{HERE}/preds/{n}_f{f}.jsonl"
        if not os.path.exists(p): per.append(None); continue
        r = [int(np.argmax(o["p"])) == gold[tuple(o["id"].split("/"))] for o in map(json.loads, open(p))]
        per.append(r)
    if any(x is None for x in per): print(f"  {n:20s} (not done)"); continue
    print(f"  {n:20s} folds 0+1: {np.mean(per[0] + per[1]):.2%}   (fold 0 {np.mean(per[0]):.1%}, fold 1 {np.mean(per[1]):.1%})")
