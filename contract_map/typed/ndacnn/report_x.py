"""The clause-vector batch (2026-10-05): 7 variants x 3 seeds (runs/gpu4/preds), each averaged over its seeds, against the
same-data baseline (96 channels, 3 seeds, preds/). Out-of-fold on the 384 training NDAs and held-out (60).
usage: python report_x.py"""
import json, os, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Q = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
     "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
gold = {(d["doc"], Q[k]): y for s in ("train", "ho") for d in pickle.load(open(f"{HERE}/data/{s}.pkl", "rb")) for k, y in enumerate(d["y"].tolist())}
B = "rising+secallw1.0+d128+k5"


def acc(names, folds, d):
    out = {}
    for f in folds:
        ps = [{o["id"]: o["p"] for o in map(json.loads, open(f"{d}/{n}_f{f}.jsonl"))} for n in names]
        for i in ps[0]: out[i] = np.mean([p[i] for p in ps], 0)
    return np.mean([int(np.argmax(p)) == gold[tuple(i.split("/"))] for i, p in out.items()])


G = f"{HERE}/runs/gpu4/preds"
rows = [("baseline: 96 channels", [f"{B}+r2" + s for s in ("", "+s1", "+s2")], f"{HERE}/preds"),
        ("A: 192 channels", [f"{B}+r2" + s for s in ("", "+s1", "+s2")], G)]
for lab, x in (("B: + bge", "xbge"), ("C1: + TF-IDF loose (2 / 0.9)", "xtfidf1"), ("C2: + TF-IDF moderate (5 / 0.7)", "xtfidf2"),
               ("C3: + TF-IDF tight (20 / 0.5)", "xtfidf3"), ("D: + bge + TF-IDF moderate", "xbge-tfidf2"),
               ("E: D + kernel across clauses", "xbge-tfidf2+ast1")):
    rows.append((lab, [f"{B}+{x}+r2" + s for s in ("", "+s1", "+s2")], G))
for lab, names, d in rows:
    if not all(os.path.exists(f"{d}/{n}_f{f}.jsonl") for n in names for f in (0, 1, 2, 3, 4, -1)):
        print(f"{lab:36s} (incomplete)"); continue
    single = [acc([n], range(5), d) for n in names]
    print(f"{lab:36s} 3-seed average: OOF {acc(names, range(5), d):.2%} | held-out {acc(names, [-1], d):.2%} | single seeds OOF "
          + " ".join(f"{x:.1%}" for x in single))
