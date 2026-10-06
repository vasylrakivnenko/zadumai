"""Out-of-fold "yes" precision at the top of the span model's ranking, per question (5-fold over the 484 NDAs)."""
import os, random, sys
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_mentioned import load
from cv_engine import model_scores, Q
docs = load(); idx = list(range(len(docs))); random.Random(0).shuffle(idx); folds = [0] * len(docs)
for r, i in enumerate(idx): folds[i] = r % 5
def run(q):
    s = np.zeros(len(docs))
    for f in range(5):
        tr = [d for d, k in zip(docs, folds) if k != f]; te = [i for i, k in enumerate(folds) if k == f]
        s[te] = [x for x, _ in model_scores(tr, [docs[i] for i in te], q)]
    g = np.array([d["gold"][q] for d in docs]); order = np.argsort(-s)
    out = []
    for k in (25, 50, 100, 200):
        top = g[order[:k]]; out.append(f"top{k}: E {np.mean(top == 'Entailment'):.0%} (N {np.sum(top == 'NotMentioned')}, C {np.sum(top == 'Contradiction')})")
    return q, out
for q, out in Parallel(n_jobs=8)(delayed(run)(q) for q in Q):
    print(f"{q:6}", " | ".join(out))
