"""Does more labeled data still help? TF-IDF + LR trained on growing slices of the training data, all scored on
the same eval rows as bench.py stage 2 (2,000). usage: learning_curve.py ledgar|cuad_clause|tos"""
import sys, time, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import bench
from sklearn.preprocessing import MultiLabelBinarizer

name = sys.argv[1]
S = bench.SETS[name](); labels = S["labels"]; multi = S["multi"]
ev = bench.sample(S["test"], 2000, multi=multi)
sizes = {"ledgar": (1000, 3000, 6000, 20000, 60000), "cuad_clause": (1500, 3000, 6000, 12000, 37616),
         "tos": (1000, 2500, 5532)}[name]
print(f"== {name}: eval {len(ev)} rows; train sizes {sizes}")
for n in sizes:
    t0 = time.time()
    tr = bench.sample(S["train"], n, seed=1, multi=multi)
    if multi:
        Y = MultiLabelBinarizer(classes=labels).fit_transform([r["labels"] for r in tr])
    else:
        Y = np.array([r["labels"][0] for r in tr])
    m = bench.Tfidf().fit([r["text"] for r in tr], Y, multi, labels)
    sc = m.scores([r["text"] for r in ev])
    if multi:
        vs = bench.sample(S["val"], 2000, multi=True); vsc = m.scores([r["text"] for r in vs])
        th = max((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: bench._f1(vs, [[labels[j] for j in np.where(s >= t)[0]] for s in vsc]))
        f1 = bench._f1(ev, [[labels[j] for j in np.where(s >= th)[0]] for s in sc])
        npos = sum(bool(r["labels"]) for r in tr)
        print(f"   train {len(tr):6} ({npos} with a type): micro-F1 {f1:.3f} (threshold {th})  [{time.time() - t0:.0f}s]", flush=True)
    else:
        cls = list(m.m.classes_); pred = [cls[i] for i in sc.argmax(1)]; gold = [r["labels"][0] for r in ev]
        acc = np.mean([p == g for p, g in zip(pred, gold)])
        mrg = np.mean([bench.MERGE.get(p, p) == bench.MERGE.get(g, g) for p, g in zip(pred, gold)])
        conf = sc.max(1); k = conf >= 0.9
        print(f"   train {len(tr):6}: accuracy {acc:.1%}, merged labels {mrg:.1%}; at p>=0.9: coverage {k.mean():.1%} "
              f"precision {np.mean([p == g for p, g, kk in zip(pred, gold, k) if kk]):.1%}  [{time.time() - t0:.0f}s]", flush=True)
