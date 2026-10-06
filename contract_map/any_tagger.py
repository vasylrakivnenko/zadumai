"""A clause tagger that doesn't need the document's kind, for short texts (a clause or a few) (2026-10-03).
One TF-IDF + one-vs-rest LR over all six datasets' classes (combined.py's label map). A row teaches class c if its
dataset marks c completely, if it's a positive for c, or if c is another kind's own class (a privacy-policy type is
not in an NDA clause): cross-kind negatives, so a type fires only on its own kind's wording. Shared (U:) classes get
no cross-kind negatives (governing law is in every kind).
Measured: each dataset's own test rows (vs its own model, same thresholds procedure) and false tags of other kinds'
own classes on every dataset's test rows. --save writes models/contract_map/tagger_any.joblib.
usage: any_tagger.py [--stage 2|3] [--save]"""
import argparse, collections, json, os, sys
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/bench")
import joblib
import bench, family
from combined import DATASETS, score_class, row_classes
from build_models import compact, scores, f1

KIND = family.KIND
OUT = "/root/projects/zadumai/legalbench_map/models/contract_map"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=2); ap.add_argument("--save", action="store_true"); a = ap.parse_args()
    n_eval, n_train = bench.STAGES[a.stage]
    S = {ds: bench.SETS[ds]() for ds in DATASETS}
    tr = {ds: bench.sample(S[ds]["train"], n_train, seed=1, multi=S[ds]["multi"]) for ds in DATASETS}
    full = {ds: {score_class(ds, l) for l in S[ds]["labels"]} for ds in DATASETS}
    own_of_kind = collections.defaultdict(set)
    for ds in DATASETS:
        own_of_kind[KIND[ds]] |= {c for c in full[ds] if not c.startswith("U:")}
    items = []
    for ds in DATASETS:
        neg = full[ds] | set().union(*(v for k, v in own_of_kind.items() if k != KIND[ds]))
        items += [(r["text"], row_classes(ds, r["labels"]), neg) for r in tr[ds]]
    print(f"== any-kind tagger, stage {a.stage}: {len(items)} training rows", flush=True)
    vec, classes, W, b = compact(family.train_items(items, max_features=100_000))
    ci = {c: i for i, c in enumerate(classes)}
    th_by_ds, report = {}, []
    for ds in DATASETS:
        D = S[ds]; cls = sorted(full[ds]); cols = [ci[c] for c in cls]
        ev = bench.sample(D["test"], n_eval, multi=D["multi"]); G = [{score_class(ds, l) for l in r["labels"]} for r in ev]
        St = scores(vec, W, b, [r["text"] for r in ev])
        if D["multi"]:
            val = bench.sample(D["val"], 2000, multi=True); Gv = [{score_class(ds, l) for l in r["labels"]} for r in val]
            Sv = scores(vec, W, b, [r["text"] for r in val])[:, cols]
            th = max((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: f1(Gv, [{cls[j] for j in np.where(s >= t)[0]} for s in Sv]))
            own = f"F1 {f1(G, [{cls[j] for j in np.where(s >= th)[0]} for s in St[:, cols]]):.3f}"
        else:
            th = 0.5; P = [{cls[int(s.argmax())]} if s.max() >= th else set() for s in St[:, cols]]
            own = f"{np.mean([bool(p & g) for p, g in zip(P, G) if p]):.1%} right on {np.mean([bool(p) for p in P]):.0%} tagged"
        th_by_ds[ds] = th
        other = [ci[c] for k, v in own_of_kind.items() if k != KIND[ds] for c in v]
        false = np.mean((St[:, other] >= 0.7).any(1))
        report.append((ds, own, false, th))
    for ds, own, false, th in report:
        print(f"   {ds:12} own classes: {own} (threshold {th}) | rows given another kind's type (p >= 0.7): {false:.1%}", flush=True)
    if a.save:
        cls_th = {}
        for ds in DATASETS:
            for c in full[ds]: cls_th[c] = max(cls_th.get(c, 0), th_by_ds[ds])
        labels = {}
        for ds in DATASETS:
            for l in S[ds]["labels"]: labels.setdefault(score_class(ds, l), l)
        joblib.dump({"vec": vec, "classes": classes, "W": W, "b": b, "thresholds": [max(0.7, cls_th.get(c, 0.7)) for c in classes],
                     "labels": labels, "kind_of": {c: KIND[ds] for ds in DATASETS for c in full[ds] if not c.startswith("U:")}},
                    f"{OUT}/tagger_any.joblib", compress=3)
        m = json.load(open(f"{OUT}/meta.json")); m["tagger_any"] = {"stage": a.stage, "report": [list(map(str, r)) for r in report]}
        json.dump(m, open(f"{OUT}/meta.json", "w"), indent=1); print("saved tagger_any.joblib")


if __name__ == "__main__":
    main()
