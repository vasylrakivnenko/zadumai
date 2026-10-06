"""Threshold-free comparison of the tagging runs (runs/oof_*.npy): per clause type, rank contracts by their best unit's
score; average precision of 'the best unit holds the clause' (AP, mean over types with >= 10 present), and recall at
equal precision per type (the most found with >= P right in the ranking, summed over types). Same units as eval_tagging."""
import collections, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import eval_tagging as E

SETS = {"cuad": lambda: E.split(), "cnli": E.cnli_split}
CONFIGS = ["live_text", "leaf_text", "leaf_head-in-text", "headed_text", "headed_head-in-text"]


def main(name):
    data = SETS[name](); types = sorted({t for d in data for t in d["gold"] if t not in E.META}); units = {}
    for cfg in CONFIGS:
        mode = cfg.split("_")[0]
        if mode not in units:
            with ProcessPoolExecutor(8) as ex: units[mode] = {d["id"]: d for d in ex.map(E.prep, [(x, mode) for x in data], chunksize=4)}
        oof = collections.defaultdict(dict)
        for row in np.load(f"runs/oof_{name}_{cfg}.npy", allow_pickle=True): oof[row[0]][int(row[1])] = np.array(row[2:], dtype=float)
        aps, rec = [], collections.Counter()
        for j, t in enumerate(types):
            rows = []
            for did, d in units[mode].items():
                if not d["units"]: continue
                ks = [oof[did][k][j] for k in range(len(d["units"]))]; k = int(np.argmax(ks))
                rows.append((ks[k], t in d["units"][k]["labels"]))
            rows.sort(key=lambda r: -r[0]); pos = sum(h for _, h in rows)
            if pos < 10: continue
            ok = 0; ap = 0.0; best = collections.Counter()
            for n, (_, h) in enumerate(rows, 1):
                ok += h
                if h: ap += ok / n
                for P in (0.9, 0.95, 0.98):
                    if ok / n >= P: best[P] = ok
            aps.append(ap / pos); rec.update(best)
        present = sum(bool(d["gold"].get(t)) for d in data for t in types)
        print(f"{name:5s} {cfg:22s} mean AP {np.mean(aps):.3f} ({len(aps)} types) | found at equal precision per type: "
              + ", ".join(f"{int(P*100)}% -> {rec[P]/present:.1%}" for P in (0.9, 0.95, 0.98)), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["cuad", "cnli"]: main(n)
