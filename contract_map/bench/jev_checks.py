"""Small Jev checks (2026-10-02; the user's cap: about 1,000 rows per task).
  router  - Jev picks the document kind for the 500 router test documents of family.py stage 1, vs the TF-IDF
            router, and a hybrid (TF-IDF when p >= 0.9, else Jev). 500 Jev calls.
  hybrid  - per clause dataset, 1,000 of the stage-2 eval rows: TF-IDF (own model, all training rows) when it's
            sure, Jev for the unsure rows. Jev's answers come from the stage-2 cache (bench.py), so no new calls.
usage: jev_checks.py router|hybrid"""
import collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench, combined, family
from combined import DATASETS, score_class, row_classes

OUT = bench.OUT
KIND_DESC = {"commercial": "business contract between companies (supply, services, licence, employment, loan, merger...)",
             "nda": "non-disclosure / confidentiality agreement", "privacy policy": "privacy policy or privacy notice",
             "terms of service": "terms of service / terms of use / cookie policy / end-user licence for an online service",
             "lease": "lease of premises or property", "not a contract": "not a contract (a letter, plan, certificate, other document)"}


def router():
    S = {ds: bench.SETS[ds]() for ds in DATASETS}
    docs = family.clause_docs(); rtr, rte = family.router_data(S, docs)
    rte = [rte[i] for i in sorted(random.Random(0).sample(range(len(rte)), 500))]  # = family.py stage 1
    route = family.train_router(rtr); cl, P = route([d["text"] for d in rte]); tf = cl[P.argmax(1)]; conf = P.max(1)
    J = bench.Jev("router", "head", 8)
    recs = J.run(rte, [d["text"][:4000] for d in rte], KIND_DESC, "What kind of legal document is this?")
    jv = np.array([r["choice"] for r in recs], dtype=object); gold = np.array([d["label"] for d in rte], dtype=object)
    lat = sorted(r["ms"] for r in recs if not r["err"])
    print(f"router, 500 test documents: TF-IDF {np.mean(tf == gold):.1%} | Jev {np.mean(jv == gold):.1%} "
          f"(errors {sum(1 for r in recs if r['err'])}, p50 {lat[len(lat) // 2]:.0f} ms)")
    for t in (0.7, 0.9, 0.95):
        hy = np.where(conf >= t, tf, jv)
        print(f"   hybrid: TF-IDF if p >= {t}, else Jev: {np.mean(hy == gold):.1%} (Jev reads {np.mean(conf < t):.0%})")
    print("   Jev confusions:", "; ".join(f"{g} -> {p} ×{n}" for (g, p), n in collections.Counter(
        (g, p) for g, p in zip(gold, jv) if g != p).most_common(6)))


def hybrid():
    for ds in DATASETS:
        S = bench.SETS[ds](); multi = S["multi"]
        ev = bench.sample(S["test"], 2000, multi=multi); ev = [ev[i] for i in sorted(random.Random(0).sample(range(len(ev)), min(1000, len(ev))))]
        cache = {}
        for line in open(f"{OUT}/jev_cache_{ds}_text.jsonl"):
            r = json.loads(line); cache[r["id"]] = r
        ev = [r for r in ev if r["id"] in cache and not cache[r["id"]]["err"]]
        full = {score_class(ds, l) for l in S["labels"]}
        model = family.train_items([(r["text"], row_classes(ds, r["labels"]), full) for r in S["train"]])
        cls = sorted(full); sc = family.predict(model, ev, cls)
        gold = [{score_class(ds, l) for l in r["labels"]} for r in ev]
        jv = [set() if cache[r["id"]]["choice"] in (None, "NONE") else {score_class(ds, cache[r["id"]]["choice"])} for r in ev]
        if not multi:
            tf = [{cls[j]} for j in sc.argmax(1)]; conf = sc.max(1)
            acc = lambda P: np.mean([bool(p & g) for p, g in zip(P, gold)])
            line = f"{ds:12} n {len(ev)}: TF-IDF {acc(tf):.1%} | Jev {acc(jv):.1%}"
            for t in (0.5, 0.7, 0.9):
                line += f" | p<{t}→Jev: {acc([j if c < t else f for f, j, c in zip(tf, jv, conf)]):.1%} (Jev {np.mean(conf < t):.0%})"
            print(line, flush=True); continue
        va = bench.sample(S["val"], 2000, multi=True); vsc = family.predict(model, va, cls)
        vgold = [{score_class(ds, l) for l in r["labels"]} for r in va]
        def f1(G, P):
            tp = sum(len(g & p) for g, p in zip(G, P)); fp = sum(len(p - g) for g, p in zip(G, P)); fn = sum(len(g - p) for g, p in zip(G, P))
            return 2 * tp / max(2 * tp + fp + fn, 1)
        th = max((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: f1(vgold, [{cls[j] for j in np.where(s >= t)[0]} for s in vsc]))
        tf = [{cls[j] for j in np.where(s >= th)[0]} for s in sc]; mx = sc.max(1)
        line = f"{ds:12} n {len(ev)}: TF-IDF F1 {f1(gold, tf):.3f} | Jev {f1(gold, jv):.3f}"
        for lo, hi in ((th - 0.2, th + 0.2), (th - 0.3, 1.01)):
            band = (mx >= lo) & (mx < hi)
            line += f" | unsure [{lo:.1f},{hi:.1f})→Jev: {f1(gold, [j if b else f for f, j, b in zip(tf, jv, band)]):.3f} (Jev {band.mean():.0%})"
        print(line, flush=True)


if __name__ == "__main__":
    {"router": router, "hybrid": hybrid}[sys.argv[1]]()
