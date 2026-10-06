"""Shared taxonomy, one clause model per kind of document, routed by a document-type classifier (2026-10-02).
1. Router: TF-IDF + LR on a document's first 4,000 characters -> commercial / nda / privacy policy /
   terms of service / lease / not a contract. Trained on documents that are NOT in any clause test set.
2. Clause models per kind (commercial = LEDGAR + CUAD in one model; the other kinds have one dataset each),
   with and without borrowing: a rare class (< 100 training positives in its kind) gets up to 300 positive rows
   from other kinds' datasets that map to the same unified class (combined.py's label map).
3. End to end: each test clause goes to the model its document is routed to (CUAD, ContractNLI, OPP-115 and lease
   have whole documents; LEDGAR and UNFAIR-ToS have none, so they're scored with their own kind).
Compared with one model per dataset and one model for everything (TF-IDF + one-vs-rest LR throughout).
usage: family.py --stage 1|2|3"""
import argparse, collections, html as H, json, os, random, re, sys, time, zipfile
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench, combined
from combined import DATASETS, UNIFIED, score_class, row_classes

KIND = {"ledgar": "commercial", "cuad_clause": "commercial", "contractnli": "nda", "opp115": "privacy policy",
        "tos": "terms of service", "lease": "lease"}
KINDS = ["commercial", "nda", "privacy policy", "terms of service", "lease"]
NJ = int(os.environ.get("BENCH_JOBS", "2"))
EXT = bench.EXT
OUT = bench.OUT
BORROWED = set()


# ---------- documents ----------
def clause_docs():
    """(dataset, doc id) -> whole document text, for the clause datasets that have documents."""
    out = {}
    for d in json.load(open(f"{bench.D}/data/CUADv1.json"))["data"]:
        out[("cuad_clause", d["title"])] = d["paragraphs"][0]["context"]
    z = zipfile.ZipFile(f"{EXT}/contractnli.zip")
    for s in ("train", "dev", "test"):
        for d in json.loads(z.read(f"contract-nli/{s}.json"))["documents"]: out[("contractnli", d["id"])] = d["text"]
    z = zipfile.ZipFile(f"{EXT}/opp115.zip")
    for n in z.namelist():
        if "sanitized_policies" in n and n.endswith(".html"):
            t = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf-8", "ignore").replace("|||", "\n"))))
            out[("opp115", n.split("/")[-1].split("_")[0])] = t
    z = zipfile.ZipFile(f"{EXT}/lease_annotated.zip")
    for n in z.namelist():
        if n.endswith(".plain.html") and not n.startswith("__MACOSX"):
            h = z.read(n).decode("utf-8", "ignore")
            out[("lease", n.split("/")[-1][:-len(".plain.html")])] = "\n".join(
                re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", m))) for m in re.findall(r'<p id="[^"]+">(.*?)</p>', h, re.S))
    return out


def router_data(S, docs):
    """Training and test documents for the router. Test = the clause sets' test documents + held-out MCC and
    ToS;DR documents; training never contains a clause test document."""
    test_docs = {(ds, r["doc"]) for ds in ("cuad_clause", "contractnli", "opp115", "lease") for r in S[ds]["test"]}
    tr, te = [], []
    for (ds, did), t in docs.items():
        item = {"id": f"{ds}/{did}", "text": t, "label": KIND[ds], "src": ds}
        (te if (ds, did) in test_docs else tr).append(item)
    # extra documents: MCC (commercial kinds, lease, not a contract) and ToS;DR (privacy, terms, cookie, EULA)
    M = bench.SETS["mcc"](); kind_of = lambda l: "lease" if l == "lease" else ("not a contract" if l == "na" else "commercial")
    for split, dest in (("train", tr), ("test", te)):
        for r in M[split]: dest.append({"id": r["id"], "text": r["text"], "label": kind_of(r["labels"][0]), "src": "mcc"})
    T = bench.SETS["doctype"]()
    tos_kind = {"privacy policy": "privacy policy", "terms of service": "terms of service", "cookie policy": "terms of service",
                "software licence / EULA": "terms of service"}
    for split, dest in (("train", tr), ("test", te)):
        for r in T[split]:
            if r["id"].startswith("tosdr/"): dest.append({"id": r["id"], "text": r["text"], "label": tos_kind[r["labels"][0]], "src": "tosdr"})
    return tr, te


def train_router(tr):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    X = v.fit_transform([d["text"][:4000] for d in tr])
    m = LogisticRegression(C=10, max_iter=3000); m.fit(X, [d["label"] for d in tr])
    return lambda texts: (m.classes_, m.predict_proba(v.transform([t[:4000] for t in texts])))


# ---------- clause models ----------
def fit_one(X, idx, y):
    """X is shared by all jobs (joblib maps it once); each job takes its own rows. Passing X[idx] per class copied
    the data once per class into /dev/shm (15 GB for 146 classes x 146k rows, 2026-10-03: the box ran out of memory)."""
    if y.min() == y.max(): return None
    m = LogisticRegression(C=10, max_iter=2000, class_weight="balanced"); m.fit(X[idx], y); return m


def train_items(items, max_features=400_000):
    """items: (text, positives, fully-marked classes). A row teaches class c if c is fully marked for it or it's
    a positive for c. Returns (vectorizer, class -> model)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True).fit([t for t, _, _ in items])
    X = vec.transform([t for t, _, _ in items])
    classes = sorted(set().union(*(f | p for _, p, f in items)))
    jobs = []
    for c in classes:
        idx = [i for i, (_, p, f) in enumerate(items) if c in f or c in p]
        jobs.append((c, idx, np.array([c in items[i][1] for i in idx], dtype=int)))
    ms = Parallel(n_jobs=NJ)(delayed(fit_one)(X, idx, y) for _, idx, y in jobs)
    return vec, {c: m for (c, _, _), m in zip(jobs, ms) if m is not None}


def kind_items(kind, S, tr, borrow, log):
    dss = [d for d in DATASETS if KIND[d] == kind]
    full = {d: {score_class(d, l) for l in S[d]["labels"]} for d in dss}
    items = [(r["text"], row_classes(d, r["labels"]), full[d]) for d in dss for r in tr[d]]
    if not borrow: return items
    npos = collections.Counter(c for _, p, _ in items for c in p)
    targets = {}
    for d in dss:
        for l in S[d]["labels"]:
            c = score_class(d, l); u, k = UNIFIED[d].get(l, (None, None))
            if npos[c] < 100 and u: targets[c] = u  # rare and has a unified class others can lend
    added = collections.Counter()
    for c, u in targets.items():
        src = [r["text"] for d in DATASETS if KIND[d] != kind for r in tr[d] if u in row_classes(d, r["labels"])]
        random.Random(0).shuffle(src)
        for t in src[:300]: items.append((t, {c}, set())); added[c] += 1
    BORROWED.update(added)
    log.append(f"{kind}: borrowed " + (", ".join(f"{c} +{n} (had {npos[c]})" for c, n in added.items()) or "nothing (no rare class has a source)"))
    return items


def predict(model, rows, cls):
    vec, ms = model; X = vec.transform([r["text"] for r in rows]); S_ = np.zeros((len(rows), len(cls)))
    for j, c in enumerate(cls):
        if c in ms: S_[:, j] = ms[c].predict_proba(X)[:, 1]
    return S_


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=1); a = ap.parse_args()
    n_eval, n_train = bench.STAGES[a.stage]
    S = {ds: bench.SETS[ds]() for ds in DATASETS}
    tr = {ds: bench.sample(S[ds]["train"], n_train, seed=1, multi=S[ds]["multi"]) for ds in DATASETS}
    ev = {ds: bench.sample(S[ds]["test"], n_eval, multi=S[ds]["multi"]) for ds in DATASETS}
    va = {ds: bench.sample(S[ds]["val"], 2000, multi=True) for ds in DATASETS}
    print(f"== family.py stage {a.stage}: eval " + ", ".join(f"{d} {len(ev[d])}" for d in DATASETS), flush=True)

    # 1. router
    t0 = time.time(); docs = clause_docs(); rtr, rte = router_data(S, docs)
    if a.stage < 3: rte = [rte[i] for i in sorted(random.Random(0).sample(range(len(rte)), min(len(rte), 500 if a.stage == 1 else 2000)))]
    route = train_router(rtr)
    cl, P = route([d["text"] for d in rte]); pred = cl[P.argmax(1)]
    ok = np.array([p == d["label"] for p, d in zip(pred, rte)])
    print(f"   router: train {len(rtr)} docs, test {len(rte)}: accuracy {ok.mean():.1%} ({time.time() - t0:.0f}s); "
          f"at p >= 0.9: {np.mean(ok[P.max(1) >= 0.9]):.1%} on {np.mean(P.max(1) >= 0.9):.0%}")
    conf = collections.Counter((d["label"], p) for d, p in zip(rte, pred) if d["label"] != p)
    print("   router confusions:", "; ".join(f"{g} -> {p} ×{n}" for (g, p), n in conf.most_common(6)))
    per_src = collections.defaultdict(list)
    for d, k in zip(rte, ok): per_src[d["src"]].append(k)
    print("   router by source:", ", ".join(f"{s} {np.mean(v):.1%} (n {len(v)})" for s, v in sorted(per_src.items())))
    json.dump([{"id": d["id"], "gold": d["label"], "pred": p, "p": round(float(x), 3)} for d, p, x in zip(rte, pred, P.max(1)) if d["label"] != p],
              open(f"{OUT}/router_errors_s{a.stage}.json", "w"), indent=0)

    # 2. clause models per kind, with and without borrowing; plus one model per dataset
    log = []; models = {}
    for kind in KINDS:
        for borrow in (False, True):
            models[(kind, borrow)] = train_items(kind_items(kind, S, tr, borrow, log))
    for ds in DATASETS:  # one model per dataset (for commercial: LEDGAR and CUAD apart)
        full = {score_class(ds, l) for l in S[ds]["labels"]}
        models[(ds, "own")] = train_items([(r["text"], row_classes(ds, r["labels"]), full) for r in tr[ds]])
    print("   " + "\n   ".join(log), flush=True)

    def score(ds, model):
        cls = sorted({score_class(ds, l) for l in S[ds]["labels"]})
        vp = (cls, predict(model, va[ds], cls), va[ds]) if va[ds] else None
        return combined.evaluate(ds, predict(model, ev[ds], cls), cls, ev[ds], vp, all_classes=True), cls, vp

    rows = []
    for ds in DATASETS:
        own, _, _ = score(ds, models[(ds, "own")])
        kd, _, _ = score(ds, models[(KIND[ds], False)])
        kb, cls, vp = score(ds, models[(KIND[ds], True)])
        e2e = None
        if ds in ("cuad_clause", "contractnli", "opp115", "lease"):
            # route each row's document; rows routed elsewhere are read by that kind's model
            dids = sorted({r["doc"] for r in ev[ds]}); rc, rp = route([docs[(ds, d)] for d in dids])
            dk = dict(zip(dids, rc[rp.argmax(1)]))
            S_ = np.zeros((len(ev[ds]), len(cls)))
            for k in set(dk.values()):
                idx = [i for i, r in enumerate(ev[ds]) if dk[r["doc"]] == k]
                if k in KINDS and idx: S_[idx] = predict(models[(k, True)], [ev[ds][i] for i in idx], cls)
            e2e = combined.evaluate(ds, S_, cls, ev[ds], vp)
            e2e["routed_right"] = np.mean([dk[r["doc"]] == KIND[ds] for r in ev[ds]])
        rows.append((ds, own, kd, kb, e2e))
    fmt = lambda r: (f"acc {100 * r['acc']:.1f}%" if "acc" in r else f"F1 {r['f1']:.3f}") if r else "—"
    print(f"\n   {'dataset':12} {'own model':>12} {'per kind':>12} {'kind+borrow':>12} {'routed e2e':>12}  rows routed right")
    for ds, own, kd, kb, e2e in rows:
        right = f"{e2e['routed_right']:.1%}" if e2e else ""
        print(f"   {ds:12} {fmt(own):>12} {fmt(kd):>12} {fmt(kb):>12} {fmt(e2e):>12}  {right}")
    for ds, own, kd, kb, e2e in rows:  # the classes that borrowed
        for c, (f, n) in sorted(kb.get("per", {}).items()):
            if c in BORROWED and c in kd.get("per", {}):
                print(f"      {ds:12} {c:36} n {n:4}  F1 {kd['per'][c][0]:.2f} -> {f:.2f}")


if __name__ == "__main__":
    main()
