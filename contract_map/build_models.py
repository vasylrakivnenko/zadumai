"""Build the contract map's models for router/contract_map.py (2026-10-02; the user approved the recommended setup).
Writes legalbench_map/models/contract_map/ (gitignored, like models/reader_net):
  router.joblib     TF-IDF + LR on a document's first 4,000 characters -> kind (family.py's training documents, which
                    exclude every clause test document; measured 95.1% on 1,156 held-out documents)
  tagger_<ds>.joblib for ledgar, cuad, contractnli, opp115, tos: TF-IDF + one-vs-rest LR (bench/family.py's own
                    model per dataset), weights as one float32 matrix, plus the threshold picked on its validation rows
  lease/            MiniLM-L6 fine-tuned on the lease paragraphs (bench/encoder.py, separate mode, all training rows)
  meta.json         per-class test F1 (CUAD's own test contracts), the classes the router sends to the LLM, sources
usage: build_models.py [--skip-encoder]"""
import argparse, json, os, sys, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/bench")
import joblib
import bench, combined, family
from combined import score_class, row_classes

OUT = "/root/projects/zadumai/legalbench_map/models/contract_map"
WEAK_F1 = 0.75  # a CUAD clause type tagged below this F1 on CUAD's test contracts counts as hard to find locally


def compact(model):
    """(vectorizer, {class: LR}) -> (vectorizer, classes, W float32 [features x classes], b float32)."""
    vec, ms = model; classes = sorted(ms)
    W = np.stack([ms[c].coef_[0] for c in classes], axis=1).astype(np.float32)
    b = np.array([ms[c].intercept_[0] for c in classes], dtype=np.float32)
    return vec, classes, W, b


def scores(vec, W, b, texts):
    z = vec.transform(texts) @ W + b
    return 1 / (1 + np.exp(-z))


def f1(G, P):
    tp = sum(len(g & p) for g, p in zip(G, P)); fp = sum(len(p - g) for g, p in zip(G, P)); fn = sum(len(g - p) for g, p in zip(G, P))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-encoder", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); meta = {"built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "taggers": {}}
    t0 = time.time()
    S = {ds: bench.SETS[ds]() for ds in combined.DATASETS}
    docs = family.clause_docs(); rtr, rte = family.router_data(S, docs)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    X = v.fit_transform([d["text"][:4000] for d in rtr]); m = LogisticRegression(C=10, max_iter=3000).fit(X, [d["label"] for d in rtr])
    P = m.predict_proba(v.transform([d["text"][:4000] for d in rte])); pred = m.classes_[P.argmax(1)]
    acc = float(np.mean([p == d["label"] for p, d in zip(pred, rte)]))
    joblib.dump({"vec": v, "lr": m}, f"{OUT}/router.joblib", compress=3)
    meta["router"] = {"kinds": list(m.classes_), "train_docs": len(rtr), "test_docs": len(rte), "test_accuracy": round(acc, 4)}
    print(f"router: {len(rtr)} docs, held-out {len(rte)}: {acc:.1%}  ({time.time() - t0:.0f}s)", flush=True)

    for ds, name in (("ledgar", "ledgar"), ("cuad_clause", "cuad"), ("contractnli", "contractnli"), ("opp115", "opp115"), ("tos", "tos")):
        t1 = time.time(); D = S[ds]; full = {score_class(ds, l) for l in D["labels"]}
        model = family.train_items([(r["text"], row_classes(ds, r["labels"]), full) for r in D["train"]], max_features=100_000)
        vec, classes, W, b = compact(model)
        keep = [i for i, c in enumerate(classes) if c in full]; classes = [classes[i] for i in keep]; W, b = W[:, keep], b[keep]
        test = D["test"]; G = [{score_class(ds, l) for l in r["labels"]} for r in test]; St = scores(vec, W, b, [r["text"] for r in test])
        if D["multi"]:
            val = bench.sample(D["val"], 2000, multi=True); Gv = [{score_class(ds, l) for l in r["labels"]} for r in val]
            Sv = scores(vec, W, b, [r["text"] for r in val])
            th = max((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: f1(Gv, [{classes[j] for j in np.where(s >= t)[0]} for s in Sv]))
            Pt = [{classes[j] for j in np.where(s >= th)[0]} for s in St]
            overall = {"f1": round(f1(G, Pt), 4)}
        else:
            th = 0.5; Pt = [{classes[int(s.argmax())]} if s.max() >= th else set() for s in St]
            overall = {"accuracy_answered": round(float(np.mean([bool(p & g) for p, g in zip(Pt, G) if p])), 4),
                       "answered": round(float(np.mean([bool(p) for p in Pt])), 4)}
        per = {}
        for c in classes:
            g = [c in x for x in G]; p = [c in x for x in Pt]; tp = sum(x and y for x, y in zip(g, p))
            per[c] = {"f1": round(2 * tp / max(sum(g) + sum(p), 1), 3), "test_n": sum(g)}
        joblib.dump({"vec": vec, "classes": classes, "W": W, "b": b, "threshold": th, "labels": {score_class(ds, l): l for l in D["labels"]}},
                    f"{OUT}/tagger_{name}.joblib", compress=3)
        meta["taggers"][name] = {"dataset": ds, "classes": len(classes), "threshold": th, "test": overall, "per_class": per}
        print(f"tagger {name}: {len(classes)} classes, threshold {th}, test {overall}  ({time.time() - t1:.0f}s)", flush=True)

    cuad = meta["taggers"]["cuad"]["per_class"]
    labels = joblib.load(f"{OUT}/tagger_cuad.joblib")["labels"]
    weak = sorted(labels[c] for c, v in cuad.items() if v["f1"] < WEAK_F1)
    meta["to_llm"] = {"cuad_types": weak, "rule": f"CUAD clause types tagged below F1 {WEAK_F1} on CUAD's test contracts, and leases"}
    print(f"to the LLM: {len(weak)} of {len(cuad)} CUAD types: {weak}", flush=True)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)

    if not a.skip_encoder:
        import torch, encoder
        torch.set_num_threads(6); torch.manual_seed(0)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(encoder.MODEL); D = S["lease"]
        net, classes = encoder.train_model(["lease"], S, {"lease": D["train"]}, tok, 2, "lease")
        cls = sorted({score_class("lease", l) for l in D["labels"]})
        val = bench.sample(D["val"], 2000, multi=True); Sv = encoder.scores(net, classes, tok, val, cls)
        Gv = [{score_class("lease", l) for l in r["labels"]} for r in val]
        th = max((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: f1(Gv, [{cls[j] for j in np.where(s >= t)[0]} for s in Sv]))
        St = encoder.scores(net, classes, tok, D["test"], cls); G = [{score_class("lease", l) for l in r["labels"]} for r in D["test"]]
        ft = f1(G, [{cls[j] for j in np.where(s >= th)[0]} for s in St])
        d = f"{OUT}/lease"; os.makedirs(d, exist_ok=True)
        net.enc.save_pretrained(d); tok.save_pretrained(d); torch.save(net.head.state_dict(), f"{d}/head.pt")
        json.dump({"classes": classes, "score_classes": cls, "threshold": th, "test_f1": round(ft, 4),
                   "labels": {score_class("lease", l): l for l in D["labels"]}}, open(f"{d}/meta.json", "w"), indent=1)
        print(f"lease encoder: threshold {th}, test F1 {ft:.3f} (all {len(D['test'])} test paragraphs)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
