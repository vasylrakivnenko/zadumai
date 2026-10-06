"""Section tagging, live units vs compiled units, with and without headings (2026-10-03). 5-fold CV over CUAD's 408
tuning contracts (data.split() folds; CUAD's 102 test contracts stay sealed). Each fold trains its own TF-IDF + one-vs-rest
LR (C=10, balanced: bench/family.py's setup) on the other 4 folds' units of the same kind, so nothing is scored in-sample.
Units:
  live     router/contract_map.split_sections: blank-line blocks, short ones joined, cut at 4,000 chars (in CUAD a block
           is usually a whole page)
  leaf     doccompile: runs of statements in the same innermost section (2.1(a), a heading's scope, ...), cut at 4,000
  headed   doccompile: the innermost section that has a heading (its unheaded children merged in), cut at 4,000
Features: text (TF-IDF 1-2 grams) | text+head (plus a second TF-IDF over the unit's heading chain).
Scores: unit-level micro F1 (best single threshold, the same search for every method); clause finding at (contract, type):
the contract's best-scoring unit for the type, right if it overlaps a gold span; coverage of the present clauses at
>= 95% / 98% precision (threshold from the pooled out-of-fold scores, the same for every method); chars per unit."""
import argparse, collections, json, os, re, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/cuadc")
from data import split
from doccompile import compile_doc

META = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
MAXC = 4000


def live_units(text):
    """router/contract_map.split_sections (56093f4), with end offsets."""
    sep = r"\n\s*\n" if len(re.findall(r"\n\s*\n", text)) >= max(3, text.count("\n") // 4) else r"\n"
    pieces, pos = [], 0
    for m in re.finditer(sep, text + "\n\n"):
        pieces.append((pos, text[pos:m.start()])); pos = m.end()
    out, carry, cs = [], "", None
    for start, raw in pieces:
        end = start + len(raw); p = re.sub(r"\s+", " ", raw).strip()
        if not p: continue
        if carry: p, start = carry + " " + p, cs
        if len(p) < 40: carry, cs = p, start; continue
        carry = ""
        cuts = list(range(0, len(p), MAXC))
        for n, k in enumerate(cuts):
            out.append({"spans": [(start + k, end if n == len(cuts) - 1 else start + k + MAXC)], "text": p[k:k + MAXC], "head": ""})
    if carry: out.append({"spans": [(cs, len(text))], "text": carry, "head": ""})
    return out


def compiled_units(text, mode):
    c = compile_doc(text); groups = []
    for st in c.stmts:
        if st.start < 0: continue
        secs = [i for i in st.secs if not c.sections[i].front]
        if mode == "headed":
            hs = [i for i in secs if c.sections[i].heading]
            key = hs[-1] if hs else (secs[0] if secs else -1)
        else:
            key = secs[-1] if secs else -1
        if groups and groups[-1][0] == key and sum(len(s.own) for s in groups[-1][1]) < MAXC: groups[-1][1].append(st)
        else: groups.append((key, [st]))
    out = []
    for key, sts in groups:
        hs = c.headings(sts[0])
        out.append({"spans": [(s.start, s.end) for s in sts], "text": " ".join(s.own for s in sts)[:MAXC], "head": " | ".join(hs)})
    return out


def prep(args):
    d, mode = args
    units = live_units(d["text"]) if mode == "live" else compiled_units(d["text"], mode)
    gold = {t: [(a, b) for a, b, _ in v] for t, v in d["gold"].items() if t not in META}
    for u in units:
        u["labels"] = sorted(t for t, g in gold.items() if any(a < ge and gs < b for a, b in u["spans"] for gs, ge in g))
        u["chars"] = sum(b - a for a, b in u["spans"])
    return {"id": d["id"], "fold": d["fold"], "units": units, "present": sorted(t for t, g in gold.items() if g)}


def fit_one(X, y):
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max(): return None
    return LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(X, y)


def run(docs, types, feats, jobs):
    from joblib import Parallel, delayed
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    oof = {}
    for f in range(5):
        tr = [u for d in docs if d["fold"] != f for u in d["units"]]
        te = [(d["id"], k, u) for d in docs if d["fold"] == f for k, u in enumerate(d["units"])]
        txt = (lambda u: (u["head"] + " . " + u["text"]) if u["head"] else u["text"]) if feats == "head-in-text" else (lambda u: u["text"])
        vt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True).fit([txt(u) for u in tr])
        Xtr, Xte = vt.transform([txt(u) for u in tr]), vt.transform([txt(u) for _, _, u in te])
        if feats == "text+head":
            vh = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([u["head"] or "_" for u in tr])
            Xtr = hstack([Xtr, vh.transform([u["head"] or "_" for u in tr])]).tocsr()
            Xte = hstack([Xte, vh.transform([u["head"] or "_" for _, _, u in te])]).tocsr()
        ms = Parallel(n_jobs=jobs)(delayed(fit_one)(Xtr, np.array([t in u["labels"] for u in tr], dtype=int)) for t in types)
        S = np.zeros((len(te), len(types)))
        for j, m in enumerate(ms):
            if m is not None: S[:, j] = m.predict_proba(Xte)[:, 1]
        for (did, k, _), s in zip(te, S): oof[did, k] = s
    return oof


def score(docs, types, oof):
    G, Sc = [], []
    for d in docs:
        for k, u in enumerate(d["units"]):
            G.append([t in u["labels"] for t in types]); Sc.append(oof[d["id"], k])
    G, Sc = np.array(G), np.array(Sc)
    best = max(((2 * ((Sc >= th) & G).sum() / max(1, (Sc >= th).sum() + G.sum())), th) for th in np.arange(0.1, 1.0, 0.05))
    # clause finding: per (contract, type) the best unit
    rows = []
    for d in docs:
        for j, t in enumerate(types):
            ks = [oof[d["id"], k][j] for k in range(len(d["units"]))]
            if not ks: continue
            k = int(np.argmax(ks)); u = d["units"][k]
            rows.append((ks[k], t in d["present"], t in u["labels"], u["chars"]))
    rows.sort(key=lambda r: -r[0])
    present = sum(r[1] for r in rows)
    out = {"unit_f1": round(best[0], 3), "f1_th": round(best[1], 2), "units": len(G), "unit_chars": int(np.mean([u["chars"] for d in docs for u in d["units"]]))}
    for target in (0.95, 0.98):
        ok = n = 0; cov = 0; chars = []
        for s, pres, hit, ch in rows:
            n += 1; ok += hit
            if ok / n >= target: cov = ok; chars_at = n
        hits = [r for r in rows[:chars_at]] if cov else []
        out[f"found@{int(target*100)}"] = f"{cov}/{present} = {cov/present:.1%}"
        out[f"chars@{int(target*100)}"] = int(np.mean([r[3] for r in hits if r[2]])) if hits else 0
    return out


def gated(docs, types, oof, target, min_n=10):
    """Per type, the contract's best unit; a threshold per type and fold from the other folds' rows (the lowest score
    whose answers there are >= target right, with >= min_n answers); applied to the held-out fold. Returns
    (found, answered, right, present)."""
    rows = collections.defaultdict(list)  # type -> [(fold, score, hit, present)]
    for d in docs:
        for j, t in enumerate(types):
            ks = [oof[d["id"], k][j] for k in range(len(d["units"]))]
            if not ks: continue
            k = int(np.argmax(ks)); rows[t].append((d["fold"], ks[k], t in d["units"][k]["labels"], t in d["present"]))
    found = answered = right = 0
    for t, rs in rows.items():
        for f in range(5):
            trn = sorted((r for r in rs if r[0] != f), key=lambda r: -r[1]); ok = n = 0; th = None
            for _, sc, hit, _ in trn:
                n += 1; ok += hit
                if n >= min_n and ok / n >= target: th = sc
            if th is None: continue
            for _, sc, hit, _ in (r for r in rs if r[0] == f):
                if sc >= th: answered += 1; right += hit
    present = sum(t in d["present"] for d in docs for t in types)
    return right, answered, present


def per_type(docs, types, oof, target=0.95):
    res = {}
    for j, t in enumerate(types):
        rows = []
        for d in docs:
            ks = [oof[d["id"], k][j] for k in range(len(d["units"]))]
            if not ks: continue
            k = int(np.argmax(ks)); rows.append((ks[k], t in d["units"][k]["labels"]))
        rows.sort(key=lambda r: -r[0]); ok = n = cov = 0
        for s, hit in rows:
            n += 1; ok += hit
            if ok / n >= target: cov = ok
        res[t] = (cov, sum(t in d["present"] for d in docs))
    return res


def cnli_split():
    """ContractNLI train + dev (484 NDAs; the test NDAs are spent): gold = each hypothesis's evidence spans; 5 folds."""
    import random, zipfile
    z = zipfile.ZipFile("/root/zadumai_nli_proto/contract_map/data/ext/contractnli.zip"); out = []
    for sp in ("train", "dev"):
        d = json.loads(z.read(f"contract-nli/{sp}.json")); lab = d["labels"]
        for doc in d["documents"]:
            gold = {v["short_description"]: [] for v in lab.values()}
            for k, an in doc["annotation_sets"][0]["annotations"].items():
                gold[lab[k]["short_description"]] += [tuple(doc["spans"][i]) + ("",) for i in an["spans"]]
            out.append({"id": f"cnli/{doc['id']}", "text": doc["text"], "gold": gold})
    order = list(range(len(out))); random.Random(1).shuffle(order)
    for r, i in enumerate(order): out[i]["fold"] = r % 5
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--set", default="cuad")
    a = ap.parse_args(); os.makedirs("runs", exist_ok=True); a.out = f"runs/tagging_{a.set}.json"
    data = split() if a.set == "cuad" else cnli_split(); results = {}; pt = {}
    for mode in ("live", "leaf", "headed"):
        t0 = time.time()
        with ProcessPoolExecutor(8) as ex: docs = list(ex.map(prep, [(d, mode) for d in data], chunksize=4))
        types = sorted({t for d in data for t in d["gold"] if t not in META})
        for feats in (("text",) if mode == "live" else ("text", "head-in-text", "text+head")):
            oof = run(docs, types, feats, a.jobs)
            r = score(docs, types, oof)
            for target in (0.95, 0.98):
                ok, n, present = gated(docs, types, oof, target)
                r[f"gated@{int(target * 100)}"] = f"found {ok}/{present} = {ok / present:.1%}, right {ok}/{n} = {ok / max(1, n):.1%}"
            results[f"{mode}/{feats}"] = r; pt[f"{mode}/{feats}"] = per_type(docs, types, oof)
            np.save(f"runs/oof_{a.set}_{mode}_{feats}.npy", np.array([[d["id"], k, *map(float, oof[d["id"], k])] for d in docs for k in range(len(d["units"]))], dtype=object), allow_pickle=True)
            print(f"{mode:7s} {feats:10s} {json.dumps(r)}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"results": results, "per_type": pt}, open(a.out, "w"), indent=1)
