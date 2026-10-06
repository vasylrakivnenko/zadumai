"""Score the IR queries (queries_ir.py) on train (rules written here) and dev (never read) (2026-10-03, overnight).
Facts are cached per split in runs/facts_<split>.pkl (delete it after changing ir.py).
usage: eval_ir.py [train|dev|both] [--q nda-7] [--errors N] [--refresh]"""
import argparse, collections, json, os, pickle, re, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import POLARITY, DEFAULT, ZIP
from cv_mentioned import wilson_lo


def facts_for(split, refresh=False):
    path = f"runs/facts_{split}.pkl"
    if os.path.exists(path) and not refresh: return pickle.load(open(path, "rb"))
    from ir import facts
    d = json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{split}.json"))["documents"]; t0 = time.time()
    out = {doc["id"]: facts(doc["text"]) for doc in d}
    pickle.dump(out, open(path, "wb")); print(f"   facts for {split}: {len(out)} NDAs in {time.time() - t0:.0f}s", flush=True)
    return out


def run(split, refresh=False):
    from queries_ir import QUERIES
    F = facts_for(split, refresh)
    d = json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{split}.json"))["documents"]; rows = []
    for doc in d:
        f = F[doc["id"]]; ann = doc["annotation_sets"][0]["annotations"]
        for q, fn in QUERIES.items():
            r = fn(f); g = ann[q]["choice"]; want = POLARITY.get(q, DEFAULT).get(g)
            rows.append({"doc": doc["id"], "q": q, "gold": g, "answer": r[0] if r else None, "evidence": r[1] if r else "", "right": bool(r) and r[0] == want,
                         "gold_ev": [re.sub(r"\s+", " ", doc["text"][doc["spans"][i][0]:doc["spans"][i][1]]) for i in ann[q]["spans"][:2]]})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split", nargs="?", default="both"); ap.add_argument("--q")
    ap.add_argument("--errors", type=int, default=0); ap.add_argument("--refresh", action="store_true"); a = ap.parse_args()
    for split in (["train", "dev"] if a.split == "both" else [a.split]):
        rows = run(split, a.refresh); json.dump(rows, open(f"runs/ir_{split}.json", "w"))
        st = collections.defaultdict(collections.Counter)
        for r in rows:
            c = st[r["q"]]; c["n"] += 1
            if r["answer"]: c["ans"] += 1; c["ok"] += r["right"]; c[r["answer"]] += 1
        tot = collections.Counter()
        print(f"== {split}")
        for q, c in st.items():
            tot.update(c)
            print(f"  {q:6} answered {c['ans']:4}/{c['n']} = {c['ans'] / c['n']:4.0%}  right {c['ok'] / max(c['ans'], 1):6.1%} [lo {wilson_lo(c['ok'], c['ans']):.3f}]  (yes {c['yes']}, no {c['no']})")
        print(f"  all    answered {tot['ans']}/{tot['n']} = {tot['ans'] / tot['n']:.1%}, right {tot['ok'] / max(tot['ans'], 1):.1%}")
        if split == "train" and a.errors:
            ws = [r for r in rows if r["answer"] and not r["right"] and (not a.q or r["q"] == a.q)]
            print(f"  {len(ws)} wrong" + (f" for {a.q}" if a.q else "") + ":", dict(collections.Counter((r['gold'], r['answer']) for r in ws)))
            for r in ws[:a.errors]:
                print(f"\n  [{r['q']}] gold {r['gold']} -> {r['answer']}\n    ours: {r['evidence'][:240]}\n    gold: {' | '.join(x[:150] for x in r['gold_ev']) or '(none)'}")


if __name__ == "__main__":
    main()
