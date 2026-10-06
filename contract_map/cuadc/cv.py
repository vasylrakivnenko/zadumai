"""Nested 5-fold CV over the 408 tuning contracts (2026-10-03). Per type, each variant of clauses.py is a source with one
answer type (yes + evidence, or no). In each outer fold a variant is trusted if it is >= TARGET right on the 4 training
folds with >= MIN_N answers; a held-out contract gets the answer of its most precise trusted variant (training precision).
yes is right when the gold has a span and our statement overlaps one; no is right when the gold is empty.
Dev contracts (82) were never read while writing rules: their rows are the honest part. usage: cv.py [--target 0.98]"""
import argparse, collections, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import split, TYPES
from front import compile_contract
from clauses import answers
import explore
from explore import overl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnli"))
from cv_mentioned import wilson_lo  # noqa: E402

CACHE = "runs/tune_front.pkl"


def tune_compiled():
    if os.path.exists(CACHE): return pickle.load(open(CACHE, "rb"))
    docs = split()
    for d in docs: d["stmts"] = compile_contract(d["text"])
    pickle.dump(docs, open(CACHE, "wb")); return docs


def right(a, d, t):
    g = d["gold"][t]
    return (a[0] == "no" and not g) or (a[0] == "yes" and bool(g) and overl(a[1], g))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.98); ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--grain", default="sentence", choices=["sentence", "paragraph"]); a = ap.parse_args(); explore.GRAIN = a.grain
    docs = tune_compiled(); out = []; trusted_log = {}
    for t in TYPES:
        ans = [answers(d["stmts"], t) for d in docs]
        trusted_log[t] = collections.Counter()
        for f in range(5):
            tr = [i for i, d in enumerate(docs) if d["fold"] != f]; te = [i for i, d in enumerate(docs) if d["fold"] == f]
            prec = {}
            for v in {v for i in tr for v in ans[i]}:
                xs = [right(ans[i][v], docs[i], t) for i in tr if v in ans[i]]
                if len(xs) >= a.min_n and sum(xs) / len(xs) >= a.target: prec[v] = sum(xs) / len(xs)
            trusted_log[t].update(prec)
            for i in te:
                vs = sorted((v for v in ans[i] if v in prec), key=lambda v: -prec[v])
                if vs:
                    x = ans[i][vs[0]]
                    out.append({"doc": docs[i]["id"], "t": t, "answer": x[0], "variant": vs[0], "right": right(x, docs[i], t),
                                "present": bool(docs[i]["gold"][t]), "dev": docs[i]["dev"],
                                "ev": [x[1].start, x[1].end, x[1].own[:300]] if x[1] else None})
                else:
                    out.append({"doc": docs[i]["id"], "t": t, "answer": None, "present": bool(docs[i]["gold"][t]), "dev": docs[i]["dev"]})
    json.dump(out, open(f"runs/cv_{a.grain}_{a.target}.json", "w"))
    def summ(rows):
        n = len(rows); an = [r for r in rows if r["answer"]]; ok = sum(r["right"] for r in an)
        return f"{len(an):4}/{n:4} = {len(an) / max(n, 1):5.1%} answered, right {ok / max(len(an), 1):6.1%} [lo {wilson_lo(ok, len(an)):.3f}]"
    print(f"== CUAD clause compiler, nested 5-fold CV over {len(docs)} tuning contracts, target {a.target}, min_n {a.min_n}, evidence grain {a.grain}")
    for t in TYPES:
        rows = [r for r in out if r["t"] == t]
        yes = [r for r in rows if r["answer"] == "yes"]; no = [r for r in rows if r["answer"] == "no"]
        print(f"  {t:28} {summ(rows)} | dev {summ([r for r in rows if r['dev']])} | yes {len(yes)} ({sum(r['right'] for r in yes)} ok) no {len(no)} ({sum(r['right'] for r in no)} ok)"
              f" | present-found {sum(r['answer'] == 'yes' for r in rows if r['present'])}/{sum(r['present'] for r in rows)} | trusted {dict(trusted_log[t])}")
    print(f"  {'all':28} {summ(out)}\n  {'dev (never read)':28} {summ([r for r in out if r['dev']])}\n  {'write':28} {summ([r for r in out if not r['dev']])}")


if __name__ == "__main__":
    main()
