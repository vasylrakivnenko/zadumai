"""Look at the WRITE contracts only (never dev/test): gold headings, gold texts, and what a candidate regex hits.
usage: explore.py TYPE [--rx REGEX] [--n 25] [--fp]"""
import argparse, collections, pickle, os, random, re
from data import write_docs
from front import compile_contract

CACHE = "runs/write_front.pkl"


def load():
    if os.path.exists(CACHE): return pickle.load(open(CACHE, "rb"))
    docs = write_docs()
    for d in docs: d["stmts"] = compile_contract(d["text"])
    pickle.dump(docs, open(CACHE, "wb")); return docs


GRAIN = "sentence"  # or "paragraph": the statement's block contains the annotated span (cv.py --grain)


def overl(s, gold, grain=None):
    a, b = (s.pstart, s.pend) if (grain or GRAIN) == "paragraph" else (s.start, s.end)
    return a >= 0 and any(a < ge and b > gs for gs, ge, _ in gold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("type"); ap.add_argument("--rx"); ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--fp", action="store_true"); ap.add_argument("--fn", action="store_true"); a = ap.parse_args()
    docs = load(); t = a.type; rnd = random.Random(0)
    if not a.rx:
        heads = collections.Counter(s.head for d in docs for s in d["stmts"] if overl(s, d["gold"][t]))
        print("gold heads:", heads.most_common(40))
        g = [(d["id"][:30], x[2]) for d in docs for x in d["gold"][t]]; rnd.shuffle(g)
        for i, x in g[:a.n]: print(f"- [{i}] {x[:400]!r}")
    else:
        rx = re.compile(a.rx, re.I | re.S); tp = fp = fn = 0; fps = []; fns = []
        for d in docs:
            hits = [s for s in d["stmts"] if rx.search(s.text)]
            good = [s for s in hits if overl(s, d["gold"][t])]
            if hits and d["gold"][t] and good: tp += 1
            elif hits: fp += 1; fps.append((d["id"][:30], hits[0], d["gold"][t][:1]))
            elif d["gold"][t]: fn += 1; fns.append((d["id"][:30], d["gold"][t][0][2]))
        print(f"docs: hit&overlap {tp}  hit-wrong {fp}  missed {fn}  (present {sum(bool(d['gold'][t]) for d in docs)}/{len(docs)})")
        if a.fp:
            for i, s, g in fps[:a.n]: print(f"- FP [{i}] head={s.head[:40]!r} | {s.text[:300]!r}\n     gold: {g[0][2][:200]!r}" if g else f"- FP [{i}] head={s.head[:40]!r} | {s.text[:300]!r}  (gold: absent)")
        if a.fn:
            for i, g in fns[:a.n]: print(f"- FN [{i}] {g[:300]!r}")
