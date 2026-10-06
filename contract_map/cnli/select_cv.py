"""Which source answers each NDA question: nested 5-fold CV over the 484 tuning NDAs (2026-10-03, overnight).
Sources per (NDA, question): Pre-Tier 0 (router), engine v2 (its own held-out answers, runs/cv_engine2.json, same folds),
the compiler queries (queries.py, runs/compiler_*.json), the IR queries (queries_ir.py, runs/ir_*.json).
In each outer fold a source is trusted for a question if its answers on the 4 training folds are >= TARGET right (and
at least MIN_N of them); the held-out NDAs get the trusted sources' answer when they agree, nothing when they disagree.
Rules of the compiler and IR passes were written from the training NDAs (not dev), so the dev rows are the honest part.
Writes runs/select_cv.json. usage: select_cv.py [--target 0.99] [--min-n 15]"""
import argparse, collections, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from baseline import QUESTIONS, POLARITY, DEFAULT
from cv_mentioned import load, wilson_lo

Q = list(QUESTIONS)


def right(ans, gold, q):
    if ans == "not mentioned": return gold == "NotMentioned"
    return POLARITY.get(q, DEFAULT).get(gold) == ans


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.99); ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--sources", default="pt0,v2,comp,ir"); ap.add_argument("--grain", default="answer", choices=["source", "answer"]); ap.add_argument("--pairs", action="store_true"); ap.add_argument("--variants", action="store_true"); ap.add_argument("--post", action="store_true"); a = ap.parse_args()
    from postcheck import keep
    docs = load(); idx = list(range(len(docs))); random.Random(0).shuffle(idx); fold = {}
    for r, i in enumerate(idx): fold[docs[i]["id"]] = r % 5
    gold = {(d["id"], q): d["gold"][q] for d in docs for q in Q}; text = {d["id"]: d["text"] for d in docs}
    dev_ids = {r["doc"] for r in json.load(open("runs/compiler_dev.json"))}
    src = collections.defaultdict(dict)  # source -> (doc, q) -> answer
    for r in json.load(open("runs/cv_engine2.json")):
        if r["answer"] and r["why"] != "Pre-Tier 0": src["v2"][(r["doc"], r["q"])] = r["answer"]
    for s, name in (("comp", "compiler"), ("ir", "ir")):
        for split in ("train", "dev"):
            for r in json.load(open(f"runs/{name}_{split}.json")):
                if r["answer"]: src[s][(r["doc"], r["q"])] = r["answer"]
    if a.variants:
        for split in ("train", "dev"):
            for r in json.load(open(f"runs/variants_{split}.json")):
                if r["answer"]: src[r["source"]][(r["doc"], r["q"])] = r["answer"]
    from router.pretier0 import check
    for d in docs:
        for q in Q:
            p = check(QUESTIONS[q], d["text"])
            if p.fired: src["pt0"][(d["id"], q)] = p.answer
    sources = a.sources.split(",") + (["var_c3", "var_c4", "var_oral"] if a.variants else [])
    if a.grain == "answer":  # a source's "yes", "no" and "not mentioned" are trusted separately
        for s0 in list(sources):
            for (i, q), v in src[s0].items(): src[f"{s0}:{v}"][(i, q)] = v
        sources = [f"{s0}:{v}" for s0 in sources for v in ("yes", "no", "not mentioned")]
    if a.pairs:  # two different sources giving the same answer: its own (usually more precise) source
        base = sorted({x.split(":")[0] for x in sources})
        for i1, s1 in enumerate(base):
            for s2 in base[i1 + 1:]:
                for v in ("yes", "no", "not mentioned"):
                    name = f"{s1}+{s2}:{v}"
                    for k, x in src[f"{s1}:{v}"].items():
                        if src[f"{s2}:{v}"].get(k) == x: src[name][k] = x
                    sources.append(name)
    out = []; trusted_log = collections.defaultdict(collections.Counter)
    for f in range(5):
        tr = [d["id"] for d in docs if fold[d["id"]] != f]; te = [d["id"] for d in docs if fold[d["id"]] == f]
        for q in Q:
            trusted = []
            for s in sources:
                ans = [(src[s][(i, q)], gold[(i, q)]) for i in tr if (i, q) in src[s]]
                if len(ans) >= a.min_n and sum(right(x, g, q) for x, g in ans) / len(ans) >= a.target: trusted.append(s)
            trusted_log[q].update(trusted)
            for i in te:
                vals = {src[s][(i, q)] for s in trusted if (i, q) in src[s]}
                ans = vals.pop() if len(vals) == 1 else None
                if ans and a.post and not keep(q, ans, text[i]): ans = None
                out.append({"doc": i, "q": q, "gold": gold[(i, q)], "answer": ans, "trusted": trusted, "conflict": len(vals) > 0, "dev": i in dev_ids})
    json.dump(out, open("runs/select_cv.json", "w"))
    def summ(rows):
        n = len(rows); ans = [r for r in rows if r["answer"]]; ok = sum(right(r["answer"], r["gold"], r["q"]) for r in ans)
        return f"{len(ans):5}/{n} = {len(ans) / n:5.1%} answered, right {ok / max(len(ans), 1):6.1%} [lo {wilson_lo(ok, len(ans)):.3f}]"
    print(f"== source selection, nested 5-fold CV, target {a.target}, sources {sources}")
    for q in Q:
        rows = [r for r in out if r["q"] == q]
        print(f"  {q:6} {summ(rows)} | dev {summ([r for r in rows if r['dev']])} | trusted (folds): {dict(trusted_log[q])}")
    print(f"  all    {summ(out)}")
    print(f"  dev    {summ([r for r in out if r['dev']])}   (never read)")
    print(f"  train  {summ([r for r in out if not r['dev']])}   (rules written here)")


if __name__ == "__main__":
    main()
