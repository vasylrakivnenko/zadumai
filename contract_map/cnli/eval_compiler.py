"""Score the compiler queries (queries.py) on train (where they were written) and dev (not read) (2026-10-03).
usage: eval_compiler.py [train|dev|both] [--show Q] [--errors N]"""
import argparse, collections, json, re, sys, zipfile
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from compiler import compile_doc
from queries import QUERIES
from baseline import POLARITY, DEFAULT, ZIP
from cv_mentioned import wilson_lo


def run(split):
    d = json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{split}.json"))["documents"]; rows = []
    for doc in d:
        p = compile_doc(doc["text"]); ann = doc["annotation_sets"][0]["annotations"]
        for q, f in QUERIES.items():
            r = f(p); g = ann[q]["choice"]; want = POLARITY.get(q, DEFAULT).get(g)
            rows.append({"doc": doc["id"], "q": q, "gold": g, "answer": r[0] if r else None, "evidence": r[1] if r else "",
                         "right": bool(r) and r[0] == want, "gold_ev": [re.sub(r"\s+", " ", doc["text"][doc["spans"][i][0]:doc["spans"][i][1]]) for i in ann[q]["spans"][:2]]})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split", nargs="?", default="both"); ap.add_argument("--show")
    ap.add_argument("--errors", type=int, default=0); a = ap.parse_args()
    for split in (["train", "dev"] if a.split == "both" else [a.split]):
        rows = run(split); json.dump(rows, open(f"runs/compiler_{split}.json", "w"))
        st = collections.defaultdict(collections.Counter)
        for r in rows:
            c = st[r["q"]]; c["n"] += 1
            if r["answer"]: c["ans"] += 1; c["ok"] += r["right"]; c[r["answer"]] += 1
        print(f"== {split}")
        for q, c in st.items():
            print(f"  {q:6} answered {c['ans']:4}/{c['n']} = {c['ans'] / c['n']:4.0%}  right {c['ok'] / max(c['ans'], 1):6.1%} [lo {wilson_lo(c['ok'], c['ans']):.3f}]  (yes {c['yes']}, no {c['no']})")
        if split == "train" and a.errors:
            for r in [r for r in rows if r["answer"] and not r["right"] and (not a.show or r["q"] == a.show)][:a.errors]:
                print(f"\n  [{r['q']}] gold {r['gold']} -> {r['answer']}\n    ours: {r['evidence'][:260]}\n    gold: {' | '.join(x[:160] for x in r['gold_ev']) or '(none)'}")


if __name__ == "__main__":
    main()
