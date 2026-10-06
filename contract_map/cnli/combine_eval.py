"""Engine v2 (held-out answers from its nested CV) with and without the compiler queries in front (2026-10-03).
The compiler answers a question only if it was >= MIN_TRAIN right on the training NDAs (where its rules were written);
dev (never read) is the honest check. usage: combine_eval.py [--min-train 0.98]"""
import argparse, collections, json, zipfile, sys
sys.path.insert(0, ".")
from cv_engine2 import right, Q
from cv_mentioned import wilson_lo
ap = argparse.ArgumentParser(); ap.add_argument("--min-train", type=float, default=0.98); a = ap.parse_args()
v2 = {(r["doc"], r["q"]): r for r in json.load(open("runs/cv_engine2.json"))}
comp = {s: {(r["doc"], r["q"]): r for r in json.load(open(f"runs/compiler_{s}.json"))} for s in ("train", "dev")}
tr = collections.defaultdict(lambda: [0, 0])
for r in comp["train"].values():
    if r["answer"]: tr[r["q"]][0] += 1; tr[r["q"]][1] += r["right"]
ok_q = [q for q, (n, k) in tr.items() if n and k / n >= a.min_train]
print(f"compiler questions (>= {a.min_train} on train): {ok_q}")
passing_v2 = ['nda-1', 'nda-3', 'nda-5', 'nda-7', 'nda-8', 'nda-10', 'nda-13', 'nda-15', 'nda-16', 'nda-17', 'nda-18']
for split in ("dev", "train"):
    st = collections.defaultdict(collections.Counter)
    for (doc, q), r in comp[split].items():
        b = v2[(doc, q)]
        sysA = b["answer"] if q in passing_v2 else None
        c_ans = r["answer"] if q in ok_q else None
        sysB = c_ans or sysA
        sysC = c_ans or b["answer"]
        for name, ans in (("v2 passing", sysA), ("compiler + v2 passing", sysB), ("compiler + v2 all", sysC)):
            c = st[name]; c["n"] += 1
            if ans: c["ans"] += 1; c["ok"] += right(ans, r["gold"], q)
    print(f"== {split}" + (" (never read)" if split == "dev" else " (compiler rules written here: optimistic)"))
    for name, c in st.items():
        print(f"  {name:24} answered {c['ans']:4}/{c['n']} = {c['ans'] / c['n']:5.1%}, right {c['ok'] / max(c['ans'], 1):6.1%} [lo {wilson_lo(c['ok'], c['ans']):.3f}]")
