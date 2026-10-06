"""From statements to the NDA's answer (2026-10-04). The statement encoder (train.py) says, per compiled statement and
question, p(no), p(yes), p(says nothing); the NDA's answer per question comes from readable rules over those:
  R1  the strongest statement decides: yes / no by the larger of max p(yes), max p(no) over the statements, if it
      reaches T, else not stated (one T for all questions)
  R2  R1 with T per question
  R3  R2 + a precedence per question when a statement says yes AND another says no (both >= T): yes wins (an exception
      beats the general rule: "may keep one archival copy" over "return all copies"), no wins, or the larger wins
T and the precedences are chosen on the training NDAs' out-of-fold predictions (4 folds), then scored on nda_ho.
Also writes per (NDA, question) features for the stacker: feats/ndaenc_{train,ho}.jsonl.
usage: python aggregate.py [--oof f0,f1,f2,f3] [--ho full]"""
import argparse, collections, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = ["no", "yes", "not_stated"]
ap = argparse.ArgumentParser(); ap.add_argument("--oof", default="f0,f1,f2,f3"); ap.add_argument("--ho", default="full")
a = ap.parse_args()
meta = json.load(open(f"{HERE}/data/docs.json")); QIDS = meta["qids"]


def gold(split):
    return {(str(d), q): LAB[v["gold"][q]] if q in v["gold"] else "not_stated" for d, v in meta["docs"][split].items() for q in QIDS}


def load(names, split):
    per = collections.defaultdict(list)  # doc -> [17 x 3]
    for n in names:
        p = f"{HERE}/preds/{n}_{split}.jsonl"
        if not os.path.exists(p): continue
        for l in open(p):
            r = json.loads(l); per[str(r["doc"])].append(r["p"])
    return per


def feats(per):
    out = {}
    for d, sts in per.items():
        mx = {q: (max(s[k][1] for s in sts), max(s[k][0] for s in sts)) for k, q in enumerate(QIDS)}
        for k, q in enumerate(QIDS):
            py = sorted((s[k][1] for s in sts), reverse=True); pn = sorted((s[k][0] for s in sts), reverse=True)
            ps = sorted((1 - s[k][2] for s in sts), reverse=True)
            f = {"e_py": py[0], "e_pn": pn[0], "e_ps": ps[0], "e_py2": py[1] if len(py) > 1 else 0.0,
                 "e_pn2": pn[1] if len(pn) > 1 else 0.0, "e_ny": sum(x >= 0.5 for x in py), "e_nn": sum(x >= 0.5 for x in pn),
                 "e_sum_s": sum(ps), "e_n": len(sts)}
            for q2 in QIDS: f[f"e_{q2}_y"], f[f"e_{q2}_n"] = mx[q2]
            out[(d, q)] = f
    return out


def decide(f, t, prec="max"):
    py, pn = f["e_py"], f["e_pn"]
    if max(py, pn) < t: return "not_stated"
    if py >= t and pn >= t and prec != "max": return prec
    return "yes" if py >= pn else "no"


def acc(F, G, keys, t, prec=None):
    return sum(decide(F[k], t if not isinstance(t, dict) else t[k[1]], (prec or {}).get(k[1], "max")) == G[k] for k in keys) / max(len(keys), 1)


TS = [x / 100 for x in range(10, 96, 5)]
oof, ho = feats(load(a.oof.split(","), "train")), feats(load([a.ho], "ho"))
Gt, Gh = gold("train"), gold("ho")
kt, kh = [k for k in oof if k in Gt], [k for k in ho if k in Gh]
print(f"out-of-fold: {len({k[0] for k in kt})} training NDAs; nda_ho: {len({k[0] for k in kh})} NDAs")
t1 = max(TS, key=lambda t: acc(oof, Gt, kt, t))
print(f"R1 T={t1}: training OOF {acc(oof, Gt, kt, t1):.2%} | held-out {acc(ho, Gh, kh, t1):.2%}")
t2 = {q: max(TS, key=lambda t: acc(oof, Gt, [k for k in kt if k[1] == q], t)) for q in QIDS}
print(f"R2 T per question: training OOF {acc(oof, Gt, kt, t2):.2%} | held-out {acc(ho, Gh, kh, t2):.2%}")
pr = {}
for q in QIDS:
    kq = [k for k in kt if k[1] == q]
    pr[q] = max(("max", "yes", "no"), key=lambda p: (acc(oof, Gt, kq, t2[q], {q: p}), p == "max"))
print(f"R3 + precedence {({q: p for q, p in pr.items() if p != 'max'})}: training OOF {acc(oof, Gt, kt, t2, pr):.2%} | "
      f"held-out {acc(ho, Gh, kh, t2, pr):.2%}")
print("   per question held-out (R3):", ", ".join(f"{q} {acc(ho, Gh, [k for k in kh if k[1] == q], t2, pr):.0%}" for q in QIDS))
os.makedirs(f"{HERE}/../feats", exist_ok=True)
for name, F in (("train", oof), ("ho", ho)):
    with open(f"{HERE}/../feats/ndaenc_{name}.jsonl", "w") as fo:
        for (d, q), f in F.items(): fo.write(json.dumps({"id": f"{d}/{q}", **f}) + "\n")
