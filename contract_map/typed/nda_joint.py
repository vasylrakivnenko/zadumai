"""Cross-question logic for NDAs (2026-10-04; the user wants a neuro-symbolic approach, not a bigger model).
1. The stacker's probabilities per (NDA, question): out-of-fold on the 384 training NDAs, a model on all of them for
   the 60 held-out NDAs (installed-network features, the NDA compiler extras, the question's identity).
2. Rules mined from training LABELS only: "a = la => b = lb" with confidence >= C_MIN and >= LIFT above b's base rate,
   each weighted log(confidence / base rate).
3. Joint decoding per NDA (iterated conditional modes): each question's answer maximizes
   log p(answer) + lam * sum of the weights of the rules whose premise holds in the current answers and whose
   conclusion is that answer. lam picked by 5-fold CV on the training NDAs.
usage: venv_xgb python nda_joint.py"""
import collections, json, math, os, zlib
import numpy as np
import xgb_stack as S, xgb_full as F

HERE = os.path.dirname(os.path.abspath(__file__))
C = S.CLASSES


def load_train():
    rows = S.load("nda_tune", "")
    for k in range(3):
        rows += [dict(json.loads(l), set="nda_more") for l in open(f"{HERE}/feats/nda_more_more{k}.jsonl")]
    return rows


def mine(rows, c_min, lift):
    lab = collections.defaultdict(dict)
    for o in rows:
        d, q = o["id"].split("/"); lab[d][q] = o["gold"]
    docs = list(lab.values()); Q = sorted({q for d in docs for q in d})
    base = {(b, lb): sum(d.get(b) == lb for d in docs) / len(docs) for b in Q for lb in C}
    rules = []
    for a in Q:
        for la in C:
            prem = [d for d in docs if d.get(a) == la]
            if len(prem) < 20: continue
            for b in Q:
                if b == a: continue
                for lb in C:
                    conf = sum(d.get(b) == lb for d in prem) / len(prem)
                    if conf >= c_min and conf - base[(b, lb)] >= lift:
                        rules.append((a, la, b, lb, math.log(conf / max(base[(b, lb)], 1e-3))))
    return rules


def decode(probs, rules, lam, iters=5):
    """probs: {q: [p_yes, p_no, p_ns]} for one NDA -> {q: label}"""
    cur = {q: C[int(np.argmax(p))] for q, p in probs.items()}
    by_b = collections.defaultdict(list)
    for a, la, b, lb, w in rules: by_b[b].append((a, la, lb, w))
    for _ in range(iters):
        changed = False
        for q, p in probs.items():
            score = {l: math.log(max(p[k], 1e-6)) for k, l in enumerate(C)}
            for a, la, lb, w in by_b.get(q, []):
                if cur.get(a) == la: score[lb] += lam * w
            best = max(score, key=score.get)
            if best != cur[q]: cur[q] = best; changed = True
        if not changed: break
    return cur


def acc(rows, P, rules, lam):
    by = collections.defaultdict(dict); gold = {}
    for o, p in zip(rows, P):
        d, q = o["id"].split("/"); by[d][q] = p; gold[o["id"]] = o
    ok = 0
    for d, probs in by.items():
        for q, l in decode(probs, rules, lam).items(): ok += S.right(gold[f"{d}/{q}"], l)
    return ok / len(rows)


def main():
    ex = F.extra(); train = load_train(); test = S.load("nda_ho", "")
    qids = sorted({o["qid"] for o in train})
    X = F.features(train, qids, ex, True, True); y = np.array([S.target(o) for o in train])
    folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
    oof = np.zeros((len(train), 3))
    for f in range(5):
        oof[folds == f] = S.fit(X[folds != f], y[folds != f]).predict_proba(X[folds == f])
    Pte = S.fit(X, y).predict_proba(F.features(test, qids, ex, True, True))
    print(f"stacker alone: training CV {acc(train, oof, [], 0):.1%}, held-out {acc(test, Pte, [], 0):.1%}")
    best = None
    for c_min, lift in ((0.95, 0.15), (0.9, 0.1), (0.85, 0.1), (0.8, 0.05)):
        for lam in (0.1, 0.25, 0.5, 1.0):
            ok = n = 0  # rules mined on 4/5 of the training NDAs, scored on the other 1/5
            for f in range(5):
                tr = [o for o, k in zip(train, folds) if k != f]; te_i = [i for i, k in enumerate(folds) if k == f]
                rules = mine(tr, c_min, lift)
                ok += acc([train[i] for i in te_i], oof[te_i], rules, lam) * len(te_i); n += len(te_i)
            r = ok / n
            print(f"   rules conf >= {c_min}, lift >= {lift}, lam {lam}: training CV {r:.1%}")
            if best is None or r > best[0]: best = (r, c_min, lift, lam)
    r, c_min, lift, lam = best
    rules = mine(train, c_min, lift)
    print(f"picked by CV: conf >= {c_min}, lift >= {lift}, lam {lam} ({len(rules)} rules) -> held-out {acc(test, Pte, rules, lam):.1%}")


if __name__ == "__main__":
    main()
