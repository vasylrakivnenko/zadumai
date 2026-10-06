"""Jev judge, step 4 (2026-10-06): how well does Jev grade LAB criteria, and with what confidence band?
Labels: Ivo Sage's records (3 judge passes per criterion: label = majority; "one LLM pass" = judge pass 1, what LAB
itself runs) and our Kimi-K3-graded pilot runs (one pass). Episodes without deliverables are excluded (trivial fails).
Split by task (crc32 % 2): fit the combiner and the confidence band on one half, report on the other.
Combiner: logistic regression on [p(whole criterion), p(checks combined by their logic, FAIL checks flipped),
mean / min / max of the checks, whole-document flag, retrieved-passages flag]. Band: Jev decides when p_pass >= hi or
<= lo; otherwise the criterion is escalated to an LLM judge (simulated with judge pass 1).
usage: dspy_venv/bin/python jev_judge/evaluate.py -> prints, data/jev_judge/eval.json"""
import json, os, zlib
import numpy as np
from sklearn.linear_model import LogisticRegression
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/jev_judge")


def feats(a):
    c = [p if pi else 1 - p for p, pi in zip(a["checks"], a["pass_if"])]
    pos = [p for p, pi in zip(a["checks"], a["pass_if"]) if pi] or [1.0]
    neg = [1 - p for p, pi in zip(a["checks"], a["pass_if"]) if not pi] or [1.0]
    logic = (max(pos) if a["logic"] == "any" else min(pos)) * min(neg)
    return [a["whole"], logic, float(np.mean(c)), min(c), max(c), float(a["whole_document"]), float(a["mode"] == "retrieved")]


def load():
    items = [json.loads(l) for l in open(f"{D}/items.jsonl")]
    A = {(r["text"], r["task"], r["cid"]): r for r in map(json.loads, open(f"{D}/jev_answers.jsonl"))}
    rows = []
    for it in items:
        if it["empty"]: continue
        a = A.get((it["text"], it["task"], it["cid"]))
        v = [x == "pass" for x in it["verdicts"] if x in ("pass", "fail")]
        if a is None or not v: continue
        rows.append({"src": it["src"], "task": it["task"], "episode": it["episode"], "x": feats(a), "y": sum(v) * 2 > len(v),
                     "one": v[0], "votes": v, "whole_document": a["whole_document"], "mode": a["mode"]})
    return rows


def band(p, y, target):
    """the widest band (most criteria decided by Jev) whose decided criteria agree with y at >= target."""
    best = (0.5, 0.5, 0.0, 0.0)
    for lo in np.arange(0.0, 0.51, 0.02):
        for hi in np.arange(0.5, 1.01, 0.02):
            m = (p <= lo) | (p >= hi)
            if m.sum() < 20: continue
            acc = float(((p >= hi) == y)[m].mean())
            if acc >= target and m.mean() > best[2]: best = (float(lo), float(hi), float(m.mean()), acc)
    return best


def report(name, R, clf, lo, hi):
    X = np.array([r["x"] for r in R]); y = np.array([r["y"] for r in R]); one = np.array([r["one"] for r in R])
    p = clf.predict_proba(X)[:, 1]; jv = p >= 0.5; m = (p <= lo) | (p >= hi)
    hyb = np.where(m, p >= hi, one)
    k = lambda a, b: (np.mean(a == b) - (np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))) / max(1e-9, 1 - (np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))))
    eps = {}
    for r, h, j in zip(R, hyb, jv): eps.setdefault(r["episode"], []).append((r["y"], h, r["one"], j))
    mae = lambda i: float(np.mean([abs(np.mean([e[0] for e in v]) - np.mean([e[i] for e in v])) for v in eps.values()]))
    out = {"criteria": len(R), "episodes": len(eps), "label_pass_rate": round(float(y.mean()), 3),
           "jev_alone_agreement": round(float((jv == y).mean()), 4), "jev_alone_kappa": round(float(k(jv, y)), 3),
           "decided_by_jev": round(float(m.mean()), 3), "agreement_when_decided": round(float(((p >= hi) == y)[m].mean()), 4) if m.any() else None,
           "hybrid_agreement": round(float((hyb == y).mean()), 4), "hybrid_kappa": round(float(k(hyb, y)), 3),
           "one_llm_pass_agreement": round(float((one == y).mean()), 4), "one_llm_pass_kappa": round(float(k(one, y)), 3),
           "task_score_mae_hybrid": round(mae(1), 4), "task_score_mae_one_llm_pass": round(mae(2), 4), "task_score_mae_jev_alone": round(mae(3), 4)}
    print(f"\n{name}: " + json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    R = load(); rec = [r for r in R if r["src"] == "rec"]; kim = [r for r in R if r["src"] == "kimi"]
    cal = [r for r in rec if zlib.crc32(r["task"].encode()) % 2 == 0]; test = [r for r in rec if zlib.crc32(r["task"].encode()) % 2 == 1]
    # judges vs judges: how often one pass agrees with the majority / with another pass (the ceiling)
    three = [r for r in rec if len(r["votes"]) == 3]
    pair = np.mean([np.mean([r["votes"][i] == r["votes"][j] for i, j in ((0, 1), (0, 2), (1, 2))]) for r in three])
    print(f"items: {len(R)} (records {len(rec)}: calibration {len(cal)} / test {len(test)}; kimi {len(kim)}); "
          f"judge pass-to-pass agreement {pair:.4f}; one pass vs majority {np.mean([r['one'] == r['y'] for r in three]):.4f}")
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.array([r["x"] for r in cal]), np.array([r["y"] for r in cal]))
    pc = clf.predict_proba(np.array([r["x"] for r in cal]))[:, 1]; yc = np.array([r["y"] for r in cal])
    target = float(np.mean([r["one"] == r["y"] for r in cal]))  # Jev's decided criteria must agree at least as often as one LLM pass
    lo, hi, cov, acc = band(pc, yc, target)
    print(f"calibration: band p <= {lo:.2f} fail / p >= {hi:.2f} pass decides {cov:.1%} at {acc:.4f} agreement (target = one LLM pass, {target:.4f})")
    print("coefficients [whole, logic, mean, min, max, whole_doc, retrieved]:", np.round(clf.coef_[0], 2).tolist())
    res = {"band": {"lo": lo, "hi": hi, "target": target}, "pass_to_pass": float(pair)}
    res["test_records"] = report("TEST (records, tasks never used to fit)", test, clf, lo, hi)
    if kim: res["kimi_runs"] = report("OUR RUNS (Kimi K3 labels, one pass: escalation = the label itself, so hybrid is optimistic)", kim, clf, lo, hi)
    for sub, f in (("whole-document criteria", lambda r: r["whole_document"]), ("retrieved passages", lambda r: r["mode"] == "retrieved")):
        S = [r for r in test if f(r)]
        if S: res[f"test_{sub}"] = report(f"TEST subset: {sub}", S, clf, lo, hi)
    json.dump(res, open(f"{D}/eval.json", "w"), indent=1)
