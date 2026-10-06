"""XGBoost on top of the engine (2026-10-04; the user: "How about training xgboost on top of our engine?").
Every signal extract.py saved per question (Pre-Tier 0 whole / located, the network's candidate and probabilities, the
related questions, the priority and typed checks, question type, document kind and length) -> yes / no / not_stated.
Trained on the tuning sets only (thresholds from 5-fold out-of-fold predictions, folds by document); scored on the
held-out sets. Modes: "yn" answers yes/no only (a not_stated prediction abstains); "3way" may also answer "the text
doesn't say" (right on gold not_stated; on CUAD, LegalBench-style, gold "no" = the clause type is absent, so a
not_stated answer is right there too). usage: venv_xgb python xgb_stack.py [--tag ft]"""
import argparse, collections, json, math, os, zlib
import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
TUNE = ["nda_tune", "cuad_tune", "wc1_open", "short_tune"]
HELD = ["nda_ho", "cuad_ho", "short_ho", "wc1_sealed"]
CLASSES = ["yes", "no", "not_stated"]


def load(name, tag):
    p = f"{HERE}/feats/{name}{('_' + tag) if tag else ''}.jsonl"
    rows, seen = [], set()
    for l in open(p):
        o = json.loads(l)
        if o["id"] in seen: continue
        seen.add(o["id"]); o["set"] = name; rows.append(o)
    return rows


def domain(o):
    return "nda" if o["set"].startswith("nda") else "commercial" if o["set"].startswith(("cuad", "wc1")) else "short"


def rel(o, k):
    for r in o.get("rel") or []:
        if r["kind"] == k: return r
    return {}


CAT = {"pt0": ["yes", "no", None], "pt0_loc": ["yes", "no", None], "net": ["yes", "no", None],
       "asks": ["CAN", "MUST", "PROHIBITED", "DOES", "EXISTS", "PROPERTY", "untyped"], "domain": ["nda", "commercial", "short"]}


def vec(o):
    asks = (o.get("frame") or {}).get("asks") or "untyped"
    pr, prn = o.get("prio") or {}, o.get("prio_no") or {}
    opp = rel(o, "ban") or rel(o, "perm")
    v = []
    for k, vals in CAT.items():
        x = {"asks": asks, "domain": domain(o)}.get(k, o.get(k))
        v += [float(x == c) for c in vals]
    v += [float(o.get("net_p") or 0), float(o.get("q_pyes") or 0), float(o.get("q_pno") or 0),
          float(opp.get("pyes", -1)), float(opp.get("pno", -1)), float(rel(o, "para").get("pyes", -1)), float(rel(o, "para").get("pno", -1)),
          float(o.get("long", False)), float(o.get("n_units") or 0), float(pr.get("pr_conflicts", 0)),
          float(bool(pr.get("pr_overridden") or pr.get("pr_unresolved") or pr.get("pr_defers_to_conflict"))),
          float(prn.get("pr_ban", 0)), float(prn.get("pr_permit", 0)), float(bool(prn.get("pr_ban_unexcepted"))),
          float(bool(pr.get("pr_ev_found"))), {True: 1.0, False: 0.0}.get(o.get("typed_ok"), -1.0), float(bool(o.get("net_checks"))),
          float(o.get("net_label") == "yes"), float(o.get("net_label") == "no")]
    return v


def target(o):
    return CLASSES.index(o["gold"])


def right(o, a):
    if a == o["gold"]: return True
    return a == "not_stated" and o["set"].startswith("cuad") and o["gold"] == "no"


def fit(X, y):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                          objective="multi:softprob", num_class=3, eval_metric="mlogloss", n_jobs=int(os.environ.get("XGB_JOBS", "6")), verbosity=0)
    return m.fit(X, y)


def decide(P, mode, t):
    out = []
    for p in P:
        k = int(np.argmax(p)) if mode == "3way" else int(np.argmax(p[:2]))
        conf = p[k] if mode == "3way" else p[k]
        out.append(CLASSES[k] if conf >= t else None)
    return out


def pick(oof_rows, P, mode, target_prec):
    best = 1.01
    for t in np.arange(0.99, 0.30, -0.005):
        a = decide(P, mode, t); ans = [(o, x) for o, x in zip(oof_rows, a) if x]
        if len(ans) >= 30 and sum(right(o, x) for o, x in ans) / len(ans) >= target_prec: best = t
        elif len(ans) >= 30: break
    return best


def lo(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d


def base(o, tag, th):
    if o.get("pt0"): return o["pt0"]
    if o.get("pt0_loc"): return o["pt0_loc"]
    t = th["long"] if o["long"] else th["short"]
    return "yes" if o.get("net") == "yes" and float(o.get("net_p") or 0) >= t else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    tune = [o for s in TUNE for o in load(s, a.tag)]
    X = np.array([vec(o) for o in tune]); y = np.array([target(o) for o in tune])
    folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in tune])
    oof = np.zeros((len(tune), 3))
    for f in range(5):
        m = fit(X[folds != f], y[folds != f]); oof[folds == f] = m.predict_proba(X[folds == f])
    model = fit(X, y)
    th_base = {"long": 0.951, "short": 0.948} if a.tag == "ft" else {"long": 0.90, "short": 0.935}
    print(f"network: {'retrained' if a.tag == 'ft' else 'installed'}; tuning rows {len(tune)}")
    for mode in ("yn", "3way"):
        for prec in (0.96, 0.98):
            t = pick(tune, oof, mode, prec)
            print(f"\n  {mode} at {prec:.0%} on tuning (threshold {t:.3f}; tuning out-of-fold coverage "
                  f"{sum(1 for x in decide(oof, mode, t) if x) / len(tune):.1%})")
            for s in HELD:
                rows = load(s, a.tag)
                P = model.predict_proba(np.array([vec(o) for o in rows]))
                ans = [(o, x) for o, x in zip(rows, decide(P, mode, t)) if x]
                ok = sum(right(o, x) for o, x in ans)
                b = [(o, base(o, a.tag, th_base)) for o in rows]; bans = [(o, x) for o, x in b if x]; bok = sum(right(o, x) for o, x in bans)
                kinds = collections.Counter(x for _, x in ans)
                print(f"    {s:10s} xgb {len(ans):5d}/{len(rows)} = {len(ans) / len(rows):5.1%} @ {ok / max(1, len(ans)):6.1%} [lo {lo(ok, len(ans)):.3f}] "
                      f"{dict(kinds)} | today's rule {len(bans) / len(rows):5.1%} @ {bok / max(1, len(bans)):6.1%}")


if __name__ == "__main__":
    main()
