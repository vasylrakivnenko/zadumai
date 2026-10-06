"""100% coverage (2026-10-04; the user: "Let's try to achieve 100% coverage at 90%"): one XGBoost per domain answers
every question with the likeliest of yes / no / not_stated. Features: xgb_stack.vec (the engine's signals) + for
checklist domains the question's identity (one-hot of qid) + for NDAs the NDA compiler's answer and the topic-word hits
(nda_extra.py). Trained on the tuning set of the domain (5-fold CV accuracy there, folds by document), scored on its
held-out set. usage: venv_xgb python xgb_full.py [--tag ft] [--no-qid] [--no-extra]"""
import argparse, collections, json, os, zlib
import numpy as np
import xgb_stack as S

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS = {"nda": ("nda_tune", "nda_ho"), "cuad": ("cuad_tune", "cuad_ho"), "short": ("short_tune", "short_ho"),
           "wc1": ("wc1_open", "wc1_sealed")}
IR = ["yes", "no", "not mentioned", None]


def extra():
    p = f"{HERE}/feats/nda_extra.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def presence():
    p = f"{HERE}/feats/cuad_presence.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def features(rows, qids, ex, use_qid, use_extra, pres=None):
    X = []
    for o in rows:
        v = S.vec(o)
        if pres:
            e = pres.get(o["id"], {})
            v += [float(e.get("pmax", -1)), float(e.get("n50", -1))]
        if use_qid and qids: v += [float(o.get("qid") == q) for q in qids]
        if use_extra and ex:
            e = ex.get(o["id"], {})
            v += [float(e.get("ir") == c) for c in IR] + [float(e.get("topic_hits", -1)), float(e.get("topic_hits", -1) == 0)]
            v += [float(e.get("span_s") if e.get("span_s") is not None else -1), float(e.get("span_pn") if e.get("span_pn") is not None else -1)]
            for name in ("v2", "compiler", "var_c3", "var_c4", "var_oral"):
                v += [float(e.get(name) == c) for c in ("yes", "no", "not mentioned")]
        X.append(v)
    return np.array(X)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--no-qid", action="store_true")
    ap.add_argument("--no-extra", action="store_true"); a = ap.parse_args()
    ex = extra()
    print(f"network {'retrained' if a.tag else 'installed'}; qid {'off' if a.no_qid else 'on'}; NDA extras {'off' if a.no_extra or not ex else 'on'}")
    for dom, (tr, te) in DOMAINS.items():
        train, test = S.load(tr, a.tag), S.load(te, a.tag)
        qids = sorted({o.get("qid") for o in train if o.get("qid")}) if dom in ("nda", "cuad") else []
        e = ex if dom == "nda" else {}
        pr = presence() if dom == "cuad" and not a.no_extra else None
        X, y = features(train, qids, e, not a.no_qid, not a.no_extra, pr), np.array([S.target(o) for o in train])
        folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
        cv_ok = 0
        if len(set(folds)) > 1 and len(train) > 100:
            for f in range(5):
                m = S.fit(X[folds != f], y[folds != f]); P = m.predict_proba(X[folds == f])
                cv_ok += sum(S.right(o, S.CLASSES[int(np.argmax(p))]) for o, p in zip([o for o, k in zip(train, folds) if k == f], P))
        m = S.fit(X, y); P = m.predict_proba(features(test, qids, e, not a.no_qid, not a.no_extra, pr))
        ok = sum(S.right(o, S.CLASSES[int(np.argmax(p))]) for o, p in zip(test, P))
        print(f"  {dom:6s} tuning CV {cv_ok / len(train):6.1%} ({len(train)}) | held-out {te}: {ok / len(test):6.1%} right at 100% coverage ({len(test)})")


if __name__ == "__main__":
    main()
