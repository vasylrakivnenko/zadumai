"""NDA, 100% coverage: every combination of tonight's parts (2026-10-04). Each config = the stacker's features (base) +
blocks: enc (the statement encoder, ndaenc/aggregate.py), cnn:<name> (the CNN, ndacnn/preds: out-of-fold for the
training NDAs, the model on all of them for held-out), jev (Jev v1; asked on 100 training NDAs + all held-out), rules
(the symbolic detectors, gated per question). 5-fold CV on the 384 training NDAs (the stacker's folds = the CNN's), then
held-out (nda_ho, 60 NDAs, bootstrap over NDAs). "alone:cnn:<name>" = the CNN's own answers.
usage: venv_xgb python final_eval.py CONFIG ...   e.g. base base+enc base+cnn:rising base+cnn:rising+jev alone:cnn:rising
       (blocks joined by "," when a CNN name has a "+": base,cnn:rising+graph,jev)"""
import collections, json, os, sys, zlib
import numpy as np
import xgb_stack as S, xgb_full as F, nda_joint as J
HERE = os.path.dirname(os.path.abspath(__file__))
train = J.load_train(); test = S.load("nda_ho", ""); qids = sorted({o["qid"] for o in train}); ex = F.extra()
y = np.array([S.target(o) for o in train]); folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
CL = ["no", "yes", "not_stated"]  # the CNN's order
import pickle  # NDA length in the reader network's tokens (the user: are NDAs over 5k tokens a weak spot?)
NTOK = {d["doc"]: len(d["ids"]) for s in ("train", "ho") for d in pickle.load(open(f"{HERE}/ndacnn/data/{s}.pkl", "rb"))}
LONG = 5000


def jl(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}


_cache = {}


def block(b):
    if b in _cache: return _cache[b]
    if b == "enc":
        d = {**jl(f"{HERE}/feats/ndaenc_train.jsonl"), **jl(f"{HERE}/feats/ndaenc_ho.jsonl")}
        keys = sorted({k for r in d.values() for k in r if k != "id"})
        fn = lambda o: [float(d.get(o["id"], {}).get(k, np.nan)) for k in keys]
    elif b == "jev":
        d = {**jl(f"{HERE}/feats/jev_nda_tune.jsonl"), **jl(f"{HERE}/feats/jev_nda_more.jsonl"), **jl(f"{HERE}/feats/jev_nda_ho.jsonl")}  # all 384 training NDAs since 2026-10-04 10:24 PT
        fn = lambda o: [d[o["id"]]["p_true"]] + [d[o["id"]]["probs"].get(k, np.nan) for k in ("yes", "no", "not stated")] if o["id"] in d else [np.nan] * 4
    elif b.startswith("cnn:"):
        n = b[4:]; d = {}
        for f in (0, 1, 2, 3, 4, -1): d.update(jl(f"{HERE}/ndacnn/preds/{n}_f{f}.jsonl"))
        fn = lambda o: d[o["id"]]["p"] if o["id"] in d else [np.nan] * 3
    elif b == "rules":
        d = jl(f"{HERE}/feats/nda_rules.jsonl"); keys = sorted({k for r in d.values() for k in r if k != "id"})
        fn = lambda o: [float(bool(d.get(o["id"], {}).get(k))) if o["qid"] == "nda-" + k.split("_")[0][1:] else np.nan for k in keys]
    else:
        raise ValueError(b)
    _cache[b] = fn
    return fn


def X(rows, cfg):
    parts = cfg.split(",") if "," in cfg else cfg.split("+")  # "," when a CNN's name has a "+" (rising+graph)
    if parts[0] == "base": M = [F.features(rows, qids, ex, True, True)]
    elif parts[0] == "lite":  # what can be computed for a NEW NDA: question id + the compiler's IR answer + topic-word hits
        M = [np.array([[float(o["qid"] == q) for q in qids] + [float(ex.get(o["id"], {}).get("ir") == c) for c in F.IR] +
                       [float(ex.get(o["id"], {}).get("topic_hits", -1)), float(ex.get(o["id"], {}).get("topic_hits", -1) == 0)] for o in rows])]
    elif parts[0] == "cheap":  # base without the engine's features (S.vec: Pre-Tier 0, located reading, the network; ~24 s
        M = [F.features(rows, qids, ex, True, True)[:, len(S.vec(rows[0])):]]  # per NDA): question id + NDA compiler extras
    else: M = []
    for b in parts[1:] if parts[0] in ("base", "cheap", "lite") else parts:
        M.append(np.array([block(b)(o) for o in rows], dtype=float))
    return np.hstack(M)


def boot(rights, rows):
    docs = collections.defaultdict(list)
    for o, r in zip(rows, rights): docs[o["id"].split("/")[0]].append(r)
    m = np.array([np.mean(v) for v in docs.values()]); rng = np.random.default_rng(0)
    bs = sorted(np.mean(rng.choice(m, len(m))) for _ in range(2000))
    return bs[50], bs[1949]


out = []
for cfg in sys.argv[1:]:
    if cfg.startswith("alone:"):  # a part's own answers (the CNN)
        fn = block(cfg[6:])
        pred = lambda o: CL[int(np.argmax(fn(o)))] if not np.isnan(fn(o)[0]) else None
        rc = [S.right(o, pred(o)) for o in train]; cv = np.mean(rc); rt = [S.right(o, pred(o)) for o in test]
    else:
        Xtr = X(train, cfg); oof = np.zeros((len(train), 3))
        for f in range(5): oof[folds == f] = S.fit(Xtr[folds != f], y[folds != f]).predict_proba(Xtr[folds == f])
        rc = [S.right(o, S.CLASSES[int(np.argmax(p))]) for o, p in zip(train, oof)]; cv = np.mean(rc)
        P = S.fit(Xtr, y).predict_proba(X(test, cfg)); rt = [S.right(o, S.CLASSES[int(np.argmax(p))]) for o, p in zip(test, P)]
    lo, hi = boot(rt, test)
    split = lambda rows, r, long: np.mean([x for o, x in zip(rows, r) if (NTOK.get(o["id"].split("/")[0], 0) > LONG) == long] or [np.nan])
    line = (f"{cfg:42s} training CV {cv:.2%} | held-out {np.mean(rt):.2%} ({lo:.1%}-{hi:.1%}) | <=5k tokens CV {split(train, rc, False):.1%} "
            f"HO {split(test, rt, False):.1%}; >5k CV {split(train, rc, True):.1%} HO {split(test, rt, True):.1%}")
    print(line, flush=True); out.append(line)
with open(f"{HERE}/runs/final_eval.txt", "a") as f:
    f.write("\n".join(out) + "\n")
