"""Teacher labels for the new SEC NDAs (2026-10-04, step 2). The teacher = the stacker on what a new NDA can give:
question id + the NDA compiler's IR answer + topic-word hits ("lite") + 2 CNNs (rising, 128-d, width 5 and 3) + Jev
(final_eval.py: CV 90.92%, held-out 93.04%). Per fold F (0-4: the teacher never saw the training NDAs of fold F, so
the student's out-of-fold check stays honest; -1: all 384, for held-out) it gives soft labels per (new NDA, question).
  features  (live venv) IR + topic hits (nda_extra.one); the CNN data (ndacnn/build.one on an unlabeled NDA); Jev: one
            request per NDA, the 17 Nouls + 17 3-way Choices of jev_nda.py + "is this an NDA?"
  cnn       (live venv) the saved teacher CNNs (ndacnn/models/<name>_f<F>.pt) -> data/cnn_<name>.jsonl, per fold
  teach     (venv_xgb) per fold: the stacker trained on the training NDAs outside the fold -> data/soft.jsonl
usage: python label.py features|cnn   then  venv_xgb python label.py teach"""
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(HERE)
sys.path.insert(0, T)
QIDS = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
        "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
CNNS = ["rising+d128+k5", "rising+d128+k3"]
FOLDS = [0, 1, 2, 3, 4, -1]


def ndas():
    return [dict(json.loads(l), doc=f"sec{k}") for k, l in enumerate(open(f"{HERE}/data/ndas.jsonl"))]


def features():
    import concurrent.futures as cf
    sys.path.insert(0, f"{T}/ndacnn"); sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
    import nda_extra, build as B, eval_docs as E
    from router.systemone import SystemOne
    docs = ndas()
    # 1. IR + topic hits
    with cf.ProcessPoolExecutor(4) as ex, open(f"{HERE}/data/extra.jsonl", "w") as f:
        for rows in ex.map(nda_extra.one, [(d["doc"], d["text"], QIDS) for d in docs], chunksize=4):
            for r in rows: f.write(json.dumps(r) + "\n")
    print("IR + topic hits done", flush=True)
    # 2. CNN data (tokens, statements, graph, clauses); no labels
    fake = lambda d: {"id": d["doc"], "text": d["text"], "spans": [],
                      "annotation_sets": [{"annotations": {q: {"choice": "NotMentioned", "spans": []} for q in QIDS}}]}
    with cf.ProcessPoolExecutor(4) as ex:
        data = [r for _, r in ex.map(B.one, [(fake(d), "new") for d in docs], chunksize=4)]
    for r in data: r["fold"] = -2
    pickle.dump(data, open(f"{HERE}/data/new.pkl", "wb")); print("CNN data done", len(data), flush=True)
    # 3. Jev, one request per NDA (resumable)
    out = f"{HERE}/data/jev.jsonl"
    done = {json.loads(l)["doc"] for l in open(out)} if os.path.exists(out) else set()
    key = next(l.split("=", 1)[1].strip() for l in open("/root/projects/zadumai/.env") if l.startswith("JEV_API="))
    llm = SystemOne("https://api.typesafe.ai/v1/systemone", "jev-latest", api_key=key)
    ch = {"yes": "the NDA says yes", "no": "the NDA says no", "not stated": "the NDA does not address this question"}

    def ask(d):
        qs = {"isnda": {"type": "noul", "instructions": "Is this document a non-disclosure or confidentiality agreement: a contract "
                        "whose main purpose is to protect confidential information shared between the parties?", "criteria": {"true": "", "false": ""}}}
        for q in QIDS:
            qs[q] = {"type": "noul", "instructions": E.QUESTIONS[q], "criteria": {"true": "", "false": ""}}
            qs[q + "#3"] = {"type": "choice", "instructions": E.QUESTIONS[q], "criteria": ch}
        try: return d["doc"], llm.ask(d["text"], qs)
        except Exception as e: return d["doc"], {"error": str(e)[:200]}  # noqa: BLE001
    with cf.ThreadPoolExecutor(4) as ex, open(out, "a") as f:
        for doc, ans in ex.map(ask, [d for d in docs if d["doc"] not in done]):
            if "error" in ans: print("Jev error", doc, ans["error"], flush=True); continue
            f.write(json.dumps({"doc": doc, "isnda": float(ans["isnda"]["noul"]),
                                "q": {q: {"p_true": float(ans[q]["noul"]), "choice": ans[q + "#3"].get("choice"),
                                          "probs": ans[q + "#3"].get("probabilities", {})} for q in QIDS}}) + "\n")
    print("Jev done; calls", llm.calls, "tokens in", llm.input_tokens, "out", llm.output_tokens, flush=True)


def cnn():
    import torch
    data = pickle.load(open(f"{HERE}/data/new.pkl", "rb"))
    src = open(f"{T}/ndacnn/train.py").read(); src = src[:src.index("net = Net()")]
    for name in CNNS:
        with open(f"{HERE}/data/cnn_{name}.jsonl", "w") as f:
            for F in FOLDS:
                ck = torch.load(f"{T}/ndacnn/models/{name}_f{F}.pt", weights_only=False); a = ck["args"]
                sys.argv = ["train.py", "--variant", a["variant"], "--fold", str(F), "--k", str(a["k"]), "--dim", str(a["dim"]),
                            "--ch", str(a["ch"]), "--threads", "8"]
                g = {"__file__": f"{T}/ndacnn/train.py", "__name__": "teacher"}; exec(compile(src, "train.py", "exec"), g)
                net = g["Net"](); net.load_state_dict(ck["state"]); net.eval()
                with torch.inference_mode():
                    for d in data:
                        p = net(d, g["ids_of"](d))[0].softmax(-1).tolist()
                        f.write(json.dumps({"doc": d["doc"], "fold": F, "p": [[round(x, 4) for x in q] for q in p]}) + "\n")
                print(name, "fold", F, "done", flush=True)


def teach():
    import numpy as np, zlib
    import xgb_stack as S, xgb_full as Fx, nda_joint as J
    jl = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p)}
    train = J.load_train(); ex = Fx.extra()
    jev = {**jl(f"{T}/feats/jev_nda_tune.jsonl"), **jl(f"{T}/feats/jev_nda_more.jsonl")}
    cnn_tr = {n: {} for n in CNNS}
    for n in CNNS:
        for f in range(5): cnn_tr[n].update(jl(f"{T}/ndacnn/preds/{n}_f{f}.jsonl"))
    new = ndas(); nex = jl(f"{HERE}/data/extra.jsonl")
    njev = {json.loads(l)["doc"]: json.loads(l) for l in open(f"{HERE}/data/jev.jsonl")}
    ncnn = {n: {} for n in CNNS}
    for n in CNNS:
        for l in open(f"{HERE}/data/cnn_{n}.jsonl"):
            r = json.loads(l); ncnn[n][(r["doc"], r["fold"])] = r["p"]

    def vec(qid, e, cnn_ps, j):
        return ([float(qid == q) for q in QIDS] + [float(e.get("ir") == c) for c in Fx.IR] +
                [float(e.get("topic_hits", -1)), float(e.get("topic_hits", -1) == 0)] + sum(cnn_ps, []) +
                ([j["p_true"]] + [j["probs"].get(k, np.nan) for k in ("yes", "no", "not stated")] if j else [np.nan] * 4))
    Xtr = np.array([vec(o["qid"], ex.get(o["id"], {}), [cnn_tr[n][o["id"]]["p"] for n in CNNS], jev.get(o["id"])) for o in train])
    y = np.array([S.target(o) for o in train]); folds = np.array([zlib.crc32(o["id"].split("/")[0].encode()) % 5 for o in train])
    rows = [(d["doc"], q) for d in new if d["doc"] in njev for q in QIDS]
    out = {}
    for F in FOLDS:
        m = S.fit(Xtr[folds != F], y[folds != F]) if F >= 0 else S.fit(Xtr, y)
        Xn = np.array([vec(q, nex.get(f"{d}/{q}", {}), [ncnn[n][(d, F)][QIDS.index(q)] for n in CNNS], njev[d]["q"][q]) for d, q in rows])
        P = m.predict_proba(Xn)  # S.CLASSES order: yes, no, not_stated
        for (d, q), p in zip(rows, P): out.setdefault(d, {}).setdefault(str(F), {})[q] = [float(p[1]), float(p[0]), float(p[2])]  # -> CNN order no, yes, ns
    with open(f"{HERE}/data/soft.jsonl", "w") as f:
        for d in new:
            if d["doc"] in out: f.write(json.dumps({"doc": d["doc"], "isnda": njev[d["doc"]]["isnda"], "soft": out[d["doc"]]}) + "\n")
    print("soft labels for", len(out), "new NDAs")


if __name__ == "__main__":
    {"features": features, "cnn": cnn, "teach": teach}[sys.argv[1]]()
