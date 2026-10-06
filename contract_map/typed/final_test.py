"""The ONE-TIME final check on the sealed sets (2026-10-04; never used for any choice before): ContractNLI test (123
NDAs, its official test split: the SCROLLS ContractNLI data; accuracy = SCROLLS exact match) and s160 (40 unused training
NDAs), plus LegalBench's 14 contract_nli_* excerpt tasks (1,927 samples, all drawn from ContractNLI test).
Systems (all trained on the 384 training NDAs; the CNN on those + the SEC NDAs with teacher labels):
  CNN alone     the final CNN, 5 seeds averaged (models trained on all training NDAs, ndacnn/preds/<name>_test.jsonl)
  local         stacker on lite (question id + IR answer + topic hits) + the CNN; no LLM at run time
  + Jev         the same + Jev's numbers (one request per NDA)
Metrics: 3-way accuracy; F1 of Entailment and Contradiction and their mean, in ContractNLI's own label space (nda-15's
user-style question is flipped back). LegalBench: accuracy and balanced accuracy per task (Yes = the excerpt supports
the statement), the CNN alone reading the excerpt as a document.
usage: python final_test.py CNN_NAME [--jev]   (live venv for the Jev requests and the LegalBench CNN; venv_xgb for the
       stacker: run `final_test.py CNN_NAME stack` with venv_xgb after the first step)"""
import collections, json, os, random, sys, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
QIDS = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
        "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
CL = ["no", "yes", "not_stated"]  # the CNN's order
LB = {"contract_nli_confidentiality_of_agreement": "nda-10", "contract_nli_explicit_identification": "nda-1",
      "contract_nli_inclusion_of_verbally_conveyed_information": "nda-3", "contract_nli_limited_use": "nda-4",
      "contract_nli_no_licensing": "nda-15", "contract_nli_notice_on_compelled_disclosure": "nda-8",
      "contract_nli_permissible_acquirement_of_similar_information": "nda-13", "contract_nli_permissible_copy": "nda-17",
      "contract_nli_permissible_development_of_similar_information": "nda-12",
      "contract_nli_permissible_post-agreement_possession": "nda-20", "contract_nli_return_of_confidential_information": "nda-16",
      "contract_nli_sharing_with_employees": "nda-5", "contract_nli_sharing_with_third-parties": "nda-7",
      "contract_nli_survival_of_obligations": "nda-19"}
LBDIR = "/root/zadumai_nli_proto/extensive/hf/data"
SEEDS = ["", "+s1", "+s2", "+s3", "+s4"]


def sealed_docs():
    z = zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip")
    tr = json.loads(z.read("contract-nli/train.json"))["documents"]; random.Random(5).shuffle(tr)
    te = json.loads(z.read("contract-nli/test.json"))["documents"]
    return {"test": te, "s160": tr[160:200]}


def gold(docs):
    """(doc, qid) -> the user-style answer (eval_docs.POLARITY: nda-15 flipped), as everywhere in this work."""
    default, polarity = {"Entailment": "yes", "Contradiction": "no"}, {"nda-15": {"Entailment": "no", "Contradiction": "yes"}}  # = eval_docs
    out = {}
    for d in docs:
        ann = d["annotation_sets"][0]["annotations"]
        for q in QIDS:
            ch = ann[q]["choice"]
            out[(str(d["id"]), q)] = "not_stated" if ch == "NotMentioned" else polarity.get(q, default)[ch]
    return out


def jev(docs, out):
    """Jev v1 (jev_nda.py's request) for the sealed NDAs; resumable."""
    import concurrent.futures as cf
    sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
    from router.systemone import SystemOne
    import eval_docs as E
    key = next(l.split("=", 1)[1].strip() for l in open("/root/projects/zadumai/.env") if l.startswith("JEV_API="))
    llm = SystemOne("https://api.typesafe.ai/v1/systemone", "jev-latest", api_key=key)
    ch = {"yes": "the NDA says yes", "no": "the NDA says no", "not stated": "the NDA does not address this question"}
    done = {json.loads(l)["id"].split("/")[0] for l in open(out)} if os.path.exists(out) else set()

    def ask(d):
        qs = {}
        for q in QIDS:
            qs[q] = {"type": "noul", "instructions": E.QUESTIONS[q], "criteria": {"true": "", "false": ""}}
            qs[q + "#3"] = {"type": "choice", "instructions": E.QUESTIONS[q], "criteria": ch}
        return str(d["id"]), llm.ask(d["text"], qs)
    with cf.ThreadPoolExecutor(4) as ex, open(out, "a") as f:
        for did, ans in ex.map(ask, [d for d in docs if str(d["id"]) not in done]):
            for q in QIDS:
                c = ans.get(q + "#3", {})
                f.write(json.dumps({"id": f"{did}/{q}", "p_true": float(ans[q]["noul"]), "choice": c.get("choice"), "probs": c.get("probabilities", {})}) + "\n")
    print("Jev calls", llm.calls, flush=True)


def legalbench(name):
    """The final CNN (5 seeds, the models trained on all training NDAs) on LegalBench's excerpts."""
    import csv, pickle, torch
    sys.path.insert(0, f"{HERE}/ndacnn")
    import build as B
    src = open(f"{HERE}/ndacnn/train.py").read(); src = src[:src.index("net = Net()")]
    rows = []
    for task in LB:
        for r in csv.DictReader(open(f"{LBDIR}/{task}/test.tsv"), delimiter="\t"): rows.append((task, r))
    docs = [B.one(({"id": f"lb{k}", "text": r["text"], "spans": [], "annotation_sets": [{"annotations": {q: {"choice": "NotMentioned", "spans": []} for q in QIDS}}]}, "lb"))[1]
            for k, (task, r) in enumerate(rows)]
    probs = [[0.0] * 3 for _ in rows]
    for sfx in SEEDS:
        ck = torch.load(f"{HERE}/ndacnn/models/{name}{sfx}_f-1.pt", weights_only=False); a = ck["args"]
        argv = ["train.py", "--variant", a["variant"], "--fold", "-1", "--k", str(a["k"]), "--dim", str(a["dim"]), "--ch", str(a["ch"]), "--threads", "8"]
        if a.get("extra"): argv += ["--extra", f"{HERE}/ndamore/data"]  # the same extended vocabulary as in training
        sys.argv = argv; g = {"__file__": f"{HERE}/ndacnn/train.py", "__name__": "final"}; exec(compile(src, "train.py", "exec"), g)
        net = g["Net"](); net.load_state_dict(ck["state"]); net.eval()
        with torch.inference_mode():
            for i, ((task, r), d) in enumerate(zip(rows, docs)):
                if not len(d["st"]): continue
                p = net(d, g["ids_of"](d))[0].softmax(-1)[QIDS.index(LB[task])].tolist()
                probs[i] = [x + y / len(SEEDS) for x, y in zip(probs[i], p)]
    per = collections.defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for (task, r), p in zip(rows, probs):
        q = LB[task]; ans = CL[max(range(3), key=lambda k: p[k])]
        supported = (ans == "no") if q == "nda-15" else (ans == "yes")  # nda-15: "supports no licensing" = our "no license"
        gold_yes = r["answer"].strip().lower() == "yes"; c = per[task]
        c["tp" if gold_yes and supported else "fn" if gold_yes else "fp" if supported else "tn"] += 1
    out = {}
    for task, c in per.items():
        n = sum(c.values()); acc = (c["tp"] + c["tn"]) / n
        tpr = c["tp"] / max(c["tp"] + c["fn"], 1); tnr = c["tn"] / max(c["tn"] + c["fp"], 1)
        out[task] = {"n": n, "acc": acc, "balanced_acc": (tpr + tnr) / 2}
    return out


def stack(name):
    import numpy as np, zlib
    import xgb_stack as S, xgb_full as F, nda_joint as J
    train = J.load_train(); ex = F.extra()
    sealed_ex = {json.loads(l)["id"]: json.loads(l) for l in open(f"{HERE}/feats/nda_extra_sealed.jsonl")}
    jl = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
    jev_tr = {**jl(f"{HERE}/feats/jev_nda_tune.jsonl"), **jl(f"{HERE}/feats/jev_nda_more.jsonl")}
    jev_se = jl(f"{HERE}/feats/jev_sealed.jsonl")

    def ens(split_suffixes):
        acc = {}
        for sfx in SEEDS:
            for fs in split_suffixes:
                for o in map(json.loads, open(f"{HERE}/ndacnn/preds/{name}{sfx}_{fs}.jsonl")):
                    acc.setdefault(o["id"], []).append(o["p"])
        return {k: list(np.mean(v, 0)) for k, v in acc.items()}
    cnn_tr = ens([f"f{f}" for f in range(5)])
    docs = sealed_docs(); G = {s: gold(d) for s, d in docs.items()}
    cnn_se = {s: ens([s]) for s in docs}

    def lite(qid, e): return [float(qid == q) for q in QIDS] + [float(e.get("ir") == c) for c in F.IR] + \
        [float(e.get("topic_hits", -1)), float(e.get("topic_hits", -1) == 0)]
    def jv(j): return [j["p_true"]] + [j["probs"].get(k, np.nan) for k in ("yes", "no", "not stated")] if j else [np.nan] * 4
    y = np.array([S.target(o) for o in train])
    res = {}
    for cfg in ("cnn_alone", "local", "jev"):
        if cfg == "jev" and not jev_se: continue
        if cfg != "cnn_alone":
            X = np.array([lite(o["qid"], ex.get(o["id"], {})) + cnn_tr[o["id"]] + (jv(jev_tr.get(o["id"])) if cfg == "jev" else []) for o in train])
            m = S.fit(X, y)
        for s, d in docs.items():
            keys = [(str(x["id"]), q) for x in d for q in QIDS]
            if cfg == "cnn_alone":
                pred = [CL[int(np.argmax(cnn_se[s][f"{k}/{q}"]))] for k, q in keys]
            else:
                Xs = np.array([lite(q, sealed_ex.get(f"{k}/{q}", {})) + cnn_se[s][f"{k}/{q}"] + (jv(jev_se.get(f"{k}/{q}")) if cfg == "jev" else []) for k, q in keys])
                pred = [S.CLASSES[int(np.argmax(p))] for p in m.predict_proba(Xs)]
            gl = [G[s][kq] for kq in keys]
            # back to ContractNLI's labels: E / C / N (nda-15 flipped)
            to_nli = lambda a, q: "N" if a == "not_stated" else (("C" if a == "yes" else "E") if q == "nda-15" else ("E" if a == "yes" else "C"))
            P = [to_nli(a, q) for a, (_, q) in zip(pred, keys)]; Gs = [to_nli(a, q) for a, (_, q) in zip(gl, keys)]
            f1 = {}
            for lab in ("E", "C"):
                tp = sum(p == g == lab for p, g in zip(P, Gs)); fp = sum(p == lab != g for p, g in zip(P, Gs)); fn = sum(g == lab != p for p, g in zip(P, Gs))
                f1[lab] = 2 * tp / max(2 * tp + fp + fn, 1)
            acc = np.mean([p == g for p, g in zip(P, Gs)])
            per_doc = collections.defaultdict(list)
            for (k, _), p, g in zip(keys, P, Gs): per_doc[k].append(p == g)
            m_ = np.array([np.mean(v) for v in per_doc.values()]); rng = np.random.default_rng(0)
            bs = sorted(np.mean(rng.choice(m_, len(m_))) for _ in range(2000))
            res[f"{cfg} | {s}"] = {"n": len(keys), "acc": acc, "ci": (bs[50], bs[1949]), "F1_E": f1["E"], "F1_C": f1["C"], "F1_mean": (f1["E"] + f1["C"]) / 2}
    for k, v in res.items():
        print(f"{k:22s} {v['n']:5d} pairs | accuracy {v['acc']:.2%} ({v['ci'][0]:.1%}-{v['ci'][1]:.1%}) | F1 E {v['F1_E']:.3f} C {v['F1_C']:.3f} mean {v['F1_mean']:.3f}")
    json.dump(res, open(f"{HERE}/runs/final_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    name = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "stack": stack(name)
    else:
        if "--jev" in sys.argv:
            d = sealed_docs(); jev(d["test"] + d["s160"], f"{HERE}/feats/jev_sealed.jsonl")
        lb = legalbench(name)
        for t, v in sorted(lb.items()): print(f"{t:62s} n {v['n']:4d} | accuracy {v['acc']:.1%} | balanced {v['balanced_acc']:.1%}")
        print(f"LegalBench contract_nli (14 tasks): mean accuracy {sum(v['acc'] for v in lb.values()) / len(lb):.1%} | "
              f"mean balanced accuracy {sum(v['balanced_acc'] for v in lb.values()) / len(lb):.1%}")
        json.dump(lb, open(f"{HERE}/runs/final_legalbench.json", "w"), indent=1)
