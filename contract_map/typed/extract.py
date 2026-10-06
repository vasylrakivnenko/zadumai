"""Feature extraction for the neuro-symbolic combinations (2026-10-03). For every question: Pre-Tier 0 on the whole
document; the text the network reads (the document if short, else the located pieces); Pre-Tier 0 on the located text;
the network's candidate at LOW thresholds (yes and no at 0.5, with its own checks), so any threshold can be replayed
offline; the raw p(yes)/p(no) for the question and its related questions (ns.related); the priority features (ns.priority)
for the candidate's evidence and for a "no"; typed_check. One JSON line per question in feats/SET[_TAG].jsonl.
Sets: nda_tune / nda_ho (100 + 100 disjoint ContractNLI train NDAs), cuad_tune / cuad_ho (40 + 42 of cuadc's dev
contracts), wc1_open, wc1_sealed (spent: secondary), short_tune (v5 + v6 gen rows) / short_ho (v6b + v7 gen rows).
usage: extract.py SET [--tag T] [--model DIR] [--n N]"""
import argparse, json, os, random, sys, time, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_docs as E
import pipeline as P
import ns

V = "/root/zadumai_nli_proto/extensive/v4"


def nda_split(which):
    z = zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip"); d = json.loads(z.read("contract-nli/train.json"))
    docs = d["documents"]; random.Random(5).shuffle(docs)
    keep = docs[:100] if which == "tune" else docs[100:160]  # held-out: 60 NDAs (time; 2026-10-03)
    ids = {x["id"] for x in keep}
    return [r for r in E.nda("train") if int(r["id"].split("/")[0]) in ids]


def nda_more():
    """The other ContractNLI NDAs (train after the first 200 of the shuffle + dev): more training rows for the NDA
    stacker (features with the INSTALLED network only: the retrained one learned from these NDAs)."""
    z = zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip")
    docs = json.loads(z.read("contract-nli/train.json"))["documents"]; random.Random(5).shuffle(docs)
    ids = {x["id"] for x in docs[200:]}
    rows = [r for r in E.nda("train") if int(r["id"].split("/")[0]) in ids]
    return rows + E.nda("dev")


def cuad_split(which):
    rows = E.cuad_dev(n=82, seed=3)
    docs = []
    for r in rows:
        k = r["id"].split("/")[0]
        if k not in docs: docs.append(k)
    keep = set(docs[:40] if which == "tune" else docs[40:65])  # held-out: 25 contracts (time; 2026-10-03)
    return [r for r in rows if r["id"].split("/")[0] in keep]


def short(names):
    out = []
    for n in names:
        for r in json.load(open(f"{V}/{n}_rows.json")):
            if r["part"].startswith("gen"):
                out.append({"id": r["id"], "question": r["question"], "doc": r["premise"], "gold": r["gold"], "shape": r.get("shape")})
    return out


SETS = {"nda_tune": lambda: nda_split("tune"), "nda_ho": lambda: nda_split("ho"), "nda_more": nda_more, "cuad_tune": lambda: cuad_split("tune"),
        "cuad_ho": lambda: cuad_split("ho"), "wc1_open": lambda: E.wc1("open"), "wc1_sealed": lambda: E.wc1("sealed"),
        "short_tune": lambda: short(["v5", "v6"]), "short_ho": lambda: short(["v6b", "v7"])}


def featurize(pipe, low, cached, norms_cache, q, doc) -> dict:
    r0 = P.pt0_check(q, doc)
    frame = (r0.frames or {}).get("frame"); parties = list(r0.parties or [])
    long = len(doc) >= P.LONG_CHARS
    o = {"question": q, "long": long, "pt0": r0.answer if r0.fired else None, "pt0_ev": (r0.evidence or [{}])[0].get("text", "") if r0.fired else "",
         "frame": {k: frame.get(k) for k in ("asks", "actor", "action", "actions")} if frame else None,
         "untyped_reason": (r0.frames or {}).get("reason", "") if not frame else "", "parties": parties}
    if long:
        cd = pipe.compiled(doc)
        fr = type("F", (), {"action": (frame or {}).get("action"), "actions": (frame or {}).get("actions") or []})()
        idx = pipe.locate(q, doc, cd, fr)
        pl = cd.party_line()
        read = (pl + "\n\n" if pl else "") + "\n\n".join(cd.pieces[i]["text"] for i in idx) if idx else ""
        r1 = P.pt0_check(q, read) if read else None
        o["pt0_loc"] = r1.answer if r1 is not None and r1.fired else None
        o["pt0_loc_ev"] = (r1.evidence or [{}])[0].get("text", "") if r1 is not None and r1.fired else ""
    else:
        read = doc; o["pt0_loc"] = None; o["pt0_loc_ev"] = ""
    o["n_units"] = len(P.NR.units(read)) if read else 0
    if read:
        rels = ns.related(q, frame, parties)
        us = P.NR.units(read)
        cached([(u, qq) for qq in [q] + [rq for _, rq in rels] for u in us])  # one batch for the question and its relatives
        n = low.answer(q, read)
        ev = (n.evidence or [{}])[0]
        o.update(net=n.answer if n.fired else None, net_p=n.confidence if n.fired else ev.get("p", 0.0),
                 net_label=ev.get("label"), net_ev=ev.get("text", ""), net_reason=n.reason[:160], net_checks=n.checks)
        o["q_pyes"], o["q_pno"] = ns.net_probs(pipe.net, q, read)
        o["rel"] = []
        for kind, rq in rels:
            py, pn = ns.net_probs(pipe.net, rq, read)
            o["rel"].append({"kind": kind, "q": rq, "pyes": round(float(py), 4), "pno": round(float(pn), 4)})
    key = hash(doc)
    if key not in norms_cache:
        norms_cache.clear(); norms_cache[key] = ns.DocNorms(doc, parties)
    dn = norms_cache[key]
    cand = o.get("pt0") or o.get("pt0_loc") or o.get("net")
    ev = o.get("pt0_ev") or o.get("pt0_loc_ev") or o.get("net_ev") or ""
    o["prio"] = ns.priority(dn, frame, cand, ev)
    o["prio_no"] = ns.priority(dn, frame, "no", "") if frame else {}
    o["typed_ok"] = P.typed_check(frame, o.get("net_ev") or "") is None if o.get("net") else None
    return o


def readers(pipe):
    """(cached network, low-threshold reader): each (text, question) pair is scored once per process (NetReader.answer,
    net_probs and the related questions share it); the reader answers yes and no from 0.5 so thresholds replay offline."""
    raw_net = pipe.net.net
    memo = {}

    def cached(pairs):
        todo = [pq for pq in dict.fromkeys(pairs) if pq not in memo]
        if todo:
            for pq, pr in zip(todo, raw_net(todo)): memo[pq] = pr
        out = [memo[pq] for pq in pairs]
        if len(memo) > 200_000: memo.clear()
        return out
    pipe.net.net = cached
    low = P.NR.NetReader(net=cached, t_yes=0.5, t_yes_doc=0.5, t_yes_long=0.5, t_no=0.5, k_lex=12, k_emb=12)
    low._embed = pipe.net._embed
    return cached, low


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--tag", default="")
    ap.add_argument("--model"); ap.add_argument("--n", type=int); a = ap.parse_args()
    rows = SETS[a.set]()
    if a.n: rows = rows[:a.n]
    P.NR.TORCH_THREADS = int(os.environ.get("TT", "3"))  # three extractions side by side on the 8 cores
    pipe = P.Pipeline(t_located=None)
    if a.model:
        pipe.net = P.NR.NetReader(model_dir=a.model)
    cached, low = readers(pipe)
    os.makedirs(f"{HERE}/feats", exist_ok=True)
    path = f"{HERE}/feats/{a.set}{('_' + a.tag) if a.tag else ''}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    norms_cache = {}
    t0 = time.time()
    with open(path, "a") as f:
        for i, r in enumerate(rows):
            if r["id"] in done: continue
            o = featurize(pipe, low, cached, norms_cache, r["question"], r["doc"])
            o.update({"id": r["id"], "set": a.set, "gold": r["gold"], "qid": r.get("qid"), "shape": r.get("shape")})
            f.write(json.dumps(o, default=float) + "\n"); f.flush()
            if (i + 1) % 100 == 0: print(f"  {a.set}: {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done {a.set} -> {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
