"""Idea 5 features (2026-10-03) for the questions the grammar left untyped (feats/SET.jsonl rows with frame None):
  fp     first person: featurize() of the question asked as each party (q5.as_party), at most 3 parties
  split  two actions: featurize() of each part (q5.split_two) and the connective
  nframe the learned parser's (asks, action) when it is sure (p >= 0.8 for both), with the priority and typed checks
         recomputed from it
The learned parser is trained on the tuning questions the grammar typed (short_tune = v5 + v6, wc1_open), its
agreement with the grammar measured by 5-fold CV first. Writes feats/SET_q5.jsonl. usage: extract5.py SET [SET ...]"""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract as X
import ns
import pipeline as P
import q5
from router import frames


def train_parser():
    rows = X.SETS["short_tune"]() + X.SETS["wc1_open"]()
    qs, fs = [], []
    for r in rows:
        p0 = P.pt0_check(r["question"], r["doc"]); fr = (p0.frames or {}).get("frame")
        if fr and fr.get("asks"): qs.append(r["question"]); fs.append(fr)
    idx = list(range(len(qs))); random.Random(3).shuffle(idx)
    agree_a = agree_c = n = sure = sure_ok = 0
    for k in range(5):
        te = [i for j, i in enumerate(idx) if j % 5 == k]; tr = [i for j, i in enumerate(idx) if j % 5 != k]
        m = q5.NeuralParser().fit([qs[i] for i in tr], [fs[i] for i in tr])
        for i, p in zip(te, m.predict([qs[i] for i in te])):
            n += 1; a_ok = p["asks"] == fs[i]["asks"]; c_ok = p["action"] == q5.NeuralParser._act(fs[i])
            agree_a += a_ok; agree_c += c_ok
            if p["p_asks"] >= 0.8 and p["p_action"] >= 0.8: sure += 1; sure_ok += a_ok and c_ok
    print(f"learned parser vs the grammar (5-fold CV, {n} typed questions): asks {agree_a / n:.1%}, action {agree_c / n:.1%}; "
          f"sure (both p >= 0.8) on {sure / n:.1%}, both right there {sure_ok / max(1, sure):.1%}", flush=True)
    return q5.NeuralParser().fit(qs, fs)


def main():
    parser = train_parser()
    P.NR.TORCH_THREADS = int(os.environ.get("TT", "3"))
    pipe = P.Pipeline(t_located=None); cached, low = X.readers(pipe)
    for name in sys.argv[1:]:
        src = {r["id"]: r for r in X.SETS[name]()}
        rows = [json.loads(l) for l in open(f"{HERE}/feats/{name}.jsonl")]
        out_path = f"{HERE}/feats/{name}_q5.jsonl"
        done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
        todo = [o for o in rows if not o.get("frame") and o["id"] not in done]
        preds = parser.predict([o["question"] for o in todo]) if todo else []
        norms_cache = {}; t0 = time.time()
        with open(out_path, "a") as f:
            for o, pr in zip(todo, preds):
                doc = src[o["id"]]["doc"]; q = o["question"]; e = {"id": o["id"], "nframe": pr}
                if q5.first_person(q):
                    named = frames.named_actors(doc)[:3]
                    e["fp"] = [{"party": p, "q": q5.as_party(q, p), "f": X.featurize(pipe, low, cached, norms_cache, q5.as_party(q, p), doc)}
                               for p in q5.candidate_parties(o.get("parties") or [], named)]
                sp = q5.split_two(q)
                if sp:
                    e["split"] = {"conn": sp[0], "parts": [X.featurize(pipe, low, cached, norms_cache, x, doc) for x in sp[1]]}
                if pr["p_asks"] >= 0.8 and pr["p_action"] >= 0.8 and pr["action"] != "OPEN":
                    fr = {"asks": pr["asks"], "actor": "ANY", "action": pr["action"], "actions": [pr["action"]], "alts": []}
                    key = hash(doc)
                    if key not in norms_cache: norms_cache.clear(); norms_cache[key] = ns.DocNorms(doc, o.get("parties") or [])
                    dn = norms_cache[key]
                    cand = o.get("pt0") or o.get("pt0_loc") or o.get("net")
                    ev = o.get("pt0_ev") or o.get("pt0_loc_ev") or o.get("net_ev") or ""
                    e["prio_n"] = ns.priority(dn, fr, cand, ev); e["prio_no_n"] = ns.priority(dn, fr, "no", "")
                    e["typed_ok_n"] = P.typed_check(fr, o.get("net_ev") or "") is None if o.get("net") else None
                f.write(json.dumps(e, default=float) + "\n"); f.flush()
        print(f"done {name}: {len(todo)} untyped questions ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
