"""Recompute the priority features (ns.priority, no network) for saved feature rows after a change to ns.py: the full
frame comes from Pre-Tier 0 again (milliseconds). usage: reprio.py SET [SET ...]  (rewrites feats/SET.jsonl in place)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract as X, ns, pipeline as P
for name in sys.argv[1:]:
    src = {r["id"]: r for r in X.SETS[name]()}
    path = f"{HERE}/feats/{name}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    cache = {}
    for o in rows:
        doc = src[o["id"]]["doc"]
        fr = (P.pt0_check(o["question"], doc).frames or {}).get("frame")
        k = hash(doc)
        if k not in cache: cache.clear(); cache[k] = ns.DocNorms(doc, o.get("parties") or [])
        cand = o.get("pt0") or o.get("pt0_loc") or o.get("net")
        ev = o.get("pt0_ev") or o.get("pt0_loc_ev") or o.get("net_ev") or ""
        o["prio"] = ns.priority(cache[k], fr, cand, ev); o["prio_no"] = ns.priority(cache[k], fr, "no", "") if fr else {}
    with open(path + ".tmp", "w") as f:
        for o in rows: f.write(json.dumps(o, default=float) + "\n")
    os.replace(path + ".tmp", path)
    print(f"{name}: {len(rows)} rows re-prioritized", flush=True)
