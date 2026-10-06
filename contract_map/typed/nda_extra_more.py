"""NDA compiler features for the 284 extra training NDAs (and a check against nda_extra.jsonl's own values): the IR
answer from cnli/runs/ir_{train,dev}.json, the topic-word hits (cnli/mentioned.py TOPIC), the out-of-fold span scores
and the rule sources, as nda_extra.jsonl has them. Appends to feats/nda_extra.jsonl."""
import collections, json, re, sys
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/cnli"); sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
from mentioned import TOPIC
import extract as X
C = "/root/zadumai_nli_proto/contract_map/cnli/runs"
ir = {}
for split in ("train", "dev"):
    for r in json.load(open(f"{C}/ir_{split}.json")): ir[f"{r['doc']}/{r['q']}"] = r["answer"]
span = json.load(open(f"{C}/oof_span.json"))
src = collections.defaultdict(dict)
for r in json.load(open(f"{C}/cv_engine2.json")):
    if r["answer"] and r["why"] != "Pre-Tier 0": src["v2"][f"{r['doc']}/{r['q']}"] = r["answer"]
for name in ("compiler", "variants"):
    for split in ("train", "dev"):
        for r in json.load(open(f"{C}/{name}_{split}.json")):
            if r["answer"]: src[name if name != "variants" else r.get("source", "var")][f"{r['doc']}/{r['q']}"] = r["answer"]
have = {json.loads(l)["id"]: json.loads(l) for l in open("feats/nda_extra.jsonl")}
agree = sum(have[k].get("ir") == ir.get(k) for k in have if k in ir); print(f"IR from runs vs recomputed: {agree}/{sum(k in ir for k in have)} agree")
n = 0
with open("feats/nda_extra.jsonl", "a") as f:
    for r in X.SETS["nda_more"]():
        k = r["id"]
        if k in have: continue
        d, q = k.split("/"); s = span.get(q, {}).get(d)
        e = {"id": k, "ir": ir.get(k), "topic_hits": len(re.findall(TOPIC[q], r["doc"], re.I)) if q in TOPIC else -1,
             "span_s": s[0] if s else -1.0, "span_pn": s[1] if s else -1.0}
        for name in ("v2", "compiler", "var_c3", "var_c4", "var_oral"): e[name] = src.get(name, {}).get(k)
        f.write(json.dumps(e) + "\n"); n += 1
print("added", n)
