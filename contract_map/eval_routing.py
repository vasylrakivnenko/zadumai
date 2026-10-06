"""What the contract map's "to the LLM" rule would change on the open question sets (2026-10-02), without LLM calls:
how many questions it sends to the LLM, and how many of those today's local tiers (p3: Pre-Tier 0 + network,
v4/runs/sys_p3_routed.json) answer, and how many right. Held-out: totals only. Also writes the routed rows (not
held-out) for an LLM check: routed_rows.jsonl.
usage: eval_routing.py"""
import collections, json, sys
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
sys.path.insert(0, "/root/zadumai_nli_proto/extensive/v4")
from router.contract_map import ContractMap
from pt0eval import load, right

SETS = ["dev", "freshA", "freshB", "genval", "v4all", "v5all", "v6ball", "v7all", "adv", "heldout"]
m = ContractMap(lease_encoder=False)
local = json.load(open("/root/zadumai_nli_proto/extensive/v4/runs/sys_p3_routed.json"))
tot = collections.Counter(); types = collections.Counter(); out = open("/root/zadumai_nli_proto/contract_map/routed_rows.jsonl", "w")
print(f"{'set':8} {'rows':>6} {'to LLM':>14} | local answers on those: {'answered':>8} {'right':>6} | local answers elsewhere")
for s in SETS:
    c = collections.Counter()
    for r in load(s):
        d = m.decide(r["question"], r["premise"])
        res = local.get(f"{s}:{r['id']}")
        ok = bool(res) and right(res[1], r["gold"], res[2])
        c["n"] += 1
        if d.to_llm:
            c["llm"] += 1; c["llm_ans"] += bool(res); c["llm_ok"] += ok
            if s != "heldout":
                types.update(d.question_types[:1] or ["(lease)"])
                out.write(json.dumps({"set": s, "id": r["id"], "question": r["question"], "premise": r["premise"], "gold": r["gold"],
                                      "local": res, "local_right": ok, "types": d.question_types, "kind": d.document["kind"]}) + "\n")
        else:
            c["rest_ans"] += bool(res); c["rest_ok"] += ok
    tot.update(c)
    print(f"{s:8} {c['n']:6} {c['llm']:6} = {c['llm'] / c['n']:5.1%} | {c['llm_ans']:8} {c['llm_ok']:6} | {c['rest_ans']} ({c['rest_ok']} right)", flush=True)
c = tot
print(f"{'all':8} {c['n']:6} {c['llm']:6} = {c['llm'] / c['n']:5.1%} | {c['llm_ans']:8} {c['llm_ok']:6} | {c['rest_ans']} ({c['rest_ok']} right)")
print("most common clause types sent to the LLM (open sets):", ", ".join(f"{t} {n}" for t, n in types.most_common(12)))
