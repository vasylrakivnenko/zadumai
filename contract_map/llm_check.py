"""How the LLM (Jev's yes/no, as harness `_fallback` asks it) does on questions the contract map sends to it
(2026-10-02; 1,000 calls, the user's cap): 500 the local tiers answer today and 500 they don't (open sets only).
Scoring: yes right on gold yes; no right on gold no; on gold not_stated the LLM's answer is counted separately."""
import collections, concurrent.futures as cf, json, random, sys
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from router.systemone import SystemOne
from router.harness import _yes_no

rows = [json.loads(l) for l in open("/root/zadumai_nli_proto/contract_map/routed_rows.jsonl")]
ans = [r for r in rows if r["local"]]; rest = [r for r in rows if not r["local"]]
random.Random(0).shuffle(ans); random.Random(0).shuffle(rest)
sample = [("local answers", r) for r in ans[:500]] + [("local defers", r) for r in rest[:500]]
jev = SystemOne.jev()
def ask(item):
    g, r = item
    try:
        a, conf, _ = _yes_no(jev.noul(r["premise"], r["question"])); return g, r, a, None
    except Exception as e:
        return g, r, None, str(e)[:100]
res = []
with cf.ThreadPoolExecutor(8) as ex:
    for out in ex.map(ask, sample): res.append(out)
json.dump([{"group": g, "set": r["set"], "id": r["id"], "gold": r["gold"], "llm": a, "local": r["local"], "local_right": r["local_right"]}
           for g, r, a, _ in res], open("/root/zadumai_nli_proto/contract_map/llm_check.json", "w"))
print(f"calls {jev.calls}, errors {sum(1 for x in res if x[3])}, tokens in {jev.input_tokens}")
for grp in ("local answers", "local defers"):
    c = collections.Counter()
    for g, r, a, err in res:
        if g != grp or err: continue
        c["n"] += 1
        if r["gold"] in ("yes", "no"):
            c["yn"] += 1; c["llm_ok"] += a == r["gold"]
            if r["local"]: c["loc"] += 1; c["loc_ok"] += r["local_right"]
        else:
            c["ns"] += 1; c["ns_yes"] += a == "yes"
    line = f"{grp:14} n {c['n']}: gold yes/no {c['yn']}: LLM right {c['llm_ok']} = {c['llm_ok'] / max(c['yn'], 1):.1%}"
    if c["loc"]: line += f" | local right {c['loc_ok']}/{c['loc']} = {c['loc_ok'] / c['loc']:.1%}"
    line += f" | gold not stated {c['ns']}: LLM said yes on {c['ns_yes']}"
    print(line)
