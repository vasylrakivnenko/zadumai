"""(run from lab_tool/) Looser evidence (2026-10-05): any passage that contains one of the criterion's specific facts (not only the passage
with the most of them). Plans as-is, plans with compound queries split, and the criterion title as the query."""
import json, sys, collections, multiprocessing as mp
sys.path.insert(0, "/root/zadumai_nli_proto/lab_tool"); sys.path.insert(0, "/root/zadumai_nli_proto/lab_tool/analysis")
import eval_tool_v0 as E
from find_variants import split, PLANS, LABELS

def one(r):
    import torch; torch.set_num_threads(1)
    from contract_tool.tool import ContractTool
    tool = ContractTool(E.docs_dir(r["task"])); tool.emb()
    U = [E.norm(u.text + " " + u.before) for u in tool.units]
    loose = [{i for a in c["anchors"] for i, u in enumerate(U) if a in u} for c in r["crit"]]
    res = {"title": sum(bool({j for _, s in tool.find_units(c["title"]) for j in s} & L) for c, L in zip(r["crit"], loose))}
    for l in LABELS:
        plan = PLANS[l].get(r["task"]); S, T = set(), set()
        for c in (plan or {}).get("calls", []):
            try: a = json.loads(c["arguments"])
            except ValueError: continue
            if a.get("action") not in ("find", "ask") or not a.get("query"): continue
            try: dn = tool.doc(a.get("document")) if a.get("document") else None
            except KeyError: dn = None
            S |= {j for _, s in tool.find_units(a["query"], dn) for j in s}
            subs = split(a["query"]); kk = max(2, -(-6 // len(subs)))
            for sq in subs: T |= {j for _, s in tool.find_units(sq, dn, k=kk) for j in s}
        res[l] = sum(bool(S & L) for L in loose); res[l + "_split"] = sum(bool(T & L) for L in loose)
    return r["fact_bearing"], res

if __name__ == "__main__":
    R = [json.loads(l) for l in open("data/tool_v0_part4.jsonl")]
    with mp.get_context("fork").Pool(8) as p: out = p.map(one, R)
    fb = sum(x[0] for x in out); tot = collections.Counter()
    for _, res in out: tot.update(res)
    print({k: f"{v / fb:.1%}" for k, v in tot.items()}, "of", fb)
