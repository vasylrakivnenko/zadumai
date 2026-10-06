"""(run from lab_tool/) Why do the callers' query plans reach so little? Variants of executing the same plans (2026-10-05)."""
import json, re, sys, collections, multiprocessing as mp
sys.path.insert(0, "/root/zadumai_nli_proto/lab_tool")
import eval_tool_v0 as E
LABELS = ["gemma-4-e4b_guided_test_gepa", "muse-bf16_guided_test_gepa"]
PLANS = {l: {json.loads(x)["task"]: json.loads(x) for x in open(f"data/out_B_{l}.jsonl")} for l in LABELS}

def split(q):
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+|or\s+)?|;\s*|\s+and\s+(?=[a-z]+\s)", q) if len(p.split()) >= 2]
    return parts if len(parts) > 1 else [q]

def one(r):
    import torch; torch.set_num_threads(1)
    from contract_tool.tool import ContractTool
    tool = ContractTool(E.docs_dir(r["task"])); tool.emb(); res = {}
    for l in LABELS:
        plan = PLANS[l].get(r["task"])
        if not plan or "error" in plan: continue
        calls = []
        for c in plan["calls"]:
            try: a = json.loads(c["arguments"])
            except ValueError: continue
            if a.get("action") in ("find", "ask") and a.get("query"):
                try: dn = tool.doc(a.get("document")) if a.get("document") else None
                except KeyError: dn = None
                calls.append((a["query"], dn))
        V = collections.defaultdict(set); n = collections.Counter()
        for q, dn in calls:
            V["as_is"] |= {j for _, s in tool.find_units(q, dn) for j in s}; n["as_is"] += 6
            V["k12"] |= {j for _, s in tool.find_units(q, dn, k=12) for j in s}; n["k12"] += 12
            V["all_docs"] |= {j for _, s in tool.find_units(q, None) for j in s}; n["all_docs"] += 6
            subs = split(q); kk = max(2, -(-6 // len(subs)))
            for sq in subs: V["split"] |= {j for _, s in tool.find_units(sq, dn, k=kk) for j in s}; n["split"] += kk * len(subs) if len(subs) > 1 else 6
            for sq in subs: V["split_alldocs"] |= {j for _, s in tool.find_units(sq, None, k=kk) for j in s}
            n["subqueries"] += len(subs)
        res[l] = {v: sum(bool(S & set(c["evidence"])) for c in r["crit"]) for v, S in V.items()}
        res[l]["passages"] = dict(n); res[l]["queries"] = len(calls)
    return r["task"], r["fact_bearing"], res

if __name__ == "__main__":
    R = [json.loads(l) for l in open("data/tool_v0_part4.jsonl")]
    with mp.get_context("fork").Pool(8) as p: out = p.map(one, R)
    fb = sum(x[1] for x in out)
    for l in LABELS:
        tot = collections.Counter(); pas = collections.Counter(); nq = 0
        for _, _, res in out:
            if l in res: tot.update({k: v for k, v in res[l].items() if isinstance(v, int) and k != "queries"}); pas.update(res[l]["passages"]); nq += res[l]["queries"]
        print(l, "find queries", nq, "| subqueries", pas["subqueries"], "|", {k: f"{v / fb:.1%}" for k, v in tot.items()}, "| passages", dict(pas))
