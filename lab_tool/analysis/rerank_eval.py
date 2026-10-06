"""Does the Cohere reranker help? (2026-10-06) The retrieval benchmark of eval_tool_v0 part 4: 273 fact-bearing
criteria of 40 training tasks, the criterion's title as the query, the passage holding its facts as the target.
Top-6 reach: contract_tool's BM25 + bge (fused) vs the same top 30 reranked by Cohere rerank v4.0 pro.
usage (from lab_tool/): dspy_venv/bin/python analysis/rerank_eval.py"""
import concurrent.futures as cf, json, sys
sys.path.insert(0, "."); sys.path.insert(0, "pipeline")
import eval_tool_v0 as E
from contract_tool.tool import ContractTool
from search import Search

def one(r):
    import torch; torch.set_num_threads(1)
    tool = ContractTool(E.docs_dir(r["task"])); tool.emb(); s = Search(tool)
    res = []
    for c in r["crit"]:
        ev = set(c["evidence"])
        base = {j for _, sh in tool.find_units(c["title"]) for j in sh}
        rr = s.search(c["title"], k=6); rr_sh = set(rr) | {i + 1 for i in rr if tool.units[i].heading or len(tool.units[i].text.split()) < 6}
        res.append((bool(base & ev), bool(rr_sh & ev)))
    return res

R = [json.loads(l) for l in open("data/tool_v0_part4.jsonl")]
with cf.ProcessPoolExecutor(8) as ex: out = [x for v in ex.map(one, R) for x in v]
n = len(out); print(f"{n} criteria: BM25+bge top-6 reach {sum(a for a, _ in out) / n:.1%} | + Cohere rerank top-6 {sum(b for _, b in out) / n:.1%}")
