"""Hybrid search for the pipeline (2026-10-06; the user: "a good hybrid search (keyword and RAG ... with a Cohere
reranker model)"): contract_tool's BM25 + bge-small candidates (reciprocal-rank fusion), the top 30 reranked by Cohere
rerank v4.0 pro on the user's Azure (POST /providers/cohere/v2/rerank). Every rerank call is logged (searches, docs).
usage: s = Search(ContractTool(docs_dir)); hits = s.search(query, docs=[...], k=6)  -> [unit index]"""
import json, os, threading, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("AZURE_API_KEY="))
URL = "https://ai-vasyl-0670.services.ai.azure.com/providers/cohere/v2/rerank"; MODEL = "cohere-rerank-v4.0-pro"
LOG = f"{HERE}/../runs/cohere_usage.jsonl"; _LOCK = threading.Lock()


def rerank(query, texts, top_n):
    """Cohere's order of `texts` for `query` (indices, best first); None on failure."""
    body = json.dumps({"model": MODEL, "query": query[:2000], "documents": [t[:4000] for t in texts], "top_n": min(top_n, len(texts))}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, data=body, headers={"api-key": KEY, "Content-Type": "application/json", "User-Agent": "zadum-pipeline"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            with _LOCK, open(LOG, "a") as f: f.write(json.dumps({"searches": 1, "docs": len(texts), "t": time.time()}) + "\n")
            return [x["index"] for x in r["results"]]
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


class Search:
    def __init__(self, tool, use_rerank=True, pool=30):
        self.tool, self.use_rerank, self.pool = tool, use_rerank, pool

    def text(self, i):
        u = self.tool.units[i]; return f"[{u.where()}] {u.marked if u.changes else u.text}"

    def search(self, query, docs=None, k=6):
        """unit indices best first: BM25 + dense candidates, Cohere-reranked when on (falls back to the fused order)."""
        idx = [i for d in (docs or self.tool.docs) for i in self.tool.docs[d] if self.tool.units[i].text or self.tool.units[i].before]
        if not idx: return []
        cand = [int(i) for i in self.tool.rank(query, idx)[:self.pool]]
        if self.use_rerank and len(cand) > 1:
            order = rerank(query, [self.text(i) for i in cand], k)
            if order is not None: return [cand[j] for j in order][:k]
        return cand[:k]
