"""Latency of the pipeline on whole contracts (2026-10-03): the first question on a new document (compile, lemmas,
embeddings, read) vs the next ones (cached), and the parts. Contracts: the open wc1 ones (MCC, <= 30k chars), each
with its open questions; run when the box is quiet (the network uses 4 torch threads, as the router)."""
import collections, statistics, sys, time
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
import eval_docs as E
import pipeline as P

rows = E.wc1("open")
by = collections.defaultdict(list)
for r in rows: by[r["doc_key"]].append(r)
pipe = P.Pipeline()
pipe.answer("warm up", next(iter(by.values()))[0]["doc"])  # loads the models
first, rest, compile_ms, lemma_ms = [], [], [], []
for k, rs in by.items():
    doc = rs[0]["doc"]
    t = time.perf_counter(); cd = P.Compiled(doc); compile_ms.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); [P.lemmas(p["text"]) for p in cd.pieces]; lemma_ms.append((time.perf_counter() - t) * 1000)
    pipe._docs.clear(); pipe.net._vectors.clear()
    for i, r in enumerate(rs):
        t = time.perf_counter(); pipe.answer(r["question"], doc); ms = (time.perf_counter() - t) * 1000
        (first if i == 0 else rest).append(ms)
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print(f"{len(by)} contracts, {len(rows)} questions")
print(f"compile only: median {statistics.median(compile_ms):.0f} ms, max {max(compile_ms):.0f}")
print(f"lemmas of all pieces: median {statistics.median(lemma_ms):.0f} ms, max {max(lemma_ms):.0f}")
print(f"first question on a new contract: median {statistics.median(first):.0f} ms, p90 {q(first, .9):.0f}, max {max(first):.0f}")
print(f"later questions (cached): median {statistics.median(rest):.0f} ms, p90 {q(rest, .9):.0f}, max {max(rest):.0f}")
