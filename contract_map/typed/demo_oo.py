"""The object model + query language end to end on small tuning samples (2026-10-03): which object answers, and how
often it is right. Sets: wc1_open (whole contracts, user questions), 8 tuning NDAs x 17, 10 tuning CUAD contracts x the
LegalBench questions of the four CUAD clause classes. usage: demo_oo.py"""
import collections, sys, time
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
import extract as X, oo, pipeline as P

pipe = P.Pipeline(t_located=0.80)
def run(name, rows, kind=None):
    st = collections.Counter(); docs = {}; t0 = time.time(); shown = 0
    for r in rows:
        c = docs.get(r["doc"]) or docs.setdefault(r["doc"], oo.contract(r["doc"], kind, pipe))
        a = c.ask(r["question"])
        if a.value:
            st[(a.via.split("[")[0].split("(")[0], a.value == r["gold"])] += 1
            if shown < 3: print("   ", a.query, "->", a.value, f"via {a.via}", "| gold", r["gold"]); shown += 1
    n = sum(st.values()); ok = sum(v for (k, good), v in st.items() if good)
    print(f"{name}: {len(rows)} questions, answered {n}, right {ok} ({time.time() - t0:.0f}s)  by object: {dict(st)}", flush=True)

run("wc1_open", X.SETS["wc1_open"]())
nda = X.SETS["nda_tune"](); first = []
for r in nda:
    if r["doc"] not in first: first.append(r["doc"])
run("nda_tune (8 NDAs)", [r for r in nda if r["doc"] in first[:8]], kind="nda")
cuad = [r for r in X.SETS["cuad_tune"]() if r["qid"] in ("governing law", "anti assignment", "cap on liability", "renewal term")]
docs = []
for r in cuad:
    if r["doc"] not in docs: docs.append(r["doc"])
run("cuad_tune (10 contracts, 4 clause types)", [r for r in cuad if r["doc"] in docs[:10]])
