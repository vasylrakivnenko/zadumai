"""Why the pipeline doesn't answer: for gold yes/no rows it leaves, was a gold evidence span located (NDA sets: the
span's first 50 normalized chars inside a located piece)? and the network's reason. usage: analyze.py SET RUN"""
import collections, json, re, sys
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
import eval_docs as E
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
rows = {r["id"]: r for r in {"nda_train": lambda: E.nda("train", 100), "nda_dev": lambda: E.nda("dev"), "wc1_open": lambda: E.wc1("open")}[sys.argv[1]]()}
out = json.load(open(f"/root/zadumai_nli_proto/contract_map/typed/runs/{sys.argv[1]}_{sys.argv[2]}.json"))
c = collections.Counter(); why = collections.Counter(); byq = collections.defaultdict(collections.Counter)
for o in out:
    r = rows[o["id"]]
    if o["gold"] == "not_stated" or o["answer"] or o.get("path") == "pt0": continue
    pieces = norm(" ".join(o.get("pieces") or []))
    ev = r.get("ev") or ([r["quote"]] if r.get("quote") else [])
    found = any(norm(e)[:50] in pieces for e in ev if e)
    c[("located" if found else "not located", o["gold"])] += 1
    byq[o.get("qid") or o.get("shape")][("located" if found else "missed")] += 1
    if found: why[re.sub(r"\(.*?\)|[0-9.]+", "", o.get("reason", ""))[:90]] += 1
print(dict(c))
print("by question:", {k: dict(v) for k, v in sorted(byq.items())})
print("network's reasons when the evidence was located:")
for k, n in why.most_common(10): print(f"  {n:4d} {k}")
