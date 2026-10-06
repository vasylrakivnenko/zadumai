"""The typed check, offline, on a saved low-threshold run: the network's located answers at p >= t, with and without
typed_check on their deciding sentence (the frame from Pre-Tier 0 on the whole document, as the pipeline gets it).
usage: checkeval.py SET RUN"""
import collections, json, sys
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
import eval_docs as E, pipeline as P
from eval_docs import wilson_lo
rows = {r["id"]: r for r in {"nda_train": lambda: E.nda("train", 100), "nda_dev": lambda: E.nda("dev"), "wc1_open": lambda: E.wc1("open")}[sys.argv[1]]()}
out = json.load(open(f"/root/zadumai_nli_proto/contract_map/typed/runs/{sys.argv[1]}_{sys.argv[2]}.json"))
net = [o for o in out if o["path"] == "net-located" and o["answer"] == "yes"]
fixed = [o for o in out if o["path"] in ("pt0", "pt0-located")]; fok = sum(o["answer"] == o["gold"] for o in fixed)
blocked = collections.Counter()
for o in net:
    r = rows[o["id"]]; p0 = P.pt0_check(r["question"], r["doc"])
    o["why"] = P.typed_check((p0.frames or {}).get("frame"), o["evidence"])
    if o["why"]: blocked[(o["gold"] == "yes", o["why"][:40])] += 1
print("blocked (right?, why):", dict(blocked))
for t in (0.5, 0.8, 0.85, 0.88, 0.9, 0.92):
    for name, keep in (("no check", lambda o: True), ("typed check", lambda o: not o["why"])):
        a = [o for o in net if o.get("p", 0) >= t and keep(o)]; ok = sum(o["answer"] == o["gold"] for o in a)
        tot, tok = len(fixed) + len(a), fok + ok
        print(f"  t {t:.2f} {name:11s}: network {ok}/{len(a)} = {ok / max(1, len(a)):.1%} | all {tok}/{tot} = {tok / max(1, tot):.1%} "
              f"[lo {wilson_lo(tok, tot):.3f}] answered {tot / len(out):.1%}")
