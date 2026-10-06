"""How well the document router guesses the kind from a single clause (2026-10-03): 500 test clauses per dataset."""
import collections, random, sys
sys.path.insert(0, "/root/projects/zadumai/legalbench_map"); sys.path.insert(0, "bench")
import bench
from router.contract_map import ContractMap
m = ContractMap(lease_encoder=False)
want = {"ledgar": "commercial", "cuad_clause": "commercial", "contractnli": "nda", "opp115": "privacy policy", "tos": "terms of service", "lease": "lease"}
for ds, k in want.items():
    rows = bench.SETS[ds]()["test"]; random.Random(0).shuffle(rows); rows = rows[:500]
    P = m.rlr.predict_proba(m.rvec.transform([r["text"] for r in rows])); top = m.rlr.classes_[P.argmax(1)]; pm = P.max(1)
    ok = top == k
    c = collections.Counter(t for t, o in zip(top, ok) if not o)
    print(f"{ds:12} top pick right {ok.mean():5.1%} | at p >= 0.6: {ok[pm >= 0.6].mean() if (pm >= 0.6).any() else 0:5.1%} on {(pm >= 0.6).mean():4.0%} | wrong picks: {dict(c.most_common(3))}")
