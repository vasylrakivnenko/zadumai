"""Step 1 baseline: how often the live Pre-Tier 0 path types a user-style question (router/pretier0.check: first person,
the document's parties, then frames.question_frame). Rows: v5, v6, v6b, v7 gen rows (open). A row is typed when
Pre-Tier 0 got as far as a frame (its reason isn't "no frame: ..." / question-level). ROUTER_PATH picks the code."""
import collections, json, os, sys
sys.path.insert(0, os.environ.get("ROUTER_PATH", "/root/projects/zadumai/legalbench_map"))
from router.pretier0 import check
V = "/root/zadumai_nli_proto/extensive/v4"
rows = [r for n in ("v5", "v6", "v6b", "v7") for r in json.load(open(f"{V}/{n}_rows.json")) if r["part"].startswith("gen")]
why = collections.Counter(); ex = collections.defaultdict(list); typed = 0; fired = right = 0
for r in rows:
    p = check(r["question"], r["premise"])
    fr = p.frames or {}
    reason = (fr.get("reason") or p.reason or "")
    if p.fired:
        fired += 1; right += (p.answer == r["gold"]) if r["gold"] != "not_stated" else 0
    if reason.startswith("no frame") or (not fr and not p.fired):
        k = reason.replace("no frame: ", "")[:60] or "(no frames result)"; why[k] += 1
        if len(ex[k]) < 2: ex[k].append(r["question"])
    else:
        typed += 1
n = len(rows)
print(f"{n} questions: typed {typed} = {typed/n:.1%}; Pre-Tier 0 answers {fired} = {fired/n:.1%}, right {right}")
for k, c in why.most_common(10): print(f"  {c:5d} {k}  e.g. {ex[k]}")
