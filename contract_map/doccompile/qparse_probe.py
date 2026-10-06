"""How many user-style questions does Pre-Tier 0's own grammar already type? (router/frames.question_frame, live code)
Rows: v5, v6, v6b, v7 gen rows (all open). A question is typed when it gets a Frame; else the parser's reason."""
import collections, json, sys
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from router import frames
V = "/root/zadumai_nli_proto/extensive/v4"
rows = [r for n in ("v5", "v6", "v6b", "v7") for r in json.load(open(f"{V}/{n}_rows.json")) if r["part"].startswith("gen")]
trie = frames._BASE_TRIE
ok = collections.Counter(); why = collections.Counter(); asks = collections.Counter(); act = collections.Counter(); ex = collections.defaultdict(list)
for r in rows:
    f = frames.question_frame(r["question"], trie)
    if isinstance(f, str):
        k = f.split(":")[0][:70]; why[k] += 1
        if len(ex[k]) < 2: ex[k].append(r["question"])
    else:
        ok[r.get("shape")] += 1; asks[f.asks] += 1; act[bool(f.action)] += 1
n = len(rows); typed = sum(ok.values())
print(f"{n} user-style questions; typed by the grammar: {typed} = {typed/n:.1%}; asks {dict(asks)}; with an action concept {act[True]}")
shapes = collections.Counter(r.get("shape") for r in rows)
print("typed by question shape:", {s: f"{ok[s]}/{c}" for s, c in shapes.most_common()})
print("not typed, top reasons:")
for k, c in why.most_common(8): print(f"  {c:5d} {k}  e.g. {ex[k][0][:90]!r}")
