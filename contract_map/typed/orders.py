"""Orderings of the local tiers, replayed from runs/router2_SET.jsonl (eval_router2.py). For long documents:
  A today:                   pt0 -> nli -> net
  B + located after:         pt0 -> nli -> net -> located
  C located before NLI:      pt0 -> net -> located -> nli
  D NLI on the located text: pt0 -> net -> located -> nli(located)
  E no NLI on long texts:    pt0 -> net -> located
Short documents: pt0 -> nli -> net in every order. usage: orders.py SET [SET ...]"""
import json, math, sys
def lo(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
ORD = {"A": ["pt0", "nli", "net"], "B": ["pt0", "nli", "net", "loc"], "C": ["pt0", "net", "loc", "nli"],
       "D": ["pt0", "net", "loc", "nli_loc"], "E": ["pt0", "net", "loc"]}
for s in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"runs/router2_{s}.jsonl")]
    print(f"== {s}: {len(rows)} questions")
    for k, order in ORD.items():
        n = ok = 0; by = {}
        for o in rows:
            seq = order if o["long"] else ["pt0", "nli", "net"]
            for t in seq:
                a = o.get(t)
                if a:
                    n += 1; ok += a == o["gold"]; by.setdefault(t, [0, 0]); by[t][0] += 1; by[t][1] += a == o["gold"]; break
        print(f"   {k} {' -> '.join(order):34s} answered {n:4d} = {n / len(rows):5.1%} @ {ok / max(1, n):6.1%} [lo {lo(ok, n):.3f}]  "
              + " ".join(f"{t}:{v[1]}/{v[0]}" for t, v in by.items()))
