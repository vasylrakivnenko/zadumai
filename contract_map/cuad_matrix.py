import json, re, collections
d = json.load(open("cuad/CUADv1.json"))["data"]
TYPES = [("co-branding", r"co.?branding"), ("joint venture", r"joint.?venture"), ("strategic alliance", r"strategic.?alliance|alliance"),
         ("non-compete", r"non.?compet|non.?solicit"), ("franchise", r"franchis"), ("endorsement", r"endorse"), ("sponsorship", r"sponsor"),
         ("hosting", r"hosting"), ("outsourcing", r"outsourc"), ("reseller", r"resell|value.?added"), ("distributor", r"distribut"),
         ("manufacturing", r"manufactur"), ("supply", r"supply"), ("maintenance", r"maintenance"), ("transportation", r"transport"),
         ("promotion", r"promot"), ("marketing", r"marketing"), ("agency", r"agency"), ("affiliate", r"affiliat"),
         ("collaboration", r"collaborat"), ("development", r"development"), ("consulting", r"consult"),
         ("ip", r"intellectual property|\bip\b|trademark|patent"), ("license", r"licen[sc]"), ("service", r"service")]
rows = []
for doc in d:
    t = doc["title"]
    tail = t.split("EX-")[-1] if "EX-" in t else t
    typ = next((name for name, pat in TYPES if re.search(pat, tail, re.I)), "other")
    present = {}
    for q in doc["paragraphs"][0]["qas"]:
        cat = q["id"].split("__")[-1]
        present[cat] = bool(q["answers"])
    rows.append((typ, present, len(doc["paragraphs"][0]["context"])))
cats = list(rows[0][1])
bytype = collections.defaultdict(list)
for typ, p, n in rows: bytype[typ].append(p)
print("contracts", len(rows), "types", {k: len(v) for k, v in sorted(bytype.items(), key=lambda kv: -len(kv[1]))})
print("median chars", sorted(n for _, _, n in rows)[len(rows)//2])
N = len(rows)
res = []
for c in cats:
    overall = sum(p[c] for _, p, _ in rows) / N
    per = {t: sum(p[c] for p in ps) / len(ps) for t, ps in bytype.items() if len(ps) >= 10}
    # spread: how many sizable types have it in >= 50% / >= 25% of contracts
    hi = sorted(per.items(), key=lambda kv: -kv[1])
    res.append((overall, c, sum(v >= 0.5 for v in per.values()), sum(v >= 0.2 for v in per.values()), len(per), hi[:3], hi[-1]))
for overall, c, n50, n20, k, top, low in sorted(res, reverse=True):
    print(f"{c:38} all {overall:5.0%}  types>=50%: {n50:2}/{k}  >=20%: {n20:2}/{k}  top {', '.join(f'{t} {v:.0%}' for t, v in top)}  low {low[0]} {low[1]:.0%}")
