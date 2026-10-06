import sys, collections, re
import xml.etree.ElementTree as ET
exec(open("parse.py").read().split("f = sys.argv[1]")[0])
f = sys.argv[1]
label, parents, alts, defs = load(f)
kids = collections.defaultdict(list)
for k, ps in parents.items():
    for p in ps: kids[p].append(k)
def sub(n, seen=None):
    seen = seen if seen is not None else set()
    for c in kids[n]:
        if c not in seen: seen.add(c); sub(c, seen)
    return seen
byname = {v: k for k, v in label.items()}
root = ET.parse(f).getroot()
langs = collections.Counter(); props = collections.Counter()
cls = {c.get(R+"about"): c for c in root.findall("owl:Class", NS)}
for name in ["Agreements", "Contractual Clause"]:
    if name not in byname: print(name, "missing"); continue
    s = sub(byname[name])
    print(f"{f} {name}: {len(s)} classes; with altLabel {sum(1 for x in s if alts[x])}, with definition {sum(1 for x in s if x in defs)}")
    for x in s:
        for ch in cls.get(x, []):
            tag = ch.tag.split('}')[1]; props[(name, tag)] += 1
            if tag == "prefLabel" or tag == "label":
                langs[ch.get("{http://www.w3.org/XML/1998/namespace}lang")] += 1
print(sorted(props.items(), key=lambda kv: -kv[1])[:30])
print("label langs:", langs.most_common(15))
print("object properties:", len(root.findall("owl:ObjectProperty", NS)), [ (p.find("rdfs:label", NS).text if p.find("rdfs:label", NS) is not None else p.get(R+"about")) for p in root.findall("owl:ObjectProperty", NS)][:40])
for n in ["Limitation of Liability Clause", "Most Favored Nation Clause", "Supply Agreement", "Termination Without Cause Clause"]:
    x = byname.get(n)
    if x: print("\n", n, "| alt:", alts[x][:8], "| def:", (defs.get(x) or "")[:300])
