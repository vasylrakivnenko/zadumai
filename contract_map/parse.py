import sys, collections, re
import xml.etree.ElementTree as ET
NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "owl": "http://www.w3.org/2002/07/owl#", "skos": "http://www.w3.org/2004/02/skos/core#"}
R = "{%s}" % NS["rdf"]
def load(f):
    root = ET.parse(f).getroot()
    label, parents, alts, defs = {}, collections.defaultdict(list), collections.defaultdict(list), {}
    for c in root.findall("owl:Class", NS):
        iri = c.get(R + "about")
        if not iri: continue
        for l in c.findall("rdfs:label", NS):
            label.setdefault(iri, l.text)
        for p in c.findall("rdfs:subClassOf", NS):
            if p.get(R + "resource"): parents[iri].append(p.get(R + "resource"))
        for a in c.findall("skos:altLabel", NS): alts[iri].append(a.text)
        d = c.find("skos:definition", NS)
        if d is not None: defs[iri] = d.text
    return label, parents, alts, defs
f = sys.argv[1]
label, parents, alts, defs = load(f)
kids = collections.defaultdict(list)
for k, ps in parents.items():
    for p in ps: kids[p].append(k)
print(f, "classes", len(label))
# top-level: children of owl:Thing
tops = [k for k in label if any(p.endswith("#Thing") for p in parents[k])]
def size(n, seen=None):
    seen = seen if seen is not None else set()
    for c in kids[n]:
        if c not in seen: seen.add(c); size(c, seen)
    return len(seen)
for t in sorted(tops, key=lambda t: label[t]):
    print(f"  {label[t]:45} {size(t)}")
for pat in sys.argv[2:]:
    hits = [k for k in label if label[k] and re.search(pat, label[k], re.I)]
    print(f"\n/{pat}/ {len(hits)} labels, e.g.:", [label[h] for h in hits[:25]])
