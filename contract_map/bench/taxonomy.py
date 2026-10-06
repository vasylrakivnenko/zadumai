"""The shared clause taxonomy (2026-10-02): every class of the six clause datasets after combined.py's label map,
with its sources, document kinds, training positives, and the FOLIO clause class whose name matches (FOLIO IRIs
make it interoperable; FOLIO itself has no clause-to-contract links). Writes ../taxonomy.json."""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench
from combined import DATASETS, UNIFIED, score_class, row_classes

FAMILY = {"ledgar": "commercial", "cuad_clause": "commercial", "contractnli": "nda", "opp115": "privacy policy",
          "tos": "terms of service", "lease": "lease"}
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
      "owl": "http://www.w3.org/2002/07/owl#", "skos": "http://www.w3.org/2004/02/skos/core#"}
STOP = {"of", "the", "and", "to", "a", "on", "for", "in", "by", "clause", "clauses", "provision", "provisions", "agreement", "u", "no"}


def norm(s):
    s = re.sub(r"^[a-z_]+:", "", s.lower()).replace("_", " ")
    toks = [re.sub(r"(ies)$", "y", re.sub(r"(?<=[a-z]{3})s$", "", t)) for t in re.findall(r"[a-z]+", s)]
    return frozenset(t for t in toks if t not in STOP)


def folio_clauses():
    R = "{%s}" % NS["rdf"]; root = ET.parse(f"{D}/data/FOLIO.owl").getroot()
    label, parents, names = {}, collections.defaultdict(list), collections.defaultdict(set)
    for c in root.findall("owl:Class", NS):
        iri = c.get(R + "about")
        l = c.find("rdfs:label", NS)
        if iri is None or l is None: continue
        label[iri] = l.text
        parents[iri] = [p.get(R + "resource") for p in c.findall("rdfs:subClassOf", NS) if p.get(R + "resource")]
        names[iri] = {l.text} | {a.text for a in c.findall("skos:altLabel", NS) if a.text and a.text.isascii()}
    top = next(k for k, v in label.items() if v == "Contractual Clause")
    kids = collections.defaultdict(list)
    for k, ps in parents.items():
        for p in ps: kids[p].append(k)
    sub, todo = set(), [top]
    while todo:
        for k in kids[todo.pop()]:
            if k not in sub: sub.add(k); todo.append(k)
    return {k: (label[k], names[k]) for k in sub if "Definition" not in label[k]}


def main():
    S = {ds: bench.SETS[ds]() for ds in DATASETS}
    classes = collections.defaultdict(lambda: {"sources": [], "families": set(), "train_positives": collections.Counter()})
    for ds in DATASETS:
        for l in S[ds]["labels"]:
            u, kind = UNIFIED[ds].get(l, (None, "own"))
            c = score_class(ds, l)
            classes[c]["sources"].append({"dataset": ds, "label": l, "kind": "eq" if kind == "eq" else "own"})
            classes[c]["families"].add(FAMILY[ds])
            if kind == "sub":
                classes[u]["sources"].append({"dataset": ds, "label": l, "kind": "sub"}); classes[u]["families"].add(FAMILY[ds])
        for r in S[ds]["train"]:
            for c in row_classes(ds, r["labels"]): classes[c]["train_positives"][ds] += 1
    folio = folio_clauses()
    index = collections.defaultdict(list)
    for iri, (lab, names) in folio.items():
        for n in names:
            k = norm(n)
            if k: index[k].append((iri, lab))
    out = []
    for c, v in sorted(classes.items()):
        k = norm(c); m = index.get(k); how = "exact name"
        if not m:  # one side's words contained in the other's, at least two words: approximate, check by hand
            m = [x for kk, xs in index.items() for x in xs if len(kk) >= 2 and len(k) >= 2 and (kk <= k or k <= kk)][:1]; how = "approximate"

        out.append({"id": c, "name": re.sub(r"^[a-z_]+:|^U:", "", c), "shared": c.startswith("U:"),
                    "families": sorted(v["families"]), "sources": v["sources"],
                    "train_positives": dict(v["train_positives"]), "folio": ({"iri": m[0][0], "label": m[0][1], "match": how} if m else None)})
    json.dump(out, open(f"{D}/taxonomy.json", "w"), indent=1)
    fam = collections.Counter(f for x in out for f in x["families"])
    print(f"{len(out)} clause types; shared by 2+ datasets: {sum(x['shared'] for x in out)}; spanning 2+ document kinds: "
          f"{sum(len(x['families']) > 1 for x in out)}; with a FOLIO match: {sum(bool(x['folio']) for x in out)} (exact name: {sum(bool(x['folio']) and x['folio']['match'] == 'exact name' for x in out)})")
    print("per document kind:", dict(fam))
    rare = [x for x in out if sum(x["train_positives"].values()) < 100]
    print(f"rare (< 100 training positives): {len(rare)}:", ", ".join(f"{x['name']} {sum(x['train_positives'].values())}" for x in rare[:40]))
    print("FOLIO matches, e.g.:", "; ".join(f"{x['name']} -> {x['folio']['label']}" for x in out if x["folio"])[:900])


if __name__ == "__main__":
    main()
