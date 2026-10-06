import sys, collections, re
exec(open("parse.py").read().split("f = sys.argv[1]")[0])
label, parents, alts, defs = load(sys.argv[1])
kids = collections.defaultdict(list)
for k, ps in parents.items():
    for p in ps: kids[p].append(k)
def path(n):
    out = []
    while n in parents and parents[n]:
        out.append(label.get(n, n)); n = parents[n][0]
    return " < ".join(out)
def size(n, seen=None):
    seen = seen if seen is not None else set()
    for c in kids[n]:
        if c not in seen: seen.add(c); size(c, seen)
    return len(seen)
byname = {v: k for k, v in label.items()}
for name in ["Termination Clause", "Supply Agreement", "Most Favored Nation Clause", "End User License Agreement", "Non-Disclosure Agreement", "Indemnification Clause", "Lease Agreement", "Employment Agreement"]:
    if name in byname: print(name, "::", path(byname[name]))
    else: print(name, ":: (not found)", [l for l in label.values() if l and name.split()[0].lower() in l.lower()][:6])
doc = byname["Document / Artifact"]
def show(n, d, maxd):
    for c in sorted(kids[n], key=lambda c: -size(c)):
        print("  " * d + f"{label.get(c)} ({size(c)})")
        if d < maxd: show(c, d + 1, maxd)
show(doc, 0, int(sys.argv[2]))
