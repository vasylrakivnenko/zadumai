import sys, collections, re
exec(open("parse.py").read().split("f = sys.argv[1]")[0])
label, parents, alts, defs = load(sys.argv[1])
kids = collections.defaultdict(list)
for k, ps in parents.items():
    for p in ps: kids[p].append(k)
def size(n, seen=None):
    seen = seen if seen is not None else set()
    for c in kids[n]:
        if c not in seen: seen.add(c); size(c, seen)
    return len(seen)
byname = {v: k for k, v in label.items()}
n = byname[sys.argv[2]]; maxd = int(sys.argv[3]); full = len(sys.argv) > 4
def show(n, d):
    for c in sorted(kids[n], key=lambda c: (-size(c), label.get(c) or "")):
        extra = (" | alt: " + "; ".join(alts[c][:4])) if full and alts[c] else ""
        print("  " * d + f"{label.get(c)} ({size(c)})" + extra)
        if d < maxd: show(c, d + 1)
print(sys.argv[2], size(n), "direct kids", len(kids[n]))
show(n, 0)
