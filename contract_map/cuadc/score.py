"""Score every variant on the WRITE contracts (development view; CV over all 408 is cv.py).
yes is right when the gold has a span and our statement overlaps one; no is right when the gold is empty."""
import collections, sys
import explore
from explore import load, overl
from data import TYPES
from clauses import answers

explore.GRAIN = "paragraph"; docs = load(); types = sys.argv[1:] or TYPES
for t in types:
    c = collections.defaultdict(lambda: [0, 0, 0])
    for d in docs:
        for v, (a, s) in answers(d["stmts"], t).items():
            g = d["gold"][t]
            ok = (a == "no" and not g) or (a == "yes" and bool(g) and overl(s, g))
            c[v][0] += 1; c[v][1] += ok; c[v][2] += (a == "yes" and bool(g) and not ok)
    print(f"== {t} (present {sum(bool(d['gold'][t]) for d in docs)}/{len(docs)})")
    for v, (n, ok, other) in sorted(c.items(), key=lambda x: -x[1][0]):
        print(f"   {v:16} n {n:4}  right {ok/n:6.1%}  (yes but other span {other})")
