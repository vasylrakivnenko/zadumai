"""Wrong answers of one variant on the WRITE contracts. usage: show.py TYPE VARIANT [N]"""
import sys
from explore import load, overl
from clauses import answers
import explore; explore.GRAIN = "paragraph"
t, v = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12; n = 0
for d in load():
    a = answers(d["stmts"], t).get(v); g = d["gold"][t]
    if not a: continue
    ok = (a[0] == "no" and not g) or (a[0] == "yes" and g and overl(a[1], g))
    if ok: continue
    n += 1
    if n > N: continue
    s = a[1]
    print(f"[{d['id'][:28]}] " + (f"head={s.head[:35]!r} @{s.start}\n   OWN: {s.own[:260]!r}" if s else "(no)"))
    print(f"   GOLD: {g[0][2][:200]!r} @{g[0][0]}" if g else "   GOLD: absent")
print("wrong:", n)
