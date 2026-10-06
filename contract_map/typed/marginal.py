"""Marginal effect of each combination against today's rule (base): the answers it adds (right / all) and the answers
it removes (wrong / all), per set; and the consistency rule's lower threshold swept on the tuning sets.
usage: marginal.py [--tag TAG]"""
import argparse, collections, sys
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
import combos as C

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--retune", action="store_true"); a = ap.parse_args()
data = {s: C.load(s, a.tag) for s in C.TUNE + C.HELD}
if a.retune:
    C.T_OVERRIDE.update(C.retune([o for s in C.TUNE for o in data[s]])); print("thresholds:", C.T_OVERRIDE)
tune_rows = [o for s in C.TUNE for o in data[s]]
gd = C.train_gates_d(tune_rows)


def dec(o, c):
    if c == "3d": return C.decide_d(o, gd)
    if c == "3v": return C.decide_v(o, gd)
    if "&v" in c: return C.decide_cv(o, c.replace("&v", ""), gd)
    return C.decide_all(o, c)


def margin(rows, c):
    add = rem = add_ok = rem_wrong = 0
    for o in rows:
        b, _ = C.decide_all(o, "base"); x, _ = dec(o, c)
        if x and x != b: add += 1; add_ok += x == o["gold"]
        if b and x != b: rem += 1; rem_wrong += b != o["gold"]
    return f"+{add_ok}/{add} right, -{rem} ({rem_wrong} were wrong)"


combos = ["1", "1+", "1+no", "2", "2+no", "12", "5", "3v", "3d", "1+&v", "1+no&v"]
for s in C.TUNE + C.HELD:
    rows = data[s]
    if not rows: continue
    print(f"== {s}")
    for c in combos: print(f"   {c:8s} {margin(rows, c)}")

# the consistency rule's lower threshold: marginal precision of the consistent "yes" answers it adds, by band
print("\nconsistent network 'yes' below today's threshold, by p band (tuning sets | held-out sets):")
for lo, hi in ((0.80, 0.85), (0.85, 0.88), (0.88, 0.90), (0.90, 0.935)):
    out = []
    for group in (C.TUNE, C.HELD[:3]):
        n = ok = 0
        for s in group:
            for o in data[s]:
                if o.get("pt0") or o.get("pt0_loc") or o.get("net") != "yes": continue
                p = float(o.get("net_p") or 0)
                if lo <= p < hi and p < C.t_base(o) and C.consistent(o, "yes"): n += 1; ok += o["gold"] == "yes"
        out.append(f"{ok}/{n}")
    print(f"   p in [{lo}, {hi}): {out[0]} | {out[1]}")
