"""Combinations of the neuro-symbolic checks, replayed offline from feats/*.jsonl (extract.py) (2026-10-03).
Decisions per question (first that answers):
  base   today's pipeline: Pre-Tier 0 on the whole text; on a long text Pre-Tier 0 on the located text; the network's
         "yes" at 0.90 on located text (short texts: its own 0.935 / 0.94); its "no" off
  1      consistency with related questions (ns.related): a network "yes" needs the opposite-modality question below 0.5
         and the paraphrase at >= 0.5; "1+" also takes consistent "yes" from 0.80; "1+no" also takes a network "no"
         (p >= 0.90) when the related questions agree with it
  2      priorities (ns.priority): no answer whose statement loses to a conflicting norm (overridden by "notwithstanding",
         deferring by "subject to" to it, or an unresolved conflict); "2+no": a "no" for may-questions whose act has a
         ban nobody excepts, when the network's p(no) for the question is >= 0.5
  3      learned gates (fold.py, default rules with exceptions; and a depth-3 decision tree for comparison) over the
         candidates' features, trained on the tuning sets only, rules kept at >= 98% training precision
usage: combos.py [--tag TAG]"""
import argparse, collections, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fold

TUNE = ["nda_tune", "cuad_tune", "wc1_open", "short_tune"]
HELD = ["nda_ho", "cuad_ho", "short_ho", "wc1_sealed"]


def wilson_lo(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def load(name, tag=""):
    p = f"{HERE}/feats/{name}{('_' + tag) if tag else ''}.jsonl"
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    seen = set(); rows = [o for o in rows if not (o["id"] in seen or seen.add(o["id"]))]  # two writers may overlap
    p5 = f"{HERE}/feats/{name}{('_' + tag) if tag else ''}_q5.jsonl"
    if os.path.exists(p5):
        q5 = {json.loads(l)["id"]: json.loads(l) for l in open(p5)}
        for o in rows:
            if o["id"] in q5: o["q5"] = q5[o["id"]]
    return rows


def rel(o, kind):
    for r in o.get("rel") or []:
        if r["kind"] == kind: return r
    return None


def feats(o) -> dict:
    asks = (o.get("frame") or {}).get("asks")
    ban, perm, para = rel(o, "ban"), rel(o, "perm"), rel(o, "para")
    opp = ban if asks in ("CAN", "MUST") else perm if asks == "PROHIBITED" else None
    pr, prn = o.get("prio") or {}, o.get("prio_no") or {}
    return {"p": float(o.get("net_p") or 0), "long": o["long"], "asks": asks or "untyped", "units": o.get("n_units", 0),
            "opp_yes": opp["pyes"] if opp else None, "para_yes": para["pyes"] if para else None,
            "para_no": para["pno"] if para else None, "q_pyes": o.get("q_pyes"), "q_pno": o.get("q_pno"),
            "pr_conflicts": pr.get("pr_conflicts", 0), "pr_bad": bool(pr.get("pr_overridden") or pr.get("pr_unresolved")
                                                                    or pr.get("pr_defers_to_conflict")),
            "pr_ev_found": bool(pr.get("pr_ev_found")), "pr_ban": prn.get("pr_ban", 0), "pr_permit": prn.get("pr_permit", 0),
            "pr_ban_unexcepted": bool(prn.get("pr_ban_unexcepted")), "typed_ok": o.get("typed_ok"),
            "checks_failed": bool(o.get("net_checks"))}


def consistent(o, answer) -> bool | None:
    f = feats(o)
    if f["opp_yes"] is None or f["para_yes"] is None: return None
    if answer == "yes": return f["opp_yes"] < 0.5 and f["para_yes"] >= 0.5
    if answer == "no": return f["opp_yes"] >= 0.5 and f["para_yes"] < 0.5
    return None


T_OVERRIDE = {}  # {"long": t, "short": t}: thresholds re-picked on the tuning rows (--retune, for another network)


def t_base(o):
    if T_OVERRIDE:
        return T_OVERRIDE["long"] if o["long"] else T_OVERRIDE["short"]
    return 0.90 if o["long"] else (0.935 if o.get("n_units", 1) <= 1 else 0.94)


def retune(rows, target_long=0.98, target_short=0.995):
    """The lowest network "yes" threshold whose tuning answers are still >= target right (long / short texts)."""
    out = {}
    for name, sel, target in (("long", lambda o: o["long"], target_long), ("short", lambda o: not o["long"], target_short)):
        cands = sorted(((float(o.get("net_p") or 0), o["gold"] == "yes") for o in rows
                        if sel(o) and not (o.get("pt0") or o.get("pt0_loc")) and o.get("net") == "yes"), reverse=True)
        ok = n = 0; best = 1.01
        for p, right in cands:
            n += 1; ok += right
            if n >= 20 and ok / n >= target: best = p
        out[name] = best
    return out


def decide(o, combo, gates=None):
    """(answer, path) under a combination."""
    f = feats(o)
    pr_bad = f["pr_bad"] and "2" in combo
    if o.get("pt0"):
        return (None, "pt0-vetoed") if pr_bad and "2all" in combo else (o["pt0"], "pt0")
    if o.get("pt0_loc"):
        return (None, "pt0loc-vetoed") if pr_bad else (o["pt0_loc"], "pt0-located")
    net, p = o.get("net"), f["p"]
    if net == "yes":
        if gates is not None:
            if fold.accept(gates["yes"], f): return "yes", "net-gate"
        else:
            c = consistent(o, "yes") if "1" in combo else None
            t = t_base(o)
            ok = p >= t
            if "1+" in combo and c: ok = p >= min(t, 0.80)
            if "1" in combo and c is False: ok = False
            if pr_bad: ok = False
            if ok: return "yes", "net"
    if net == "no" and ("1+no" in combo or gates is not None):
        if gates is not None:
            if fold.accept(gates["no"], f): return "no", "net-no-gate"
        elif p >= 0.90 and consistent(o, "no") and not pr_bad:
            return "no", "net-no"
    if "2+no" in combo and f["asks"] == "CAN" and f["pr_ban_unexcepted"] and (f["q_pno"] or 0) >= 0.5 and net != "yes" \
            and (f["opp_yes"] or 0) >= 0.5:  # the network's "prohibited" question agrees (ideas 1 + 2)
        return "no", "prio-no"
    return None, "none"


def decide5(o, combo, gates=None):
    """Idea 5 for a question the grammar left untyped: first person by supervaluation, two actions by composition;
    the learned parser's frame drives the priority veto ("5n")."""
    e = o.get("q5") or {}
    base_combo = combo.replace("5", "")
    if e.get("fp"):
        ans = [decide({**x["f"], "gold": o["gold"]}, base_combo, gates)[0] for x in e["fp"]]
        if len(ans) >= 2 and ans[0] and all(a == ans[0] for a in ans): return ans[0], "first-person"
    if e.get("split"):
        parts = [decide({**x, "gold": o["gold"]}, base_combo, gates)[0] for x in e["split"]["parts"]]
        a = q5c(e["split"]["conn"], parts)
        if a: return a, "split"
    return None, "none"


def q5c(conn, answers):
    if conn == "or":
        if any(a == "yes" for a in answers): return "yes"
        if answers and all(a == "no" for a in answers): return "no"
        return None
    if all(a == "yes" for a in answers): return "yes"
    if any(a == "no" for a in answers): return "no"
    return None


def decide_all(o, combo, gates=None):
    if "5" in combo and not o.get("frame"):
        if "5n" in combo and (o.get("q5") or {}).get("prio_n"):
            o = {**o, "prio": o["q5"]["prio_n"], "prio_no": o["q5"].get("prio_no_n") or {}, "typed_ok": o["q5"].get("typed_ok_n")}
        a, p = decide(o, combo.replace("5n", "").replace("5", ""), gates)
        if a: return a, p
        return decide5(o, combo, gates)
    return decide(o, combo, gates)


def score(rows, combo, gates=None):
    a = [(decide_all(o, combo, gates), o["gold"]) for o in rows]
    ans = [(x, g) for (x, _), g in a if x]; ok = sum(x == g for x, g in ans)
    paths = collections.Counter(p for (x, p), _ in a if x)
    return len(ans), ok, len(rows), paths


def domain(o):
    s = o.get("set", "")
    return "nda" if s.startswith("nda") else "commercial" if s.startswith(("cuad", "wc1")) else "short"


def gfeats(o):
    f = feats(o); f["domain"] = domain(o); f["base_ok"] = f["p"] >= t_base(o)
    return f


def train_gates_d(rows, add_prec=0.98, veto_ratio=0.5):
    """Idea 3 as FOLD's default + exceptions: today's rule is the default; learned VETO rules (when its accepted network
    "yes" is wrong) and ADD rules (a rejected "yes", or a network "no", that is right at >= add_prec)."""
    fs = ["p", "long", "asks", "units", "opp_yes", "para_yes", "para_no", "q_pyes", "q_pno", "pr_conflicts", "pr_bad",
          "pr_ev_found", "pr_ban", "pr_permit", "pr_ban_unexcepted", "typed_ok", "checks_failed", "domain"]
    yes = [o for o in rows if not (o.get("pt0") or o.get("pt0_loc")) and o.get("net") == "yes"]
    acc = [o for o in yes if gfeats(o)["base_ok"]]; rej = [o for o in yes if not gfeats(o)["base_ok"]]
    veto = fold.learn([gfeats(o) for o in acc], [o["gold"] != "yes" for o in acc], fs, min_prec=veto_ratio, min_cover=3)
    add = fold.learn([gfeats(o) for o in rej], [o["gold"] == "yes" for o in rej], fs, min_prec=add_prec, min_cover=10)
    nos = [o for o in rows if not (o.get("pt0") or o.get("pt0_loc")) and o.get("net") == "no"]
    add_no = fold.learn([gfeats(o) for o in nos], [o["gold"] == "no" for o in nos], fs, min_prec=add_prec, min_cover=10)
    return {"veto": veto, "add": add, "add_no": add_no, "n": (len(acc), len(rej), len(nos))}


def decide_v(o, g):
    """Vetoes only: today's rule minus the learned veto rules."""
    if o.get("pt0"): return o["pt0"], "pt0"
    if o.get("pt0_loc"): return o["pt0_loc"], "pt0-located"
    f = gfeats(o)
    if o.get("net") == "yes" and f["base_ok"] and not fold.accept(g["veto"], f): return "yes", "net"
    return None, "none"


def decide_cv(o, combo, g):
    """A hand combination (1 / 2 / 5 ...) with the learned vetoes on its network "yes" answers (ideas 1-2-5 + 3)."""
    a, p = decide_all(o, combo)
    if a == "yes" and p == "net" and fold.accept(g["veto"], gfeats(o)): return None, "vetoed"
    return a, p


def decide_d(o, g):
    if o.get("pt0"): return o["pt0"], "pt0"
    if o.get("pt0_loc"): return o["pt0_loc"], "pt0-located"
    f = gfeats(o)
    if o.get("net") == "yes":
        if f["base_ok"] and not fold.accept(g["veto"], f): return "yes", "net"
        if not f["base_ok"] and fold.accept(g["add"], f): return "yes", "net-added"
    if o.get("net") == "no" and fold.accept(g["add_no"], f): return "no", "net-no-added"
    return None, "none"


def train_gates(rows, min_prec=0.98):
    out = {}
    for which in ("yes", "no"):
        xs, ys = [], []
        for o in rows:
            if o.get("pt0") or o.get("pt0_loc") or o.get("net") != which: continue
            xs.append(feats(o)); ys.append(o["gold"] == which)
        fs = ["p", "long", "asks", "units", "opp_yes", "para_yes", "para_no", "q_pyes", "q_pno", "pr_conflicts", "pr_bad",
              "pr_ev_found", "pr_ban", "pr_permit", "pr_ban_unexcepted", "typed_ok", "checks_failed"]
        out[which] = fold.learn(xs, ys, fs, min_prec=min_prec)
        out[which + "_n"] = (len(xs), sum(ys))
    return out


def doc_of(o):
    i = o["id"]
    return i.split("/")[0] if "/" in i else i


def cv_gates_d(rows, k=5):
    import zlib
    tot = collections.Counter()
    for f in range(k):
        tr = [o for o in rows if zlib.crc32(doc_of(o).encode()) % k != f]
        te = [o for o in rows if zlib.crc32(doc_of(o).encode()) % k == f]
        g = train_gates_d(tr)
        for o in te:
            if o.get("pt0") or o.get("pt0_loc"): continue
            a, _ = decide_d(o, g); b, _ = decide(o, "base"); v, _ = decide_v(o, g)
            if a: tot["gate"] += 1; tot["gate_ok"] += a == o["gold"]
            if v: tot["veto"] += 1; tot["veto_ok"] += v == o["gold"]
            if b: tot["base"] += 1; tot["base_ok"] += b == o["gold"]
    return tot


def cv_gates(rows, k=5):
    """The learned gates trained on 4/5 of the tuning documents, applied to the other 1/5: network candidates only."""
    import zlib
    tot = collections.Counter()
    for f in range(k):
        tr = [o for o in rows if zlib.crc32(doc_of(o).encode()) % k != f]
        te = [o for o in rows if zlib.crc32(doc_of(o).encode()) % k == f]
        g = train_gates(tr)
        for o in te:
            if o.get("pt0") or o.get("pt0_loc"): continue
            a, _ = decide(o, "3", g)
            b, _ = decide(o, "base")
            if a: tot["gate"] += 1; tot["gate_ok"] += a == o["gold"]
            if b: tot["base"] += 1; tot["base_ok"] += b == o["gold"]
    return tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--retune", action="store_true")
    a = ap.parse_args()
    data = {s: load(s, a.tag) for s in TUNE + HELD}
    if a.retune:
        T_OVERRIDE.update(retune([o for s in TUNE for o in data[s]]))
        print("thresholds re-picked on the tuning rows:", T_OVERRIDE)
    combos = ["base", "1", "1+", "1+no", "2", "2+no", "2all", "12", "1+2+no", "1+no2+no", "5", "5n", "1+no2+no5n"]
    tune_rows = [o for s in TUNE for o in data[s]]
    gates = train_gates(tune_rows)
    print("learned gates (trained on", ", ".join(f"{s} {len(data[s])}" for s in TUNE), "):")
    for w in ("yes", "no"):
        print(f"  {w}: {gates[w + '_n'][0]} candidates, {gates[w + '_n'][1]} right")
        for r in gates[w]: print("   ", r.show().replace("\n", "\n    "))
    gd = train_gates_d(tune_rows)
    print(f"default + exceptions gates (accepted / rejected yes, no candidates: {gd['n']}):")
    for name in ("veto", "add", "add_no"):
        for r in gd[name]: print(f"  {name}: " + r.show().replace("\n", "\n    "))
    cvd = cv_gates_d(tune_rows)
    print(f"default+exceptions gates, 5-fold CV over the tuning documents (network candidates): {cvd['gate_ok']}/{cvd['gate']}"
          f" = {cvd['gate_ok'] / max(1, cvd['gate']):.1%}; vetoes only {cvd['veto_ok']}/{cvd['veto']} = {cvd['veto_ok'] / max(1, cvd['veto']):.1%};"
          f" base {cvd['base_ok']}/{cvd['base']} = {cvd['base_ok'] / max(1, cvd['base']):.1%}")
    cv = cv_gates(tune_rows)
    print(f"gates, 5-fold CV over the tuning documents (network candidates only): gates {cv['gate_ok']}/{cv['gate']}"
          f" = {cv['gate_ok'] / max(1, cv['gate']):.1%} vs base {cv['base_ok']}/{cv['base']} = {cv['base_ok'] / max(1, cv['base']):.1%}")
    for s in TUNE + HELD:
        rows = data[s]
        if not rows: continue
        print(f"\n== {s} ({len(rows)} questions{'; tuning' if s in TUNE else '; HELD-OUT' if s != 'wc1_sealed' else '; spent'})")
        for c in combos + ["3", "3d", "3v", "1+v", "1+no+v", "1+no2+no5+v"]:
            if c in ("3d", "3v") or c.endswith("+v"):
                fn = decide_d if c == "3d" else decide_v if c == "3v" else (lambda o, g, c=c: decide_cv(o, c[:-2], g))
                a_ = [(fn(o, gd), o["gold"]) for o in rows]
                ans = [(x, g_) for (x, _), g_ in a_ if x]; n, ok, tot = len(ans), sum(x == g_ for x, g_ in ans), len(rows)
                paths = collections.Counter(p_ for (x, p_), _ in a_ if x)
            else:
                n, ok, tot, paths = score(rows, c, gates if c.startswith("3") else None)
            print(f"  {c:9s} answered {n:5d} = {n / tot:6.1%}  right {ok:5d} = {ok / max(1, n):6.1%} [lo {wilson_lo(ok, n):.3f}]  "
                  + " ".join(f"{k}:{v}" for k, v in sorted(paths.items())))


if __name__ == "__main__":
    main()
