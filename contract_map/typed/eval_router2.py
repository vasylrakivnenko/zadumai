"""Every local tier's answer per question (router chain, dev copy), so orderings can be compared offline (2026-10-04):
Pre-Tier 0 on the whole text; Tier 0 NLI on the whole text; the reader network on the whole text; the located reading
(Pre-Tier 0 + network on the located pieces); Tier 0 NLI on the located text. usage: eval_router2.py SET [--n N]"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/dev"); sys.path.insert(0, HERE)
from router.pretier0 import check
from router.tier0 import Tier0
from router.netreader import NetReader
from router import located as L
import extract as X

ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--n", type=int); ap.add_argument("--ft", action="store_true")
a = ap.parse_args()
rows = X.SETS[a.set]()
if a.n: rows = rows[:a.n]
import torch
from router import netreader as NRmod
NRmod.TORCH_THREADS = int(os.environ.get("TT", "6"))
t0, net = Tier0(), NetReader(); loc = L.LocatedReader(net)
torch.set_num_threads(int(os.environ.get("TT", "6")))
if a.ft:  # the retrained network for the whole-text read (short texts), at its own threshold
    from router.netreader import T_YES_FT
    net = NetReader(model_dir=f"{HERE}/models/ft1", t_yes=T_YES_FT, t_yes_doc=T_YES_FT)
path = f"{HERE}/runs/router2_{a.set}{'_ft' if a.ft else ''}.jsonl"
done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
with open(path, "a") as f:
    for r in rows:
        if r["id"] in done: continue
        q, doc = r["question"], r["doc"]; o = {"id": r["id"], "gold": r["gold"], "long": len(doc) >= L.LONG_CHARS}
        tt = time.perf_counter(); p0 = check(q, doc); o["pt0"] = p0.answer if p0.fired else None; o["ms_pt0"] = (time.perf_counter() - tt) * 1000
        qq = p0.rewritten or q
        tt = time.perf_counter(); x = t0.answer(qq, doc); o["nli"] = x.answer if x.fired else None; o["ms_nli"] = (time.perf_counter() - tt) * 1000
        tt = time.perf_counter(); n = net.answer(qq, doc); o["net"] = n.answer if n.fired else None; o["ms_net"] = (time.perf_counter() - tt) * 1000
        if o["long"]:
            tt = time.perf_counter(); lr = loc.answer(qq, doc, p0); o["loc"] = lr.answer if lr.fired else None
            o["loc_by"] = "pt0" if lr.fired and lr.confidence == 1.0 and "Pre-Tier 0" in lr.reason else ("net" if lr.fired else None)
            o["ms_loc"] = (time.perf_counter() - tt) * 1000
            cd = loc.compiled(doc); idx = loc.locate(qq, cd, set())
            located = cd.party_line() + "\n\n" + "\n\n".join(cd.pieces[i] for i in idx)
            tt = time.perf_counter(); y = t0.answer(qq, located); o["nli_loc"] = y.answer if y.fired else None; o["ms_nli_loc"] = (time.perf_counter() - tt) * 1000
        f.write(json.dumps(o) + "\n"); f.flush()
print("done", a.set)
