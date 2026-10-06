"""Threshold sweep for the network on located documents (a run made with a low --t-located): answers at p >= t,
with the Pre-Tier 0 answers (whole and located) unchanged. usage: sweep.py SET RUN [RUN2 ...]"""
import json, os, sys
NO_T = float(os.environ.get("NO_T", "1.01"))  # a network "no" counts only at this probability
from eval_docs import wilson_lo
for run in sys.argv[2:]:
    out = json.load(open(f"/root/zadumai_nli_proto/contract_map/typed/runs/{sys.argv[1]}_{run}.json"))
    n = len(out); fixed = [o for o in out if o["path"] in ("pt0", "pt0-located")]
    fok = sum(o["answer"] == o["gold"] for o in fixed)
    net = [o for o in out if o["path"] == "net-located"]
    nos = [o for o in net if o["answer"] == "no"]
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        x = [o for o in nos if o.get("p", 0) >= t]; print(f"  network no at p >= {t}: {sum(o['gold'] == 'no' for o in x)}/{len(x)} right")
    print(f"{run}: {n} rows; Pre-Tier 0 (whole + located) {fok}/{len(fixed)}; network candidates {len(net)}")
    for t in (0.5, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94):
        a = [o for o in net if o.get("p", 0) >= t and (o["answer"] == "yes" or len(sys.argv) > 1 and NO_T and o.get("p", 0) >= NO_T)]
        ok = sum(o["answer"] == o["gold"] for o in a)
        tot, tok = len(fixed) + len(a), fok + ok
        print(f"  t {t:.2f}: network {ok}/{len(a)} = {ok / max(1, len(a)):.1%} | all {tok}/{tot} = {tok / max(1, tot):.1%} "
              f"[lo {wilson_lo(tok, tot):.3f}], answered {tot / n:.1%}")
