"""Contract kind from the parties' roles (a symbol-table signal), vs the live router (2026-10-03).
Documents: bench/family.router_data (the router's own training / held-out split; held-out = the clause sets' test
documents + held-out MCC and ToS;DR, 1,156 documents). Rules role -> kind are counted on the training documents only;
scored on the held-out ones, alone and as a tiebreaker when the router is unsure (p < 0.9)."""
import collections, json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/bench")
import joblib
import numpy as np
from doccompile import compile_doc

MODEL = "/root/projects/zadumai/legalbench_map/models/contract_map/router.joblib"


def roles(text):
    try:
        c = compile_doc(text[:60000])
    except Exception:  # noqa: BLE001
        return []
    return sorted({p.role for p in c.parties if p.role})


if __name__ == "__main__":
    import bench, family
    S = {ds: bench.SETS[ds]() for ds in ("cuad_clause", "contractnli", "opp115", "lease")}
    tr, te = family.router_data(S, family.clause_docs())
    print(f"train {len(tr)}, held-out {len(te)}", collections.Counter(d["label"] for d in te))
    with ProcessPoolExecutor(8) as ex:
        R_tr = list(ex.map(roles, [d["text"] for d in tr], chunksize=8)); R_te = list(ex.map(roles, [d["text"] for d in te], chunksize=8))
    # role -> kind, counted on training documents; a rule when >= 95% of >= 10 documents with the role are that kind
    cnt = collections.defaultdict(collections.Counter)
    for d, rs in zip(tr, R_tr):
        for r in rs: cnt[r][d["label"]] += 1
    rules = {}
    for r, c in cnt.items():
        k, n = c.most_common(1)[0]
        if sum(c.values()) >= 10 and n / sum(c.values()) >= 0.95: rules[r] = k
    print("rules:", rules)
    print("roles with mixed kinds:", {r: dict(c) for r, c in cnt.items() if r not in rules and sum(c.values()) >= 10})
    m = joblib.load(MODEL); P = m["lr"].predict_proba(m["rvec"].transform([d["text"][:4000] for d in te])) if "rvec" in m else \
        m["lr"].predict_proba(m["vec"].transform([d["text"][:4000] for d in te]))
    classes = m["lr"].classes_
    fired = right = 0; sure = sure_ok = unsure = unsure_ok = unsure_fix = unsure_fired = 0
    per = collections.Counter()
    for d, rs, p in zip(te, R_te, P):
        ks = {rules[r] for r in rs if r in rules}
        rk = ks.pop() if len(ks) == 1 else None
        if rk: fired += 1; right += rk == d["label"]; per[rk, rk == d["label"]] += 1
        top = classes[int(p.argmax())]
        if p.max() >= 0.9: sure += 1; sure_ok += top == d["label"]
        else:
            unsure += 1; unsure_ok += top == d["label"]
            if rk: unsure_fired += 1; unsure_fix += rk == d["label"]
    print(f"role rule alone: fires on {fired}/{len(te)} = {fired/len(te):.1%}, right {right}/{fired} = {right/max(1,fired):.1%}; {dict(per)}")
    print(f"router sure (p>=0.9): {sure_ok}/{sure} = {sure_ok/sure:.1%}; unsure: router {unsure_ok}/{unsure} = {unsure_ok/max(1,unsure):.1%}; "
          f"role rule fires on {unsure_fired} of the unsure, right {unsure_fix}")
    json.dump({"rules": rules, "fired": fired, "right": right, "unsure": unsure, "unsure_fired": unsure_fired, "unsure_fix": unsure_fix},
              open("runs/role_kind.json", "w"), indent=1)
