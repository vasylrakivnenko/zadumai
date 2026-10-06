"""The one sealed run on CUAD's 102 test contracts (2026-10-03). Configuration from all 408 tuning contracts, exactly as
cv.py / cv2.py do inside a fold: per type, sources >= TARGET right with >= MIN_N answers are trusted; a test contract
gets its most precise trusted source's answer. Reports three systems on the same rows: compiler only, compiler + LLM
(gpt-oss-120b, same prompt as llm_baseline.py), and the LLM alone (answers everything). Logged in sealed_runs.log;
refuses a second run. usage: final.py [--target 0.98] [--grain paragraph]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/zadumai_nli_proto/reader_net")
import explore
from data import TYPES, test_docs
from front import compile_contract
from clauses import answers
from cv import tune_compiled, right
from cv2 import llm_answers
from llm_baseline import PROMPT, MODELS, defs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnli"))
from cv_mentioned import wilson_lo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def sources(d, t, L):
    m = {f"comp:{v}": x for v, x in answers(d["stmts"], t).items()}
    l = L.get((d["id"], t))
    if l:
        m[f"llm:{l[0]}"] = l
        for v, x in list(m.items()):
            if not v.startswith("comp:"): continue
            if x[0] == "no" and l[0] == "no": m[f"both:{v[5:]}"] = x
            if x[0] == "yes" and l[0] == "yes" and l[1].start >= 0 and l[1].start < x[1].pend and l[1].end > x[1].pstart: m[f"both:{v[5:]}"] = x
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.98); ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--grain", default="paragraph"); a = ap.parse_args(); explore.GRAIN = a.grain
    log = f"{HERE}/sealed_runs.log"
    if os.path.exists(log) and "final" in open(log).read(): raise SystemExit("the sealed test was already run; refusing a second run")
    open(log, "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
    tune = tune_compiled(); Lt = llm_answers(tune, "gptoss")
    test = test_docs()
    for d in test: d["stmts"] = compile_contract(d["text"])
    import llm
    keys = ", ".join(f'"{t}": {{"present": true|false, "quote": "..."}}' for t in TYPES)
    model, effort = MODELS["gptoss"]
    llm.run("cuad6_gptoss_test", {d["id"]: {"text": d["text"]} for d in test}, lambda v: PROMPT.format(defs=defs(), text=v["text"], keys=keys),
            workers=8, model=model, effort=effort, max_tokens=6000, progress=25)
    Lx = llm_answers(test, "gptoss", name="cuad6_gptoss_test")
    rows = []
    for system in ("compiler", "compiler+llm"):
        for t in TYPES:
            tm = [sources(d, t, Lt if system != "compiler" else {}) for d in tune]
            prec = {}
            for v in {v for m in tm for v in m}:
                xs = [right(m[v], d, t) for m, d in zip(tm, tune) if v in m]
                if len(xs) >= a.min_n and sum(xs) / len(xs) >= a.target: prec[v] = sum(xs) / len(xs)
            for d in test:
                m = sources(d, t, Lx if system != "compiler" else {})
                vs = sorted((v for v in m if v in prec), key=lambda v: -prec[v])
                r = {"system": system, "doc": d["id"], "t": t, "answer": None, "present": bool(d["gold"][t])}
                if vs: x = m[vs[0]]; r.update(answer=x[0], source=vs[0], right=right(x, d, t))
                rows.append(r)
            if system == "compiler+llm":
                for d in test:
                    l = Lx.get((d["id"], t)); r = {"system": "llm alone", "doc": d["id"], "t": t, "answer": None, "present": bool(d["gold"][t])}
                    if l: r.update(answer=l[0], right=right(l, d, t))
                    rows.append(r)
    json.dump(rows, open(f"{HERE}/runs/final_test.json", "w"))
    def summ(rs):
        n = len(rs); an = [r for r in rs if r["answer"]]; ok = sum(r["right"] for r in an)
        return f"{len(an):4}/{n} = {len(an) / max(n, 1):5.1%} answered, right {ok}/{len(an)} = {ok / max(len(an), 1):6.1%} [lo {wilson_lo(ok, len(an)):.3f}]"
    print(f"== SEALED TEST: {len(test)} CUAD test contracts x {len(TYPES)} types (target {a.target}, min_n {a.min_n}, grain {a.grain})")
    for system in ("compiler", "compiler+llm", "llm alone"):
        print(f"  -- {system}")
        for t in TYPES: print(f"     {t:28} {summ([r for r in rows if r['system'] == system and r['t'] == t])}")
        print(f"     {'all':28} {summ([r for r in rows if r['system'] == system])}")


if __name__ == "__main__":
    main()
