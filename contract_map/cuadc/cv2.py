"""Compiler + LLM, nested 5-fold CV over the 408 tuning contracts (2026-10-03). Sources per type:
  comp:<variant>       a clauses.py variant (yes + statement, or no)
  llm:yes / llm:no     gpt-oss-120b's answer (llm_baseline.py; its quote located in the text)
  both:<variant>       the compiler variant and the LLM agree: both no, or both yes with the LLM's quote inside the
                       compiler statement's paragraph (the compiler's statement is the evidence)
Gating as cv.py: trusted if >= TARGET right on the training folds with >= MIN_N answers; the held-out contract gets the
answer of its most precise trusted source. usage: cv2.py [--target 0.98] [--grain sentence|paragraph] [--model gptoss]"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore
from data import TYPES
from front import Located
from clauses import answers
from cv import tune_compiled, right
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnli"))
from cv_mentioned import wilson_lo  # noqa: E402


class Span:
    def __init__(self, s, e, ps=None, pe=None):
        self.start, self.end = s, e; self.pstart, self.pend = (ps, pe) if ps is not None else (s, e); self.own = ""


def llm_answers(docs, model, name=None):
    path = f"/root/zadumai_nli_proto/reader_net/llm/{name or f'cuad6_{model}'}.jsonl"
    cache = {}
    for line in open(path):
        r = json.loads(line); cache[r["key"]] = r.get("json")
    out = {}
    for d in docs:
        j = cache.get(d["id"])
        if not isinstance(j, dict): continue
        loc = Located(d["text"])
        for t in TYPES:
            x = j.get(t)
            if not isinstance(x, dict) or "present" not in x: continue
            if not x["present"]: out[(d["id"], t)] = ("no", None); continue
            q = str(x.get("quote") or ""); s, e = loc.find(q)
            if s < 0 and len(q) > 80: s, e = loc.find(q[:80])
            st = next((y for y in d["stmts"] if 0 <= y.start <= s < max(y.end, y.start + 1)), None) if s >= 0 else None
            out[(d["id"], t)] = ("yes", Span(s, e, *((st.pstart, st.pend) if st else (s, e))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.98); ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--grain", default="paragraph", choices=["sentence", "paragraph"]); ap.add_argument("--model", default="gptoss")
    ap.add_argument("--no-llm-alone", action="store_true"); a = ap.parse_args(); explore.GRAIN = a.grain
    docs = tune_compiled(); L = llm_answers(docs, a.model)
    have = {d["id"] for d in docs if any((d["id"], t) in L for t in TYPES)}
    docs = [d for d in docs if d["id"] in have]
    out = []; trusted_log = {}
    for t in TYPES:
        src = []
        for d in docs:
            m = {f"comp:{v}": x for v, x in answers(d["stmts"], t).items()}
            l = L.get((d["id"], t))
            if l:
                if not a.no_llm_alone: m[f"llm:{l[0]}"] = l
                for v, x in list(m.items()):
                    if not v.startswith("comp:"): continue
                    if x[0] == "no" and l[0] == "no": m[f"both:{v[5:]}"] = x
                    if x[0] == "yes" and l[0] == "yes" and l[1].start >= 0 and l[1].start < x[1].pend and l[1].end > x[1].pstart:
                        m[f"both:{v[5:]}"] = x
            src.append(m)
        trusted_log[t] = collections.Counter()
        for f in range(5):
            tr = [i for i, d in enumerate(docs) if d["fold"] != f]; te = [i for i, d in enumerate(docs) if d["fold"] == f]
            prec = {}
            for v in {v for i in tr for v in src[i]}:
                xs = [right(src[i][v], docs[i], t) for i in tr if v in src[i]]
                if len(xs) >= a.min_n and sum(xs) / len(xs) >= a.target: prec[v] = sum(xs) / len(xs)
            trusted_log[t].update({v: 1 for v in prec})
            for i in te:
                vs = sorted((v for v in src[i] if v in prec), key=lambda v: -prec[v])
                r = {"doc": docs[i]["id"], "t": t, "present": bool(docs[i]["gold"][t]), "dev": docs[i]["dev"], "answer": None}
                if vs:
                    x = src[i][vs[0]]; r.update(answer=x[0], source=vs[0], right=right(x, docs[i], t))
                out.append(r)
    json.dump(out, open(f"runs/cv2_{a.grain}_{a.target}.json", "w"))
    def summ(rows):
        n = len(rows); an = [r for r in rows if r["answer"]]; ok = sum(r["right"] for r in an)
        return f"{len(an):4}/{n:4} = {len(an) / max(n, 1):5.1%} answered, right {ok / max(len(an), 1):6.1%} [lo {wilson_lo(ok, len(an)):.3f}]"
    print(f"== compiler + LLM ({a.model}), nested 5-fold CV over {len(docs)} contracts, target {a.target}, grain {a.grain}{', no LLM alone' if a.no_llm_alone else ''}")
    for t in TYPES:
        rows = [r for r in out if r["t"] == t]
        print(f"  {t:28} {summ(rows)} | dev {summ([r for r in rows if r['dev']])} | trusted (folds) {dict(trusted_log[t])}")
    print(f"  {'all':28} {summ(out)}\n  {'dev (never read)':28} {summ([r for r in out if r['dev']])}")


if __name__ == "__main__":
    main()
