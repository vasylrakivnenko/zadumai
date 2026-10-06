"""LLM baseline on the same 408 tuning contracts and the same metric (2026-10-03): one request per contract, the six
CUAD questions at once, the model answers present yes/no + an exact quote (Fireworks, reader_net/llm.py's guarded
client). The quote is located in the text; yes is right when it overlaps a gold span (sentence grain) or when the
statement holding it has a paragraph overlapping one (paragraph grain), as in cv.py. usage: llm_baseline.py [--model gptoss|kimi] [--dev-only]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/zadumai_nli_proto/reader_net")
from data import TYPES, question
from front import Located
from cv import tune_compiled
from explore import overl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnli"))
from cv_mentioned import wilson_lo  # noqa: E402

MODELS = {"gptoss": ("accounts/fireworks/models/gpt-oss-120b", "medium"), "kimi": ("accounts/fireworks/models/kimi-k3", None)}
PROMPT = """You are reviewing a commercial contract for six clause types (definitions from the CUAD dataset):
{defs}

Contract:
<<<
{text}
>>>

For each clause type, say whether the contract contains it, and if so quote the single sentence that best shows it,
copied exactly from the contract. Answer with JSON only:
{{{keys}}}"""


def defs():
    return "\n".join(f'- "{t}": {question(t).split("Details:")[1].strip()}' for t in TYPES)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gptoss"); ap.add_argument("--dev-only", action="store_true"); a = ap.parse_args()
    import llm
    docs = tune_compiled()
    if a.dev_only: docs = [d for d in docs if d["dev"]]
    keys = ", ".join(f'"{t}": {{"present": true|false, "quote": "..."}}' for t in TYPES)
    items = {d["id"]: {"text": d["text"]} for d in docs}
    model, effort = MODELS[a.model]
    res = llm.run(f"cuad6_{a.model}", items, lambda v: PROMPT.format(defs=defs(), text=v["text"], keys=keys), workers=8,
                  model=model, effort=effort, max_tokens=6000, progress=25)
    out = []
    for d in docs:
        j = (res.get(d["id"]) or {}).get("json") or {}
        loc = Located(d["text"])
        for t in TYPES:
            x = j.get(t) if isinstance(j, dict) else None
            if not isinstance(x, dict) or "present" not in x:
                out.append({"doc": d["id"], "t": t, "answer": None, "dev": d["dev"]}); continue
            g = d["gold"][t]
            if not x["present"]:
                out.append({"doc": d["id"], "t": t, "answer": "no", "right_s": not g, "right_p": not g, "dev": d["dev"]}); continue
            q = str(x.get("quote") or ""); s, e = loc.find(q)
            if s < 0 and len(q) > 80: s, e = loc.find(q[:80])
            st = next((y for y in d["stmts"] if y.start >= 0 and y.start <= s < max(y.end, y.start + 1)), None) if s >= 0 else None
            class Q: pass
            qq = Q(); qq.start, qq.end = s, e; qq.pstart, qq.pend = (st.pstart, st.pend) if st else (s, e)
            out.append({"doc": d["id"], "t": t, "answer": "yes", "located": s >= 0, "dev": d["dev"],
                        "right_s": bool(g) and overl(qq, g, "sentence"), "right_p": bool(g) and overl(qq, g, "paragraph"), "right_doc": bool(g)})
    json.dump(out, open(f"runs/llm_{a.model}.json", "w"))
    print(f"== LLM baseline {model} on {len(docs)} tuning contracts (answers everything)")
    for t in [*TYPES, None]:
        rows = [r for r in out if (t is None or r["t"] == t) and r["answer"]]
        doc_ok = sum(r["right_s"] if r["answer"] == "no" else r["right_doc"] for r in rows)
        print(f"  {t or 'all':28} answered {len(rows)}/{sum(1 for r in out if t is None or r['t'] == t)}  right sentence {sum(r['right_s'] for r in rows) / max(len(rows), 1):6.1%}"
              f"  paragraph {sum(r['right_p'] for r in rows) / max(len(rows), 1):6.1%}  doc-level {doc_ok / max(len(rows), 1):6.1%}"
              f"  | quotes not found {sum(r['answer'] == 'yes' and not r['located'] for r in rows)}")


if __name__ == "__main__":
    main()
