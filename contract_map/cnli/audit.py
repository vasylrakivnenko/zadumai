"""Label audit (2026-10-03, the user: "do ~360 first, use Fireworks"): two blind LLM checkers (gpt-oss-120b, kimi-k3 on
Fireworks, via reader_net/llm.py's guarded client: 20k output tokens/min, 8 in flight) re-label ContractNLI rows:
 - every disagreement between our system and the gold label: the sealed test's errors (runs/final2_test.json) and the
   nested-CV errors over the 484 tuning NDAs (runs/select_cv.json)
 - a control sample of 100 answers where we agree with the gold label (50 test, 50 CV)
The checkers see the hypothesis and the whole NDA, never the gold label or our answer. Outcome per row: both checkers agree with gold / both agree with us /
split. usage: audit.py [--dry]"""
import argparse, collections, json, os, random, re, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/zadumai_nli_proto/reader_net")
from baseline import POLARITY, DEFAULT, ZIP

MAX_CHARS = 60000  # every NDA (longest 54.6k chars) goes in whole: the throttle counts output tokens only
MODELS = {"gptoss": "accounts/fireworks/models/gpt-oss-120b", "kimi": "accounts/fireworks/models/kimi-k3"}
LAB = {"Entailment": "ENTAILED", "Contradiction": "CONTRADICTED", "NotMentioned": "NOT_MENTIONED"}


def to_label(ans, q):
    if ans == "not mentioned": return "NotMentioned"
    w = POLARITY.get(q, DEFAULT)
    return next(k for k, v in w.items() if v == ans)


def excerpt(doc, hyp, extra):
    t = doc["text"]
    if len(t) <= MAX_CHARS: return t, True
    paras = [p for p in re.split(r"\n\s*\n|\n", t) if p.strip()]
    words = {w for w in re.findall(r"[a-z]{4,}", hyp.lower())}
    scored = sorted(range(len(paras)), key=lambda i: -len(words & set(re.findall(r"[a-z]{4,}", paras[i].lower()))))
    keep = set(); size = sum(len(e) for e in extra)
    for i in scored:
        if size + len(paras[i]) > MAX_CHARS: continue
        keep.add(i); size += len(paras[i])
    body = "\n".join(paras[i] for i in sorted(keep))
    return body + ("\n\n[Also in the NDA:]\n" + "\n".join(extra) if extra else ""), False


PROMPT = """You are checking a non-disclosure agreement (NDA) against a statement, as in the ContractNLI dataset.
Labels:
- ENTAILED: the NDA states or clearly implies the statement.
- CONTRADICTED: the NDA states or clearly implies the opposite of the statement.
- NOT_MENTIONED: the NDA does neither.
Statement: "{hyp}"

NDA{part}:
<<<
{text}
>>>

Reason briefly, then answer with JSON only on the last line: {{"label": "ENTAILED|CONTRADICTED|NOT_MENTIONED", "quote": "<the exact sentence from the NDA that decides it, or empty>"}}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    z = zipfile.ZipFile(ZIP); docs = {}; hyps = None
    for s in ("train", "dev", "test"):
        d = json.loads(z.read(f"contract-nli/{s}.json")); hyps = d["labels"]
        for x in d["documents"]: docs[x["id"]] = x
    rows = []
    for src, path in (("test", "runs/final2_test.json"), ("cv", "runs/select_cv.json")):
        data = json.load(open(path)); data = data["rows"] if isinstance(data, dict) else data
        ans = [r for r in data if r["answer"]]
        wrong = [r for r in ans if to_label(r["answer"], r["q"]) != r["gold"]]
        ok = [r for r in ans if to_label(r["answer"], r["q"]) == r["gold"]]
        random.Random(7).shuffle(ok)
        rows += [{**r, "src": src, "kind": "disagree"} for r in wrong] + [{**r, "src": src, "kind": "control"} for r in ok[:50]]
    items = {}
    for r in rows:
        doc = docs[r["doc"]]; ann = doc["annotation_sets"][0]["annotations"][r["q"]]
        gold_ev = [doc["text"][doc["spans"][i][0]:doc["spans"][i][1]] for i in ann["spans"][:3]]
        text, whole = excerpt(doc, hyps[r["q"]]["hypothesis"], gold_ev)
        items[f"{r['src']}|{r['doc']}|{r['q']}"] = {"hyp": hyps[r["q"]]["hypothesis"], "text": text, "part": "" if whole else " (the parts most relevant to the statement)"}
    print(f"rows: {len(rows)} ({collections.Counter((r['src'], r['kind']) for r in rows)}); requests: {2 * len(items)}; "
          f"prompt chars avg {sum(len(v['text']) for v in items.values()) // max(len(items), 1)}", flush=True)
    if a.dry: return
    import llm
    res = {}
    for name, model in MODELS.items():
        res[name] = llm.run(f"cnli_audit_{name}", items, lambda v: PROMPT.format(**v), workers=8, model=model,
                            effort="medium" if name == "gptoss" else None, max_tokens=4000, progress=50)
    out = []
    inv = {v: k for k, v in LAB.items()}
    for r in rows:
        k = f"{r['src']}|{r['doc']}|{r['q']}"
        labs = {}
        for name in MODELS:
            j = (res[name].get(k) or {}).get("json") or {}
            labs[name] = inv.get(str(j.get("label", "")).upper().strip())
        ours = to_label(r["answer"], r["q"])
        both = labs["gptoss"] if labs["gptoss"] and labs["gptoss"] == labs["kimi"] else None
        verdict = "split" if both is None else ("gold" if both == r["gold"] else ("ours" if both == ours else "neither"))
        out.append({**r, "ours": ours, "checkers": labs, "verdict": verdict})
    json.dump(out, open("runs/audit.json", "w"), indent=0)
    for src in ("test", "cv"):
        for kind in ("disagree", "control"):
            xs = [r for r in out if r["src"] == src and r["kind"] == kind]
            print(f"  {src:4} {kind:9} n {len(xs):3}: checkers agree with gold {sum(r['verdict'] == 'gold' for r in xs):3} | with us "
                  f"{sum(r['verdict'] == 'ours' for r in xs):3} | another label {sum(r['verdict'] == 'neither' for r in xs):3} | split {sum(r['verdict'] == 'split' for r in xs):3}")


if __name__ == "__main__":
    main()
