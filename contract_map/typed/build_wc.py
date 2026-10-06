"""wc1: user-style yes/no questions over WHOLE contracts (2026-10-03). No earlier set has them: v4-v7 are over single
provisions, cuad_blind3 / docs_pt0 ask CUAD's checklist questions. Contracts: the Material Contracts Corpus sample
(SEC filings; texts stored up to 30,000 characters), labels other than "na", >= 15,000 characters, 45 drawn with
random.Random(7). Each contract: gpt-oss-120b writes 12 questions with answers and quotes (one call), kimi-k3 answers
them blind (one call). Kept: the two agree, and a yes/no row's quote is found in the contract. Split by contract:
18 open (tune on them), 27 sealed (run once, at the end). 90 LLM calls (ledger.py).
usage: build_wc.py"""
import collections, json, random, re, sys, zlib
sys.path.insert(0, "/root/zadumai_nli_proto/reader_net")
import llm
from ledger import pending, reserve
llm.chat.__defaults__ = tuple(600 if d == 120 else d for d in llm.chat.__defaults__)  # kimi on 30k-char contracts takes > 120 s

HERE = "/root/zadumai_nli_proto/contract_map/typed"
N, OPEN = 45, 18
KIMI = "accounts/fireworks/models/kimi-k3"

WRITE = """You write test questions for a system that answers yes/no questions about a whole contract.

Contract:
\"\"\"{text}\"\"\"

Write 12 yes/no questions that a party to this contract (or its lawyer) might type, each answered from this contract alone:
- 4 that the contract answers "yes";
- 3 that the contract answers "no": the contract itself forbids, denies or excludes it. Silence is never "no": if the contract just doesn't mention something, that question is "not_stated";
- 4 that the contract does NOT settle although they are close to what it covers: the same action by another party, "must" where it only says "may", a detail or amount it doesn't give, or a related right it doesn't grant;
- 1 more of any kind.
The answer must hold for the whole contract: check definitions, exceptions and other sections that change it.
Use each of these shapes twice:
  "direct": party first: "Can the Supplier ...?", "Does the Tenant have to ...?"
  "condition_first": starts with the situation: "If ..., does/can/must ...?", "After termination, can ...?"
  "passive": no actor up front: "Must notices be given in writing?", "Can this agreement be assigned ...?"
  "two_actions": two actions at once: "Can ... sell or transfer ...?", "Does ... have to return or destroy ...?"
  "noun": a noun phrase or -ing form instead of a verb: "Is there a cap on ...?", "Does the agreement include a non-compete?"
  "plain": the way a non-lawyer talks: "Can I ...?", "Do I have to ...?", "Can they ...?"
Rules: the examples only show each shape's form, never reuse them; ask in your own words, don't copy the contract's
phrasing; name parties only as the contract names them, or generically ("either party", "the other party"); in "plain"
questions "I" is a party you name in the "who" field; each question must make sense on its own.
For each question give its shape, who "I"/"we" is for plain questions ("" otherwise), the answer, whether the answer
depends on a condition or exception the contract states, and the shortest exact quote that decides it ("" for not_stated).

Reply with JSON only: {{"questions": [{{"shape": "...", "who": "...", "q": "...", "answer": "yes" | "no" | "not_stated", "conditional": true | false, "quote": "..."}}, ...]}}"""

CHECK = """You answer yes/no questions about a contract, using only the contract.

Contract:
\"\"\"{text}\"\"\"

Questions:
{questions}

For each question, considering the whole contract (definitions, exceptions, other sections):
- "yes": the contract says so.
- "no": the contract itself says the opposite: it forbids, denies or excludes it. Silence is never "no".
- "not_stated": the contract doesn't settle it: it is silent, it is about another party or another action, it only permits what the question asks is required, or it leaves the point open.
"conditional" is true when that answer depends on a condition or exception the contract states. "quote" is the shortest exact span that decides it ("" for not_stated).

Reply with JSON only: {{"answers": [{{"n": 1, "answer": "yes" | "no" | "not_stated", "conditional": true | false, "quote": "..."}}, ...]}}"""


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    rows = [json.loads(l) for l in open("/root/zadumai_nli_proto/contract_map/data/ext/mcc/mcc_sample.jsonl")]
    rows = [r for r in rows if r.get("text") and r["label"] != "na" and len(r["text"]) >= 15000]
    random.Random(7).shuffle(rows); docs = rows[:N]
    print(f"{len(docs)} contracts:", collections.Counter(d["label"] for d in docs))
    items = {d["doc_key"]: {"text": d["text"], "who": ""} for d in docs}
    reserve("wc1 write (gpt-oss-120b)", pending("wc1_write", items))
    w = llm.run("wc1_write", items, lambda it: WRITE.format(text=it["text"]), workers=8, effort="medium", max_tokens=12000,
                backend="fireworks", model=llm.GPT_OSS)
    qs = {k: ((w.get(k) or {}).get("json") or {}).get("questions") or [] for k in items}
    citems = {k: {"text": items[k]["text"], "qs": v} for k, v in qs.items() if v}
    reserve("wc1 blind check (kimi-k3)", pending("wc1_check", citems))
    c = llm.run("wc1_check", citems, lambda it: CHECK.format(text=it["text"], questions="\n".join(f"{i + 1}. {q['q']}" for i, q in enumerate(it["qs"]))),
                workers=8, effort=None, max_tokens=8000, backend="fireworks", model=KIMI)
    out, st = [], collections.Counter()
    for k, it in citems.items():
        ans = {a.get("n"): a for a in (((c.get(k) or {}).get("json") or {}).get("answers") or []) if isinstance(a, dict)}
        doc = norm(it["text"]); half = "open" if sorted(citems).index(k) < OPEN else "sealed"
        for i, q in enumerate(it["qs"], 1):
            g, b = q.get("answer"), (ans.get(i) or {}).get("answer")
            st["written"] += 1
            if g not in ("yes", "no", "not_stated"): st["bad answer"] += 1; continue
            if b != g: st[f"disagree {g}->{b}"] += 1; continue
            if g != "not_stated" and norm(q.get("quote", ""))[:60] not in doc: st["quote not found"] += 1; continue
            st["kept"] += 1
            out.append({"id": f"wc1-{len(out)}", "doc": k, "label": next(d["label"] for d in docs if d["doc_key"] == k), "half": half,
                        "question": q["q"].strip(), "who": q.get("who", ""), "shape": q.get("shape"), "gold": g,
                        "conditional": bool(q.get("conditional")), "quote": q.get("quote", ""), "quote_check": (ans.get(i) or {}).get("quote", "")})
    json.dump({"docs": {d["doc_key"]: d["text"] for d in docs}, "rows": out}, open(f"{HERE}/wc1.json", "w"))
    print(dict(st)); print("kept by half/gold:", collections.Counter((r["half"], r["gold"]) for r in out))


if __name__ == "__main__":
    main()
