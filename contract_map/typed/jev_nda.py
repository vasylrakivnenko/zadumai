"""Jev on whole NDAs (2026-10-04; the user: "test Jev on 100 NDAs giving it 17 questions so it returns probability of
true / false for each"). ONE request per NDA (systemone takes several questions per request): the whole NDA as the
state, the 17 user-style questions twice:
  "<qid>"    noul: p(true)                                  (what the user asked for)
  "<qid>#3"  choice: yes / no / not stated, with probabilities (the 3-way read: ContractNLI's "not mentioned")
NDAs: nda_tune (the first 100; the stacker has out-of-fold answers for them). Resumable; never re-asks a done NDA.
usage: python jev_nda.py [N] [SET]   (live venv; router/systemone.py from the live tree, read-only)
       -> feats/jev_nda_tune.jsonl {id: "<doc>/<qid>", p_true, choice, probs}"""
import concurrent.futures as cf, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map"); sys.path.insert(0, HERE)
from router.systemone import SystemOne
import extract as X

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SET = sys.argv[2] if len(sys.argv) > 2 else "nda_tune"  # nda_ho: the 60 held-out NDAs (2026-10-04, the user's go-ahead)
OUT = f"{HERE}/feats/jev_{SET}.jsonl"
CHOICES = {"yes": "the NDA says yes", "no": "the NDA says no",
           "not stated": "the NDA does not address this question"}


def main():
    rows = X.SETS[SET]()
    docs = {}
    for r in rows:
        d, q = r["id"].split("/"); docs.setdefault(d, [r["doc"], {}])[1][q] = r["question"]
    done = {json.loads(l)["id"].split("/")[0] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [d for d in list(docs)[:N] if d not in done]
    llm = SystemOne.jev()
    print(f"{len(todo)} NDAs to ask ({len(done)} done)", flush=True)

    def one(d):
        text, qs = docs[d]
        questions = {}
        for q, question in qs.items():
            questions[q] = {"type": "noul", "instructions": question, "criteria": {"true": "", "false": ""}}
            questions[q + "#3"] = {"type": "choice", "instructions": question, "criteria": CHOICES}
        t = time.time(); ans = llm.ask(text, questions)
        return d, ans, time.time() - t

    with cf.ThreadPoolExecutor(4) as ex, open(OUT, "a") as f:
        for k, (d, ans, sec) in enumerate(ex.map(one, todo)):
            for q in docs[d][1]:
                c = ans.get(q + "#3", {})
                f.write(json.dumps({"id": f"{d}/{q}", "p_true": float(ans[q]["noul"]), "choice": c.get("choice"),
                                    "probs": c.get("probabilities", {})}) + "\n")
            f.flush()
            print(f"{k + 1}/{len(todo)} NDA {d} {len(docs[d][0])} chars {sec:.0f}s | calls {llm.calls} tokens in {llm.input_tokens} "
                  f"out {llm.output_tokens} | {llm.model_version}", flush=True)


if __name__ == "__main__":
    main()
