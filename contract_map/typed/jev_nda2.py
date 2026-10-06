"""Jev on whole NDAs, v2 (2026-10-04): the multi-label pattern of TypeSafe's docs (one Noul per label in one request;
a separate judgment for "nothing applies"). ONE request per NDA, state {"agreement": <the NDA>}, per question two Nouls:
  "<qid>#yes"  does the agreement say the statement holds (states it or clearly implies it)?
  "<qid>#no"   does the agreement say the opposite (contradicts it or clearly rules it out)?
"not stated" = neither. The statement is ContractNLI's hypothesis for the question (it defines what "yes" means: often
existential, "some information", "in some circumstances"); answers map to the user-style question through
eval_docs.POLARITY (nda-15 is flipped). v1 (jev_nda.py) asked the user-style question alone.
usage: python jev_nda2.py SET [N]   (SET: nda_tune | nda_ho) -> feats/jev2_<SET>.jsonl {id, p_yes, p_no}"""
import concurrent.futures as cf, json, os, sys, time, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map"); sys.path.insert(0, HERE)
from router.systemone import SystemOne
import eval_docs as E
import extract as X

SET = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6
OUT = f"{HERE}/feats/jev2_{SET}.jsonl"
HYP = json.loads(zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip").read("contract-nli/train.json"))["labels"]


def questions(qids):
    out = {}
    for q in qids:
        h = HYP[q]["hypothesis"]
        out[q + "#yes"] = {"type": "noul", "instructions": {
            "statement": h, "question": "Does the agreement say that this statement holds: it states it, or it clearly follows from its terms?"},
            "criteria": {"true": "the agreement's terms state or clearly imply the statement",
                         "false": "the agreement says the opposite, or does not address the statement"}}
        out[q + "#no"] = {"type": "noul", "instructions": {
            "statement": h, "question": "Does the agreement say the opposite of this statement: its terms contradict it or clearly rule it out?"},
            "criteria": {"true": "the agreement's terms contradict the statement or rule it out",
                         "false": "the agreement supports the statement, or does not address it"}}
    return out


def main():
    docs = {}
    for r in X.SETS[SET]():
        d, q = r["id"].split("/"); docs.setdefault(d, [r["doc"], []])[1].append(q)
    done = {json.loads(l)["id"].split("/")[0] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [d for d in list(docs)[:N] if d not in done]
    llm = SystemOne.jev()
    print(f"{SET}: {len(todo)} NDAs to ask ({len(done)} done)", flush=True)

    def one(d):
        t = time.time(); return d, llm.ask({"agreement": docs[d][0]}, questions(docs[d][1])), time.time() - t

    with cf.ThreadPoolExecutor(4) as ex, open(OUT, "a") as f:
        for k, (d, ans, sec) in enumerate(ex.map(one, todo)):
            for q in docs[d][1]:
                pe, pc = float(ans[q + "#yes"]["noul"]), float(ans[q + "#no"]["noul"])  # entailment, contradiction
                pol = E.POLARITY.get(q, E.DEFAULT)
                p = {pol["Entailment"]: pe, pol["Contradiction"]: pc}
                f.write(json.dumps({"id": f"{d}/{q}", "p_yes": p["yes"], "p_no": p["no"]}) + "\n")
            f.flush()
            print(f"{k + 1}/{len(todo)} NDA {d} {sec:.0f}s | calls {llm.calls} tokens in {llm.input_tokens} out {llm.output_tokens}", flush=True)


if __name__ == "__main__":
    main()
