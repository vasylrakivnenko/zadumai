"""Training data for the NDA statement encoder (2026-10-04; the user's idea: a tiny encoder, fully fine-tuned, for the
17 NDA questions only). One example per compiled statement (doccompile: lead-ins prefixed, page furniture out, its
section heading in front), with 17 labels, one per question: 0 = says no, 1 = says yes, 2 = says nothing about it.
A statement says yes / no on a question when it overlaps one of ContractNLI's evidence spans for that question
(the document's label under the user-style question: eval_docs.nda, nda-15 flipped); everything else is 2.
Splits (documents; the order of extract.py: train.json shuffled by random.Random(5)):
  train  the stacker's 384 training NDAs (nda_tune 0-99 + nda_more: 200- and dev), with fold = crc32(id) % 4
  ho     nda_ho (100-159)
  s160   160-199 (never used by anything: sealed)
  test   ContractNLI test (no network of ours trained on it: sealed)
usage: python build.py   -> data/{train,ho,s160,test}.jsonl"""
import json, os, random, re, sys, zipfile, zlib
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/.."); sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/doccompile")
import eval_docs as E
from doccompile import compile_doc

QIDS = list(E.QUESTIONS)  # nda-1 .. nda-20 (17)
MAX_CHARS = 1500


def doc_rows(item):
    d, split = item
    ann = d["annotation_sets"][0]["annotations"]
    gold = {}
    for k in QIDS:
        ch = ann[k]["choice"]
        if ch == "NotMentioned": continue
        g = E.POLARITY.get(k, E.DEFAULT)[ch]
        gold[k] = (0 if g == "no" else 1, [tuple(d["spans"][i]) for i in ann[k]["spans"]])
    c = compile_doc(d["text"])
    out, hit = [], {k: 0 for k in gold}
    for n, st in enumerate(c.stmts):
        if len(st.own.strip()) < 4: continue
        secs = [i for i in st.secs if not c.sections[i].front and c.sections[i].heading]
        head = c.sections[secs[-1]].heading if secs else ""
        t = re.sub(r"\s+", " ", st.text).strip()
        text = (f"{head}. {t}" if head and not t.lower().startswith(head.lower()[:20]) else t)[:MAX_CHARS]
        lab = [2] * len(QIDS)
        if st.start >= 0:
            for k, (y, spans) in gold.items():
                for a, b in spans:
                    ov = min(b, st.end) - max(a, st.start)
                    if ov >= 20 or (ov > 0 and ov >= 0.5 * min(b - a, st.end - st.start)):
                        lab[QIDS.index(k)] = y; hit[k] += 1; break
        out.append({"doc": d["id"], "sid": n, "text": text, "labels": lab})
    gold_q = {k: v[0] for k, v in gold.items()}
    return split, d["id"], out, gold_q, sum(1 for k in gold if not hit[k])


if __name__ == "__main__":
    z = zipfile.ZipFile(f"{HERE}/../../data/ext/contractnli.zip")
    tr = json.loads(z.read("contract-nli/train.json"))["documents"]; random.Random(5).shuffle(tr)
    dev = json.loads(z.read("contract-nli/dev.json"))["documents"]
    te = json.loads(z.read("contract-nli/test.json"))["documents"]
    items = ([(d, "train") for d in tr[:100] + tr[200:] + dev] + [(d, "ho") for d in tr[100:160]]
             + [(d, "s160") for d in tr[160:200]] + [(d, "test") for d in te])
    files = {s: open(f"{HERE}/data/{s}.jsonl", "w") for s in ("train", "ho", "s160", "test")}
    docs = {s: {} for s in files}
    stats = {s: [0, 0, 0, 0] for s in files}  # docs, statements, evidence labels, gold answers with no statement found
    with ProcessPoolExecutor(6) as ex:
        for split, did, rows, gold_q, missed in ex.map(doc_rows, items, chunksize=4):
            fold = zlib.crc32(str(did).encode()) % 4
            for r in rows: files[split].write(json.dumps({**r, "fold": fold}) + "\n")
            docs[split][did] = {"fold": fold, "gold": gold_q}
            st = stats[split]; st[0] += 1; st[1] += len(rows); st[2] += sum(l != 2 for r in rows for l in r["labels"]); st[3] += missed
    json.dump({"qids": QIDS, "docs": docs}, open(f"{HERE}/data/docs.json", "w"))
    for s, (n, m, e, miss) in stats.items():
        print(f"{s}: {n} NDAs, {m} statements, {e} evidence labels, {miss} yes/no answers whose evidence matched no statement")
