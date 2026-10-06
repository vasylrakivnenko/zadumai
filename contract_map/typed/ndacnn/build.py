"""Data for the NDA CNN (2026-10-04; the user: tokens -> very low-dimensional embeddings -> convolutional kernels,
dilated kernels, inter-sentence kernels). Per NDA:
  ids        the reader network's tokens (DeBERTa-v3 sentencepiece; offsets kept), at most MAX_TOK
  st         compiled statements (doccompile) as token ranges [lo, hi)
  ev         per statement x question: 0 says no, 1 says yes, 2 nothing (overlap with ContractNLI's evidence spans,
             mapped to the user-style question as eval_docs.POLARITY: nda-15 flipped)
  y          the NDA's 17 answers: 0 no, 1 yes, 2 not stated
  sec        per statement, its innermost section (the "same section" kernel)
  cl_ids     per statement, its own tokens WITH its lead-ins (the "clause" variant: kernels inside each clause only)
  edges      statement graph: (kind, from, to); kind 0 next statement, 1 cross-reference (both ways), 2 defined term ->
             each statement that uses it
  fold       the stacker's 5 folds (crc32 of the NDA id % 5)
Splits as ndaenc/build.py: train (384), ho (60), s160 / test (sealed; built, not read).
usage: python build.py -> data/{train,ho,s160,test}.pkl"""
import bisect, json, os, pickle, random, re, sys, zipfile, zlib
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/.."); sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/doccompile")
import numpy as np
import eval_docs as E
import doccompile as DC
from transformers import AutoTokenizer

QIDS = list(E.QUESTIONS)
MAX_TOK = 20000
TOK = None


def one(item):
    global TOK
    d, split = item
    if TOK is None: TOK = AutoTokenizer.from_pretrained("/root/projects/zadumai/legalbench_map/models/reader_net")
    text = d["text"]
    enc = TOK(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"][:MAX_TOK]; starts = [a for a, _ in enc["offset_mapping"][:MAX_TOK]]
    ann = d["annotation_sets"][0]["annotations"]
    y = np.full(len(QIDS), 2, dtype=np.int64); spans = {}
    for k, q in enumerate(QIDS):
        ch = ann[q]["choice"]
        if ch == "NotMentioned": continue
        y[k] = 0 if E.POLARITY.get(q, E.DEFAULT)[ch] == "no" else 1
        spans[k] = [tuple(d["spans"][i]) for i in ann[q]["spans"]]
    c = DC.compile_doc(text)
    st, ev, sec, keep, cl_ids = [], [], [], [], []
    for n, s in enumerate(c.stmts):
        if s.start < 0 or len(s.own.strip()) < 4: continue
        lo, hi = bisect.bisect_left(starts, s.start), bisect.bisect_left(starts, s.end)
        if hi <= lo or lo >= len(ids): continue
        lab = [2] * len(QIDS)
        for k, sp in spans.items():
            for a, b in sp:
                ov = min(b, s.end) - max(a, s.start)
                if ov >= 20 or (ov > 0 and ov >= 0.5 * min(b - a, s.end - s.start)):
                    lab[k] = int(y[k]); break
        secs = [i for i in s.secs if not c.sections[i].front]
        st.append((lo, hi)); ev.append(lab); sec.append(secs[-1] if secs else -1); keep.append(n)
        # the clause on its own, with its lead-ins in front (the "clause" variant; the user, 2026-10-04): at most 256 tokens
        cl_ids.append(np.array(TOK(re.sub(r"\s+", " ", s.text).strip(), add_special_tokens=False)["input_ids"][:256], dtype=np.int64))
    idx = {n: i for i, n in enumerate(keep)}
    edges = [(0, i, i + 1) for i in range(len(st) - 1)]
    sstart = [c.stmts[n].start for n in keep]
    for x in c.xrefs:  # the statement holding the reference <-> the statements of the section it names
        if x.target < 0: continue
        i = bisect.bisect_right(sstart, x.start) - 1
        if i < 0: continue
        for j, n in enumerate(keep):
            if x.target in c.stmts[n].secs and j != i: edges += [(1, i, j), (1, j, i)]
    defs = {}
    for j, n in enumerate(keep):
        t = c.stmts[n].text
        for m in list(DC._DEF.finditer(t)) + list(DC._PAREN.finditer(t)):
            term = m.group("t").strip()
            if len(term) >= 3 and term[0].isupper(): defs.setdefault(term, set()).add(j)
    for term, js in defs.items():
        rx = re.compile(r"\b" + re.escape(term) + r"\b")
        users = [i for i, n in enumerate(keep) if i not in js and rx.search(c.stmts[n].own)]
        for j in js:
            edges += [(2, j, i) for i in users]
    return split, {"doc": str(d["id"]), "ids": np.array(ids, dtype=np.int64), "st": np.array(st, dtype=np.int64).reshape(-1, 2),
                   "ev": np.array(ev, dtype=np.int64).reshape(-1, len(QIDS)), "y": y, "sec": np.array(sec, dtype=np.int64),
                   "edges": np.array(sorted(set(edges)), dtype=np.int64).reshape(-1, 3), "fold": zlib.crc32(str(d["id"]).encode()) % 5, "cl_ids": cl_ids}


if __name__ == "__main__":
    z = zipfile.ZipFile(f"{HERE}/../../data/ext/contractnli.zip")
    tr = json.loads(z.read("contract-nli/train.json"))["documents"]; random.Random(5).shuffle(tr)
    dev = json.loads(z.read("contract-nli/dev.json"))["documents"]; te = json.loads(z.read("contract-nli/test.json"))["documents"]
    items = ([(d, "train") for d in tr[:100] + tr[200:] + dev] + [(d, "ho") for d in tr[100:160]]
             + [(d, "s160") for d in tr[160:200]] + [(d, "test") for d in te])
    out = {s: [] for s in ("train", "ho", "s160", "test")}
    with ProcessPoolExecutor(int(os.environ.get("PROCS", "2"))) as ex:
        for split, r in ex.map(one, items, chunksize=4): out[split].append(r)
    for s, rows in out.items():
        pickle.dump(rows, open(f"{HERE}/data/{s}.pkl", "wb"))
        n_tok = [len(r["ids"]) for r in rows]
        print(f"{s}: {len(rows)} NDAs, tokens median {int(np.median(n_tok))} max {max(n_tok)}, statements {sum(len(r['st']) for r in rows)}, "
              f"edges next/xref/def {[int(sum((r['edges'][:, 0] == k).sum() for r in rows)) for k in range(3)]}, "
              f"statement evidence labels {int(sum((r['ev'] != 2).sum() for r in rows))}")
