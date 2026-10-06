"""Training data for idea 4 (2026-10-03): the reader network fine-tuned on LOCATED text, with labels deduced from
existing gold answers and their annotated evidence (no LLM). For each (document, question): the pipeline's located text
(pipeline.Pipeline.locate, 6 pieces + the party line), cut into the network's units; the unit holding a gold evidence
span gets the gold answer (yes / no), the other located units get "doesn't settle it" (2); a question whose gold is
"not stated" gives every located unit 2. Sources (none of them in a tuning or held-out set of this work):
  ContractNLI: train NDAs after the first 200 of random.Random(5) (nda_tune = 0-99, nda_ho = 100-199) + the dev NDAs,
               x the 17 user-style questions (cnli/baseline.py; Entailment -> yes, Contradiction -> no, nda-15 flipped)
  CUAD: cuadc's "write" contracts (not dev: cuad_tune / cuad_ho are dev) x the LegalBench CUAD questions; a unit with
               an annotated span of the category -> yes; no "no" labels (CUAD has none)
  plus a sample of the network's own training data (reader_net/data/v3c_train.jsonl), so it keeps what it knew.
Per question: every evidence unit + at most 2 other located units. Writes data/ft_{train,val}.jsonl (val = 5% of the
documents). usage: build_ft.py [--cuad-docs N] [--keep-old N]"""
import argparse, json, os, random, re, sys, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_docs as E
import pipeline as P

norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()


def key(e):
    e = re.sub(r"^\s*(?:\(?[0-9a-zA-Z]{1,3}[.)]|[•\-–])\s+", "", norm(e))
    return e[:40] if len(e) >= 20 else ""


def located_units(pipe, q, doc):
    cd = pipe.compiled(doc)
    p0 = P.pt0_check(q, doc); fd = (p0.frames or {}).get("frame") or {}
    fr = type("F", (), {"action": fd.get("action"), "actions": fd.get("actions") or []})()
    idx = pipe.locate(q, doc, cd, fr)
    pl = cd.party_line()
    text = (pl + "\n\n" if pl else "") + "\n\n".join(cd.pieces[i]["text"] for i in idx) if idx else ""
    return P.NR.units(text) if text else []


def rows_for(pipe, q, doc, gold, evs, src, rid, rng):
    us = located_units(pipe, q, doc)
    keys = [k for k in (key(e) for e in evs) if k]
    out, other = [], []
    for u in us:
        hit = gold in ("yes", "no") and any(k in norm(u) for k in keys)
        if hit: out.append({"text": u, "question": q, "label": 1 if gold == "yes" else 0, "src": src, "id": rid})
        else: other.append({"text": u, "question": q, "label": 2, "src": src, "id": rid})
    rng.shuffle(other)
    return out + other[:2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cuad-docs", type=int, default=160); ap.add_argument("--keep-old", type=int, default=12000)
    a = ap.parse_args()
    rng = random.Random(11); pipe = P.Pipeline(); out = []
    z = zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip")
    tr = json.loads(z.read("contract-nli/train.json")); docs = tr["documents"]; random.Random(5).shuffle(docs)
    nda_docs = [("train", d) for d in docs[200:]] + [("dev", d) for d in json.loads(z.read("contract-nli/dev.json"))["documents"]]
    lab = tr["labels"]
    for n, (sp, d) in enumerate(nda_docs):
        ann = d["annotation_sets"][0]["annotations"]
        for k, q in E.QUESTIONS.items():
            ch = ann[k]["choice"]; g = "not_stated" if ch == "NotMentioned" else E.POLARITY.get(k, E.DEFAULT)[ch]
            evs = [d["text"][d["spans"][i][0]:d["spans"][i][1]] for i in ann[k]["spans"]]
            for r in rows_for(pipe, q, d["text"], g, evs, "ft_cnli", f"cnli/{d['id']}/{k}", rng): r["doc"] = f"cnli/{d['id']}"; out.append(r)
        if (n + 1) % 50 == 0: print(f"  NDAs {n + 1}/{len(nda_docs)}: {len(out)} rows", flush=True)
    sys.path.insert(0, f"{HERE}/../cuadc")
    from data import write_docs
    qs = json.load(open("/root/zadumai_nli_proto/extensive/questions.json"))
    cat_q = {t[5:].replace("_", " ").replace("-", " ").lower(): q for t, q in qs.items() if t.startswith("cuad_")}
    wd = sorted(write_docs(), key=lambda d: d["id"]); rng.shuffle(wd)
    for n, d in enumerate(wd[:a.cuad_docs]):
        for cat, spans in d["gold"].items():
            k = cat.lower().replace("/", " ").replace("-", " ")
            if k not in cat_q: continue
            g = "yes" if spans else "not_stated"
            for r in rows_for(pipe, cat_q[k], d["text"], g, [s[2] for s in spans], "ft_cuad", f"cuad/{d['id'][:40]}/{k}", rng):
                r["doc"] = f"cuad/{d['id'][:40]}"; out.append(r)
        if (n + 1) % 20 == 0: print(f"  CUAD {n + 1}/{a.cuad_docs}: {len(out)} rows", flush=True)
    old = [json.loads(l) for l in open("/root/zadumai_nli_proto/reader_net/data/v3c_train.jsonl")]
    rng.shuffle(old)
    for r in old[:a.keep_old]: r = dict(r); r["doc"] = "old/" + str(r.get("id")); r.pop("soft", None); out.append(r)
    docs_all = sorted({r["doc"] for r in out}); rng.shuffle(docs_all); val = set(docs_all[:len(docs_all) // 20])
    os.makedirs(f"{HERE}/data", exist_ok=True)
    with open(f"{HERE}/data/ft_train.jsonl", "w") as f, open(f"{HERE}/data/ft_val.jsonl", "w") as g:
        for r in out: (g if r["doc"] in val else f).write(json.dumps(r) + "\n")
    import collections
    print("rows:", len(out), collections.Counter((r["src"] if r["src"].startswith("ft_") else "old", r["label"]) for r in out))


if __name__ == "__main__":
    main()
