"""CUAD for the clause compiler (2026-10-03; the user: "port the compiler to a few high-value CUAD clause types").
Splits, fixed before any rule is written:
  test  = CUAD's own 102 test contracts (data/test.json) - SEALED: never read, one run at the end (final.py)
  tune  = the other 408: "write" (326; rules are written from these) + "dev" (82, random seed 0; never read, scored only)
Nested 5-fold CV over all 408 (folds: random.Random(1) shuffle, r % 5).
Gold per (contract, type): the list of annotated spans (start, end, text); empty = the clause is absent."""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f"{HERE}/../data/CUADv1.json"; TEST = f"{HERE}/../data/test.json"
TYPES = ["Governing Law", "Anti-Assignment", "Cap On Liability", "Termination For Convenience", "Renewal Term", "Change Of Control"]


def _docs(path):
    out = []
    for d in json.load(open(path))["data"]:
        p = d["paragraphs"][0]; gold = {}
        for qa in p["qas"]:
            t = qa["id"].split("__")[-1]
            spans = sorted({(a["answer_start"], a["answer_start"] + len(a["text"]), a["text"]) for a in qa["answers"]})
            gold[t] = [list(s) for s in spans]
        out.append({"id": d["title"], "text": p["context"], "gold": gold})
    return out


def split():
    test_ids = {d["title"] for d in json.load(open(TEST))["data"]}
    tune = [d for d in _docs(SRC) if d["id"] not in test_ids]
    ids = sorted(d["id"] for d in tune); random.Random(0).shuffle(ids); dev = set(ids[:82])
    for d in tune: d["dev"] = d["id"] in dev
    order = list(range(len(tune))); random.Random(1).shuffle(order)
    for r, i in enumerate(order): tune[i]["fold"] = r % 5
    return tune


def write_docs():
    return [d for d in split() if not d["dev"]]


def test_docs():
    """Only final.py may call this."""
    return _docs(TEST)


def question(t):
    d = json.load(open(TEST))["data"][0]["paragraphs"][0]["qas"]
    return next(q["question"] for q in d if q["id"].endswith("__" + t))
