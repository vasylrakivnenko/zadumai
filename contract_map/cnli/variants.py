"""Graded variants of some queries, each its own source for the selector (select_cv.py --variants) (2026-10-03).
The selector's cross-validated gate keeps only the strictness levels that are >= target right on the training folds."""
import json, re, sys, zipfile
sys.path.insert(0, ".")
from compiler import compile_doc
from queries import NONTECH
from baseline import POLARITY, DEFAULT, ZIP


def nda2_cats(p, k):
    body = " ".join(p.ci_body)
    cats = {m.group(0).lower()[:6] for m in re.finditer(NONTECH, body, re.I)}
    return "no" if body and len(cats) >= k else None


from queries import nda_3_fallback
VARIANTS = {"var_c3": ("nda-2", lambda p: nda2_cats(p, 3)), "var_c4": ("nda-2", lambda p: nda2_cats(p, 4)),
            "var_oral": ("nda-3", lambda p: (nda_3_fallback(p) or (None,))[0])}

if __name__ == "__main__":
    for split in ("train", "dev"):
        rows = []
        for doc in json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{split}.json"))["documents"]:
            p = compile_doc(doc["text"]); ann = doc["annotation_sets"][0]["annotations"]
            for name, (q, fn) in VARIANTS.items():
                a = fn(p); g = ann[q]["choice"]
                rows.append({"doc": doc["id"], "q": q, "source": name, "answer": a, "right": bool(a) and a == POLARITY.get(q, DEFAULT).get(g)})
        json.dump(rows, open(f"runs/variants_{split}.json", "w"))
        for name in VARIANTS:
            rr = [r for r in rows if r["source"] == name and r["answer"]]
            print(split, name, len(rr), "answers,", sum(r["right"] for r in rr), "right")
