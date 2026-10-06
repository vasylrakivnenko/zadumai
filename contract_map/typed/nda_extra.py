"""Extra NDA signals for the stacker (2026-10-04): per (NDA, question) the NDA compiler's answer (cnli/queries_ir.py over
ir.facts: yes / no / not mentioned / none), whether the question's broad topic words appear anywhere in the NDA
(cnli/mentioned.py TOPIC) and how often. For nda_tune and nda_ho. Writes feats/nda_extra.jsonl."""
import json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/../cnli"); sys.path.insert(0, HERE)


def one(item):
    sys.path.insert(0, f"{HERE}/../cnli")
    import queries_ir as QI
    from ir import facts
    from mentioned import TOPIC
    doc_id, text, qids = item
    try:
        f = facts(text)
    except Exception as e:  # noqa: BLE001
        f = None
    out = []
    for q in qids:
        a = None
        if f is not None:
            try:
                r = QI.QUERIES[q](f); a = r[0] if r else None
            except Exception:  # noqa: BLE001
                a = "error"
        hits = len(re.findall(TOPIC[q], text, re.I)) if q in TOPIC else -1
        out.append({"id": f"{doc_id}/{q}", "ir": a, "topic_hits": hits})
    return out


if __name__ == "__main__":
    import extract as X
    items = {}
    for s in ("nda_tune", "nda_ho"):
        for r in X.SETS[s]():
            d, q = r["id"].split("/")
            items.setdefault(d, [r["doc"], []])[1].append(q)
    work = [(d, t, qs) for d, (t, qs) in items.items()]
    with ProcessPoolExecutor(4) as ex, open(f"{HERE}/feats/nda_extra.jsonl", "w") as f:
        for rows in ex.map(one, work, chunksize=2):
            for r in rows: f.write(json.dumps(r) + "\n")
    print("done", len(work), "NDAs")
