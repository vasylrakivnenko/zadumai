"""Light features (IR answer + topic-word hits; nda_extra.one) for the SEALED sets: ContractNLI test (123 NDAs) and the
40 unused training NDAs (s160), for the one-time final check (2026-10-04). No labels are read.
usage: python sealed_feats.py -> feats/nda_extra_sealed.jsonl"""
import json, os, random, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nda_extra
QIDS = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
        "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
if __name__ == "__main__":
    z = zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip")
    tr = json.loads(z.read("contract-nli/train.json"))["documents"]; random.Random(5).shuffle(tr)
    te = json.loads(z.read("contract-nli/test.json"))["documents"]
    items = [(str(d["id"]), d["text"], QIDS) for d in tr[160:200] + te]
    with ProcessPoolExecutor(4) as ex, open(f"{HERE}/feats/nda_extra_sealed.jsonl", "w") as f:
        for rows in ex.map(nda_extra.one, items, chunksize=4):
            for r in rows: f.write(json.dumps(r) + "\n")
    print("done", len(items), "sealed NDAs")
