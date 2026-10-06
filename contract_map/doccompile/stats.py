"""Coverage stats of doccompile over CUAD's 408 tuning contracts (test stays sealed) and ContractNLI train+dev NDAs."""
import collections, json, statistics, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/cuadc")
from doccompile import compile_doc


def one(item):
    did, text = item
    t = time.time()
    try:
        c = compile_doc(text)
    except Exception as e:  # noqa: BLE001
        return {"id": did, "error": repr(e)}
    ms = (time.time() - t) * 1000
    located = [s for s in c.stmts if s.start >= 0]
    headed = [s for s in located if c.headings(s)]
    cov = sum(s.end - s.start for s in headed) / max(1, sum(s.end - s.start for s in located))
    xs = collections.Counter((x.kind, x.resolved) for x in c.xrefs)
    return {"id": did, "ms": ms, "chars": len(text), "stmts": len(c.stmts), "located": len(located) / max(1, len(c.stmts)),
            "headed_chars": cov, "sections": len(c.sections), "numbered": sum(bool(s.number) for s in c.sections),
            "parties": len(c.parties), "roles": c.roles, "furniture": collections.Counter(w for *_, w in c.furniture),
            "symbols": len(c.symbols), "xrefs": {f"{k}:{r}": n for (k, r), n in xs.items()}, "title": c.title}


def cnli_docs():
    z = zipfile.ZipFile("/root/zadumai_nli_proto/contract_map/data/ext/contractnli.zip")
    out = []
    for split in ("train", "dev"):
        for d in json.loads(z.read(f"contract-nli/{split}.json"))["documents"]: out.append((f"cnli/{d['id']}", d["text"]))
    return out


if __name__ == "__main__":
    from data import split
    sets = {"cuad": [(d["id"], d["text"]) for d in split()], "cnli": cnli_docs()}
    for name, items in sets.items():
        with ProcessPoolExecutor(8) as ex: rows = list(ex.map(one, items, chunksize=4))
        err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
        q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
        print(f"\n== {name}: {len(rows)} docs, errors {len(err)} {err[:2]}")
        print(f"ms median {q([r['ms'] for r in rows], .5):.0f}, p95 {q([r['ms'] for r in rows], .95):.0f}, max {max(r['ms'] for r in rows):.0f}")
        print(f"statements located in the original: median {q([r['located'] for r in rows], .5):.1%}, p5 {q([r['located'] for r in rows], .05):.1%}")
        print(f"text under a heading: median {q([r['headed_chars'] for r in rows], .5):.0%}, p25 {q([r['headed_chars'] for r in rows], .25):.0%}, "
              f"docs < 50%: {sum(r['headed_chars'] < .5 for r in rows)}")
        print(f"sections median {q([r['sections'] for r in rows], .5)}, numbered median {q([r['numbered'] for r in rows], .5)}")
        print(f"docs with >= 2 parties: {sum(r['parties'] >= 2 for r in rows)}, >= 1 role: {sum(bool(r['roles']) for r in rows)}, title: {sum(bool(r['title']) for r in rows)}")
        f = collections.Counter(); x = collections.Counter()
        for r in rows: f.update({k: 1 for k in r["furniture"]}); x.update(r["xrefs"])
        print("docs with furniture:", dict(f)); print("xrefs:", dict(x))
        print("roles:", collections.Counter(ro for r in rows for ro in r["roles"]).most_common(25))
