"""Heading probe, old front end (cuadc/front.py) vs doccompile: rules 'heading word -> CUAD type' mined on write (326),
scored on dev (82); test sealed. Also the namespace view (a type's top-3 heading words: recall, share of text)."""
import collections, re, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/cuadc")
from data import split
from front import compile_contract
from doccompile import compile_doc

META = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
STOP = {"of", "and", "or", "to", "the", "for", "in", "on", "by", "a", "an", "with", "upon", "section", "article", "agreement", "this", "s"}


def words(h):
    h = re.sub(r"^(section|article)\s+[\w.]+", " ", h.lower())
    return frozenset(w[:-1] if w.endswith("s") and len(w) > 4 else w for w in re.findall(r"[a-z]+", h) if w not in STOP)


def prep(args):
    d, mode = args
    heads = collections.defaultdict(list); allsp = []
    if mode == "old":
        for s in compile_contract(d["text"]):
            if s.start < 0: continue
            sp = (s.pstart, s.pend) if s.pstart >= 0 else (s.start, s.end); allsp.append(sp)
            for w in words(s.head) if s.head else (): heads[w].append(sp)
    else:
        c = compile_doc(d["text"])
        for s in c.stmts:
            if s.start < 0: continue
            sp = (s.pstart, s.pend) if s.pstart >= 0 else (s.start, s.end); allsp.append(sp)
            hs = c.headings(s)
            ws = set().union(*(words(h) for h in (hs[-1:] if mode == "near" else hs))) if hs else set()
            for w in ws: heads[w].append(sp)
    gold = {t: [(a, b) for a, b, _ in v] for t, v in d["gold"].items() if t not in META}
    return d["dev"], dict(heads), gold, allsp


def hit(spans, gold):
    return any(a < ge and gs < b for a, b in spans for gs, ge in gold)


def cover(sp):
    return len({i for a, b in sp for i in range(a // 50, b // 50 + 1)})


if __name__ == "__main__":
    docs = split()
    for mode in ("old", "near", "chain"):
        with ProcessPoolExecutor(8) as ex: rows = list(ex.map(prep, [(d, mode) for d in docs], chunksize=4))
        write = [r for r in rows if not r[0]]; dev = [r for r in rows if r[0]]; types = sorted(write[0][2])
        n = collections.Counter(); k = collections.Counter()
        for _, heads, gold, _ in write:
            for w, sp in heads.items():
                n[w] += 1
                for t in types:
                    if gold[t] and hit(sp, gold[t]): k[w, t] += 1
        rules = collections.defaultdict(set)
        for (w, t), c in k.items():
            if n[w] >= 5 and c / n[w] >= 0.8: rules[t].add(w)
        P = R = PR = 0; nsr = nsp = 0; share = []
        for t in types:
            ns = [w for (w2, tt), c in sorted(k.items(), key=lambda kv: -kv[1]) if tt == t for w in [w2]][:3]
            for _, heads, gold, allsp in dev:
                present = bool(gold[t]); P += present
                sp = [x for w in rules[t] if w in heads for x in heads[w]]
                if sp: PR += 1; R += present and hit(sp, gold[t])
                nsp_ = [x for w in ns if w in heads for x in heads[w]]
                if present: nsp += 1; nsr += hit(nsp_, gold[t])
                if present: share.append(cover(nsp_) / max(1, cover(allsp)))
        print(f"{mode:6s} rules: found {R}/{P} = {R/P:.1%} of dev clauses, precision {R/max(1,PR):.1%} ({PR} predicted) | "
              f"namespace: recall {nsr/nsp:.1%}, text share {sum(share)/len(share):.1%}")
