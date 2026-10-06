"""The locate step alone (2026-10-03; the user: "don't forget about lemmatization or stemming if needed"): how often is
a gold evidence span inside the top-K located pieces? Rows: NDA tuning sample (eval_docs.nda("train", 100)), gold
yes/no (they have evidence spans). Word normalizers for the lexical rank (each with its own lexicon-synonym map):
  pref5   NetReader's: the first 5 letters (merges assign/assignment, also compete/compensation, confirm/confidential)
  lemma   spaCy en_core_web_sm lemmas
  lemd    lemma + derivational suffixes off (assignment -> assign, termination -> terminat, confidentiality -> confidential)
Rankers: lexical alone, embedder alone (bge-small, as NetReader), fused by rank (pipeline.locate without the concept bonus).
usage: locate_eval.py [--set nda_train]"""
import argparse, collections, functools, re, sys
import numpy as np
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/typed")
import eval_docs as E
import pipeline as P
NR = P.NR

STOP = NR._STOP
_DERIV = ("ization", "isation", "ations", "ation", "ments", "ment", "ities", "ity", "ances", "ance", "ences", "ence",
          "ables", "able", "ibles", "ible", "ions", "ion", "ives", "ive", "als", "al", "ness")


@functools.lru_cache(maxsize=1)
def nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmas(text):
    return [t.lemma_.lower() for t in nlp()(text.lower()[:20000]) if t.is_alpha]


def deriv(w):
    for s in _DERIV:
        if w.endswith(s) and len(w) - len(s) >= 5:
            return w[:-len(s)]
    return w


NORMS = {
    "lem+p5": lambda text: {w for w in lemmas(text) if w not in STOP and len(w) > 2} | {"~" + x for x in NR._stems(text)},
    "pref5": lambda text: NR._stems(text),
    "lemma": lambda text: {w for w in lemmas(text) if w not in STOP and len(w) > 2},
    "lemd": lambda text: {deriv(w) for w in lemmas(text) if w not in STOP and len(w) > 2},
}


def syn_map(norm):
    out = {}
    for phrases in NR.LEXICON.values():
        group = set().union(*(norm(p) for p in phrases)) if phrases else set()
        for p in phrases:
            for s in norm(p): out.setdefault(s, set()).update(group)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="nda_train"); a = ap.parse_args()
    rows = [r for r in (E.nda("train", 100) if a.set == "nda_train" else E.nda("dev")) if r["gold"] != "not_stated"]
    pipe = P.Pipeline()
    norm_text = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    def key(e):  # 40 characters from inside the span: its list marker and its last words may be cut differently
        e = re.sub(r"^\s*(?:\(?[0-9a-zA-Z]{1,3}[.)]|[•\-–])\s+", "", norm_text(e))
        return e[:40] if len(e) >= 20 else ""
    by_doc = collections.defaultdict(list)
    for r in rows: by_doc[r["doc"]].append(r)
    syns = {k: syn_map(f) for k, f in NORMS.items()}
    ranks = collections.defaultdict(list)  # (norm, ranker) -> rank of the first gold piece (or None)
    for doc, rs in by_doc.items():
        cd = pipe.compiled(doc); texts = [p["text"] for p in cd.pieces]
        if not texts: continue
        vecs = pipe.net._embed(texts)
        piece_norm = {k: [f(t) for t in texts] for k, f in NORMS.items()}
        for r in rs:
            gold = {i for i, t in enumerate(texts) for e in r["ev"] if key(e) and key(e) in norm_text(t)}
            if not gold: ranks["(evidence not in any piece)", ""].append(None); continue
            qv = pipe.net._embed([NR.QUERY_PREFIX + r["question"]])[0]
            emb = [int(i) for i in (-(vecs @ qv)).argsort()]
            for k, f in NORMS.items():
                q = f(r["question"]); qx = set().union(q, *(syns[k].get(s, ()) for s in q))
                lex = sorted(range(len(texts)), key=lambda i: -(2 * len(q & piece_norm[k][i]) + len(qx & piece_norm[k][i])))
                fused = {i: 0.0 for i in range(len(texts))}
                for order in (lex, emb):
                    for rk, i in enumerate(order): fused[i] += 1 / (10 + rk)
                fz = sorted(fused, key=lambda i: -fused[i])
                for name, order in (("lex", lex), ("fused", fz)):
                    ranks[k, name].append(min(order.index(i) for i in gold))
            ranks["-", "emb"].append(min(emb.index(i) for i in gold))
    n = len(rows)
    print(f"{n} gold yes/no rows; evidence not inside any compiled piece: {len(ranks['(evidence not in any piece)', ''])}")
    for key, rk in sorted(ranks.items()):
        if key[0].startswith("("): continue
        print(f"{key[0]:6s} {key[1]:6s} " + "  ".join(f"top{K}: {sum(x is not None and x < K for x in rk) / n:.1%}" for K in (2, 4, 6, 8, 10)))


if __name__ == "__main__":
    main()
