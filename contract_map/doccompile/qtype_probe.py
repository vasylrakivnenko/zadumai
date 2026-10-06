"""Can a user's question be typed with the document side's clause types? (2026-10-03; no LLM calls)
Rows: gen_ledgar user-style questions of v5, v6, v6b, v7 (all open now) with gold yes/no (the provision states the
answer, so the question is about the provision's LEDGAR type). Two ways:
  (a) zero-shot: the live LEDGAR section tagger (models/contract_map/tagger_ledgar.joblib) applied to the question text
  (b) a question classifier: TF-IDF + LR trained on questions, 5-fold CV grouped by provision (no provision in two folds)
Reports top-1 / top-3 accuracy and accuracy on the share the model is sure of."""
import collections, json, random, zlib
import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

V = "/root/zadumai_nli_proto/extensive/v4"
rows = [r for n in ("v5", "v6", "v6b", "v7") for r in json.load(open(f"{V}/{n}_rows.json"))
        if r["part"] == "gen_ledgar" and r["gold"] in ("yes", "no") and r.get("ledgar_type")]
print(f"{len(rows)} questions over {len({r['src'] for r in rows})} provisions, {len({r['ledgar_type'] for r in rows})} types")
gold = [r["ledgar_type"] for r in rows]


def report(name, P, classes):
    top = np.argsort(-P, 1)
    t1 = np.mean([classes[top[i, 0]] == g for i, g in enumerate(gold)])
    t3 = np.mean([g in {classes[j] for j in top[i, :3]} for i, g in enumerate(gold)])
    conf = P.max(1); out = [f"{name}: top-1 {t1:.1%}, top-3 {t3:.1%}"]
    for c in (0.5, 0.7, 0.9):
        m = conf >= c
        if m.sum(): out.append(f"p>={c}: {np.mean([classes[top[i, 0]] == gold[i] for i in np.where(m)[0]]):.1%} on {m.mean():.0%}")
    print(" | ".join(out))


# (a) the section tagger on the question
d = joblib.load("/root/projects/zadumai/legalbench_map/models/contract_map/tagger_ledgar.joblib")
Z = d["vec"].transform([r["question"] for r in rows]) @ d["W"] + d["b"]
P = 1 / (1 + np.exp(-Z)); labels = d.get("labels", {})
classes = [labels.get(c, c) for c in d["classes"]]
known = set(classes); print(f"(a) gold types the tagger knows: {np.mean([g in known for g in gold]):.0%}")
report("(a) section tagger on the question", np.asarray(P), classes)

# (b) a question classifier, 5-fold CV grouped by provision
fold = [zlib.crc32(r["src"].encode()) % 5 for r in rows]; cls = sorted(set(gold)); P = np.zeros((len(rows), len(cls)))
for f in range(5):
    tr = [i for i in range(len(rows)) if fold[i] != f]; te = [i for i in range(len(rows)) if fold[i] == f]
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit([rows[i]["question"] for i in tr])
    m = LogisticRegression(C=10, max_iter=3000).fit(v.transform([rows[i]["question"] for i in tr]), [gold[i] for i in tr])
    p = m.predict_proba(v.transform([rows[i]["question"] for i in te]))
    for k, i in enumerate(te): P[i, [cls.index(c) for c in m.classes_]] = p[k]
report("(b) question classifier (CV)", P, cls)
top1 = [cls[int(P[i].argmax())] for i in range(len(rows))]
conf = collections.Counter((g, t) for g, t in zip(gold, top1) if g != t)
print("most common confusions:", conf.most_common(8))

# (c) the same, scored with bench.py's MERGE groups (near-synonym LEDGAR headings; written 2026-10-02, not tuned here)
import sys
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/bench")
from bench import MERGE
groups = sorted({MERGE.get(c, c) for c in cls}); G = np.zeros((len(rows), len(groups)))
for j, c in enumerate(cls): G[:, groups.index(MERGE.get(c, c))] += P[:, j]
gold = [MERGE.get(g, g) for g in gold]
report(f"(c) question classifier, {len(groups)} merged types", G, groups)
