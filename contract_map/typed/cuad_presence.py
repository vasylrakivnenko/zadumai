"""A presence signal for CUAD checklist questions (2026-10-04): TF-IDF + one-vs-rest LR section tagger trained on CUAD's
training contracts EXCEPT cuadc's 82 dev contracts (cuad_tune and cuad_ho come from those) and CUAD's test contracts;
for each (contract, category) the highest section probability and how many sections score >= 0.5.
Writes feats/cuad_presence.jsonl keyed like the feature rows ("<doc[:40]>/<category key>")."""
import collections, json, re, sys
import numpy as np
sys.path.insert(0, "/root/zadumai_nli_proto/contract_map/cuadc")
from data import split, test_docs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

META = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
key = lambda cat: cat.lower().replace("/", " ").replace("-", " ")


def paras(text):
    out, pos = [], 0
    for block in text.split("\n\n"):
        s, e = pos, pos + len(block); pos = e + 2
        t = re.sub(r"\s+", " ", block).strip()
        if len(t) >= 40: out.append((s, e, t[:4000]))
    return out


tune = split(); dev = [d for d in tune if d["dev"]]; train_docs = [d for d in tune if not d["dev"]]
cats = sorted({c for d in tune for c in d["gold"] if c not in META})
X_text, Y = [], []
for d in train_docs:
    for s, e, t in paras(d["text"]):
        X_text.append(t); Y.append({c for c in cats if any(a < e and b > s for a, b, _ in d["gold"].get(c, []))})
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True).fit(X_text)
X = vec.transform(X_text)
def fit(c):
    y = np.array([c in ys for ys in Y], dtype=int)
    return c, (LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(X, y) if y.min() != y.max() else None)
models = dict(Parallel(n_jobs=4)(delayed(fit)(c) for c in cats))
print(f"tagger: {len(train_docs)} training contracts, {len(X_text)} sections, {len(cats)} categories", flush=True)
with open("/root/zadumai_nli_proto/contract_map/typed/feats/cuad_presence.jsonl", "w") as f:
    for d in dev:
        ps = paras(d["text"]); Xd = vec.transform([t for _, _, t in ps]) if ps else None
        for c in cats:
            m = models.get(c)
            p = m.predict_proba(Xd)[:, 1] if (m is not None and Xd is not None) else np.zeros(1)
            f.write(json.dumps({"id": f"{d['id'][:40]}/{key(c)}", "pmax": float(p.max()), "n50": int((p >= 0.5).sum())}) + "\n")
print("done")
