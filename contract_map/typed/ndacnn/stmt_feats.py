"""Extra per-clause vectors for the CNN (2026-10-04; the user: stack token -> sentence -> clause views; TF-IDF per clause
with three moderate thresholds). For every compiled statement (its own words with its lead-ins: the "clause" tokens
of build.py, decoded) of every NDA:
  bge       bge-small-en-v1.5 sentence vector (384-d, normalized): contextual meaning, negation and scope included
  tfidf1-3  TF-IDF (word 1-2-grams, sublinear tf) -> TruncatedSVD 256-d, L2-normalized, at three document-frequency
            thresholds: 1 = min_df 2 / max_df 0.9, 2 = min_df 5 / max_df 0.7, 3 = min_df 20 / max_df 0.5;
            fitted on the training + SEC clauses only (no labels; never on held-out or sealed NDAs)
usage: python stmt_feats.py -> data/xf_<name>.pkl  {split: {doc: float16 [statements, dim]}}"""
import os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = {"train": f"{HERE}/data/train.pkl", "ho": f"{HERE}/data/ho.pkl", "test": f"{HERE}/data/test.pkl",
          "s160": f"{HERE}/data/s160.pkl", "new": f"{HERE}/../ndamore/data/new.pkl"}
TFIDF = {"tfidf1": (2, 0.9), "tfidf2": (5, 0.7), "tfidf3": (20, 0.5)}

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/projects/zadumai/legalbench_map/models/reader_net")
    texts = {}
    for split, p in SPLITS.items():
        texts[split] = {d["doc"]: [tok.decode(c.tolist(), skip_special_tokens=True).strip() or "." for c in d["cl_ids"]]
                        for d in pickle.load(open(p, "rb"))}
        print(split, len(texts[split]), "NDAs,", sum(len(v) for v in texts[split].values()), "clauses", flush=True)
    # TF-IDF x 3 thresholds, fitted on training + SEC clauses
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    fit_texts = [t for s in ("train", "new") for v in texts[s].values() for t in v]
    for name, (mn, mx) in TFIDF.items():
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=mn, max_df=mx, sublinear_tf=True, lowercase=True)
        svd = TruncatedSVD(256, random_state=0).fit(vec.fit_transform(fit_texts))
        out = {}
        for split, docs in texts.items():
            out[split] = {}
            for doc, ts in docs.items():
                z = svd.transform(vec.transform(ts)); z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-8)
                out[split][doc] = z.astype(np.float16)
        pickle.dump(out, open(f"{HERE}/data/xf_{name}.pkl", "wb"))
        print(f"{name}: vocabulary {len(vec.vocabulary_):,} terms, SVD keeps {svd.explained_variance_ratio_.sum():.1%}, {time.time() - t0:.0f}s", flush=True)
    # bge-small sentence vectors
    import torch
    torch.set_num_threads(8)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); t0 = time.time(); out = {}
    for split, docs in texts.items():
        out[split] = {}
        for doc, ts in docs.items():
            out[split][doc] = m.encode(ts, batch_size=64, normalize_embeddings=True).astype(np.float16)
        print(f"bge {split} done {time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(f"{HERE}/data/xf_bge.pkl", "wb"))
    print("done")
