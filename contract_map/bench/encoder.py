"""Does a fine-tuned small encoder get more out of combined data than TF-IDF? (2026-10-02)
MiniLM-L6 (paraphrase-MiniLM-L6-v2, 22M params) + mean pooling + one linear output per class, trained with a
masked binary loss: a row teaches class c only if its dataset marks c completely or the row is a positive for c
(combined.py's label map). Modes: `separate` (one model per dataset) and `combined` (one model on all six).
Scoring as combined.py: each dataset on its own test rows; LEDGAR by argmax (near-synonyms merged); the others
with a threshold picked on their own validation rows.
usage: encoder.py --stage 1|2 --mode separate|combined [--threads 6]"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench, combined
from combined import DATASETS, UNIFIED, score_class, row_classes

MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
STAGE = {1: dict(train=3000, eval=500, val=500, epochs=1), 2: dict(train=10000, eval=2000, val=1000, epochs=2)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.enc = AutoModel.from_pretrained(MODEL); self.head = torch.nn.Linear(self.enc.config.hidden_size, n)
    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float(); return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))


def batches(lengths, bs, shuffle, seed=0):
    """Length-bucketed batches (less padding); batch order shuffled when training."""
    idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle: random.Random(seed).shuffle(out)
    return out


def encode(tok, texts, max_len=256):
    e = tok(texts, truncation=True, max_length=max_len)
    return e["input_ids"]


def pad(seqs):
    L = max(len(s) for s in seqs); ids = torch.zeros(len(seqs), L, dtype=torch.long); att = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    return ids, att


def train_model(datasets, S, tr, tok, epochs, tag):
    full = {ds: {score_class(ds, l) for l in S[ds]["labels"]} for ds in datasets}
    rows = [(ds, r) for ds in datasets for r in tr[ds]]
    pos = [row_classes(ds, r["labels"]) for ds, r in rows]
    classes = sorted(set().union(*full.values()) | set().union(*pos)); ci = {c: i for i, c in enumerate(classes)}
    Y = torch.zeros(len(rows), len(classes)); M = torch.zeros(len(rows), len(classes))
    for i, (ds, _) in enumerate(rows):
        for c in full[ds]: M[i, ci[c]] = 1
        for c in pos[i]: Y[i, ci[c]] = 1; M[i, ci[c]] = 1
    npos = (Y * M).sum(0); nneg = ((1 - Y) * M).sum(0)
    pw = torch.sqrt(nneg / npos.clamp(min=1)).clamp(1, 30)
    ids = encode(tok, [r["text"] for _, r in rows])
    net = Net(len(classes))
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 5e-5}, {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    bl = [b for e in range(epochs) for b in batches([len(x) for x in ids], 32, True, seed=e)]
    warm = max(1, len(bl) // 20); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.05, 1 - s / len(bl)))
    net.train(); t0 = time.time(); run = 0
    for step, b in enumerate(bl):
        x, att = pad([ids[i] for i in b]); y, m = Y[b], M[b]
        loss = (lossf(net(x, att), y) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0 or step == len(bl) - 1:
            print(f"   [{tag}] step {step + 1}/{len(bl)} loss {run:.4f}  {(step + 1) * 32 / (time.time() - t0):.0f} rows/s", flush=True)
    net.eval(); return net, classes


@torch.no_grad()
def scores(net, classes, tok, rows, cls):
    ids = encode(tok, [r["text"] for r in rows]); out = np.zeros((len(rows), len(cls)))
    ci = {c: i for i, c in enumerate(classes)}; cols = [ci.get(c) for c in cls]
    for b in batches([len(x) for x in ids], 64, False):
        p = torch.sigmoid(net(*pad([ids[i] for i in b]))).numpy()
        for j, k in enumerate(cols):
            if k is not None: out[b, j] = p[:, k]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=1); ap.add_argument("--mode", default="combined")
    ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    st = STAGE[a.stage]; tok = AutoTokenizer.from_pretrained(MODEL)
    S = {ds: bench.SETS[ds]() for ds in DATASETS}
    tr = {ds: bench.sample(S[ds]["train"], st["train"], seed=1, multi=S[ds]["multi"]) for ds in DATASETS}
    ev = {ds: bench.sample(S[ds]["test"], st["eval"], multi=S[ds]["multi"]) for ds in DATASETS}
    va = {ds: bench.sample(S[ds]["val"], st["val"], multi=True) for ds in DATASETS}
    print(f"== encoder {a.mode}, stage {a.stage}: train " + ", ".join(f"{d} {len(tr[d])}" for d in DATASETS) + f", {st['epochs']} epoch(s)", flush=True)
    groups = [[d] for d in DATASETS] if a.mode == "separate" else [DATASETS]
    res = {}
    for g in groups:
        net, classes = train_model(g, S, tr, tok, st["epochs"], "+".join(g))
        for ds in g:
            cls = sorted({score_class(ds, l) for l in S[ds]["labels"]})
            Sc = scores(net, classes, tok, ev[ds], cls)
            vp = (cls, scores(net, classes, tok, va[ds], cls), va[ds]) if va[ds] else None
            res[ds] = combined.evaluate(ds, Sc, cls, ev[ds], vp)
            r = res[ds]
            print(f"   {ds:12} " + (f"acc {100 * r['acc']:.1f}%" if "acc" in r else f"F1 {r['f1']:.3f} (P {100 * r['p']:.0f} R {100 * r['r']:.0f})"), flush=True)
    json.dump({d: {k: v for k, v in r.items() if k != "per"} for d, r in res.items()}, open(f"{OUT}/encoder_s{a.stage}_{a.mode}.json", "w"))


if __name__ == "__main__":
    main()
