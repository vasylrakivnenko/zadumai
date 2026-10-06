"""The NDA statement encoder (2026-10-04): DeBERTa-v3-xsmall, every weight trained (word embeddings too), from the
router's reader network (models/reader_net: its encoder already reads contracts), with a new head of 17 x 3 logits,
one (no, yes, says nothing) per NDA question, on one compiled statement (data/*.jsonl from build.py). The document
itself is decided by rules over its statements (aggregate.py), not here.
Per epoch: every statement with a yes / no label + a fresh random NONE_KEEP of the statements that say nothing on
any question (79% of them). Loss: cross-entropy per question, yes / no weighted CLASS_W.
  --fold F   train on the training NDAs of the other folds, write predictions for fold F (cross-fitting)
  --fold -1  train on all of them, write predictions for --predict splits (ho; the sealed s160 / test only when asked)
usage: train.py --fold F [--epochs E] [--lr LR] [--threads T] [--predict ho,...]   -> runs/<name>/, preds/<name>_<split>.jsonl"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/root/projects/zadumai/legalbench_map/models/reader_net"
NQ = 17
ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True); ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=4e-5); ap.add_argument("--head-lr", type=float, default=5e-4)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--none-keep", type=float, default=0.4)
ap.add_argument("--class-w", type=float, default=3.0); ap.add_argument("--smooth", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--predict", default="ho")
ap.add_argument("--name", default="")
a = ap.parse_args()
name = a.name or (f"f{a.fold}" if a.fold >= 0 else "full")
out_dir = f"{HERE}/runs/{name}"; os.makedirs(out_dir, exist_ok=True); os.makedirs(f"{HERE}/preds", exist_ok=True)
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); random.seed(a.seed)
log = open(f"{out_dir}/train_log.txt", "a")


def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()


tok = AutoTokenizer.from_pretrained(BASE)
old = AutoModelForSequenceClassification.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3 * NQ, ignore_mismatched_sizes=True)
with torch.no_grad():  # each question's head starts as the reader network's own (no, yes, doesn't settle it)
    model.classifier.weight.copy_(old.classifier.weight.repeat(NQ, 1)); model.classifier.bias.copy_(old.classifier.bias.repeat(NQ))
del old


def load(split):
    rows = [json.loads(l) for l in open(f"{HERE}/data/{split}.jsonl")]
    enc = tok([r["text"] for r in rows], truncation=True, max_length=a.max_len)["input_ids"]
    for r, i in zip(rows, enc): r["ids"] = i
    return rows


allrows = load("train")
train = [r for r in allrows if r["fold"] != a.fold]
pos = [r for r in train if any(l != 2 for l in r["labels"])]; none = [r for r in train if all(l == 2 for l in r["labels"])]
per_epoch = len(pos) + int(a.none_keep * len(none))
total = int(math.ceil(per_epoch / a.bs) * a.epochs)
say(f"{time.strftime('%H:%M:%S')} {name}: train {len(train)} statements ({len(pos)} with a label), {per_epoch}/epoch, "
    f"{total} steps; lr {a.lr} head {a.head_lr} epochs {a.epochs} class_w {a.class_w}")


def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle: random.shuffle(idx)
    out = []
    for i in range(0, len(idx), bs * 50):
        part = sorted(idx[i:i + bs * 50], key=lambda j: len(rows[j]["ids"]))
        out += [part[k:k + bs] for k in range(0, len(part), bs)]
    if shuffle: random.shuffle(out)
    return out


def collate(rows, ids):
    enc = tok.pad({"input_ids": [rows[j]["ids"] for j in ids]}, return_tensors="pt")
    return enc, torch.tensor([rows[j]["labels"] for j in ids])


head = [p for n, p in model.named_parameters() if n.startswith(("classifier", "pooler"))]
body = [p for n, p in model.named_parameters() if not n.startswith(("classifier", "pooler"))]
opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([a.class_w, a.class_w, 1.0]), label_smoothing=a.smooth)
model.train(); step, t0, run = 0, time.time(), 0.0
while step < total:
    epoch_rows = pos + random.sample(none, int(a.none_keep * len(none)))
    for ids in batches(epoch_rows, a.bs, True):
        if step >= total: break
        enc, y = collate(epoch_rows, ids)
        logits = model(**enc).logits.view(-1, NQ, 3)
        loss = lossf(logits.reshape(-1, 3), y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); step += 1; run += loss.item()
        if step % 25 == 0 and os.path.exists(f"{out_dir}/threads"):  # the scheduler (sched.py) shares the 8 cores
            try:
                n = int(open(f"{out_dir}/threads").read())
                if n != torch.get_num_threads(): torch.set_num_threads(n)
            except ValueError: pass
        if step % 100 == 0:
            el = time.time() - t0
            say(f"step {step}/{total} loss {run / 100:.4f} {el / step:.2f}s/step eta {(total - step) * el / step / 60:.0f} min"); run = 0.0
model.save_pretrained(out_dir); tok.save_pretrained(out_dir); json.dump(vars(a), open(f"{out_dir}/train_args.json", "w"))


@torch.inference_mode()
def predict(rows, path):
    model.eval()
    if os.path.exists(f"{out_dir}/threads"): torch.set_num_threads(int(open(f"{out_dir}/threads").read()))
    with open(path, "w") as f:
        for ids in batches(rows, 64, False):
            enc, _ = collate(rows, ids)
            p = torch.softmax(model(**enc).logits.view(-1, NQ, 3), -1)
            for j, pj in zip(ids, p.tolist()):
                f.write(json.dumps({"doc": rows[j]["doc"], "sid": rows[j]["sid"], "p": [[round(x, 4) for x in q] for q in pj]}) + "\n")


if a.fold >= 0:
    predict([r for r in allrows if r["fold"] == a.fold], f"{HERE}/preds/{name}_train.jsonl")
else:
    for s in a.predict.split(","):
        predict(load(s), f"{HERE}/preds/{name}_{s}.jsonl")
say(f"{time.strftime('%H:%M:%S')} done")
