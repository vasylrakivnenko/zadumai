"""The NDA CNN (2026-10-04; the user's idea: very low-dimensional token embeddings + convolutional kernels, dilated kernels
in several orders, inter-sentence kernels). One model reads the whole NDA and answers the 17 questions:
  tokens -> 64-d vectors (the reader network's word embeddings, PCA to 64; trained further) -> kernels over the whole
  token stream (VARIANTS below) -> each compiled statement = max + mean of its tokens -> [graph: 2 rounds of kernels over
  the statement graph: next statement, same section, cross-reference, defined term -> its users] -> per question an
  attention over the statements (1 slot, or 2: a rule and an exception) -> yes / no / not stated.
Supervision: the 17 answers + the attention on ContractNLI's evidence statements + each statement's own label per question
(dense, 7k evidence labels).
Kernel stacks (9 taps each):  plain    1 layer, dilation 1
                              parallel one layer, dilations 1, 2, 3 ... 16 side by side (the user's first idea)
                              rising   1, 2, 4 ... 64 stacked;  falling 64 ... 1;  updown 1 ... 64 ... 1
                              rising2  the rising cycle twice (14 layers)
                              plain3   3 layers, kernels 13 -> 9 -> 5;  pyramid  DPCNN: conv, then 6 x [pool / 2 -> conv]
                              clause   kernels inside each clause (with its lead-ins) only, dilations 1..16, then a
                                       3-wide kernel across neighbouring clauses (--across N: N layers, dilations 1, 2, 4..)
                              clausetok  kernels inside each clause (1, 2, 4), then the clauses' tokens end to end and
                                       kernels over the whole NDA (8 .. 64);  --sep: clause starts marked
teachers: --jev head|mix (Jev's answers on the training NDAs), --regex W (the detectors' flags); run time stays local.
usage: train.py --variant V [--graph] [--slots 2] [--jev head|mix] [--regex W] --fold F (0-4: out-of-fold for the training NDAs; -1: all of them ->
       held-out)  -> preds/<name>_f<F>.jsonl {id, p: [p_no, p_yes, p_not_stated]}"""
import argparse, json, math, os, pickle, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
NQ, DIM = 17, 64
QIDS = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
        "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]  # = eval_docs.QUESTIONS order (build.py)
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["plain", "parallel", "rising", "falling", "updown", "plain3", "pyramid", "clause", "clausetok", "rising2"])
ap.add_argument("--across", type=int, default=1)  # clause variants: layers of kernels over the sequence of clause vectors
ap.add_argument("--sep", action="store_true")  # mark where each clause starts (a learned vector added to its first token)
ap.add_argument("--graph", action="store_true"); ap.add_argument("--slots", type=int, default=1)
ap.add_argument("--fold", type=int, required=True); ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--ch", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--att-w", type=float, default=0.5); ap.add_argument("--st-w", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--accum", type=int, default=4); ap.add_argument("--max-docs", type=int, default=0)
# teachers (2026-10-04, the user: distill Jev, add signals like our regex): Jev's 3-way answers on the TRAINING NDAs only
ap.add_argument("--jev", default="none", choices=["none", "head", "mix"])  # head: an extra head learns Jev; mix: (1-a) gold + a Jev
ap.add_argument("--jev-w", type=float, default=0.5); ap.add_argument("--alpha", type=float, default=0.3)
# student (step 2, 2026-10-04): extra NDAs from SEC with the teacher's soft labels (ndamore/label.py; the fold's own teacher)
ap.add_argument("--extra", default="")  # ndamore/data dir with new.pkl + soft.jsonl
ap.add_argument("--stmt-feats", default="")  # extra per-clause vectors (stmt_feats.py): bge, tfidf1, tfidf2, tfidf3 (comma list)
ap.add_argument("--across-st", type=int, default=0)
ap.add_argument("--doc-vec", default="")  # a whole-NDA vector (the mean of one clause-vector set, e.g. bgel) joined to every question's decision  # kernels (3 wide) across neighbouring clauses, after the clause vectors
ap.add_argument("--sealed", action="store_true")  # fold -1: also write predictions for the sealed sets (test, s160); no score printed
ap.add_argument("--tag", default="")  # appended to the run name (e.g. r2: the second round of SEC NDAs)
ap.add_argument("--extra-w", type=float, default=1.0); ap.add_argument("--isnda-min", type=float, default=0.5)
ap.add_argument("--extra-n", type=int, default=0)  # use only the first N extra NDAs (learning curve); 0 = all
ap.add_argument("--save", action="store_true")  # keep the weights: models/<name>_f<F>.pt (the teacher for new NDAs)
ap.add_argument("--device", default="cpu")  # "cuda": every tensor on the GPU (the user's 4090, 2026-10-04)
ap.add_argument("--dim", type=int, default=64)  # word-vector size (PCA of the reader network's 384-d vectors; 384 = all)
ap.add_argument("--k", type=int, default=9)  # kernel width (odd: centred); the user asked for wider (2026-10-04)
ap.add_argument("--frac", type=float, default=1.0)  # learning curve: train on this share of the training NDAs
ap.add_argument("--regex", type=float, default=0.0)  # weight of a head that predicts the symbolic detectors' flags (nda_rules.py)
a = ap.parse_args()
if a.device != "cpu": torch.set_default_device(a.device)  # factories, parameters, as_tensor(numpy) all land there
DIM = a.dim  # 64 kept 33.9% of the vectors' variance, 128 53.1%, 384 all (the user asked for 128, 2026-10-04)
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
name = a.variant + ("+graph" if a.graph else "") + ("+2slot" if a.slots == 2 else "") + (f"+jev{a.jev}" if a.jev != "none" else "") + \
    ("+regex" if a.regex else "") + ("+sep" if a.sep else "") + (f"+sec{a.extra_n or 'all'}w{a.extra_w}" if a.extra else "") + (f"+d{a.dim}" if a.dim != 64 else "") + (f"+across{a.across}" if a.across != 1 else "") + (f"+k{a.k}" if a.k != 9 else "") + (f"+frac{a.frac}" if a.frac < 1 else "")
if a.stmt_feats: name += "+x" + "-".join(a.stmt_feats.split(","))
if a.across_st: name += f"+ast{a.across_st}"
if a.doc_vec: name += f"+dv{a.doc_vec}"
if a.tag: name += f"+{a.tag}"
if a.seed: name += f"+s{a.seed}"  # seed ensembles (2026-10-04)
assert a.k % 2 == 1, "odd kernel widths only (centred)"
os.makedirs(f"{HERE}/preds", exist_ok=True)

train_all = pickle.load(open(f"{HERE}/data/train.pkl", "rb"))
ho = pickle.load(open(f"{HERE}/data/ho.pkl", "rb"))
tr = [d for d in train_all if d["fold"] != a.fold]; te = [d for d in train_all if d["fold"] == a.fold] if a.fold >= 0 else ho
if a.max_docs: tr, te = tr[:a.max_docs], te[:max(4, a.max_docs // 4)]
if a.frac < 1: tr = random.Random(1).sample(sorted(tr, key=lambda d: d["doc"]), int(round(a.frac * len(tr))))
EXTRA = []
if a.extra:
    soft = {json.loads(l)["doc"]: json.loads(l) for l in open(f"{a.extra}/soft.jsonl")}
    for d in pickle.load(open(f"{a.extra}/new.pkl", "rb")):
        s_ = soft.get(d["doc"])
        if not s_ or s_["isnda"] < a.isnda_min or str(a.fold) not in s_["soft"]: continue
        d["soft_y"] = [s_["soft"][str(a.fold)][q] for q in QIDS]; EXTRA.append(d)
    if a.extra_n: EXTRA = EXTRA[:a.extra_n]
    print(f"extra NDAs with the fold's teacher labels: {len(EXTRA)}", flush=True)
tr = tr + EXTRA
XF = {n: pickle.load(open(f"{HERE}/data/xf_{n}.pkl", "rb")) for n in a.stmt_feats.split(",") if n}
XDIM = sum(next(iter(v["train"].values())).shape[1] for v in XF.values())
DV = pickle.load(open(f"{HERE}/data/xf_{a.doc_vec}.pkl", "rb")) if a.doc_vec else None  # the user, 2026-10-05: a document embedding
DVDIM = next(iter(DV["train"].values())).shape[1] if DV else 0


def attach(docs, split):
    """each statement's extra vectors (token -> sentence -> clause views; the user, 2026-10-04)"""
    for d in docs:
        if XF: d["xf"] = torch.tensor(np.concatenate([XF[n][split][d["doc"]] for n in XF], 1), dtype=torch.float32)
        if DV: d["dvec"] = torch.tensor(DV[split][d["doc"]].astype(np.float32).mean(0))


attach(train_all, "train"); attach(ho, "ho"); attach(EXTRA, "new")

# vocabulary: every token id of the four splits (ids never seen in training keep their pretrained vector)
VOC = os.path.join(HERE, "data", f"vocab_pca{DIM}{'_ext' if a.extra else ''}.pt")
if not os.path.exists(VOC):
    from safetensors.torch import load_file
    ids = set()
    for s in ("train", "ho", "s160", "test"):
        for d in pickle.load(open(f"{HERE}/data/{s}.pkl", "rb")): ids.update(d["ids"].tolist())
    if a.extra:
        for d in pickle.load(open(f"{a.extra}/new.pkl", "rb")): ids.update(d["ids"].tolist())
    ids = sorted(ids)
    W = load_file("/root/projects/zadumai/legalbench_map/models/reader_net/model.safetensors")
    W = next(v for k, v in W.items() if k.endswith("word_embeddings.weight"))[ids].float()
    W = W - W.mean(0)
    U, S, V = torch.linalg.svd(W, full_matrices=False)
    E = W @ V[:DIM].T; E = E / E.std(0)
    torch.save({"ids": ids, "E": E}, VOC)
vv = torch.load(VOC); remap = {t: i + 1 for i, t in enumerate(vv["ids"])}  # 0 = padding / unknown


def ids_of(d):
    return torch.tensor([remap.get(t, 0) for t in d["ids"].tolist()])


FEATS = f"{HERE}/../feats"
JEV = {}  # (doc, qid) -> [p_no, p_yes, p_not_stated], Jev's choice (training NDAs)
for f in ("jev_nda_tune.jsonl", "jev_nda_more.jsonl"):
    if a.jev != "none" and os.path.exists(f"{FEATS}/{f}"):
        for l in open(f"{FEATS}/{f}"):
            r = json.loads(l); pr = r["probs"]; v = [pr.get("no", 0.0), pr.get("yes", 0.0), pr.get("not stated", 0.0)]
            t = sum(v); JEV[tuple(r["id"].split("/"))] = [x / t for x in v] if t > 0 else None
RX, RKEYS = {}, []
if a.regex:
    for l in open(f"{FEATS}/nda_rules.jsonl"):
        r = json.loads(l); RX.setdefault(r["id"].split("/")[0], r)
    RKEYS = sorted({k for r in RX.values() for k in r if k != "id"})
RQ = [QIDS.index("nda-" + k.split("_")[0][1:]) for k in RKEYS]  # each flag is read from its own question's vector
_TID = {}


def token_stmt(d):
    if d["doc"] not in _TID:
        t = torch.full((len(d["ids"]),), len(d["st"]), dtype=torch.long)
        for i, (lo, hi) in enumerate(d["st"].tolist()): t[lo:hi] = i
        _TID[d["doc"]] = t
    return _TID[d["doc"]]


RATES = {"plain": [1], "rising": [1, 2, 4, 8, 16, 32, 64],
         "rising2": [1, 2, 4, 8, 16, 32, 64] * 2,  # twice as deep (14 layers; the user, 2026-10-04): the cycle repeated "falling": [64, 32, 16, 8, 4, 2, 1],
         "updown": [1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1]}


class Block(nn.Module):
    def __init__(self, ch, d):
        super().__init__(); self.conv = nn.Conv1d(ch, ch, a.k, dilation=d, padding=(a.k // 2) * d); self.norm = nn.LayerNorm(ch)

    def forward(self, h):  # h [C, L]
        return h + Fn.gelu(self.norm(self.conv(h.unsqueeze(0)).squeeze(0).T).T)


class Pyramid(nn.Module):
    """DPCNN-style (Johnson & Zhang 2017; the user: "at least 3 ... shrinking"): two convolutions at full length, then
    LEVELS times [halve the length by max-pooling (3 wide, stride 2) -> two convolutions], residual. Every level's
    output is brought back to token positions (nearest) and summed, so statements still pool per token."""
    LEVELS = 6

    def __init__(self, ch):
        super().__init__()
        self.c0 = nn.ModuleList([nn.Conv1d(ch, ch, a.k, padding=a.k // 2) for _ in range(2)])
        self.lv = nn.ModuleList([nn.ModuleList([nn.Conv1d(ch, ch, a.k, padding=a.k // 2) for _ in range(2)]) for _ in range(self.LEVELS)])
        self.mix = nn.Linear(ch * (self.LEVELS + 1), ch)

    def forward(self, h):  # [C, L] -> [C, L]
        L = h.shape[1]; x = h.unsqueeze(0)
        for c in self.c0: x = x + c(Fn.gelu(x))
        outs = [x]
        for convs in self.lv:
            if x.shape[2] < 3: break
            x = Fn.max_pool1d(x, 3, stride=2, padding=1)
            for c in convs: x = x + c(Fn.gelu(x))
            outs.append(x)
        ups = [Fn.interpolate(o, size=L, mode="nearest") for o in outs]
        ups += [torch.zeros_like(ups[0])] * (self.LEVELS + 1 - len(ups))
        return Fn.gelu(self.mix(torch.cat(ups, 1).squeeze(0).T)).T


class ClauseBlock(nn.Module):
    """A dilated kernel over a batch of clauses [S, C, L]; padding stays zero (the "clause" variant)."""
    def __init__(self, ch, d):
        super().__init__(); self.conv = nn.Conv1d(ch, ch, a.k, dilation=d, padding=(a.k // 2) * d); self.norm = nn.LayerNorm(ch)

    def forward(self, h, m):  # m [S, 1, L]
        return (h + Fn.gelu(self.norm(self.conv(h).transpose(1, 2)).transpose(1, 2))) * m


_CL = {}


def clause_ids(d, group=16):
    """The NDA's clauses as batches of `group` clauses of similar length (each padded only to its own longest): most
    clauses are ~34 tokens, a few 256. -> [(positions, ids [g, L], mask [g, 1, L])]"""
    if d["doc"] not in _CL:
        cl = [c.tolist() or [0] for c in d["cl_ids"]]
        order = sorted(range(len(cl)), key=lambda i: len(cl[i])); out = []
        for k in range(0, len(order), group):
            idx = order[k:k + group]; L = max(len(cl[i]) for i in idx)
            ids = torch.zeros(len(idx), L, dtype=torch.long); m = torch.zeros(len(idx), 1, L)
            for j, i in enumerate(idx): ids[j, :len(cl[i])] = torch.tensor([remap.get(t, 0) for t in cl[i]]); m[j, 0, :len(cl[i])] = 1
            out.append((torch.tensor(idx), ids, m))
        _CL[d["doc"]] = out
    return _CL[d["doc"]]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        C = a.ch
        self.emb = nn.Embedding(len(vv["ids"]) + 1, DIM, padding_idx=0)
        with torch.no_grad(): self.emb.weight[1:] = vv["E"]; self.emb.weight[0] = 0
        if a.variant == "parallel":
            self.par = nn.ModuleList([nn.Conv1d(DIM, 16, a.k, dilation=d, padding=(a.k // 2) * d) for d in range(1, 17)])
            self.proj = nn.Conv1d(16 * 16, C, 1); self.blocks = nn.ModuleList()
        elif a.variant == "plain":
            self.proj = nn.Conv1d(DIM, C, a.k, padding=a.k // 2); self.blocks = nn.ModuleList()
        elif a.variant == "plain3":  # 3 layers, shrinking kernels 13 -> 9 -> 5, no dilation (reach ~25 tokens)
            self.proj = nn.Conv1d(DIM, C, 13, padding=6)
            self.blocks = nn.ModuleList([nn.Conv1d(C, C, 9, padding=4), nn.Conv1d(C, C, 5, padding=2)])
        elif a.variant == "pyramid":
            self.proj = nn.Conv1d(DIM, C, 1); self.blocks = nn.ModuleList([Pyramid(C)])
        elif a.variant in ("clause", "clausetok"):  # the user (2026-10-04): kernels inside each clause only, then across
            inner = (1, 2, 4, 8, 16) if a.variant == "clause" else (1, 2, 4)
            self.proj = nn.Conv1d(DIM, C, 1); self.blocks = nn.ModuleList([ClauseBlock(C, d) for d in inner])
            if a.variant == "clausetok":  # then the clauses' tokens end to end, kernels over the whole NDA (dilations 8 .. 64)
                self.outer = nn.ModuleList([Block(C, d) for d in (8, 16, 32, 64)])
            # kernels over the sequence of clause vectors: --across layers, 3 wide, dilations 1, 2, 4, 8 ...
            self.across = nn.ModuleList([nn.Conv1d(C, C, 3, dilation=2 ** i, padding=2 ** i) for i in range(a.across)])
            self.across_norm = nn.ModuleList([nn.LayerNorm(C) for _ in range(a.across)])
        else:
            self.proj = nn.Conv1d(DIM, C, 1); self.blocks = nn.ModuleList([Block(C, d) for d in RATES[a.variant]])
        if a.sep: self.sep = nn.Parameter(torch.randn(DIM) * 0.1)
        self.pool = nn.Linear(2 * C + XDIM, C)
        if a.across_st:
            self.across_s = nn.ModuleList([nn.Conv1d(C, C, 3, padding=1) for _ in range(a.across_st)])
            self.across_sn = nn.ModuleList([nn.LayerNorm(C) for _ in range(a.across_st)])
        if a.graph:
            self.g = nn.ModuleList([nn.ModuleDict({k: nn.Linear(C, C, bias=False) for k in ("self", "next", "prev", "xref", "def", "sec")})
                                    for _ in range(2)])
            self.gn = nn.ModuleList([nn.LayerNorm(C) for _ in range(2)])
        self.u = nn.Parameter(torch.randn(a.slots, NQ, C) * 0.05)
        self.head = nn.Parameter(torch.randn(NQ, a.slots * C + (C if DV else 0), 3) * 0.02); self.hb = nn.Parameter(torch.zeros(NQ, 3))
        if DV: self.dproj = nn.Linear(DVDIM, C)
        self.st_head = nn.Parameter(torch.randn(NQ, C, 3) * 0.02); self.sb = nn.Parameter(torch.zeros(NQ, 3))
        self.drop = nn.Dropout(0.1)
        if a.jev == "head": self.jhead = nn.Parameter(torch.randn(NQ, a.slots * C, 3) * 0.02); self.jb = nn.Parameter(torch.zeros(NQ, 3))
        if RKEYS: self.rhead = nn.Parameter(torch.randn(len(RKEYS), a.slots * C) * 0.02); self.rb = nn.Parameter(torch.zeros(len(RKEYS)))

    def forward(self, d, ids):
        if a.variant in ("clause", "clausetok"): return self.tail(d, self.clauses(d))
        x = self.emb(ids)
        if a.sep: x = x.index_add(0, torch.as_tensor(d["st"][:, 0]), self.sep.expand(len(d["st"]), -1))
        x = self.drop(x).T  # [64, L]
        if a.variant == "parallel":
            h = self.proj(torch.cat([Fn.gelu(c(x.unsqueeze(0))) for c in self.par], 1)).squeeze(0)
        else:
            h = self.proj(x.unsqueeze(0)).squeeze(0)
            if a.variant in ("plain", "plain3"): h = Fn.gelu(h)
        if a.variant == "plain3":
            for b in self.blocks: h = h + Fn.gelu(b(h.unsqueeze(0)).squeeze(0))
        else:
            for b in self.blocks: h = b(h)
        h = h.T  # [L, C]
        tid, S = token_stmt(d), len(d["st"])  # each token's statement (S: none)
        ix = tid.unsqueeze(1).expand(-1, h.shape[1])
        mx = torch.zeros(S + 1, h.shape[1]).scatter_reduce(0, ix, h, "amax", include_self=False)[:S]
        mean = torch.zeros(S + 1, h.shape[1]).index_add(0, tid, h)[:S] / torch.as_tensor(d["st"][:, 1] - d["st"][:, 0]).clamp(min=1).unsqueeze(1)
        return self.tail(d, torch.cat([mx, mean], 1))

    def clauses(self, d):
        parts, toks = [], {}
        for idx, ids, m in clause_ids(d):
            h = self.proj(self.drop(self.emb(ids)).transpose(1, 2)) * m  # [g, C, L]
            for b in self.blocks: h = b(h, m)
            if a.variant == "clausetok":
                for j, i in enumerate(idx.tolist()): toks[i] = h[j, :, :int(m[j, 0].sum())]
                continue
            mx = h.masked_fill(m == 0, -1e4).max(-1).values; mean = h.sum(-1) / m.sum(-1).clamp(min=1)
            parts.append((idx, torch.cat([mx, mean], 1)))
        if a.variant == "clausetok":  # the clauses' token features end to end, in document order -> kernels over the NDA
            S = len(d["cl_ids"]); h = torch.cat([toks[i] for i in range(S)], 1)  # [C, T]
            for b in self.outer: h = b(h)
            h = h.T; lens = torch.tensor([toks[i].shape[1] for i in range(S)])
            tid = torch.repeat_interleave(torch.arange(S), lens); ix = tid.unsqueeze(1).expand(-1, h.shape[1])
            mx = torch.zeros(S, h.shape[1]).scatter_reduce(0, ix, h, "amax", include_self=False)
            mean = torch.zeros(S, h.shape[1]).index_add(0, tid, h) / lens.clamp(min=1).unsqueeze(1)
            return torch.cat([mx, mean], 1)
        out = torch.zeros(len(d["cl_ids"]), parts[0][1].shape[1])
        for idx, v in parts: out = out.index_copy(0, idx, v)
        return out

    def tail(self, d, s):
        if XDIM: s = torch.cat([s, d["xf"]], 1)  # + the clause's sentence / TF-IDF vectors
        s = Fn.gelu(self.pool(self.drop(s)))  # [S, C]
        if a.across_st:
            for conv, norm in zip(self.across_s, self.across_sn):
                s = s + Fn.gelu(norm(conv(s.T.unsqueeze(0)).squeeze(0).T))
        if a.variant in ("clause", "clausetok"):  # kernels over the sequence of clause vectors (neighbouring clauses)
            for conv, norm in zip(self.across, self.across_norm):
                s = s + Fn.gelu(norm(conv(s.T.unsqueeze(0)).squeeze(0).T))
        if a.graph: s = self.graph(d, s)
        att = torch.einsum("sc,kqc->kqs", s, self.u).softmax(-1)  # [slots, 17, S]
        v = torch.einsum("kqs,sc->qkc", att, s).reshape(NQ, -1)
        if DV: v = torch.cat([v, Fn.gelu(self.dproj(self.drop(d["dvec"]))).expand(NQ, -1)], 1)  # + the whole-NDA vector
        logits = torch.einsum("qc,qcz->qz", v, self.head) + self.hb
        st_logits = torch.einsum("sc,qcz->sqz", s, self.st_head) + self.sb
        self.v = v
        return logits, att, st_logits

    def graph(self, d, s):
        S = len(s); e = torch.as_tensor(d["edges"]); sec = torch.as_tensor(d["sec"])
        for g, norm in zip(self.g, self.gn):
            msg = g["self"](s)
            for kind, key, rev in ((0, "next", False), (0, "prev", True), (1, "xref", False), (2, "def", False)):
                ek = e[e[:, 0] == kind]
                if not len(ek): continue
                src, dst = (ek[:, 2], ek[:, 1]) if rev else (ek[:, 1], ek[:, 2])
                agg = torch.zeros_like(s).index_add_(0, dst, s[src]); cnt = torch.zeros(S).index_add_(0, dst, torch.ones(len(dst)))
                msg = msg + g[key](agg / cnt.clamp(min=1).unsqueeze(1))
            if (sec >= 0).any():  # same section: the mean of the section's statements
                ks = sec.clamp(min=0); tot = torch.zeros(int(ks.max()) + 1, s.shape[1]).index_add_(0, ks, s)
                n = torch.zeros(int(ks.max()) + 1).index_add_(0, ks, torch.ones(S))
                msg = msg + g["sec"]((tot[ks] / n[ks].unsqueeze(1)) * (sec >= 0).float().unsqueeze(1))
            s = s + Fn.gelu(norm(msg))
        return s


net = Net()
emb_p = [net.emb.weight]; rest = [p for n, p in net.named_parameters() if n != "emb.weight"]
opt = torch.optim.AdamW([{"params": emb_p, "lr": a.lr / 4}, {"params": rest, "lr": a.lr}], weight_decay=0.01)
steps = math.ceil(len(tr) / a.accum) * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr / 4, a.lr], total_steps=steps, pct_start=0.1)
cw = torch.tensor([3.0, 3.0, 1.0])  # statement labels: says no / says yes weighted (most statements say nothing)
cache = {d["doc"]: ids_of(d) for d in tr + te}
t0 = time.time(); step = 0
for ep in range(a.epochs):
    net.train(); random.shuffle(tr); tot = 0.0
    for k, d in enumerate(tr):
        logits, att, st_logits = net(d, cache[d["doc"]])
        if "soft_y" in d:  # an extra NDA: the teacher's soft labels only (no evidence statements)
            loss = a.extra_w * -(torch.tensor(d["soft_y"]) * logits.log_softmax(-1)).sum(-1).mean()
            (loss / a.accum).backward(); tot += loss.item()
            if (k + 1) % a.accum == 0 or k == len(tr) - 1:
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
                if step < steps - 1: sched.step()
                step += 1
            continue
        y = torch.as_tensor(d["y"]); ev = torch.as_tensor(d["ev"])
        jt = [JEV.get((d["doc"], q)) for q in QIDS] if a.jev != "none" else []
        jm = torch.tensor([t is not None for t in jt]) if jt else None
        if a.jev == "mix" and jm.any():
            soft = Fn.one_hot(y, 3).float()
            soft[jm] = (1 - a.alpha) * soft[jm] + a.alpha * torch.tensor([t for t in jt if t is not None])
            loss = -(soft * logits.log_softmax(-1)).sum(-1).mean()
        else:
            loss = Fn.cross_entropy(logits, y)
        if a.jev == "head" and jm.any():
            jl_ = torch.einsum("qc,qcz->qz", net.v, net.jhead) + net.jb
            loss = loss + a.jev_w * -(torch.tensor([t for t in jt if t is not None]) * jl_[jm].log_softmax(-1)).sum(-1).mean()
        if RKEYS and d["doc"] in RX:
            fl = torch.tensor([float(bool(RX[d["doc"]].get(k))) for k in RKEYS])
            rl = (net.v[RQ] * net.rhead).sum(-1) + net.rb
            loss = loss + a.regex * Fn.binary_cross_entropy_with_logits(rl, fl)
        has = (ev != 2)  # [S, 17]
        q_ev = has.any(0)
        if q_ev.any():
            mass = (att[0] * has.T.float()).sum(-1)[q_ev]  # slot 1's attention on the evidence statements
            loss = loss + a.att_w * -(mass.clamp(min=1e-6).log()).mean()
        loss = loss + a.st_w * Fn.cross_entropy(st_logits.reshape(-1, 3), ev.reshape(-1), weight=cw)
        (loss / a.accum).backward(); tot += loss.item()
        if (k + 1) % a.accum == 0 or k == len(tr) - 1:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
            if step < steps - 1: sched.step()
            step += 1
    print(f"{name} fold {a.fold} epoch {ep + 1}/{a.epochs} loss {tot / len(tr):.3f} {time.time() - t0:.0f}s", flush=True)

if a.save:
    os.makedirs(f"{HERE}/models", exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "args": vars(a), "vocab": VOC}, f"{HERE}/models/{name}_f{a.fold}.pt")
net.eval(); right = 0
with torch.inference_mode(), open(f"{HERE}/preds/{name}_f{a.fold}.jsonl", "w") as f:
    for d in te:
        p = net(d, cache[d["doc"]])[0].softmax(-1)
        right += int((p.argmax(-1) == torch.as_tensor(d["y"])).sum())
        for k in range(NQ):
            f.write(json.dumps({"id": f"{d['doc']}/{QIDS[k]}", "p": [round(float(x), 4) for x in p[k]]}) + "\n")
print(f"{name} fold {a.fold}: {right / (NQ * len(te)):.2%} on {len(te)} NDAs, {time.time() - t0:.0f}s", flush=True)
if a.sealed and a.fold == -1:  # the one-time final check: predictions only, read later by final scoring
    for split in ("test", "s160"):
        docs = pickle.load(open(f"{HERE}/data/{split}.pkl", "rb")); attach(docs, split)
        with torch.inference_mode(), open(f"{HERE}/preds/{name}_{split}.jsonl", "w") as f:
            for d in docs:
                p = net(d, ids_of(d))[0].softmax(-1)
                for k in range(NQ):
                    f.write(json.dumps({"id": f"{d['doc']}/{QIDS[k]}", "p": [round(float(x), 4) for x in p[k]]}) + "\n")
        print(f"{name}: predictions written for the sealed {split} set ({len(docs)} NDAs)", flush=True)
