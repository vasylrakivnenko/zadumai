"""Clause-type and contract-type classification benchmark (2026-10-02): cheap methods vs Jev with fixed classes.

Ground truth (all expert- or source-labeled, nothing generated):
  ledgar       LEDGAR (LexGLUE): one clause type per provision, 100 types; 60k train / 10k val / 10k test.
  cuad_clause  CUAD paragraphs: types of every expert-marked passage overlapping the paragraph (0..n of 37
               categories; none = no marked passage). Train/test split = CUAD's own split (by contract).
  cuad_type    CUAD's 508 contracts with their contract type (25 types, from CUAD's folders). Too small for a
               held-out split: local methods are scored on 5-fold out-of-fold predictions.
Methods: tfidf (TF-IDF word 1-2 grams + logistic regression), emb (bge-small-en-v1.5 embeddings + logistic
regression), jev (Jev `choice` over the class names, zero-shot; one call per item; answers cached in out/).
Stages grow the eval sample (the same fixed order, so stage 1 ⊂ stage 2 ⊂ stage 3) and the training data:
  stage 1: eval 500 (cuad_type 200), train <= 6,000;  stage 2: eval 2,000, train <= 20,000;  stage 3: all.
usage: bench.py SET --stage N [--methods tfidf,emb,jev] [--jev-workers 4]
"""
import argparse, collections, concurrent.futures as cf, json, os, random, re, sys, threading, time, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
RN = "/root/zadumai_nli_proto/reader_net"
OUT = f"{HERE}/out"
NJOBS = int(os.environ.get("BENCH_JOBS", "8"))  # CPU cores for one-vs-rest fits
STAGES = {1: (500, 6000), 2: (2000, 20000), 3: (10**9, 10**9)}
META = {"Document Name", "Parties", "Agreement Date", "Effective Date"}  # CUAD fields that aren't clauses


# ---------- ground truth ----------
def _ledgar():
    rows = {s: [{"id": r["id"], "text": r["text"], "labels": [r["type"]]} for r in json.load(open(f"{RN}/ledgar_{s}.json"))]
            for s in ("train", "validation", "test")}
    labels = sorted({r["labels"][0] for r in rows["train"]})
    return dict(train=rows["train"], val=rows["validation"], test=rows["test"], labels=labels, multi=False,
                desc={l: l for l in labels}, question="Which type of contract provision is this text?")


def _cuad_clause():
    data = json.load(open(f"{D}/data/CUADv1.json"))["data"]
    test_titles = {x["title"] for x in json.load(open(f"{D}/data/test.json"))["data"]}
    desc, out = {}, {"train": [], "test": []}
    for doc in data:
        p = doc["paragraphs"][0]; ctx = p["context"]
        spans = []
        for q in p["qas"]:
            cat = q["id"].split("__")[-1]
            if cat in META: continue
            desc.setdefault(cat, q["question"].split("Details:")[-1].strip())
            spans += [(a["answer_start"], a["answer_start"] + len(a["text"]), cat) for a in q["answers"]]
        pos = 0
        for i, block in enumerate(ctx.split("\n\n")):
            s, e = pos, pos + len(block); pos = e + 2
            text = re.sub(r"\s+", " ", block).strip()
            if len(text) < 40: continue
            cats = sorted({c for a, b, c in spans if a < min(e, s + 4000) and b > s})  # only what the 4,000-char cut shows
            out["test" if doc["title"] in test_titles else "train"].append(
                {"id": f"{doc['title'][:60]}#{i}", "doc": doc["title"], "text": text[:4000], "labels": cats})
    # validation = 10% of the training contracts
    docs = sorted({r["doc"] for r in out["train"]}); random.Random(0).shuffle(docs); vd = set(docs[:len(docs) // 10])
    val = [r for r in out["train"] if r["doc"] in vd]; train = [r for r in out["train"] if r["doc"] not in vd]
    labels = sorted(desc)
    return dict(train=train, val=val, test=out["test"], labels=labels, multi=True, desc=desc,
                question="Which type of contract clause is this text? If it is none of these, pick NONE.")


def _cuad_type():
    names = zipfile.ZipFile(f"{D}/CUAD_v1.zip").namelist()
    norm = {"Joint Venture _ Filing": "Joint Venture", "Endorsement Agreement": "Endorsement",
            "Affiliate Agreement": "Affiliate_Agreements"}
    lab = {}
    for x in names:
        if x.lower().endswith(".pdf"):
            t = x.split("/")[-2]; lab[os.path.splitext(x.split("/")[-1])[0].strip().lower()] = norm.get(t, t)
    pretty = lambda t: re.sub(r"_?Agreements?$", "", t).replace("_", " ").replace("Co Branding", "Co-Branding").strip() + " Agreement"
    rows = []
    txt = f"{D}/cuad_full/CUAD_v1/full_contract_txt"
    for f in sorted(os.listdir(txt)):
        k = os.path.splitext(f)[0].strip().lower()
        if k in lab:
            rows.append({"id": f, "text": open(f"{txt}/{f}", errors="ignore").read(), "labels": [pretty(lab[k])]})
    labels = sorted({r["labels"][0] for r in rows})
    return dict(train=rows, val=[], test=rows, labels=labels, multi=False, desc={l: l for l in labels}, cv=True,
                question="What type of contract is this document?")


def _tos():
    from datasets import load_dataset
    d = load_dataset("coastalcph/lex_glue", "unfair_tos"); names = d["train"].features["labels"].feature.names
    rows = {s: [{"id": f"tos/{s}/{i}", "text": x["text"], "labels": sorted(names[j] for j in x["labels"])} for i, x in enumerate(d[s])]
            for s in ("train", "validation", "test")}
    desc = {"Limitation of liability": "limits or excludes the provider's liability",
            "Unilateral termination": "the provider may suspend or terminate the service or account at its discretion",
            "Unilateral change": "the provider may change the terms or the service unilaterally",
            "Content removal": "the provider may remove or delete user content",
            "Contract by using": "the user is bound by the terms simply by using the service",
            "Choice of law": "names the law that governs the contract",
            "Jurisdiction": "names the courts or place where disputes are heard",
            "Arbitration": "disputes go to arbitration"}
    return dict(train=rows["train"], val=rows["validation"], test=rows["test"], labels=sorted(names), multi=True, desc=desc,
                question="Which type of terms-of-service clause is this sentence? If it is none of these, pick NONE.")


EXT = f"{D}/data/ext"


def _split_docs(rows, test=0.2, val=0.1, seed=0):
    """Split by document (row["doc"]) so no document is in two splits."""
    docs = sorted({r["doc"] for r in rows}); random.Random(seed).shuffle(docs)
    nt, nv = int(len(docs) * test), int(len(docs) * val)
    te, va = set(docs[:nt]), set(docs[nt:nt + nv])
    return ([r for r in rows if r["doc"] not in te | va], [r for r in rows if r["doc"] in va], [r for r in rows if r["doc"] in te])


def _opp115():
    """OPP-115 privacy policies: segments × 10 data-practice categories (consolidated at 0.5 annotator overlap)."""
    import csv, io, html
    z = zipfile.ZipFile(f"{EXT}/opp115.zip"); names = z.namelist()
    cats = collections.defaultdict(set)
    for n in names:
        if "threshold-0.5-overlap-similarity" in n and n.endswith(".csv"):
            pid = n.split("/")[-1].split("_")[0]
            for row in csv.reader(io.TextIOWrapper(z.open(n), "utf-8")):
                cats[(pid, int(row[4]))].add(row[5])
    rows = []
    for n in names:
        if "sanitized_policies" in n and n.endswith(".html"):
            pid = n.split("/")[-1].split("_")[0]
            for i, seg in enumerate(z.read(n).decode("utf-8", "ignore").split("|||")):
                t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", seg))).strip()
                if len(t) >= 30:
                    rows.append({"id": f"opp/{pid}/{i}", "doc": pid, "text": t[:4000], "labels": sorted(cats.get((pid, i), set()) - {"Other"})})
    tr, va, te = _split_docs(rows)
    desc = {"First Party Collection/Use": "how and why the service collects or uses user data",
            "Third Party Sharing/Collection": "sharing user data with or collection by third parties",
            "User Choice/Control": "choices and control options the user has", "User Access, Edit and Deletion": "user can access, edit or delete their data",
            "Data Retention": "how long user data is kept", "Data Security": "how user data is protected",
            "Policy Change": "how users are told about changes to the policy", "Do Not Track": "Do Not Track signals",
            "International and Specific Audiences": "rules for children, specific regions or audiences"}
    labels = sorted({l for r in rows for l in r["labels"]})
    return dict(train=tr, val=va, test=te, labels=labels, multi=True, desc={l: desc.get(l, l) for l in labels},
                question="Which type of privacy-policy section is this text? If it is none of these, pick NONE.")


def _contractnli():
    """ContractNLI NDAs: each annotated span × the NDA questions it is evidence for (17 types; none = evidence for nothing)."""
    z = zipfile.ZipFile(f"{EXT}/contractnli.zip"); out = {}
    for split in ("train", "dev", "test"):
        d = json.loads(z.read(f"contract-nli/{split}.json")); lab = d["labels"]; rows = []
        for doc in d["documents"]:
            ev = collections.defaultdict(set)
            for k, a in doc["annotation_sets"][0]["annotations"].items():
                for s in a["spans"]: ev[s].add(lab[k]["short_description"])
            for i, (s, e) in enumerate(doc["spans"]):
                t = re.sub(r"\s+", " ", doc["text"][s:e]).strip()
                if len(t) >= 30: rows.append({"id": f"cnli/{doc['id']}/{i}", "doc": doc["id"], "text": t[:4000], "labels": sorted(ev.get(i, set()))})
        out[split] = rows
    labels = sorted({v["short_description"] for v in lab.values()})
    return dict(train=out["train"], val=out["dev"], test=out["test"], labels=labels, multi=True,
                desc={v["short_description"]: v["hypothesis"] for v in lab.values()},
                question="Which type of NDA clause is this text? If it is none of these, pick NONE.")


LEASE_TYPES = {"term_of_payment": "payment terms (when and how rent is paid)", "leased_space": "description of the leased premises",
               "designated_use": "what the premises may be used for", "notice_period": "notice period for termination",
               "vat": "VAT on rent", "extension_period": "extension or renewal of the lease term",
               "indexation_rent": "rent indexation or review", "redflag": "a clause that is risky or unusual for the tenant (red flag)"}


def _lease():
    """Lease benchmark (Leivaditi et al., 179 leases): paragraphs × which terms they contain (+ red flag)."""
    import html as H
    z = zipfile.ZipFile(f"{EXT}/lease_annotated.zip"); names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
    leg = json.load(open(f"{EXT}/lease_legend.json")); rows = []
    for n in names:
        if not n.endswith(".plain.html"): continue
        base = n[:-len(".plain.html")]
        if base + ".ann.json" not in names: continue
        h = z.read(n).decode("utf-8", "ignore")
        paras = {m.group(1): re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", m.group(2)))).strip()
                 for m in re.finditer(r'<p id="([^"]+)">(.*?)</p>', h, re.S)}
        tags = collections.defaultdict(set)
        for e in json.loads(z.read(base + ".ann.json"))["entities"]:
            c = leg.get(e["classId"])
            if c in LEASE_TYPES: tags[e["part"]].add(c)
        doc = base.split("/")[-1]
        for pid, t in paras.items():
            if len(t) >= 40: rows.append({"id": f"lease/{doc}/{pid}", "doc": doc, "text": t[:4000], "labels": sorted(tags.get(pid, set()))})
    tr, va, te = _split_docs(rows)
    return dict(train=tr, val=va, test=te, labels=sorted(LEASE_TYPES), multi=True, desc=LEASE_TYPES,
                question="Which of these does this lease paragraph contain? If none, pick NONE.")


def _doctype():
    """Document type from whole documents (English). Labels come from each source: CUAD (commercial agreements),
    ContractNLI (NDAs), OPP-115 + ToS;DR (privacy policies), ToS;DR document names (terms of service, cookie policy,
    licence/EULA), lease benchmark (leases). ToS;DR samples: up to 300 per type. Split 70/30 per type (seed 0)."""
    import html as H
    rows = []
    for r in _cuad_type()["test"]: rows.append({"id": "cuad/" + r["id"], "text": r["text"], "labels": ["commercial agreement"]})
    z = zipfile.ZipFile(f"{EXT}/contractnli.zip")
    for s in ("train", "dev", "test"):
        for d in json.loads(z.read(f"contract-nli/{s}.json"))["documents"]:
            rows.append({"id": f"nda/{d['id']}", "text": d["text"], "labels": ["non-disclosure agreement"]})
    z = zipfile.ZipFile(f"{EXT}/opp115.zip")
    for n in z.namelist():
        if "sanitized_policies" in n and n.endswith(".html"):
            t = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf-8", "ignore").replace("|||", "\n"))))
            rows.append({"id": "opp/" + n.split("/")[-1], "text": t, "labels": ["privacy policy"]})
    z = zipfile.ZipFile(f"{EXT}/lease_annotated.zip")
    for n in z.namelist():
        if n.endswith(".plain.html") and not n.startswith("__MACOSX"):
            h = z.read(n).decode("utf-8", "ignore")
            t = "\n".join(re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", m))) for m in re.findall(r'<p id="[^"]+">(.*?)</p>', h, re.S))
            rows.append({"id": "lease/" + n.split("/")[-1], "text": t, "labels": ["lease"]})
    z = zipfile.ZipFile(f"{EXT}/tosdr.zip"); kinds = collections.defaultdict(list)
    for n in sorted(z.namelist()):
        if "/corpus/text/" not in n or not n.endswith(".txt"): continue
        kind = n.split("/")[-1].rsplit("_", 1)[-1][:-4].lower()
        if "cookie" in kind: k = "cookie policy"
        elif "privacy" in kind: k = "privacy policy"
        elif "eula" in kind or "licen" in kind: k = "software licence / EULA"
        elif re.search(r"terms|conditions|user agreement", kind): k = "terms of service"
        else: continue
        kinds[k].append(n)
    for k, ns in kinds.items():
        random.Random(0).shuffle(ns)
        for n in ns[:300]:
            t = z.read(n).decode("utf-8", "ignore")
            if len(t) >= 500: rows.append({"id": "tosdr/" + n.split("/")[-1], "text": t, "labels": [k]})
    by = collections.defaultdict(list)
    for r in rows: by[r["labels"][0]].append(r)
    tr, te = [], []
    for k, rs in by.items():
        random.Random(0).shuffle(rs); cut = int(len(rs) * 0.7); tr += rs[:cut]; te += rs[cut:]
    labels = sorted(by)
    return dict(train=tr, val=[], test=te, labels=labels, multi=False, desc={l: l for l in labels}, views=True,
                question="What type of legal document is this?")


MCC_DESC = {"employment": "employment agreement (incl. executive, severance, compensation, benefit plans)",
            "security": "securities or financing agreement (credit, loan, notes, indentures, warrants, derivatives)",
            "purchase": "purchase or merger agreement (M&A, asset or stock purchase)",
            "services": "services or supply agreement (services, goods, licensing, distribution)",
            "shareholder": "shareholder or governance agreement (shareholder rights, voting, formation)",
            "lease": "lease agreement", "other": "other agreement", "na": "not a substantive contract"}


def _mcc():
    """Material Contracts Corpus sample: 250 SEC-filed contracts per MCC label (labels are MCC's own model labels,
    ~95% accurate per its paper, so this is silver ground truth). Split 70/30 per label."""
    rows = [json.loads(l) for l in open(f"{EXT}/mcc/mcc_sample.jsonl")]
    rows = [{"id": "mcc/" + r["doc_key"], "text": r["text"], "labels": [r["label"]]} for r in rows if "text" in r and len(r["text"]) >= 300]
    by = collections.defaultdict(list)
    for r in rows: by[r["labels"][0]].append(r)
    tr, te = [], []
    for k, rs in sorted(by.items()):
        random.Random(0).shuffle(rs); cut = int(len(rs) * 0.7); tr += rs[:cut]; te += rs[cut:]
    labels = sorted(by)
    return dict(train=tr, val=[], test=te, labels=labels, multi=False, desc={l: MCC_DESC[l] for l in labels}, views=True,
                question="What type of contract is this document?")


SETS = {"ledgar": _ledgar, "cuad_clause": _cuad_clause, "cuad_type": _cuad_type, "tos": _tos,
        "opp115": _opp115, "contractnli": _contractnli, "lease": _lease, "doctype": _doctype, "mcc": _mcc}

# LEDGAR labels are each contract's own section heading, so near-synonyms are separate classes. Scored also with
# these merged (a stand-in for a taxonomy of our own).
MERGE = {l: g for g, ls in {
    "assignment/successors": ["Assignments", "Assigns", "Successors", "Binding Effects"], "waivers": ["Waivers", "No Waivers"],
    "withholding": ["Withholdings", "Tax Withholdings"], "governing law": ["Governing Laws", "Applicable Laws"],
    "jurisdiction/venue": ["Jurisdictions", "Submission To Jurisdiction", "Consent To Jurisdiction", "Venues"],
    "costs": ["Expenses", "Costs", "Fees"], "entire agreement": ["Entire Agreements", "Integration"],
    "amendments": ["Amendments", "Modifications"], "indemnity": ["Indemnifications", "Indemnity"],
    "definitions": ["Definitions", "Defined Terms"], "organization": ["Organizations", "Existence"],
    "consents": ["Consents", "Approvals", "Authorizations"], "interpretation": ["Construction", "Interpretations", "Headings"],
    "remedies": ["Remedies", "Specific Performance"], "miscellaneous": ["Miscellaneous", "General"],
    "effectiveness": ["Effective Dates", "Effectiveness"]}.items() for l in ls}


def sample(rows, n, seed=0, multi=False):
    """Fixed order, so a bigger stage contains the smaller one. cuad_clause: 60% rows with a type, 40% none."""
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    if multi:
        pos = [i for i in order if rows[i]["labels"]]; neg = [i for i in order if not rows[i]["labels"]]
        k = min(len(pos), int(n * 0.6)); order = pos[:k] + neg[:min(len(neg), n - k)]
        return [rows[i] for i in order]
    return [rows[i] for i in order[:n]]


def view(text, kind):
    """What a method reads. Contracts: the first 4,000 characters (title included), or 'body' = 2,000-6,000."""
    if kind == "head": return text[:4000]
    if kind == "body": return text[2000:6000]
    return text


# ---------- methods ----------
class Tfidf:
    name = "tfidf"
    def fit(self, X, Y, multi, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        self.v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300_000, sublinear_tf=True)
        A = self.v.fit_transform(X)
        self.m = OneVsRestClassifier(LogisticRegression(C=10, max_iter=2000, class_weight="balanced"), n_jobs=NJOBS) if multi else LogisticRegression(C=10, max_iter=2000)
        self.m.fit(A, Y); return self
    def scores(self, X):
        return self.m.predict_proba(self.v.transform(X))


class Emb:
    name = "emb"
    _model = None
    _cache = {}  # (contract, text) -> vector: cross-validation folds and validation reuse the same texts
    def encode(self, X, contract=False):
        todo = [t for t in dict.fromkeys(X) if (contract, t) not in Emb._cache]
        if todo:
            for t, e in zip(todo, self._encode(todo, contract)): Emb._cache[(contract, t)] = e
        return np.array([Emb._cache[(contract, t)] for t in X])
    def _encode(self, X, contract):
        from sentence_transformers import SentenceTransformer
        if Emb._model is None:
            Emb._model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); Emb._model.max_seq_length = 512
        if not contract:
            return Emb._model.encode(X, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        out = []  # a contract: mean of its first 8 chunks of 1,500 characters
        for t in X:
            ch = [t[i:i + 1500] for i in range(0, min(len(t), 12000), 1500)] or [""]
            e = Emb._model.encode(ch, batch_size=8, normalize_embeddings=True, show_progress_bar=False).mean(0)
            out.append(e / np.linalg.norm(e))
        return np.array(out)
    def fit(self, X, Y, multi, labels, contract=False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        self.contract = contract
        A = self.encode(X, contract)
        self.m = OneVsRestClassifier(LogisticRegression(C=10, max_iter=3000, class_weight="balanced"), n_jobs=NJOBS) if multi else LogisticRegression(C=10, max_iter=3000)
        self.m.fit(A, Y); return self
    def scores(self, X):
        return self.m.predict_proba(self.encode(X, self.contract))


class Jev:
    """Zero-shot: Jev `choice` over the class names (+ NONE for cuad_clause). Answers cached per (set, view, id)."""
    name = "jev"
    def __init__(self, setname, kind, workers):
        sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
        from router.systemone import SystemOne
        self.c = SystemOne.jev(); self.workers = workers
        self.path = f"{OUT}/jev_cache_{setname}_{kind}.jsonl"
        self.cache = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line); self.cache[r["id"]] = r
        self.lock = threading.Lock()
    def run(self, rows, texts, criteria, question):
        """criteria: one dict for all rows, or a list (one per row: the cascade's candidates)."""
        per_row = isinstance(criteria, list)
        key = lambda i, r: r["id"] + ("|" + "|".join(sorted(criteria[i])) if per_row else "")
        todo = [(i, r, t) for i, (r, t) in enumerate(zip(rows, texts)) if key(i, r) not in self.cache]
        print(f"   jev: {len(rows) - len(todo)} cached, {len(todo)} to call", flush=True)
        def one(irt):
            i, r, t = irt; t0 = time.time()
            try:
                a = self.c.choice(t, question, criteria[i] if per_row else criteria); err = None
            except Exception as e:  # keep going; the row is reported as an error
                a, err = {"choice": None, "probabilities": {}}, str(e)[:200]
            rec = {"id": key(i, r), "choice": a["choice"], "p": a["probabilities"], "ms": (time.time() - t0) * 1000, "err": err}
            with self.lock:
                self.cache[key(i, r)] = rec
                with open(self.path, "a") as f: f.write(json.dumps(rec) + "\n")
            return rec
        done = 0
        with cf.ThreadPoolExecutor(self.workers) as ex:
            for _ in ex.map(one, todo):
                done += 1
                if done % 100 == 0: print(f"   jev: {done}/{len(todo)}", flush=True)
        return [self.cache[key(i, r)] for i, r in enumerate(rows)]


# ---------- scoring ----------
def wilson_lo(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d


def report_single(name, gold, pred, conf, ms, errs_out, rows, merge=None):
    from sklearn.metrics import f1_score
    gold = np.array(gold); pred = np.array(pred, dtype=object); conf = np.array(conf)
    ok = pred == gold
    print(f"  {name:12} accuracy {ok.mean():6.1%} [lo {wilson_lo(ok.sum(), len(ok)):.3f}]  macro-F1 "
          f"{f1_score(gold, pred.astype(str), average='macro'):.3f}  ({ms:.1f} ms/item)"
          + (f"  | near-synonym labels merged: {np.mean([merge.get(g, g) == merge.get(p, p) for g, p in zip(gold, pred)]):6.1%}" if merge else ""))
    for th in (0.5, 0.7, 0.9):
        k = conf >= th
        if k.any():
            print(f"      answer only if p >= {th}: coverage {k.mean():6.1%}  precision {ok[k].mean():6.1%} "
                  f"[lo {wilson_lo(ok[k].sum(), k.sum()):.3f}]")
    conf_pairs = collections.Counter((g, p) for g, p in zip(gold, pred) if g != p)
    print("      most confused (gold -> predicted):", "; ".join(f"{g} -> {p} ×{n}" for (g, p), n in conf_pairs.most_common(6)))
    with open(errs_out, "w") as f:
        for r, g, p, c in zip(rows, gold, pred, conf):
            if g != p: f.write(json.dumps({"id": r["id"], "gold": g, "pred": p, "conf": round(float(c), 3), "text": r["text"][:600]}) + "\n")


def report_multi(name, rows, predsets, ms, errs_out, top1=None):
    tp = fp = fn = 0; exact = 0; none_ok = none_n = 0; per = collections.defaultdict(collections.Counter)
    for r, P in zip(rows, predsets):
        G = set(r["labels"]); P = set(P)
        tp += len(G & P); fp += len(P - G); fn += len(G - P); exact += G == P
        if not G: none_n += 1; none_ok += not P
        for l in G | P: per[l]["tp"] += l in G and l in P; per[l]["fp"] += l in P and l not in G; per[l]["fn"] += l in G and l not in P
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    macro = np.mean([2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1) for c in per.values()])
    print(f"  {name:12} micro P {pr:6.1%} R {rc:6.1%} F1 {f1:.3f}  macro-F1 {macro:.3f}  exact {exact / len(rows):6.1%}  "
          f"'none' rows kept empty {none_ok}/{none_n}  ({ms:.1f} ms/item)")
    if top1 is not None:
        print(f"      (Jev picks one class per paragraph: top choice is one of the gold types or NONE when none: "
              f"{np.mean(top1):6.1%})")
    worst = sorted(per.items(), key=lambda kv: 2 * kv[1]["tp"] / max(2 * kv[1]["tp"] + kv[1]["fp"] + kv[1]["fn"], 1))
    print("      weakest types (F1):", ", ".join(f"{l} {2 * c['tp'] / max(2 * c['tp'] + c['fp'] + c['fn'], 1):.2f}" for l, c in worst[:6]))
    with open(errs_out, "w") as f:
        for r, P in zip(rows, predsets):
            if set(r["labels"]) != set(P): f.write(json.dumps({"id": r["id"], "gold": r["labels"], "pred": sorted(P), "text": r["text"][:600]}) + "\n")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=SETS); ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--methods", default="tfidf,emb,jev"); ap.add_argument("--jev-workers", type=int, default=8)
    ap.add_argument("--views", default="head,body", help="cuad_type only")
    a = ap.parse_args()
    n_eval, n_train = STAGES[a.stage]
    if a.set == "cuad_type" and a.stage == 1: n_eval = 200
    S = SETS[a.set](); labels = S["labels"]; multi = S["multi"]; methods = a.methods.split(",")
    ev = sample(S["test"], n_eval, multi=multi)
    print(f"== {a.set} stage {a.stage}: {len(labels)} classes, eval {len(ev)} rows"
          + ("" if S.get("cv") else f", train {min(n_train, len(S['train']))} of {len(S['train'])}"), flush=True)
    if multi:
        print(f"   eval rows with a type {sum(bool(r['labels']) for r in ev)}, none {sum(not r['labels'] for r in ev)}")
    views = a.views.split(",") if (a.set == "cuad_type" or S.get("views")) else ["text"]
    from sklearn.preprocessing import MultiLabelBinarizer
    merge = MERGE if a.set == "ledgar" else None
    tfidf_scores = {}
    for kind in views:
        if len(views) > 1: print(f" -- view: {kind} ({'first 4,000 chars, title included' if kind == 'head' else 'chars 2,000-6,000, no title'})")
        tag = f"{a.set}_s{a.stage}_{kind}"
        for meth in methods:
            t0 = time.time()
            if meth in ("tfidf", "emb"):
                if S.get("cv"):  # cuad_type: 5-fold out-of-fold over all 508; scored on the stage's eval rows
                    from sklearn.model_selection import StratifiedKFold
                    rows = S["train"]; y = np.array([r["labels"][0] for r in rows]); X = [view(r["text"], kind) for r in rows]
                    sc = np.zeros((len(rows), len(labels))); lab_ix = {l: i for i, l in enumerate(labels)}
                    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                        m = (Tfidf() if meth == "tfidf" else Emb())
                        if meth == "emb": m.fit([X[i] for i in tr], y[tr], False, labels, contract=True)
                        else: m.fit([X[i] for i in tr], y[tr], False, labels)
                        P = m.scores([X[i] for i in te])
                        for j, c in enumerate(m.m.classes_): sc[te, lab_ix[c]] = P[:, j]
                    ids = {r["id"]: i for i, r in enumerate(rows)}; sc = sc[[ids[r["id"]] for r in ev]]
                    ms = (time.time() - t0) * 1000 / len(rows)
                    pred = [labels[i] for i in sc.argmax(1)]
                    report_single(meth, [r["labels"][0] for r in ev], pred, sc.max(1), ms, f"{OUT}/errors_{tag}_{meth}.jsonl", ev)
                    continue
                tr = sample(S["train"], n_train, seed=1, multi=multi)  # multi: 60% rows with a type (bug at stage 1: natural mix, ~13%)
                X = [view(r["text"], kind) for r in tr]
                if multi:
                    mlb = MultiLabelBinarizer(classes=labels); Y = mlb.fit_transform([r["labels"] for r in tr])
                else:
                    Y = np.array([r["labels"][0] for r in tr])
                m = (Tfidf() if meth == "tfidf" else Emb()).fit(X, Y, multi, labels)
                t_fit = time.time() - t0; t1 = time.time()
                sc = m.scores([view(r["text"], kind) for r in ev]); ms = (time.time() - t1) * 1000 / len(ev)
                if meth == "tfidf": tfidf_scores[kind] = (list(labels) if multi else list(m.m.classes_), sc)
                print(f"   ({meth}: trained on {len(tr)} rows in {t_fit:.0f}s)")
                if multi:
                    # threshold on validation: the one with the best micro-F1
                    vs = sample(S["val"], 2000, multi=True); vsc = m.scores([r["text"] for r in vs])
                    best = max((0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda th: _f1(vs, [[labels[j] for j in np.where(s >= th)[0]] for s in vsc]))
                    print(f"   ({meth}: threshold {best}, picked on {len(vs)} validation rows)")
                    report_multi(meth, ev, [[labels[j] for j in np.where(s >= best)[0]] for s in sc], ms, f"{OUT}/errors_{tag}_{meth}.jsonl")
                else:
                    cls = list(m.m.classes_)
                    report_single(meth, [r["labels"][0] for r in ev], [cls[i] for i in sc.argmax(1)], sc.max(1), ms,
                                  f"{OUT}/errors_{tag}_{meth}.jsonl", ev, merge)
            elif meth in ("jev", "cascade"):
                if meth == "cascade":  # TF-IDF's 5 best classes, Jev picks one (needs tfidf earlier in --methods)
                    assert kind in tfidf_scores, "cascade needs tfidf before it in --methods"
                    cls, sc = tfidf_scores[kind]
                    crit = [{cls[j]: S["desc"][cls[j]] for j in np.argsort(-s)[:5]} for s in sc]
                    if multi: crit = [{**c, "NONE": "none of the clause types above"} for c in crit]
                    J = Jev(a.set, kind + "_cascade", a.jev_workers)
                else:
                    J = Jev(a.set, kind, a.jev_workers)
                    crit = {l: S["desc"][l] for l in labels}
                    if multi: crit["NONE"] = "none of the clause types above"
                calls0, tin0, tout0 = J.c.calls, J.c.input_tokens, J.c.output_tokens
                recs = J.run(ev, [view(r["text"], kind) if kind != "text" else r["text"][:4000] for r in ev], crit, S["question"])
                errs = sum(1 for r in recs if r["err"])
                lat = sorted(r["ms"] for r in recs if not r["err"]) or [0]
                new = J.c.calls - calls0
                print(f"   ({meth}: {errs} errors; latency p50 {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * .95) - 1]:.0f} ms; "
                      f"this run: {new} calls, tokens in {J.c.input_tokens - tin0}, out {J.c.output_tokens - tout0}, model {J.c.model_version})")
                good = [(r, x) for r, x in zip(ev, recs) if not x["err"]]
                if multi:
                    preds = [[] if x["choice"] in (None, "NONE") else [x["choice"]] for _, x in good]
                    top1 = [(x["choice"] in r["labels"]) or (x["choice"] == "NONE" and not r["labels"]) for r, x in good]
                    report_multi(meth, [r for r, _ in good], preds, lat[len(lat) // 2], f"{OUT}/errors_{tag}_{meth}.jsonl", top1)
                else:
                    report_single(meth, [r["labels"][0] for r, _ in good], [x["choice"] for _, x in good],
                                  [x["p"].get(x["choice"], 0) for _, x in good], lat[len(lat) // 2], f"{OUT}/errors_{tag}_{meth}.jsonl",
                                  [r for r, _ in good], merge)


def _f1(rows, predsets):
    tp = fp = fn = 0
    for r, P in zip(rows, predsets):
        G = set(r["labels"]); P = set(P); tp += len(G & P); fp += len(P - G); fn += len(G - P)
    return 2 * tp / max(2 * tp + fp + fn, 1)


if __name__ == "__main__":
    main()
