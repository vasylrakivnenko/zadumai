"""Contract tool v0 (2026-10-05; the user approved "v0": find / changes / compare). One command over a matter's documents
folder, for an agent that should not read whole documents into its context. Actions:
  find     the passages most relevant to `query` (all documents, or `document`), each with its document and location;
           deleted text is searchable too and shows as [-deleted-] / {+inserted+}
  changes  the tracked changes in one .docx, clause by clause (LAB's `read` accepts them silently: deletions vanish)
  compare  two versions of an agreement (`document` = older, `against` = newer), aligned paragraph by paragraph, word diff
  outline  the documents (no `document`), or one document's headings and clause numbers
  show     chosen paragraphs of one document in full: a clause number ("4.2") or a ¶ range ("62-73") from outline / find
  ask      = find within one document in v0 (no reader model yet)
Search is hybrid: BM25 (prefix-stemmed) + bge-small-en-v1.5 on CPU, fused by reciprocal rank. Outputs are capped
(~1,200 tokens per call; `offset` pages through changes / compare). Parsed units and embeddings are cached per file.
usage: dspy_venv/bin/python -m contract_tool.tool DOCS_DIR ACTION [--query Q] [--document D] [--against D2] [--offset N]"""
import argparse, collections, difflib, hashlib, itertools, math, os, pickle, re, sys
import numpy as np
from .parse import parse

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache")
STOP = set("a an and are as at be been by for from has have in is it its of on or that the this to was were will with shall "
           "any all such which into under upon than then there these those who whom whose may must can not no".split())
TOK = re.compile(r"[a-z0-9$€£%§]+(?:[.,/\-'][a-z0-9%]+)*")
BUDGET = 5000  # characters per call (~1,200 tokens)
_MODEL = None


def toks(s):
    return [t[:7] if t.isalpha() else t for t in TOK.findall(s.lower()) if t not in STOP]


def model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu"); _MODEL.max_seq_length = 256  # a paragraph; 2x faster on CPU
    return _MODEL


def search_text(u):
    dele = " ".join(t for s, t, _, _ in u.changes if s == "del")
    return " ".join(x for x in (" ".join(u.path), u.label, u.text, dele) if x)


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.len = np.array([len(d) for d in docs], float); self.avg = self.len.mean() if len(docs) else 1.0
        self.post = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in collections.Counter(d).items(): self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + .5) / (len(p) + .5)) for t, p in self.post.items()}

    def scores(self, q):
        s = np.zeros(self.n)
        for t in set(q):
            for i, f in self.post.get(t, ()):
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def wdiff(a, b):
    """word-level diff: (marked text, deleted words, inserted words)."""
    A, B, out, de, ins = a.split(), b.split(), [], [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal": out += A[i1:i2]; continue
        if i2 > i1: out.append("[-" + " ".join(A[i1:i2]) + "-]"); de += A[i1:i2]
        if j2 > j1: out.append("{+" + " ".join(B[j1:j2]) + "+}"); ins += B[j1:j2]
    return " ".join(out), de, ins


def _jac(a, b):
    return len(a & b) / max(1, len(a | b))


def _key(t):
    return re.sub(r"\s+", " ", re.sub(r"^\s*(\(?[0-9a-zA-Z]{1,4}[.)]|\d+(\.\d+)+)\s*", "", t.lower())).strip()


def snippet(text, q=(), words=120):
    """the text, or the `words`-word window with the most query words."""
    w = text.split()
    if len(w) <= words: return text
    qs = set(q); hit = [1 if toks(x) and toks(x)[0] in qs else 0 for x in w]
    c = np.convolve(hit, np.ones(words, int), "valid"); i = int(c.argmax())
    return ("… " if i else "") + " ".join(w[i:i + words]) + (" …" if i + words < len(w) else "")


def change_snippet(marked, ctx=90):
    """the marked text cut to the changes and `ctx` characters around each."""
    spans = [m.span() for m in re.finditer(r"\[-.*?-\]|\{\+.*?\+\}", marked)]
    if not spans or len(marked) <= 600: return marked
    win = []
    for a, b in spans:
        a, b = max(0, a - ctx), min(len(marked), b + ctx)
        if win and a <= win[-1][1]: win[-1] = (win[-1][0], b)
        else: win.append((a, b))
    out = " … ".join(marked[a:b].strip() for a, b in win)
    return ("… " if win[0][0] else "") + out + (" …" if win[-1][1] < len(marked) else "")


def diff_units(A, B):
    """two versions as unit lists -> ([(kind, unit of the newer (or, when removed, older) version, marked text, deleted
    words, inserted words)], unchanged count): paragraphs aligned in order (difflib), paired inside each differing block
    by word overlap, leftovers paired across blocks (moved and edited), then a word diff per pair."""
    ka, kb = [_key(u.text) for u in A], [_key(u.text) for u in B]
    sa, sb = [set(k.split()) for k in ka], [set(k.split()) for k in kb]
    items, rem, add, same = [], [], [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
        if tag == "equal": same += i2 - i1; continue
        used = set()
        for i in range(i1, i2):
            best = max(((_jac(sa[i], sb[j]), j) for j in range(j1, j2) if j not in used), default=(0, None))
            if best[0] >= 0.35: used.add(best[1]); items.append((best[1], i))
            else: rem.append(i)
        add += [j for j in range(j1, j2) if j not in used]
    for i in list(rem):
        best = max(((_jac(sa[i], sb[j]), j) for j in add), default=(0, None))
        if best[0] >= 0.5: add.remove(best[1]); rem.remove(i); items.append((best[1], i))
    out = []
    for j, i in items:
        m, de, ins = wdiff(A[i].text, B[j].text)
        if de or ins: out.append(("changed", B[j], m, de, ins))
        else: same += 1
    out += [("added", B[j], "{+" + B[j].text + "+}", [], B[j].text.split()) for j in add]
    out += [("removed", A[i], "[-" + A[i].text + "-]", A[i].text.split(), []) for i in rem]
    pos = {id(u): n for n, u in enumerate(B)}; posa = {id(u): n for n, u in enumerate(A)}
    out.sort(key=lambda x: pos[id(x[1])] if x[0] != "removed" else posa[id(x[1])] * len(B) / max(1, len(A)))
    return out, same


class ContractTool:
    def __init__(self, docs_dir, cache=CACHE):
        os.makedirs(cache, exist_ok=True); self.cache, self.units, self.docs, self.errors, self._emb = cache, [], {}, {}, None
        self.files = {}
        for root, _, fs in os.walk(docs_dir):
            for f in sorted(fs):
                p = os.path.join(root, f); name = os.path.relpath(p, docs_dir)
                try: us = self._parsed(p)
                except Exception as e: self.errors[name] = f"{type(e).__name__}: {e}"; continue
                for u in us: u.doc = name
                self.files[name] = p; self.docs[name] = list(range(len(self.units), len(self.units) + len(us))); self.units += us
        self.bm25 = BM25([toks(search_text(u)) for u in self.units])

    def _key(self, p):
        return hashlib.sha1(open(p, "rb").read()).hexdigest()

    def _parsed(self, p):
        c = os.path.join(self.cache, self._key(p) + ".units.pkl")
        if os.path.exists(c): return pickle.load(open(c, "rb"))
        us = parse(p); tmp = f"{c}.{os.getpid()}"; pickle.dump(us, open(tmp, "wb")); os.replace(tmp, c); return us  # atomic: parallel runs share the cache

    def emb(self):
        """unit embeddings, cached per file."""
        if self._emb is None:
            parts = []
            for name, idx in self.docs.items():
                c = os.path.join(self.cache, self._key(self.files[name]) + ".bge-small.npy")
                if os.path.exists(c): parts.append(np.load(c)); continue
                e = model().encode([search_text(self.units[i])[:2000] for i in idx], batch_size=64, normalize_embeddings=True) if idx else np.zeros((0, 384), np.float32)
                tmp = f"{c}.{os.getpid()}.npy"; np.save(tmp, e.astype(np.float32)); os.replace(tmp, c); parts.append(e.astype(np.float32))
            self._emb = np.concatenate(parts) if parts else np.zeros((0, 384), np.float32)
        return self._emb

    # ---------- helpers ----------
    def doc(self, name):
        """resolves a document name the way an agent writes it (path, case, missing extension, small typos)."""
        if not name: return None
        n = os.path.basename(name.strip().strip("'\"")); names = list(self.docs)
        for d in names:
            if d.lower() == n.lower() or os.path.splitext(d)[0].lower() == os.path.splitext(n)[0].lower(): return d
        m = difflib.get_close_matches(n.lower(), [d.lower() for d in names], 1, 0.6)
        if m: return names[[d.lower() for d in names].index(m[0])]
        raise KeyError(name)

    def rank(self, query, idx):
        """unit indices `idx` ordered by relevance to `query` (BM25 + dense, reciprocal-rank fusion)."""
        idx = np.asarray(idx)
        if not len(idx): return []
        bm = self.bm25.scores(toks(query))[idx]
        qv = model().encode(["Represent this sentence for searching relevant passages: " + query], normalize_embeddings=True)[0]
        de = self.emb()[idx] @ qv
        r = np.zeros(len(idx))
        for s in (bm, de):
            order = np.argsort(-s); rr = np.empty(len(s)); rr[order] = np.arange(len(s)); r += 1 / (60 + rr)
        return list(idx[np.argsort(-r)])

    def _body(self, u):
        return u.marked if u.changes else u.text

    # ---------- actions ----------
    def find_units(self, query, document=None, k=6):
        """[(unit index, [unit indices shown])], best first."""
        idx = self.docs[document] if document else range(len(self.units))
        idx = [i for i in idx if self.units[i].text or self.units[i].before]
        out, seen, per = [], set(), collections.Counter()
        for i in self.rank(query, idx):
            if i in seen: continue
            u, shown = self.units[i], [i]
            if (u.heading or len(u.text.split()) < 6) and i + 1 < len(self.units) and self.units[i + 1].doc == u.doc:
                shown.append(i + 1)  # a heading or a stub: show the paragraph after it
            if any(j in seen for j in shown) or per[(u.doc, u.path)] >= 2: continue  # at most 2 passages per section / sheet
            seen.update(shown); out.append((i, shown)); per[(u.doc, u.path)] += 1
            if len(out) == k: break
        return out

    def find(self, query, document=None):
        if not query: return "Error: find needs a query (a topic, clause, fact, number or defined term)."
        hits = self.find_units(query, document); q = toks(query); lines, n = [], 0
        for r, (i, shown) in enumerate(hits, 1):
            body = " ".join(self._body(self.units[j]) for j in shown)
            s = f"[{r}] {self.units[i].where()} (¶{self.units[i].i})\n{snippet(body, q)}"
            if n + len(s) > BUDGET and lines: break
            lines.append(s); n += len(s)
        where = f" in {document}" if document else f" in {len(self.docs)} documents"
        return f'find "{query}"{where}: top {len(lines)} passages\n' + "\n".join(lines)

    def changes_items(self, document):
        return [i for i in self.docs[document] if self.units[i].changes]

    def changes(self, document, query=None, offset=0):
        if not document: return "Error: changes needs a document (a .docx with tracked changes)."
        items = self.changes_items(document)
        if not items:
            red = [d for d in self.docs if self.changes_items(d)]
            return (f"{document} has no tracked changes (a clean document). To see how it differs from another version, use "
                    f"action=compare with against=<the other version>." + (f" Documents with tracked changes: {', '.join(red)}." if red else ""))
        ch = [c for i in items for c in self.units[i].changes]
        nins, ndel = sum(c[0] == "ins" for c in ch), sum(c[0] == "del" for c in ch)
        who = collections.Counter(c[2] for c in ch if c[2]); dates = sorted({c[3] for c in ch if c[3]})
        if query: items = self.rank(query, items)
        head = (f"tracked changes in {document}: {nins} insertions, {ndel} deletions in {len(items)} paragraphs"
                + (f"; by {', '.join(f'{a} ({n})' for a, n in who.most_common(3))}" if who else "")
                + (f"; dated {dates[0]}" + (f" to {dates[-1]}" if len(dates) > 1 else "") if dates else "")
                + (f"; most relevant to \"{query}\" first" if query else "") + ". [-deleted-] {+inserted+}")
        return self._page(head, [(i, change_snippet(self.units[i].marked)) for i in items], offset)

    def _page(self, head, items, offset, local=None):
        lines, n = [], 0
        for r, (i, s) in enumerate(items[offset:], offset + 1):
            u = (self.units[self.docs[local][i]] if local else self.units[i]) if isinstance(i, int) else i
            line = f"[{r}] {u.where()} (¶{u.i}): {s}"
            if n + len(line) > BUDGET and lines: break
            lines.append(line); n += len(line)
        end = offset + len(lines)
        foot = f"(showing {offset + 1}-{end} of {len(items)}" + (f"; offset={end} for more)" if end < len(items) else ")")
        return head + "\n" + "\n".join(lines) + "\n" + foot

    def compare_items(self, older, newer):
        return diff_units([self.units[i] for i in self.docs[older] if self.units[i].text], [self.units[i] for i in self.docs[newer] if self.units[i].text])

    def compare(self, document, against, query=None, offset=0):
        if not document or not against: return "Error: compare needs document (the older version) and against (the newer version)."
        items, same = self.compare_items(document, against)
        c = collections.Counter(k for k, *_ in items)
        head = (f"compare {document} (older) -> {against} (newer): {c['changed']} paragraphs changed, {c['added']} added, "
                f"{c['removed']} removed, {same} unchanged. [-deleted-] {{+inserted+}}")
        if same < 0.2 * (same + len(items)):
            head += " Few paragraphs match: these look like different documents rather than two versions (matched by similarity)."
        if query:
            gi = {(u.doc, u.i): n for n, u in enumerate(self.units)}
            order = self.rank(query, [gi[(u.doc, u.i)] for _, u, *_ in items])
            rank = {i: r for r, i in enumerate(order)}; items = sorted(items, key=lambda x: rank[gi[(x[1].doc, x[1].i)]])
            head += f' Most relevant to "{query}" first.'
        return self._page(head, [(u, f"({k}) " + change_snippet(m)) for k, u, m, _, _ in items], offset)

    def show(self, document, query, offset=0):
        """paragraphs in full: query = a ¶ range ("¶62-73", "62-73", "62") or a clause number ("4.2" -> it and its items);
        no query: the document from the top (or from `offset`), one page at a time (2026-10-06: agents call it that way)."""
        if not document: return "Error: show needs a document (and optionally a ¶ range like 62-73, or a clause number like 4.2)."
        us = [self.units[i] for i in self.docs[document]]
        if not (query or "").strip():
            return self._page(f"{document}, {len(us)} paragraphs", [(u.i, self._body(u)) for u in us], int(offset or 0), local=document)
        q = query.strip().lstrip("¶§").strip()
        m = re.fullmatch(r"(\d+)\s*(?:-|to|–)\s*¶?(\d+)", q)
        if m and not re.fullmatch(r"\d+\.\d+.*", q): sel = us[int(m.group(1)):int(m.group(2)) + 1]
        elif re.fullmatch(r"\d+", q) and int(q) < len(us) and not any(u.label == q for u in us): sel = us[int(q):int(q) + 1]
        else:
            k = next((n for n, u in enumerate(us) if u.label == q or u.label.lower() == q.lower()), None)
            if k is None: return f"Error: no ¶ or clause {query} in {document}; use action=outline to see its clause numbers."
            sel = [us[k]] + [u for u in itertools.takewhile(lambda u: u.label.startswith(q) or not u.label and not u.heading, us[k + 1:])]
        return self._page(f"{document}, {len(sel)} paragraphs", [(u.i, self._body(u)) for u in sel], 0, local=document)

    def outline(self, document=None):
        if not document:
            lines = []
            for d, idx in self.docs.items():
                us = [self.units[i] for i in idx]; words = sum(len(u.text.split()) for u in us)
                ch = [c for u in us for c in u.changes]
                title = next((u.text for u in us if u.heading or u.kind == "head"), us[0].text if us else "")[:90]
                lines.append(f"- {d}: {len(us)} {'rows' if d.endswith(('.xlsx', '.xlsm')) else 'paragraphs'}, ~{int(words * 1.35):,} tokens"
                             + (f", tracked changes: {sum(c[0] == 'ins' for c in ch)} insertions / {sum(c[0] == 'del' for c in ch)} deletions" if ch else "")
                             + (f"; {title}" if title else ""))
            lines += [f"- {d}: could not be parsed ({e})" for d, e in self.errors.items()]
            return f"{len(self.docs)} documents\n" + "\n".join(lines)
        us = [self.units[i] for i in self.docs[document]]
        if document.endswith((".xlsx", ".xlsm")):
            sheets = collections.OrderedDict()
            for u in us: sheets.setdefault(u.path[0] if u.path else "", []).append(u)
            return f"{document}: " + "; ".join(f"{s} ({len(v)} rows): {v[0].text[:300]}" for s, v in sheets.items())
        heads = [u for u in us if u.heading] or [u for u in us if u.label and re.fullmatch(r"\d+(\.\d+)*|[A-Z][a-z]+ \S+", u.label)]
        lines, n = [], 0
        for u in heads:
            s = f"¶{u.i} " + (u.text if u.heading else f"§{u.label} {u.text[:80]}")[:110] + (" *" if any(x.changes for x in us[u.i:u.i + 1]) else "")
            if n + len(s) > BUDGET: lines.append(f"... ({len(heads) - len(lines)} more headings)"); break
            lines.append(s); n += len(s)
        nch = sum(1 for u in us if u.changes)
        return (f"{document}: {len(us)} paragraphs, {len(heads)} headings" + (f", {nch} paragraphs with tracked changes (see action=changes)" if nch else "")
                + "\n" + "\n".join(lines))

    def call(self, action, query="", document="", against="", offset=0):
        """the tool's answer as text (errors included), as an agent sees it."""
        try:
            d = self.doc(document) if document else None; a = self.doc(against) if against else None
        except KeyError as e:
            return f"Error: no document named {e}. Documents: {', '.join(self.docs)}"
        try:
            if action in ("find", "ask"): return self.find(query, d)
            if action == "changes": return self.changes(d, query or None, int(offset or 0))
            if action == "compare": return self.compare(d, a, query or None, int(offset or 0))
            if action in ("outline", "checklist"): return self.outline(d)
            if action == "show": return self.show(d, query, int(offset or 0))
        except Exception as e:  # a tool must answer, not crash the agent
            return f"Error: {action} failed: {type(e).__name__}: {e}"
        return "Error: action must be one of find, show, changes, compare, outline."


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("docs"); ap.add_argument("action")
    for o in ("query", "document", "against"): ap.add_argument("--" + o, default="")
    ap.add_argument("--offset", type=int, default=0); a = ap.parse_args()
    print(ContractTool(a.docs).call(a.action, a.query, a.document, a.against, a.offset))
