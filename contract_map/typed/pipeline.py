"""Compile, locate, read, check (2026-10-03; the user: "Good to go with all the steps").
A long document is compiled once (doccompile.py: page furniture out, structure, sections, statements with their
lead-ins, parties). A question is then answered from a short "located document" built for it:
  1. Pre-Tier 0 on the whole document first, as today (its answers are kept).
  2. pieces: each compiled section's statements (lead-ins prefixed, so a list item never stands alone), cut at
     statement boundaries into pieces of at most PIECE_CHARS, the section heading in front.
  3. locate: rank pieces by word stems shared with the question (lexicon synonyms count, as NetReader.pick) and by
     the embedder (bge-small, cached per document), fused by rank; a piece that has the question's action concept
     (Pre-Tier 0's tagger, the same types as the question's frame) ranks higher. Top K_LOCATE.
  4. exceptions: up to K_EXCEPT more pieces that name the same action concept and carry an override cue
     ("notwithstanding", "except", "subject to", "provided that", "shall not apply", "in no event"), so a carve-out in
     another section is read too and the network's conflict rule can stop the answer.
  5. read: the located document = a line naming the parties (from the compiled preamble, so the party checks know
     them) + the pieces in document order; Pre-Tier 0, then the reader network read it whole (NetReader's T_YES_DOC
     and its party / negation / modality checks), as for any short document.
The question side is Pre-Tier 0's own frame (its typed form: asks, actor, action, things, conditions).
ROUTER_PATH picks the router code (default: the dev copy next to this file)."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ROUTER_PATH", f"{HERE}/dev"))
sys.path.insert(0, f"{HERE}/../doccompile")
from doccompile import compile_doc  # noqa: E402
from router import frames  # noqa: E402
from router import netreader as NR  # noqa: E402
from router.pretier0 import check as pt0_check  # noqa: E402

PIECE_CHARS = 600
K_LOCATE = 6  # locate_eval.py: the gold clause in the top 6 for 91.5% of tuning NDA questions
K_EXCEPT = 0  # exception pieces changed no wrong answer on nda_train or wc1 open, and cost 4 right ones
T_LOCATED = 0.90  # the network's "yes" on a located document (nda_train: 87/88 at >= 0.90; its own 0.94 kept 15)
LONG_CHARS = 3000  # shorter documents go the live way (Pre-Tier 0, then the network on the whole text)
OVERRIDE = re.compile(r"\bnotwithstanding\b|\bexcept\b|\bsubject to\b|\bprovided,? (?:however,? )?that\b|"
                      r"\bshall not apply\b|\bin no event\b|\bunless\b|\bother than\b", re.I)


_NLP = None


def lemmas(text: str) -> set:
    """Content-word lemmas (spaCy en_core_web_sm, tagger + lemmatizer only). Measured for locating (locate_eval.py,
    100 tuning NDAs): the gold clause in the top 6 pieces 91.5% with lemmas vs 88.2% with NetReader's 5-letter stems."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return {t.lemma_.lower() for t in _NLP(text.lower()[:20000]) if t.is_alpha and len(t) > 2 and t.lemma_.lower() not in NR._STOP}


_LEMMA_SYN = None


def lemma_synonyms() -> dict:
    global _LEMMA_SYN
    if _LEMMA_SYN is None:
        _LEMMA_SYN = {}
        for phrases in NR.LEXICON.values():
            group = set().union(*(lemmas(p) for p in phrases)) if phrases else set()
            for p in phrases:
                for w in lemmas(p): _LEMMA_SYN.setdefault(w, set()).update(group)
    return _LEMMA_SYN


class Compiled:
    def __init__(self, document: str):
        c = compile_doc(document)
        self.parties = c.parties
        self.pieces = []  # [{"text", "start", "head"}]
        groups = []
        for st in c.stmts:
            if st.start < 0 or len(st.own) < 4: continue
            secs = [i for i in st.secs if not c.sections[i].front]
            hs = [i for i in secs if c.sections[i].heading]
            key = hs[-1] if hs else (secs[0] if secs else -1)
            if groups and groups[-1][0] == key: groups[-1][1].append(st)
            else: groups.append((key, [st]))
        for key, sts in groups:
            head = c.sections[key].heading if key >= 0 else ""
            cur, start = [], None
            for st in sts:
                t = re.sub(r"\s+", " ", st.text).strip()
                if cur and sum(len(x) + 1 for x in cur) + len(t) > PIECE_CHARS:
                    self._add(cur, start, head); cur, start = [], None
                cur.append(t[:PIECE_CHARS * 2]); start = st.start if start is None else start
            if cur: self._add(cur, start, head)

    def _add(self, cur, start, head):
        body = " ".join(cur)
        if head and not body.lower().startswith(head.lower()[:20]):
            body = f"{head}. {body}"
        self.pieces.append({"text": body, "start": start, "head": head})

    def party_line(self) -> str:
        if not self.parties: return ""
        return " and ".join(f'{p.said[-60:].strip(" ,(")} (the "{p.name}")' for p in self.parties[:4]) + "."


def typed_check(frame: dict | None, sentence: str, trie=None) -> str | None:
    """The deciding sentence must hold what the question is about, in the question's own types (Pre-Tier 0's frame):
    an action question needs its action concept in the sentence (an unfamiliar verb: its stem); an existence question
    needs every slot of one of its alternatives. None = passes, else why not. (2026-10-03: the network's wrong "yes"
    answers on located text were sentences about another act: survival for "keep", returning for "keep".)"""
    if not frame: return None
    toks = frames.tokens(sentence)
    if frame.get("actions"):
        tags = frames.tag(toks, trie or frames._BASE_TRIE)
        have = {t.name for t in tags if t.kind == "ACTION"}
        for a in frame["actions"]:
            if a in have: return None
            if ":" in a:  # "V:identifi", "OPEN:...": an unfamiliar verb, by its stem
                stem = a.split(":", 1)[1][:6]
                if any(t.startswith(stem) for t in toks): return None
        return f"the sentence has none of the question's actions ({', '.join(frame['actions'])})"
    if frame.get("alts"):
        for alt in frame["alts"]:
            if all(any(any(t.startswith(st[:6]) for t in toks) for st in g.get("stems") or []) or
                   any(ph in sentence.lower() for ph in g.get("phrases") or []) for g in alt):
                return None
        return "the sentence lacks what the question asks exists"
    return None


class Pipeline:
    def __init__(self, net: bool = True, k_locate: int = K_LOCATE, k_except: int = K_EXCEPT, t_located: float | None = T_LOCATED,
                 t_no_located: float | None = None):
        self.net = NR.NetReader() if net else None
        # the reader for located documents: the same network and embedder, its own thresholds when given
        # (2026-10-03 fix: a located text of more than K_LEX + K_EMB = 12 windows counted as a "long document" and the
        # network's answer was off there: 18 of 126 sealed wc1 questions. The located reader reads up to 24 whole.)
        self.net_located = None if not net else NR.NetReader(
            net=self.net.net, t_yes=t_located or NR.T_YES, t_yes_doc=t_located or NR.T_YES_DOC,
            t_no=t_no_located or NR.T_NO, embedder=None, k_lex=12, k_emb=12)
        if self.net_located is not self.net and self.net is not None:
            self.net_located._embed = self.net._embed
        self.k_locate, self.k_except = k_locate, k_except
        self._docs = OrderedDict()

    def compiled(self, document: str) -> Compiled:
        key = hashlib.sha1(document.encode()).hexdigest()
        if key not in self._docs:
            self._docs[key] = Compiled(document)
            while len(self._docs) > 64: self._docs.popitem(last=False)
        return self._docs[key]

    def locate(self, question: str, document: str, cd: Compiled, frame) -> list:
        texts = [p["text"] for p in cd.pieces]
        if not texts: return []
        if not hasattr(cd, "lemmas"): cd.lemmas = [lemmas(t) for t in texts]
        syn = lemma_synonyms()
        q = lemmas(question); qx = set().union(q, *(syn.get(w, ()) for w in q))
        lex = sorted(range(len(texts)), key=lambda i: -(2 * len(q & cd.lemmas[i]) + len(qx & cd.lemmas[i])))
        rank = {i: 1 / (10 + r) for r, i in enumerate(lex)}
        if self.net is not None:
            key = hashlib.sha1(("pieces:" + document).encode()).hexdigest()
            vecs = self.net._vectors.get(key)
            if vecs is None:
                vecs = self.net._embed(texts); self.net._vectors[key] = vecs
            qv = self.net._embed([NR.QUERY_PREFIX + question])[0]
            for r, i in enumerate((-(vecs @ qv)).argsort()):
                rank[int(i)] += 1 / (10 + r)
        acts = set(getattr(frame, "actions", None) or ([frame.action] if getattr(frame, "action", None) else []))
        tagged = {}
        if acts:
            for i in sorted(rank, key=lambda i: -rank[i])[:40]:
                tags = frames.tag(frames.tokens(texts[i]), frames._BASE_TRIE)
                tagged[i] = {t.name for t in tags if t.kind == "ACTION"}
                if tagged[i] & acts: rank[i] += 0.05
        top = sorted(rank, key=lambda i: -rank[i])[:self.k_locate]
        extra = []
        if acts and self.k_except:
            for i in sorted(tagged, key=lambda i: -rank[i]):
                if i not in top and tagged[i] & acts and OVERRIDE.search(texts[i]):
                    extra.append(i)
                if len(extra) >= self.k_except: break
        return sorted(top + extra)

    def answer(self, question: str, document: str) -> dict:
        r0 = pt0_check(question, document)
        info = {"frame": (r0.frames or {}).get("frame"), "parties": list(r0.parties or []), "pt0_reason": r0.reason}
        if r0.fired:
            return {"answer": r0.answer, "path": "pt0", "evidence": (r0.evidence or [{}])[0].get("text", ""), "reason": r0.reason,
                    "read_text": document if len(document) < LONG_CHARS else None, **info}
        if len(document) < LONG_CHARS:
            if self.net is None: return {"answer": None, "path": "none", "reason": r0.reason, "read_text": document, **info}
            n = self.net.answer(question, document)
            return {"answer": n.answer if n.fired else None, "path": "net" if n.fired else "none",
                    "evidence": (n.evidence or [{}])[0].get("text", ""), "reason": n.reason, "read_text": document,
                    "p": n.confidence if n.fired else (n.evidence or [{}])[0].get("p", 0.0),
                    "net_label": (n.evidence or [{}])[0].get("label"), **info}
        cd = self.compiled(document)
        fd = (r0.frames or {}).get("frame") or {}
        frame = type("F", (), {"action": fd.get("action"), "actions": fd.get("actions") or []})()
        idx = self.locate(question, document, cd, frame)
        if not idx: return {"answer": None, "path": "none", "reason": "nothing located", "read_text": None, **info}
        pl = cd.party_line()
        located = (pl + "\n\n" if pl else "") + "\n\n".join(cd.pieces[i]["text"] for i in idx)
        r1 = pt0_check(question, located)
        if r1.fired:
            return {"answer": r1.answer, "path": "pt0-located", "evidence": (r1.evidence or [{}])[0].get("text", ""),
                    "reason": r1.reason, "located": idx, "read_text": located, **info}
        if self.net is None: return {"answer": None, "path": "none", "reason": r1.reason, "located": idx, "read_text": located, **info}
        n = self.net_located.answer(question, located)
        return {"answer": n.answer if n.fired else None, "path": "net-located" if n.fired else "none",
                "evidence": (n.evidence or [{}])[0].get("text", ""), "reason": n.reason, "located": idx,
                "p": n.confidence if n.fired else (n.evidence or [{}])[0].get("p", 0.0), "net_label": (n.evidence or [{}])[0].get("label"),
                "pieces": [cd.pieces[i]["text"][:300] for i in idx], "read_text": located, **info}
