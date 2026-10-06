"""Idea 5 (2026-10-03): the questions the grammar leaves untyped.
  a. first person ("Am I allowed to ...", "Do I have to ..."): who "I" is isn't known, so the question is asked once
     per party the document names and answered only if every reading gives the same answer (supervaluation)
  b. two actions ("Can the Corporation reimburse X or pay Y?"): split at the "or"/"and" joining two verbs (spaCy's
     parse); "or" -> yes if a part is yes, no if every part is no; "and" -> yes if every part is yes, no if a part is no
  c. a learned parser (TF-IDF + logistic regression) trained on the questions the grammar types, predicting the
     question's type (asks) and action concept for the rest, so the priority check and the typed check can run there."""
from __future__ import annotations

import functools
import re

from router.pretier0 import ROLES

_ROLE_WORDS = {r.lower() for r in ROLES}
_FIRST = re.compile(r"(?<![\w'])(?:I|I'm|I've)(?![\w'])|\b(?:me|my|myself|mine)\b")
_NOT_PARTY = {"party", "parties", "agreement", "company's", "section", "exhibit", "schedule"}


def first_person(q: str) -> bool:
    return bool(_FIRST.search(q))


def candidate_parties(parties: list, named: list, limit: int = 3) -> list:
    out = []
    for p in list(parties) + list(named):
        p = p.strip()
        if not p or p.lower() in _NOT_PARTY or p.lower() in (x.lower() for x in out): continue
        out.append(p)
        if len(out) >= limit: break
    return out


def as_party(q: str, party: str) -> str:
    s = f"the {party}" if party.lower() in _ROLE_WORDS or party.lower().endswith(("ee", "or", "er", "ant", "ent")) else party
    rules = [(r"\bAm I\b", f"Is {s}"), (r"\bam I\b", f"is {s}"), (r"\bDo I\b", f"Does {s}"), (r"\bdo I\b", f"does {s}"),
             (r"\bHave I\b", f"Has {s}"), (r"\bhave I\b", f"has {s}"), (r"\bWas I\b", f"Was {s}"),
             (r"\bI am\b", f"{s} is"), (r"\bI'm\b", f"{s} is"), (r"\bI have\b", f"{s} has"), (r"\bI've\b", f"{s} has"),
             (r"\bI do\b", f"{s} does"), (r"(?<![\w'])I(?![\w'])", s), (r"\bmy\b", f"{s}'s"), (r"\bmyself\b", "itself"),
             (r"\bme\b", s), (r"\bmine\b", f"{s}'s")]
    for rx, rep in rules:
        q = re.sub(rx, rep, q)
    return q[:1].upper() + q[1:]


@functools.lru_cache(maxsize=1)
def _parser():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


def split_two(q: str):
    """("or" | "and", [q1, q2]) when two verbs are joined, else None."""
    doc = _parser()(q.rstrip(" ?"))
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None: return None
    heads = [root] + [t for t in root.children if t.dep_ == "xcomp" and t.pos_ == "VERB"]
    for h in heads:
        for c in h.children:
            if c.dep_ == "conj" and c.pos_ == "VERB":
                cc = next((x for x in h.children if x.dep_ == "cc" and x.i < c.i and x.lower_ in ("or", "and")), None)
                if cc is None: continue
                prefix = re.sub(r"\s+either\b", "", doc[:h.i].text)  # "Can the Corporation also"
                has_obj = any(x.dep_ in ("dobj", "obj", "prep", "dative") and x.i < cc.i for x in h.children)
                if has_obj:
                    q1 = re.sub(r"\s+either\b", "", doc[:cc.i].text).rstrip(" ,") + "?"
                else:  # "give or destroy the records": the object is shared
                    q1 = f"{prefix} {h.text} {doc[c.i + 1:].text}?"
                q2 = f"{prefix} {doc[c.i:].text}?"
                return cc.lower_, [q1, q2]
    return None


def combine(conn: str, answers: list):
    if conn == "or":
        if any(a == "yes" for a in answers): return "yes"
        if answers and all(a == "no" for a in answers): return "no"
        return None
    if all(a == "yes" for a in answers): return "yes"
    if any(a == "no" for a in answers): return "no"
    return None


class NeuralParser:
    """asks and action concept from the question's words, trained on the grammar's frames."""
    def fit(self, questions: list, frames: list):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import hstack
        self.vw = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(questions)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(questions)
        X = hstack([self.vw.transform(questions), self.vc.transform(questions)]).tocsr()
        self.asks = LogisticRegression(C=5, max_iter=3000).fit(X, [f["asks"] for f in frames])
        acts = [self._act(f) for f in frames]
        self.act = LogisticRegression(C=5, max_iter=3000).fit(X, acts)
        return self

    @staticmethod
    def _act(f):
        a = (f.get("actions") or [f.get("action")] or [None])[0]
        return "OPEN" if not a or ":" in a else a

    def predict(self, questions: list):
        from scipy.sparse import hstack
        X = hstack([self.vw.transform(questions), self.vc.transform(questions)]).tocsr()
        pa, pc = self.asks.predict_proba(X), self.act.predict_proba(X)
        out = []
        for a, c in zip(pa, pc):
            out.append({"asks": self.asks.classes_[a.argmax()], "p_asks": float(a.max()),
                        "action": self.act.classes_[c.argmax()], "p_action": float(c.max())})
        return out
