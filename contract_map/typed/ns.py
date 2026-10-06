"""Neuro-symbolic checks on top of the pipeline (2026-10-03; the user: "Go with 1,2,3,4,5! Try different combinations").
  1. consistency (ConCoRD-style, without a solver library): from the question's typed form, related questions are
     written by rule ("Can S V R?" <-> "Is S prohibited from V-ing R?", a paraphrase), the network reads the same text
     for each, and an answer must agree with them.
  2. priorities (defeasible / Catala-style default logic): every statement of the compiled document becomes norms
     (actor, action concept, permit / oblige / ban), with its override markers ("notwithstanding"), deferrals ("subject
     to Section X", resolved by the compiler's cross-references) and exceptions ("except as provided in Section X",
     "unless", "provided that"). An answer's statement must not lose to a conflicting norm; a ban nobody excepts is a
     candidate "no".
Feature functions only: the decisions are made by the combinations in combos.py (hand rules) and gates.py (learned)."""
from __future__ import annotations

import functools
import re
import sys
from dataclasses import dataclass, field

import pipeline as P
from pipeline import NR, frames
from router import qshapes

sys.path.insert(0, f"{P.HERE}/../doccompile")
from doccompile import compile_doc  # noqa: E402

YES, NO = NR.YES, NR.NO


# ---------- 1. related questions ----------
def _words(text: str) -> list:
    """Words as written (case kept), aligned with frames.tokens(text)."""
    t2 = re.sub(r"'s\b", "", text.replace("’", "'"))
    return [t2[m.start():m.end()] for m in frames._TOKEN.finditer(t2.lower())]


@functools.lru_cache(maxsize=4096)
def base_form(word: str) -> str:
    """A verb's base form, read in a verb's place ("engineering" alone is a noun to the lemmatizer)."""
    P.lemmas("x")  # loads spaCy
    doc = P._NLP(f"they are {word.lower()} it") if word.lower().endswith("ing") else P._NLP(f"they {word.lower()} it")
    return doc[2].lemma_ if word.lower().endswith("ing") and len(doc) > 2 else (doc[1].lemma_ if len(doc) > 1 else word.lower())


def related(question: str, frame: dict | None, parties: list) -> list:
    """[(kind, question)]: "ban" / "perm" (the opposite modality: an answer "yes" to one makes the other "no") and
    "para" (the same question in other words: the answers must agree)."""
    if not frame or frame.get("asks") not in ("CAN", "MUST", "PROHIBITED"):
        return []
    canon = qshapes.canonical(question).rstrip(" ?")
    if frame["asks"] == "PROHIBITED":
        m = re.match(r"^(?:Is|Are)\s+(?P<s>.+?)\s+prohibited\s+from\s+(?P<v>(?:\w+\s+)?\w+ing)\b(?P<r>.*)$", canon)
        if not m:
            return []
        s, ger, r = m.group("s"), m.group("v"), m.group("r")
        head, last = (ger.rsplit(" ", 1) if " " in ger else ("", ger))
        v = (head + " " if head else "") + base_form(last)
        return [("perm", f"Can {s} {v}{_or_base(r)}?"), ("para", f"Is {s} barred from {ger}{r}?")]
    trie = frames._build_trie(frames._party_lexicon(parties)) if parties else frames._BASE_TRIE
    toks, words = frames.tokens(canon), _words(canon)
    if len(toks) != len(words) or len(toks) < 3:
        return []
    tags = frames.tag(toks[1:], trie)
    actor = next((t for t in tags if t.kind == "ACTOR"), None)
    if actor is None:
        return []
    end = (actor.end if actor.end > 0 else actor.start + 1)
    k = end
    while k < len(toks) - 1 and (toks[1 + k] in frames._PREFACE or words[1 + k].lower().endswith("ly")
                                 or words[1 + k].lower() in ("be", "also", "still")):
        k += 1
    if 1 + k >= len(words):
        return []
    s = " ".join(words[1:1 + end]); v = words[1 + k]; r = (" " + " ".join(words[2 + k:])) if 2 + k < len(words) else ""
    adv = " ".join(w for w in words[1 + end:1 + k] if w.lower().endswith("ly"))  # "independently develop": kept
    adv = adv + " " if adv else ""
    vb = base_form(v)
    if frame["asks"] == "CAN":
        return [("ban", f"Is {s} prohibited from {adv}{gerund(vb)}{_or_gerund(r)}?"), ("para", f"Is {s} permitted to {adv}{vb}{r}?")]
    return [("ban", f"Is {s} prohibited from {adv}{gerund(vb)}{_or_gerund(r)}?"), ("para", f"Is {s} required to {adv}{vb}{r}?")]


_DOUBLE = {"permit", "refer", "transfer", "commit", "omit", "submit", "admit", "occur", "prefer", "control", "compel",
           "regret", "forget", "begin", "equip", "remit", "acquit", "expel", "rebut", "recur", "abet"}


def gerund(verb: str) -> str:
    """"share" -> "sharing", "transfer" -> "transferring", "develop" -> "developing" (no doubling past one syllable
    unless the verb is a known stress-final one)."""
    v = verb.lower()
    if v.endswith("ie"): return v[:-2] + "ying"
    if v.endswith("e") and not v.endswith(("ee", "ye", "oe")): return v[:-1] + "ing"
    syll = len(re.findall(r"[aeiouy]+", v))
    if re.search(r"[^aeiou][aeiou][bdgmnprt]$", v) and (syll == 1 or v in _DOUBLE): return v + v[-1] + "ing"
    return v + "ing"


def _or_gerund(rest: str) -> str:
    """"return or destroy X" -> "returning or destroying X": the second verb of a pair takes the same form."""
    m = re.match(r"^(\s+or\s+)(\w+)\b(.*)$", rest)
    return f"{m.group(1)}{gerund(base_form(m.group(2)))}{m.group(3)}" if m else rest


def _or_base(rest: str) -> str:
    m = re.match(r"^(\s+or\s+)(\w+ing)\b(.*)$", rest)
    return f"{m.group(1)}{base_form(m.group(2))}{m.group(3)}" if m else rest


def net_probs(net, question: str, text: str) -> tuple:
    """The network's highest p(yes) and p(no) over the text's units (no checks)."""
    us = NR.units(text)
    if not us:
        return 0.0, 0.0
    probs = net.net([(u, question) for u in us])
    return max(p[YES] for p in probs), max(p[NO] for p in probs)


# ---------- 2. norms and priorities ----------
_NOTWITH = re.compile(r"\bnotwithstanding\b", re.I)
_SUBJECT_TO = re.compile(r"\bsubject\s+to\b", re.I)
_EXCEPT_REF = re.compile(r"\b(?:except|other\s+than|save)\s+(?:as\s+)?(?:otherwise\s+)?(?:expressly\s+)?"
                         r"(?:provided|set\s+forth|permitted|described|contemplated)\s+(?:in|under|by)\b", re.I)
_EXCEPTION = re.compile(r"\bexcept\b|\bunless\b|\bprovided,?\s+(?:however,?\s+)?that\b|\bother\s+than\b|\bsave\s+(?:as|for)\b", re.I)
_BAN_WORDS = re.compile(r"\b(?:shall|will|may|must|can)\s+not\b|\bcannot\b|\bnever\b|\bno\s+\w+\s+shall\b|\bprohibited\b|"
                        r"\bforbidden\b|\bin\s+no\s+event\b|\bneither\b.{0,80}\bnor\b", re.I)


@dataclass
class Norm:
    stmt: int
    text: str
    start: int
    actions: set
    actors: set
    polarity: str            # "permit" | "oblige" | "ban" | ""
    override: bool = False   # notwithstanding ...
    defers: set = field(default_factory=set)      # section ids it is "subject to"
    except_secs: set = field(default_factory=set)  # sections it excepts ("except as provided in Section X")
    inline_exception: bool = False
    secs: tuple = ()


def _polarity(tags: list, raw: list, action) -> str:
    before = [t for t in tags if t.start < action.start and action.start - t.start <= 4]  # the modal run right before the act
    names = {t.name for t in before if t.kind == "MODAL"}
    if names & {"NOT", "BAN", "NEITHER"} or frames._negative_subject(raw, max(0, action.start - 8), action.start):
        return "ban"
    if "MAY" in names:
        return "permit"
    if "MUST" in names:
        return "oblige"
    return ""


class DocNorms:
    def __init__(self, document: str, parties: list):
        c = compile_doc(document)
        self.c = c
        trie = frames._build_trie(frames._party_lexicon(parties)) if parties else frames._BASE_TRIE
        sec_xrefs = {}
        for x in c.xrefs:
            if x.kind == "section" and x.resolved:
                sec_xrefs.setdefault(x.start, x.target)
        xstarts = sorted(sec_xrefs)
        self.norms = []
        for i, st in enumerate(c.stmts):
            if st.start < 0:
                continue
            text = st.text
            toks, raw = frames.tokens(text), frames._raw_tokens(text)
            tags = frames.tag(toks, trie)
            acts = [t for t in tags if t.kind == "ACTION"]
            if not acts:
                continue
            actors = {t.name for t in tags if t.kind == "ACTOR"}
            refs = {sec_xrefs[s] for s in xstarts if st.start <= s < st.end}
            pol = _polarity(tags, raw, acts[0])
            n = Norm(i, text, st.start, {t.name for t in acts}, actors, pol, bool(_NOTWITH.search(st.own)),
                     secs=st.secs, inline_exception=bool(_EXCEPTION.search(st.own)))
            n.things = {t.name for t in tags if t.kind == "THING"}
            if _SUBJECT_TO.search(st.own): n.defers = refs
            if _EXCEPT_REF.search(st.own): n.except_secs = refs
            self.norms.append(n)

    def find(self, evidence: str):
        key = re.sub(r"\s+", " ", evidence).strip().lower()[:60]
        if not key:
            return None
        for n in self.norms:
            if key[:40] in re.sub(r"\s+", " ", n.text).lower():
                return n
        return None


_OPPOSITE = {("CAN", "yes"): "ban", ("PROHIBITED", "yes"): "permit", ("MUST", "yes"): "ban",
             ("CAN", "no"): "permit", ("PROHIBITED", "no"): "ban"}


def _actor_ok(asked, actors: set) -> bool:
    if not actors:
        return asked in (None, "ANY", "ALL")
    return any(frames._actor_matches(asked or "ANY", a) for a in actors)


def priority(dn: DocNorms, frame: dict | None, answer: str | None, evidence: str) -> dict:
    """What the document's other norms say about this answer, by priority (features)."""
    out = {"pr_has_frame": bool(frame and frame.get("actions")), "pr_ev_found": False, "pr_conflicts": 0,
           "pr_overridden": False, "pr_ev_overrides": False, "pr_defers_to_conflict": False, "pr_conflict_is_exception": False,
           "pr_unresolved": False, "pr_ban": 0, "pr_permit": 0, "pr_ban_unexcepted": False}
    if not out["pr_has_frame"]:
        return out
    acts, asked = set(frame["actions"]), frame.get("actor")
    q_things = {t[1] for t in frame.get("things") or [] if t[0] == "THING"} - {"AGREEMENT"}
    same = [n for n in dn.norms if n.actions & acts and _actor_ok(asked, n.actors)
            and (not q_things or getattr(n, "things", set()) & q_things)]  # the same act on the same thing (2026-10-03)
    out["pr_ban"] = sum(n.polarity == "ban" for n in same)
    out["pr_permit"] = sum(n.polarity == "permit" for n in same)
    bans = [n for n in same if n.polarity == "ban"]
    permits = [n for n in same if n.polarity == "permit"]
    # a ban nobody excepts: no permit for the act, no exception inside it, no "except as provided in" pointing anywhere
    out["pr_ban_unexcepted"] = bool(bans) and not permits and not any(n.inline_exception or n.except_secs for n in bans)
    if not answer:
        return out
    ev = dn.find(evidence) if evidence else None
    out["pr_ev_found"] = ev is not None
    want = _OPPOSITE.get((frame.get("asks"), answer))
    if not want:
        return out
    conflicts = [n for n in same if n.polarity == want and (ev is None or n.stmt != ev.stmt)]
    out["pr_conflicts"] = len(conflicts)
    if not conflicts:
        return out
    unresolved = False
    for n in conflicts:
        in_ev_secs = ev is not None and bool(set(n.secs) & set(ev.secs))
        if n.override and not (ev is not None and ev.override):
            out["pr_overridden"] = True; continue
        if ev is not None and ev.override:
            out["pr_ev_overrides"] = True; continue
        if ev is not None and n.secs and (set(n.secs) & ev.defers):
            out["pr_defers_to_conflict"] = True; continue
        if ev is not None and (n.secs and set(n.secs) & ev.except_secs or n.inline_exception and in_ev_secs):
            out["pr_conflict_is_exception"] = True; continue
        if ev is not None and (ev.inline_exception or ev.except_secs):
            continue  # the answer's own statement carries the exception: it is the more specific rule
        unresolved = True
    out["pr_unresolved"] = unresolved
    return out
