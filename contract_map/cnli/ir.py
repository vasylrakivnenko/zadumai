"""Compiler middle + back end for contracts (2026-10-03; overnight run, the user's instructions).
  1. preprocessor   canonical verbs ("reverse engineer" -> decompile, "make available" -> provide, "give notice" ->
                    notify ...), party roles (receiving / disclosing / both) and defined-term expansion, on a copy of
                    each statement (the original text is kept for quoting)
  2. grammar        each statement parsed (spaCy dependency tree) into predicates; Adams' categories of contract
                    language: obligation (shall/will/must/agrees to), prohibition (shall not/may not/agrees not to/
                    neither ... shall/prohibited from), permission (may/is entitled/free/permitted to), definition,
                    exclusion, declaration
  3. IR             Norm(kind, actor role, action class, verb, object, recipients, exceptions, conditions, time, text)
  4. scope          exceptions ("except", "other than", "unless", "provided that", "save", "but not") bind to the
                    statement's norms; "notwithstanding" makes a permission override the bans in scope; "without
                    (prior written) consent" is a consent gate, not a permission; lead-ins give list items their
                    modality (compiler.py)
  5. semantic passes  derived facts: exception -> permission ("shall not disclose ... except to X" => may disclose to X),
                    "only/solely to X" => may disclose to X, definition exclusions => may develop / may receive from
                    third parties, a return-or-destroy duty's proviso => may retain, the CI definition's membership
                    criteria => is marking necessary or only sufficient, ...
  6. queries        queries_ir.py: the 17 questions over the facts
Deterministic. What doesn't parse yields no norm (the queries abstain)."""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

from compiler import compile_doc

I = re.I

# ---------- 1. preprocessor ----------
VERB_MACROS = [
    (r"\breverse[- ]?engineer(?:ing|ed|s)?\b", "decompile"), (r"\bde-?compil(?:e|ing|ed|es)\b", "decompile"),
    (r"\bdisassembl(?:e|ing|ed|es)\b", "decompile"), (r"\bmake\s+(?:\w+\s+){0,3}?available\b", "provide"),
    (r"\bgive\s+(?:\w+\s+){0,3}?notice\b", "notify"), (r"\bprovide\s+(?:\w+\s+){0,3}?notice\b", "notify"),
    (r"\bkeep\s+(?:\w+\s+){0,4}?(?:confidential|secret|in (?:strict )?confidence)\b", "protect"),
    (r"\bhold\s+(?:\w+\s+){0,4}?in\s+(?:strict(?:est)?\s+)?(?:confidence|trust)\b", "protect"),
    (r"\bis\s+(?:hereby\s+)?prohibited\s+from\b", "shall not"), (r"\bare\s+(?:hereby\s+)?prohibited\s+from\b", "shall not"),
    (r"\bshall\s+refrain\s+from\b", "shall not"), (r"\bundertakes?\s+not\s+to\b", "shall not"),
    (r"\bcovenants?\s+not\s+to\b", "shall not"), (r"\bagrees?\s+not\s+to\b", "shall not"), (r"\bpromises?\s+not\s+to\b", "shall not"),
    (r"\bundertakes?\s+to\b", "shall"), (r"\bcovenants?\s+to\b", "shall"), (r"\bagrees?\s+to\b", "shall"), (r"\bis\s+obligated\s+to\b", "shall"),
    (r"\bis\s+required\s+to\b", "shall"), (r"\bis\s+(?:hereby\s+)?(?:permitted|entitled|free|allowed|authori[sz]ed)\s+to\b", "may"),
    (r"\bare\s+(?:hereby\s+)?(?:permitted|entitled|free|allowed|authori[sz]ed)\s+to\b", "may"), (r"\bshall\s+have\s+the\s+right\s+to\b", "may"),
    (r"\bin\s+no\s+event\s+shall\b", "shall not"),
    (r"\bmake\s+(?:\w+\s+){0,2}?(?:copies|copy|reproductions?|duplicates?)\b", "copy"),  # "may make copies" (train, 2026-10-03)
]
_MACROS = [(re.compile(p, I), r) for p, r in VERB_MACROS]
CI_ALIASES = r"\b(?:proprietary information|evaluation materials?|confidential materials?|confidential data|protected information|secret information|the information)\b"
RECV = (r"\b(?:recipient|receiving part(?:y|ies)|receiver|disclosee|transferee|you|each party|either party|both parties|the parties|"
        r"neither party|each of the parties|the undersigned|consultant|contractor|visitor|participant|bidder|licensee|mentor|"
        r"attendee|evaluator|proponent|investor|buyer|purchaser)\b")
DISC = r"\b(?:disclos(?:ing|er)\s*part(?:y|ies)|discloser|disclosing party|owner|provider|transmitter|originating party|licensor|company)\b"

ACTIONS = {
    "DISCLOSE": r"disclose|reveal|divulge|publish|communicate|provide|share|disseminate|distribute|transfer|furnish|give|release|deliver|transmit|discuss|show|furnish",
    "USE": r"use|utilize|utilise|exploit|employ",
    "COPY": r"copy|reproduce|duplicate|replicate|photocopy",
    "DECOMPILE": r"decompile|analyze|analyse|decompose|derive|deconstruct",
    "RETURN": r"return|destroy|erase|delete|purge|expunge|surrender|shred",
    "RETAIN": r"retain|keep|archive|maintain|store",
    "NOTIFY": r"notify|inform|advise|alert|tell",
    "SOLICIT": r"solicit|hire|employ|recruit|induce|entice|engage|persuade|offer",
    "SURVIVE": r"survive|continue|remain",
    "PROTECT": r"protect|safeguard|secure",
    "DEVELOP": r"develop|create|conceive|generate|invent",
    "ACQUIRE": r"receive|obtain|acquire|learn",
    "GRANT": r"grant|confer|transfer|license|assign",
}
_ACTION_RX = {k: re.compile(rf"^(?:{v})$", I) for k, v in ACTIONS.items()}
EXC = r"\bexcept\b|\bother than\b|\bunless\b|\bprovided\s*(?:,\s*)?(?:however|further)?\s*,?\s*that\b|\bprovided\s*,?\s*however\b|\bsave\s+(?:as|that|for)\b|\bbut not\b|\bexcluding\b|\bwith the exception of\b"
CONSENT = r"\bwithout\s+(?:the\s+)?(?:express\s+)?(?:prior\s+)?(?:written\s+)?(?:consent|approval|authori[sz]ation|permission)\b|\bwithout\s+first\s+obtaining\b"


def preprocess(text: str) -> str:
    t = text
    for rx, r in _MACROS: t = rx.sub(r, t)
    return t


@functools.lru_cache(maxsize=1)
def nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


# ---------- 3. IR ----------
@dataclass
class Norm:
    kind: str  # OBLIGATION | PROHIBITION | PERMISSION
    actor: str  # RECV | DISC | BOTH | OTHER | PASSIVE | UNKNOWN
    action: str  # ACTIONS key or OTHER
    verb: str
    obj: str
    recipients: str
    exceptions: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    consent_gate: bool = False
    overrides: bool = False  # "notwithstanding ..."
    text: str = ""  # the source statement (original wording)


@dataclass
class Facts:
    norms: list
    permit_recipients: list  # [(who (resolved), source text)] the receiving party may disclose CI to
    ci_body: list
    ci_exclusions: list
    marking: str | None  # "necessary" | "sufficient" | None (not discussed)
    marking_text: str
    stmts: list
    symbols: dict
    recv_terms: list


def _action(lemma: str) -> str:
    for k, rx in _ACTION_RX.items():
        if rx.match(lemma): return k
    return "OTHER"


def _role(subj: str, recv_terms) -> str:
    s = subj.lower()
    if not s.strip(): return "UNKNOWN"
    if re.search(r"\b(?:each|either|both|neither|the parties|a party|any party|each of)\b", s): return "BOTH"
    if any(t.lower() in s for t in recv_terms) or re.search(RECV, s, I): return "RECV"
    if re.search(DISC, s, I): return "DISC"
    return "OTHER"


def _span(tok) -> str:
    return " ".join(t.text for t in tok.subtree)


# ---------- 2. grammar: predicates of one parsed statement ----------
def predicates(doc, recv_terms) -> list:
    """[(kind, actor, action, verb, obj, recipients)] for the statement's governed verbs."""
    out = []
    for v in doc:
        if v.pos_ not in ("VERB", "AUX") and v.dep_ not in ("ROOT", "conj", "xcomp"): continue
        if v.dep_ not in ("ROOT", "conj", "xcomp", "ccomp", "relcl", "advcl", "pcomp"): continue
        # modality: aux of the verb, or of the verb it hangs on (conj/xcomp share it)
        chain = [v]; h = v
        while h.dep_ in ("conj", "xcomp") and h.head is not h:
            h = h.head; chain.append(h)
        auxes = [c.lower_ for x in chain for c in x.children if c.dep_ in ("aux", "auxpass")]
        neg = any(c.dep_ == "neg" for x in chain for c in x.children)
        subj_tok = next((c for x in chain for c in x.children if c.dep_ in ("nsubj", "nsubjpass", "csubj")), None)
        subj = _span(subj_tok) if subj_tok is not None else ""
        if re.match(r"\s*(?:neither|no|nor|none)\b", subj, I) or re.search(r"\bnor\b", " ".join(t.text for t in doc[:v.i])[-12:], I): neg = True
        passive = any(c.dep_ in ("nsubjpass", "auxpass") for x in chain for c in x.children)
        modal = next((a for a in auxes if a in ("shall", "will", "must", "may", "can", "should", "might", "could")), None)
        if modal is None: continue
        if neg and modal in ("shall", "will", "must", "may", "can", "should"): kind = "PROHIBITION"
        elif modal in ("may", "can", "might", "could"): kind = "PERMISSION"
        else: kind = "OBLIGATION"
        actor = "PASSIVE" if passive else _role(subj, recv_terms)
        if passive and re.search(r"confidential information|information|materials?", subj, I): obj = subj
        else: obj = " ".join(_span(c) for c in v.children if c.dep_ in ("dobj", "attr", "oprd"))
        rec = " ".join(_span(c) for c in v.children if c.dep_ == "prep" and c.lower_ in ("to", "with"))
        out.append((kind, actor, _action(v.lemma_.lower()), v.lemma_.lower(), obj, rec))
    return out


# ---------- 4 + 5. scope and semantic passes ----------
GRANT = re.compile(
    r"(?:\bmay\s+(?:\w+\s+){0,3}(?:disclose|share|provide|communicate|give)\b.{0,120}?\bto\s+"
    r"|\b(?:except|other than)\s+(?:\w+\s+){0,4}?(?:to|with|for disclosure to)\s+"
    r"|\b(?:only|solely|exclusively)\s+(?:be\s+)?(?:\w+\s+){0,2}?(?:to|with)\s+"
    r"|\blimit\w*\s+(?:\w+\s+){0,3}?(?:access|disclosure|distribution|dissemination)\b.{0,60}?\bto\s+"
    r"|\brestrict\w*\s+(?:\w+\s+){0,3}?(?:access|disclosure)\b.{0,60}?\bto\s+)", I)
LEGAL = r"\brequired by (?:law|statute|regulation|rule)|\bcompelled|\bsubpoena|\bcourt order|\bfreedom of information|\blegal process|\bopen records|\bgovernmental (?:authority|agency|body)|\bjudicial|\bstock exchange"


def _expand(prog, phrase: str) -> str:
    out = phrase
    for k, bodies in prog.symbols.items():
        if len(k) > 3 and re.search(r"\b" + re.escape(k) + r"\b", phrase):
            out += " [" + k + ": " + " ".join(bodies[:2])[:600] + "]"
    return out


def _marking(body: str):
    """Is marking (identification as confidential) a necessary condition of being CI, or one route among others?"""
    if not body: return None, ""
    mark = re.search(r"(?:marked|labell?ed|designated|identified|stamped|legended|indicated)\b.{0,50}\b(?:as\s+)?[\"“'‘]?(?<!non-)(?:confidential|proprietary|secret|restricted)", body, I)
    whether = re.search(r"(?:whether or not|regardless of whether|irrespective of whether)\s+(?:\w+\s+){0,6}?(?:marked|designated|identified|labell?ed)|"
                        r"need not be (?:marked|designated|identified)|without (?:being )?(?:so )?(?:marked|designated|identified)", body, I)
    if whether: return "sufficient", whether.group(0)
    if not mark: return None, ""
    alt = re.search(r"\bor\b.{0,80}\b(?:reasonabl\w*|by (?:its|their) (?:nature|character)|ought to|should (?:reasonably )?(?:be )?(?:understood|known|considered)|"
                    r"would be (?:understood|considered)|is (?:understood|known) to be|under the circumstances|from the (?:nature|circumstances))", body, I)
    alt2 = re.search(r"\b(?:including|includes|such as)\b.{0,80}\b(?:business|financial|technical|commercial|customer|pricing|plans|know-how)", body, I)
    if alt or alt2: return "sufficient", (alt or alt2).group(0)
    return "necessary", mark.group(0)


def facts(text: str) -> Facts:
    prog = compile_doc(text)
    stmts = [s for s, _ in prog.stmts]
    # role inference: the term most often the subject of a ban on disclosure/use = the receiving party
    recv_terms = [k for k, b in prog.symbols.items() if re.search(r"receiv|recipient", k + " " + " ".join(b[:1]), I) and len(k) < 40]
    docs = list(nlp().pipe([preprocess(s)[:1500] for s in stmts], batch_size=64))
    norms = []
    for s, d in zip(stmts, docs):
        exc = [m.group(0) + s[m.end():m.end() + 200] for m in re.finditer(EXC, s, I)]
        cond = [m.group(0) + s[m.end():m.end() + 160] for m in re.finditer(r"\bif\b|\bupon\b|\bin the event\b|\bwhen\b|\bwhere\b(?= \w+ is)", s, I)]
        gate = bool(re.search(CONSENT, s, I)); over = bool(re.search(r"\bnotwithstanding\b", s, I))
        for kind, actor, action, verb, obj, rec in predicates(d, recv_terms):
            norms.append(Norm(kind, actor, action, verb, obj, rec, exc, cond, gate, over, s))
    # semantic pass: who the receiving party may disclose to
    permit = []
    for s in stmts:
        if re.search(r"\bmeans\b|\bshall mean\b|\bis defined as\b|\bwhereas\b|\brecitals?\b|\bthe term\b", s, I): continue
        if re.search(LEGAL, s, I): continue
        if not re.search(r"\bdisclos|\bshare|\bcommunicat|\bprovide\b|\bmake\s+(?:\w+\s+)?available|\baccess\b|\bdistribut|\bdissemin|\breveal", s, I): continue
        for m in GRANT.finditer(s):
            who = re.split(r"[.;]\s|\bprovided\b|\bunless\b", s[m.end():m.end() + 240])[0]
            permit.append((_expand(prog, who), s))
    marking, mtext = _marking(" ".join(prog.ci_body))
    return Facts(norms, permit, prog.ci_body, prog.ci_exclusions, marking, mtext, stmts, prog.symbols, recv_terms)
