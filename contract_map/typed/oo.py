"""Contract object model + a small query language (draft, 2026-10-03; the user: "combine both ideas. Make about 8 core
classes the compiler's output, keep the small query language as the interface questions compile into, and have the
specialized clause classes host the per-type answerers we already built").

Object model (what the compiler outputs):
  Contract     the document: kind, title, parties, sections, clauses, terms, references, norms; ask(); subclass NDA
  Party        a party and the role the contract calls it ("Licensee")
  Section      a numbered block or a heading's scope: number, heading, span, parent
  Clause       a section seen as a clause type; subclasses host per-type answerers (registered by clause type)
  Norm         who may / must / must not do what to what (permission, obligation, prohibition), with its override
               ("notwithstanding"), deferral ("subject to Section X") and exception markers: idea 2's priorities
  DefinedTerm  a defined term and its definition
  Reference    a cross-reference: to a section (resolved), an exhibit, another document
  Answer       value + evidence + which object answered + a trace

Query language (CQL), what questions compile into:
  MAY(actor=..., act=CONCEPT, object=...)        may a party do it        -> yes / no
  MUST(actor=..., act=CONCEPT, object=...)       must it                  -> yes / no
  MUST_NOT(actor=..., act=CONCEPT, object=...)   is it prohibited         -> yes / no
  HAS(clause="Governing Law")                    does the contract have such a clause
  VALUE(clause="Governing Law", slot=law)        a clause's value (draft: governing law only)
  WHO(modality=MUST, act=CONCEPT)                which parties (draft: from the norms)
  ANSWER(q="...")                                the neural operator: anything the forms above don't cover
Execution of a query: (1) a specialized clause class (or NDA method) registered for it answers if it can, with only
the variants that held >= 98% in earlier cross-validation; (2) otherwise the neural ANSWER operator (pipeline.py:
locate, Pre-Tier 0, the reader network) with idea 1's consistency check, idea 2's priority veto from the norms.
Draft: the specialized answerers come from cuadc/clauses.py (CUAD) and cnli/queries_ir.py (NDA)."""
from __future__ import annotations

import functools
import re
import sys
from dataclasses import dataclass, field

import ns
import pipeline as P
from router import qshapes

sys.path.insert(0, f"{P.HERE}/../doccompile")
sys.path.insert(0, f"{P.HERE}/../cuadc")
sys.path.insert(0, f"{P.HERE}/../cnli")
from doccompile import compile_doc  # noqa: E402


# ======================================================== the object model
@dataclass
class Party:
    name: str
    role: str
    said: str = ""


@dataclass
class Section:
    id: int
    number: str
    heading: str
    start: int
    end: int
    parent: int


@dataclass
class DefinedTerm:
    name: str
    definition: list


@dataclass
class Reference:
    kind: str           # section | attachment | document | document-section | incorporation
    label: str
    target: int = -1    # section id when resolved
    resolved: bool = False


@dataclass
class Norm:
    kind: str           # permission | obligation | prohibition | statement
    holders: set
    acts: set
    objects: set
    text: str
    sections: tuple
    overrides: bool = False
    subject_to: set = field(default_factory=set)
    has_exception: bool = False


@dataclass
class Answer:
    value: str | None   # "yes" | "no" | None (no answer)
    evidence: str = ""
    via: str = ""       # which object answered: "GoverningLawClause", "NDA.nda-5", "ANSWER(network)", ...
    query: str = ""
    trace: list = field(default_factory=list)


class Clause:
    """A section seen as a clause type. Subclasses register themselves for a clause type and answer what they can."""
    TYPE = ""
    REGISTRY: dict = {}

    def __init__(self, contract: "Contract", section: Section | None, statements: list):
        self.contract, self.section, self.statements = contract, section, statements

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.TYPE: Clause.REGISTRY[cls.TYPE] = cls

    def answer(self, q: "Query") -> Answer | None:
        return None


# ---------- specialized clause classes: CUAD answerers (cuadc/clauses.py), trusted variants only ----------
class _CuadClause(Clause):
    TRUSTED: set = set()   # variants >= 98% on cuadc's write contracts (cuadc/RESULTS.md)

    @classmethod
    def find(cls, contract: "Contract"):
        import clauses as C
        out = C.answers(contract.cuad_stmts, cls.TYPE)
        return out

    def answer(self, q: "Query") -> Answer | None:
        if q.form not in ("HAS", "VALUE"): return None
        out = self.find(self.contract)
        for v in sorted(out, key=lambda v: v not in self.TRUSTED):
            val, st = out[v]
            if v in self.TRUSTED and val == "yes":
                if q.form == "VALUE": return self.value(q, st)
                return Answer("yes", st.own, f"{type(self).__name__}[{v}]", str(q), [f"{self.TYPE}: variant {v} found"])
        if "no_mention" in out and self.NO_MENTION_TRUSTED:
            return Answer("no", "", f"{type(self).__name__}[no_mention]", str(q), [f"nothing in the contract mentions {self.TYPE}"])
        return None

    NO_MENTION_TRUSTED = False

    def value(self, q, st):
        return None


class GoverningLawClause(_CuadClause):
    TYPE = "Governing Law"
    TRUSTED = {"head+core", "agr+core"}

    def value(self, q, st):
        m = re.search(r"\blaws?\s+of\s+(?:the\s+)?((?:State|Commonwealth|Province|Republic)\s+of\s+)?([A-Z][\w ]{2,40}?)(?=[,.;)]|\s+(?:without|and|applicable|excluding|as)\b)", st.own)
        return Answer(m.group(0) if m else "yes", st.own, f"{type(self).__name__}.value", str(q), ["governing law clause: the law named"]) if m else None


class AssignmentClause(_CuadClause):
    TYPE = "Anti-Assignment"
    TRUSTED = {"head+canon"}


class LiabilityCapClause(_CuadClause):
    TYPE = "Cap On Liability"
    TRUSTED = {"head+excl", "head+amount"}


class RenewalClause(_CuadClause):
    TYPE = "Renewal Term"
    TRUSTED = {"head+auto"}


# ======================================================== the query language
@dataclass
class Query:
    form: str            # MAY | MUST | MUST_NOT | HAS | VALUE | WHO | ANSWER
    args: dict
    text: str = ""       # the question it came from

    def __str__(self):
        a = ", ".join(f"{k}={v!r}" if isinstance(v, str) and not v.isupper() else f"{k}={v}" for k, v in self.args.items())
        return f"{self.form}({a})"


_Q = re.compile(r"^\s*(?P<f>MAY|MUST_NOT|MUST|HAS|VALUE|WHO|ANSWER)\s*\((?P<a>.*)\)\s*$", re.S)
_ARG = re.compile(r"\s*(?P<k>\w+)\s*=\s*(?:\"(?P<s>[^\"]*)\"|'(?P<s2>[^']*)'|(?P<w>[\w.:-]+))\s*(?:,|$)")


def parse(text: str) -> Query:
    """CQL text -> Query (the grammar above); raises ValueError."""
    m = _Q.match(text)
    if not m: raise ValueError(f"not a query: {text!r}")
    args, pos, a = {}, 0, m.group("a")
    while pos < len(a):
        mm = _ARG.match(a, pos)
        if not mm: raise ValueError(f"bad arguments: {a[pos:]!r}")
        args[mm.group("k")] = mm.group("s") if mm.group("s") is not None else mm.group("s2") if mm.group("s2") is not None else mm.group("w")
        pos = mm.end()
    need = {"MAY": {"act"}, "MUST": {"act"}, "MUST_NOT": {"act"}, "HAS": {"clause"}, "VALUE": {"clause", "slot"},
            "WHO": {"act"}, "ANSWER": {"q"}}[m.group("f")]
    if not need <= set(args): raise ValueError(f"{m.group('f')} needs {sorted(need)}")
    return Query(m.group("f"), args)


# question -> query: Pre-Tier 0's frame is the typed form; clause-type questions from the contract map's triggers
_FORM = {"CAN": "MAY", "MUST": "MUST", "PROHIBITED": "MUST_NOT"}
_HAS_TRIGGERS = {
    "Governing Law": r"\bgovern\w*\s+law|\bwhich (?:state'?s?|country'?s?) laws?|\blaw\b.{0,30}\b(?:applies|govern)|\bchoice of law",
    "Anti-Assignment": r"\b(?:restrict|limit|prohibit)\w*\b.{0,30}\bassign|\bassignment\b.{0,40}\b(?:restrict|consent|prohibit)|\bcan(?:not)?\b.{0,30}\bassign",
    "Cap On Liability": r"\b(?:cap|limit\w*|maximum)\b.{0,30}\bliabilit|\bliabilit\w*\b.{0,30}\b(?:cap|capped|limited)",
    "Renewal Term": r"\b(?:renew\w*|auto-?renew\w*)\b",
}


def compile_question(question: str, frame: dict | None) -> Query:
    if frame and frame.get("asks") in _FORM and frame.get("actions"):
        things = [t[1] for t in frame.get("things") or [] if t[0] == "THING"]
        specific = [t for t in things if t not in ("CONFIDENTIAL_INFO", "AGREEMENT")]  # "share CI with employees" -> EMPLOYEES
        args = {"actor": frame.get("actor") or "ANY", "act": frame["actions"][0]}
        if things: args["object"] = (specific or things)[0]
        return Query(_FORM[frame["asks"]], args, question)
    if (frame or {}).get("asks") == "EXISTS" or re.match(r"(?i)^(?:does|do|is|are)\b", question):
        for t, rx in _HAS_TRIGGERS.items():
            if re.search(rx, question, re.I) and re.search(r"(?i)\b(?:clause|provision|agreement|contract|is there|does)\b", question):
                return Query("HAS", {"clause": t}, question)
    return Query("ANSWER", {"q": question}, question)


# ======================================================== the contract
class Contract:
    KIND = "contract"

    def __init__(self, text: str, pipe: "P.Pipeline | None" = None):
        self.text = text
        c = compile_doc(text)
        self.title = c.title
        self.parties = [Party(p.name, p.role, p.said) for p in c.parties]
        self.sections = [Section(s.id, s.number, s.heading, s.start, s.end, s.parent) for s in c.sections]
        self.terms = [DefinedTerm(k, v) for k, v in c.symbols.items()]
        self.references = [Reference(x.kind, x.label, x.target, x.resolved) for x in c.xrefs]
        self._compiled = c
        self.pipe = pipe

    # ---- derived objects, computed when first asked for
    @functools.cached_property
    def norms(self) -> list:
        dn = self.doc_norms
        kind = {"permit": "permission", "oblige": "obligation", "ban": "prohibition"}
        return [Norm(kind.get(n.polarity, "statement"), set(n.actors), set(n.actions), set(getattr(n, "things", set())),
                     n.text, n.secs, n.override, set(n.defers), n.inline_exception) for n in dn.norms]

    @functools.cached_property
    def doc_norms(self):
        return ns.DocNorms(self.text, [p.name for p in self.parties])

    @functools.cached_property
    def cuad_stmts(self):
        from front import compile_contract
        return compile_contract(self.text)

    def clause(self, clause_type: str) -> Clause | None:
        cls = Clause.REGISTRY.get(clause_type)
        return cls(self, None, []) if cls else None

    # ---- queries
    def ask(self, question: str) -> Answer:
        r0 = P.pt0_check(question, self.text)
        frame = (r0.frames or {}).get("frame")
        q = compile_question(question, frame)
        trace = [f"query: {q}"]
        a = self.execute(q, frame)
        if a is not None and a.value:
            a.trace = trace + a.trace; return a
        a = self.neural(question, q, frame)
        a.trace = trace + a.trace
        return a

    def execute(self, q: Query, frame) -> Answer | None:
        """The specialized objects first."""
        if q.form in ("HAS", "VALUE"):
            c = self.clause(q.args["clause"])
            return c.answer(q) if c else None
        if q.form == "WHO":
            holders = sorted({h for n in self.norms if q.args["act"] in n.acts and n.kind == {"MUST": "obligation", "MAY": "permission"}.get(q.args.get("modality", "MUST"), "obligation") for h in n.holders})
            return Answer(", ".join(holders) or None, "", "Contract.norms", str(q), [f"norms with {q.args['act']}: {len(holders)} holders"])
        return None

    def neural(self, question: str, q: Query, frame) -> Answer:
        """ANSWER(q): the pipeline (Pre-Tier 0, locate, the network), the network's "yes" kept only when its related
        questions agree (idea 1) and no conflicting norm outranks its statement (idea 2)."""
        pipe = self.pipe or P.Pipeline(t_located=0.80)
        res = pipe.answer(question, self.text)
        trace = [f"ANSWER via {res.get('path')}: {res.get('answer')} ({res.get('reason', '')[:80]})"]
        val, ev = res.get("answer"), res.get("evidence", "")
        if val == "yes" and res.get("path", "").startswith("net") and res.get("read_text"):
            rel = ns.related(question, frame, res.get("parties") or [])
            agree = True
            for kind, rq in rel:
                py, _ = ns.net_probs(pipe.net, rq, res["read_text"])
                trace.append(f"  related {kind}: {rq} -> p(yes) {py:.2f}")
                if (kind in ("ban", "perm") and py >= 0.5) or (kind == "para" and py < 0.5): agree = False
            p = float(res.get("p") or 0)
            if not agree or (p < 0.90 and not rel):
                trace.append("  dropped: the related questions disagree" if not agree else "  dropped: below 0.90 with nothing to confirm it")
                val = None
        if val and frame:
            pr = ns.priority(self.doc_norms, frame, val, ev)
            if pr["pr_overridden"] or pr["pr_unresolved"] or pr["pr_defers_to_conflict"]:
                trace.append(f"  dropped by priority: {', '.join(k for k in ('pr_overridden', 'pr_unresolved', 'pr_defers_to_conflict') if pr[k])}")
                val = None
        return Answer(val, ev, f"ANSWER({res.get('path')})", str(q), trace)


class NDA(Contract):
    """An NDA hosts the 17 NDA answerers (cnli/queries_ir.py), only those >= 98% in cross-validation."""
    KIND = "nda"
    TRUSTED = {"nda-3", "nda-5", "nda-8", "nda-15", "nda-16", "nda-19"}

    @functools.cached_property
    def facts(self):
        from ir import facts
        return facts(self.text)

    @functools.cached_property
    def _templates(self):
        """The 17 questions' own queries, to match an incoming query against."""
        from baseline import QUESTIONS
        out = {}
        for k, qq in QUESTIONS.items():
            fr = (P.pt0_check(qq, "The Receiving Party shall keep the Confidential Information confidential.").frames or {}).get("frame")
            out[k] = (compile_question(qq, fr), qq)
        return out

    def execute(self, q: Query, frame) -> Answer | None:
        for k, (tq, qq) in self._templates.items():
            if k in self.TRUSTED and tq.form == q.form and tq.args.get("act") == q.args.get("act") and \
                    tq.args.get("object") == q.args.get("object") and (q.form != "ANSWER" or q.args["q"] == qq):
                import queries_ir as QI
                r = QI.QUERIES[k](self.facts)
                if r and r[0] in ("yes", "no"):
                    return Answer(r[0], r[1] if isinstance(r[1], str) else "", f"NDA.{k}", str(q), [f"matched NDA question {k}"])
        return super().execute(q, frame)


def contract(text: str, kind: str | None = None, pipe=None) -> Contract:
    """The compiler's entry point: the right Contract class for the document's kind."""
    return (NDA if kind == "nda" else Contract)(text, pipe)
