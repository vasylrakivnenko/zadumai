"""The 17 NDA questions as queries over the IR facts (ir.py) (2026-10-03, overnight). Each returns (answer, evidence)
or None. Answers are in the question's terms (baseline.QUESTIONS; nda-15 asks whether a license IS granted).
Written from the training NDAs only."""
from __future__ import annotations

import re

import queries as Q  # the compiler-pass queries (definition / exclusion based), reused where they are the right tool
from ir import Facts, LEGAL

I = re.I
R = ("RECV", "BOTH", "PASSIVE", "UNKNOWN")
CI = r"confidential information|proprietary information|information|materials?|data|documents?|evaluation material"
THIRD = Q.THIRD
INTERNAL = Q.INTERNAL


def _norms(f: Facts, kind=None, action=None, actors=R):
    return [n for n in f.norms if (kind is None or n.kind == kind) and (action is None or n.action == action) and n.actor in actors]


def nda_1(f: Facts):  # all CI must be expressly identified
    if f.marking == "necessary": return "yes", f.marking_text
    if f.marking == "sufficient": return "no", f.marking_text
    return None


def nda_2(f: Facts):
    body = " ".join(f.ci_body)
    if not body: return None
    cats = {m.group(0).lower()[:6] for m in re.finditer(Q.NONTECH, body, I)}
    return ("no", body) if len(cats) >= 2 else None


def nda_3(f: Facts):
    for t in f.ci_body:
        if re.search(Q.ORAL, t, I) and not re.search(r"\bnot\b.{0,30}\b(?:oral|verbal)", t, I): return "yes", t
    return None


PURPOSE = r"\bpurposes?\b|\bevaluat|\bthe (?:project|transaction|proposal|business relationship|services|discussions)\b"


def nda_4(f: Facts):  # use only for the agreement's purpose
    for n in _norms(f, action="USE"):
        lim = re.search(r"\b(?:solely|only|exclusively)\b", n.text, I) or any(re.search(r"\bexcept|\bother than", e, I) for e in n.exceptions)
        if re.search(PURPOSE, n.text, I) and lim and n.kind in ("PROHIBITION", "OBLIGATION", "PERMISSION") \
                and not re.search(r"\bfree to use\b|\bfor any purpose whatsoever\b(?!.{0,20}\bother)", n.text, I):
            return "yes", n.text
    return None


def nda_5(f: Facts):  # may share with employees
    for who, t in f.permit_recipients:
        if re.search(INTERNAL, who, I): return "yes", t
    return None


THIRD7 = (r"\bconsultant|\badvis[oe]r|\bagents?\b|\bcontractor|\bsubcontractor|\battorney|\blawyer|\baccountant|\bcounsel\b|\bauditor|"
          r"\blender|\bfinancing source|\bbankers?\b|\bprofessional advis|\bthird part(?:y|ies)\b(?!\s+without)")
# not affiliates or investors: ContractNLI doesn't count those as the hypothesis' third parties when another clause gates
# them (train, 2026-10-03); no "no" answers: bans on "any third party" are in almost every NDA, permissions elsewhere


def nda_7(f: Facts):  # may share with third parties (consultants, agents, advisors)
    for who, t in f.permit_recipients:
        if re.search(THIRD7, who, I) and not re.search(r"\bnot\b.{0,40}\b(?:consultant|agent|advis|contractor|third part)", who, I):
            return "yes", t
    return None


def nda_8(f: Facts):  # notice on compelled disclosure
    for n in _norms(f, kind="OBLIGATION", action="NOTIFY"):
        if re.search(LEGAL, n.text, I) or re.search(r"\brequired\b|\brequest(?:ed)?\b.{0,40}\b(?:court|government|authorit|law)", n.text, I):
            return "yes", n.text
    return None


EXIST = (r"\bexistence (?:or|and|of)\b|\bexistence\b(?= of (?:this|the|any|such)\b)|\bfact that\b.{0,60}\b(?:discussions|negotiations|agreement|transaction|evaluation|relationship|confidential information has been|investigations)|"
         r"\bnature of (?:the|these|its|any) (?:discussions|negotiations)|\b(?:status|content|substance) of (?:the|any|such) (?:discussions|negotiations)|"
         r"\bthat (?:any )?(?:discussions|negotiations) (?:are|have|is)\b")
# not "terms of this Agreement" alone: "agreed to be bound by the terms of the Agreement" (train errors, 2026-10-03)


BANWORDS = r"\b(?:disclose|reveal|divulge|announce|publici[sz]e|make public|publish|discuss|confirm|communicate|inform\w*|protect|confidential)\w*"


def nda_10(f: Facts):  # the fact of the agreement / discussions is confidential: what is banned must be that fact
    for n in _norms(f, kind="PROHIBITION") + _norms(f, kind="OBLIGATION", action="PROTECT"):
        if re.search(EXIST, n.obj, I): return "yes", n.text
        if re.search(BANWORDS + r".{0,80}?(?:" + EXIST + ")", n.text, I) and re.search(r"\bnot\b|\bno\b|\bneither\b|\bnor\b|confidential", n.text, I):
            return "yes", n.text
    return None


def nda_11(f: Facts):  # no reverse engineering
    for n in _norms(f, kind="PROHIBITION", action="DECOMPILE"):
        if re.search(r"reverse[- ]?engineer|decompil|disassembl", n.text, I): return "yes", n.text
    return None


def nda_12(f: Facts):
    return Q.nda_12(_prog(f))


def nda_13(f: Facts):
    return Q.nda_13(_prog(f))


NOLICENSE = (r"\bno (?:\w+ )?(?:licen[cs]e|rights?)\b.{0,80}\b(?:granted|conveyed|implied|transferred|created|given|conferred)|"
             r"\bnothing\b.{0,120}\b(?:construed|deemed|interpreted)\b.{0,60}\b(?:grant|licen[cs]e|confer|transfer)|"
             r"\b(?:shall|will) not\b.{0,40}\b(?:be (?:construed|deemed|interpreted) (?:as|to) (?:grant|confer|give|transfer|convey|create)\w*|grant|confer|convey)\b.{0,60}\b(?:licen[cs]e|right|title|interest)|"
             r"\b(?:remain|is|are)\b.{0,20}\b(?:the )?(?:sole |exclusive )?property of\b|\bretains? all (?:right|title|ownership)|"
             r"\ball (?:right|rights), title,? and interest\b.{0,60}\b(?:remain|retain|belong|vest)")


def nda_15(f: Facts):  # does the agreement grant a license? (E = no license -> "no")
    for s in f.stmts:
        if re.search(NOLICENSE, s, I): return "no", s
    return None


END = (r"\bterminat|\bexpir|\b(?:end|conclusion|completion|cessation)\b|\bcomes to an end|\bceas(?:e|es|ing)\b|\bno longer (?:needed|required|necessary)|"
       r"\bnot to proceed|\bdecides? not to|\bdetermin\w* not to|\bdiscontinu|\bwithdraw")


END = END.replace(r"|\bceas(?:e|es|ing)\b", r"|\bceas(?:e|es|ing) to be interested|\bcessation of")  # not "cease to use" (2026-10-03)


def nda_16(f: Facts):  # return or destroy upon the end of the agreement. ContractNLI: a duty "upon request" is not
    # mentioned, even after termination ("following termination ..., upon the request ...") - unless the end itself is
    # an alternative trigger ("upon request or upon termination") (train + dev errors, 2026-10-03)
    for n in _norms(f, kind="OBLIGATION", action="RETURN"):
        if not (re.search(CI + r"|\bcop(?:y|ies)|\bdocuments", n.obj + " " + n.text, I) and re.search(END, n.text, I)): continue
        if re.search(r"\brequest|\bdemand|\bdirect(?:ed|ion)\b|\binstruct", n.text, I) and not re.search(
                r"\b(?:request|demand)\w*\b.{0,60}\bor\b.{0,40}(?:" + END + r")|(?:" + END + r").{0,60}\bor\b.{0,40}\b(?:request|demand)", n.text, I):
            continue
        return "yes", n.text
    return None


COPYOK = (r"\bcop(?:y|ies|ying)\b.{0,80}?\b(?:except|other than|unless|only|save)\b.{0,60}?\b(?:necessary|required|needed|purpose|evaluat|internal)|"
          r"\b(?:except|other than|unless|save)\b.{0,50}?\b(?:necessary|required|needed)\b.{0,50}?\bcop(?:y|ies)|"
          r"\bcop(?:y|ies)\b.{0,40}?\b(?:as|to the extent) (?:reasonably )?(?:necessary|required|needed)")


def nda_17(f: Facts):  # may make copies (the exception must be about copying, close to it)
    for n in _norms(f, kind="PERMISSION", action="COPY"):
        return "yes", n.text
    for n in _norms(f, kind="PROHIBITION", action="COPY"):
        if re.search(COPYOK, n.text, I): return "yes", n.text
    if any(re.search(COPYOK, s, I) for s in f.stmts): return None
    for n in _norms(f, kind="PROHIBITION", action="COPY"):
        if not n.exceptions and not n.consent_gate: return "no", n.text
    return None


def nda_18(f: Facts):  # no solicitation
    for n in _norms(f, kind="PROHIBITION", action="SOLICIT"):
        if n.verb in ("engage", "offer") : continue
        if re.search(r"employee|personnel|staff|officer|representative|consultant|agent|customer|client|worker|individual|anyone|any person", n.obj + " " + n.recipients, I):
            return "yes", n.text
    return None


def nda_19(f: Facts):
    return Q.nda_19(_prog(f))


RETAINX = (r"\bretain|\bkeep\b|\barchiv|\bback-?up|\bone (?:archival )?copy|\blegal (?:department|counsel|purposes|requirements?)|"
           r"\bcompliance|\binfeasible|\bnot (?:be )?feasible|\bimpracticable|\bautomatic(?:ally)? (?:created|stored)")


def nda_20(f: Facts):  # may retain some CI after return/destruction
    for n in _norms(f, kind="PERMISSION", action="RETAIN"):
        if re.search(r"\bresidual", n.text, I): continue  # memory of residuals isn't keeping copies
        return "yes", n.text
    for n in _norms(f, kind="OBLIGATION", action="RETURN"):
        if any(re.search(RETAINX, e, I) and not re.search(r"\b(?:not|no)\s+(?:\w+\s+){0,2}(?:retain|keep)", e, I) for e in n.exceptions):
            return "yes", n.text
    for n in _norms(f, kind="PROHIBITION", action="RETAIN"):
        if not n.exceptions and not n.consent_gate: return "no", n.text
    return None


def _prog(f: Facts):
    """The compiler-pass view the definition queries read (queries.py takes a Program)."""
    class P: pass
    p = P(); p.ci_body = f.ci_body; p.ci_exclusions = f.ci_exclusions; p.stmts = [(s, []) for s in f.stmts]; p.symbols = f.symbols
    return p


# ---------- "not mentioned" from a located structure that lacks the item (2026-10-03) ----------
def _doc(f: Facts) -> str:
    return " ".join(f.stmts)


def nm_rule(q, f: Facts):
    t = _doc(f)
    if q == "nda-12" and f.ci_exclusions and not re.search(Q.INDEP + r"|independent", t, I): return "not mentioned"
    if q == "nda-13" and f.ci_exclusions and not re.search(Q.THIRDSRC + r"|third[- ]part", t, I): return "not mentioned"
    if q == "nda-3" and f.ci_body and not re.search(Q.ORAL + r"|\bin any form|\bwhatever form|\bany (?:form|medium|manner)", t, I): return "not mentioned"
    if q == "nda-19" and not re.search(r"\bsurviv|\bcontinue|\bremain in (?:full )?(?:force|effect)|\bnotwithstanding (?:the |any )?(?:termination|expiration)|\bafter (?:the )?(?:termination|expiration)", t, I):
        return "not mentioned"
    if q == "nda-10" and not re.search(EXIST + r"|\bterms of (?:this|the) agreement|\bannounce|\bpublicity|\bpress release", t, I): return "not mentioned"
    if q == "nda-16":
        if not re.search(r"\breturn|\bdestr|\bdelet|\berase|\bsurrender|\bpurge", t, I): return "not mentioned"
        rets = _norms(f, kind="OBLIGATION", action="RETURN")
        if rets and all(re.search(r"\brequest|\bdemand", n.text, I) and not re.search(END, n.text, I) for n in rets): return "not mentioned"
    if q == "nda-20" and not re.search(RETAINX + r"|\ball copies|\bno copies|\bretain", t, I): return "not mentioned"
    if q == "nda-1" and f.ci_body and not re.search(r"\bmark|\blabel|\bdesignat|\bidentif|\blegend|\bstamp", t, I): return "not mentioned"
    return None


def _with_nm(q, fn):
    def g(f: Facts):
        r = fn(f)
        if r: return r
        nm = nm_rule(q, f)
        return (nm, "(no such clause found)") if nm else None
    return g


QUERIES = {"nda-1": nda_1, "nda-2": nda_2, "nda-3": nda_3, "nda-4": nda_4, "nda-5": nda_5, "nda-7": nda_7, "nda-8": nda_8,
           "nda-10": nda_10, "nda-11": nda_11, "nda-12": nda_12, "nda-13": nda_13, "nda-15": nda_15, "nda-16": nda_16,
           "nda-17": nda_17, "nda-18": nda_18, "nda-19": nda_19, "nda-20": nda_20}
QUERIES = {k: _with_nm(k, v) for k, v in QUERIES.items()}
