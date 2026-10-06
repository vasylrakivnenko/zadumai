"""The 17 NDA questions as queries over a compiled NDA (compiler.py) (2026-10-03). Each returns
(answer, evidence statement) or None (abstain). Answers are in the question's own terms (baseline.QUESTIONS): "yes" /
"no" - for nda-15 "yes" means a license IS granted. Written from the training NDAs only.
Queries for the questions the engine fails or barely passes: nda-1, 2, 3, 4, 5, 7, 11, 12, 13, 19, 20."""
from __future__ import annotations

import re

from compiler import Program, resolve

I = re.I
NEG = r"\b(?:not|no|never|neither|nor|prohibit\w*|refrain\w*|forbid\w*|restrict\w*)\b"
PERMIT = (r"(?:\bmay\b|\bcan\b|\bpermitted to\b|\bentitled to\b|\ballowed to\b|\bauthori[sz]ed to\b|\bexcept\b|\bother than\b|"
          r"\bonly to\b|\bsolely to\b|\blimit\w* (?:access|disclosure|distribution)\b|\brestrict\w* (?:access|disclosure)\b|"
          r"\bunless\b|\bprovided that\b|\bwithout (?:the )?(?:prior )?(?:written )?consent\b)")
THIRD = (r"\bconsultant|\badvis[oe]r|\bagents?\b|\bcontractor|\bsubcontractor|\battorney|\blawyer|\baccountant|\bcounsel\b|\bauditor|"
         r"\baffiliate|\blender|\bfinancing source|\bbank(?:er)?s?\b|\binvestor|\bprofessional advis|\bthird part(?:y|ies)\b(?!\s+without)")
INTERNAL = r"\bemployee|\bstaff\b|\bpersonnel\b|\bofficers?\b|\bdirectors?\b|\bpartners?\b|\bmembers?\b|\bboard\b"


def _first(stmts, *pats, exclude=None):
    for t in stmts:
        if all(re.search(p, t, I) for p in pats) and not (exclude and re.search(exclude, t, I)):
            return t
    return None


def nda_1(p: Program):  # all CI must be expressly identified
    body = " ".join(p.ci_body)
    no = re.search(r"whether or not (?:\w+ )?(?:marked|designated|identified|labell?ed|so marked)|regardless of whether|"
                   r"irrespective of whether|need not be (?:marked|designated|identified)|without (?:being )?(?:so )?(?:marked|designated)|"
                   r"whether or not .{0,40}(?:marked|identified|designated)", body, I)
    yes = re.search(r"(?:marked|labell?ed|designated|identified|stamped|legend)\b.{0,40}\b(?:as )?[\"“']?(?:confidential|proprietary)", body, I)
    broad = re.search(r"\b(?:all|any and all|any)\s+(?:\w+\s+){0,3}(?:information|data)\b.{0,80}\b(?:relating|concerning|regarding|whether|of any kind|whatsoever)", body, I)
    if no or (broad and not yes): return "no", body
    return None  # "yes" isn't answered: the labels call most marking clauses "no" when any other route exists (train, 2026-10-03)


NONTECH = (r"\bbusiness|\bfinanc|\bcommercial|\bmarketing|\bcustomer|\bclient (?:list|data|information)|\bpric(?:e|es|ing)\b|\bstrateg|"
           r"\b(?:business|marketing|product|strategic) plans?\b|\bpersonnel|\bsales\b|\bcorporate|\bnon-technical|\boperation|\baccounting|"
           r"\bbudget|\bforecast|\bemployee (?:information|data)|\blegal\b|\bpersonal (?:information|data)")
# not: "any information", "of any kind" - ContractNLI labels a definition with only those "not mentioned" (train errors 2026-10-03)
TECH = r"\btechnical|\btechnology|\bsoftware|\bengineering|\bscientific|\bknow-how|\bdesign|\bspecification|\bsource code|\bformula|\bprocess|\balgorithm|\binvention|\bresearch"


def nda_2(p: Program):  # CI only technical (the usual answer: no)
    body = " ".join(p.ci_body)
    if not body: return None
    cats = {m.group(0).lower()[:6] for m in re.finditer(NONTECH, body, I)}
    if len(cats) >= 2: return "no", body  # at least two kinds of non-technical information named
    return None  # "yes" isn't answered: the labels for technical-only definitions are inconsistent (train, 2026-10-03)


ORAL = r"\boral(?:ly)?\b|\bverbal(?:ly)?\b|\bvisual(?:ly)?\b|\bby (?:way of )?observation|\bspoken"
# not "in any form", "in whatever form", "whether in writing, printed ... or otherwise": labeled not mentioned (train, 2026-10-03)


def nda_3(p: Program):  # CI may include oral information - from the definition only
    for t in p.ci_body:
        if re.search(ORAL, t, I) and not re.search(r"\bnot\b.{0,30}\b(?:oral|verbal)", t, I): return "yes", t
    return None


def nda_3_fallback(p: Program):  # a statement about the information disclosed that names oral/verbal/visual forms
    # (a graded variant for the selector: 90.6% on train when used alone, 2026-10-03)
    r = nda_3(p)
    if r: return r
    for t, _ in p.stmts:
        if re.search(ORAL, t, I) and re.search(r"\binformation\b|\bmaterials?\b", t, I) and re.search(r"\bdisclos|\bprovid|\bfurnish|\bconvey|\bcommunicat|\btransmit|\breceiv", t, I) \
                and not re.search(r"\b(?:oral|verbal)\w*\s+(?:or\s+written\s+)?(?:notice|consent|agreement|request|approval|statement of|amendment|waiver)|\bnot\b.{0,30}\b(?:oral|verbal)", t, I):
            return "yes", t
    return None


def nda_4(p: Program):  # use only for the agreement's purpose
    t = _first([s for s, _ in p.stmts], r"\b(?:use[ds]?|utiliz\w*)\b.{0,80}\b(?:solely|only|exclusively|other than|except)\b.{0,80}\bpurpose|\b(?:solely|only|exclusively)\b.{0,20}\b(?:use|utiliz)\w*\b.{0,60}\bpurpose",
               r"\b(?:use[ds]?|utiliz\w*)\b",
               exclude=r"\bfree to use\b|\bmay use .{0,30}\bfor any purpose")
    return ("yes", t) if t else None


_GRANT = re.compile(
    r"(?:\bmay\s+(?:\w+\s+){0,3}(?:disclose|share|provide|make\s+(?:\w+\s+)?available|communicate|give\s+access)\b.{0,120}?\bto\s+"
    r"|\b(?:except|other than)\s+(?:\w+\s+){0,4}?(?:to|with|for disclosure to|by)\s+"
    r"|\b(?:only|solely|exclusively)\s+(?:be\s+)?(?:\w+\s+){0,2}?(?:to|with)\s+"
    r"|\blimit\w*\s+(?:\w+\s+){0,3}?(?:access|disclosure|distribution|dissemination)\b.{0,60}?\bto\s+"
    r"|\brestrict\w*\s+(?:\w+\s+){0,3}?(?:access|disclosure)\b.{0,60}?\bto\s+"
    r"|\bpermitted\s+(?:to\s+)?(?:disclose|share)\b.{0,80}?\bto\s+)", I)


def _permitted_recipients(p: Program):
    """Who the receiving party may disclose to: the phrase after a granting construction ("except to X", "only to X",
    "may disclose ... to X", "limit access ... to X"), defined groups resolved. Not definitions, recitals, or bare
    "without consent" (a ban with a consent exception: ContractNLI labels that a contradiction)."""
    out = []
    for t, _ in p.stmts:
        if re.search(r"\bmeans\b|\bshall mean\b|\bis defined as\b|\bwhereas\b|\brecitals?\b|\bthe term\b", t, I): continue
        if re.search(r"\brequired by (?:law|statute|regulation)|\bcompelled|\bsubpoena|\bcourt order|\bfreedom of information|\blegal process|\bopen records", t, I): continue
        if not re.search(r"\bdisclos|\bshare|\bcommunicat|\bprovide\b|\bmake\s+(?:\w+\s+)?available|\baccess\b|\bdistribut|\bdissemin", t, I): continue
        for m in _GRANT.finditer(t):
            who = t[m.end():m.end() + 220]
            who = re.split(r"[.;]\s|\bprovided\b|\bunless\b", who)[0]
            out.append((resolve(p, who), t))
    return out


def nda_5(p: Program):  # may share with employees
    for who, t in _permitted_recipients(p):
        if re.search(INTERNAL, who, I): return "yes", t
    return None


def nda_7(p: Program):  # may share with third parties (consultants, agents, advisors)
    rec = _permitted_recipients(p)
    for who, t in rec:
        if re.search(THIRD, who, I) and not re.search(r"\bnot\b.{0,40}\b(?:consultant|agent|advis|contractor|third part)", who, I):
            return "yes", t
    for who, t in rec:  # only internal people may receive it, and no third party anywhere among the permitted
        if re.search(INTERNAL, who, I) and re.search(r"\bonly\b|\bsolely\b|\blimit|\brestrict", t, I):
            if not any(re.search(THIRD, w, I) for w, _ in rec): return "no", t
    return None


def nda_11(p: Program):  # no reverse engineering
    t = _first([s for s, _ in p.stmts], r"reverse[- ]?engineer|decompil|disassembl", NEG)
    return ("yes", t) if t else None


INDEP = (r"independent(?:ly)?\s+(?:develop|creat|conceiv|generat|deriv|discover|produc)|(?:develop|creat|conceiv|deriv|discover)\w*\s+(?:\w+\s+){0,6}independently|"
         r"independent development|independent(?:ly)? (?:of|from) (?:any|the) (?:use|disclosure|confidential)")


def nda_12(p: Program):  # may independently develop similar information
    for t in p.ci_exclusions:
        if re.search(INDEP, t, I): return "yes", t
    t = _first([s for s, _ in p.stmts], r"\b(?:nothing|not)\b.{0,100}\b(?:prevent|restrict|limit|preclude|prohibit)", INDEP)
    return ("yes", t) if t else None


THIRDSRC = (r"(?:receiv|obtain|acquir|disclos|learn)\w*\s+(?:\w+\s+){0,6}(?:from|by)\s+(?:a|any|another)\s+(?:third[- ]part|source|person)|"
            r"third[- ]part(?:y|ies)\s+(?:who|which|that|having|with)|\brightfully (?:received|obtained|acquired|in)|"
            r"\blawfully (?:received|obtained|acquired)|source other than|on a non-confidential basis from")


def nda_13(p: Program):  # may acquire similar information from a third party
    for t in p.ci_exclusions:
        if re.search(THIRDSRC, t, I) and not re.search(r"\bshall include\b|\bincludes?\b.{0,40}\bthird part", t, I): return "yes", t
    return None


def nda_19(p: Program):  # some obligations survive termination
    t = _first([s for s, _ in p.stmts],
               r"\b(?:obligations?|provisions?|terms?|covenants?|agreement|duties|restrictions|undertakings?|sections?|paragraphs?|clauses?)\b.{0,100}\bsurviv(?:e|es|ing the)\b"
               r"|\bshall\s+(?:\w+\s+){0,2}survive\b|\bcontinue\w* in (?:full )?(?:force|effect)\b.{0,80}\b(?:after|following|notwithstanding|beyond)\b.{0,30}\b(?:terminat|expir|end of|return)"
               r"|\b(?:continue|remain)\w* (?:to be )?bound\b|\bnotwithstanding (?:the |any )?(?:termination|expiration|return|destruction)\b.{0,120}\b(?:obligations?|bound|continue|remain)"
               r"|\bobligations?\b.{0,80}\b(?:shall|will)\s+(?:continue|remain)\b.{0,80}\b(?:after|following|beyond)\b.{0,30}\b(?:terminat|expir)",
               exclude=r"\bsurviving (?:corporation|entity|company|party)|\bnon-?compet|\bcovenant not to compete|\bremain liable for (?:all )?antecedent")
    return ("yes", t) if t else None


RETAIN = (r"(?:except|provided|notwithstanding|however|save|other than)\b.{0,160}\b(?<!not )(?<!no )(?:retain|keep|archiv|back-?up|one (?:archival )?copy|"
          r"copy for (?:its )?(?:legal|records|files|archiv))|\bmay (?:retain|keep)\b|\bnot (?:be )?(?:required|obligated) to (?:delete|destroy|return|erase|purge)|"
          r"(?<!not )(?<!no )\bretain\w*\b.{0,60}\b(?:archival|back-?up|legal|compliance)|\binfeasible|\bnot (?:be )?feasible|\bimpracticable")
_NOT_RETAIN = r"\b(?:not|no|never)\s+(?:\w+\s+){0,2}(?:retain|keep)|\bretain\w* (?:all |any )?(?:rights?|title|ownership|interest)"


def nda_20(p: Program):  # may retain some CI after return/destruction
    st = [s for s, _ in p.stmts]
    t = _first(st, RETAIN, r"\bretur|\bdestr|\bdelet|\berase|\bcop(?:y|ies)|\bretain|\bkeep", exclude=_NOT_RETAIN)
    if t: return "yes", t
    t = _first(st, r"\b(?:return|destroy|deliver)\w*\b.{0,80}\ball\b", r"\ball (?:copies|reproductions)|\bretain no\b|\bshall not retain|\bnot (?:retain|keep) any")
    if t and not any(re.search(RETAIN, s, I) and not re.search(_NOT_RETAIN, s, I) for s in st) \
            and not any(re.search(r"\bexcept as (?:otherwise )?provided|\bsubject to (?:section|paragraph|clause)", s, I) and re.search(r"\bretur|\bdestr", s, I) for s in st):
        return "no", t
    return None


QUERIES = {"nda-1": nda_1, "nda-2": nda_2, "nda-3": nda_3, "nda-4": nda_4, "nda-5": nda_5, "nda-7": nda_7,
           "nda-11": nda_11, "nda-12": nda_12, "nda-13": nda_13, "nda-19": nda_19, "nda-20": nda_20}
