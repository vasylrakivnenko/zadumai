"""Clause queries over the compiled contract (front.compile_contract) for six CUAD types (2026-10-03).
Each type has graded variants, strictest first; a variant returns the statement it relies on. CV (cv.py) decides which
variants are trusted for "yes" (and the "no" variants: nothing that even mentions the topic anywhere).
Rules are written from the 326 "write" contracts only (explore.py); dev and test are never read."""
import re

I = re.I | re.S


def rx(p): return re.compile(p, I)


NEG = r"(?:shall|will|may|can|does|do)\s+not|cannot|can't|won't|shall\s+in\s+no\s+event|neither|\bno\b|\bnot\s+(?:to|be)\b|\bwithout\b|\bprohibit"

# ---------- Governing Law ----------
GL_VERB = r"\b(?:governed|construed|interpreted|enforced|governs?|construction|interpretation)\b"
GL_LAW = r"\b(?:laws?\s+(?:of|in)\s+(?:the\s+)?(?:state|commonwealth|province|republic|people'?s|federal|kingdom|united|(?-i:[A-Z]))|(?-i:(?!Applicable|Export|Securities|Tax|Data|Privacy|Environmental|Employment|Labou?r|Antitrust|Competition|Bankruptcy|Insolvency|Health|Patent|Copyright|Trademark|Foreign|Anti)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+laws?\b|laws?\s+(?:governing|applicable\s+to)\s+the\s+\w+\s+agreement)"
GL_CORE = rx(GL_VERB + r"[^.;]{0,160}?" + GL_LAW)
GL_AGR = rx(r"\b(?:this|the|such)\s+(?:\w+\s+){0,3}(?:agreement|contract|amendment|assignment|letter|guarant\w*|confirmation|license|addendum|terms)\b[^.;]{0,200}?\b(?:governed|construed|interpreted)\b|\b(?:governed|construed|interpreted)\b[^.;]{0,60}?\b(?:this|the)\s+(?:\w+\s+){0,2}(?:agreement|contract)\b")
GL_REV = rx(r"\blaws?\s+of\s+(?:the\s+)?(?:state|commonwealth|province|republic|people'?s|federal|kingdom|united|england|(?-i:[A-Z]))[^.;]{0,120}?\b(?:shall|will)\s+(?:govern|apply|control)|\blaws?\s+of\b[^.;]{0,80}\bgovern\s+(?:all|this|the)\b")
GL_HEAD = rx(r"governing|applicable law|choice of law|\blaw\b")
GL_MENTION = rx(r"\bgovern\w*\s+(?:by\s+)?(?:the\s+)?laws?\b|\blaws?\s+of\s+the\s+(?:state|commonwealth)|\bapplicable\s+law\b.{0,40}\bgovern|governing\s+law|choice\s+of\s+law|\bjurisdiction\b|\blaws?\s+of\b.{0,80}\b(?:govern|apply)")


def governing_law(st):
    out = []
    for s in st:
        if re.search(r"\bconflicts?\s+of\s+interest\b", s.own, re.I): continue
        if GL_CORE.search(s.own):
            v = "head+core" if GL_HEAD.search(s.head) else ("agr+core" if GL_AGR.search(s.own) else "core")
            out.append((v, s))
        elif GL_REV.search(s.own):
            out.append(("rev", s))
    return out


# ---------- Anti-Assignment ----------
ASSIGN = r"\b(?:assign\w*|transfer\w*|delegat\w*|cede|novat\w*)\b"
AA_OBJ = (r"\b(?:this\s+(?:agreement|contract|license|amendment)|(?:its|any|all|the|such)\s+(?:of\s+(?:its|the)\s+)?(?:\w+\s+){0,2}(?:rights?|obligations?|interests?|duties|benefits)"
          r"\s+(?:and\s+(?:\w+\s+){0,2})?(?:under|in|hereunder|of\s+this)|(?:rights?|obligations?|duties)\s+(?:hereunder|herein)|hereunder|this\s+agreement)\b")
GATE = r"\b(?:consent|approval|approve|agreement\s+of|permission|authori[sz]ation|notice|notif\w*|void|null|invalid|except|unless|other\s+than|provided|without)\b"
AA_NEG = r"(?:shall|will|may|can|does|do)\s+not|cannot|neither|\bnor\b|\bno\s+(?:party|right)|\bnot\s+(?:to|be)\b|\bprohibit\w*"
AA_CANON = rx(rf"(?:{AA_NEG})[^.]{{0,100}}?{ASSIGN}[^.]{{0,160}}?{AA_OBJ}|{AA_OBJ}[^.]{{0,120}}?\b(?:shall|may|will)\s+not\s+be\s+(?:\w+\s+){{0,2}}(?:assign\w*|transfer\w*|delegat\w*)|\b(?:not|non-?)\s*(?:be\s+)?(?:assignable|transferable)\b[^.]{{0,60}}?(?:by\s+(?:either|any|the)|without)")
AA_VOID = rx(r"\b(?:any|an|each)\s+(?:purported\s+|attempted\s+)?(?:assignment|transfer|delegation)\b[^.]{0,250}?\b(?:void|null|invalid|ineffective|of\s+no\s+(?:force|effect))")
AA_PERMIT = rx(rf"\bmay\s+(?:freely\s+)?(?:assign|transfer)\b[^.]{{0,160}}?(?:{AA_OBJ})[^.]{{0,200}}?\b(?:consent|notice|notif\w*|provided|condition|subject\s+to)\b")
AA_HEAD = rx(r"assign|transfer|successors")
AA_SKIP = re.compile(r"for the benefit of creditors|non-?transferable\b[^.]{0,80}\b(?:licen[cs]e|right\s+to\s+use)|\blicen[cs]e\b[^.]{0,60}\bnon-?transferable|\b(?:no|any)\s+(?:transfer|grant)\s+(?:or\s+grant\s+)?is\s+made|"
                     r"\btitle\b[^.]{0,60}\b(?:transfer|pass)|\brisk\s+of\s+loss|work\s+assigned|\bassigned\s+to\s+(?:perform|work)|\bassignments?\s+of\s+(?:inventions|intellectual|patent|copyright)|\bhereby\s+assigns?\b", re.I)
AA_MENTION = rx(r"\bassign(?:ment|ed|able|s)?\b|\btransfer\w*\b.{0,40}\b(?:agreement|rights|obligations)|\bdelegat|successors\s+and\s+assigns")


def anti_assignment(st):
    out = []
    for s in st:
        own = s.own
        if AA_SKIP.search(own) or len(own) > 1500: continue
        canon = AA_CANON.search(own)
        if canon and re.search(GATE, own, re.I):
            out.append(("head+canon" if AA_HEAD.search(s.head) else "canon", s))
        elif AA_VOID.search(own):
            out.append(("void", s))
        elif canon:
            out.append(("canon_nogate", s))
        elif AA_PERMIT.search(own):
            out.append(("permit_cond", s))
    return out


# ---------- Cap On Liability ----------
LIAB = r"\b(?:liabilit\w*|liable|damages|recover\w*)\b"
CAP_SUBJ = (r"(?:\b(?:aggregate|total|cumulative|maximum|entire|overall|collective)\s+(?:\w+\s+){0,3}liabilit\w*|\b\w+['’]s?\s+(?:\w+\s+){0,2}liabilit\w*|"
            r"\bliabilit\w*\s+of\s+(?:\w+\s+){0,3}\w+|\bin\s+no\s+event\s+(?:shall|will)\s+(?:\w+\s+){0,5}(?:liabilit\w*|be\s+liable)|\b(?:shall|will)\s+(?:only\s+)?be\s+liable\b|\bliabilit\w*\s+(?:hereunder|under\s+this))")
CAP_AMT = rx(CAP_SUBJ + r"[^.]{0,250}?\b(?:shall\s+not\s+exceed|not\s+to\s+exceed|will\s+not\s+exceed|exceed\w*|(?:shall\s+be|will\s+be|is|be|are)\s+limited\s+to|\blimited\s+to\s+(?:the\s+)?(?:amount|aggregate|total|fees|sum|\$|actual|direct)|in\s+excess\s+of|(?:be\s+)?equal\s+to|capped|greater\s+than|more\s+than|maximum)")
CAP_EXCL = rx(r"(?:neither\s+party|\bin\s+no\s+event|under\s+no\s+circumstances|(?:shall|will)\s+not\s+(?:be\s+)?(?:liable|responsible)|\bno\s+liability|\bnot\s+be\s+liable|\bwaives?\b)[^.]{0,250}?"
              r"\b(?:indirect|incidental|consequential|special|punitive|exemplary|multiple|lost\s+profits?|loss\s+of\s+(?:profits?|revenue|business|data|goodwill))\b")
CAP_TIME = rx(r"\b(?:claims?|actions?|suits?|proceedings?)\b[^.]{0,120}?\b(?:must\s+be|shall\s+be|may\s+(?:only\s+)?be|be)\s+(?:filed|brought|commenced|made|asserted|instituted)\b[^.]{0,80}?\bwithin\b|\bno\s+(?:claim|action|suit)\b[^.]{0,120}?\b(?:more\s+than|after)\b[^.]{0,40}\b(?:years?|months?|days?)")
CAP_HEAD = rx(r"limitation|limit\w*\s+(?:of|on)\s+liab|liabilit|damages")
CAP_SKIP = re.compile(r"\binsur\w*|\bpolic(?:y|ies)\b|\bcoverage\b|force\s+majeure|beyond\s+(?:its|their|the)\s+(?:\w+\s+)?control|criminal|arbitrat\w*|\bsolven|third[- ]party\s+beneficiar|\bcovenant\s+not\s+to\s+sue|more\s+than\s+one\s+(?:separate\s+)?(?:\w+\s+)?(?:counsel|firm|attorney)|\bswap\b|\bpurchase\s+orders?\b", re.I)
CAP_MENTION = rx(r"\bliab(?:le|ility)\b[^.]{0,200}(?:exceed|limited|consequential|indirect|special|punitive)|(?:consequential|indirect|punitive)\s+damages")


def cap_on_liability(st):
    out = []
    for s in st:
        own = s.own
        if CAP_SKIP.search(own) or len(own) > 2000: continue
        if CAP_AMT.search(own):
            out.append(("head+amount" if CAP_HEAD.search(s.head) else "amount", s))
        elif CAP_EXCL.search(own):
            out.append(("head+excl" if CAP_HEAD.search(s.head) else "excl", s))
        elif CAP_TIME.search(own):
            out.append(("time", s))
    return out


# ---------- Termination For Convenience ----------
TFC_AGR = r"(?:this|the|such)\s+(?:\w+\s+){0,2}(?:agreement|contract|term|relationship|arrangement|alliance|engagement|appointment|employment|license|statement\s+of\s+work|sow)\b"
TFC_CORE = rx(rf"\b(?:may|can|shall\s+(?:have|be\s+entitled)|has\s+the\s+right|have\s+the\s+right|is\s+entitled|be\s+entitled|reserves?\s+the\s+right|entitled|right)\b[^.]{{0,60}}?\b(?:terminat\w*|cancel\w*)\b[^.]{{0,40}}?{TFC_AGR}"
              rf"|{TFC_AGR}[^.]{{0,40}}?\b(?:may|can)\s+(?:\w+\s+){{0,3}}be\s+(?:terminated|cancell?ed)|\buntil\s+terminated\s+by\b")
TFC_FREE = rx(r"\bwithout\s+(?:cause|reason|penalty)\b|\bfor\s+(?:any\s+reason|any\s+or\s+no\s+reason|no\s+reason|convenience)\b|\bat\s+any\s+time\b|\bin\s+(?:its|their|his|her)\s+(?:sole\s+)?(?:and\s+absolute\s+)?(?:discretion|option)\b|\bwith\s+or\s+without\s+cause\b|\bat\s+will\b|\bfor\s+whatever\s+reason")
TFC_NOTICE = rx(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty-five|sixty|ninety|\[\*+\]|\*+|\[\.{3}\*+\.{3}\]|\[…\*+…\])\b[^.]{0,30}?\b(?:days?|months?|weeks?|years?)['’]?\s*(?:\([^)]*\)\s*)?(?:prior\s+|advance\s+|written\s+|\w+\s+){0,3}notice|\bnotice\b[^.]{0,40}?\b(?:days?|months?)\b")
TFC_CAUSE = rx(r"\b(?:breach\w*|default\w*|insolven\w*|bankrupt\w*|fail\w*|violat\w*|cure|change\s+(?:of|in)\s+control|force\s+majeure|upon\s+the\s+occurrence|in\s+the\s+event\b|if\b|should\b|unless|provided\s+that|in\s+accordance\s+with\s+(?:section|clause|paragraph)|"
               r"pursuant\s+to\s+(?:section|clause|paragraph)|under\s+(?:section|clause|paragraph)|expir\w*|cease|ceases|incapacit\w*|death|disab\w*|misconduct|for\s+cause|with\s+cause|good\s+reason|material|mutual\w*|"
               r"(?:by\s+)?(?:written\s+)?agreement\s+(?:in\s+writing\s+)?of\s+(?:all|both|the\s+parties)|prior\s+to\s+(?:the\s+)?(?:closing|effective\s+date)|at\s+or\s+prior\s+to)\b")
TFC_SKIP = re.compile(r"\bpurchase\s+orders?\b|\bpo['’]?s\b|\bsub-?servic|\bmodif(?:y|ied)\s+or\s+terminat|\b(?:means|defined)\b|\beffect\s+of\s+termination|\bupon\s+termination\b|\bfollowing\s+termination|\bafter\s+termination", re.I)
TFC_HEAD = rx(r"terminat|\bterm\b|convenience")
TFC_MENTION = rx(r"\bterminat\w*|\bcancel\w*")


def termination_convenience(st):
    out = []
    for s in st:
        own = s.own
        if len(own) > 1500 or TFC_SKIP.search(own) or not TFC_CORE.search(own) or re.search(r"(?:as\s+follows|following)\s*:?\s*$|:\s*$", own, re.I): continue
        free = TFC_FREE.search(own); notice = TFC_NOTICE.search(own); cause = TFC_CAUSE.search(own)
        strong = re.search(r"\bwithout\s+cause\b|\bfor\s+convenience\b|\bwith\s+or\s+without\s+cause", own, re.I) and not re.search(r"\bfor\s+cause\b|\bwith\s+cause\b|\bif\b", own, re.I)
        strong = strong or (re.search(r"\bfor\s+any\s+(?:or\s+no\s+)?reason|\bfor\s+no\s+reason", own, re.I) and not cause)
        if strong:
            out.append(("strong", s))
        elif free and not cause:
            out.append(("free", s))
        elif notice and not cause:
            out.append(("notice", s))
    return out


# ---------- Renewal Term ----------
REN_VERB = r"\b(?:renew\w*|extend\w*|extension|continu\w*|roll\w*\s+over|prolong\w*)\b"
REN_MORE = r"\b(?:additional|further|successive|subsequent|another|consecutive|like|similar)\b[^.]{0,40}?\b(?:periods?|terms?|years?|months?)\b|\b(?:from\s+)?(?:year\s+to\s+year|month\s+to\s+month)\b|\bon\s+(?:each|an?)\s+(?:annual|yearly)\s+basis\b|\beach\s+(?:year|anniversary)\b"
REN_AUTO = rx(r"\b(?:automatic(?:ally)?|auto-?)\s*(?:be\s+)?(?:renew\w*|extend\w*|continu\w*)|\b(?:renew\w*|extend\w*|continu\w*|prolong\w*)\s+(?:\w+\s+){0,2}automatically|\bautomatic\s+(?:renewal|extension)"
              r"|" + REN_VERB + r"[^.]{0,80}?(?:" + REN_MORE + r")")
REN_OPT = rx(r"\b(?:option|right)\s+to\s+(?:renew|extend)\b[^.]{0,60}?\b(?:agreement|term|for)\b|\bmay\s+(?:be\s+)?(?:renew\w*|extend\w*)\b[^.]{0,40}?\b(?:agreement|term|for\s+(?:an?\s+)?(?:additional|further|successive|another))|\brenewal\s+(?:term|period)s?\b")
REN_SKIP = re.compile(r"\bmeans\s+(?:each|the|a)\b|\b(?:calendar|fiscal)\s+(?:quarter|year)\b|\blicen[cs]e\s+renewal|\brenew(?:al)?\s+(?:of\s+)?(?:the\s+)?(?:insurance|polic|permit|registration)|\bincreases?\b|\bshall\s+not\s+cause", re.I)
REN_HEAD = rx(r"\bterm\b|renewal|duration")
REN_MENTION = rx(r"\brenew\w*|\bextend\w*|\bextension\b|\bsuccessive\b|year\s+to\s+year")


REN_SUBJ = rx(r"\b(?:this\s+(?:agreement|contract)|the\s+(?:agreement|contract)|(?:the\s+)?(?:initial\s+)?term(?:\s+of\s+this\s+(?:agreement|contract))?|the\s+(?:initial\s+)?(?:distribution\s+)?period)\b[^.]{0,160}?\b(?:renew\w*|extend\w*|extension|continu\w*|prolong\w*|roll\w*)\b"
               r"|\b(?:renew\w*|extend\w*|continu\w*)\b[^.]{0,40}?\b(?:this\s+agreement|the\s+(?:initial\s+)?term|the\s+agreement)\b")
REN_NOT = re.compile(r"\bmutual(?:ly)?\b|\bwaiver\b|\bapproved\s+(?:at\s+least\s+)?annually|\bby\s+the\s+number\s+of\s+days|\bfailure\s+to\s+supply|\btoll\w*", re.I)


def renewal_term(st):
    out = []
    for s in st:
        own = s.own
        if len(own) > 1500 or REN_SKIP.search(own): continue
        if not REN_SUBJ.search(own) or REN_NOT.search(own): continue
        if REN_AUTO.search(own):
            out.append(("head+auto" if REN_HEAD.search(s.head) else "auto", s))
        elif REN_OPT.search(own):
            out.append(("option", s))
    return out


# ---------- Change Of Control ----------
COC_WORDS = rx(r"\bchange\s+(?:of|in)\s+(?:the\s+)?(?:control|ownership)\b|\bchange\s+(?:of|in)\s+(?:its|the)\s+(?:\w+\s+){0,2}(?:control|ownership)\b")
COC_EVENT = rx(r"\b(?:merger|merges?|merged|consolidat\w*\s+with)\b|\b(?:all\s+or\s+substantially\s+all)\s+of\s+(?:its|the|such)\b|\bcontrolling\s+(?:interest|stake)\b|\b(?:majority|fifty\s+percent|50%)\s+(?:or\s+more\s+)?of\s+(?:the\s+|its\s+)?(?:\w+\s+){0,2}(?:voting|outstanding|equity|shares|stock)|\bacqui\w+\s+(?:by|of)\b[^.]{0,60}\b(?:control|voting|shares|stock|equity)|\bis\s+acquired\b")
COC_EFFECT = rx(r"\bterminat\w*|\bconsent\b|\bapproval\b|\bnotice\b|\bnotif\w*|\bdeemed\s+(?:to\s+be\s+)?(?:an?\s+)?(?:assignment|transfer)\b|\bassign\w*|\btransfer\w*")
COC_SKIP = re.compile(r"\bentire\s+agreement\b|\bsupersede|\bmerges?\s+all\s+prior|\bmerger\s+clause|\bbankrupt|\binsolven|\breceiver\b|\breorgani[sz]ation\s+under|\bseverance\b|\bchange\s+in\s+control\s+period|\bmeans\b|\bcreditors\b", re.I)
COC_HEAD = rx(r"change\s+(?:of|in)\s+control|assign|transfer|terminat|merger")
COC_MENTION = rx(r"\bchange\s+(?:of|in)\s+(?:control|ownership)|\bmerg|\bconsolidat|substantially\s+all|operation\s+of\s+law|controlling\s+interest|\bacqui")


def change_of_control(st):
    out = []
    for s in st:
        own = s.own
        if len(own) > 1500 or len(own) < 50 or COC_SKIP.search(own) or not COC_EFFECT.search(own): continue
        if re.search(r"[\"“]\s*(?:change\s+(?:of|in)\s+control|asset\s+transfer|acquisition)\w*\s*[\"”]\s*(?:shall\s+)?means?|\bshall\s+mean\b", s.text, re.I): continue
        cw = COC_WORDS.pattern; ef = COC_EFFECT.pattern
        if re.search(rf"(?:{cw})[^.]{{0,150}}?(?:{ef})|(?:{ef})[^.]{{0,150}}?(?:{cw})", own, re.I):
            out.append(("coc_words", s))
        elif COC_WORDS.search(own):
            out.append(("coc_far", s))
        elif COC_EVENT.search(own):
            out.append(("head+event" if COC_HEAD.search(s.head) else "event", s))
    return out


QUERIES = {
    "Governing Law": (governing_law, GL_MENTION),
    "Anti-Assignment": (anti_assignment, AA_MENTION),
    "Cap On Liability": (cap_on_liability, CAP_MENTION),
    "Termination For Convenience": (termination_convenience, TFC_MENTION),
    "Renewal Term": (renewal_term, REN_MENTION),
    "Change Of Control": (change_of_control, COC_MENTION),
}


RANK = {  # within a variant, the statement most like the textbook clause first
    "Anti-Assignment": lambda s: 2 * bool(re.search(r"\b(?:assign\w*|transfer\w*)\s+(?:\w+\s+){0,2}(?:this|the)\s+agreement\b|\bagreement\s+(?:\w+\s+){0,4}(?:may|shall|will)\s+not\s+be\s+(?:\w+\s+){0,2}assign", s.own, re.I))
                          + bool(AA_HEAD.search(s.head)),
    "Governing Law": lambda s: bool(GL_HEAD.search(s.head)),
    "Cap On Liability": lambda s: bool(CAP_HEAD.search(s.head)),
    "Termination For Convenience": lambda s: bool(TFC_HEAD.search(s.head)),
    "Renewal Term": lambda s: bool(REN_HEAD.search(s.head)),
    "Change Of Control": lambda s: bool(COC_HEAD.search(s.head)),
}


def answers(stmts, t):
    """{variant: ("yes", stmt) | ("no", None)} for one contract and one type."""
    fn, mention = QUERIES[t]; out = {}; best = {}
    for k, (v, s) in enumerate(fn(stmts)):  # per variant, the statement about the main agreement first, then the earliest
        main = bool(re.search(r"\b(?:this|the)\s+agreement\b", s.own, re.I)) and not re.search(
            r"\bthis\s+(?:letter|assignment|guarant\w*|instrument|amendment|addendum|confirmation|exhibit|schedule)\b", s.own, re.I)
        key = (-RANK.get(t, lambda x: 0)(s), not main, k)
        if v not in best or key < best[v][0]: best[v] = (key, s)
    for v, (_, s) in best.items():
        out[v] = ("yes", s)
    if not any(mention.search(s.own) for s in stmts): out["no_mention"] = ("no", None)
    elif not out: out["no_rule"] = ("no", None)
    return out
