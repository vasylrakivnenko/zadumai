"""A compiler-style front end for contracts (2026-10-03; the user: "borrow from how compilers work").
  lexing + structure parsing: lines -> numbered/bulleted blocks -> a tree (a marker class seen before closes the deeper
      levels, like an indentation stack); a block that ends in ":" (or "shall", "agrees to", ...) is a lead-in
  statements: every leaf with its lead-ins prefixed ("Recipient shall not: (a) disclose ..." -> "Recipient shall not
      disclose ...") - one self-contained statement per item, so a fragment never stands alone
  symbol table: defined terms ("X" means / includes / refers to ...; ... (the "X")), each with its body statements
  the Confidential Information definition: its body and its exclusions list ("shall not include", "obligations shall
      not apply to", ...)
Deterministic; what doesn't parse stays plain text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_MARK = re.compile(
    r"^\s*(?:(?:section|article|clause|paragraph)\s+)?"
    r"(?P<m>\d+(?:\.\d+)+\.?|\d+\.|\d+\)|\(\d+\)|\([a-z]{1,2}\)|\([ivxlc]{1,6}\)|[a-z]\)|[a-z]\.(?=\s)|[ivxlc]{1,6}\.(?=\s)"
    r"|\([A-Z]\)|[A-Z]\.(?=\s)|[•●▪◦\-–—*·]|☐|□)\s+", re.I)
_TERMINAL = re.compile(r"[.;:!?]\s*$|[.;:]\s*(?:and|or)\s*$", re.I)
_LEADIN = re.compile(r"(?::|—|–|-|\bshall|\bwill|\bagrees?(?: to)?|\bnot|\bto|\bthat|\bmeans|\binclude[s]?|\bincluding|"
                     r"\bfollowing|\bor|\band|\bexcept|\bprovided)\s*$", re.I)


def _mclass(m: str) -> str:
    m = m.strip()
    if re.fullmatch(r"\d+(?:\.\d+)+\.?", m): return f"num{m.rstrip('.').count('.') + 1}"
    if re.fullmatch(r"\d+[.)]|\(\d+\)", m): return "num1" if m.endswith(".") else "paren_num"
    if re.fullmatch(r"\([ivxlc]+\)|[ivxlc]+\.", m, re.I) and m.strip("().").lower() not in ("c", "d", "i", "l", "v", "x") or m.lower() in ("(i)", "i.", "(ii)", "ii."):
        return "roman"
    if re.fullmatch(r"\([a-z]{1,2}\)|[a-z][.)]", m): return "alpha"
    if re.fullmatch(r"\([A-Z]\)|[A-Z]\.", m): return "ALPHA"
    return "bullet"


@dataclass
class Node:
    text: str
    mclass: str = "root"
    marker: str = ""
    children: list = field(default_factory=list)

    @property
    def leadin(self) -> bool:
        return bool(self.children) and bool(_LEADIN.search(self.text) or not _TERMINAL.search(self.text))


def blocks(text: str) -> list:
    """Lines joined back into blocks: a line without a marker continues the previous one if that didn't end a sentence."""
    out = []
    for raw in text.replace("\r", "").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line: continue
        m = _MARK.match(line)
        if m: out.append([m.group("m"), line[m.end():].strip()])
        elif out and not _TERMINAL.search(out[-1][1]) and not line[:1].isupper(): out[-1][1] += " " + line
        elif out and not _TERMINAL.search(out[-1][1]) and len(out[-1][1]) > 60: out[-1][1] += " " + line
        else: out.append(["", line])
    return out


def parse(text: str) -> Node:
    root = Node("", "root"); stack = [root]
    for marker, body in blocks(text):
        if not marker:
            top = stack[-1]
            last = top.children[-1] if top.children else None
            if (last is not None and last.mclass == "text" and last.text.rstrip().endswith(":")
                    and (not last.children or re.search(r"(?:[;,:]|\b(?:and|or))\s*$", last.children[-1].text))):
                last.children.append(Node(body, "text_item"))  # "does not include:" + plain lines: its items, until one ends a sentence
            elif (last is not None and last.mclass == "text" and last.children
                  and re.search(r"(?:[;,]|\b(?:and|or))\s*$", last.children[-1].text)):  # the list goes on only after ";", ",", "and", "or"
                last.children.append(Node(body, "text_item"))
            else:
                top.children.append(Node(body, "text"))
            continue
        c = _mclass(marker); n = Node(body, c, marker)
        classes = [s.mclass for s in stack]
        if c in classes:  # a sibling of an open level: close everything below it
            del stack[classes.index(c):]
        else:  # a new list right after an unmarked lead-in ("The Recipient shall not:") hangs under it (2026-10-03)
            top = stack[-1]; last = top.children[-1] if top.children else None
            if (last is not None and last.mclass == "text" and last.text.rstrip().endswith(":") and not last.children
                    and not c.startswith("num")  # top-level section numbering never hangs under a sentence
                    and not re.search(r"as follows|witnesseth|now,? therefore|\bwhereas\b|in consideration of|hereby agree", last.text, re.I)):
                stack.append(last)
        stack[-1].children.append(n); stack.append(n)
    return root


_INLINE = re.compile(r"(?:(?<=[:;])|(?<=;\s)|(?<=:\s)|(?<=\bor\s)|(?<=\band\s))\s*\((?:[a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)\s+", re.I)


_INLINE_MARK = re.compile(r"(?:(?<=\s)|^)\((?P<l>[a-z]|[ivx]{1,4}|\d{1,2})\)\s+", re.I)
_FIRST = {"a", "i", "1"}


def _inline_split(text: str):
    """'lead: (a) x; (b) y' or 'lead (a) x; (b) y' -> ('lead', ['x', 'y']), else None (2026-10-03: no colon needed)."""
    ms = list(_INLINE_MARK.finditer(text))
    if len(ms) < 2 or ms[0].group("l").lower() not in _FIRST: return None
    lead = text[:ms[0].start()].rstrip(" :—-")
    items = [text[m.end():(ms[k + 1].start() if k + 1 < len(ms) else len(text))].strip(" ;,") for k, m in enumerate(ms)]
    items = [re.sub(r"\s*(?:;|,)?\s*(?:and|or)\s*$", "", x) for x in items if x]
    return (lead, items) if lead and len(items) >= 2 else None


def statements(root: Node) -> list:
    """Every statement with its lead-ins: [(text, path of markers)]."""
    out = []
    def walk(n: Node, lead: list, path: list):
        here = [*lead]
        if n.text:
            if n.leadin: here.append(n.text.rstrip(" :—–-"))
            elif (sp := _inline_split(n.text)):
                for item in sp[1]: out.append((" ".join([*lead, sp[0], item]).strip(), path))
            else:
                for s in re.split(r"(?<=[.;])\s+(?=[A-Z(\"“])", n.text):
                    out.append((" ".join([*lead, s]).strip(), path))
        for ch in n.children:
            walk(ch, here if n.leadin or n.mclass == "root" else lead, [*path, ch.marker])
        if n.text and n.leadin and not n.children:
            out.append((" ".join(here).strip(), path))
    walk(root, [], [])
    return [(re.sub(r"\s+", " ", t), p) for t, p in out if len(t) > 3]


_DEF = re.compile(r"[\"“”']{1,2}(?P<t>[A-Z][\w&\-’' ]{1,60}?)[\"“”']{1,2}\s*(?:,\s*)?(?:shall\s+|will\s+|as used herein,?\s+)?"
                  r"(?:mean|means|refers? to|include|includes|shall include|is defined as|has the meaning|consists? of)\b", re.I)
_PAREN = re.compile(r"\((?:the\s+|collectively,?\s+|hereinafter\s+(?:referred to as\s+)?|herein\s+|each\s+a\s+|individually\s+)?"
                    r"[\"“”']{1,2}(?P<t>[A-Z][\w&\-’' ]{1,50}?)[\"“”']{1,2}\)")
_DEF_BARE = re.compile(r"(?:^|[.:;]\s+|\b(?:the\s+(?:term|expression|words?)|for (?:the )?purposes? of this agreement,?)\s+)"
                       r"(?P<t>(?i:the\s+)?(?:Confidential Information|Proprietary Information|Evaluation Materials?|Confidential Materials?|"
                       r"CONFIDENTIAL INFORMATION|PROPRIETARY INFORMATION))\s+(?:as used (?:herein|in this agreement),?\s+)?(?:shall\s+|will\s+)?"
                       r"(?i:mean|means|refers? to|include|includes|is defined as|consists? of)\b")
CI_TERMS = re.compile(r"confidential information|proprietary information|evaluation material|confidential material|information", re.I)
_EXCL = re.compile(r"\bshall not be (?:held )?liable for (?:the )?(?:any )?(?:disclosure|use)|\bno (?:liability|obligation)s?\b.{0,60}\b(?:with respect to|for|regarding|if|in respect)|"
                   r"\binoperative\b|\bimposes? no obligation|\bexcludes?\b|\bnot (?:be )?subject to\b|\bexceptions? (?:to|from)\b|\bshall not be bound\b|"
                   r"\b(?:shall|does|will|do|should)\s+not\s+(?:include|apply|extend|cover|be (?:deemed|considered|construed))|"
                   r"\bnot include\b|\bexclud|\bno obligation|\bshall have no (?:obligation|liability)|\bdo(?:es)? not apply|"
                   r"\bshall not be (?:deemed|considered) (?:to be )?confidential|\bobligations\b.{0,80}\bshall not apply", re.I)


@dataclass
class Program:
    root: Node
    stmts: list
    symbols: dict  # term -> list of body statements
    ci_term: str | None
    ci_body: list
    ci_exclusions: list


def compile_doc(text: str) -> Program:
    root = parse(text); st = statements(root); sym = {}
    for t, _ in st:
        for m in _DEF.finditer(t):
            sym.setdefault(m.group("t").strip(), []).append(t)
        for m in _PAREN.finditer(t):
            sym.setdefault(m.group("t").strip(), []).append(t[:m.start()])
        if not _DEF.search(t):
            for m in _DEF_BARE.finditer(t):
                k = " ".join(w.capitalize() for w in re.sub(r"^(?i:the)\s+", "", m.group("t")).split())
                sym.setdefault(k, []).append(t)
    terms = [k for k in sym if CI_TERMS.search(k)]
    ci = next((k for k in terms if re.search(r"confidential information", k, re.I)), terms[0] if terms else None)
    body = sym.get(ci, [])
    excl = [t for t, _ in st if _EXCL.search(t) and re.search(r"\binformation|\bmaterials?\b|\bobligation|\brestriction", t, re.I)
            and not re.search(r"\bwaiver|\bwaive|\bsevera|\bconstrued (?:for|against)|\bassign", t, re.I)]
    return Program(root, st, sym, ci, body, excl)


def resolve(prog: Program, phrase: str, depth: int = 2) -> str:
    """The phrase with every defined term in it followed by its definition (one or two levels)."""
    out = phrase
    for k, bodies in prog.symbols.items():
        if depth and len(k) > 3 and re.search(r"\b" + re.escape(k) + r"\b", phrase):
            out += " [" + k + ": " + " ".join(bodies[:2])[:600] + "]"
    return out
