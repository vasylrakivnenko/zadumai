"""Document compiler (2026-10-03; the user: "let's build it"): one pass over a document at upload, shared by Pre-Tier 0
and the contract map, cached per document. Self-contained so it can move to router/ as is; the parser and heading rules
come from cnli/compiler.py and cuadc/front.py (those copies stay frozen with their recorded results).
Passes:
  1. preprocess  page furniture blanked with spaces, so every offset still points into the original text: EDGAR
                 footers ("Source: PF HOSPITALITY GROUP INC., 10-12G, 9/23/2015", in 145 of CUAD's 408 tuning contracts,
                 which the old front end read as section headings), page numbers ("- 7 -", "Page 3 of 12", "A-2", a run
                 of bare page numbers), running headers (a short line repeated on 4+ pages), a table of contents
  2. parse       lines -> numbered / lettered / bulleted blocks -> a tree (a marker class seen before closes the deeper
                 levels; a block ending in ":" / "shall" / ... is a lead-in)
  3. statements  every item with its lead-ins prefixed, its own span and its paragraph's span in the original text, and
                 the sections it sits in (outermost first)
  4. sections    every marked block and every heading's scope: number ("12.1", "12.1(a)"), heading, span, parent
  5. symbols     defined terms ("X" means ...; ... (the "X")) with the text that defines them
  6. parties     the preamble's parties and what the contract calls each ("Licensee", "Disclosing Party", "ABC")
  7. xrefs       "Section 9.2", "Article V", "Exhibit B" resolved to a section when the document has it; other documents
                 the contract leans on ("as defined in the Master Agreement", "incorporated herein by reference")
Deterministic; what doesn't parse stays plain text."""
from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass, field

# ---------- 1. preprocess ----------
_SOURCE = re.compile(r"^[ \t]*Source:[ \t]+[^\n]{2,160}?,[ \t]*[\w\-/ ]{1,16},[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]*$", re.M)
_PAGE = re.compile(r"^[ \t]*(?:-[ \t]*\d{1,3}[ \t]*-|Page[ \t]+\d{1,3}(?:[ \t]+of[ \t]+\d{1,3})?|Page[ \t]*-[ \t]*\d{1,3}[ \t]*-"
                   r"|[A-Z]{1,2}[ \t]*-[ \t]*\d{1,3}(?:[ \t]*-[ \t]*\d{1,3})?)[ \t]*$", re.M)
_BARE_NUM = re.compile(r"^[ \t]*(\d{1,3})[ \t]*$", re.M)
_NOT_HEADER = re.compile(r"^(?:by|name|title|date|its|signature|address|witness|attn|attention|e-?mail|fax|phone|telephone|tel)\b"
                         r"|:\s*$|\[\s*\*|\*\*\*|^\W*$", re.I)


def preprocess(text: str) -> tuple:
    """(text with page furniture blanked by spaces and line breaks restored, [(start, end, what)]). Same length as the
    input, so offsets into it are offsets into the original."""
    text = _GAP.sub(lambda m: "\n" + m.group(0)[1:], text)  # PDF text: a run of 3+ spaces is a line break
    spans = [(m.start(), m.end(), "footer") for m in _SOURCE.finditer(text)]
    spans += [(m.start(), m.end(), "page") for m in _PAGE.finditer(text)]
    bare = [(m.start(), m.end(), int(m.group(1))) for m in _BARE_NUM.finditer(text)]
    if len(bare) >= 2:  # bare numbers are page numbers when they count up one by one, far apart
        steps = [b[2] - a[2] == 1 for a, b in zip(bare, bare[1:])]
        gaps = sorted(b[0] - a[0] for a, b in zip(bare, bare[1:]))
        if sum(steps) / len(steps) >= 0.7 and gaps[len(gaps) // 2] >= 1000:
            spans += [(s, e, "page") for s, e, _ in bare]
    lines, pos = [], 0
    for raw in text.split("\n"):
        lines.append((pos, pos + len(raw), re.sub(r"\s+", " ", raw).strip())); pos += len(raw) + 1
    seen = {}  # running headers: the same short line on 4+ pages
    for s, e, l in lines:
        if 3 <= len(l) <= 100 and len(re.findall(r"[A-Za-z]{2,}", l)) >= 2 and not _NOT_HEADER.search(l) and not _MARK.match(l):
            seen.setdefault(l, []).append((s, e))
    for l, occ in seen.items():
        if len(occ) >= 4 and min(b[0] - a[0] for a, b in zip(occ, occ[1:])) >= 300:  # 3 can be a party's name
            spans += [(s, e, "header") for s, e in occ]
    spans += _toc(lines)
    for m in _TRAIL_PAGE.finditer(text):  # a page number at the end of a page-long line ("... above. C - 2")
        spans.append((m.start("p"), m.end("p"), "page"))
    trail = [(m.start("p"), m.end("p"), int(m.group("p"))) for m in _TRAIL_NUM.finditer(text)]
    if len(trail) >= 3 and sum(b[2] - a[2] == 1 for a, b in zip(trail, trail[1:])) / (len(trail) - 1) >= 0.7:
        spans += [(s, e, "page") for s, e, _ in trail]
    spans.sort(); merged = []
    for sp in spans:  # one span per stretch of text
        if merged and sp[0] < merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], sp[1]), merged[-1][2])
        else: merged.append(sp)
    out = list(text)
    for s, e, _ in merged:
        out[s:e] = " " * (e - s)
    return _split_inline("".join(out)), merged


_TOC_HEAD = re.compile(r"^(?:TABLE OF CONTENTS|Table of Contents|CONTENTS|Contents|INDEX|Index)\s*:?$")
_TOC_LINE = re.compile(r"(?:\.{3,}|…|\s)\s*(?:\d{1,3}|[ivxlc]{1,5}|[A-Z]-\d{1,3})$|^(?:ARTICLE|Article|SECTION|Section|EXHIBIT|Exhibit|SCHEDULE|Schedule)\b")


def _toc(lines: list) -> list:
    """A table of contents: its heading and the entries after it (a page number at the end, or a heading-like line);
    it ends at the first line that reads like text. Its entries would otherwise be parsed as sections."""
    out = []
    for k, (s, e, l) in enumerate(lines):
        if not _TOC_HEAD.match(l): continue
        last, misses = k, 0
        for j in range(k + 1, min(len(lines), k + 400)):
            l2 = lines[j][2]
            if not l2: continue
            if len(l2) <= 160 and (_TOC_LINE.search(l2) or head_only(l2)) and not re.search(r"[a-z]\.\s+[A-Z]", l2):
                last, misses = j, 0
            else:
                misses += 1  # one stray short line (a page numeral) is allowed; text ends it
                if misses > 1 or len(l2) > 160 or re.search(r"[a-z]\.\s+[A-Z]|[a-z]{3,}[.;]$", l2): break
        if last - k >= 3: out.append((s, lines[last][1], "toc"))
    return out


_GAP = re.compile(r"(?<=\S)[ \t]{3,}(?=\S)")


_TRAIL_PAGE = re.compile(r"\S[ \t]+(?P<p>[A-Z]{1,2}[ \t]?-[ \t]?\d{1,3}|-[ \t]?\d{1,3}[ \t]?-|Page[ \t]+\d{1,3}(?:[ \t]+of[ \t]+\d{1,3})?)[ \t]*$", re.M)
_TRAIL_NUM = re.compile(r"(?<=[^\n]{200})[.;:)\"”][ \t]+(?P<p>\d{1,3})[ \t]*$", re.M)
_INLINE_SEC = re.compile(r"(?:[.;:!?)\"”’]|^)(?P<ws>[ \t]+)(?:(?:Section|SECTION|Article|ARTICLE)[ \t]+)?"
                         r"(?P<n>\d{1,2}(?:\.\d{1,2}){1,3}\.?|\d{1,2}\.)(?!\d)[ \t]+(?=[A-Z(\"“])", re.M)
_LINE_SEC = re.compile(r"^[ \t]*(?:(?:Section|SECTION|Article|ARTICLE)[ \t]+)?(?P<n>\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+\S", re.M)


def _next_ok(n: tuple, cur) -> bool:
    """Does section number n continue the numbering after cur? (a sibling at some level, or a first child)"""
    if cur is None: return n[0] <= 2 and all(x <= 1 for x in n[1:])
    if n == cur + (1,) or n == cur + (0,): return True
    for d in range(len(cur)):
        for step in (1, 2):
            if n == cur[:d] + (cur[d] + step,) or n == cur[:d] + (cur[d] + step, 1): return True
    return False


def _split_inline(text: str) -> str:
    """A numbered section that starts inside a line ("... remedies. 4. Severability. All ...") gets its own line: the space
    before its number becomes a newline (offsets unchanged). Only numbers that continue the document's numbering count."""
    cands = [(m.start("n"), m.start("ws"), m.group("n"), False) for m in _INLINE_SEC.finditer(text) if m.start("ws") > 0
             and text[m.start("ws") - 1] != "\n"]
    cands += [(m.start("n"), -1, m.group("n"), True) for m in _LINE_SEC.finditer(text)]
    cands.sort()
    out = list(text); cur = None
    for pos, ws, n, at_line in cands:
        t = tuple(int(x) for x in n.rstrip(".").split("."))
        if at_line:
            if _next_ok(t, cur) or cur is None: cur = t
            continue
        if _next_ok(t, cur):
            out[ws] = "\n"; cur = t
    return "".join(out)


# ---------- 2. parse (cnli/compiler.py) ----------
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


_GLUED = re.compile(r"^(?P<m>\d{1,2}\.)(?=[A-Z][a-z])")
_HEADLIKE = re.compile(r"^(?:ARTICLE|Article|SECTION|EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|APPENDIX)\b|^[A-Z][A-Z0-9 ,;'’&/\-]{3,80}$")


def blocks(text: str) -> list:
    """Lines joined back into blocks: a line without a marker continues the previous one if that didn't end a sentence."""
    out = []
    for raw in text.replace("\r", "").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line: continue
        m = _MARK.match(line) or _GLUED.match(line)  # "5.Term of this Agreement": a number glued to its heading
        if m: out.append([m.group("m"), line[m.end():].strip()])
        elif out and not _TERMINAL.search(out[-1][1]) and not line[:1].isupper(): out[-1][1] += " " + line
        elif out and not _TERMINAL.search(out[-1][1]) and len(out[-1][1]) > 60 and not _HEADLIKE.match(line) \
                and not head_only(out[-1][1]): out[-1][1] += " " + line
        else: out.append(["", line])
    return out


def parse(text: str) -> Node:
    root = Node("", "root"); stack = [root]
    for marker, body in blocks(text):
        if not marker:
            top = stack[-1]
            last = top.children[-1] if top.children else None
            if _HEADLIKE.match(body) and head_only(body):  # "ARTICLE 2 ...", "GOVERNING LAW": a top-level heading
                del stack[1:]; root.children.append(Node(body, "text"))
            elif (last is not None and last.mclass == "text" and last.text.rstrip().endswith(":")
                    and (not last.children or re.search(r"(?:[;,:]|\b(?:and|or))\s*$", last.children[-1].text))):
                last.children.append(Node(body, "text_item"))
            elif (last is not None and last.mclass == "text" and last.children
                  and re.search(r"(?:[;,]|\b(?:and|or))\s*$", last.children[-1].text)):
                last.children.append(Node(body, "text_item"))
            else:
                top.children.append(Node(body, "text"))
            continue
        c = _mclass(marker); n = Node(body, c, marker)
        classes = [s.mclass for s in stack]
        if c in classes:
            del stack[classes.index(c):]
        else:
            top = stack[-1]; last = top.children[-1] if top.children else None
            if (last is not None and last.mclass == "text" and last.text.rstrip().endswith(":") and not last.children
                    and not c.startswith("num")
                    and not re.search(r"as follows|witnesseth|now,? therefore|\bwhereas\b|in consideration of|hereby agree", last.text, re.I)):
                stack.append(last)
        stack[-1].children.append(n); stack.append(n)
    return root


_INLINE_MARK = re.compile(r"(?:(?<=\s)|^)\((?P<l>[a-z]|[ivx]{1,4}|\d{1,2})\)\s+", re.I)
_FIRST = {"a", "i", "1"}


def _inline_split(text: str):
    """'lead: (a) x; (b) y' or 'lead (a) x; (b) y' -> ('lead', ['x', 'y']), else None."""
    ms = list(_INLINE_MARK.finditer(text))
    if len(ms) < 2 or ms[0].group("l").lower() not in _FIRST: return None
    lead = text[:ms[0].start()].rstrip(" :—-")
    items = [text[m.end():(ms[k + 1].start() if k + 1 < len(ms) else len(text))].strip(" ;,") for k, m in enumerate(ms)]
    items = [re.sub(r"\s*(?:;|,)?\s*(?:and|or)\s*$", "", x) for x in items if x]
    return (lead, items) if lead and len(items) >= 2 else None


# ---------- headings (cuadc/front.py) ----------
_SMALL = r"(?:of|and|or|to|the|for|in|on|by|a|an|with|upon|&|/)"
_WORD = r"(?:[A-Z][\w'’&/\-]*|" + _SMALL + r")"
INLINE_HEAD = re.compile(r"^(?P<h>[A-Z][\w'’&/\-]*(?:[ ,;]+" + _WORD + r"){0,8}?)\s*(?:\.|:|—|–|-)\s+(?=[A-Z(\"“])")
CAPS_HEAD = re.compile(r"^(?P<h>[A-Z][A-Z0-9 ,;'’&/\-]{2,70}?)\s*(?:\.|:|—|–|-)?\s+(?=[A-Z][a-z]|\(|[\"“])")


def head_only(text: str) -> str | None:
    t = text.strip().rstrip(".:").strip()
    if not t or len(t) > 80 or re.search(r"[.;]\s|\bshall\b|\bwill\b|\bmay\b", t): return None
    words = re.findall(r"[A-Za-z][\w'’\-]*", t)
    if not words or len(words) > 10: return None
    caps = sum(w[0].isupper() for w in words if not re.fullmatch(_SMALL, w))
    return t if caps >= max(1, len([w for w in words if not re.fullmatch(_SMALL, w)]) - 1) else None


def inline_head(text: str) -> str | None:
    m = INLINE_HEAD.match(text) or CAPS_HEAD.match(text)
    if not m: return None
    h = m.group("h").strip(" ,;")
    words = [w for w in re.findall(r"[A-Za-z][\w'’\-]*", h) if not re.fullmatch(_SMALL, w)]
    if not words or len(words) > 8 or not all(w[0].isupper() for w in words): return None
    if re.fullmatch(r"(?:The|This|Each|Either|Neither|No|Any|All|In|If|Upon|Notwithstanding|Except|Subject|Such|"
                    r"Company|Licensee|Licensor|Party|Parties)", words[0]) and len(words) == 1: return None
    return h


SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(\"“]|\d+(?:\.\d+)*\.?\s)")


class Located:
    """Whitespace-normalized text with a map back to the original offsets."""
    def __init__(self, text: str):
        self.idx = []; out = []; prev_space = True
        for i, ch in enumerate(text):
            if ch.isspace():
                if prev_space: continue
                out.append(" "); self.idx.append(i); prev_space = True
            else:
                out.append(ch); self.idx.append(i); prev_space = False
        self.norm = "".join(out); self.cursor = 0

    def find(self, s: str):
        s = re.sub(r"\s+", " ", s).strip()
        if not s: return -1, -1
        k = self.norm.find(s, self.cursor)
        if k < 0: k = self.norm.find(s)
        if k < 0: return -1, -1
        self.cursor = k + 1
        return self.idx[k], self.idx[k + len(s) - 1] + 1


# ---------- 3-4. statements and sections ----------
@dataclass
class Section:
    id: int
    number: str      # "12", "12.1", "12.1(a)", "V" (articles), "" for an unnumbered heading's scope
    heading: str     # as written, "" if none
    parent: int      # -1 at the top
    start: int = -1  # span in the original text (from its statements)
    end: int = -1
    front: bool = False  # a title-page line ("Exhibit 10.13", "LICENSE AGREEMENT"): not a heading of what follows


@dataclass
class Stmt:
    text: str        # with lead-ins
    own: str         # its own words
    start: int       # span of `own` in the original text (-1 if not found)
    end: int
    secs: tuple      # section ids, outermost first
    pstart: int = -1  # its paragraph: the block, from its lead-in if any (at most PARA_MAX around the statement)
    pend: int = -1


PARA_MAX = 600
_ARTICLE = re.compile(r"^(?:ARTICLE|Article|SECTION|Section)\s+(?P<n>[IVXLC]+|\d+(?:\.\d+)*)\b\.?\s*[-—–:.]?\s*(?P<h>.*)$")


def _number(parent_num: str, marker: str, mclass: str) -> str:
    m = marker.strip().rstrip(".")
    if mclass.startswith("num"): return m.rstrip(")")
    if mclass in ("alpha", "roman", "ALPHA", "paren_num"):
        x = m.strip("()").rstrip(")")
        return f"{parent_num}({x})" if parent_num else f"({x})"
    return ""


def _structure(text: str):
    root = parse(text); loc = Located(text); ploc = Located(text)
    stmts, secs = [], []
    para = [-1, -1]

    def new_sec(number, heading, parent):
        secs.append(Section(len(secs), number, heading, parent)); return len(secs) - 1

    def emit(full, own, chain):
        s, e = loc.find(own)
        ps, pe = para
        if s >= 0 and (ps < 0 or not ps <= s < pe): ps, pe = s, e
        if s >= 0: ps, pe = max(ps, s - PARA_MAX), min(max(pe, e), e + PARA_MAX)
        stmts.append(Stmt(re.sub(r"\s+", " ", full).strip(), own, s, e, tuple(chain), ps, pe))

    def walk(n: Node, lead: list, chain: list, lead_start: int = -1, own: bool = False):
        here = [*lead]; body = n.text
        if body:
            bs, be = ploc.find(body)
            para[0], para[1] = (lead_start if 0 <= lead_start <= bs else bs), be
            h = inline_head(body)
            if h and own and not secs[chain[-1]].heading:
                secs[chain[-1]].heading = h  # "12.1 Assignment. Neither party ..." names its own section
            elif h and not (chain and secs[chain[-1]].heading == h):
                chain = [*chain, new_sec("", h, chain[-1] if chain else -1)]
            if n.leadin:
                here.append(body.rstrip(" :—–-"))
                for s in SENT.split(body):
                    if len(s) > 3: emit(" ".join([*lead, s]), s, chain)
            elif (sp := _inline_split(body)):
                for item in sp[1]: emit(" ".join([*lead, sp[0], item]), item, chain)
                emit(sp[0], sp[0], chain)
            else:
                for s in SENT.split(body):
                    if len(s) > 3: emit(" ".join([*lead, s]), s, chain)
        sib = chain; para_of = list(para) if body else [-1, -1]
        parent_num = next((secs[i].number for i in reversed(chain) if secs[i].number), "")
        for ch in n.children:
            ho = head_only(ch.text) if ch.text else None
            if ho and not ch.children:  # a heading line: a scope for the siblings after it
                para[0] = para[1] = -1
                am = _ARTICLE.match(ho)
                num = am.group("n") if am else _number(parent_num, ch.marker, ch.mclass) if ch.marker else ""
                sib = [*chain, new_sec(num, ho, chain[-1] if chain else -1)]
                emit(ch.text, ch.text, sib); continue
            ls = (para_of[0] if n.leadin and body else lead_start) if (n.leadin or n.mclass == "root") else -1
            base, mine = sib, False
            if ch.marker and ch.mclass != "bullet":
                pn = next((secs[i].number for i in reversed(base) if secs[i].number), "")
                base, mine = [*base, new_sec(_number(pn, ch.marker, ch.mclass), ho or "", base[-1] if base else -1)], True
            elif ho:
                base = [*base, new_sec("", ho, base[-1] if base else -1)]
            walk(ch, here if n.leadin or n.mclass == "root" else lead, base, ls, mine)

    walk(root, [], [])
    first = next((st for st in stmts if st.start >= 0 and not head_only(st.own) and len(st.own) > 60), None)
    for sec in secs:  # unnumbered top-level headings before the first sentence are the title page
        if not sec.number and sec.parent == -1 and first is not None and any(
                st.start >= 0 and st.start < first.start and st.own.strip().rstrip(".:").strip() == sec.heading for st in stmts[:40]):
            sec.front = True
    for st in stmts:  # each section's span = its statements'
        if st.start < 0: continue
        for i in st.secs:
            s = secs[i]
            s.start = st.start if s.start < 0 else min(s.start, st.start)
            s.end = max(s.end, st.end)
    return root, stmts, secs


# ---------- 5. symbols ----------
_DEF = re.compile(r"[\"“”']{1,2}(?P<t>[A-Z][\w&\-’' ]{1,60}?)[\"“”']{1,2}\s*(?:,\s*)?(?:shall\s+|will\s+|as used herein,?\s+)?"
                  r"(?:mean|means|refers? to|include|includes|shall include|is defined as|has the meaning|consists? of)\b", re.I)
_PAREN = re.compile(r"\((?:the\s+|collectively,?\s+|hereinafter\s+(?:referred to as\s+|called\s+)?|herein\s+|each\s+a\s+|individually\s+|"
                    r"referred to (?:herein )?as\s+(?:the\s+)?)?[\"“”']{1,2}(?P<t>[A-Z][\w&\-’'. ]{1,50}?)[\"“”']{1,2}"
                    r"(?:\s*(?:or|and)\s*(?:the\s+)?[\"“”']{1,2}[A-Z][\w&\-’' ]{1,50}?[\"“”']{1,2})?\s*\)")


def _symbols(stmts: list) -> dict:
    sym = {}
    for st in stmts:
        t = st.text
        for m in _DEF.finditer(t):
            sym.setdefault(m.group("t").strip(), []).append(t)
        for m in _PAREN.finditer(t):
            sym.setdefault(m.group("t").strip(), []).append(t[:m.start()][-300:])
    return sym


# ---------- 6. parties ----------
ROLES = ("licensor", "licensee", "sublicensee", "distributor", "supplier", "customer", "client", "vendor", "buyer", "seller",
         "purchaser", "franchisor", "franchisee", "landlord", "tenant", "lessor", "lessee", "sublessee", "sublessor",
         "disclosing party", "receiving party", "recipient", "discloser", "employer", "employee", "executive", "consultant",
         "contractor", "subcontractor", "service provider", "provider", "manufacturer", "reseller", "agent", "principal",
         "sponsor", "developer", "publisher", "borrower", "lender", "guarantor", "investor", "partner", "member", "owner",
         "operator", "manager", "marketer", "promoter", "endorser", "affiliate", "host", "carrier", "shipper", "producer",
         "artist", "talent", "author", "institution", "university", "collaborator", "co-branding partner", "outsourcer",
         "maintenance provider", "strategic partner", "joint venturer", "issuer", "holder", "trustee", "transferor",
         "transferee", "assignor", "assignee", "grantor", "grantee", "company", "corporation", "bank", "user")
_ROLE_RX = re.compile(r"^(?:the\s+)?(?:" + "|".join(re.escape(r) for r in sorted(ROLES, key=len, reverse=True)) + r")s?$", re.I)
_ENTITY = re.compile(r"\b(?:inc|llc|l\.l\.c|ltd|limited|corp|corporation|company|co|plc|gmbh|s\.a|n\.v|b\.v|ag|lp|l\.p|llp|"
                     r"partnership|trust|bank|university|an? individual|organized under|incorporated under|existing under|"
                     r"principal (?:place of business|office)|having (?:its|an) (?:office|address)|residing at|with offices at)\b", re.I)
_PREAMBLE_END = re.compile(r"\bWHEREAS\b|\bRECITALS?\b|\bW\s*I\s*T\s*N\s*E\s*S\s*S\s*E\s*T\s*H\b|\bNOW,?\s+THEREFORE\b|\bBACKGROUND\b", re.I)


@dataclass
class Party:
    name: str    # the defined name ("Licensee", "ABC")
    role: str    # the role noun when the name is one ("licensee"), else ""
    said: str    # the words that introduce it (up to 200 chars)
    start: int


def _parties(text: str, symbols: dict) -> list:
    head = text[:4000]
    m = _PREAMBLE_END.search(head, 40)
    head = head[:m.start()] if m else head[:2500]
    out, seen = [], set()
    for pm in _PAREN.finditer(head):
        name = pm.group("t").strip().strip(".")
        before = head[max(0, pm.start() - 250):pm.start()]
        before = re.split(r"\)\s*(?:,|;)?\s*(?:and|&)?\s*|\bby and (?:between|among)\b|\bbetween\b", before)[-1]
        if re.fullmatch(r"(?i)(?:this\s+)?(?:agreement|contract|effective date|execution date|amendment|parties|party|"
                        r"lease|license|term|services?)", name): continue
        role = name.lower() if _ROLE_RX.match(name) else ""
        if not (role or _ENTITY.search(before)): continue
        if name.lower() in seen: continue
        seen.add(name.lower()); out.append(Party(name, re.sub(r"^the\s+", "", role), re.sub(r"\s+", " ", before).strip()[-200:], pm.start()))
    return out


# ---------- 7. cross-references ----------
_XREF = re.compile(r"\b(?P<k>Sections?|Articles?|Clauses?|Paragraphs?|§§?)\s*(?P<n>\d+(?:\.\d+)*(?:\.[a-z]\b)?(?:\s*\([a-zA-Z0-9]{1,4}\))*|[IVXLC]+\b)")
_ATTACH = re.compile(r"\b(?P<k>Exhibit|Schedule|Annex|Appendix|Attachment|Addendum)\s+(?P<n>[A-Z0-9]{1,3}(?:[.\-][A-Z0-9]{1,3})?)\b")
_ATTACH_HEAD = re.compile(r"^[ \t]*(?:EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|Annex|APPENDIX|Appendix|ATTACHMENT|Attachment|ADDENDUM|Addendum)"
                          r"[ \t]+(?P<n>[A-Z0-9]{1,3}(?:[.\-][A-Z0-9]{1,3})?)\b", re.M)
_EXTERNAL = re.compile(r"\b(?:as defined in|defined in|as set forth in|set forth in|pursuant to|under|in accordance with|subject to|"
                       r"terms of|the provisions of)\s+(?:the|that certain|such)\s+(?P<d>(?:[A-Z][\w\-&,]*\s+){0,6}?"
                       r"(?:Agreement|Contract|Plan|Policy|Lease|License|Order|Statement of Work|Indenture|Charter|Bylaws))\b")
_INCORP = re.compile(r"\bincorporated\s+(?:herein|into\s+(?:this|the)\s+\w+|in\s+this\s+\w+)\s+by\s+reference\b|"
                     r"\bby\s+reference\s+(?:herein|into)\b|\bmade a part (?:hereof|of this \w+) by reference\b", re.I)


@dataclass
class XRef:
    kind: str     # "section" | "attachment" | "document" | "document-section" | "incorporation"
    label: str    # "9.2", "Exhibit B", "Master Services Agreement"
    start: int
    end: int
    target: int = -1  # section id, when resolved
    resolved: bool = False


_OF_DOC = re.compile(r"\s+of\s+(?P<d>(?:the\s+|that\s+certain\s+)?(?:[A-Z][\w\-&.]*\s*){1,7})")


def _norm_num(n: str) -> str:
    n = re.sub(r"\s+", "", n).rstrip(".")
    return re.sub(r"\.([a-z])$", r"(\1)", n)  # "7.e" -> "7(e)"


def _xrefs(text: str, secs: list, title: str, symbols: dict) -> list:
    by_num = {}
    for s in secs:  # the last section with a number wins over a table of contents' entry
        if s.number and s.start >= 0:
            prev = by_num.get(s.number)
            if prev is None or (s.end - s.start) > (prev.end - prev.start): by_num[s.number] = s
    out = []
    for m in _XREF.finditer(text):
        of = _OF_DOC.match(text, m.end())
        if of and not re.fullmatch(r"(?i)(?:this|the)\s+(?:agreement|contract)", of.group("d").strip()):
            out.append(XRef("document-section", f"{_norm_num(m.group('n'))} of {of.group('d').strip()}", m.start(), of.end(), -1, False))
            continue  # "Section 7.09 of the Separation Agreement", "Section 4043(c) of ERISA"
        n = _norm_num(m.group("n")); s = by_num.get(n)
        while s is None and "(" in n:
            n = n[:n.rindex("(")]; s = by_num.get(n)
        if s is None and "." in n and n.endswith(".0"): s = by_num.get(n[:-2])
        out.append(XRef("section", _norm_num(m.group("n")), m.start(), m.end(), s.id if s else -1, s is not None))
    heads = {re.sub(r"\s+", "", h.group("n")).upper() for h in _ATTACH_HEAD.finditer(text)}
    for m in _ATTACH.finditer(text):
        lab = f"{m.group('k').capitalize()} {m.group('n')}"
        out.append(XRef("attachment", lab, m.start(), m.end(), -1, m.group("n").upper() in heads))
    own = (title or "").lower()
    defined = {k.lower() for k in symbols}
    for m in _EXTERNAL.finditer(text):
        d = m.group("d").strip()
        if d.lower() in own or re.fullmatch(r"(?i)(?:this\s+)?agreement", d) or d.lower() in defined and \
                re.search(r"\bthis\s+" + re.escape(d), text[:3000], re.I):
            continue
        if re.search(r"\bthis\s+" + re.escape(d) + r"\b", text, re.I): continue  # the document names itself so
        out.append(XRef("document", d, m.start("d"), m.end("d"), -1, False))
    for m in _INCORP.finditer(text):
        out.append(XRef("incorporation", m.group(0), m.start(), m.end(), -1, False))
    out.sort(key=lambda x: x.start)
    return out


# ---------- the compiled document ----------
_TITLE_WORDS = re.compile(r"\b(?:agreement|contract|lease|license|licence|policy|terms|addendum|amendment|memorandum|"
                          r"undertaking|deed|indenture|plan|charter|bylaws|order|statement of work|letter)\b", re.I)


@dataclass
class CompiledDoc:
    length: int
    title: str
    furniture: list   # [(start, end, what)]
    stmts: list       # [Stmt], document order
    sections: list    # [Section]
    symbols: dict     # term -> [defining text]
    parties: list     # [Party]
    xrefs: list       # [XRef]
    _starts: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._order = sorted((s.start, k) for k, s in enumerate(self.stmts) if s.start >= 0)
        self._starts = [a for a, _ in self._order]

    def stmts_in(self, start: int, end: int) -> list:
        """Statements whose own span overlaps [start, end)."""
        k = bisect.bisect_left(self._starts, start - 2000)
        out = []
        for a, i in self._order[k:]:
            if a >= end: break
            s = self.stmts[i]
            if s.end > start: out.append(s)
        return out

    def headings(self, st: Stmt) -> list:
        return [self.sections[i].heading for i in st.secs if self.sections[i].heading and not self.sections[i].front]

    def headings_at(self, start: int, end: int) -> list:
        """The heading chain most of [start, end) sits under (by characters), outermost first."""
        best, size = [], -1
        tally = {}
        for s in self.stmts_in(start, end):
            key = tuple(self.headings(s)); tally[key] = tally.get(key, 0) + min(s.end, end) - max(s.start, start)
        for key, n in tally.items():
            if n > size: best, size = list(key), n
        return best

    @property
    def roles(self) -> list:
        return [p.role for p in self.parties if p.role]

    def to_dict(self) -> dict:
        d = asdict(self); d.pop("_starts", None); return d


def _title(text: str, stmts: list) -> str:
    for st in stmts[:12]:
        if st.start < 0 or st.start > 1500: continue
        t = st.own.strip()
        if len(t) <= 120 and _TITLE_WORDS.search(t) and not re.search(r"\b(?:shall|will|hereby|made and entered|is entered)\b", t, re.I):
            return t
    return ""


def compile_doc(text: str) -> CompiledDoc:
    clean, furniture = preprocess(text)
    _, stmts, secs = _structure(clean)
    sym = _symbols(stmts)
    title = _title(clean, stmts)
    return CompiledDoc(len(text), title, furniture, stmts, secs, sym, _parties(clean, sym), _xrefs(clean, secs, title, sym))
