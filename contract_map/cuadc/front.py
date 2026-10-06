"""Front end for commercial contracts (2026-10-03): the NDA compiler's structure parser (cnli/compiler.py: numbered /
lettered / bulleted blocks -> tree, lead-ins prefixed to their items) plus what CUAD needs:
  offsets   every statement keeps the character span of its own words in the original text, so evidence can be
            checked against CUAD's annotated spans
  headings  each statement knows its section heading: a heading-only block ("12. ASSIGNMENT", "Governing Law") or an
            inline one ("12.1 Assignment. Neither party ...", "GOVERNING LAW: This Agreement ...")
Stmt.text is the statement with its lead-ins, Stmt.own its own words, Stmt.head the heading (lower case, "" if none)."""
from __future__ import annotations

import os, re, sys
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnli"))  # after ours: cnli has its own queries.py
from compiler import parse, _inline_split, Node  # noqa: E402

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


@dataclass
class Stmt:
    text: str      # with lead-ins
    own: str       # its own words
    start: int     # span of `own` in the original text (-1 if not found)
    end: int
    head: str      # section heading, lower case
    path: tuple
    pstart: int = -1  # its paragraph (the block, from its lead-in if any; at most PARA_MAX around the statement)
    pend: int = -1


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


PARA_MAX = 600  # chars kept on each side of the statement when the block is longer (unstructured text)


def compile_contract(text: str) -> list:
    root = parse(text); loc = Located(text); ploc = Located(text); out = []
    para = [-1, -1]

    def emit(full, own, head, path):
        s, e = loc.find(own)
        ps, pe = para
        if s >= 0 and (ps < 0 or not ps <= s < pe): ps, pe = s, e
        if s >= 0: ps, pe = max(ps, s - PARA_MAX), min(max(pe, e), e + PARA_MAX)
        out.append(Stmt(re.sub(r"\s+", " ", full).strip(), own, s, e, head.lower(), tuple(path), ps, pe))

    def walk(n: Node, lead: list, head: str, path: list, lead_start: int = -1):
        here = [*lead]; body = n.text
        if body:
            bs, be = ploc.find(body)
            para[0], para[1] = (lead_start if 0 <= lead_start <= bs else bs), be
            h = inline_head(body)
            if h: head = h
            if n.leadin:
                here.append(body.rstrip(" :—–-"))
                for s in SENT.split(body):  # located too: a lead-in can hold the clause itself
                    if len(s) > 3: emit(" ".join([*lead, s]), s, head, path)
            elif (sp := _inline_split(body)):
                for item in sp[1]: emit(" ".join([*lead, sp[0], item]), item, head, path)
                emit(sp[0], sp[0], head, path)
            else:
                for s in SENT.split(body):
                    if len(s) > 3: emit(" ".join([*lead, s]), s, head, path)
        sib = head; para_of = list(para) if body else [-1, -1]
        for ch in n.children:
            ho = head_only(ch.text) if ch.text else None
            if ho and not ch.children:
                para[0] = para[1] = -1; sib = ho; emit(ch.text, ch.text, ho, [*path, ch.marker]); continue
            ls = (para_of[0] if n.leadin and body else lead_start) if (n.leadin or n.mclass == "root") else -1
            walk(ch, here if n.leadin or n.mclass == "root" else lead, ho or sib, [*path, ch.marker], ls)

    walk(root, [], "", [])
    return out
