"""A small FOLD-R++-style learner (2026-10-03; idea 3: learned answer-or-abstain rules): default rules with exceptions,
readable, from labeled examples. Written here (no package: the router's venv is the live one).
  rule  = literals (feature <= v, feature > v, feature == v)  AND NOT exception
  learn = cover the positives greedily, a literal at a time by FOLD's information gain, until a rule covers no negative
          or no literal helps; the negatives it still covers become the positives of a recursive call that learns the
          exception (exceptions of exceptions likewise), as in Wang & Gupta's FOLD-R++.
Here a positive is a candidate answer that is right; the rules say when to answer. Thresholds are the features'
quantiles. `min_prec` drops learned rules below that precision on the training data (we want few, very precise rules)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Lit:
    f: str
    op: str   # "<=", ">", "=="
    v: object

    def ok(self, x: dict) -> bool:
        a = x.get(self.f)
        if a is None: return False
        if self.op == "==": return a == self.v
        if not isinstance(a, (int, float)) or isinstance(a, bool): return False
        return a <= self.v if self.op == "<=" else a > self.v

    def __str__(self):
        v = f"{self.v:.3g}" if isinstance(self.v, float) else self.v
        return f"{self.f} {self.op} {v}"


@dataclass
class Rule:
    lits: list
    exceptions: list = field(default_factory=list)  # [Rule]

    def ok(self, x: dict) -> bool:
        return all(l.ok(x) for l in self.lits) and not any(e.ok(x) for e in self.exceptions)

    def show(self, depth=0) -> str:
        s = "  " * depth + "answer if " + " and ".join(map(str, self.lits))
        for e in self.exceptions:
            s += "\n" + "  " * (depth + 1) + "except " + e.show(depth + 1).strip().replace("answer if ", "")
        return s


def _ig(tp, fn, tn, fp):
    if tp == 0: return -math.inf
    tot = tp + fn + tn + fp
    def h(a, b):
        n = a + b
        return 0.0 if n == 0 or a == 0 or b == 0 else -(a / n) * math.log2(a / n) - (b / n) * math.log2(b / n)
    # FOLD's gain prefers covering positives while shedding negatives (entropy of the covered and uncovered parts)
    return -(((tp + fp) / tot) * h(tp, fp) + ((tn + fn) / tot) * h(tn, fn)) + (0.0 if tp > fp else -1.0)


def _candidates(xs: list, feats: list) -> list:
    lits = []
    for f in feats:
        vals = [x.get(f) for x in xs if x.get(f) is not None]
        if not vals: continue
        if all(isinstance(v, (bool, str)) for v in vals):
            for v in sorted(set(vals), key=str): lits.append(Lit(f, "==", v))
        else:
            vs = sorted(float(v) for v in vals)
            qs = sorted({vs[int(q * (len(vs) - 1))] for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)})
            for v in qs: lits += [Lit(f, "<=", v), Lit(f, ">", v)]
    return lits


def _learn_rule(pos, neg, lits, max_lits):
    rule, P, N = [], pos, neg
    while N and len(rule) < max_lits:
        best, bg = None, -math.inf
        for l in lits:
            if any(l.f == r.f and l.op == r.op for r in rule): continue
            tp = sum(l.ok(x) for x in P); fp = sum(l.ok(x) for x in N)
            g = _ig(tp, len(P) - tp, len(N) - fp, fp)
            if g > bg: best, bg = l, g
        if best is None or bg == -math.inf: break
        nP = [x for x in P if best.ok(x)]; nN = [x for x in N if best.ok(x)]
        if len(nN) == len(N) and len(nP) == len(P): break
        rule.append(best); P, N = nP, nN
    return rule, P, N


def fold(pos, neg, feats, max_lits=3, depth=0, max_depth=2, min_cover=5):
    lits = _candidates(pos + neg, feats)
    rules, P = [], list(pos)
    while len(P) >= min_cover:
        lits_r, cP, cN = _learn_rule(P, neg, lits, max_lits)
        if not lits_r or len(cP) < min_cover: break
        r = Rule(lits_r)
        if cN and depth < max_depth:
            r.exceptions = fold(cN, cP, feats, max_lits, depth + 1, max_depth, min_cover=2)
        rules.append(r)
        P = [x for x in P if not r.ok(x)]
        if len(rules) >= 6: break
    return rules


def learn(xs: list, ys: list, feats: list, min_prec=0.98, **kw) -> list:
    pos = [x for x, y in zip(xs, ys) if y]; neg = [x for x, y in zip(xs, ys) if not y]
    rules = fold(pos, neg, feats, **kw)
    keep = []
    for r in rules:
        cov = [y for x, y in zip(xs, ys) if r.ok(x)]
        if cov and sum(cov) / len(cov) >= min_prec: keep.append(r)
    return keep


def accept(rules: list, x: dict) -> bool:
    return any(r.ok(x) for r in rules)
