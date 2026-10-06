"""NDA question bank, engine v2 (2026-10-03): v1 (cv_engine.py) + a polarity model + concept checks, all choices made
inside the training folds of a nested 5-fold cross-validation over the 484 tuning NDAs (test NDAs untouched).
Per question q and NDA:
 1. span model (TF-IDF + LR, q's gold evidence spans vs all others): best span s at index i; window W = spans i-1..i+2
    (ContractNLI cuts list items into fragments, so the words that matter can sit in a neighbour).
 2. not mentioned: s < t_nm, or no topic word (mentioned.py), or either - the variant the inner folds pick.
 3. addressed: s >= t_m (the inner folds' loosest threshold where >= TARGET of NDAs above it address q). Then
    - questions with >= 15 "no" (Contradiction) NDAs in the training folds: a polarity model (TF-IDF + LR on the
      evidence windows of addressed NDAs, E vs C) gives p_no for W; "no" if p_no >= t_no, "yes" if p_no <= t_yes;
    - otherwise "yes" (the label of almost every addressed NDA);
    - the concept check (CORE[q] must occur in W) is on if the inner folds answer more at >= TARGET with it.
 4. Pre-Tier 0's answer, if the bank has none, for questions where it was >= TARGET on the training folds.
Writes runs/cv_engine2.json. usage: cv_engine2.py [--target 0.99]"""
import argparse, collections, json, os, random, re, sys
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from mentioned import RX
from baseline import QUESTIONS, POLARITY, DEFAULT
from cv_mentioned import load, wilson_lo

Q = list(QUESTIONS)
MIN_CALLS = 20
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = {k: re.compile(v, re.I) for k, v in {
    "nda-1": r"\bmark|\blabel|\bdesignat|\bidentif|\blegend|\bstamp",
    "nda-2": r"\bbusiness|\bfinanc|\bcommercial|\bmarketing|\bcustomer|\bclient|\bpric|\bstrateg|\bplans?\b|\bpersonnel|\bsales|\bsupplier|\bcorporate|\bnon-technical|\boperat|\btechnical",
    "nda-3": r"\boral|\bverbal|\bvisual|\bin any form|\bany (?:form|medium|manner)|\bwhether (?:written|in writing|oral|tangible)|\bobservation|\bintangible|\bspoken",
    "nda-4": r"\bpurpose|\bsolely|\bonly\b|\bexclusively|\bother than",
    "nda-5": r"\bemployee|\bstaff|\bpersonnel|\bofficer|\bdirector|\brepresentative|\bneed[- ]to[- ]know",
    "nda-7": r"\bconsultant|\badvis|\bagent|\bcontractor|\battorney|\baccountant|\bcounsel|\bthird[- ]part|\brepresentative|\baffiliate|\blender|\bfinancing|\bsubcontract|\bbanker|\bauditor",
    "nda-8": r"(?=.*\b(?:notif|notice|inform|advise|consult))(?=.*\b(?:requir|compel|order|subpoena|law|regulat|court|judicial|governmental|legal process))",
    "nda-10": r"\bexistence|\bfact that|\bterms of (?:this|the)|\bdiscussions|\bnegotiations|\btransaction|\bannounce|\bpublicity|\bpress|\bcontents of",
    "nda-11": r"reverse|decompil|disassembl|derive|analy[sz]|decompos|deconstruct|source code",
    "nda-12": r"\bindependent",
    "nda-13": r"third[- ]part|\bfrom (?:a|any|another) (?:source|person|party)|\blawfully|\brightfully|\bsource other|\bwithout restriction|\bnon-confidential basis|\breceived from",
    "nda-15": r"\blicen[cs]|\bright|\btitle\b|\bownership|\bproperty of|\bgrant|\bremain",
    "nda-16": r"\breturn|\bdestroy|\bdestruction|\bdelete|\berase|\bdeliver|\bsurrender",
    "nda-17": r"\bcop(?:y|ies|ying)|\breproduc|\bduplicat",
    "nda-18": r"\bsolicit|\bhire|\bhiring|\bemploy|\brecruit|\bentice|\binduce|\bpersuade|\bconduct business|\bcontact",
    "nda-19": r"\bsurviv|\bcontinue|\bremain|\bafter (?:the )?(?:termination|expiration)|\bnotwithstanding|\bperiod of|\byears?\b|\bterminat|\bexpir",
    "nda-20": r"\bretain|\bkeep|\barchiv|\bback-?up|\bone copy|\brecords|\brequired to (?:delete|destroy|return)|\bsave\b|\ball copies|\bno copies",
}.items()}


def window(d, i):
    return " ".join(d["spans"][max(0, i - 1):i + 3])


def span_model(train_docs, q):
    X, y = [], []
    for d in train_docs:
        for i, s in enumerate(d["spans"]): X.append(s); y.append(int(i in d["ev"][q]))
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000)
    m = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(v.fit_transform(X), y)
    def best(d):
        if not d["spans"]: return 0.0, 0
        p = m.predict_proba(v.transform(d["spans"]))[:, 1]; return float(p.max()), int(p.argmax())
    return best


def polarity_model(train_docs, q, best):
    """E vs C on the best-span windows of addressed NDAs (the windows the engine will read)."""
    X, y = [], []
    for d in train_docs:
        g = d["gold"][q]
        if g == "NotMentioned": continue
        ev = sorted(d["ev"][q]) or [best(d)[1]]
        for i in ev[:3]: X.append(window(d, i)); y.append(int(g == "Contradiction"))
    if sum(y) < 15: return None
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40_000)
    m = LogisticRegression(C=4, max_iter=2000, class_weight="balanced").fit(v.fit_transform(X), y)
    return lambda text: float(m.predict_proba(v.transform([text]))[0, 1])


def loosest(scores, good, target, above):
    ts = sorted(set(np.round(scores, 4)), reverse=above)  # strict to loose
    best = None
    for t in ts:
        k = scores >= t if above else scores < t
        if k.sum() >= MIN_CALLS and good[k].mean() >= target: best = t
        elif k.sum() >= MIN_CALLS: break
    return best


def features(train_docs, test_docs, q):
    """For each test doc: (best score, best index, p_no or None)."""
    best = span_model(train_docs, q); pol = polarity_model(train_docs, q, best)
    out = []
    for d in test_docs:
        s, i = best(d); out.append((s, i, pol(window(d, i)) if pol else None))
    return out, pol is not None


def decide(feat, d, q, cfg):
    s, i, pn = feat; want = POLARITY.get(q, DEFAULT)
    kw = bool(RX[q].search(d["text"]))
    nm = {"model": cfg["t_nm"] is not None and s < cfg["t_nm"], "keywords": not kw}
    nm["either"] = nm["model"] or nm["keywords"]
    if cfg["nm"] and nm[cfg["nm"]]: return "not mentioned", f"{cfg['nm']}: best span {s:.2f}"
    if cfg["t_m"] is None or s < cfg["t_m"]: return None, ""
    if cfg["check"] and not CORE[q].search(window(d, i)): return None, "concept check failed"
    if pn is None: return want["Entailment"], f"addressed (span {s:.2f}); answer of almost every addressed NDA"
    if cfg["t_no"] is not None and pn >= cfg["t_no"]: return want["Contradiction"], f"polarity p_no {pn:.2f} >= {cfg['t_no']}"
    if cfg["t_yes"] is not None and pn <= cfg["t_yes"]: return want["Entailment"], f"polarity p_no {pn:.2f} <= {cfg['t_yes']}"
    return None, "polarity unsure"


def right(ans, gold, q):
    if ans == "not mentioned": return gold == "NotMentioned"
    return POLARITY.get(q, DEFAULT).get(gold) == ans


def configure(tr, q, target):
    """Pick every threshold and switch from inner 4-fold out-of-fold features of the training folds."""
    inner = np.array([i % 4 for i in range(len(tr))]); feats = [None] * len(tr); has_pol = False
    for g in range(4):
        a = [d for d, k in zip(tr, inner) if k != g]; b = [i for i, k in enumerate(inner) if k == g]
        f, hp = features(a, [tr[i] for i in b], q); has_pol = has_pol or hp
        for j, x in zip(b, f): feats[j] = x
    s = np.array([f[0] for f in feats]); g = np.array([d["gold"][q] for d in tr])
    cfg = {"t_nm": loosest(s, g == "NotMentioned", target, above=False),
           "t_m": loosest(s, g != "NotMentioned", target, above=True), "t_yes": None, "t_no": None}
    pn = np.array([f[2] if f[2] is not None else np.nan for f in feats])
    if not np.all(np.isnan(pn)) and cfg["t_m"] is not None:
        k = (s >= cfg["t_m"]) & ~np.isnan(pn)
        cfg["t_no"] = loosest(pn[k], g[k] == "Contradiction", target, above=True)
        cfg["t_yes"] = loosest(pn[k], g[k] == "Entailment", target, above=False)
    best_n, choice = -1, (None, False)
    for nmv in (None, "model", "keywords", "either"):
        for check in (False, True):
            c = {**cfg, "nm": nmv, "check": check}
            ans = [decide(f, d, q, c)[0] for f, d in zip(feats, tr)]
            calls = [(a, gg) for a, gg in zip(ans, g) if a]
            if len(calls) >= MIN_CALLS and np.mean([right(a, gg, q) for a, gg in calls]) >= target and len(calls) > best_n:
                best_n, choice = len(calls), (nmv, check)
    cfg["nm"], cfg["check"] = choice
    return cfg


def one_question(q, docs, folds, target, p0):
    out = []
    for f in range(5):
        tr = [d for d, k in zip(docs, folds) if k != f]; te = [d for d, k in zip(docs, folds) if k == f]
        cfg = configure(tr, q, target)
        pa = [(p0[(d["id"], q)], d["gold"][q]) for d in tr if (d["id"], q) in p0]
        use_pt0 = bool(len(pa) >= 10 and np.mean([right(x, g, q) for x, g in pa]) >= target)
        feats, _ = features(tr, te, q)
        for d, ft in zip(te, feats):
            ans, why = decide(ft, d, q, cfg)
            if ans is None and use_pt0 and (d["id"], q) in p0: ans, why = p0[(d["id"], q)], "Pre-Tier 0"
            out.append({"doc": d["id"], "q": q, "gold": d["gold"][q], "answer": ans, "why": why, "span": ft[1], "fold": f,
                        "cfg": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cfg.items()}})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.99); a = ap.parse_args()
    docs = load(); idx = list(range(len(docs))); random.Random(0).shuffle(idx); folds = [0] * len(docs)
    for r, i in enumerate(idx): folds[i] = r % 5
    from router.pretier0 import check
    p0 = {(d["id"], q): p.answer for d in docs for q in Q if (p := check(QUESTIONS[q], d["text"])).fired}
    print(f"== engine v2, nested 5-fold CV, {len(docs)} NDAs x 17, target {a.target}", flush=True)
    res = [r for rs in Parallel(n_jobs=8)(delayed(one_question)(q, docs, folds, a.target, p0) for q in Q) for r in rs]
    json.dump(res, open(f"{HERE}/runs/cv_engine2.json", "w"))
    st = collections.defaultdict(collections.Counter)
    for r in res:
        c = st[r["q"]]; c["n"] += 1; c[r["gold"]] += 1
        if r["answer"]:
            k = "nm" if r["answer"] == "not mentioned" else ("pt0" if r["why"] == "Pre-Tier 0" else ("no" if r["answer"] == POLARITY.get(r["q"], DEFAULT)["Contradiction"] else "yes"))
            c[k] += 1; c[k + "_ok"] += right(r["answer"], r["gold"], r["q"])
    print(f"{'question':7} {'E/C/N':>12} | {'yes':>9} {'no':>9} {'not ment.':>10} {'Pre-T0':>8} | answered  right")
    tot = collections.Counter()
    for q in Q:
        c = st[q]; tot.update(c); ks = ("yes", "no", "nm", "pt0")
        ans = sum(c[k] for k in ks); ok = sum(c[k + "_ok"] for k in ks)
        cf = collections.Counter((r["cfg"]["nm"], r["cfg"]["check"]) for r in res if r["q"] == q).most_common(1)[0][0]
        print(f"{q:7} {c['Entailment']:4}/{c['Contradiction']:3}/{c['NotMentioned']:3} | {c['yes_ok']:4}/{c['yes']:<4} {c['no_ok']:4}/{c['no']:<4} "
              f"{c['nm_ok']:4}/{c['nm']:<4}  {c['pt0_ok']:3}/{c['pt0']:<3} | {ans / c['n']:4.0%}  {ok / max(ans, 1):6.1%} [lo {wilson_lo(ok, ans):.3f}]  nm={cf[0]} check={cf[1]}")
    ks = ("yes", "no", "nm", "pt0"); ans = sum(tot[k] for k in ks); ok = sum(tot[k + "_ok"] for k in ks)
    print(f"all: answered {ans} / {tot['n']} = {ans / tot['n']:.1%}, right {ok / ans:.1%} [lo {wilson_lo(ok, ans):.3f}] | "
          + ", ".join(f"{k} {tot[k + '_ok']}/{tot[k]}" for k in ks))


if __name__ == "__main__":
    main()
