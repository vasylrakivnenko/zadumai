"""ContractNLI as a mechanical question bank (2026-10-03, the user's go): today's baseline on the 17 NDA questions.
Protocol: tune on train + dev (484 NDAs); the 123 test NDAs stay sealed until one final run (`--final`, logged).
Each question is asked as a user would, over the whole NDA. Gold: Entailment -> the question's "yes" answer,
Contradiction -> its "no" answer (POLARITY flips nda-15), NotMentioned -> any answer is wrong.
Measured per question: Pre-Tier 0 alone, then the reader network on what it leaves (both as the router runs them),
whether the NDA section tagger (models/contract_map/tagger_contractnli.joblib) finds a section of that question's
type, and whether an answer's evidence overlaps the gold evidence spans.
usage: baseline.py dev|train [--save NAME] [--no-net]"""
import argparse, collections, json, os, sys, time, zipfile
sys.path.insert(0, "/root/projects/zadumai/legalbench_map")
from router.pretier0 import check
from router.contract_map import ContractMap, split_sections

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = "/root/zadumai_nli_proto/contract_map/data/ext/contractnli.zip"
QUESTIONS = {  # how a user would ask it; ContractNLI's hypothesis text is in the data
    "nda-1": "Must all confidential information be expressly identified as confidential by the disclosing party?",
    "nda-2": "Is confidential information limited to technical information?",
    "nda-3": "Does confidential information include information disclosed orally?",
    "nda-4": "Is the receiving party prohibited from using confidential information for any purpose other than the purpose of the agreement?",
    "nda-5": "May the receiving party share confidential information with its employees?",
    "nda-7": "May the receiving party share confidential information with third parties such as consultants, agents or professional advisors?",
    "nda-8": "Must the receiving party notify the disclosing party if it is required by law to disclose confidential information?",
    "nda-10": "Is the receiving party prohibited from disclosing the existence of this agreement?",
    "nda-11": "Is the receiving party prohibited from reverse engineering the confidential information?",
    "nda-12": "May the receiving party independently develop information similar to the confidential information?",
    "nda-13": "May the receiving party acquire similar information from a third party?",
    "nda-15": "Does the agreement grant the receiving party any license or right to the confidential information?",
    "nda-16": "Must the receiving party return or destroy the confidential information when the agreement ends?",
    "nda-17": "May the receiving party make copies of the confidential information?",
    "nda-18": "Is the receiving party prohibited from soliciting the disclosing party's employees?",
    "nda-19": "Do any obligations survive termination of the agreement?",
    "nda-20": "May the receiving party keep some confidential information after returning or destroying it?",
}
POLARITY = {"nda-15": {"Entailment": "no", "Contradiction": "yes"}}  # the question asks the opposite of the hypothesis
DEFAULT = {"Entailment": "yes", "Contradiction": "no"}


def overlaps(ev_texts, doc, spans):
    """Whether any evidence text sits inside (or across) a gold evidence span."""
    gold = [doc["spans"][i] for i in spans]
    for t in ev_texts:
        t = (t or "").strip()
        if len(t) < 15: continue
        k = doc["text"].find(t[:80])
        if k < 0: continue
        if any(a <= k < b or k <= a < k + len(t) for a, b in gold): return True
    return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split", choices=["dev", "train", "test"]); ap.add_argument("--save")
    ap.add_argument("--no-net", action="store_true"); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    if a.split == "test":
        assert a.final, "test is sealed: one final run with --final"
        open(f"{HERE}/sealed_runs.log", "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
    d = json.loads(zipfile.ZipFile(ZIP).read(f"contract-nli/{a.split}.json")); lab = d["labels"]
    net = None
    if not a.no_net:
        from router import netreader
        net = netreader.NetReader()
    cm = ContractMap(lease_encoder=False); tg = cm.taggers["contractnli"]
    by_label = {v["short_description"]: k for k, v in lab.items()}
    st = collections.defaultdict(collections.Counter); kinds = collections.Counter(); rows = []
    t0 = time.time()
    for n, doc in enumerate(d["documents"]):
        kinds[cm.kind(doc["text"])[0] or "not sure"] += 1
        secs = split_sections(doc["text"]); S = tg.scores([t for _, _, t in secs]) if secs else None
        tagged = set()
        if S is not None:
            for s in S:
                for j in (s >= tg.th).nonzero()[0]: tagged.add(by_label.get(tg.labels[tg.classes[j]]))
        for k, q in QUESTIONS.items():
            ann = doc["annotation_sets"][0]["annotations"][k]; g = ann["choice"]
            want = POLARITY.get(k, DEFAULT).get(g)  # None for NotMentioned
            p = check(q, doc["text"]); tier, ans, ev = None, None, []
            if p.fired:
                tier, ans, ev = "pt0", p.answer, [e.get("text") for e in (p.evidence or [])]
            elif net is not None:
                r = net.answer(p.rewritten or q, doc["text"])
                if r.fired: tier, ans, ev = "net", r.answer, [e.get("text") for e in (r.evidence or [])]
            c = st[k]; c["n"] += 1; c[g] += 1
            if ans:
                ok = ans == want; c[tier] += 1; c[tier + "_ok"] += ok
                if not ok: c["wrong_on_" + ("notmentioned" if g == "NotMentioned" else "polarity")] += 1
                c["ev_overlap"] += overlaps(ev, doc, ann["spans"]) if g != "NotMentioned" else 0
            mentioned = g != "NotMentioned"; found = k in tagged
            c["tag_found_mentioned"] += found and mentioned; c["tag_none_notmentioned"] += (not found) and not mentioned
            c["tag_none"] += not found; c["mentioned"] += mentioned
            rows.append({"doc": doc["id"], "q": k, "gold": g, "want": want, "tier": tier, "answer": ans, "tagged": found})
        if (n + 1) % 10 == 0: print(f"   {n + 1}/{len(d['documents'])} NDAs ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n== ContractNLI {a.split}: {len(d['documents'])} NDAs x 17 questions; router kinds: {dict(kinds)}")
    print(f"{'question':48} {'E/C/N':>11} | {'Pre-Tier 0':>10} {'network':>8} | {'answered':>8} {'right':>6} | wrong: N / polarity | "
          f"tagger: finds mentioned | 'none' -> not mentioned")
    tot = collections.Counter()
    for k in QUESTIONS:
        c = st[k]; tot.update(c)
        ans = c["pt0"] + c["net"]; ok = c["pt0_ok"] + c["net_ok"]
        print(f"{k:6} {lab[k]['short_description'][:41]:41} {c['Entailment']:3}/{c['Contradiction']:2}/{c['NotMentioned']:3} | "
              f"{c['pt0_ok']:3}/{c['pt0']:<3}    {c['net_ok']:3}/{c['net']:<3}  | {ans:4} = {ans / c['n']:4.0%} {ok:4}   | "
              f"{c['wrong_on_notmentioned']:3} / {c['wrong_on_polarity']:<3}         | {c['tag_found_mentioned']:3}/{c['mentioned']:<3} = "
              f"{c['tag_found_mentioned'] / max(c['mentioned'], 1):4.0%}      | {c['tag_none_notmentioned']}/{c['tag_none']} = "
              f"{c['tag_none_notmentioned'] / max(c['tag_none'], 1):4.0%}")
    c = tot; ans = c["pt0"] + c["net"]; ok = c["pt0_ok"] + c["net_ok"]
    print(f"{'all':48} {c['Entailment']:4}/{c['Contradiction']:3}/{c['NotMentioned']:4} | Pre-Tier 0 {c['pt0_ok']}/{c['pt0']}, network "
          f"{c['net_ok']}/{c['net']} | answered {ans} = {ans / c['n']:.1%}, right {ok} = {ok / max(ans, 1):.1%} | evidence overlaps gold "
          f"on {c['ev_overlap']} | tagger finds {c['tag_found_mentioned']}/{c['mentioned']} mentioned; 'none' right {c['tag_none_notmentioned']}/{c['tag_none']}")
    if a.save:
        json.dump(rows, open(f"{HERE}/runs/{a.save}.json", "w"))


if __name__ == "__main__":
    main()
