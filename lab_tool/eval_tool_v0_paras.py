"""Part 3 at paragraph level (2026-10-05): word recall against the tracked changes understates compare(), because a
minimal word diff keeps words the redline happened to delete and re-insert ("[-three and three-quarters percent-]
{+two and one-half percent+}" vs "[-three-] {+two+} and [-three-quarters-] {+one-half+} percent"). Here: of the
paragraphs the redline changed, how many does compare() flag (recall), and how many it flags are really changed
(precision).
usage: dspy_venv/bin/python eval_tool_v0_paras.py -> prints, adds "3_compare_paragraphs" to data/tool_v0_eval.json"""
import json, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def one(path):
    from contract_tool.parse import parse, Unit
    from contract_tool.tool import diff_units
    us = parse(path)
    gt = {i for i, u in enumerate(us) if " ".join(u.before.split()) != " ".join(u.text.split())}
    if not gt: return None
    A = [Unit("before", i, "p", u.before) for i, u in enumerate(us) if u.before]
    B = [Unit("after", i, "p", u.text) for i, u in enumerate(us) if u.text]
    items, _ = diff_units(A, B)
    flagged = {u.i for _, u, *_ in items}
    return {"gt": len(gt), "flagged": len(flagged), "both": len(gt & flagged)}


if __name__ == "__main__":
    docs = [json.loads(l)["doc"] for l in open(f"{HERE}/data/tool_v0_part23.jsonl")]
    with mp.get_context("fork").Pool(8) as p: R = [r for r in p.map(one, [f"{HERE}/harvey-labs/{d}" for d in docs]) if r]
    S = lambda k: sum(r[k] for r in R)
    o = {"docs": len(R), "changed_paragraphs": S("gt"), "flagged": S("flagged"), "recall": round(S("both") / S("gt"), 4), "precision": round(S("both") / S("flagged"), 4)}
    print(o)
    p = f"{HERE}/data/tool_v0_eval.json"; d = json.load(open(p)); d["3_compare_paragraphs"] = o; json.dump(d, open(p, "w"), indent=1)
