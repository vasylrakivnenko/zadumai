"""Jev judge, step 2 (2026-10-06): every rubric criterion -> 1-5 simple yes/no statements about the deliverable, once per
task, cached (GPT-6 Luna on the user's Azure deployment, one call per task; the first 10 tasks were done with Kimi
K3 before the user switched us to Luna: kept in decomp_kimi/ for comparison). Input: the rubric only (no verdicts,
no deliverables).
usage: dspy_venv/bin/python jev_judge/decompose.py -> data/jev_judge/decomp/<task>.json"""
import concurrent.futures as cf, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/../data/jev_judge/decomp"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from luna import ask, spent
PROMPT = """You turn the grading rubric of a legal deliverable into simple yes/no checks that a careful reader can verify \
from the deliverable's text alone (the deliverable is shown as text; tracked changes appear as insertions / deletions).

Task: {title}

For every criterion below, return:
- "id": the criterion id
- "checks": 1-5 statements about the deliverable. Each states ONE concrete thing the deliverable does or says, in plain \
words, keeping the criterion's exact names, numbers, dates, sections and quoted language. Write each as a statement, \
e.g. "The memo rates the exclusivity provision as high risk." / "The redline deletes the 'generally available within \
the data analytics industry' carve-out."
- each check has "pass_if": true when the statement must be TRUE to pass, false when the statement being true means \
FAIL (use false only for a FAIL condition that is not simply the opposite of a pass check).
- "logic": "all" if every pass_if check must hold (and no fail check), "any" if one of the alternatives is enough.
- "whole_document": true if judging needs the whole deliverable at once (a count, "every", "all N", coverage, \
consistency throughout, something being absent everywhere), else false.
Do not add checks the criterion does not require. Do not restate a check as its own negation.

Return JSON only: {{"criteria": [{{"id": "...", "logic": "all", "whole_document": false, "checks": [{{"s": "...", "pass_if": true}}]}}]}}

Criteria:
{criteria}"""


def one(task):
    out = f"{OUT}/{task.replace('/', '__')}.json"
    if os.path.exists(out): return task, "cached"
    cfg = json.load(open(f"{HERE}/../harvey-labs/tasks/{task}/task.json"))
    crit = "\n".join(f"[{c['id']}] {c['title']}: {c['match_criteria']}" for c in cfg["criteria"])
    for attempt in range(3):
        try:
            txt = ask(PROMPT.format(title=cfg["title"], criteria=crit), json_mode=True, tag="decompose")
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            got = {c["id"] for c in d["criteria"]}; want = {c["id"] for c in cfg["criteria"]}
            if want - got: raise ValueError(f"missing {sorted(want - got)[:5]}")
            d["model"] = "gpt-6-luna"
            json.dump(d, open(out, "w"), indent=1); return task, f"ok {len(got)} criteria, {sum(len(c['checks']) for c in d['criteria'])} checks"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"; time.sleep(5)
    return task, "FAILED " + err


if __name__ == "__main__":
    tasks = sorted({json.loads(l)["task"] for l in open(f"{HERE}/../data/jev_judge/items.jsonl")})
    with cf.ThreadPoolExecutor(16) as ex:  # Azure: 5,000 requests/min
        for t, msg in ex.map(one, tasks): print(t, msg, flush=True)
    print(f"Luna decomposition cost so far: ${spent('decompose'):.2f}")
