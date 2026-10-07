"""Cost ledger (2026-10-06; the user: "only count our costs starting from Microsoft Azure and also from today with Jev.
All of the costs that are Fireworks, let's leave them so we don't mix things"). Azure models (GPT-6 Luna, Kimi K2.6,
DeepSeek V4 Flash, Cohere rerank) and Jev, from 2026-10-06, from our usage logs; Fireworks and Runpod are not counted here.
Writes ../COSTS.md.  usage: dspy_venv/bin/python analysis/costs.py   (run from lab_tool/)"""
import glob, json, os, re
L = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# USD per 1M tokens (the user, 2026-10-06): (input, cached input, cache write, output); None = no separate rate
PRICES = {
    "GPT-6 Luna, prompt < 250k tokens": (0.11, 0.011, 0.14, 0.55),
    "GPT-6 Luna, prompt >= 250k tokens": (0.20, 0.02, 0.25, 0.75),
    "Kimi K2.6 Thinking": (0.95, 0.16, None, 4.00),
    "DeepSeek V4 Flash": (0.19, 0.03, None, 0.51),
    "Jev (TypeSafe)": (0.042, None, None, 0.0),
    "GPT-OSS-120B (Fireworks, from 2026-10-07)": (0.15, 0.015, None, 0.60),
    "DeepSeek V4.1 Flash (Fireworks, from 2026-10-07)": (0.30, 0.006, None, 1.20),
}
# USD per 1K searches (the user, 2026-10-06); our pipeline uses the Pro model (cohere-rerank-v4.0-pro)
SEARCH_PRICES = {"Cohere Rerank 4.0 Fast": 2.00, "Cohere Rerank 4.0 Pro": 2.50}


def usd(price, i, c=0, w=0, o=0):
    """Azure's input count includes cached and cache-written tokens."""
    p = PRICES[price]
    return (max(0, i - c - w) * p[0] + c * (p[1] if p[1] is not None else p[0]) + w * (p[2] if p[2] is not None else p[0]) + o * p[3]) / 1e6


def luna_price(i):
    return "GPT-6 Luna, prompt >= 250k tokens" if i >= 250_000 else "GPT-6 Luna, prompt < 250k tokens"


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def rows():
    R = []
    def add(model, job, U, price):
        n = len(U); i = sum(u[0] for u in U); c = sum(u[1] for u in U); w = sum(u[2] for u in U); o = sum(u[3] for u in U)
        if n: R.append((model, job, n, i, c, w, o, sum(usd(price(u[0]) if callable(price) else price, *u) for u in U)))
    tags, runs = {}, set()
    for u in load(f"{L}/runs/luna_usage.jsonl"):  # pipeline runs (tags pipe:<run id>) and v5 replays (v5val:...) are summed into one row each
        t = "pipe:*" if u["tag"].startswith("pipe:") else "v5val:*" if u["tag"].startswith("v5val:") else u["tag"]
        if t == "pipe:*": runs.add(u["tag"])
        tags.setdefault(t, []).append((u["input"], u.get("cached", 0), u.get("cache_write", 0), u["output"]))
    names = {"pipe:*": f"pipeline runs ({len(runs)} runs)", "v5val:*": "v5 validation: replays of logged pipeline decisions"}
    for tag, U in tags.items(): add("GPT-6 Luna", names.get(tag, f"jev_judge: {tag}"), U, luna_price)
    U = [(u["prompt"], u["cached"], u.get("cache_write", 0), u["completion"]) for f in glob.glob(f"{L}/runs/pilot/*-luna*.usage.jsonl") for u in load(f)]
    add("GPT-6 Luna", "LAB agent episodes (max effort)", U, luna_price)
    U = [(u["input"], u["cached"], u.get("cache_write", 0), u["output"]) for u in load(f"{L}/runs/kimi26_usage.jsonl")]
    add("Kimi K2.6 Thinking", "judge validation", U, "Kimi K2.6 Thinking")
    G = load(f"{L}/runs/judge_usage_grade_k26.jsonl")
    add("Kimi K2.6 Thinking", "grading: criteria Jev is unsure of (jev_judge/grade.py), LAB's prompt", [(u["input"], u["cached"], 0, u["output"]) for u in G if not u.get("brief")], "Kimi K2.6 Thinking")
    add("Kimi K2.6 Thinking", "grading: criteria Jev is unsure of, LAB's prompt + 'reason briefly'", [(u["input"], u["cached"], 0, u["output"]) for u in G if u.get("brief")], "Kimi K2.6 Thinking")
    R.append(("Kimi K2.6 Thinking", "judge pilot: 'reason briefly' on 30 hard items (a scratch test; total from its printout)", 33, 0, 0, 0, 0, 0.91))
    U = [(u["input"], u["cached"], 0, u["output"]) for u in load(f"{L}/runs/kimi26b_usage.jsonl")]
    add("Kimi K2.6 Thinking", "judge validation, 'reason briefly' prompt", U, "Kimi K2.6 Thinking")
    # pipeline / replay calls of the OpenAI-compatible models (providers.load_usage(): runs/llm_usage.jsonl + the older logs)
    import sys; sys.path.insert(0, L); import providers
    LABEL = {("deepseek-v4-flash", "azure"): "DeepSeek V4 Flash", ("kimi-k2.6", "azure"): "Kimi K2.6 Thinking",
             ("deepseek-v4.1-flash", "azure"): "DeepSeek V4.1 Flash (Azure)", ("deepseek-v4.1-flash", "fireworks"): "DeepSeek V4.1 Flash (Fireworks, from 2026-10-07)",
             ("gpt-oss-120b", "fireworks"): "GPT-OSS-120B (Fireworks, from 2026-10-07)"}
    groups = {}
    for u in providers.load_usage():
        if u["tag"].startswith("test:"): continue
        job = "v5 validation: replays of logged pipeline decisions" if u["tag"].startswith("v5val:") else "pipeline runs"
        groups.setdefault((u["model"], u["provider"], job), []).append(u)
    for (m, pv, job), U in sorted(groups.items()):
        c = [providers.cost(u) for u in U]
        R.append((LABEL.get((m, pv), f"{m} ({pv})"), job + ("" if None not in c else " (price not set: $0 here)"), len(U), sum(u["input"] for u in U),
                  sum(u["cached"] for u in U), 0, sum(u["output"] for u in U), sum(x or 0 for x in c)))
    C = [u for u in load(f"{L}/runs/cohere_usage.jsonl") if u.get("searches", 1)]  # failed searches (logged since v4.1) are not billed
    if C: R.append(("Cohere Rerank 4.0 Pro", f"pipeline search ({len(C)} searches, ${SEARCH_PRICES['Cohere Rerank 4.0 Pro']:g} / 1K)", len(C), 0, 0, 0, 0,
                    len(C) * SEARCH_PRICES["Cohere Rerank 4.0 Pro"] / 1000))
    U = [(u["prompt"], u["cached"], u.get("cache_write", 0), u["completion"]) for f in glob.glob(f"{L}/runs/pilot/*-dsv4*.usage.jsonl") for u in load(f)]
    add("DeepSeek V4 Flash", "LAB agent episodes", U, "DeepSeek V4 Flash")
    # Jev: the first check run (before per-run logging) from its log; later runs from jev_usage.jsonl
    if os.path.exists(f"{L}/runs/jev_check.log"):
        m = re.findall(r"Jev calls (\d+), tokens in ([\d,]+)", open(f"{L}/runs/jev_check.log").read())
        if m: R.append(("Jev", "jev_judge: check, all recorded criteria (from its log)", int(m[-1][0]), int(m[-1][1].replace(",", "")), 0, 0, 0,
                        usd("Jev (TypeSafe)", int(m[-1][1].replace(",", "")))))
    # Fireworks, from 2026-10-07 only (the user added $20 for the GPT-OSS-120B tests; earlier Fireworks costs stay out)
    F = "GPT-OSS-120B (Fireworks, from 2026-10-07)"
    side = lambda f: os.path.exists(f.replace(".usage.jsonl", ".model.json"))  # runs from 2026-10-07 record their model + provider
    add(F, "LAB agent episodes (LAB harness)", [(u["prompt"], u.get("cached", 0), 0, u["completion"]) for f in glob.glob(f"{L}/runs/pilot/*-gptoss*.usage.jsonl") if not side(f) for u in load(f)], F)
    eps = {}
    for f in glob.glob(f"{L}/runs/pilot/*.model.json"):  # LAB agent episodes with a model record: priced from providers.py
        m = json.load(open(f)); U = load(f.replace(".model.json", ".usage.jsonl"))
        eps.setdefault((m["model"], m["provider"]), []).extend(U)
    for (m, pv), U in sorted(eps.items()):
        pr = providers.price(m, pv); c = sum(((u["prompt"] - u.get("cached", 0)) * pr[0] + u.get("cached", 0) * pr[1] + u["completion"] * pr[2]) / 1e6 for u in U) if pr else 0.0
        R.append((LABEL.get((m, pv), f"{m} ({pv})"), "LAB agent episodes (LAB harness)" + ("" if pr else " (price not set: $0 here)"), len(U), sum(u["prompt"] for u in U),
                  sum(u.get("cached", 0) for u in U), 0, sum(u["completion"] for u in U), c))
    add(F, "judge validation (hard items)", [(u["input"], u["cached"], 0, u["output"]) for u in load(f"{L}/runs/gptoss_judge_usage.jsonl")], F)
    jobs = {}
    for u in load(f"{L}/runs/jev_usage.jsonl"):  # one line per run, summed by job
        name = "fixed-pipeline runs, abandoned (Jev labels)" if u["job"].startswith("pipeline:") else {"grade": "jev_judge: grade", "v5_validate": "Jev labels (v5 validation)"}.get(u["job"], u["job"])
        j = jobs.setdefault(name, [0, 0, 0]); j[0] += u["calls"]; j[1] += u["input"]; j[2] += 1
    for job, (calls, inp, runs) in jobs.items():
        R.append(("Jev", f"{job} ({runs} runs)", calls, inp, 0, 0, 0, usd("Jev (TypeSafe)", inp)))
    return R


if __name__ == "__main__":
    R = rows(); fmt = lambda x: "-" if x is None else f"${x:g}"
    out = ["# Cost ledger: Azure models + Jev (from 2026-10-06)", "",
           "Generated by `analysis/costs.py` from our usage logs (re-run it to update). Fireworks and Runpod costs are not",
           "counted here (the user: keep them apart). Azure's input count includes cached and cache-written tokens. Calls logged",
           "before cache writes were recorded (jev_judge: luna_judge, luna_judge_max) count every input token at the input rate.", "",
           "## Prices (USD per 1M tokens)", "", "| model | input | cached input | cache write | output |", "|---|---|---|---|---|"]
    out += [f"| {k} | " + " | ".join(fmt(x) for x in v) + " |" for k, v in PRICES.items()]
    out += ["", "| reranker | USD per 1K searches |", "|---|---|"] + [f"| {k} | ${v:.2f} |" for k, v in SEARCH_PRICES.items()]
    out += ["", "## Usage", "", "| model | job | calls | input | cached | cache write | output | USD |", "|---|---|---|---|---|---|---|---|"]
    tot = {}
    for m, job, n, i, c, w, o, d in R:
        out.append(f"| {m} | {job} | {n} | {i / 1e6:.1f}M | {c / 1e6:.1f}M | {w / 1e6:.2f}M | {o / 1e6:.2f}M | ${d:.2f} |"); tot[m] = tot.get(m, 0) + d
    out += ["", "| model | USD |", "|---|---|"] + [f"| {m} | ${d:.2f} |" for m, d in tot.items()] + [f"| **total** | **${sum(tot.values()):.2f}** |"]
    open(f"{L}/COSTS.md", "w").write("\n".join(out) + "\n"); print("\n".join(out[out.index("## Usage"):]))
