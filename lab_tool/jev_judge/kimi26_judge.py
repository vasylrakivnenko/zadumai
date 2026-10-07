"""Kimi K2.6 Thinking on the user's Azure deployment ("Kimi-K2.6"; $0.95 / 1M in, $0.16 cached, $4 out; 1,000 req/min)
as LAB's judge (2026-10-06; the user: "Should we try it / validate it as judge?"). LAB's own judge prompt on the exact
text LAB's judge reads, on the same paired items as luna_max.py: every criterion inside Jev's uncertain band, 300 of the
Kimi-K3-graded ones (our runs), 200 random recorded ones. Responses API (reports cached tokens: exact cost).
usage: dspy_venv/bin/python jev_judge/kimi26_judge.py -> data/jev_judge/kimi26_verdicts.jsonl, kimi26_judge_eval.json"""
import concurrent.futures as cf, json, os, random, sys, threading, time
import numpy as np
from openai import OpenAI
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data/jev_judge"
KEY = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("AZURE_API_KEY="))
CLIENT = OpenAI(base_url="https://ai-vasyl-0670.services.ai.azure.com/openai/v1", api_key=KEY, timeout=900, max_retries=4)
MODEL = "Kimi-K2.6"; PRICE = (0.95, 0.16, 4.00)  # input, cached input, output per 1M (the user, 2026-10-06)
TEMPLATE = open(f"{HERE}/../harvey-labs/evaluation/prompts/rubric_criterion.txt").read()
OUT = f"{D}/kimi26_verdicts.jsonl"; USAGE = f"{HERE}/../runs/kimi26_usage.jsonl"; _LOCK = threading.Lock()
# K26_BRIEF=1: the grader's prompt (jev_judge/grade.py BRIEF: LAB's prompt + "keep your reasoning brief"), its own files
BRIEF = os.environ.get("K26_BRIEF") == "1"
if BRIEF:
    OUT = f"{D}/kimi26b_verdicts.jsonl"; USAGE = f"{HERE}/../runs/kimi26b_usage.jsonl"
# K26_MODEL=DeepSeek-V4-Flash: the same validation for DeepSeek V4 Flash (Azure; answers without reasoning there)
if os.environ.get("K26_MODEL") == "DeepSeek-V4-Flash":
    MODEL = "DeepSeek-V4-Flash"; PRICE = (0.19, 0.03, 0.51); OUT = f"{D}/dsv4_verdicts.jsonl"; USAGE = f"{HERE}/../runs/dsv4_judge_usage.jsonl"
# K26_MODEL=gpt-oss-120b: GPT-OSS-120B on Fireworks (2026-10-07, the user: "a small test to validate GPT OS 120B as a
# judge"), chat completions, reasoning effort medium; K26_ONLY=uncertain: only the hard items (Jev's uncertain band)
FIREWORKS = os.environ.get("K26_MODEL") == "gpt-oss-120b"
# K26_MODEL=claude-sonnet-5: Claude Sonnet 5 on Azure through ../providers.py (Anthropic Messages API, adaptive thinking low),
# the judge prompt cached up to the deliverable (consecutive criteria of one deliverable read it back)
CLAUDE = os.environ.get("K26_MODEL") == "claude-sonnet-5"
if CLAUDE:
    sys.path.insert(0, f"{HERE}/.."); import providers
    EP = providers.endpoint("claude-sonnet-5"); MODEL = "claude-sonnet-5"; PRICE = providers.price("claude-sonnet-5", "azure-anthropic")[:3]
    OUT = f"{D}/claude_verdicts.jsonl"; USAGE = f"{HERE}/../runs/claude_judge_usage.jsonl"
if FIREWORKS:
    FWKEY = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("FIREWORKS_API_KEY="))
    CLIENT = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=FWKEY, timeout=900, max_retries=4)
    MODEL = "accounts/fireworks/models/gpt-oss-120b"; PRICE = (0.15, 0.015, 0.60)
    OUT = f"{D}/gptoss_verdicts.jsonl"; USAGE = f"{HERE}/../runs/gptoss_judge_usage.jsonl"


def cost():
    if not os.path.exists(USAGE): return 0.0
    w = lambda u: u.get("cache_write", 0)  # Claude: cache writes at 1.25x input (assumed list price)
    return sum(((u["input"] - u["cached"] - w(u)) * PRICE[0] + u["cached"] * PRICE[1] + w(u) * PRICE[0] * 1.25 + u["output"] * PRICE[2]) / 1e6
               for u in map(json.loads, open(USAGE)))


def grade(r):
    cfg = json.load(open(f"{HERE}/../harvey-labs/tasks/{r['task']}/task.json")); c = next(x for x in cfg["criteria"] if x["id"] == r["cid"])
    prompt = TEMPLATE.format(task_description=cfg["title"], agent_output=open(f"{D}/texts/{r['text']}.txt").read(),
                             criterion_title=c["title"], match_criteria=c["match_criteria"])
    if BRIEF: prompt += "\n\nKeep your reasoning brief: decide from the decisive passage, in a few sentences, then give the JSON."
    v = "error"
    for attempt in range(5):
        try:
            if CLAUDE:
                pre = TEMPLATE.split("{criterion_title}")[0].format(task_description=cfg["title"], agent_output=open(f"{D}/texts/{r['text']}.txt").read())
                txt = EP.chat(prompt, tag="judge-val:claude", cache_prefix=pre) or ""
                with _LOCK, open(USAGE, "a") as f: f.write(json.dumps(EP.last_usage) + "\n")
            elif FIREWORKS:
                resp = CLIENT.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=16384,
                                                      response_format={"type": "json_object"}, extra_body={"reasoning_effort": "medium"})
                u = resp.usage; cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                with _LOCK, open(USAGE, "a") as f: f.write(json.dumps({"input": u.prompt_tokens, "cached": cached, "output": u.completion_tokens}) + "\n")
                txt = resp.choices[0].message.content or ""
            else:
                resp = CLIENT.responses.create(model=MODEL, input=prompt, text={"format": {"type": "json_object"}}, max_output_tokens=32768)  # Azure clamps to 4,096 (reasoning included): no verdict -> retried
                u = resp.usage
                with _LOCK, open(USAGE, "a") as f:
                    f.write(json.dumps({"input": u.input_tokens, "cached": getattr(u.input_tokens_details, "cached_tokens", 0) or 0, "output": u.output_tokens}) + "\n")
                txt = resp.output_text
            v = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]).get("verdict", "").lower(); break
        except Exception as e:  # Azure's token-per-minute cap answers 429: back off long (first run: 64 in flight -> 654 x 429)
            v = f"error: {type(e).__name__}"; time.sleep(30 * (attempt + 1))
    row = {k: r[k] for k in ("src", "episode", "task", "cid", "text", "verdicts", "why")}; row.update(kimi26=v, luna_default=r["luna_default"], luna_max=r["luna"])
    with _LOCK, open(OUT, "a") as f: f.write(json.dumps(row) + "\n")


def kappa(a, b):
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)); return (np.mean(a == b) - pe) / max(1e-9, 1 - pe)


def report():
    R = [r for r in map(json.loads, open(OUT)) if r["kimi26"] in ("pass", "fail")]
    out = {"model": MODEL, "cost_usd": round(cost(), 2), "errors": sum(1 for r in map(json.loads, open(OUT)) if r["kimi26"] not in ("pass", "fail"))}
    K = [r for r in R if r["why"] == "kimi"]
    if K:
        k = np.array([r["verdicts"][0] == "pass" for r in K]); d = {"n": len(K), "kimi_k3_pass_rate": round(float(k.mean()), 3)}
        for name in ("kimi26", "luna_max"):
            l = np.array([r[name] == "pass" for r in K]); d[name] = round(float(np.mean(l == k)), 4); d[name + "_kappa"] = round(float(kappa(l, k)), 3); d[name + "_pass_rate"] = round(float(l.mean()), 3)
        out["vs_kimi_k3"] = d
    for why in ("uncertain", "random"):
        S = [r for r in R if r["why"] == why and len(r["verdicts"]) == 3 and r["verdicts"][1] == r["verdicts"][2]]
        if not S: continue
        y = np.array([r["verdicts"][1] == "pass" for r in S]); one = np.array([r["verdicts"][0] == "pass" for r in S])
        d = {"n": len(S), "one_recorded_pass": round(float(np.mean(one == y)), 4), "one_recorded_pass_kappa": round(float(kappa(one, y)), 3)}
        for name in ("kimi26", "luna_max"):
            l = np.array([r[name] == "pass" for r in S]); d[name] = round(float(np.mean(l == y)), 4); d[name + "_kappa"] = round(float(kappa(l, y)), 3)
        out[f"recorded_{why}"] = d
    U = [json.loads(l) for l in open(USAGE)]
    out["per_call"] = {"input_tokens": int(np.mean([u["input"] for u in U])), "cached_share": round(sum(u["cached"] for u in U) / max(1, sum(u["input"] for u in U)), 3),
                       "output_tokens": int(np.mean([u["output"] for u in U])), "usd": round(cost() / len(U), 4)}
    name = "dsv4" if MODEL.startswith("DeepSeek") else "gptoss" if FIREWORKS else "claude" if CLAUDE else f"kimi26{'b' if BRIEF else ''}"
    print(json.dumps(out, indent=1)); json.dump(out, open(f"{D}/{name}_judge_eval.json", "w"), indent=1)


if __name__ == "__main__":
    items = [r for r in map(json.loads, open(f"{D}/luna_verdicts_max.jsonl")) if r["luna"] in ("pass", "fail")]  # luna_max's paired set
    # second pass (rate limits): every uncertain criterion + 100 of the Kimi-K3-graded ones, same deliverable back to back
    kimi = [r for r in items if r["why"] == "kimi"]; random.Random(1).shuffle(kimi)
    items = sorted([r for r in items if r["why"] == "uncertain"] + ([] if os.environ.get("K26_ONLY") == "uncertain" else kimi[:100]), key=lambda r: r["text"])
    if os.path.exists(OUT):  # redo the items with no verdict (Azure's 4,096-token output cap, reasoning included, cut them short)
        keep = [l for l in open(OUT) if json.loads(l)["kimi26"] in ("pass", "fail")]; open(OUT, "w").writelines(keep)
    done = {(r["text"], r["task"], r["cid"], r["why"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    todo = [r for r in items if (r["text"], r["task"], r["cid"], r["why"]) not in done]
    print(len(todo), "criteria", flush=True)
    with cf.ThreadPoolExecutor(int(os.environ.get("K26_PAR", 8))) as ex:  # 64 in flight hit Azure's token-per-minute cap; 8, same deliverables together
        for i, _ in enumerate(ex.map(grade, todo), 1):
            if i % 100 == 0: print(i, f"${cost():.2f}", flush=True)
    report()
