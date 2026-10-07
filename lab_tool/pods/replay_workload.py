"""A fixed Gemma workload for comparing GPU setups (2026-10-06; the user: "Make sure you measure that serverless costs
correctly. We want to compare them correctly with our dedicated pod with storage-optimized start."). The 453 logged
pipeline prompts of the 4 dev tasks (runs/pipe_items/pipe-v42L-*.jsonl: review, markup and gap-check decisions), sent
exactly as the pipeline sends them to Gemma (temperature 0.3, max_tokens 8192, json_object, thinking off), 4 in flight.
Logs every call (tokens, seconds, errors) and a summary; billed time and dollars come from Runpod's billing API.
usage: dspy_venv/bin/python pods/replay_workload.py URL TAG [--model NAME] [--key KEY]   (URL: an OpenAI-compatible /v1 base)"""
import argparse, concurrent.futures as cf, glob, json, os, threading, time
import openai
L = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("tag")
ap.add_argument("--model", default="accounts/fireworks/models/gemma-4-26b"); ap.add_argument("--key", default="local")
ap.add_argument("--parallel", type=int, default=4)
A = ap.parse_args()
P = [json.loads(l)["prompt"] for f in sorted(glob.glob(f"{L}/runs/pipe_items/pipe-v42L-*.jsonl")) for l in open(f)
     if json.loads(l).get("step") in ("review", "markup", "gap_check")]
C = openai.OpenAI(base_url=A.url, api_key=A.key, timeout=900, max_retries=0)
LOG = f"{L}/runs/gpu_compare/{A.tag}.jsonl"; os.makedirs(os.path.dirname(LOG), exist_ok=True); lock = threading.Lock()


def one(i):
    t = time.time(); err = None; u = None
    for attempt in range(6):  # a serverless worker may be cold or busy: retry
        try:
            r = C.chat.completions.create(model=A.model, messages=[{"role": "user", "content": P[i]}], temperature=0.3, max_tokens=8192,
                                          response_format={"type": "json_object"}, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            u = r.usage; err = None; break
        except Exception as e:
            err = repr(e)[:200]; time.sleep(5 * (attempt + 1))
    row = {"i": i, "s": round(time.time() - t, 2), "input": getattr(u, "prompt_tokens", 0), "output": getattr(u, "completion_tokens", 0), "error": err, "t": time.time()}
    with lock, open(LOG, "a") as f: f.write(json.dumps(row) + "\n")
    return row


t0 = time.time()
with cf.ThreadPoolExecutor(A.parallel) as ex: R = list(ex.map(one, range(len(P))))
S = {"tag": A.tag, "url": A.url, "prompts": len(P), "errors": sum(1 for r in R if r["error"]), "wall_s": round(time.time() - t0, 1),
     "input_tokens": sum(r["input"] for r in R), "output_tokens": sum(r["output"] for r in R), "start": t0, "end": time.time()}
json.dump(S, open(f"{L}/runs/gpu_compare/{A.tag}.summary.json", "w"), indent=1); print(json.dumps(S))
