"""GPT-6 Luna on the user's Azure AI deployment (2026-10-06; the user: "Where possible - let's switch to this API with
GPT-6-Luna"). Responses API; 5,000 requests/min; $0.10 in / $0.50 out per 1M tokens (prompts under 250k tokens),
$0.20 / $0.75 above. Every call's usage is appended to a log so costs are exact.
usage: from luna import ask; text = ask(prompt, json_mode=True)"""
import json, os, threading, time
from openai import OpenAI
ENDPOINT = "https://ai-vasyl-0670.services.ai.azure.com/openai/v1"; MODEL = "gpt-6-luna"
KEY = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("AZURE_API_KEY="))
CLIENT = OpenAI(base_url=ENDPOINT, api_key=KEY, timeout=900, max_retries=4)
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../runs/luna_usage.jsonl"); _LOCK = threading.Lock()


def cost(u):
    long = u["input"] > 250_000
    return (u["input"] * (0.20 if long else 0.10) + u["output"] * (0.75 if long else 0.50)) / 1e6


def ask(prompt, json_mode=False, effort=None, tag=""):
    """the model's text answer (JSON text when json_mode); retries transient errors."""
    kw = {"model": MODEL, "input": prompt}
    if json_mode: kw["text"] = {"format": {"type": "json_object"}}
    if effort: kw["reasoning"] = {"effort": effort}
    for attempt in range(5):
        try:
            r = CLIENT.responses.create(**kw); break
        except Exception as e:
            if attempt == 4: raise
            time.sleep(5 * (attempt + 1))
    u = {"tag": tag, "input": r.usage.input_tokens, "cached": getattr(r.usage.input_tokens_details, "cached_tokens", 0) or 0,
         "output": r.usage.output_tokens, "t": time.time()}
    with _LOCK, open(LOG, "a") as f: f.write(json.dumps(u) + "\n")
    return r.output_text


def spent(tag=None):
    if not os.path.exists(LOG): return 0.0
    return sum(cost(u) for u in map(json.loads, open(LOG)) if tag is None or u["tag"] == tag)
