"""Model endpoints in one place (2026-10-07; the user: "Let's use MS Azure's DeepSeek-V4.1-Flash since now. But make some
kind of easy swiping between Azure and Fireworks so you don't rewrite the code all the time").

A model (a logical name) has one deployment per provider. The provider used is, first match:
  the `provider` argument  >  env PROVIDER_<MODEL> (e.g. PROVIDER_DEEPSEEK_V4_1_FLASH=fireworks)  >  env PROVIDER  >  the
  model's default (Azure wherever the model is there).
Every call made through `Endpoint.chat()` is logged to runs/llm_usage.jsonl (model, provider, tag, tokens), and
`load_usage()` reads that log plus the older per-backend logs in one shape, priced by `price()`.

usage:
  from providers import endpoint
  ep = endpoint("deepseek-v4.1-flash")                 # Azure (default)
  ep = endpoint("deepseek-v4.1-flash", "fireworks")    # or PROVIDER=fireworks in the environment
  text = ep.chat(prompt, tag="pipe:<run id>", effort="low", json=True)
  ep.lab_env()  -> env vars for LAB's harness (its OpenAI-compatible "Fireworks" adapter pointed at this endpoint)"""
import json, os, threading, time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__)); LOG = f"{HERE}/runs/llm_usage.jsonl"; _LOCK = threading.Lock()
PROVIDERS = {"azure": ("https://ai-vasyl-0670.services.ai.azure.com/openai/v1", "AZURE_API_KEY"),
             "azure-anthropic": ("https://ai-vasyl-0670.services.ai.azure.com/anthropic", "AZURE_API_KEY"),  # Claude on Azure: Anthropic Messages API
             "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY")}
# per provider: (deployment id, USD per 1M tokens (input, cached input, output) or None if not given yet)
MODELS = {
    "deepseek-v4.1-flash": {"azure": ("DeepSeek-V4.1-Flash", None),  # Azure: reasoning off unless reasoning_effort is sent
                            "fireworks": ("accounts/fireworks/models/deepseek-v4p1-flash", (0.30, 0.006, 1.20)),
                            "effort": "low", "temperature": 0.6},  # as the best LAB agent ran it (low effort, 0.6)
    "deepseek-v4-flash": {"azure": ("DeepSeek-V4-Flash", (0.19, 0.03, 0.51)), "effort": None, "temperature": 0.3},
    "kimi-k2.6": {"azure": ("Kimi-K2.6", (0.95, 0.16, 4.00)), "effort": None, "temperature": 0.3},  # 4,096 output tokens max incl. reasoning
    "gpt-oss-120b": {"fireworks": ("accounts/fireworks/models/gpt-oss-120b", (0.15, 0.015, 0.60)), "effort": "medium", "temperature": 0.3},
    # Claude Sonnet 5 on the user's Azure (2026-10-07): only the Anthropic Messages API (x-api-key); adaptive thinking with
    # output_config.effort (budget_tokens is rejected); no temperature; 5,000 requests or 5M tokens per minute; price not given yet
    "claude-sonnet-5": {"azure-anthropic": ("claude-sonnet-5", (2.00, 0.20, 10.00, 2.50)), "effort": "low", "temperature": None},  # $2 / $10 (the user, corrected 2026-10-07); cache read 0.1x, write 1.25x (Anthropic's standard, assumed)
}
# assumed prices for models whose Azure price the user has not given yet (input, cache read, output, cache write)
ASSUMED = {}  # (claude-sonnet-5 had Anthropic's list price here until the user gave the Azure price)


def _key(name):
    for f in ("/root/.env", "/root/projects/zadumai/.env"):
        if os.path.exists(f):
            for l in open(f):
                if l.startswith(name + "="): return l.split("=", 1)[1].strip().strip('"\'')
    raise KeyError(name)


def provider_for(model, provider=None):
    m = MODELS[model]
    p = provider or os.environ.get("PROVIDER_" + model.upper().replace("-", "_").replace(".", "_")) or os.environ.get("PROVIDER")
    if p and p in m: return p
    return next(x for x in ("azure", "azure-anthropic", "fireworks") if x in m)  # default: Azure where available (also if PROVIDER names a provider without this model)


def price(model, provider):
    m = MODELS.get(model, {}); return m[provider][1] if provider in m else None


@dataclass
class Endpoint:
    model: str; provider: str; deployment: str; base_url: str; key_env: str

    def client(self, timeout=900, max_retries=4):
        import openai
        return openai.OpenAI(base_url=self.base_url, api_key=_key(self.key_env), timeout=timeout, max_retries=max_retries)

    def chat(self, prompt, tag="", effort="default", json=True, max_tokens=16384, client=None, cache_prefix=None):
        """one call; effort "default" = the model's own setting, None / "none" = no reasoning. Anthropic models: the
        Messages API; cache_prefix = the leading part of the prompt to cache (e.g. the judge prompt up to the deliverable)"""
        m = MODELS[self.model]; eff = m.get("effort") if effort == "default" else (None if effort in (None, "none") else effort)
        if self.provider == "azure-anthropic": return self._anthropic(prompt, tag, eff, max_tokens, cache_prefix)
        kw = {"max_tokens": max_tokens}
        if json: kw["response_format"] = {"type": "json_object"}
        if eff: kw["extra_body"] = {"reasoning_effort": eff}
        if m.get("temperature") is not None: kw["temperature"] = m["temperature"]
        r = (client or self.client()).chat.completions.create(model=self.deployment, messages=[{"role": "user", "content": prompt}], **kw)
        u = r.usage; cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        with _LOCK, open(LOG, "a") as f:
            f.write(json_dumps({"model": self.model, "provider": self.provider, "effort": eff, "tag": tag, "input": u.prompt_tokens, "cached": cached,
                                "output": u.completion_tokens, "t": time.time()}) + "\n")
        return r.choices[0].message.content

    def _anthropic(self, prompt, tag, eff, max_tokens, cache_prefix):
        import urllib.request
        if cache_prefix and prompt.startswith(cache_prefix) and len(cache_prefix) > 4000:
            content = [{"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": prompt[len(cache_prefix):]}]
        else:
            content = prompt
        body = {"model": self.deployment, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        if eff: body.update(thinking={"type": "adaptive"}, output_config={"effort": eff})
        req = urllib.request.Request(self.base_url + "/v1/messages", data=json_dumps(body).encode(),
                                     headers={"x-api-key": _key(self.key_env), "anthropic-version": "2023-06-01", "Content-Type": "application/json", "User-Agent": "zadum"})
        for attempt in range(5):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=900).read()); break
            except Exception as e:
                if attempt == 4: raise
                time.sleep(10 * (attempt + 1))
        u = r.get("usage", {}); cr = u.get("cache_read_input_tokens", 0) or 0; cw = u.get("cache_creation_input_tokens", 0) or 0
        self.last_usage = {"input": u.get("input_tokens", 0) + cr + cw, "cached": cr, "cache_write": cw, "output": u.get("output_tokens", 0)}
        with _LOCK, open(LOG, "a") as f:
            f.write(json_dumps(dict({"model": self.model, "provider": self.provider, "effort": eff, "tag": tag}, **self.last_usage, t=time.time())) + "\n")
        return "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")

    def lab_env(self):
        """LAB's harness: its OpenAI-compatible adapter (harness/adapters/fireworks.py) pointed at this endpoint; our patch
        LAB_API_MODEL sends the deployment id. Pass the harness --model accounts/fireworks/models/<anything> (routing only)."""
        if self.provider == "azure-anthropic":  # LAB's Anthropic adapter (anthropic.Anthropic() reads these)
            return {"ANTHROPIC_BASE_URL": self.base_url, "ANTHROPIC_API_KEY": _key(self.key_env)}
        return {"FIREWORKS_API_BASE": self.base_url, "FIREWORKS_API_KEY": _key(self.key_env), "LAB_API_MODEL": self.deployment}


def json_dumps(o):
    return json.dumps(o)


def endpoint(model, provider=None):
    p = provider_for(model, provider); base, key = PROVIDERS[p]
    return Endpoint(model, p, MODELS[model][p][0], base, key)


def load_usage():
    """every logged call of the OpenAI-compatible models, one shape: {model, provider, tag, input, cached, output, t}"""
    rows = []
    def read(f):
        return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    rows += read(LOG)
    old = {"DeepSeek-V4-Flash": "deepseek-v4-flash", "Kimi-K2.6": "kimi-k2.6"}  # before 2026-10-07: per-backend logs
    rows += [dict(u, model=old[u["model"]], provider="azure") for u in read(f"{HERE}/runs/pipeline_azure_usage.jsonl")]
    rows += [dict(u, model="gpt-oss-120b", provider="fireworks") for u in read(f"{HERE}/runs/pipeline_gptoss_usage.jsonl")]
    rows += [dict(u, model="deepseek-v4.1-flash", provider="fireworks") for u in read(f"{HERE}/runs/pipeline_ds41_usage.jsonl")]
    return rows


def cost(u, assumed=False):
    """USD for one logged call; None if the price is not set (assumed=True: use ASSUMED list prices where given)"""
    p = price(u["model"], u["provider"]) or (ASSUMED.get(u["model"]) if assumed else None)
    if p is None: return None
    w = u.get("cache_write", 0); wp = p[3] if len(p) > 3 else p[0]
    return ((u["input"] - u["cached"] - w) * p[0] + u["cached"] * p[1] + w * wp + u["output"] * p[2]) / 1e6


if __name__ == "__main__":
    for name in MODELS:
        ep = endpoint(name); print(f"{name:22s} -> {ep.provider:9s} {ep.deployment:48s} price {price(name, ep.provider)}")
