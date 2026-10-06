"""Model clients for the pipeline (2026-10-06): JSON answers from GPT-6 Luna (Azure, Responses API, reasoning effort from
PIPE_EFFORT, default max) or Gemma 4 26B-A4B 4-bit (llama.cpp on a Runpod 4090, tunnel 127.0.0.1:18081, json_object).
Usage is logged per call (Luna: runs/luna_usage.jsonl via jev_judge/luna.py; Gemma: runs/pipeline_gemma_usage.jsonl).
usage: m = LLM("luna" | "gemma26", tag); d = m.json(prompt)"""
import json, os, re, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{HERE}/../jev_judge")
GEMMA_URL = "http://127.0.0.1:18081/v1"; GEMMA_MODEL = "accounts/fireworks/models/gemma-4-26b"
_LOCK = threading.Lock()


def parse_json(txt):
    txt = (txt or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: raise ValueError("no JSON object")
    return json.loads(m.group(0))


class LLM:
    def __init__(self, name, tag):
        self.name, self.tag, self.calls, self.errors = name, tag, 0, 0
        if name == "gemma26":
            import openai
            self.client = openai.OpenAI(base_url=GEMMA_URL, api_key="local", timeout=1800, max_retries=2)

    def _raw(self, prompt):
        if self.name == "luna":
            from luna import ask
            return ask(prompt, json_mode=True, effort=os.environ.get("PIPE_EFFORT", "max"), tag=self.tag)
        # thinking off: with it, Gemma often spent the whole 8,192-token budget reasoning and returned no JSON (dev run)
        r = self.client.chat.completions.create(model=GEMMA_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3,
                                                max_tokens=8192, response_format={"type": "json_object"},
                                                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        u = r.usage
        with _LOCK, open(f"{HERE}/../runs/pipeline_gemma_usage.jsonl", "a") as f:
            f.write(json.dumps({"tag": self.tag, "input": u.prompt_tokens, "output": u.completion_tokens, "t": time.time()}) + "\n")
        return r.choices[0].message.content

    def json(self, prompt, default=None):
        """the model's JSON object; one repair retry; `default` (logged as an error) when it still fails."""
        with _LOCK: self.calls += 1
        for attempt in range(3):
            try:
                return parse_json(self._raw(prompt if attempt == 0 else prompt + "\n\nReturn ONE valid JSON object only, nothing else."))
            except Exception as e:
                err = e; time.sleep(3 * attempt)
        with _LOCK: self.errors += 1
        print(f"  [{self.tag}] JSON failed: {type(err).__name__}: {str(err)[:120]}", flush=True)
        return default
