"""Runs candidate tool-caller LLMs (Fireworks, OpenAI-style function calling) on the replay benchmark (build_replay.py).
Tools offered: LAB's six (bash, read, write, edit, glob, grep) + contract_tool (ours: find / checklist / ask).
  A  per state: the model's next tool call. Right if it calls contract_tool on a "call" state (the agent was about to
     read a source document) and does not on a "no_call" state (the agent was about to write / build the deliverable).
  B  per task: the model plans its contract_tool calls up front; coverage = the share of rubric criteria with a
     query whose bge-small cosine to the criterion (title + pass condition) >= T (0.60 / 0.70; an approximation).
usage: python run_replay.py A|B MODEL N   (MODEL: a Fireworks model id; resumable: data/out_<A|B>_<model>.jsonl)"""
import concurrent.futures as cf, json, os, random, sys, time
import requests
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("FIREWORKS_API_KEY="))
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
fn = lambda name, desc, props, req: {"type": "function", "function": {"name": name, "description": desc,
                                     "parameters": {"type": "object", "properties": props, "required": req}}}
S_ = {"type": "string"}
LAB_TOOLS = [
    fn("bash", "Run a shell command in the workspace.", {"command": S_}, ["command"]),
    fn("read", "Read a file (documents/ holds the matter's source documents: .docx, .xlsx, .eml, .pdf).", {"file_path": S_}, ["file_path"]),
    fn("write", "Write a file.", {"file_path": S_, "content": S_}, ["file_path", "content"]),
    fn("edit", "Replace text in a file.", {"file_path": S_, "old_string": S_, "new_string": S_}, ["file_path", "old_string", "new_string"]),
    fn("glob", "List files matching a pattern.", {"pattern": S_}, ["pattern"]),
    fn("grep", "Search file contents with a regular expression.", {"pattern": S_, "path": S_}, ["pattern"])]
CONTRACT_TOOL = fn(
    "contract_tool",
    "Reads the matter's source documents for you, so you do not need to read whole documents into your context. "
    "action=find: the passages of the documents most relevant to `query` (a topic, clause type, fact, number or defined term), "
    "each with its document and section. action=checklist: the standard review checklist for one contract (`document`; its type "
    "is detected): every item answered yes / no / not stated, with the clause. action=ask: a yes/no or value question about one "
    "`document`, answered with the supporting clause.",
    {"action": {"type": "string", "enum": ["find", "checklist", "ask"]}, "query": S_, "document": S_}, ["action"])


GKEY = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("GEMINI_API_KEY="))
GURL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"  # Gemma 4 (2026-10-05, the user: try Gemma)


_MUSE_TOK = None
_MUSE_LOCK = __import__("threading").Lock()
ATEM_INVOKE = __import__("re").compile(r'<atem:invoke name="([^"]+)">(.*?)</atem:invoke>', __import__("re").S)
ATEM_PARAM = __import__("re").compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', __import__("re").S)


def call_atem(model, messages, tools, max_tokens):
    """Muse Glimmer (2026-10-05): the servers' tool parsers don't read its ATEM format (Fireworks returns HTTP 400), so
    render Meta's own chat template with the tools, call the plain completions endpoint, parse <atem:invoke> blocks."""
    global _MUSE_TOK
    with _MUSE_LOCK:  # one load, not one per thread (concurrent `import transformers` breaks)
        if _MUSE_TOK is None:
            from transformers import AutoTokenizer
            _MUSE_TOK = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")
    prompt = _MUSE_TOK.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    url, key = (("http://127.0.0.1:18081/v1/completions", "local") if model == "muse-q4"
                else ("https://api.fireworks.ai/inference/v1/completions", KEY))
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    for i in range(6):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=300)
            if r.status_code == 200:
                d = r.json(); text = d["choices"][0].get("text") or ""; u = d.get("usage") or {}
                calls = []
                for name, inner in ATEM_INVOKE.findall(text):
                    args = {k: v.strip() for k, v in ATEM_PARAM.findall(inner)}
                    calls.append({"name": name.split(".")[-1], "arguments": json.dumps(args)})
                return {"calls": calls, "text": text[:500] if not calls else "", "in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}
            if r.status_code not in (429, 500, 502, 503, 529): return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1))
    return {"error": "retries exhausted"}


def call(model, messages, tools, max_tokens=1500):
    if model == "muse-q4" or "muse-glimmer" in model: return call_atem(model, messages, tools, max_tokens)
    body = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens, "temperature": 0.0}
    url, key = (("http://127.0.0.1:18080/v1/chat/completions", "local") if model == "gemma-4-e4b"  # vLLM on our pod via a tunnel
                else ("http://127.0.0.1:18081/v1/chat/completions", "local") if model == "muse-q4"  # llama.cpp, Muse Glimmer Q4_K_M
                else (GURL, GKEY) if model.startswith("gemma") else (URL, KEY))
    for i in range(6):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
            if r.status_code == 200:
                d = r.json(); m = d["choices"][0]["message"]; u = d.get("usage") or {}
                calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"} for c in m.get("tool_calls") or []]
                return {"calls": calls, "text": (m.get("content") or "")[:500], "in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}
            if r.status_code not in (429, 500, 502, 503, 529): return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except requests.RequestException as e:
            err = str(e)
        time.sleep(3 * (i + 1))
    return {"error": "retries exhausted"}


def history_text(h, last=30):
    lines = []
    for x in h[-last:]:
        if x[0] == "model": lines.append(f"[you] {(x[1] or '').strip()[:200]} -> " + ", ".join(x[2]))
        else: lines.append(f"[{x[1]} result] {(x[2] or '').strip()[:160]}")
    return "\n".join(lines)


GUIDE = ("To keep your context small, use contract_tool to find what you need in the documents instead of reading whole "
         "documents; read a document in full only when you must quote it at length or edit it.")
if os.environ.get("INSTR_FILE"):  # an optimized instruction (optimize_dspy.py) in place of the one-line guide
    GUIDE = open(os.environ["INSTR_FILE"]).read().strip()
MODE = os.environ.get("MODE", "guided")  # guided: the one-line usage instruction a deployment would add; neutral: none


def prompt_A(r):
    sys_ = ("You are a lawyer working as an agent in a sandbox workspace. Complete the task by calling tools. The matter's "
            "source documents are in documents/: " + ", ".join(r["documents"] or ["(see documents/)"]) + "."
            + (" " + GUIDE if MODE == "guided" else ""))
    user = (f"TASK: {r['title']}\n{r['instructions']}\n\nWHAT YOU HAVE DONE SO FAR ({len(r['history'])} steps; most recent last; "
            f"results are shortened here, you saw them in full):\n{history_text(r['history']) or '(nothing yet)'}\n\nMake your next tool call.")
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]


def prompt_B(r):
    sys_ = ("You are a lawyer working as an agent. The matter's source documents are in documents/: " + ", ".join(r["documents"]) + "."
            + (" " + GUIDE if MODE == "guided" else ""))
    user = (f"TASK: {r['title']}\n{r['instructions']}\n\nBefore drafting anything, gather every fact, clause, position and number "
            "you will need from the documents. Make all the contract_tool calls you need now (call it as many times as useful, in parallel).")
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]


def run(part, model, n):
    rows = [json.loads(l) for l in open(f"{HERE}/data/replay_{part}.jsonl")]
    split = os.environ.get("SPLIT")  # test: the held-out tasks of the optimization split (crc32 % 10 >= 7), all of them
    if split == "test":
        import zlib
        if part == "A": rows = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 >= 7]
        else: rows = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 >= 7][:n]
    elif part == "A":  # a balanced sample, same for every model
        rng = random.Random(1); c = [r for r in rows if r["label"] == "call"]; nc = [r for r in rows if r["label"] == "no_call"]
        rows = rng.sample(c, n // 2) + rng.sample(nc, n // 2)
    else:
        rows = random.Random(1).sample(rows, n)
    tag = os.environ.get("TAG", "")
    label = os.environ.get("LABEL") or model.split("/")[-1]  # e.g. LABEL=muse-bf16 for a Fireworks deployment string
    out = f"{HERE}/data/out_{part}_{label}_{MODE}{('_' + tag) if tag else ''}.jsonl"
    done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
    key = lambda r: f"{r['task']}|{r.get('panel', '')}|{len(r.get('history', []))}"
    todo = [r for r in rows if key(r) not in done]
    tools = LAB_TOOLS + [CONTRACT_TOOL] if part == "A" else [CONTRACT_TOOL]

    def one(r):
        return r, call(model, prompt_A(r) if part == "A" else prompt_B(r), tools, 1500 if part == "A" else 4000)
    with cf.ThreadPoolExecutor(4) as ex, open(out, "a") as f:  # 2 runs at a time = 8 in flight (live router keeps headroom)
        for r, res in ex.map(one, todo):
            f.write(json.dumps({"key": key(r), "task": r["task"], "label": r.get("label"), "actual": r.get("actual"), **res}) + "\n"); f.flush()
    print(f"{part} {model}: {len(todo)} asked, file {out}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
