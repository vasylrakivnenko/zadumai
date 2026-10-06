"""DSPy prompt optimization of the tool caller (2026-10-05; the user: "Before fine tuning we should try DSPy prompt
optimization", then "Go for it"). Model being tuned: Gemma 4 E4B (vLLM on a Runpod 4090, tunnel 127.0.0.1:18080);
reflection / proposal model: Kimi K3 (Fireworks). Native function calling (LAB's six tools + contract_tool).
Part A (WHEN to call): replay_A states split BY TASK (crc32 % 10 < 7: optimization, else test). Optimization tasks give
a balanced train set and a balanced val set; the test set is every state of the test tasks, scored as balanced accuracy.
Optimizers compared on the same test set: the starting instruction (= run_replay's guided line), BootstrapFewShot
(worked examples picked from the program's own correct answers), GEPA (instruction rewritten from failures).
usage: dspy_venv/bin/python optimize_dspy.py A   -> data/dspy_A_*.json, printed scores"""
import json, os, random, sys, zlib
import dspy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_replay as R  # history_text, GUIDE, tool descriptions
THREADS = int(os.environ.get("THREADS", "24"))  # ~24 requests in flight keep vLLM's prefill saturated (shared prefixes are cached)
FW = next(l.split("=", 1)[1].strip().strip('"\'') for l in open("/root/.env") if l.startswith("FIREWORKS_API_KEY="))

gemma = dspy.LM("openai/gemma-4-e4b", api_base="http://127.0.0.1:18080/v1", api_key="local", temperature=0.0, max_tokens=800)
kimi = dspy.LM("fireworks_ai/accounts/fireworks/models/kimi-k3", api_key=FW, temperature=1.0, max_tokens=16000)
dspy.configure(lm=gemma, adapter=dspy.ChatAdapter(use_native_function_calling=True))


# the tools, as the caller sees them (never executed here: only the choice is scored)
def bash(command: str):
    """Run a shell command in the workspace."""
def read(file_path: str):
    """Read a file (documents/ holds the matter's source documents: .docx, .xlsx, .eml, .pdf)."""
def write(file_path: str, content: str):
    """Write a file."""
def edit(file_path: str, old_string: str, new_string: str):
    """Replace text in a file."""
def glob(pattern: str):
    """List files matching a pattern."""
def grep(pattern: str, path: str = ""):
    """Search file contents with a regular expression."""
def contract_tool(action: str, query: str = "", document: str = ""):
    pass
contract_tool.__doc__ = R.CONTRACT_TOOL["function"]["description"] + " action is one of: find, checklist, ask."
TOOLS = [dspy.Tool(f) for f in (bash, read, write, edit, glob, grep, contract_tool)]


class NextStep(dspy.Signature):
    __doc__ = ("You are a lawyer working as an agent in a sandbox workspace. Complete the task by calling tools. " + R.GUIDE
               + " Make your next tool call.")
    task: str = dspy.InputField(desc="the task title and instructions")
    documents: str = dspy.InputField(desc="the matter's source documents in documents/")
    history: str = dspy.InputField(desc="what you have done so far, most recent last (results shortened; you saw them in full)")
    tools: list[dspy.Tool] = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


class Caller(dspy.Module):
    def __init__(self):
        super().__init__(); self.step = dspy.Predict(NextStep)

    def forward(self, task, documents, history):
        return self.step(task=task, documents=documents, history=history, tools=TOOLS)


def example(r):
    return dspy.Example(task=f"{r['title']}\n{r['instructions']}", documents=", ".join(r["documents"]) or "(see documents/)",
                        history=R.history_text(r["history"]) or "(nothing yet)", label=r["label"]).with_inputs("task", "documents", "history")


def used_tool(pred):
    calls = getattr(getattr(pred, "tool_calls", None), "tool_calls", None) or []
    return any(c.name == "contract_tool" for c in calls), [c.name for c in calls]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    used, names = used_tool(pred); ok = used if gold.label == "call" else not used
    if pred_name is None and pred_trace is None: return float(ok)
    if gold.label == "call":
        fb = ("Correct: the agent needed information from the source documents and you used contract_tool." if ok else
              f"Wrong: you called {names or 'nothing'}. The agent's next need was information from a source document; "
              "contract_tool should have found it instead of reading the whole document.")
    else:
        fb = ("Correct: the agent was writing, editing or building its own deliverable, and you did not call contract_tool." if ok else
              "Wrong: you called contract_tool, but the agent already had the information and was writing, editing or "
              "checking its own deliverable (or its output files). contract_tool is only for finding information in the source documents.")
    return dspy.Prediction(score=float(ok), feedback=fb)


def split_A():
    rows = [json.loads(l) for l in open(f"{HERE}/data/replay_A.jsonl")]
    opt = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 < 7]; test = [r for r in rows if zlib.crc32(r["task"].encode()) % 10 >= 7]
    rng = random.Random(0)
    bal = lambda rs, n: rng.sample([r for r in rs if r["label"] == "call"], n // 2) + rng.sample([r for r in rs if r["label"] == "no_call"], n // 2)
    pool = bal(opt, 240); rng.shuffle(pool)
    return [example(r) for r in pool[:120]], [example(r) for r in pool[120:]], [example(r) for r in test], test


def balanced(prog, data):
    ev = dspy.Evaluate(devset=data, metric=metric, num_threads=THREADS, display_progress=False, provide_traceback=True)
    res = ev(prog)
    rows = res.results if hasattr(res, "results") else []
    by = {"call": [], "no_call": []}
    for ex, pred, score in rows: by[ex.label].append(float(score))
    rc, rn = sum(by["call"]) / max(len(by["call"]), 1), sum(by["no_call"]) / max(len(by["no_call"]), 1)
    return {"uses_when_should": rc, "holds_off_when_writing": rn, "balanced": (rc + rn) / 2, "n": len(rows)}


if __name__ == "__main__":
    train, val, test, test_rows = split_A()
    print(f"A: train {len(train)}, val {len(val)} (optimization tasks, balanced), test {len(test)} states from test tasks "
          f"(call {sum(e.label == 'call' for e in test)}, no_call {sum(e.label == 'no_call' for e in test)})", flush=True)
    out = {}
    base = Caller(); out["start"] = balanced(base, test); print("start instruction:", out["start"], flush=True)
    fs = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=0, max_rounds=1).compile(Caller(), trainset=train)
    out["fewshot"] = balanced(fs, test); print("BootstrapFewShot:", out["fewshot"], flush=True); fs.save(f"{HERE}/data/dspy_A_fewshot.json")
    gepa = dspy.GEPA(metric=metric, auto="light", reflection_lm=kimi, num_threads=THREADS, track_stats=True, seed=0)
    opt = gepa.compile(Caller(), trainset=train, valset=val)
    out["gepa"] = balanced(opt, test); print("GEPA:", out["gepa"], flush=True); opt.save(f"{HERE}/data/dspy_A_gepa.json")
    out["gepa_instruction"] = opt.step.signature.instructions
    print("GEPA instruction:\n" + out["gepa_instruction"], flush=True)
    json.dump(out, open(f"{HERE}/data/dspy_A_results.json", "w"), indent=1)
