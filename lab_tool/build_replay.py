"""The tool-caller benchmark for the Legal Agent Benchmark (LAB) tool approach (2026-10-05; the user: benchmark the tool
caller on its own, before any full LAB run). No API calls here: it builds the cases; run_replay.py runs candidate LLMs.
  A. WHEN to call (per turn): states cut from real agent transcripts (Ivo Sage's evaluation record: the base model's
     and Ivo Sage's episodes on LAB's held-out GENERAL tasks; the 55 held-out Contracts tasks stay clean for the pilot).
     positive = the agent's next action reads or extracts a SOURCE document (it should call contract_tool instead);
     negative = its next action writes, edits or builds the deliverable (it should not call contract_tool).
     Context given: task instructions, the document list, and a compact history of the previous actions and results.
  B. WHAT to ask (per task): LAB TRAINING-split Contracts tasks (harvey-labs at the pinned commit): instructions +
     documents + the rubric. The caller plans its contract_tool queries up front; scored by how many rubric criteria
     (what the graders check) the queries cover.
usage: python build_replay.py -> data/replay_A.jsonl, data/replay_B.jsonl"""
import csv, glob, gzip, json, os, random, re
HERE = os.path.dirname(os.path.abspath(__file__))
S = "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace"
EP = f"{S}/art/un/train/deepseek-v4-flash"
REPO = f"{HERE}/harvey-labs"
os.makedirs(f"{HERE}/data", exist_ok=True)
DOCS = re.compile(r"documents/")
BUILD = re.compile(r"\b(scratch|output)/|generate_from_md|build_|python-docx|Document\(|\.save\(|pandoc .* -o ", re.I)


def action_kind(calls):
    """the agent's next action: 'doc' (reads / extracts a source document), 'build' (writes / edits / builds the
    deliverable), or None (anything else: listing files, reading the skills, checks)"""
    kinds = set()
    for c in calls:
        try: a = json.loads(c.get("arguments") or "{}")
        except ValueError: a = {}
        name, cmd = c.get("name"), a.get("command", "") if isinstance(a, dict) else ""
        path = (a.get("file_path") or a.get("path") or "") if isinstance(a, dict) else ""
        if name == "read" and DOCS.search(path): kinds.add("doc")
        elif name == "bash" and DOCS.search(cmd) and not re.search(r"\bls\b|\bfind\b", cmd) and not BUILD.search(cmd): kinds.add("doc")
        elif name in ("write", "edit") or (name == "bash" and BUILD.search(cmd) and not DOCS.search(cmd)): kinds.add("build")
    return "doc" if kinds == {"doc"} else "build" if kinds == {"build"} else None


def short(c):
    try: a = json.loads(c.get("arguments") or "{}")
    except ValueError: a = {}
    arg = (a.get("command") or a.get("file_path") or a.get("path") or a.get("pattern") or "") if isinstance(a, dict) else ""
    return f"{c.get('name')}({' '.join(str(arg).split())[:120]})"


def build_A(per_episode=4, seed=0):
    tasks = json.load(gzip.open(f"{S}/space/data/episodes.json.gz"))["tasks"]
    general = {tid: t for tid, t in tasks.items() if t["suite"] == "General"}
    rng = random.Random(seed); out = []
    for panel, root in (("base", f"{EP}/base_model/episodes/eval_base"), ("sage", f"{EP}/run_16x8_v12/episodes/eval_round_005")):
        for tid, t in general.items():
            p = f"{root}/{tid.replace('/', '__')}/rollout_0/transcript.jsonl"
            if not os.path.exists(p): continue
            rows = [json.loads(l) for l in open(p)]
            hist, cands = [], []
            for i, x in enumerate(rows):
                if x.get("role") == "assistant":
                    k = action_kind(x.get("tool_calls") or [])
                    if k: cands.append({"kind": k, "history": list(hist), "actual": [short(c) for c in x.get("tool_calls") or []]})
                    hist.append(("model", (x.get("text") or "")[:300], [short(c) for c in x.get("tool_calls") or []]))
                else:
                    hist.append(("result", x.get("tool_name"), (x.get("result_preview") or "")[:200]))
            docs = sorted({d for d in (t.get(panel, {}) or {}).get("docs_read", [])})
            for kind in ("doc", "build"):
                pool = [c for c in cands if c["kind"] == kind]
                for c in rng.sample(pool, min(per_episode // 2, len(pool))):
                    out.append({"task": tid, "panel": panel, "title": t.get("title"), "instructions": t.get("instructions"),
                                "documents": docs, "label": "call" if kind == "doc" else "no_call", **c})
    rng.shuffle(out)
    with open(f"{HERE}/data/replay_A.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"A: {len(out)} states from {len({r['task'] for r in out})} General tasks; call {sum(r['label'] == 'call' for r in out)}, "
          f"no_call {sum(r['label'] == 'no_call' for r in out)}")


def build_B():
    train = [r["task_id"] for r in csv.DictReader(open(f"{S}/art/splits/train.csv")) if r["task_id"].startswith("contracts/")]
    out = []
    for tid in train:
        p = f"{REPO}/tasks/{tid}/task.json"
        if not os.path.exists(p): continue
        t = json.load(open(p))
        docs = sorted(os.listdir(f"{REPO}/tasks/{tid}/documents")) if os.path.isdir(f"{REPO}/tasks/{tid}/documents") else []
        out.append({"task": tid, "title": t.get("title"), "instructions": t.get("instructions"), "documents": docs,
                    "criteria": [{"id": c["id"], "title": c.get("title", ""), "match": c.get("match_criteria", "")} for c in t.get("criteria", [])]})
    with open(f"{HERE}/data/replay_B.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"B: {len(out)} training-split Contracts tasks, {sum(len(r['criteria']) for r in out)} rubric criteria "
          f"(median {sorted(len(r['criteria']) for r in out)[len(out) // 2]} per task), median {sorted(len(r['documents']) for r in out)[len(out) // 2]} documents")


if __name__ == "__main__":
    build_A(); build_B()
