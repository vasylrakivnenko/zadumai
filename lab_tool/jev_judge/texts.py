"""Jev judge, step 1 (2026-10-06): the labeled items. For every graded LAB episode we hold, every criterion with its
verdicts and exactly the text LAB's judge reads for it (the criterion's deliverable files, pandoc markdown, tracked
changes shown when the criterion asks; LAB's own scoring helpers, so no drift).
  rec   Ivo Sage's records: 55 held-out Contracts tasks x {base, ivo} x 3 judge passes
  kimi  our pilot runs graded by Kimi K3 (one pass)
Texts are stored once by content hash. Episodes with no deliverable file are flagged (trivial fails).
usage (harvey venv, pandoc on PATH): harvey-labs/.venv/bin/python jev_judge/texts.py -> data/jev_judge/items.jsonl"""
import glob, hashlib, json, os, sys
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = f"{HERE}/../harvey-labs"; sys.path.insert(0, LAB)
os.environ["PATH"] = "/root/zadumai_nli_proto/dspy_venv/lib/python3.12/site-packages/pypandoc/files:" + os.environ["PATH"]
from evaluation.scoring import DocxTrackChanges, _load_all_output, _match_deliverables, _read_file_as_text
OUT = f"{HERE}/../data/jev_judge"; os.makedirs(f"{OUT}/texts", exist_ok=True)
REC = {"base": "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace/art/un/train/deepseek-v4-flash/base_model/episodes/eval_base",
       "ivo": "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace/art/un/train/deepseek-v4-flash/run_16x8_v12/episodes/eval_round_005"}


def texts_for(task_id, output_dir):
    """{criterion id: text} as LAB's scorer builds it (score_rubric), without its LLM file-matching fallback."""
    config = json.load(open(f"{LAB}/tasks/{task_id}/task.json")); criteria = config["criteria"]
    out_dir = Path(output_dir); filenames = []
    for c in criteria:
        for d in c.get("deliverables", []):
            if d not in filenames: filenames.append(d)
    resolved = {}
    if filenames and out_dir.exists():
        actual = [f.name for f in out_dir.rglob("*") if f.is_file()]
        resolved = _match_deliverables({f: f for f in filenames}, actual, output_dir=None)
    full = None; res = {}
    for c in criteria:
        ds = c.get("deliverables", [])
        if ds and resolved:
            secs = []
            for name in ds:
                fp = out_dir / resolved[name]
                if not fp.exists(): secs.append(f"## Agent Output: {name}\n(File not found: {resolved[name]})"); continue
                tc = DocxTrackChanges.ALL if c.get("evaluation_options", {}).get("include_docx_redlines", False) else DocxTrackChanges.ACCEPT
                secs.append(f"## Agent Output: {name}\n{_read_file_as_text(fp, track_changes=tc)}")
            res[c["id"]] = "\n\n".join(secs) if secs else "(No agent output found)"
        else:
            if full is None: full = _load_all_output(out_dir) if out_dir.exists() else ""
            res[c["id"]] = full
    return criteria, res


def store(text):
    h = hashlib.sha1(text.encode()).hexdigest()[:16]; p = f"{OUT}/texts/{h}.txt"
    if not os.path.exists(p): open(p, "w").write(text)
    return h


def has_files(output_dir):
    return any(os.path.isfile(f) for f in glob.glob(f"{output_dir}/**/*", recursive=True))  # like LAB's _load_all_output (work/ included)


if __name__ == "__main__":
    import contextlib, io
    items = []
    for arm, root in REC.items():  # Ivo Sage's records
        for d in sorted(glob.glob(f"{root}/contracts__*/rollout_0")):
            task = os.path.basename(os.path.dirname(d)).replace("__", "/")
            J = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/judge_*.json"))]
            if len(J) < 3: continue
            with contextlib.redirect_stdout(io.StringIO()): criteria, T = texts_for(task, f"{d}/output")
            V = [{v["criterion_id"]: v["verdict"] for v in j["verdicts"]} for j in J]
            for c in criteria:
                items.append({"src": "rec", "episode": f"{arm}:{task}", "task": task, "cid": c["id"], "title": c["title"], "match": c["match_criteria"],
                              "text": store(T[c["id"]]), "verdicts": [v.get(c["id"]) for v in V], "empty": not has_files(f"{d}/output")})
            print(arm, task, len(criteria), flush=True)
    for d in sorted(glob.glob(f"{LAB}/results/pilot-*")):  # our runs, Kimi K3 verdicts
        if not os.path.exists(f"{d}/scores.json") or not os.path.exists(f"{d}/config.json"): continue
        S = json.load(open(f"{d}/scores.json")); task = S["task"]
        with contextlib.redirect_stdout(io.StringIO()): criteria, T = texts_for(task, f"{d}/output")
        V = {c["id"]: c["verdict"] for c in S["criteria_results"]}
        for c in criteria:
            items.append({"src": "kimi", "episode": os.path.basename(d), "task": task, "cid": c["id"], "title": c["title"], "match": c["match_criteria"],
                          "text": store(T[c["id"]]), "verdicts": [V.get(c["id"])], "empty": not has_files(f"{d}/output")})
        print("kimi", os.path.basename(d), len(criteria), flush=True)
    with open(f"{OUT}/items.jsonl", "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    print(len(items), "items;", len({i['text'] for i in items}), "distinct texts;", sum(i["empty"] for i in items), "in empty episodes")
