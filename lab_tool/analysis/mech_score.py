"""Can LAB be scored mechanically? On the recorded held-out Contracts episodes (base V4 Flash + Ivo Sage), compare a
mechanical check (the criterion's exact facts: amounts, percentages, dates, durations, quoted terms, names, found in the
deliverables' text, tracked changes included) with the recorded LLM judge verdicts (judge_1.json)."""
import glob, json, os, re, subprocess, sys, collections
sys.path.insert(0, "/root/zadumai_nli_proto/lab_tool")
import eval_tool_v0 as E
import pypandoc
REC = {"base": "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace/art/un/train/deepseek-v4-flash/base_model/episodes/eval_base",
       "ivo": "/tmp/claude-0/-root/ed0c4a54-2a7f-4b5e-9d25-af162b3c051b/scratchpad/hfspace/art/un/train/deepseek-v4-flash/run_16x8_v12/episodes/eval_round_005"}

def text_of(out):
    parts = []
    for f in glob.glob(f"{out}/**/*", recursive=True):
        if "/work/" in f or os.path.isdir(f): continue
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext == ".docx": parts.append(subprocess.run([pypandoc.get_pandoc_path(), f, "-t", "plain", "--track-changes=all"], capture_output=True, text=True, timeout=60).stdout)
            elif ext in (".md", ".txt"): parts.append(open(f, errors="replace").read())
            elif ext == ".xlsx": parts.append(E.lab_read(f))
        except Exception: pass
    return E.norm(" ".join(parts))

agree = collections.Counter(); n_crit = 0
for arm, root in REC.items():
    for d in glob.glob(f"{root}/contracts__*/rollout_0"):
        if not os.path.exists(f"{d}/judge_1.json"): continue
        J = json.load(open(f"{d}/judge_1.json")); txt = text_of(f"{d}/output")
        task = json.load(open(f"/root/zadumai_nli_proto/lab_tool/harvey-labs/tasks/{J['task_id']}/task.json"))
        crit = {c["id"]: c for c in task["criteria"]}
        for v in J["verdicts"]:
            n_crit += 1; c = crit.get(v["criterion_id"])
            if not c or not txt: continue
            an = {a: s for a, s in E.anchors({"title": c["title"], "match": c["match_criteria"]}).items() if s}
            if not an: continue
            mech = all(a in txt for a in an)
            agree[(mech, v["verdict"] == "pass")] += 1
tot = sum(agree.values())
print(f"criteria judged: {n_crit}; with strong anchors and a deliverable: {tot} ({tot / n_crit:.0%})")
print(f"agreement with the LLM judge: {(agree[(True, True)] + agree[(False, False)]) / tot:.1%}  "
      f"| mech pass & judge pass {agree[(True, True)]}, mech pass & judge fail {agree[(True, False)]}, mech fail & judge pass {agree[(False, True)]}, both fail {agree[(False, False)]}")
