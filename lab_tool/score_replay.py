"""Scores the replay runs (run_replay.py).
  A  call states: share where the model called contract_tool (it was about to read a source document);
     no_call states: share where it did NOT call contract_tool (it was about to write / build); balanced = their mean.
  B  per task: contract_tool calls made, actions used, and rubric coverage = the share of criteria (title + pass
     condition) with at least one query (query + document) at bge-small cosine >= 0.60 / 0.70; "shuffled" = the same
     model's queries scored against ANOTHER task's rubric (the floor that generic legal wording reaches by itself).
usage: python score_replay.py"""
import collections, glob, json, os, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def part_A():
    print("A. WHEN to call (120 states each: 60 'call', 60 'no_call')")
    for f in sorted(glob.glob(f"{HERE}/data/out_A_*.jsonl")):
        rows = list({json.loads(l)["key"]: json.loads(l) for l in open(f)}.values()); ok = [r for r in rows if "error" not in r]  # dedupe
        uses = lambda r: any(c["name"] == "contract_tool" for c in r["calls"])
        c = [r for r in ok if r["label"] == "call"]; n = [r for r in ok if r["label"] == "no_call"]
        rc = np.mean([uses(r) for r in c]) if c else float("nan"); rn = np.mean([not uses(r) for r in n]) if n else float("nan")
        first = collections.Counter((r["calls"][0]["name"] if r["calls"] else "(no call)") for r in c)
        acts = collections.Counter(json.loads(x["arguments"]).get("action") for r in ok for x in r["calls"] if x["name"] == "contract_tool")
        name = os.path.basename(f)[6:-6]
        print(f"  {name:34s} uses tool when it should {rc:5.1%} | holds off when writing {rn:5.1%} | balanced {np.mean([rc, rn]):5.1%} | "
              f"errors {len(rows) - len(ok)} | first call on 'call' states {dict(first.most_common(3))} | actions {dict(acts)}")


def part_B():
    files = sorted(glob.glob(f"{HERE}/data/out_B_*.jsonl"))
    if not files: return
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    B = {json.loads(l)["task"]: json.loads(l) for l in open(f"{HERE}/data/replay_B.jsonl")}
    print("B. WHAT to ask (20 training-split Contracts tasks): rubric coverage of the planned queries")
    for f in files:
        rows = [r for r in {json.loads(l)["key"]: json.loads(l) for l in open(f)}.values() if "error" not in r]  # dedupe
        plans = {}
        for r in rows:
            qs = []
            for c in r["calls"]:
                if c["name"] != "contract_tool": continue
                try: a = json.loads(c["arguments"])
                except ValueError: continue
                qs.append(" ".join(str(a.get(k) or "") for k in ("action", "query", "document")).strip())
            plans[r["task"]] = qs
        tasks = [t for t in plans if plans[t]]
        rng = random.Random(0); other = {t: rng.choice([u for u in tasks if u != t]) for t in tasks}
        cov = {0.6: [], 0.7: []}; shuf = {0.6: [], 0.7: []}
        for t in tasks:
            crit = [f"{c['title']}. {c['match']}"[:400] for c in B[t]["criteria"]]
            C = emb.encode(crit, normalize_embeddings=True)
            for which, qs in (("own", plans[t]), ("shuffled", plans[other[t]])):
                Qv = emb.encode(qs, normalize_embeddings=True); best = (C @ Qv.T).max(1)
                for th in (0.6, 0.7): (cov if which == "own" else shuf)[th].append(float((best >= th).mean()))
        acts = collections.Counter(json.loads(c["arguments"]).get("action") for r in rows for c in r["calls"] if c["name"] == "contract_tool")
        n_calls = [len(plans[t]) for t in plans]
        name = os.path.basename(f)[6:-6]
        print(f"  {name:34s} tasks with a plan {len(tasks)}/{len(rows)} | calls per task median {sorted(n_calls)[len(n_calls) // 2] if n_calls else 0} | "
              f"coverage @0.60 {np.mean(cov[0.6]):5.1%} (shuffled {np.mean(shuf[0.6]):5.1%}) | @0.70 {np.mean(cov[0.7]):5.1%} "
              f"(shuffled {np.mean(shuf[0.7]):5.1%}) | actions {dict(acts)}")


if __name__ == "__main__":
    part_A(); part_B()
