"""Whole-document evaluation of the compiled pipeline vs today's path (2026-10-03).
Sets:
  nda_train  ContractNLI train NDAs (100, random.Random(5)) x cnli/baseline.py's 17 user-style questions: tune here, read errors here
  nda_dev    ContractNLI dev NDAs (61) x the 17: scored only, errors not read
  wc1_open   wc1.json open half (user-style questions over whole MCC contracts; build_wc.py)
  wc1_sealed wc1.json sealed half: only with --final (run once, logged in sealed_runs.log)
Gold: yes / no / not_stated; "yes"/"no" right only on the same gold; any answer on not_stated is wrong.
Today = Pre-Tier 0 on the whole document, then the network on documents it reads (long ones: off).
usage: eval_docs.py SET [--save NAME] [--n N] [--k K] [--except E] [--final]"""
import argparse, collections, json, math, os, random, sys, time, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline as P  # noqa: E402

sys.path.insert(0, f"{HERE}/../cnli")
from baseline import QUESTIONS, POLARITY, DEFAULT  # noqa: E402


def wilson_lo(k, n, z=1.96):
    if not n: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def nda(split, n=None, seed=5):
    z = zipfile.ZipFile(f"{HERE}/../data/ext/contractnli.zip"); d = json.loads(z.read(f"contract-nli/{split}.json"))
    lab = d["labels"]; docs = d["documents"]
    if n: random.Random(seed).shuffle(docs); docs = docs[:n]
    rows = []
    for doc in docs:
        ann = doc["annotation_sets"][0]["annotations"]
        for k, q in QUESTIONS.items():
            choice = ann[k]["choice"]
            g = "not_stated" if choice == "NotMentioned" else POLARITY.get(k, DEFAULT)[choice]
            ev = [doc["text"][doc["spans"][i][0]:doc["spans"][i][1]] for i in ann[k]["spans"]]
            rows.append({"id": f"{doc['id']}/{k}", "qid": k, "question": q, "doc": doc["text"], "gold": g, "ev": ev})
    return rows


def cuad_dev(n=40, seed=3):
    """CUAD tuning contracts of cuadc's dev split (never read there; CUAD test stays sealed) x the LegalBench CUAD
    questions (as extensive/v4/docs_pt0.py): gold yes when CUAD annotated that category, else no (LegalBench's labels)."""
    sys.path.insert(0, f"{HERE}/../cuadc")
    from data import split
    qs = json.load(open("/root/zadumai_nli_proto/extensive/questions.json"))
    cat_q = {t[5:].replace("_", " ").replace("-", " ").lower(): q for t, q in qs.items() if t.startswith("cuad_")}
    docs = sorted((d for d in split() if d["dev"]), key=lambda d: d["id"]); random.Random(seed).shuffle(docs)
    rows = []
    for d in docs[:n]:
        for cat, spans in d["gold"].items():
            k = cat.lower().replace("/", " ").replace("-", " ")
            if k in cat_q: rows.append({"id": f"{d['id'][:40]}/{k}", "question": cat_q[k], "doc": d["text"], "gold": "yes" if spans else "no", "qid": k})
    return rows


def wc1(half):
    d = json.load(open(f"{HERE}/wc1.json"))
    return [{**r, "doc": d["docs"][r["doc"]], "doc_key": r["doc"]} for r in d["rows"] if r["half"] == half]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--save"); ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int, default=P.K_LOCATE); ap.add_argument("--except", dest="exc", type=int, default=P.K_EXCEPT)
    ap.add_argument("--final", action="store_true"); ap.add_argument("--t-located", type=float, default=P.T_LOCATED)
    ap.add_argument("--t-no-located", type=float); a = ap.parse_args()
    if a.set == "wc1_sealed":
        assert a.final, "wc1_sealed is sealed: run once, at the end, with --final"
        with open(f"{HERE}/sealed_runs.log", "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} wc1_sealed {a.save}\n")
    rows = {"nda_train": lambda: nda("train", 100), "nda_dev": lambda: nda("dev"), "wc1_open": lambda: wc1("open"),
            "wc1_sealed": lambda: wc1("sealed"), "cuad_dev": cuad_dev}[a.set]()
    if a.n: rows = rows[:a.n]
    pipe = P.Pipeline(k_locate=a.k, k_except=a.exc, t_located=a.t_located, t_no_located=a.t_no_located)
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        base = P.pt0_check(r["question"], r["doc"])
        b = base.answer if base.fired else None
        if not base.fired and len(r["doc"]) < P.LONG_CHARS:
            nr = pipe.net.answer(r["question"], r["doc"]); b = nr.answer if nr.fired else None
        res = pipe.answer(r["question"], r["doc"])
        out.append({"id": r["id"], "question": r["question"], "gold": r["gold"], "today": b, **res,
                    "qid": r.get("qid"), "shape": r.get("shape")})
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    report(out)
    if a.save:
        os.makedirs(f"{HERE}/runs", exist_ok=True); json.dump(out, open(f"{HERE}/runs/{a.set}_{a.save}.json", "w"), indent=0)


def report(out):
    n = len(out)
    for name, key in (("today", "today"), ("pipeline", "answer")):
        ans = [r for r in out if r[key]]; ok = sum(r[key] == r["gold"] for r in ans)
        print(f"{name:9s} answered {len(ans)}/{n} = {len(ans) / n:.1%}, right {ok}/{len(ans)} = {ok / max(1, len(ans)):.1%} "
              f"[lo {wilson_lo(ok, len(ans)):.3f}]")
    by = collections.defaultdict(lambda: [0, 0])
    for r in out:
        if r["answer"]: by[r["path"]][0] += 1; by[r["path"]][1] += r["answer"] == r["gold"]
    print("  by path:", {k: f"{v[1]}/{v[0]}" for k, v in by.items()})
    g = collections.Counter(r["gold"] for r in out); print("  gold:", dict(g))


if __name__ == "__main__":
    main()
