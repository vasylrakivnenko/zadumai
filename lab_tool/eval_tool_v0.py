"""Contract tool v0 evaluation (2026-10-05; the user: "gtg with v0! Report when done"). LAB training-split Contracts tasks
only (the 55 held-out tasks stay untouched); no LLM calls, no GPU. Against LAB's own `read` (pandoc / pandas parse of the
whole document, harvey-labs sandbox/parsers/parse_doc.py) and `grep` (raw file text):
 1. cost (318 tasks): tokens an agent loads by reading every document with `read`; the share of document text `grep` can
    see at all (plain-text files only); files the tool fails to parse
 2. redline fidelity (every training .docx with tracked changes): our insertions / deletions vs pandoc --track-changes=all
    (an independent parser); our current text vs what `read` shows
 3. compare: each redlined .docx split into its original and its current version, as two clean documents; compare() must
    recover the tracked changes (word-level precision / recall)
 4. find (the 40 test tasks of replay part B): fact-bearing rubric criteria = criteria whose exact facts (amounts,
    percentages, dates, durations, quoted terms, company names) occur in a passage of the documents (in at most 5
    passages); is that passage among what the tool returns (a) asked with the criterion's title, (b) running the query
    plans the callers wrote without seeing the rubric (Gemma 4 E4B, Muse Glimmer); tokens returned vs `read`
usage: dspy_venv/bin/python eval_tool_v0.py [1 2 3 4] -> data/tool_v0_eval.json (+ data/tool_v0_part*.jsonl)"""
import collections, glob, json, multiprocessing as mp, os, re, subprocess, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = f"{HERE}/harvey-labs"
OUT = f"{HERE}/data/tool_v0_eval.json"
_ENC = None


def ntok(s):
    global _ENC
    if _ENC is None:
        import tiktoken; _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(s, disallowed_special=()))


def pandoc(path, *extra):
    import pypandoc
    r = subprocess.run([pypandoc.get_pandoc_path(), path, "-t", "markdown", "--wrap=none", *extra], capture_output=True, text=True, timeout=120)
    return r.stdout


def lab_read(path):
    """what LAB's `read` returns for a file (pptx: python-pptx text instead of markitdown; 39 files in all of Contracts)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx": return pandoc(path)
    if ext == ".xlsx":
        import pandas as pd
        return "\n".join(f"=== Sheet: {n} ===\n" + df.to_string(index=False) for n, df in pd.read_excel(path, sheet_name=None).items())
    if ext == ".pptx":
        from pptx import Presentation
        return "\n".join(sh.text_frame.text for s in Presentation(path).slides for sh in s.shapes if sh.has_text_frame)
    return open(path, "rb").read().decode("utf-8", "replace")


def rows_B():
    return [json.loads(l) for l in open(f"{HERE}/data/replay_B.jsonl")]


def docs_dir(task):
    return f"{REPO}/tasks/{task}/documents"


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("\\", ""))


# ---------- 1. cost ----------
def part1_task(task):
    from contract_tool.tool import ContractTool
    t0 = time.time(); rec = {"task": task, "read_tokens": 0, "by_ext": collections.Counter(), "grep_visible": 0, "files": 0}
    for p in sorted(glob.glob(docs_dir(task) + "/*")):
        ext = os.path.splitext(p)[1].lower(); n = ntok(lab_read(p))
        rec["read_tokens"] += n; rec["by_ext"][ext] += n; rec["files"] += 1
        if ext not in (".docx", ".xlsx", ".pptx", ".pdf"): rec["grep_visible"] += n
    rec["read_s"] = round(time.time() - t0, 2); t0 = time.time()
    tool = ContractTool(docs_dir(task))
    rec["parse_s"] = round(time.time() - t0, 2); rec["parse_errors"] = tool.errors; rec["units"] = len(tool.units)
    rec["outline_tokens"] = ntok(tool.call("outline")); rec["redline_docs"] = sum(1 for d in tool.docs if tool.changes_items(d))
    return rec


# ---------- 2 + 3. redline fidelity, compare ----------
SPAN = re.compile(r"\[((?:\\.|[^\[\]\\])*)\]\{\.(insertion|deletion)\b[^}]*\}")


def part23_doc(path):
    from contract_tool.parse import parse, Unit
    from contract_tool.tool import diff_units
    us = parse(path)
    ours = {"ins": collections.Counter(), "del": collections.Counter()}
    for u in us:
        for s, t, _, _ in u.changes: ours[s].update(words(t))
    if not (ours["ins"] or ours["del"]): return None
    allm = pandoc(path, "--track-changes=all"); theirs = {"ins": collections.Counter(), "del": collections.Counter()}
    for m in SPAN.finditer(allm): theirs["ins" if m.group(2) == "insertion" else "del"].update(words(m.group(1)))
    read_view = pandoc(path)  # what LAB's read shows
    cur = collections.Counter(w for u in us for w in words(u.text)); rv = collections.Counter(words(read_view))
    rec = {"doc": os.path.relpath(path, REPO), "read_marks": len(SPAN.findall(read_view)),
           "pandoc_spans": sum(1 for _ in SPAN.finditer(allm)), "our_changes": sum(len(u.changes) for u in us)}
    for k in ("ins", "del"):
        rec[f"{k}_ours"], rec[f"{k}_pandoc"], rec[f"{k}_both"] = sum(ours[k].values()), sum(theirs[k].values()), sum((ours[k] & theirs[k]).values())
    rec["read_words"], rec["read_words_in_ours"] = sum(rv.values()), sum((rv & cur).values())
    rec["deleted_words_visible_in_read"] = sum((ours["del"] & (rv - cur)).values())  # deleted words that `read` still shows somewhere
    # 3. compare: original and current version as two clean documents
    A = [Unit("before", i, "p", u.before) for i, u in enumerate(us) if u.before]
    B = [Unit("after", i, "p", u.text) for i, u in enumerate(us) if u.text]
    t0 = time.time(); items, same = diff_units(A, B); rec["compare_s"] = round(time.time() - t0, 3)
    got = {"ins": collections.Counter(w for it in items for x in it[4] for w in words(x)),
           "del": collections.Counter(w for it in items for x in it[3] for w in words(x))}
    for k in ("ins", "del"):
        rec[f"cmp_{k}_got"], rec[f"cmp_{k}_true"], rec[f"cmp_{k}_both"] = sum(got[k].values()), sum(ours[k].values()), sum((got[k] & ours[k]).values())
    rec["changed_paras"] = sum(1 for u in us if u.changes)
    rec["cmp_items"] = len(items); rec["cmp_unchanged"] = same
    return rec


# ---------- 4. find ----------
NUMW = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|forty-five|fifty|sixty|ninety|hundred|[a-z\-]+)"
ANCH = [r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion))?", r"\b\d+(?:\.\d+)?\s?%",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        r"\b\d+\s+(?:calendar |business |banking )?(?:days|months|years|weeks)\b", r"\b\d+\.\d+\b", r"\b\d{1,3}(?:,\d{3})+\b",
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:MW|kW|MWh|kWh|GW|MMBtu|bps|basis points)(?:-?(?:dc|ac|DC|AC))?\b",
        r"[\"“‘']([^\"”’']{4,60})[\"”’']",
        r"\b(?:[A-Z][A-Za-z&\-]+,?\s+){1,4}(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|LLP|L\.P\.|LP|Corp\.?|Corporation|Company|plc|N\.A\.|GmbH|AG|S\.A\.|Bank|Trust)\b"]


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').lower()
    s = re.sub(rf"\b{NUMW}(?:[\- ]{NUMW})?\s*\((\d+)\)", r"\1", s)  # "thirty (30) days" -> "30 days"
    return re.sub(r"\s+", " ", s)


WEAK = {6, 7}  # bare decimals and comma numbers: count only next to another anchor


def anchors(c):
    """{anchor: strong?} from a criterion's title and pass condition; an anchor inside a longer one is dropped."""
    t = f"{c['title']}. {c['match']}"; out = {}
    for k, p in enumerate(ANCH):
        for m in re.finditer(p, t):
            a = norm(m.group(1) if m.groups() else m.group(0)).strip(" ,.;:")
            if len(a) >= 3 and a not in ("pass", "fail"): out[a] = out.get(a, False) or k not in WEAK
    return {a: v for a, v in out.items() if not any(a != b and a in b for b in out)}


def part4_task(task):
    import torch; torch.set_num_threads(1)
    from contract_tool.tool import ContractTool
    r = next(x for x in rows_B() if x["task"] == task)
    t0 = time.time(); tool = ContractTool(docs_dir(task)); tool.emb(); build_s = time.time() - t0
    U = [norm(u.text + " " + u.before) for u in tool.units]
    crit = []
    for c in r["criteria"]:
        an = anchors(c); df = {a: [i for i, u in enumerate(U) if a in u] for a in an}
        spec = {a: v for a, v in df.items() if 1 <= len(v) <= 5}
        if not spec: continue
        cnt = collections.Counter(i for v in spec.values() for i in v); best = max(cnt.values())
        ev = sorted(i for i, n in cnt.items() if n == best)
        if best < 2 and not any(an[a] for a in spec): continue  # a lone weak number is not evidence
        crit.append({"id": c["id"], "title": c["title"], "anchors": sorted(spec), "evidence": ev, "n_anchor": best,
                     "redline": any(tool.units[i].changes for i in ev)})
    res = {"task": task, "criteria": len(r["criteria"]), "fact_bearing": len(crit), "units": len(tool.units), "build_s": round(build_s, 1)}
    # (a) the criterion's own title as the query
    t0 = time.time()
    for c in crit:
        shown = {j for _, s in tool.find_units(c["title"]) for j in s}; c["title_hit"] = bool(shown & set(c["evidence"]))
    res["find_s"] = round((time.time() - t0) / max(1, len(crit)), 3)
    # (b) the callers' query plans (part B test outputs), run on the tool; and `read` of the documents they named
    for label in ("gemma-4-e4b_guided_test_gepa", "muse-bf16_guided_test_gepa", "muse-q4_guided_test_gepa", "gemma-4-e4b_guided_test_plain", "muse-bf16_guided_test_plain"):
        p = f"{HERE}/data/out_B_{label}.jsonl"
        if not os.path.exists(p): continue
        plan = next((json.loads(l) for l in open(p) if json.loads(l)["task"] == task), None)
        if plan is None or "error" in plan: continue
        shown, toks, named, errs, n = set(), 0, set(), 0, 0
        for call in plan["calls"]:
            if call["name"] != "contract_tool": continue
            try: a = json.loads(call["arguments"])
            except ValueError: errs += 1; continue
            n += 1; act, q, d = a.get("action", ""), a.get("query") or "", a.get("document") or ""
            txt = tool.call(act, q, d); toks += ntok(txt); errs += txt.startswith("Error")
            try: dn = tool.doc(d) if d else None
            except KeyError: dn = None
            if dn: named.add(dn)
            if act in ("find", "ask") and q: shown |= {j for _, s in tool.find_units(q, dn) for j in s}
        ev_docs = lambda c: {tool.units[i].doc for i in c["evidence"]}
        res[label] = {"calls": n, "errors": errs, "tool_tokens": toks, "reach": sum(bool(shown & set(c["evidence"])) for c in crit),
                      "named_docs": len(named), "read_named_reach": sum(bool(ev_docs(c) & named) for c in crit),
                      "read_named_tokens": sum(ntok(lab_read(tool.files[d])) for d in named)}
    res["crit"] = crit
    return res


def pool_map(f, xs, procs=8):
    with mp.get_context("fork").Pool(procs) as p: return list(p.imap_unordered(f, xs, chunksize=1))


if __name__ == "__main__":
    parts = sys.argv[1:] or ["1", "2", "3", "4"]
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rows = rows_B(); tasks = [r["task"] for r in rows]
    if "1" in parts:
        t0 = time.time(); R = pool_map(part1_task, tasks)
        with open(f"{HERE}/data/tool_v0_part1.jsonl", "w") as f:
            for r in R: f.write(json.dumps(r) + "\n")
        rt = np.array([r["read_tokens"] for r in R]); gv = np.array([r["grep_visible"] for r in R])
        by = collections.Counter()
        for r in R: by.update(r["by_ext"])
        out["1_cost"] = {"tasks": len(R), "files": sum(r["files"] for r in R), "read_tokens_median": int(np.median(rt)), "read_tokens_p90": int(np.percentile(rt, 90)),
                         "read_tokens_max": int(rt.max()), "tasks_over_100k": int((rt > 100_000).sum()), "tasks_over_200k": int((rt > 200_000).sum()),
                         "read_tokens_by_ext": {k: int(v) for k, v in by.most_common()}, "grep_visible_share": round(float(gv.sum() / rt.sum()), 4),
                         "parse_errors": {r["task"]: r["parse_errors"] for r in R if r["parse_errors"]}, "outline_tokens_median": int(np.median([r["outline_tokens"] for r in R])),
                         "tasks_with_redline_docs": sum(r["redline_docs"] > 0 for r in R), "parse_s_median": float(np.median([r["parse_s"] for r in R])), "wall_s": round(time.time() - t0)}
        print("1 cost:", json.dumps(out["1_cost"], indent=1), flush=True)
    if "2" in parts or "3" in parts:
        docs = sorted({p for t in tasks for p in glob.glob(docs_dir(t) + "/*.docx")})
        t0 = time.time(); R = [r for r in pool_map(part23_doc, docs) if r]
        with open(f"{HERE}/data/tool_v0_part23.jsonl", "w") as f:
            for r in R: f.write(json.dumps(r) + "\n")
        S = lambda k: sum(r[k] for r in R)
        out["2_redline"] = {"docx_scanned": len(docs), "docx_with_tracked_changes": len(R), "read_change_marks": S("read_marks"),
                            "changes_ours": S("our_changes"), "pandoc_spans": S("pandoc_spans"),
                            "ins_word_recall_vs_pandoc": round(S("ins_both") / max(1, S("ins_pandoc")), 4), "ins_word_precision": round(S("ins_both") / max(1, S("ins_ours")), 4),
                            "del_word_recall_vs_pandoc": round(S("del_both") / max(1, S("del_pandoc")), 4), "del_word_precision": round(S("del_both") / max(1, S("del_ours")), 4),
                            "deleted_words": S("del_ours"), "inserted_words": S("ins_ours"),
                            "read_words_covered_by_our_current_text": round(S("read_words_in_ours") / max(1, S("read_words")), 4)}
        out["3_compare"] = {"docs": len(R), "ins_precision": round(S("cmp_ins_both") / max(1, S("cmp_ins_got")), 4), "ins_recall": round(S("cmp_ins_both") / max(1, S("cmp_ins_true")), 4),
                            "del_precision": round(S("cmp_del_both") / max(1, S("cmp_del_got")), 4), "del_recall": round(S("cmp_del_both") / max(1, S("cmp_del_true")), 4),
                            "changed_paragraphs_true": S("changed_paras"), "items_reported": S("cmp_items"), "compare_s_median": float(np.median([r["compare_s"] for r in R])),
                            "wall_s": round(time.time() - t0)}
        print("2 redline:", json.dumps(out["2_redline"], indent=1), "\n3 compare:", json.dumps(out["3_compare"], indent=1), flush=True)
    if "4" in parts:
        test = [r["task"] for r in rows if zlib.crc32(r["task"].encode()) % 10 >= 7][:40]
        t0 = time.time(); R = pool_map(part4_task, test)
        with open(f"{HERE}/data/tool_v0_part4.jsonl", "w") as f:
            for r in R: f.write(json.dumps(r) + "\n")
        P1 = {json.loads(l)["task"]: json.loads(l) for l in open(f"{HERE}/data/tool_v0_part1.jsonl")} if os.path.exists(f"{HERE}/data/tool_v0_part1.jsonl") else {}
        fb = sum(r["fact_bearing"] for r in R); cr = [c for r in R for c in r["crit"]]
        o = {"tasks": len(R), "criteria": sum(r["criteria"] for r in R), "fact_bearing": fb, "fact_bearing_in_redlined_text": sum(c["redline"] for c in cr),
             "title_query_reach": round(sum(c["title_hit"] for c in cr) / max(1, fb), 4), "read_all_tokens_per_task": int(np.mean([P1[r["task"]]["read_tokens"] for r in R if r["task"] in P1] or [0])),
             "find_s_per_query": float(np.median([r["find_s"] for r in R])), "build_s_median": float(np.median([r["build_s"] for r in R])), "wall_s": round(time.time() - t0)}
        for label in ("gemma-4-e4b_guided_test_gepa", "muse-bf16_guided_test_gepa", "muse-q4_guided_test_gepa", "gemma-4-e4b_guided_test_plain", "muse-bf16_guided_test_plain"):
            L = [r for r in R if label in r]
            if not L: continue
            f = sum(r["fact_bearing"] for r in L)
            o[label] = {"tasks": len(L), "calls_per_task": round(np.mean([r[label]["calls"] for r in L]), 1), "errors": sum(r[label]["errors"] for r in L),
                        "tool_reach": round(sum(r[label]["reach"] for r in L) / max(1, f), 4), "tool_tokens_per_task": int(np.mean([r[label]["tool_tokens"] for r in L])),
                        "read_named_reach": round(sum(r[label]["read_named_reach"] for r in L) / max(1, f), 4),
                        "read_named_tokens_per_task": int(np.mean([r[label]["read_named_tokens"] for r in L]))}
        out["4_find"] = o; print("4 find:", json.dumps(o, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
