"""The fixed pipeline (2026-10-06; plan: PLAN.md): review the other side's draft (redline + issues memo) or respond to
their markup of our draft (counter-redline + cover note), with small, focused model calls and deterministic documents.
Reads only the task's instructions and documents (never the rubric). Writes LAB-style results so LAB's graders apply:
harvey-labs/results/<run_id>/{output/, config.json, metrics.json, trace.json}.
usage: dspy_venv/bin/python pipeline/run.py MODEL TASK [RUN_ID]      MODEL: luna | gemma26"""
import collections, concurrent.futures as cf, datetime, json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); LT = os.path.dirname(HERE); LAB = f"{LT}/harvey-labs"
sys.path.insert(0, HERE); sys.path.insert(0, LT)
from contract_tool.tool import ContractTool, diff_units
from docx_redline import Redline
from llm import LLM
from search import Search

PARALLEL = {"luna": 16, "gemma26": 4}
ROLES = ("their_draft", "their_markup", "our_draft", "standard_form", "playbook", "instructions", "correspondence", "reference")


def load_task(task):
    """the instructions and the documents folder; the rest of task.json (the rubric) is dropped unread."""
    cfg = json.load(open(f"{LAB}/tasks/{task}/task.json")); instructions = cfg["instructions"]
    docs = f"{LAB}/tasks/{task}/" + (cfg.get("docs_dir") or "documents"); del cfg
    out = instructions.split("### Output:")[-1].split() if "### Output:" in instructions else []
    return instructions, os.path.realpath(docs), [o for o in out if "." in o]


def clip(s, n):
    return s if len(s) <= n else s[:n] + " …"


class Pipeline:
    def __init__(self, model, task, run_id):
        self.model, self.task, self.run_id = model, task, run_id
        self.out = f"{LAB}/results/{run_id}/output"; os.makedirs(self.out, exist_ok=True)
        self.llm = LLM(model, f"pipe:{run_id}"); self.trace = {"task": task, "model": model, "steps": {}}
        self.instructions, self.docs_dir, self.deliverables = load_task(task)
        self.tool = ContractTool(self.docs_dir); self.tool.emb(); self.search = Search(self.tool)

    # ---------- helpers ----------
    def units(self, doc):
        return [self.tool.units[i] for i in self.tool.docs[doc]]

    def doc_text(self, doc, limit):
        return clip("\n".join(f"¶{u.i}: {u.marked if u.changes else u.text}" for u in self.units(doc)), limit)

    def evidence(self, query, docs, k=6, limit=6000):
        if not docs: return "(none)"
        hits = self.search.search(query, docs=docs, k=k); out = []
        for i in hits:
            u = self.tool.units[i]; out.append(f"[{u.where()}] {u.text}")
            nxt = self.tool.units[i + 1] if i + 1 < len(self.tool.units) and self.tool.units[i + 1].doc == u.doc else None
            if nxt is not None and (u.heading or len(u.text.split()) < 8): out.append(f"[{nxt.where()}] {nxt.text}")
        return clip("\n".join(out), limit)

    def pmap(self, f, xs):
        with cf.ThreadPoolExecutor(PARALLEL[self.model]) as ex: return list(ex.map(f, xs))

    # ---------- 1. roles ----------
    def roles(self):
        files = []
        for d, idx in self.tool.docs.items():
            us = self.units(d); ch = sum(len(u.changes) for u in us)
            heads = [u.text for u in us if u.heading][:12] or [u.text for u in us[:6]]
            files.append(f"- {d} ({len(us)} paragraphs/rows{', ' + str(ch) + ' tracked changes' if ch else ''}): " + clip(" | ".join(heads), 600))
        p = f"""You are a senior contracts lawyer opening a new matter. Below are the assignment and the files received.

ASSIGNMENT:
{self.instructions}

FILES:
{chr(10).join(files)}

Give each file one role: their_draft (an agreement drafted by the other side, for us to review), their_markup (our draft
as marked up by the other side, with tracked changes), our_draft (our own draft or form as we sent it), standard_form (our
standard template / precedent), playbook (internal negotiation playbook or policy: positions, fallbacks, approvals),
instructions (an email or note telling us what to do), correspondence (other emails, letters, transmittals, notes),
reference (other supporting material: term sheets, schedules, orders, executed agreements, data).
Also name the client we act for, the other side, and the workflow: review_draft (review the other side's draft:
redline + issues memo) or respond_to_markup (answer the other side's markup of our draft: counter-redline + cover note).
Return JSON: {{"client": "...", "counterparty": "...", "our_role": "e.g. licensee / borrower / buyer", "workflow":
"review_draft" | "respond_to_markup", "files": {{"<file name>": "<role>"}}, "primary": "<the file our redline is made on>",
"our_original": "<our draft as sent, or null>"}}"""
        r = self.llm.json(p, default={}) or {}
        files = {d: (r.get("files") or {}).get(d, "reference") for d in self.tool.docs}
        files = {d: (v if v in ROLES else "reference") for d, v in files.items()}
        marked = [d for d in self.tool.docs if d.endswith(".docx") and self.tool.changes_items(d)]
        wf = r.get("workflow") if r.get("workflow") in ("review_draft", "respond_to_markup") else ("respond_to_markup" if marked else "review_draft")
        prim = r.get("primary") if r.get("primary") in self.tool.docs else None
        if wf == "respond_to_markup" and (prim is None or prim not in marked) and marked: prim = marked[0]
        if prim is None: prim = next((d for d, v in files.items() if v in ("their_draft", "their_markup")), next(iter(self.tool.docs)))
        orig = r.get("our_original") if r.get("our_original") in self.tool.docs and r.get("our_original") != prim else None
        self.r = {"client": r.get("client") or "our client", "counterparty": r.get("counterparty") or "the other side", "our_role": r.get("our_role") or "",
                  "workflow": wf, "files": files, "primary": prim, "our_original": orig}
        self.trace["steps"]["roles"] = self.r

    # ---------- 2. brief ----------
    def brief(self):
        mail = [d for d, v in self.r["files"].items() if v in ("instructions", "correspondence")]
        refs = [d for d, v in self.r["files"].items() if v == "reference"]
        stds = [d for d, v in self.r["files"].items() if v in ("playbook", "standard_form")]
        body = "\n\n".join(f"=== {d} ===\n" + self.doc_text(d, 12000) for d in mail)
        ref = "\n".join(f"- {d}: " + clip(" | ".join(u.text for u in self.units(d)[:5]), 400) for d in refs)
        p = f"""You act for {self.r['client']} ({self.r['our_role']}) against {self.r['counterparty']}. Write the working brief for this
assignment from the assignment text and the correspondence below.

ASSIGNMENT:
{self.instructions}

CORRESPONDENCE AND NOTES:
{body or '(none)'}

OUR PLAYBOOK / STANDARD FORM (opening lines: title, version, date):
{chr(10).join(f"- {d}: " + " | ".join(u.text for u in self.units(d)[:8]) for d in stds) or '(none)'}

OTHER MATERIALS (titles only):
{ref or '(none)'}

Return JSON: {{"deal": "one paragraph: the transaction and where the negotiation stands", "objectives": ["..."],
"checklist": [{{"id": "I1", "item": "one concrete instruction, client priority, position, red line, deadline or required
content of the deliverables, with its numbers, names and dates", "source": "file name"}}],
"memo_to": "...", "memo_from": "...", "memo_cc": "...", "note_to": "who a cover note to the other side goes to", "note_from": "...",
"note_cc": "reply-all: everyone on the incoming email's To and Cc lines except the note's addressee and ourselves, with names and organizations", "benchmark": "the playbook / standard form / policy our review measures against:
exact title, version and date", "dates": ["key dates and deadlines"]}}
Put every instruction, priority and deadline in the checklist, one per entry."""
        self.b = self.llm.json(p, default={}) or {}
        self.b.setdefault("checklist", [])
        self.trace["steps"]["brief"] = self.b

    def brief_text(self):
        ck = "\n".join(f"{c.get('id')}: {c.get('item')}" for c in self.b.get("checklist", []))
        return f"Client: {self.r['client']} ({self.r['our_role']}); other side: {self.r['counterparty']}.\nDeal: {self.b.get('deal', '')}\nInstructions and priorities:\n{ck}"

    def support_docs(self):
        return [d for d, v in self.r["files"].items() if v in ("playbook", "standard_form", "reference", "instructions", "correspondence") and d != self.r["primary"]]

    # ---------- 3-5. work items and decisions ----------
    def clause_groups(self, doc):
        groups, cur = [], []
        words = lambda g: sum(len(u.text.split()) for u in g)
        for u in self.units(doc):
            lab = u.label or ""
            start = u.heading or bool(re.fullmatch(r"\d+(\.\d+)?|(Section|SECTION|Article|ARTICLE|Clause|CLAUSE) \S+", lab))
            if start and cur and words(cur) > 0 and not (len(cur) == 1 and cur[0].heading):
                groups.append(cur); cur = []
            cur.append(u)
            if words(cur) > 700: groups.append(cur); cur = []
        if cur: groups.append(cur)
        return groups

    def review_items(self):
        prim = self.r["primary"]; sup = self.support_docs(); bt = self.brief_text()
        groups = [g for g in self.clause_groups(prim) if sum(len(u.text.split()) for u in g) >= 4]
        def decide(g):
            text = "\n".join(f"¶{u.i}: {u.text}" for u in g)
            ev = self.evidence(" ".join(u.text for u in g)[:1500], sup)
            p = f"""You are reviewing an agreement drafted by the other side, for {self.r['client']} ({self.r['our_role']}).
{bt}

CLAUSE OF THEIR DRAFT ({prim}):
{text}

OUR PLAYBOOK, STANDARD FORM AND RELEVANT MATERIALS (search results):
{ev}

Decide whether this clause is acceptable for our client under the playbook, the standard form and the client's
instructions. Recitals, headings and boilerplate are acceptable unless they misstate facts or shift rights. Accept acceptable language as it is; change only what the playbook or the instructions require, using our
preferred language (or a fallback the playbook allows). Keep clause numbers. Return JSON:
{{"topic": "short name", "deviation": true | false, "risk": "High" | "Medium" | "Low" | "None",
"issue": "what is off-market, missing or contrary to the instructions (cite the draft's terms, numbers)",
"playbook_position": "our standard position (cite the playbook section)", "fallback": "acceptable fallback, if any",
"recommendation": "what we ask for", "exposure": "what the client risks if this stays",
"edits": [{{"para": <¶ number>, "new_text": "the full revised text of that paragraph"}}],
"inserts": [{{"after": <¶ number>, "text": "a new paragraph to add"}}],
"comment": "margin comment for the redline: what we changed and why (playbook reference)", "checklist": ["I1"]}}
Use empty lists when no change is needed."""
            d = self.llm.json(p, default=None)
            return {"paras": [u.i for u in g], "label": g[0].label or g[0].text[:60], "decision": d}
        items = self.pmap(decide, groups)
        self.trace["steps"]["clause_items"] = len(items)
        return items

    def gap_items(self):
        """provisions the playbook requires that the draft lacks."""
        pb = [d for d, v in self.r["files"].items() if v in ("playbook", "standard_form")]
        if not pb: return []
        text = "\n\n".join(f"=== {d} ===\n" + self.doc_text(d, 120000) for d in pb)  # whole playbooks: a 40k cut lost later sections
        p = f"""From our playbook / standard form below, list the provisions our client requires or strongly prefers in this
agreement (each one a protection the other side's draft must contain), at most 40.
{text}
Return JSON: {{"provisions": [{{"name": "...", "requirement": "what it must say (numbers, terms)"}}]}}"""
        prov = (self.llm.json(p, default={}) or {}).get("provisions", [])[:40]
        prim = self.r["primary"]; bt = self.brief_text()
        def check(pv):
            hits = self.search.search(f"{pv.get('name')}: {pv.get('requirement')}", docs=[prim], k=4)
            found = "\n".join(f"¶{i - self.tool.docs[prim][0]}: {self.tool.units[i].text}" for i in hits)
            p = f"""For {self.r['client']} ({self.r['our_role']}). {bt}
Required provision: {pv.get('name')}: {pv.get('requirement')}
The passages of the other side's draft ({prim}) most likely to contain it:
{found}
Does the draft contain this provision adequately? If not, draft the clause to add and say where.
Return JSON: {{"present": true | false, "risk": "High" | "Medium" | "Low", "issue": "...", "insert_after": <¶ number or null>,
"text": "the clause to insert (when missing)", "comment": "margin comment: why we add it (playbook reference)"}}"""
            d = self.llm.json(p, default=None) or {}
            return {"provision": pv, "decision": d} if d and d.get("present") is False and d.get("text") else None
        gaps = [g for g in self.pmap(check, prov) if g]
        self.trace["steps"]["gaps"] = {"provisions": len(prov), "missing": len(gaps)}
        return gaps

    def markup_items(self):
        prim = self.r["primary"]; sup = self.support_docs(); bt = self.brief_text()
        changed = [u for u in self.units(prim) if u.changes]
        silent = []
        if self.r["our_original"]:  # edits made without tracking: our draft vs the markup's original text
            from contract_tool.parse import Unit
            A = [Unit("ours", u.i, "p", u.text) for u in self.units(self.r["our_original"]) if u.text]
            B = [Unit("theirs", u.i, "p", u.before) for u in self.units(prim) if u.before and not u.changes]
            for kind, u, m, de, ins in diff_units(A, B)[0]:
                if kind == "changed" and len(de) + len(ins) >= 2: silent.append((u.i, m))
        allu = self.units(prim)
        def decide(x):
            u, silent_mark = x
            ctx = "\n".join(f"¶{v.i}: {v.text}" for v in allu[max(0, u.i - 2):u.i] + allu[u.i + 1:u.i + 3])
            ev = self.evidence((u.before + " " + u.text)[:1500], sup)
            change = (f"their change (untracked! they edited our text without marking it): {silent_mark}" if silent_mark
                      else f"our text: {u.before or '(none: they added this paragraph)'}\ntheir text: {u.text or '(they deleted this paragraph)'}\nmarked: {u.marked}")
            p = f"""The other side marked up our draft. You act for {self.r['client']} ({self.r['our_role']}).
{bt}

THE CHANGE (¶{u.i} of {prim}, {u.where()}):
{change}

NEARBY TEXT:
{ctx}

OUR PLAYBOOK AND RELEVANT MATERIALS (search results):
{ev}

Decide: accept their change, reject it (restore our text), or counter (compromise language the playbook allows), following
the playbook and the client's instructions. Accept only a change that the playbook or the instructions allow, or a
clarification that does not weaken our client's position (scope, amounts, triggers, remedies, approvals). Where the
playbook or the instructions allow a compromise (a fallback, a threshold, a carve-out, a cap), counter with it rather than
reject flatly. Return JSON:
{{"topic": "short name", "decision": "accept" | "reject" | "counter", "risk": "High" | "Medium" | "Low",
"text": "the full paragraph as we want it (accept: their text; reject: our text; counter: new language)",
"comment": "margin comment to the other side: our position and the reason", "rationale": "internal reasoning and playbook
reference", "escalate": true | false, "checklist": ["I1"]}}"""
            d = self.llm.json(p, default=None)
            return {"para": u.i, "label": u.label or u.where(), "silent": bool(silent_mark), "theirs": u.text, "ours": u.before, "decision": d}
        xs = [(u, None) for u in changed] + [(allu[i], m) for i, m in silent if i < len(allu)]
        items = self.pmap(decide, xs)
        self.trace["steps"]["markup_items"] = {"tracked": len(changed), "silent": len(silent)}
        return items

    # ---------- 6. checklist coverage ----------
    def coverage(self, items):
        summ = []
        for n, it in enumerate(items):
            d = it.get("decision") or {}
            summ.append(f"{n}: {it.get('label')} | {d.get('topic', '')} | {d.get('decision') or ('deviation' if d.get('deviation') else 'ok')} | {clip(d.get('comment') or d.get('issue') or '', 160)}")
        p = f"""{self.brief_text()}

WHAT OUR REVIEW DID, per item:
{chr(10).join(summ)}

Which checklist entries (I1, I2, ...) are not yet addressed by any item? For each, say what to add. Return JSON:
{{"uncovered": [{{"id": "I3", "action": "edit" | "insert" | "memo", "para": <¶ number of {self.r['primary']} or null>,
"text": "the paragraph text (edit: full revised paragraph; insert: new paragraph)", "comment": "margin comment",
"note": "what the memo / cover note must say about it"}}]}}"""
        unc = (self.llm.json(p, default={}) or {}).get("uncovered", [])
        self.trace["steps"]["uncovered"] = unc
        return unc

    # ---------- 7. redline ----------
    def redline(self, items, gaps, unc, name):
        prim = self.r["primary"]; base = self.tool.files[prim]; respond = self.r["workflow"] == "respond_to_markup"
        rl = Redline(base, author=f"Counsel for {self.r['client']}", accept_existing=respond)
        edits, inserts, comments = {}, [], collections.defaultdict(list)
        for it in items:
            d = it.get("decision") or {}
            if respond:
                if d.get("decision") in ("reject", "counter") and d.get("text"): edits[it["para"]] = d["text"]
                c = d.get("comment"); tag = {"accept": "Accepted", "reject": "Rejected", "counter": "Counter-proposal"}.get(d.get("decision"), "")
                if c or tag: comments[it["para"]].append(f"{tag}. {c or ''}".strip())
            else:
                if not d.get("deviation"): continue
                for e in d.get("edits") or []:
                    if isinstance(e, dict) and isinstance(e.get("para"), int) and e.get("new_text"): edits[e["para"]] = e["new_text"]
                for e in d.get("inserts") or []:
                    if isinstance(e, dict) and isinstance(e.get("after"), int) and e.get("text"): inserts.append((e["after"], e["text"], None))
                if d.get("comment"): comments[it["paras"][0]].append(d["comment"])
        for g in gaps:
            d = g["decision"]; a = d.get("insert_after") if isinstance(d.get("insert_after"), int) else len(rl.units) - 1
            inserts.append((a, d["text"], d.get("comment")))
        for u in unc:
            if u.get("action") == "edit" and isinstance(u.get("para"), int) and u.get("text"): edits.setdefault(u["para"], u["text"]); comments[u["para"]].append(u.get("comment") or "")
            elif u.get("action") == "insert" and u.get("text"): inserts.append((u["para"] if isinstance(u.get("para"), int) else len(rl.units) - 1, u["text"], u.get("comment")))
        applied = 0
        for i, t in edits.items(): applied += rl.edit(i, t)
        for i, cs in comments.items():
            cs = [c for c in cs if c.strip()]
            if cs: rl.comment(i, "\n".join(cs))
        for a, t, c in sorted(inserts, key=lambda x: -x[0]): rl.insert_after(min(a, len(rl.units) - 1), t, comment=c)
        rl.save(f"{self.out}/{name}")
        self.trace["steps"]["redline"] = {"file": name, "edits": len(edits), "applied": applied, "inserts": len(inserts), "comments": len(rl.comments), "log": rl.log}

    # ---------- 8. memo / cover note ----------
    def memo(self, items, gaps, unc, name):
        respond = self.r["workflow"] == "respond_to_markup"; today = datetime.date.today().strftime("%B %d, %Y")
        rows = []
        if respond:
            for it in items:
                d = it.get("decision") or {}
                rows.append(f"- **{it['label']}{' (untracked change)' if it.get('silent') else ''}: {d.get('topic', '')}**: "
                            f"{ {'accept': 'Accepted', 'reject': 'Rejected', 'counter': 'Counter-proposed'}.get(d.get('decision'), 'Open')}. {d.get('comment', '')}")
        else:
            rows = ["| # | Provision | Issue | Playbook position | Risk | Exposure | Recommendation / fallback |", "|---|---|---|---|---|---|---|"]
            k = 0
            for it in items:
                d = it.get("decision") or {}
                if not d.get("deviation"): continue
                k += 1; cell = lambda s: clip(str(s or "").replace("|", "/").replace("\n", " "), 400)
                rows.append(f"| {k} | {cell(it['label'])} ({cell(d.get('topic'))}) | {cell(d.get('issue'))} | {cell(d.get('playbook_position'))} | {cell(d.get('risk'))} | "
                            f"{cell(d.get('exposure'))} | {cell(d.get('recommendation'))}{(' Fallback: ' + cell(d.get('fallback'))) if d.get('fallback') else ''} |")
            for g in gaps:
                k += 1; d = g["decision"]; pv = g["provision"]
                rows.append(f"| {k} | Missing: {pv.get('name')} | {clip(str(d.get('issue') or ''), 400)} | {clip(str(pv.get('requirement') or ''), 300)} | {d.get('risk', '')} | | Add the clause (see redline) |")
        summ = "\n".join(rows)
        notes = "\n".join(f"- {u.get('id')}: {u.get('note')}" for u in unc if u.get("note"))
        p = f"""{self.brief_text()}

{'OUR RESPONSES TO THEIR MARKUP' if respond else 'DEVIATIONS FOUND IN THEIR DRAFT'}:
{clip(summ, 30000)}
{('OTHER POINTS FROM THE INSTRUCTIONS:' + chr(10) + notes) if notes else ''}

Write the {'cover note to the other side transmitting our counter-redline' if respond else 'narrative sections of our internal issues and risk memorandum'}.
Return JSON: {{"subject": "...", {'"opening": "...", "key_points": ["our position on each key disputed provision, with the reason"], "closing": "next steps, timing, signature block"' if respond else '"summary": "executive summary", "missing_protections": "...", "preliminary_understandings": "departures from what was agreed or discussed earlier", "strategy": "negotiation strategy, priorities, leverage", "deadlines": "...", "escalations": "points needing client or committee approval"'}}}
Be specific: names, numbers, sections, dates."""
        n = self.llm.json(p, default={}) or {}
        b = self.b
        if respond:
            md = [f"**To:** {b.get('note_to', self.r['counterparty'])}  ", f"**From:** {b.get('note_from', 'Counsel for ' + self.r['client'])}  "] + \
                 ([f"**Cc:** {b['note_cc']}  "] if b.get("note_cc") else []) + [f"**Date:** {today}  ",
                  f"**Re:** {n.get('subject', 'Our counter-redline')}", "", n.get("opening", ""), "", "## Key points", ""]
            md += [f"- {k}" for k in n.get("key_points", [])] + ["", "## Our response to each change", ""] + rows + ["", n.get("closing", "")]
        else:
            md = ["# Issues and Risk Memorandum", "", f"**To:** {b.get('memo_to', self.r['client'])}  ", f"**From:** {b.get('memo_from', 'Counsel')}  "] + \
                 ([f"**Cc:** {b['memo_cc']}  "] if b.get("memo_cc") else []) + [f"**Date:** {today}  ", f"**Re:** {n.get('subject', 'Review of the draft')}  "] + \
                 ([f"**Benchmark:** {b['benchmark']}"] if b.get("benchmark") else []) + ["", "## Executive summary", "", n.get("summary", ""), "",
                  "## Deviations", "", summ, "", "## Missing protections", "", n.get("missing_protections", ""), "",
                  "## Departures from preliminary understandings", "", n.get("preliminary_understandings", ""), "",
                  "## Strategy and leverage", "", n.get("strategy", ""), "", "## Deadlines", "", n.get("deadlines", ""), "",
                  "## Escalations", "", n.get("escalations", "")]
            if notes: md += ["", "## Other instructions", "", notes]
        md_path = f"{self.out}/../{os.path.splitext(name)[0]}.md"; open(md_path, "w").write("\n".join(str(x) for x in md))
        import pypandoc
        subprocess.run([pypandoc.get_pandoc_path(), md_path, "-o", f"{self.out}/{name}"], check=True)
        self.trace["steps"]["memo"] = {"file": name, "rows": len(rows)}

    # ---------- run ----------
    def run(self):
        t0 = time.time()
        self.roles(); self.brief()
        respond = self.r["workflow"] == "respond_to_markup"
        items = self.markup_items() if respond else self.review_items()
        gaps = [] if respond else self.gap_items()
        unc = self.coverage(items + [{"label": "missing: " + g["provision"].get("name", ""), "decision": g["decision"]} for g in gaps])
        docx = [d for d in self.deliverables if d.endswith(".docx")] or ["redline.docx", "memo.docx"]
        red = next((d for d in docx if "redline" in d or "markup" in d), docx[0]); other = next((d for d in docx if d != red), "memo.docx")
        self.redline(items, gaps, unc, red); self.memo(items, gaps, unc, other)
        for d in self.deliverables:  # any other requested deliverable we did not build: say so in a note rather than skip silently
            if not os.path.exists(f"{self.out}/{d}"): self.trace.setdefault("missing_deliverables", []).append(d)
        self.trace["items"] = items; self.trace["gaps"] = gaps
        res = f"{LAB}/results/{self.run_id}"
        json.dump({"model": self.model, "task": self.task, "run_id": self.run_id, "pipeline": "pipeline/run.py"}, open(f"{res}/config.json", "w"), indent=1)
        json.dump({"model": self.model, "task": self.task, "llm_calls": self.llm.calls, "llm_errors": self.llm.errors, "wall_clock_seconds": round(time.time() - t0, 1),
                   "items": len(items), "gaps": len(gaps), "workflow": self.r["workflow"]}, open(f"{res}/metrics.json", "w"), indent=1)
        json.dump(self.trace, open(f"{res}/trace.json", "w"), indent=1, default=str)
        print(f"{self.run_id}: {self.r['workflow']}, {len(items)} items, {len(gaps)} gaps, {self.llm.calls} calls ({self.llm.errors} failed), {time.time() - t0:.0f}s -> {os.listdir(self.out)}", flush=True)


if __name__ == "__main__":
    model, task = sys.argv[1], sys.argv[2]
    rid = sys.argv[3] if len(sys.argv) > 3 else f"pipe-{model}-{task.split('/')[-1] if 'scenario' not in task else '-'.join(task.split('/')[-2:])}"
    Pipeline(model, task, rid).run()
