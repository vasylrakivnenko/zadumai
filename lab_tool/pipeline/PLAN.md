# Fixed pipeline for counterparty paper review and first-turn redline: plan (2026-10-06)

The user: "build it for counterparty paper review and first-turn redline. Run it with Luna and Gemma 26B on our 5 tasks
against the agent-loop results. Maybe also a good hybrid search (keyword + RAG, Cohere reranker). Think well, run a
self-critique -> refinement cycle. Then detailed plan and implementation. Write-ups and checkpoints."

## Why
In the agent loop the losses are process failures, not legal judgment: context overflow, reasoning loops, hand-building
the redline .docx (the biggest token sink), missing / misnamed deliverables. Both task types follow one professional
workflow every time, so a fixed workflow with small, focused LLM calls should be more reliable, cheaper, and usable by
a small local model.

## Rules (the user's principle: a general system, no benchmark tricks)
- Inputs: the task instructions and the documents only (the harness's load_task gives the instructions; the rubric is
  never read). Prompts describe legal practice, not LAB's rubric wording.
- Developed and debugged on TRAINING-split tasks of the same two types (dev set below); the 5 held-out tasks of our
  working set are run once at the end, for the comparison.

## Design v1 and its critique
v1: give the LLM the whole contract + playbook and ask for the revised contract and the memo.
- Long contexts again (the agent's failure mode); a small model cannot do it; the diff is uncontrolled (rewrites of
  acceptable text); the memo can silently skip deviations; no tracked changes unless something builds them.
=> v2: many small calls over explicit work items, deterministic document building, deterministic completeness.

## Design v2 (implemented)
0. Load: instructions, deliverable names ("### Output:"), documents parsed by contract_tool (units with ¶ ids, clause
   numbers, sections, tracked changes kept apart).
1. Roles (1 call): each document's role: their_draft | their_markup | our_draft | standard_form | playbook |
   instructions | correspondence | reference; who we represent; the counterparty. Hint: a .docx with tracked changes is
   a markup.
2. Brief (1 call): from the instructions, emails and notes: client, counterparty, objectives, a numbered checklist of
   explicit instructions / positions / deadlines, who the memo or cover note is to / from.
3. Work items (no LLM):
   - paper review: the clauses of their draft (paragraphs grouped by clause number / heading; long ones split);
   - first-turn redline: every paragraph their markup changed (our text vs theirs), plus silent edits (our draft vs
     the markup's original text, where they differ without tracked marks).
4. Evidence per item: hybrid search over the playbook, standard form, brief and references: BM25 + bge-small, fused by
   reciprocal rank, top 30 reranked by Cohere rerank v4.0 pro (Azure), top passages up to ~3k tokens.
5. Decide per item (1 call each, in parallel), JSON:
   - paper review: deviation? topic, risk (High / Medium / Low), issue, playbook position, fallback, recommendation,
     edits [{para, new_text}] and inserts [{after, text}] (none when the clause is acceptable), margin comment;
   - first-turn redline: accept | reject | counter, the paragraph text we want, margin comment (rationale), playbook
     reference, escalation flag.
6. Gaps (paper review): required provisions from the playbook (1 call on the playbook outline + text) checked against
   the draft by search; missing ones become insert items (1 call each). Both types: every brief checklist item must be
   addressed by some item; uncovered ones get an item.
7. Redline .docx (no LLM): paper review on their draft; first-turn redline on their markup with their changes accepted
   as the baseline, our responses as new tracked changes; word-level w:ins / w:del, inserted paragraphs, a margin
   comment on every item we act on (and every change of theirs we accept, so each change is answered).
8. Memo / cover note (.docx via pandoc): the deviation table / the per-change list built from the items (no LLM, so
   nothing is dropped); narrative sections (summary, missing protections, departures from preliminary understandings,
   strategy and leverage, deadlines, next steps) written by 1 call.
9. QA: exact deliverable names, .docx parse (pandoc --track-changes=all), counts of changes / comments; 1 call checks
   the memo / note against the brief checklist; one repair pass if something is missing.

## Self-critique of v2 -> refinements
- ¶ ids must map exactly to XML paragraphs -> the writer walks document.xml with the parser's own traversal and checks
  every mapped paragraph's text equals the unit's text before editing (else the edit is skipped and logged).
- Full-paragraph replacement can mangle numbering -> edits keep the clause number; the word diff shows only real
  changes; formatting of untouched words is kept (the run properties of the first run).
- Over-editing acceptable text -> default "acceptable, no edit"; an edit must cite the playbook or an instruction.
- Small-model JSON errors -> json_object mode, one repair retry, then "no change" (logged).
- Completeness -> deterministic tables (8) and checklist coverage (6, 9).
- Reranker: unknown value and price -> measured first on our retrieval benchmark (fact-bearing criteria, title
  queries: 75.8% without it); used only if it helps; every search counted.
- Table rows: comments only (no in-cell edits) in this version.

## Models, compute
- GPT-6 Luna (Azure, Responses API, effort max: as in its agent run), 16 calls in flight.
- Gemma 4 26B-A4B 4-bit on one RTX 4090 (llama.cpp, json_object), 4 slots.
- 8 CPUs: tasks run as parallel processes; parsing, search, docx building are local.

## Dev set (training split) and test
- Dev: joint-defense-agreement-counterparty-paper-review; license-agreement-counterparty-paper-review/scenario-03;
  guaranty-agreement-first-turn-redline; account-control-agreement-first-turn-redline/scenario-01.
- Test (held-out working set, runs/pilot/set5_tasks.txt): NDA first-turn redline, NDA paper review, easement paper
  review, quality-agreement paper review, easement first-turn redline.
- Grader for everything: jev_judge/grade.py (Jev + Kimi K3; within 0.009 of full Kimi on 14 runs). Agent-loop runs
  on the 5 tasks are re-graded with it where needed, so all columns use one grader.

## Checkpoints
pipeline/STATUS.md, updated after every step.
