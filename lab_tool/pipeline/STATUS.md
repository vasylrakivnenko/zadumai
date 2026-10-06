# Pipeline: status and checkpoints

- 2026-10-06 ~07:00 UTC: plan written (PLAN.md, with the self-critique round). Azure has `cohere-rerank-v4.0-pro`
  (POST /providers/cohere/v2/rerank). Gemma 26B pod 0l1w4j0egtvdts (103.196.86.110:56722) building llama.cpp +
  downloading the model (pods/gemma26b_llamacpp_setup.sh). DELETE it when done.
- Redline writer (pipeline/docx_redline.py) done: ¶ mapping exact on dev docs (175/175, 223/223), word-level
  w:ins/w:del, inserted paragraphs, margin comments, optional accept-their-changes baseline; outputs pass pandoc
  --track-changes=all and LAB's docx validator.
- Hybrid search (pipeline/search.py): BM25 + bge (fused) top 30 -> Cohere rerank v4.0 pro (Azure). On the retrieval
  benchmark (273 fact-bearing criteria, title queries, top 6): 75.8% -> 79.5% with the reranker (analysis/rerank_eval.py;
  273 rerank searches, logged in runs/cohere_usage.jsonl; Cohere price per search not known yet). Adopted.
- pipeline/llm.py (Luna / Gemma 26B JSON clients, usage logs) and pipeline/run.py (the 9 steps) written. Dev runs
  started on the training-split dev tasks (guaranty first-turn redline, joint defense paper review), Luna and Gemma.
  Gemma 26B server up (tunnel 18081). Agent-loop Gemma runs on the 5 test tasks re-graded with jev_judge/grade.py.
- Dev v1 (Luna): joint defense review 0.935, guaranty markup response 0.875 (jev_judge/grade.py; dev = training split).
  Failures -> four process fixes (no task-specific wording): memo states its benchmark (playbook title / version / date);
  cc lines from the correspondence on memo / note headers; up to 40 required provisions checked for gaps; markup
  decisions accept only what doesn't weaken the client and prefer an allowed compromise to a flat rejection. Also:
  recitals / boilerplate acceptable unless they misstate facts (v1 flagged 68 of 104 clauses).
- Gemma dev v1: thinking ran to the 8,192-token limit without JSON on some calls, ~100 s per call at 8 in flight ->
  thinking off (chat_template_kwargs enable_thinking false). v1 outputs moved to runs/pipe_dev_v1/.
- Dev v2: all four dev runs restarted with the fixes.
- Dev v2/v3 (training split): grader bug found and fixed: re-running under the same run id re-graded the OLD texts
  (jev_judge/grade.py now drops a run's old rows before re-reading). So Luna's "v2" scores were v1's. Three more fixes
  (v3): the brief sees the playbook's opening lines (benchmark title / version / date), cc = reply-all from the incoming
  email, the gap step reads whole playbooks (a 40k-character cut lost §15.3 Limitation of Liability).
  Gemma 26B pipeline v3 (thinking off): joint defense review 1.000 (v2 0.903), guaranty markup response 0.938 (v2 0.906);
  135 + 33 calls, 4.5 + 2 min. Luna dev rerun with v3 and the 5 TEST tasks started (both models).
- 09:11 UTC: Gemma pod 0l1w4j0egtvdts DELETED after the test runs; tunnel closed.
- TEST (5 held-out tasks, run once, graded by Jev + Kimi K3): Gemma 26B pipeline 0.870 (agent loop 0.087 / 0.238),
  Luna pipeline 0.836 (agent 0.884 / 0.904), DeepSeek agent 0.954, Ivo Sage (rec.) 0.674. 0 failed calls, all
  deliverables. Luna dev v3: joint defense 0.968, guaranty 0.719 (v1 0.875: the "protect the client / prefer a
  compromise" change may have hurt it there; Gemma got 0.938 on the same task).
- Critique: no consolidation into distinct issues (over-flagging, ~70-row memos), no approval-authority field, markup
  answered per paragraph instead of per substantive change. v4 fixes listed in ../RESULTS.md; need a dev round first.
- Write-up: ../RESULTS.md "Fixed pipeline vs agent loop"; table script analysis/pipeline_compare.py.
