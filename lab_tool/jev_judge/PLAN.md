# Jev as a cheap LAB judge: plan (2026-10-06)

The user: "make a good detailed plan... put some kind of threshold in case Jev gives low confidence... measure Jev's
agreement... split each criterion into simple yes/no statements. Go build it."

## Goal
Grade LAB deliverables with Jev (systemone noul: P(statement true | document), ~0.1-0.2 s per request, ~18k-token
documents fine) instead of an LLM call per criterion (Kimi K3: ~22k tokens per call, ~$2.2 per deliverable set).
Uncertain criteria go to the LLM judge. Never used by the agent: grading only.

## Data we already have (labels)
1. Ivo Sage's records, 55 held-out Contracts tasks x {base V4 Flash, Ivo Sage} x 3 judge passes (judge_1..3.json):
   ~6,000 criteria with 3 verdicts each. Label = majority of 3. Ceiling = how often one pass agrees with the others.
2. Our pilot runs graded by Kimi K3 (scores.json, one pass): a second, independent check.
Episodes without deliverables (overflow, nothing written) are trivial fails: excluded from agreement, counted apart.

## Pipeline
1. Texts (texts.py, harvey venv): for each episode x criterion, exactly the text LAB's judge sees: the criterion's
   deliverable files, pandoc markdown, tracked changes shown when the criterion asks for them
   (evaluation_options.include_docx_redlines), via LAB's own scoring helpers. Cached; grouped by identical text.
2. Decomposition (decompose.py, once per task, cached; GPT-6 Luna on Azure, $0.13 for 55 tasks): each criterion -> 1-5 simple yes/no statements about
   the deliverable, each tagged pass_if_true / fail_if_true, with "all" / "any" logic and a flag when it needs the whole
   document at once (counts, "every deviation", consistency). Uses the rubric only (no verdicts, no deliverables).
3. Jev (check.py): one request per (episode, text group): state = the text, questions = every statement of every
   criterion that reads that text (chunks of 40), plus the whole criterion as one more statement. Cached answers.
4. Combine (evaluate.py): features per criterion = p(whole criterion), min / mean of the statement checks after
   polarity, the logic. A small logistic regression on calibration tasks -> p_pass.
5. Confidence thresholds: p_pass >= hi -> pass, <= lo -> fail, in between -> "escalate" to the LLM judge. lo / hi
   chosen on calibration tasks: the widest confident band whose agreement with the majority label is >= that of one
   LLM judge pass (evaluate.py; the pass-to-pass agreement is reported as the ceiling).

## Measurement (test tasks never used to fit anything; split by task, crc32)
- Agreement with the majority label: all criteria; confident ones; kappa (chance-corrected).
- Escalation rate (share sent to the LLM judge) and the hybrid agreement (escalated = a judge pass).
- Task-level criterion pass rate: hybrid vs majority, mean absolute error (target <= 0.02).
- Same on our Kimi-graded pilot runs.
- Cost and time: Jev requests / tokens / seconds vs the LLM judge's calls and dollars.

## Status (2026-10-06)
Built and measured: see ../RESULTS.md, "Cheap LAB judge". Band p <= 0.10 fail / p >= 0.94 pass: Jev decides 89% at
99.5%; hybrid 99.15% vs one LLM pass 99.32% (test tasks, fair labels). Next: validate Luna as the escalation judge
(luna_judge.py), then a grader for new runs.
