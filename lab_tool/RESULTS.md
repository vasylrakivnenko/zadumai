# Tool callers for the Legal Agent Benchmark (LAB): replay benchmark + DSPy prompt optimization

_Code + write-ups are also in the zadumai repo (`lab_tool/`, small result files only). The working copy with data, models and caches is `/root/zadumai_nli_proto/lab_tool` on the server; scripts use its absolute paths._

2026-10-05. Directory: `/root/zadumai_nli_proto/lab_tool`. Nothing here touches the live router.

## Why
The user's idea: instead of RL-training a huge model (Ivo Sage: a LoRA on the 284B DeepSeek V4 Flash, trained on LAB),
give an off-the-shelf LLM a small specialist **contract tool** (our document compiler, located reading, clause taggers,
NDA engine behind one command). Ivo's own evaluation record shows why this can work: the base model's Contracts
failures are mostly **context overflow** (29 of 55 held-out Contracts episodes ended in overflow or truncation);
where it finished normally it scored 0.953 vs Ivo Sage's 0.963. A tool that reads the documents and returns only what
matters keeps the context small. The user asked to benchmark the **tool caller** first, on its own.

## The replay benchmark (build_replay.py; no API calls)
- **A. When to call** (per turn): 924 decision moments cut from real transcripts (base + Ivo Sage episodes on LAB's
  125 held-out **General** tasks). "call" = the agent's next action read or extracted a source document (436); "no_call"
  = it wrote, edited or built its deliverable (488). The model sees the task, the document list and a compact history,
  and LAB's six tools (bash, read, write, edit, glob, grep) plus `contract_tool` (find / checklist / ask).
- **B. What to ask** (per task): 318 LAB **training-split** Contracts tasks (harvey-labs @ a2b429e) with their rubrics
  (20,096 criteria). The model plans its `contract_tool` queries up front; scored by the share of rubric criteria a
  query reaches (bge-small cosine), against a floor (the same queries scored on another task's rubric).
- The 55 held-out Contracts tasks were not used here; the end-to-end pilots below run on them (never for tuning).

## Callers compared (run_replay.py, score_replay.py)
A: 120 states (60 / 60); "guided" = one system-prompt line asking to prefer the tool; "neutral" = none.

| model | where | A guided: uses tool when it should / holds off while writing / balanced | A neutral balanced | B: queries per task, coverage @0.60, floor, lift |
|---|---|---|---|---|
| **Gemma 4 E4B** (4B effective) | vLLM on one RTX 4090 | **93.3% / 70.5% / 81.9%** | 66.7% | 7, 77.8%, 57.1%, +20.7 |
| DeepSeek V4.1 Flash | Fireworks | 48.3% / 91.7% / 70.0% | 50.0% | 8, 80.6%, 58.5%, **+22.1** |
| gpt-oss-120b | Fireworks | 76.7% / 55.0% / 65.8% | 57.5% | 1, 38.0%, 21.2%, +16.8 |
| Gemma 4 26B-A4B | Gemini API | 41.7% / 78.3% / 60.0% | 47.4% | (stopped) |
| Kimi K3 | Fireworks | 23.3% / 90.0% / 56.7% | 50.0% | 4, 70.2%, 49.1%, +21.1 |

- Without an instruction the big models almost never use a new tool; the 4B Gemma follows the instruction best.
- Gemma 4 E4B: 260 calls in 3.5 min on one 4090. Its weakness: calling the tool while it should be writing (29.5%).

## DSPy prompt optimization of Gemma 4 E4B (optimize_dspy.py, optimize_dspy_b.py)
Split by task (crc32 % 10 < 7: optimization; >= 7: test). Model tuned: Gemma 4 E4B (vLLM, ~48 requests in flight:
8,900-10,000 prompt tokens/s); prompt rewriter for GEPA: Kimi K3. DSPy 3.4, native function calling.

### Part A: when to call (test = 235 moments from 32 General tasks never used in optimization)
| prompt | format | uses tool when it should | holds off while writing | balanced |
|---|---|---|---|---|
| starting instruction | DSPy | 100% | 12.7% | 56.3% |
| + 4 examples (BootstrapFewShot) | DSPy | 100% | 41.3% | 70.6% |
| **GEPA instruction** (10 rounds, val 0.90) | DSPy | 86.2% | 92.1% | **89.2%** |
| plain one-line instruction | deployment | 92.7% | 62.7% | 77.7% |
| GEPA instruction, transplanted | deployment, native calls only | 13.8% | 100% | 56.9% |
| same, decisions (JSON-text calls counted) | deployment | 84.4% | 91.3% | 87.8% |
| **GEPA + one output-format line**, decisions | deployment | 83.5% | 98.4% | **90.9%** |
| same, native calls only | deployment | 34.9% | 100% | 67.4% |
| GEPA with its output section hand-rewritten ("reply NO_CALL") | deployment | 0.9% | 100% | 50.5% |

- GEPA (Kimi K3 rewriting from failures) wrote a ~2,500-word instruction (data/gepa_A_instruction.txt): a "gate"
  (no contract_tool once drafting / building / proofing has begun, or when the information is already in context),
  phase detection (survey -> extract -> draft -> build), one query per remaining source, recorded failure patterns.
- The decisions transfer to the deployment prompt (90.9% balanced vs 77.7% for the plain line), but the instruction
  was tuned in DSPy's layout ("output a list, [] when none"), so Gemma often writes the calls as JSON text instead of
  native tool calls. A controller harness that also accepts JSON-text calls (a few lines; we own that code) gets the
  90.9%. Hand-editing the tuned instruction backfired (50.5%): tuned prompts are fragile; re-optimize instead.
- Fix for the format properly: run GEPA directly in the deployment prompt layout (custom DSPy adapter); not done.

### Part B: what to ask (test = 40 training-split Contracts tasks never used in optimization; coverage at bge cosine 0.65)
| prompt | format | coverage | floor (another task's rubric) | lift | queries per task |
|---|---|---|---|---|---|
| starting instruction | DSPy | 41.8% | 12.3% | +29.5 | 9 |
| GEPA instruction (14 rounds; best = candidate 5, val 0.533) | DSPy | 47.9% | 15.7% | **+32.2** | 12 |
| plain one-line instruction | deployment | 31.0% | 10.5% | +20.5 | 7 |
| **GEPA instruction** | deployment | 45.4% | 18.1% | **+27.2** | 12 |

- GEPA's B instruction (data/gepa_B_instruction.txt, 1,503 words): a hard budget of 12 calls, `checklist` on the
  principal drafts / standard form, one provision-specific `find` per file, no generic queries. It transfers to the
  deployment prompt without the format problem of part A (+6.7 lift points); part of the gain is using the full budget.
- optimize_dspy_b.py's own final test step crashed on one truncated answer (800-token cap), after GEPA had finished;
  the best instruction was recovered from the GEPA log and scored by final_B.py (3,000-token cap, failures = empty plan).

### Muse Glimmer 30B (Meta) as the caller: full precision vs 4-bit (lab_tool/muse/, PROGRESS.md)
Same held-out tests, deployment format; Gemma's GEPA instructions transplanted as they are (not re-tuned for Muse).
Fireworks could not parse Muse's own tool-call format (ATEM), so run_replay renders Meta's chat template and parses the
ATEM blocks itself (call_atem).

| caller | where | A plain (balanced) | A GEPA (balanced, native calls) | B plain lift (queries) | B GEPA lift (queries) |
|---|---|---|---|---|---|
| Gemma 4 E4B | vLLM, one 4090 | **77.7%** | 67.4% (90.9% counting JSON-text calls) | +20.5 (7) | +27.2 (12) |
| Muse Glimmer 30B BF16 | Fireworks, 1x B300 | 47.1% | 85.2% | +26.5 (5) | **+34.8** (12) |
| Muse Glimmer 30B Q4_K_M | llama.cpp, one 4090 | 45.9% | 83.8% | +28.1 (6) | +32.9 (12) |

- 4-bit costs 1-2 points (within noise: +-3 points on A's 235 moments). It fits the 4090 (16.3 GB) but is slow there:
  ~66 min for the 550 requests vs ~3.5 min for Gemma's 235 part-A requests on vLLM.
- Muse ignores a new tool without a tuned instruction (plain A 46-47%), but with Gemma's GEPA instruction it calls
  natively (no JSON-text problem) and plans the best queries (B lift +33-35). Gemma E4B stays the better and far
  cheaper "when to call" model once its output format is fixed; Muse is the better planner.
- Both temporary deployments (Fireworks, Runpod pod) were deleted after the runs.

## Contract tool v0 (contract_tool/, eval_tool_v0.py; 2026-10-05)
Built after reading LAB's harness: `read` returns the whole parsed document and silently accepts tracked changes;
`grep` reads raw files, so it finds nothing inside .docx / .xlsx / .pptx. The tool: `outline`, `find` (BM25 +
bge-small on CPU, reciprocal-rank fusion, deleted text searchable), `show` (a clause or ¶ range in full), `changes`
(tracked changes clause by clause), `compare` (two versions aligned, word diff); ~1,200 tokens max per call. General
purpose: no LAB task files or rubrics. Evaluated on the 318 training-split Contracts tasks only; no LLM calls, no GPU.

| check | LAB harness | contract tool v0 |
|---|---|---|
| tokens to load a task's documents | `read` everything: median 110k, p90 182k; 192 of 318 tasks over 100k | `outline` of all documents: median 383 tokens; `find`: ~600 per call |
| text a search can see | `grep`: 6.6% (emails / .txt only) | 100%; 3,274 files, 0 parse errors |
| tracked changes (248 redlined .docx) | 0 shown (374,812 deleted words silently dropped) | 37,445 changes; words match pandoc --track-changes=all 100% / 100% |
| current text vs what `read` shows | | covers 99.8% of `read`'s words |
| compare two versions (each redline split into original and current) | none (agents hand-diff) | paragraphs: recall 94.7%, precision 99.7% (words: precision 97.5%, recall 83-85%, a minimal diff keeps re-inserted words) |
| speed | | find 0.06 s per query; compare 0.02 s; one-time embedding ~3.5 min per task on 1 core (cached) |

**find, on 273 fact-bearing rubric criteria of the 40 replay test tasks** (criteria whose exact facts: amounts,
percentages, dates, durations, quoted terms, company names occur in at most 5 passages; evidence = the passage with
the most of them; the rubric is used only to score, never by the tool or the callers):

| who asks | passages with the fact returned | tokens per task |
|---|---|---|
| a precise question (the criterion's title) | **75.8%** (78.0% counting any passage with the fact) | |
| Gemma 4 E4B GEPA plan, run on the tool (11 queries, written before seeing any document) | 12.5% | 6,587 |
| Muse Glimmer GEPA plan, run on the tool | 11.0% | 6,627 |
| the same plans, 12 passages per query instead of 6 | 13.9-19.4% | ~13,000 |
| the same plans, compound queries split into sub-queries (no budget cap) | 17.2-30.4% | 2-9x passages |
| `read` the documents those plans named | 82.8-97.8% | 96,657-104,012 |
| `read` everything | ~100% | 118,090 |

- The tool reads well: redlines exact, comparisons near-exact, every format, 1/20 of the context.
- Search works when the question is precise (76-78%), but up-front query plans reach little (11-15%): they are long
  compound lists of topics written before the caller has seen any document. Part B's "coverage" score (cosine between a
  query and a criterion) overstated the planners: being similar to a criterion is not retrieving its fact.
- So the bottleneck moves from reading to asking: the caller has to ask narrow questions iteratively (outline -> find
  -> show), as an agent loop does. Only a real agent loop can measure that (next step; needs LLM calls).
- Redlines matter for 179 criteria in 69 training tasks (counterparty's edits, what was accepted), but only 9 of the 273
  anchored criteria here have their fact inside redlined text, so this part doesn't measure `changes`' value.

## Tiny LAB pilot, end to end (lab_pilot.py; 2026-10-05/06)
LAB's own harness and podman sandbox (harvey-labs @ a2b429e, patched: optional contract_tool, emulated 262,144-token
window like the recorded runs, Fireworks judge). Agent: DeepSeek V4.1 Flash (Fireworks serverless), temperature 0.6
(at LAB's default 0 its reasoning loops "Let me go. Let me run it." until the output limit), 32k output cap, a looping
response redrawn twice at most. Judge: Kimi K3, one pass per criterion, LAB's rubric prompt; it also re-judged the
recorded base and Ivo Sage deliverables. Two held-out tasks (the user: "maybe an even smaller test first"), both arms.

| task (criteria) | base V4 Flash (recorded) | Ivo Sage (recorded) | V4.1 Flash, LAB tools | V4.1 Flash + contract_tool |
|---|---|---|---|---|
| teaming agreement, counterparty paper review (58) | 0.00 (overflow) | 0.966 (original judge 0.95) | **0.948** | 0.00: redline built (172 ins / 86 del / 35 comments) in work/, then the reasoning looped 3x at ~210k context before saving it or writing the memo |
| quality agreement, first-turn redline (45) | 0.00 (no deliverable) | 0.978 (original 0.98) | 0.00: reasoning looped 3x at ~154k context | 0.00: context limit at 252k while debugging its own docx-building script |

- Kimi K3 reproduces the recorded judge on Ivo's deliverables (0.966 vs 0.95, 0.978 vs 0.98).
- When it doesn't break, V4.1 Flash in LAB's plain harness nearly matches Ivo Sage (0.948 vs 0.966).
- 3 of 4 episodes died of the model's long-context reasoning loops (150k+) or the context limit. Not a test of the
  tool yet. The tool cut reading, but building the tracked-changes .docx by hand (python + XML, debugging) filled the
  context instead: the next tool capability is "make the redline" (the agent gives clause edits, the tool writes the
  tracked-changes .docx with comments).
- First judging was invalid (LAB's scorer runs `pandoc` from PATH; missing here: every .docx unreadable); re-judged.
- Cost (Fireworks list prices): agent $3.4-16.5 (cached share unknown; includes aborted attempts and looping draws),
  judge $9.9 (incl. $2.5 for the invalid pass), total $13-26.

## LAB pilot, round 2: 7 held-out tasks (2026-10-06)
Fixes after round 1: DeepSeek V4.1 Flash at **low reasoning effort** (at default effort it overthinks or loops past the
32k output limit at the drafting step, even at ~60-86k context); LAB's scorer gets `pandoc` on PATH; exact usage logs.
Judge: Kimi K3 (one pass; it reproduced Ivo's recorded scores within 0.02 on two tasks). Base and Ivo columns: recorded,
original judge. Tasks: the 7 smallest held-out counterparty-paper-review / first-turn-redline tasks.

| task | base V4 Flash (rec.) | Ivo Sage (rec.) | V4.1 Flash, LAB tools | V4.1 Flash + tool | Gemma 26B-A4B 4-bit, LAB tools | Gemma 26B + tool |
|---|---|---|---|---|---|---|
| NDA first-turn redline | 0.647 | 0.880 | 0.880 | 0.840 | 0.060 (overflow) | **0.680** |
| NDA counterparty paper review | 0.090 | 0.936 | 0.937 | 0.905 | 0.000 | 0.508 (a scratch analysis only) |
| easement paper review | 0.000 | 1.000 | 1.000 | 0.982 | 0.175 (response.md only) | 0.000 |
| quality-agreement paper review | 0.296 | 0.000 | 1.000 | 1.000 | 0.244 (response.md only) | 0.000 |
| ISDA confirmation paper review | 0.000 | 0.931 | 0.925 | 0.887 | 0.340 | 0.000 |
| cyber-insurance paper review | 1.000 | 0.968 | 0.968 | 0.968 | 0.000 | 0.000 |
| easement first-turn redline | 0.000 | 0.552 | 0.931 | 0.966 | 0.000 | 0.000 |
| **mean (7)** | 0.290 | 0.752 | **0.949** | 0.935 | 0.117 | 0.170 |

- DeepSeek V4.1 Flash with LAB's own tools, low reasoning effort, no training: 0.949, above Ivo Sage's recorded 0.752 on
  these 7 (Ivo failed two of them) and at Ivo's level where Ivo finished. Caveats: different judge for the recorded
  columns, a newer base model, 7 tasks.
- Our tool doesn't help a strong model here (0.935 vs 0.949, within noise). For a small local model it helped on the
  NDA redline (Gemma 26B-A4B 4-bit on one RTX 4090: 0.68 with the tool, 0.06 without: overflow), but on the other six
  tasks Gemma 26B mostly ran out of its window either way (131k KV shared, 98k usable with a 32k output reserve; one
  episode at a time): means 0.17 with the tool, 0.12 without. Too small a window for these tasks on one 4090.
- Gemma 4 E4B can't drive LAB's loop: it answers with a plan in text, no tool call, and LAB ends the episode.
- Cost: judge $42.6 in all (Kimi K3 sends the deliverables with every criterion: ~$2.2 per deliverable set), DeepSeek
  agent ~$7.3 (cached input $0.006/M), pods ~$2.

## Cheap LAB judge: Jev + an LLM for the uncertain criteria (jev_judge/, 2026-10-06)
Plan in jev_judge/PLAN.md. Rubric criteria split once into simple yes/no statements (GPT-6 Luna on Azure, all 55
held-out Contracts tasks: $0.13); Jev answers every statement plus the whole criterion (P(true)), on the exact text LAB's
judge reads (deliverables longer than Jev's ~32k-token input: the criterion's most relevant passages); a logistic
combiner and a confidence band fitted on half the tasks; criteria inside the band go to an LLM judge.
Labels: Ivo Sage's records (6,438 non-empty criteria; 3 judge passes each) + our Kimi-graded runs.

| test tasks (never used to fit), 2,349 criteria where judge passes 2 and 3 agree | agreement |
|---|---|
| one LLM judge pass (pass 1) | 99.32% |
| Jev alone | 97.32% |
| Jev on the criteria it is sure of (89%; band p <= 0.10 fail / >= 0.94 pass) | 99.52% |
| **hybrid: Jev on 89%, the LLM judge on the other 11%** | **99.15%** |
| stricter band 0.05 / 0.97: Jev 82%, LLM 18% | 99.23% |

- Task-level scores: hybrid within 0.009 of the 3-pass majority on average (Jev alone 0.022).
- Our runs (Kimi labels): Jev alone 97.2%; on its confident 82%, 99.9%.
- Cost: Jev $2.30 for all 6,438 criteria (54.7M input tokens at $0.042/M, output free), 427 s. A full Kimi K3 grading
  of one deliverable set is ~$2.2; hybrid ~$0.3 with Kimi on the uncertain 11%, a few cents with Luna (not yet
  validated as the judge).
- Caveat: labels are 94% "pass", so agreement numbers start high; kappa: hybrid 0.90 vs one LLM pass 0.94.
- GPT-6 Luna (Azure) as the LLM judge, LAB's own judge prompt (jev_judge/luna_judge.py, $4.29): NOT validated. vs Kimi
  K3 on our runs 94.2% (kappa 0.68; Luna stricter: 87% pass vs 93%); on recorded criteria 97.3% vs one recorded pass
  99.3%; on Jev's uncertain criteria (where it would be used) 82.8% vs 96.1%. The uncertain criteria stay with Kimi
  K3. Untested: Luna at higher reasoning effort.

## Where things stand (2026-10-06)
- **Tool callers** (replay benchmark): prompt optimization (DSPy GEPA) gets Gemma 4 E4B to ~89-91% on "when to call";
  no fine-tuning needed. But Gemma 4 E4B can't drive LAB's agent loop on its own (it answers with a text plan).
- **contract_tool v0** reads every format, shows tracked changes exactly (LAB's `read` hides them, `grep` can't see
  inside .docx) and compares versions, at ~1/20 of the context; search works on precise questions (76-78%), not on
  up-front compound query plans (11-15%).
- **End to end in LAB's own harness** (7 smallest held-out paper-review / first-turn-redline tasks): untrained
  DeepSeek V4.1 Flash at low reasoning effort 0.949, above Ivo Sage's recorded 0.752 (different judge, newer base);
  the tool adds nothing for it (0.935). Small local models: Gemma 26B-A4B on one 4090 mostly overflows.
- **Grading:** Jev + an LLM for Jev's uncertain 11% matches one LLM judge pass (99.15% vs 99.32%) at a fraction of the
  cost. The LLM for the uncertain criteria stays Kimi K3: Luna (Azure) failed validation (82.8% vs 96.1% there).
- **Open:** a "make the redline" action (drafting the .docx, not reading, fills the context); DeepSeek V4 Flash (now on
  Azure, $0.19 / $0.51) for a same-base comparison with Ivo Sage; the grader for new runs (Jev + Kimi K3); Luna at
  higher reasoning effort as a cheaper judge.

## Files
- `build_replay.py`, `run_replay.py`, `score_replay.py`, `optimize_dspy*.py`, `final_B.py`, `run_*.sh`: replay benchmark
  and DSPy; `muse/`: Muse Glimmer track (PROGRESS.md log); `data/`: replay sets, caller outputs, GEPA instructions.
- `contract_tool/`: the tool (README.md), `server.py` for the harness; `eval_tool_v0*.py`: its evaluation.
- `lab_pilot.py`: LAB episodes (`PILOT_MODEL`, `PILOT_EFFORT`), judging, report; `analysis/`: summaries and costs;
  `pods/`: the Gemma pod setup scripts; `harvey_labs_patches.diff`: our changes to LAB's harness (below).
- `jev_judge/`: the cheap judge (PLAN.md; texts, decompose, check, evaluate, luna, luna_judge).
- LAB's harness patches (harvey-labs @ a2b429e; all opt-in by env var): `contract_tool` as a 7th tool
  (`LAB_CONTRACT_TOOL`) and one system-prompt line about it; Fireworks adapter: emulated context window
  (`LAB_CONTEXT_CAP`), output cap (`LAB_MAX_OUTPUT`), call timeout, redraw of looping responses
  (`LAB_RESAMPLE_LENGTH`), usage / truncation logs, llama.cpp error mapping; judge: Fireworks models.
- Throughput / cost notes: vLLM on one 4090 ran Gemma 4 E4B at ~9-10k prompt tokens/s (~48 requests in flight).
  Replay + DSPy: Fireworks ~810 calls, a B300 Muse deployment ~15 min, three Runpod pods ~3.5 h. Pilots: judge
  ~$43, DeepSeek ~$7.3, pods ~$3.5; Jev judge $2.30 + Luna $0.13.
