# Neuro-symbolic ideas 1-5, the object model + query language (2026-10-03), and the NDA engine (2026-10-04/05)

> **Review notes (2026-10-06)** — discrepancies found in a review; the text below is unchanged unless noted.
>
> - Scope: most of this file is the 2026-10-04/05 NDA work, not only the 2026-10-03 ideas of the title.
> - NDA held-out set: 1,020 questions in some tables, 959 in others: `feats/nda_ho_ft.jsonl` has 1,120 rows for 959
>   unique ids (61 questions never extracted; `xgb_stack.load` keeps the first duplicate). The 87.7% and 87.45%
>   baselines are the same model on these two sets. (`cuad_ho_ft.jsonl`: 1,501 rows for 950 ids, same pattern.)
> - s160 and the ContractNLI test were spent by the one-time sealed test (section "12."); lines calling them "sealed
>   and never used" predate it.
> - The sealed numbers are for the 96-channel CNN (models `ndacnn/models/rising+secallw1.0+d128+k5+r2{,+s1..+s4}_f-1.pt`;
>   the glob `r2*_f-1.pt` would also match the r2c192/256/384 models). The final 384-channel CNN was never sealed-tested.
> - `ndacnn/runs/gpu4/models/rising+secallw1.0+d128+k5+r2{,+s1,+s2}_f-1.pt` are DIFFERENT files from the same names in
>   `ndacnn/models/` (the sealed-test models); `report_x.py` reads `runs/gpu4/preds`.
> - "Next (running)" / "Running:" lines are stale: those runs finished (results follow them). The latency script cited
>   as "scratchpad latency.py" was not kept (`typed/latency.py` is a different one). `nda_joint.py` has no reported result.
> - Installing the fine-tuned reader (PORT.md) is still pending.


The user: "Go with 1,2,3,4,5! Try different combinations, test, then report." Then: "combine both ideas. Make about 8
core classes the compiler's output, keep the small query language as the interface questions compile into, and have the
specialized clause classes host the per-type answerers we already built." No LLM calls in any of this.

## Setup
- Code: ns.py (ideas 1-2), fold.py (idea 3's learner), q5.py + extract5.py (idea 5), build_ft.py + reader_net/train.py
  (idea 4), extract.py (one feature pass per set: every signal per question, candidates at low thresholds so any rule
  replays offline), combos.py / marginal.py (combinations), oo.py (object model + query language), test_typed.py (10 pass).
- Bug fixed on the way (pipeline.py): a located text of more than 12 network windows counted as a "long document" and
  the network's answer was off there (18 of the 126 sealed wc1 questions, 2-3% elsewhere); the located reader now reads
  up to 24 windows. With the fix, today's rule answers 27/27 sealed wc1 questions (was 23/23).
- Sets. Tuning: nda_tune (100 ContractNLI train NDAs x 17), cuad_tune (40 of cuadc's dev contracts x LegalBench CUAD
  questions), wc1_open, short_tune (v5 + v6 user-style questions over single provisions). Held-out, never read in this
  work: nda_ho (60 other train NDAs, 1,020), cuad_ho (25 other dev contracts, 950), short_ho (v6b + v7, 2,321);
  wc1_sealed is spent (reported only).

## What each idea is
1. Consistency (ConCoRD-style): related questions written by rule from the typed form ("Can S V R?" <-> "Is S prohibited
   from V-ing R?", a paraphrase), read by the network on the same text; "1" drops an inconsistent "yes", "1+" also takes
   a consistent "yes" from p 0.80, "+no" takes a consistent network "no".
2. Priorities (defeasible / Catala-style): the contract's statements as norms (actor, act, object, permit / oblige /
   ban, "notwithstanding", "subject to Section X" via the compiler's cross-references, exceptions); "2" vetoes an answer
   whose statement loses to a conflicting norm; "2+no" answers "no" to a may-question whose act has an unexcepted ban.
3. Learned gates (FOLD-R++-style default rules with exceptions, fold.py): "3d" keeps today's rule as the default and
   learns VETO and ADD rules; "3v" vetoes only. Trained on the tuning sets.
4. A fine-tuned reader network on located text (labels deduced from gold answers + evidence spans): see below.
5. Untyped questions: first person by supervaluation (every party reading must agree), two-action questions split and
   recombined by "or"/"and", a learned parser (asks 82.4%, action 87.5% agreement with the grammar in CV; when sure, on
   52%, 90.7% both right) feeding the priority and typed checks.

## Results with today's network (marginal effect vs today's rule: answers added, right / all; answers removed, of which wrong)
| idea | short_ho | nda_ho | cuad_ho | short_tune | nda_tune | cuad_tune |
|---|---|---|---|---|---|---|
| today's rule (answered @ right) | 449 @ 100% | 204 @ 93.1% | 164 @ 85.4% | 735 @ 99.5% | 297 @ 97.3% | 286 @ 79.0% |
| 1 (veto) | -3 (0 wrong) | -1 (0) | 0 | -4 (0) | -2 (0) | -2 (0) |
| 1+ (consistent yes from 0.80) | +50/51 | +20/23 | +4/7 | +94/102 | +31/36 | +6/12 |
| 1+no | +56/58 | +22/26 | +4/7 | +113/121 | +34/46 | +7/13 |
| 2 (priority veto) | -3 (0) | -6 (1) | 0 | 0 | -6 (0) | -3 (0) |
| 2+no | +3/3 | +3/8 | 0 | +1/1 | +3/20 | 0 |
| 3v (learned vetoes) | 0 | 0 | -16 (3) | 0 | 0 | -37 (28) |
| 3d (default + exceptions) | +106/111 | +5/8 | +1/2, -16 (3) | +194/197 | +12/12 | +1/1, -37 (28) |
| 5 | 0 | 0 | 0 | +1/2 | 0 | 0 |
Consistent network "yes" below today's threshold, by p band (tuning | held-out): [0.80, 0.85) 21/27 | 16/20;
[0.85, 0.88) 26/31 | 9/11; [0.88, 0.90) 16/21 | 14/14; [0.90, 0.935) (short texts) 65/68 | 33/34.
Consistent network "no" on short texts: 19/19 (tuning) + 6/7 (held-out); on NDAs 3/10 + 2/3.

Reading of it:
- None of the layers built on today's network brings ADDED answers to 98% across sets. The network's misreadings are
  systematic: it reads the related questions the same wrong way (same network, same text), so consistency can't catch
  them; it only confirms what is already right (as a veto it removed only right answers).
- The priority veto removes mostly right answers: the norms (actor / act / object from the lexicon tagger) are too coarse.
  The "defeasible no" from unexcepted bans is wrong on NDAs (3/20, 3/8): exceptions live in clauses the extractor
  doesn't link (definitions, "Representatives").
- The learned vetoes found the CUAD category problem on the tuning contracts (28 of 37 removed were wrong) but didn't
  generalize: on the held-out contracts 3 of 16 removed were wrong. The default+exceptions ADD rule works on short texts
  (+106/111 = 95.5% held-out), below the bar.
- Idea 5 adds ~nothing: first-person questions rarely get the same answer for every party; splits are rare.
- What holds up at ~96-97%: consistent "yes" just under today's short-text threshold (0.90-0.935) and consistent "no" on
  short texts. Not 98-99%.

## The object model + query language (oo.py, draft; tests in test_typed.py)
Eight core classes are the compiler's output:
| class | what it holds | built from |
|---|---|---|
| Contract (subclass NDA) | kind, title, parties, sections, clauses, terms, references, norms; ask(question) | doccompile + ns |
| Party | name, role ("Licensee"), how the preamble introduces it | doccompile parties |
| Section | number ("12.1(a)"), heading, span, parent | doccompile sections |
| Clause (GoverningLawClause, AssignmentClause, LiabilityCapClause, RenewalClause) | a section as a clause type; answer(query) | registry by clause type |
| Norm | permission / obligation / prohibition: holders, acts, objects, overrides, subject_to, has_exception | ns.DocNorms |
| DefinedTerm | name, definition | doccompile symbols |
| Reference | section / exhibit / other document, resolved target | doccompile xrefs |
| Answer | value, evidence, via (which object answered), query, trace | every answer |
Query language (CQL), the interface questions compile into:
```
MAY(actor="Licensee", act=ASSIGN, object="this Agreement")     MUST(...)     MUST_NOT(...)
HAS(clause="Governing Law")     VALUE(clause="Governing Law", slot=law)     WHO(modality=MUST, act=PAY)
ANSWER(q="...")                 # the neural operator: whatever the forms above don't cover
```
compile_question(): Pre-Tier 0's frame -> MAY / MUST / MUST_NOT (actor, act concept, the most specific object);
clause-presence questions -> HAS(clause) by trigger patterns; anything else -> ANSWER(q). Execution: a specialized
object answers first (clause classes with only the cuadc variants >= 98% in CV: governing law head+core / agr+core,
anti-assignment head+canon, cap head+excl / head+amount, renewal head+auto; the NDA class with the cnli/queries_ir
answers for nda-3, 5, 8, 15, 16, 19); otherwise ANSWER runs the pipeline with idea 1's consistency check and idea 2's
priority veto. Every Answer carries its query and a trace, e.g.:
```
query: MAY(actor=RECEIVER, act=SHARE, object=EMPLOYEES)  -> NDA.nda-5 (cnli/queries_ir)
query: HAS(clause='Governing Law')                       -> GoverningLawClause[head+core]
query: MUST(actor=ANY, act=NOTIFY)                       -> ANSWER(net-located); related ban 0.03, para 0.97
```
End-to-end sample (demo_oo.py, tuning data): first run, the NDA methods answered 11 (8 right: 3 nda-7 questions were
matched to nda-5 because the query kept the generic object; fixed: the object is the most specific thing named).
After the fix (runs/demo_oo2.txt): wc1_open 9/9 (all ANSWER); 8 NDAs x 17: 29/31 (NDA methods 10/11: nda-15 4/4,
nda-5 2/3, nda-3 / 8 / 16 / 19 1/1 each; ANSWER 19/20); 10 CUAD contracts x 4 clause types: 22/22 (GoverningLawClause 7,
RenewalClause 1, ANSWER 14).

## Idea 4: the reader network fine-tuned on located text (no LLM)
Data (build_ft.py, 40,334 pairs): for each question, the pipeline's located text cut into the network's windows; the
window holding a gold evidence span gets the gold answer, the other located windows "doesn't settle it". ContractNLI:
284 NDAs in no tuning or held-out set (train after the first 200 of the shuffle + dev), x the 17 questions; CUAD: 160
"write" contracts x the LegalBench CUAD questions (yes / doesn't settle it); plus 12,000 rows of the network's own
training data. Training (reader_net/train.py, CPU, from the installed model, lr 1e-5, 1 epoch, 2,377 steps, 64 min):
validation 81.7% -> 90.0% (yes found 188 -> 247 of 306 windows; false "yes" on windows that don't settle it 68 -> 54).
Model: typed/models/ft1 (not installed). Its thresholds re-picked on the tuning sets (lowest >= 98% on located text,
>= 99.5% on short texts): located 0.951, short 0.948.
Paired comparison on the same held-out questions (runs/paired_ft.txt), today's rule with each network:
| set | today | fine-tuned | added (right) | dropped (were wrong) |
|---|---|---|---|---|
| short_ho (2,321) | 449 @ 100% | **508 @ 100%** | 73 (73) | 14 (0) |
| nda_ho (959) | 186 @ 92.5%, 14 wrong | **217 @ 95.4%, 10 wrong** | 44 (43) | 13 (5) |
| cuad_ho (950) | 164 @ 85.4% | 140 @ 87.9% | 3 (3) | 27 (7) |
| wc1_sealed (spent, 126) | 27 @ 100% | 22 @ 100% | 1 (1) | 6 (0) |
| tuning: nda / cuad / short / wc1_open | 297 / 281 / 298 / 8 | 375 / 212 / 336 / 8 | | |
With the fine-tuned network, ideas 1 / 2 / 3 again add nothing at the bar (runs/combos_ft.txt, marginal_ft.txt): "1+" on
nda_ho 245 @ 90.6%, on short_ho 537 @ 99.4%; the priority veto drops right answers; the learned vetoes don't fire.
Caveat: on whole commercial contracts (wc1) the fine-tuned network at the pooled located threshold answers fewer (22 vs
27, both 100%); that threshold is set mostly by NDA and CUAD tuning rows. A per-kind threshold needs user-style
whole-contract tuning data we don't have (wc1_open has 4 network rows).

## Conclusions
1. Idea 4 is the one that pays: a better-calibrated reader (labels deduced from existing gold, no LLM) adds 13% answers
   at 100% on held-out user questions over provisions and +31 answers with fewer errors on held-out NDAs.
2. Ideas 1, 2, 3 on top of either network don't move answers past the 98% bar: the consistency checks confirm only what
   the network already gets right (its misreadings are systematic), the norms are too coarse for priority vetoes or
   "defeasible no", and learned gates overfit (they did find the CUAD category weakness on tuning data).
3. Idea 5 adds ~nothing (first-person supervaluation rarely agrees across parties; two-action splits are rare).
4. The object model + query language works end to end as the interface; its value is structure (clause classes host
   the trusted per-type answerers; every answer has a query and a trace), not coverage by itself.
Next, by value: install the fine-tuned reader behind a switch (after a fresh sealed check, which needs LLM budget for a
new whole-contract set, or hand-labeled questions); grow the clause classes (each a trusted answerer + its tests);
consistency / priorities only once the rules extracted from contracts are precise enough (they are the weak part).

## XGBoost on top of the engine (2026-10-04; xgb_stack.py, isolated venv /root/zadumai_nli_proto/venv_xgb; runs/xgb_stack.txt)
Every saved signal per question -> yes / no / not_stated; trained on the tuning sets, thresholds from 5-fold out-of-fold
predictions (folds by document), scored on the held-out sets. With the retrained network's features, 3-way ("the text
doesn't say" allowed), threshold for 98% on tuning: short_ho 67.3% @ 99.3% (today's rule 21.9% @ 100%); cuad_ho 29.6% @
98.2% (14.7% @ 87.9%); nda_ho 30.0% @ 97.6% (22.6% @ 95.4%); at the 96% threshold: 78.5% @ 98.4%, 53.2% @ 97.4%,
46.8% @ 95.3%. It FAILS on whole-contract user questions (wc1_sealed 22-33% @ 63-71%): its "yes" answers are right,
its "no" answers are to questions the contract is silent on (gold not_stated here; CUAD's labels call absence "no").
The stacker learns each dataset's labeling convention for silence; it needs training data in the target question style.

## 100% coverage (2026-10-04; the user: "Let's try to achieve 100% coverage at 90%"; xgb_full.py, xgb_perq.py)
One XGBoost per domain answers EVERY question (the likeliest of yes / no / not_stated), trained on the domain's tuning
set, scored on its held-out set. Local only, no LLM. Retrained-network features unless noted.
| domain | held-out | right at 100% coverage | tuning CV | what got it there |
|---|---|---|---|---|
| single-clause user questions (v6b+v7) | 2,321 | **92.7%** | 91.7% | the engine's signals |
| CUAD checklist (38 clause types, 25 contracts) | 950 | **92.0%** | 89.0% | + question identity + a section-tagger presence score (tagger retrained without the 82 dev contracts: cuad_presence.py) |
| NDA checklist (17 ContractNLI questions) | 959 | 87.7% | 87.6% | + question identity + NDA compiler answers, topic hits, out-of-fold evidence-span scores (nda_extra*.py) |
| whole-contract user questions (wc1) | 126 | 81.0% | - | trained on wc1_open's 59 questions only |
Reference: ContractNLI paper, fine-tuned BERT-large 87.5% / BERT-base 83.8% (3-way, its test set).
NDA levers tried: hyperparameters (CV 87.6-87.9%, flat); 384 training NDAs instead of 100 (installed network: 86.8% ->
87.5%); separate models per question (user's question): worse than one model with the question's identity (100 NDAs
85.4% vs 86.8%; 384 NDAs 86.7% vs 87.5%), better on a few questions, worse on more.
Pooled across domains (one model, domain as a feature) was worse: wc1 47.6% (it learned CUAD's "absence = no").

## NDA to 90% at 100% coverage (2026-10-04, night; the user: "We want neuro symbolic approach", then Jev, then a tiny encoder)
All on the NDA checklist (17 ContractNLI questions); training = 384 NDAs (nda_tune + nda_more), held-out = nda_ho (60
NDAs, 1,020 questions). Sealed and never used: ndaenc/data/s160 (40 unused training NDAs) and ContractNLI test (123).
1. Symbolic detectors per weak question (nda_rules.py; nda-1 marking, nda-2 technical-only, nda-7 third parties,
   nda-16 return trigger, nda-20 retention), gated to their own question (nda_eval.py): held-out 87.45% -> 87.75-88.14%,
   CV +0.1-0.3. Each detector has signal on its own (e.g. "all copies returned, nothing kept" -> gold no 77/92) but
   the stacker already has most of it from the network.
2. Label-noise audit, 24 random training errors of the stacker read by hand: ~4 gold mistakes (e.g. 405/nda-1 gold
   "must be identified" on "regardless of whether identified as confidential"), 2 debatable, ~18 ours -> the ceiling
   is well above 90%. Our errors cluster on conventions: "no disclosure to third parties without consent" = NO for
   nda-7; a broad definition / "whether marked or not" = NO for nda-1; "return all copies" = NO for nda-20.
3. Jev (systemone, jev-1.13.0): ONE request per NDA, the whole NDA as state, the 17 questions as 17 Nouls + 17 3-way
   Choices (jev_nda.py; < 0.5 s per request; ~4.4k tokens in, ~1k out). TypeSafe has no multi-label type: its docs say
   one Noul per label in one request, which is this. 266 Jev calls in all (100 tune + 60 held-out + v2 100 + 6 probes).
   | 100 tuning NDAs (1,700 q), out-of-fold | 3-way |
   |---|---|
   | Jev's choice alone | 82.3% (binary p(true) on gold yes/no: 92.2%) |
   | v2: Nouls "says yes" / "says no" on ContractNLI's statement (jev_nda2.py) | 82.2% |
   | our stacker | 87.1% |
   | stacker + Jev v1 numbers | **90.0%** |
   | stacker + Jev v2 | 88.5% (v2 wins nda-1 / nda-16, breaks the double-negative nda-15) |
   | stacker + v1 + v2 | 89.9% |
   HELD-OUT (jev_ho.py, 60 NDAs, Jev for all; the stacker learned Jev's numbers from 100 of its 384 NDAs):
   Jev alone 81.37%; stacker 87.45% (bootstrap over NDAs 84.9-90.0); **stacker + Jev 90.98% (88.8-93.0)**; 67 answers
   changed, 50 fixed, 14 broken. nda-1 73->87, nda-7 68->80, nda-16 82->90, nda-20 83->90; only nda-3 lower (90->88).
4. The NDA statement encoder (ndaenc/; the user's idea: a tiny encoder, fully fine-tuned, only for the 17 questions):
   DeBERTa-v3-xsmall from the reader network, all weights, a 17 x 3 head on ONE compiled statement (labels from
   ContractNLI's evidence spans: 24,161 statements, every yes/no answer has >= 1), 4-fold cross-fitting + a full model,
   in parallel on the 8 cores (sched.py). Documents are decided by rules (aggregate.py: strongest statement, threshold
   per question, yes/no precedence per question) or by the stacker with its features (nda_eval2.py).
   Training: 5 runs, ~2.5 h on CPU. Alone (rules, aggregate.py): out-of-fold 78.8%, held-out 80.8%. In the stacker:
   CV 87.15 -> 87.35%, held-out 87.45 -> 88.43%; with Jev it adds nothing (90.78% vs 90.98%). Not the road to 90%.
5. The NDA CNN (ndacnn/; the user's ideas: very low-dimensional embeddings + convolutional kernels; dilated kernels side by
   side; stacked rising / falling / up-then-down; inter-sentence kernels). Reader-network tokens -> 64-d vectors (its
   word embeddings, PCA; trained further) -> kernels over the WHOLE NDA (no window; 3.1% of NDAs > 5k tokens, max 11k)
   -> statements (max + mean of their tokens) -> [graph kernels over the compiler's links: next statement, same section,
   cross-reference, defined term -> users] -> attention per question over statements (trained on ContractNLI's evidence
   statements) -> yes / no / not stated. 96 channels, 12 epochs, 1.5-14 min per run on 2 threads (sched.py: 42 runs).
   | CNN (5-fold out-of-fold / held-out) | alone | in the stacker | 
   |---|---|---|
   | plain (one 9-token kernel) | 83.66 / 84.22 | 87.96 / 89.22 |
   | parallel: dilations 1..16 side by side | 81.33 / 80.49 | 87.55 / 88.14 |
   | rising 1, 2, 4 .. 64 stacked | **85.14 / 85.29** | **88.33 / 89.02** |
   | falling 64 .. 1 | 82.95 / 82.55 | 87.82 / 88.82 |
   | up-then-down 1 .. 64 .. 1 | 84.93 / 85.69 | 88.13 / 88.92 |
   | rising + graph kernels | 85.03 / 85.20 | 88.11 / 88.43 |
   | rising + graph + two slots (rule / exception) | 85.05 / 86.18 | 88.07 / 87.75 |
   As predicted: sparse side-by-side dilations are brittle on text; bottom-up (rising) beats top-down (falling); the
   graph kernels and two-slot pooling don't help at this data size. The CNN alone beats Jev alone (81.4%) and the
   encoder (80.8%), trains in minutes and reads a whole NDA in ~0.05 s (estimated from training speed).
6. Final table (final_eval.py, runs/final_eval.txt; CV with Jev is handicapped: Jev asked on 100 of the 384 NDAs):
   | config | training CV | held-out (95% bootstrap) |
   |---|---|---|
   | stacker (start of the night) | 87.15% | 87.45% (84.9-90.0) |
   | LOCAL: + CNN rising + encoder | 88.60% | 89.31% (86.7-91.9) |
   | LOCAL: + 3 CNNs (rising, up-down, plain) + encoder + detectors | 88.43% | 89.41% (86.7-91.9) |
   | + Jev | 87.91% | 90.98% (88.8-93.0) |
   | + CNN up-down + Jev | 88.45% | 91.08% (88.9-93.0) |
   | + 3 CNNs + Jev | 89.12% | 90.98% (88.7-93.1) |
   NDAs over 5k tokens are not a weak spot in any config (held-out 92.6-97.1% on the 4 such NDAs).
   Local: +2 points (87.45 -> 89.3-89.4% held-out), not yet 90%. With one Jev call per NDA: ~91%.
7. Jev on ALL 384 training NDAs (2026-10-04 10:24 PT, 284 more calls, 8 s; 550 Jev calls today), so the stacker learns
   how far to trust Jev from every NDA, not 100 (final_eval.py):
   | config | training CV | held-out (95% bootstrap) |
   |---|---|---|
   | stacker + Jev | 90.53% | 92.06% (90.1-93.8) |
   | + CNN rising + Jev | 90.82% | 91.86% (89.8-93.9) |
   | + CNN up-down + Jev | 90.72% | **92.55% (90.5-94.4)** |
   | + CNN rising + encoder + Jev | 90.81% | 92.45% (90.3-94.5) |
   | + 3 CNNs + encoder + Jev + detectors | **90.84%** | 92.45% (90.3-94.5) |
   With one Jev call per NDA: ~92.5% held-out, ~90.8% CV; the bootstrap's lower end is above 90%.
   Next (done, below): the CNN taught by Jev (its answers on the training NDAs as an extra head / mixed into the targets)
   and by the detectors (a head predicting their flags): local at run time (ndacnn/sched_teach.py).
8. Latency for ONE NDA x 17 questions (median NDA 231, 1,884 tokens, 4 threads, other jobs paused; scratchpad
   latency.py): engine features (Pre-Tier 0 + located reading + reader network + related questions) 24.5 s; statement
   encoder 3.6 s; NDA compiler facts (spaCy) + queries + topic words 0.17 s; regex detectors 0.01 s; CNN tokens +
   compile + graph 0.01 s, forward 0.01 s; Jev one request 0.09 s; XGBoost < 1 ms. (TF-IDF span models and the old NDA
   engine's answers not timed; light.) -> FAST configs without the engine ("cheap" = question id + NDA compiler extras):
   | config (final_eval.py) | training CV | held-out | ~latency |
   |---|---|---|---|
   | base + Jev (engine) | 90.53% | 92.06% | 28 s |
   | cheap + Jev | 90.70% | **92.25% (90.4-94.0)** | 0.3 s |
   | cheap + CNN rising + Jev | **91.05%** | 91.67% | 0.3 s |
   | cheap (local) | 86.47% | 87.45% | 0.2 s |
   | cheap + CNN rising (local) | 87.97% | 88.24% | 0.2 s |
   The engine's features add nothing once Jev is there, and ~0.4 pt locally.
9. The CNN taught by Jev / the detectors (ndacnn train.py --jev head|mix --regex; Jev's answers on the training NDAs
   only; nothing called at run time). Alone (out-of-fold / held-out): rising 85.14 / 85.29; + Jev head 85.69 / 86.27;
   + Jev mix 85.14 / 86.47; + regex 84.74 / 85.10; + Jev mix + regex 84.60 / 87.25. In the stacker, LOCAL and fast:
   cheap + CNN rising 87.97 / 88.24 -> cheap + CNN taught by Jev (head) 88.50 / 88.92; best local:
   cheap + CNNs (Jev head, Jev mix, up-down, plain) + regex detectors: CV 88.45%, held-out 89.90% (87.4-92.3), ~0.2 s.
   Teaching on the same 384 NDAs: +0.5 pt, as expected. Not yet 90% locally on CV.
10. CNN architecture round (2026-10-04, 11:30-13:10 PT; the user's ideas; CPU + a new Runpod RTX 4090 pod
    nu7pb7bcpo78qh created via RUNPOD_API_KEY, 30 runs at once, ~10 min; pod STOPPED after). CNN alone, 5-fold
    out-of-fold / held-out (ndacnn/report_k.py):
    | variant | OOF | HO |
    |---|---|---|
    | rising, width 13 / 9 / 5 / 3 | 84.74 / 85.14 / 86.00 / 86.23 | 84.90 / 85.29 / 86.76 / 87.35 |
    | rising width 5, 128-d word vectors (64-d keep 34% of the variance, 128-d 53%) | **86.96** | **87.84** |
    | rising width 3, 128-d | 86.93 | 87.25 |
    | rising width 5, 128-d, taught by Jev | 86.96 | 86.37 |
    | plain (1 layer) / plain width 13 / plain3 (3 layers 13-9-5) | 83.66 / 83.64 / 83.62 | 84.22 / 83.82 / 83.43 |
    | pyramid (DPCNN, 6 x [pool/2 -> conv]) | 85.06 | 86.08 |
    | clause starts marked (rising w5 --sep) | 85.68 | 85.49 |
    | per-clause kernels (folds 0+1 only, vs rising 84.78 / rising w5 84.66 there): clause 84.18, clause + 4 clause-level layers 82.55, clause -> whole NDA 83.31 | | |
    Learning curve (rising w9): 25/50/75/100% of the training NDAs -> 75.7 / 81.2 / 83.8 / 85.1% (power-law fit:
    2x data ~87.9%, 4x ~89.8% CNN alone). Narrow kernels + deep dilated stacks + 128-d vectors win; isolating clauses,
    wider kernels, extra clause-level layers lose.
    In the stacker the stronger CNNs add little (final_eval.py): LOCAL fast cheap + 6 CNNs + regex CV 88.68%, HO 89.71%
    (was 88.45 / 89.90); with Jev: cheap + CNN (rising w5 128-d) + Jev CV 90.95%, HO **92.65% (90.6-94.5)**.
    Local plateau ~88.7% CV: the remaining lever is more training data (step 2) or Jev at run time.
11. Step 2: more training NDAs from SEC (2026-10-04 13:00-14:00 PT; ndamore/collect.py, label.py; ndacnn train.py
    --extra). SEC full-text search (user's email in the User-Agent, approved): 11,477 exhibit hits -> 7,870 fetched ->
    476 stand-alone NDAs (title or "Evaluation Material" near the top; 158 overlapping ContractNLI and 357 duplicates
    dropped) -> Jev "is this an NDA?" >= 0.5: 407. Teacher = lite (qid + IR + topic) + CNNs rising 128-d width 5 / 3 +
    Jev (CV 90.92%, HO 93.04%), one per fold (never saw that fold's NDAs). 476 Jev calls (1,026 today in all).
    Student CNN (rising, width 5, 128-d; 384 gold + 407 soft-labeled NDAs): alone OOF 86.96 -> **88.11%**, HO 87.84 ->
    88.73% (no fold worse). In the stacker, LOCAL (no Jev at run time, ~0.3 s):
    | config | training CV | held-out |
    |---|---|---|
    | cheap + 6 CNNs + regex (before) | 88.68% | 89.71% |
    | cheap + student CNN | 88.91% | **90.10% (87.7-92.5)** |
    | cheap + student + CNN width 3 + regex | **89.02%** | 89.51% |
    | base (slow engine) + all CNNs + encoder + regex | 89.45% | 89.61% |
    With Jev: lite + student + CNN w3 + Jev CV 90.76%, HO 92.35% (no gain over 90.95 / 92.65 before).
    More data helped the CNN by ~1 pt but the local stacker by only ~0.3 pt: local ~89% CV / ~90% held-out.
12. FINAL, 2026-10-04 ~16:00 PT. Round 2 of SEC NDAs (wider search, 18,284 hits, 12,063 fetched): 573 candidates, 438
    confirmed NDAs (+31; 97 more Jev calls). Final CNN: rising 7-layer, width 5, 128-d, trained on 384 + 438 SEC NDAs,
    5 seeds averaged (models in ndacnn/models/rising+secallw1.0+d128+k5+r2*_f-1.pt). 14 layers (rising2, 3 seeds): OOF
    88.27% vs 88.77% for 7 layers with 3 seeds -> not used. Before the test: CNN alone CV 89.02% / HO 89.61%; lite + CNN
    89.15 / 89.41; lite + CNN + Jev 90.85 / 91.96.
    ONE-TIME SEALED TEST (final_test.py; runs/final_test.json, runs/final_legalbench.json):
    | system | ContractNLI official test (123 NDAs, 2,091 pairs) | s160 (40 unused NDAs, 680) |
    |---|---|---|
    | CNN alone | **87.47%** (86.1-88.8), F1 E 0.903 / C 0.706 | 90.15% |
    | local stacker (lite + CNN) | 87.52%, F1 E 0.908 / C 0.688 | 89.85% |
    | + Jev | **89.05%** (87.7-90.3), F1 E 0.923 / C 0.732 | 91.47% |
    Published on the same data: ContractNLI paper BERT-large 87.5%; SCROLLS ContractNLI CoLT5-XL 88.4 EM; LLMs zero/few-shot
    46-78% (setups vary). LegalBench contract_nli (14 tasks, 1,927 excerpts from ContractNLI test), CNN alone: mean accuracy
    91.8%, mean balanced accuracy 90.8% (weakest: explicit identification 75.5%, return of information 81.2%).
    The test split scores ~1.5 pt below CV (s160, same split family as training, matches CV). Weak spot: Contradiction F1.
13. After the sealed test (2026-10-04/05; judged by CV + held-out only). Per-question calibration of the CNN's classes
    (ndacnn/calibrate.py, nested): OOF 89.02 -> 88.43%, held-out unchanged -> dropped. The user's token -> sentence ->
    clause stack (ndacnn/stmt_feats.py: bge-small sentence vector per clause, encoded on the GPU in 86 s for 107k clauses;
    TF-IDF 1-2-grams -> SVD 256 at three thresholds) and wider CNNs, 3 seeds each, rising width 5, 128-d, round-2 data
    (report_x.py):
    | variant | 3-seed OOF | held-out | single seeds OOF |
    |---|---|---|---|
    | baseline 96 channels | 88.77% | 89.80% | 87.4-88.0 |
    | **192 channels** | **89.52%** | **90.39%** | **89.0-89.2** |
    | + bge | 89.05% | 88.73% | 87.8-88.1 |
    | + TF-IDF loose (min_df 2 / max_df 0.9) | 88.69% | 88.92% | 87.8-88.0 |
    | + TF-IDF moderate (5 / 0.7) | 88.62% | 89.61% | 87.4-88.0 |
    | + TF-IDF tight (20 / 0.5) | 89.00% | 89.22% | 88.3-88.4 |
    | + bge + TF-IDF moderate | 88.82% | 89.12% | 87.9-88.6 |
    | + both + a kernel across clauses | 88.97% | 90.59% | 87.7-88.4 |
    Only width helps (every 192-channel seed beats every 96-channel seed); the extra clause views are within noise.
    Running: 256 and 384 channels (tags r2c256 / r2c384), then the winner to 5 seeds.
    Width, 3 seeds each: 96 ch 88.77% OOF / 89.80% HO; 192 ch 89.52 / 90.39; 256 ch 89.31 / 90.98; **384 ch 89.72 / 90.49**
    (single seeds 89.1-89.5). Speed per NDA, one model, 4 threads: 96 ch 8 ms, 192 ch 21 ms, 384 ch 66 ms (9.6M params).
    NEW FINAL CNN: 384 channels, 5 seeds averaged (models ndacnn/models/rising+secallw1.0+d128+k5+r2c384*_f-1.pt, ~0.33 s
    per NDA for all 5): alone CV 89.64% / HO 90.49%; lite + CNN CV **89.95%** / HO 90.78%; cheap + CNN 89.89 / 91.67;
    lite + CNN + Jev CV 91.27% / HO 92.65%. (Previous final, 96 ch: alone 89.02 / 89.61; + Jev 90.85 / 91.96.)
14. The user's next ideas on the 384-channel CNN (2026-10-05; 3 seeds each, out-of-fold / held-out; a second Runpod
    pod, stopped after): bge-large-en-v1.5 clause vectors (1024-d, encoded on the GPU in 150 s), a whole-NDA vector
    (the mean of the clause vectors, joined to every question's decision; train.py --doc-vec), and both.
    | variant | 3-seed OOF | held-out | single seeds OOF |
    |---|---|---|---|
    | 384 channels (final) | **89.72%** | 90.49% | 89.1-89.5 |
    | + bge-large clause vectors | 89.58% | 90.49% | 89.1-89.2 |
    | + document vector | 89.66% | 89.22% | 88.4-89.3 |
    | + both | 89.26% | 90.39% | 88.5-89.1 |
    None helps (all within seed noise, "both" slightly worse). A 2D 3x9 kernel over clauses x tokens was not tried
    (token positions in neighbouring clauses don't align; cross-clause kernels already failed). The 384-channel CNN
    stays final. The CNN appears data-limited, not architecture-limited.
