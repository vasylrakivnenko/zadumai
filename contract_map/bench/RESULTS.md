# Clause / contract-type classification benchmark — stage 1 (2026-10-02)

> **Review notes (2026-10-06)** — discrepancies found in a review; the text below is unchanged unless noted.
>
> - "fetch notes there" (MCC data): the only fetch script is `data/ext/mcc/fetch_mcc.py`.
> - The lease encoder's 0.38-0.40 here and the deployed encoder's 0.278 (`../build_models.log`) are on different eval
>   sets; they are not a regression.

bench.py (sets, methods, stages), learning_curve.py; logs + error files in out/. Jev = jev-1.13.0 `choice`, zero-shot.

| set (ground truth) | eval rows | TF-IDF + LR | bge-small + LR | Jev, all classes | cascade (TF-IDF top 5 → Jev) |
|---|---|---|---|---|---|
| LEDGAR, 100 clause types (train 6k) | 500 | 84.0% acc (≥0.9 conf: 96.9% on 31.8%) | 82.4% | 73.8% (≥0.9: 89.8% on 64.4%) | stage 2 |
| CUAD paragraphs, 37 types, multi-label | 500 (300 typed) | F1 0.638 (P 74.8 R 55.6) | F1 0.528 | F1 0.618 (P 79.1 R 50.7) | stage 2 |
| UNFAIR-ToS, 8 types, multi-label | 500 (172 typed) | F1 0.793 (P 83.8 R 75.3) | F1 0.728 | F1 0.715 (P 63.2 R 82.3) | F1 0.739 (P 65.7 R 84.4) |
| CUAD contract type, 25 types, first 4k chars | 200 | 72.0% | 66.5% | 89.5% | — |
| same, title cut (chars 2k–6k) | 200 | 52.0% | 51.5% | 65.0% | — |
Title rule (one type named in the first 300 chars): answers 69% of contracts, 97.4% right.
Speed per item: TF-IDF 0.1–0.3 ms; bge-small 30–60 ms (contracts ~900 ms); Jev p50 ~70 ms, p95 ~120–135 ms,
0 errors in 2,400 calls; ~2,000 input tokens per call with 100 classes, ~540 with 8.

Learning curves (TF-IDF + LR, 2,000 eval rows):
- LEDGAR: 1k 70.0% · 3k 77.3% · 6k 81.2% · 20k 84.7% · 60k 87.9% (merged near-synonym labels 91.8%; ≥0.9 conf: 97.9% on 63%)
- CUAD paragraphs: 900 typed 0.604 · 1.8k 0.650 · 3.6k 0.674 · 4.6k (all) 0.684 · + all untyped 0.669 → out of labels
- UNFAIR-ToS (all 1,607 test rows): 600 typed 0.695 · 630 (all) 0.723 · + all untyped 0.739 → out of labels

Stage-1 bugs caught: CUAD training sample had the natural mix (13% typed) → recall 24% (fixed: F1 0.38 → 0.64);
embeddings re-encoded per fold (cached); labels of passages beyond the 4,000-char cut (dropped).
Caveats: LEDGAR labels = each contract's own heading (near-synonyms are separate classes); UNFAIR-ToS marks only
*unfair* clauses, so topically right tags count as errors (about half of Jev's 75 false tags looked right).

# Stage 2 (2,000 rows; contract types and ToS = all rows) and stage 3 (TF-IDF only, full test sets)
| set | TF-IDF + LR | Jev, all classes | cascade |
|---|---|---|---|
| LEDGAR 2,000 (TF-IDF train 20k) | 84.7% (merged 89.4%; ≥0.9: 97.5% on 48.9%) | 74.6% (merged 82.7%; ≥0.9: 89.3% on 65.8%) | 75.9% (83.7%) |
| LEDGAR full 10,000 (train 60k), stage 3 | 87.7% (merged 91.0%; ≥0.9: 97.4% on 61.0%) | — | — |
| CUAD paragraphs 2,000 (1,115 typed) | F1 0.678 (P 74.0 R 62.5) | F1 0.636 (P 77.9 R 53.7) | F1 0.626 (P 79.8 R 51.5) |
| CUAD paragraphs full 8,838, natural mix (1,115 typed), stage 3 | F1 0.590 (P 55.6 R 62.9) | — | — |
| UNFAIR-ToS all 1,607, natural mix (172 typed) | F1 0.739 (P 72.5 R 75.3) | F1 0.430 (P 29.1 R 82.3) | F1 0.447 (P 30.4 R 84.4) |
| contract type, all 508, first 4k chars | 74.2% | 89.4% (≥0.9: 94.1% on 86.2%) | — |
| contract type, title cut | 54.9% | 66.1% | — |
Jev stage 2: 9,830 calls, 0 errors, p50 ~70 ms, p95 ~120–140 ms; ~10.7M input / ~2.6M output tokens.
Stage 1 → 2 changes were within noise; ToS precision fell on the natural mix (1,435 untyped rows: Jev tagged 358).

# New datasets (2026-10-02 evening): stage 1 (500) then stage 2 (2,000 or all)
Data in ../data/ext/ (fetch notes there). Not available: LEXDEMOD (repo gone), Braun & Matthes 2022 EN/DE set (not public). German sets (AGB-DE,
German employment clauses) were tried and deleted (user: US only). CFPB credit card agreements (2.3 GB of PDFs) not used yet.
| set | ground truth | eval rows | TF-IDF + LR | Jev |
|---|---|---|---|---|
| opp115 | OPP-115 privacy-policy segments, 9 categories (experts) | 763 (all) | F1 0.769 (P 78.7 R 75.2) | F1 0.741 (P 89.0 R 63.5) |
| contractnli | NDA passages × 17 NDA questions (experts) | 2,000 (1,200 typed) | F1 0.744 | F1 0.584 |
| lease | lease paragraphs × 8 term types + red flag (Leivaditi) | 2,000 (310 typed, natural mix) | F1 0.275 (stage 1, 60% typed: 0.443) | F1 0.295 (stage 1: 0.518) |
| doctype | whole documents, 7 types (CUAD / NDA / privacy / ToS / cookie / EULA / lease) | 500 | 93.2% first 4k chars (≥0.9: 99.0% on 60%); 88.6% title cut | 93.8%; 88.2% |
| mcc | MCC contracts, 8 types (silver: MCC's model labels) | 500 | 84.8% (≥0.9: 99.3% on 29%); 82.6% title cut | 78.2%; 76.0% |
Jev this round: ~7,700 calls, 0 errors, p50 ~70-85 ms. Runs in parallel (6 processes, one-vs-rest on 2 cores each).

# One model on all clause datasets vs one per dataset (combined.py, TF-IDF + one-vs-rest LR in both)
31 shared classes via a label map (eq = same meaning: positives + negatives; sub = narrower: positives only).
| dataset | stage 2 separate → combined | stage 3 (all rows) separate → combined |
|---|---|---|
| LEDGAR (acc, merged labels) | 89.5% → 89.4% | 90.1% → 90.2% |
| CUAD paragraphs (F1) | 0.678 → 0.672 | 0.590 → 0.574 |
| UNFAIR-ToS | 0.739 → 0.716 | 0.739 → 0.721 |
| OPP-115 | 0.769 → 0.768 | 0.769 → 0.768 |
| ContractNLI | 0.744 → 0.748 | 0.682 → 0.681 |
| lease | 0.275 → 0.279 | 0.213 → 0.231 |
Shared classes: rare ones gain (CUAD non-disparagement 0.17 → 0.25–0.29, change of control +0.04 at stage 2);
classes whose "same" label means different things across sources lose (CUAD insurance 0.61 → 0.47 at stage 3:
LEDGAR's insurance clauses are lender covenants; OPP-115 policy change 0.80 → 0.73 with ToS unilateral changes).

# Shared taxonomy + per-kind models + router (family.py, taxonomy.py, jev_checks.py; 2026-10-02 evening)
Taxonomy: ../taxonomy.json — 146 clause types (31 shared by 2+ datasets, 11 span 2+ document kinds), 50 linked
to a FOLIO clause class (45 by exact name, 5 approximate), 25 rare (< 100 training examples).
Router (TF-IDF, first 4,000 chars → commercial / NDA / privacy policy / terms of service / lease / not a contract;
trained on 3,294 docs that are in no clause test set): 95.1% on 1,156 held-out docs (≥0.9 conf: 99.1% on 75%);
clause test docs 97–100% (CUAD 99%, NDA 98%, OPP 100%, lease 97%). Errors: MCC "not a contract" → commercial,
ToS ↔ privacy. Jev on 500 docs: 92.6% (TF-IDF 94.4%); hybrid TF-IDF if p ≥ 0.7 else Jev: 95.4% (Jev reads 7%).
Clause models, all test rows (stage 3):
| dataset | own model | per kind | kind + borrowing | routed end to end |
|---|---|---|---|---|
| LEDGAR (acc) | 90.1% | 90.3% (with CUAD) | 90.2% | — (no documents) |
| CUAD | 0.590 | 0.562 (with LEDGAR) | 0.560 | 0.560 (99.6% rows routed right) |
| UNFAIR-ToS | 0.739 | 0.739 | 0.740 | — (no documents) |
| OPP-115 | 0.769 | 0.769 | 0.769 | 0.769 |
| ContractNLI | 0.682 | 0.682 | 0.682 | 0.679 |
| lease | 0.213 | 0.213 | 0.234 | 0.237 |
Borrowing (rare class, < 100 positives, gets ≤ 300 positives from another kind): lease VAT 0.17 → 0.26; ToS
arbitration 0.93 → 0.74, lease renewal 0.35 → 0.25. Only 8 rare classes have any source to borrow from.
Hybrid clause tagging, 1,000 rows per set, Jev answers from the stage-2 cache (no new calls): TF-IDF when sure,
Jev for the unsure band: CUAD 0.673 → 0.689 (Jev 11%), ToS 0.723 → 0.741 (5%), OPP 0.769 → 0.786 (16%);
LEDGAR, NDA flat; lease worse.
Encoder (MiniLM-L6 fine-tuned, masked multi-label), stage 1 (3k rows/dataset, 1 epoch): combined beats separate
everywhere (LEDGAR 55% vs 19%, CUAD 0.41 vs 0.16, NDA 0.59 vs 0.47, ToS 0.62 vs 0.56, OPP 0.62 vs 0.61, lease
0.52 vs 0.50) but is far below TF-IDF except on lease. Stage 2 below.
Encoder stage 2 (10k rows/dataset, 2 epochs, 2,000 eval rows; TF-IDF = own model on the same rows, all training data):
| dataset | encoder combined | encoder separate | TF-IDF |
|---|---|---|---|
| LEDGAR (acc) | 82.5% | 81.8% | 89.5% |
| CUAD | 0.559 | 0.501 | 0.678 |
| UNFAIR-ToS | 0.675 | 0.597 | 0.739 |
| OPP-115 | 0.746 | 0.745 | 0.769 |
| ContractNLI | 0.747 | 0.730 | 0.744 |
| lease | 0.384 | 0.399 | 0.275 |
Combining helps the encoder (CUAD +0.06, ToS +0.08) though not TF-IDF; the encoder beats TF-IDF only on lease
(+0.11) and ties on NDA. CPU-only (~35 rows/s); more epochs or a GPU might close the rest — untested.

## Recommended setup (from all of the above)
1. Route the document (TF-IDF router; Jev for p < 0.7): 95.4%.
2. Tag clauses with that kind's own TF-IDF model (LEDGAR and CUAD stay separate models); send the unsure band to
   Jev for CUAD / ToS / privacy (+0.015–0.02 F1, 5–16% of rows). Leases: the fine-tuned encoder (0.38–0.40 vs 0.28).
3. Don't borrow examples across kinds; get more labels for leases, employment and insurance instead.
