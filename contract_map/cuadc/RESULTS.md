# CUAD clause compiler (2026-10-03): does the NDA compiler approach generalize?

> **Review notes (2026-10-06)** — discrepancies found in a review; the text below is unchanged unless noted.
>
> - The numbers come from the `front.py` front end, which (doccompile/RESULTS.md) misreads EDGAR footers in
>   141 / 408 contracts; they were not re-run after that fix. The sealed test is still not run.


The user (2026-10-03): "Let's try Prove it generalizes: port the compiler to a few high-value CUAD clause types".

## Protocol (fixed before any rule was written)
- Types (6): Governing Law, Anti-Assignment, Cap On Liability, Termination For Convenience, Renewal Term, Change Of Control.
- test = CUAD's own 102 test contracts: SEALED (data.test_docs(), only a final.py may read them, once).
- tune = the other 408: write 326 (rules written from these only, explore.py/show.py/score.py) + dev 82 (seed 0; never read).
- Nested 5-fold CV over the 408 (folds random.Random(1)); a source is trusted per type if >= target right on the
  training folds with >= 15 answers.
- Right: "no" when CUAD has no span; "yes" when CUAD has a span and our evidence overlaps one. Grain "sentence" = our
  statement overlaps; grain "paragraph" = the statement's block (from its lead-in; at most 600 chars each side) overlaps.

## Pieces
- front.py: the NDA compiler's structure parser (cnli/compiler.py) + character offsets + section headings (heading-only
  blocks and inline "12.1 Assignment. ..." headings) + paragraph spans; lead-ins split into sentences and located.
- clauses.py: per type graded variants (head+core / agr+core / canon / void / strong / free / auto / ...), plus
  "no_mention" (nothing on the topic anywhere) and "no_rule".
- cv.py: compiler-only gating. llm_baseline.py: gpt-oss-120b, one request per contract, six questions, quote.
  cv2.py: compiler + LLM sources incl. agreement ("both:<variant>").

## Results, nested 5-fold CV over the 408 tuning contracts (2026-10-03 ~11:05 PT), paragraph grain
| system | answered | right | dev (never read) |
|---|---|---|---|
| LLM alone (gpt-oss-120b, answers everything) | 100% | 86.3% | 88.8%-89.2% |
| compiler alone, target 98% | 13.9% | 96.8% | 12.6% @ 98.4% |
| compiler + LLM agreement, target 98% | 47.4% | 98.3% [lo 0.974] | 46.7% @ 99.1% |
| compiler + LLM agreement, target 95% | 64.0% | 97.7% | 63.8% @ 97.8% |
Per type at 98%: Governing Law 86.8% @ 99.2%, Anti-Assignment 50.2% @ 98.0%, Cap On Liability 27.7% @ 100%,
Termination For Convenience 10.3% @ 92.9%, Renewal Term 49.8% @ 96.1%, Change Of Control 59.8% @ 99.2%.
- The LLM alone scores 95.4% on the rows the combined system answers and 78.1% on the rest: the compiler is mostly a
  selector of trustworthy LLM answers (plus the located evidence); on Cap On Liability it corrects the LLM (75% -> 100%).
- Verified 47% + LLM for the rest = 87.7% overall vs 86.3% LLM alone; the gain is knowing which half needs no review.
- Best compiler variants on the write contracts: GL agr+core 97.9%, AA head+canon 97.7%, Cap head+excl/head+amount 100%,
  Renewal head+auto 97.6%, TfC strong 92.6%, CoC coc_words 79% (CUAD labels the same merger-assignment template both ways).
- Sealed test: final.py written, NOT run (waiting for the user).
