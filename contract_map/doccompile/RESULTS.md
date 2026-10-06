# Document compiler (doccompile.py), 2026-10-03

> **Review notes (2026-10-06)** — discrepancies found in a review; the text below is unchanged unless noted.
>
> - `test_doccompile.py` has 9 tests (the text says 8).
> - "Next: router port": done as `../typed/pt0_engine.patch` (plan in ../typed/PORT.md, not applied); the tagger retrain
>   mentioned there was never done.
> - Typed-question shares (65.9% / 75.8%, 1,182 unknown actors, 58 two-action questions) differ from
>   ../typed/RESULTS.md (76.4%, 1,145, 59) on the same 6,227 questions: different scripts, not a contradiction.


The user asked for one compiler pass shared by Pre-Tier 0 and the contract map ("Sure. Let's build it!", after the
recommendation to start with the footer fix, `compile.py` and the section-tagging comparison). Research folder first;
nothing in the router yet. CUAD's 102 test contracts stay sealed (all numbers: CUAD's 408 tuning contracts, ContractNLI
train + dev 484 NDAs).

## What it does (doccompile.py, self-contained; tests: test_doccompile.py, 8 pass)
1. preprocess, offsets unchanged (same length as the input):
   - PDF line breaks: a run of 3+ spaces becomes a line break. CUAD keeps a whole page on one line with "   " between
     paragraphs, so the old front end (cuadc/front.py) saw each page as one block.
   - page furniture blanked: EDGAR footers "Source: X, 10-K, date" (141 of 408 contracts), page numbers ("- 7 -",
     "Page 3 of 12", "A-1", a run of bare or trailing page numbers; 243), running headers (the same short line 4+ times;
     50), a table of contents (32; its entries were parsed as sections and swallowed the document).
   - inline sections: "... remedies. 4. Severability. All ..." gets a line break before "4." only when 4 continues the
     document's numbering (3 -> 4, 4 -> 4.1); "in Section 4. The ..." is left alone.
2. parse: cnli/compiler.py's parser, plus "5.Term" (number glued to its heading), heading lines that never continue
   the previous line or join a list, and "ARTICLE 2 ..." / ALL-CAPS headings that close the open levels.
3. statements with lead-ins, original offsets, paragraph spans, and their section chain.
4. sections: number ("2.1(a)", articles "V"), heading, parent, span; title-page lines ("Exhibit 10.13", the title) are
   marked `front` and are not headings of what follows.
5. symbols (defined terms), 6. parties + roles from the preamble, 7. cross-references (Section / Article / Exhibit,
   other documents, incorporation by reference).

Coverage (stats.py): no errors on 408 + 484 documents; median 42 ms per CUAD contract (p95 232, max 671 ms), 12 ms per
NDA. Statements located in the original: median 100%. Text under a heading: CUAD median 98% (58 contracts < 50%),
NDAs median 89% (152 < 50%: short NDAs with unheaded paragraphs). Section cross-references resolved: CUAD 9,800 of
11,754 (83%), NDAs 786 of 910. Parties: >= 2 found in 269/408 contracts, a role noun in 164.

## Headings as a signal for CUAD clause types (head_probe2.py; rules on write 326, scored on dev 82)
| front end | heading-word rules: clauses found, precision | a type's top-3 heading words: recall, text share |
|---|---|---|
| old (cuadc/front.py) | 13.8% @ 93.5% | 48.7% in 10.9% of the text |
| doccompile, nearest heading | 18.7% @ 90.2% | 56.8% in 4.4% |
| doccompile, heading chain | 28.9% @ 86.5% | 60.7% in 7.9% |

## Section tagging: live units vs compiled units (eval_tagging.py, curves.py)
5-fold CV (each fold trains its own TF-IDF + one-vs-rest LR, C=10 balanced, bench/family.py's setup) over CUAD's 408
tuning contracts (37 types, 3,946 present clauses) and ContractNLI train + dev (484 NDAs, 17 types, 4,985). Units:
live = router/contract_map.split_sections; leaf = runs of statements in the same innermost section; headed = the
innermost section with a heading. "head-in-text" = the heading chain written in front of the unit's text.
Clause finding: per (document, type) the best-scoring unit, right if it overlaps a gold span.

At equal precision per type (each type's ranking cut where it is still >= P right; the same rule for every method):
| set | units / features | unit F1 | mean AP | found @90% | @95% | @98% |
|---|---|---|---|---|---|---|
| CUAD | live / text (today) | 0.606 | 0.687 | 48.5% | 40.8% | 28.9% |
| CUAD | leaf / text | 0.617 | 0.704 | 50.7% | 35.7% | 27.6% |
| CUAD | leaf / head-in-text | 0.625 | 0.722 | 53.4% | 44.6% | **35.1%** |
| CUAD | headed / text | 0.651 | 0.713 | 54.0% | 42.5% | 30.6% |
| CUAD | headed / head-in-text | **0.653** | 0.719 | 54.0% | 44.5% | 33.5% |
| NDA | live / text (today) | 0.769 | 0.931 | 77.3% | 57.5% | 33.0% |
| NDA | leaf / head-in-text | 0.772 | 0.926 | 77.5% | 56.9% | 35.7% |
| NDA | headed / text | 0.818 | 0.931 | 81.6% | 64.1% | 38.1% |
| NDA | headed / head-in-text | **0.820** | 0.933 | 81.8% | 62.9% | **41.1%** |
Nested per-type gating (threshold per type from the other folds; what the router would do) shows the same order but
lands below its target (93-94% at a 95% target, 96-97% at 98%) for every method: NDA live 35.2% @ 96.8% -> headed
head-in-text 39.7% @ 97.4%; CUAD live 26.3% @ 97.1% -> 32.3% @ 96.4% (more found, slightly less precise).
A separate TF-IDF over the heading (text+head) was worse than writing the heading into the text.
**Pick: headed units, heading in the text** (one setup for both kinds): CUAD clauses found at 98% 28.9% -> 33.5%
(+16% relative), NDAs 33.0% -> 41.1% (+25%), unit F1 +0.05 on both. Units stay small (CUAD mean 596 chars, NDA 519).

## Contract kind from the parties' roles (role_kind.py): not worth it for the kind
Rules counted on the router's 3,294 training documents (a role -> kind when >= 95% of >= 10 documents): distributor /
consultant / licensor / licensee / executive -> commercial; disclosing / receiving party -> NDA; landlord / tenant /
lessor / lessee -> lease ("company", "buyer", "seller", "customer", "recipient" ... are mixed). On the 1,156 held-out
documents the rule fires on 68 (5.9%), 66 right (97.1%); the router is 99.1% on the 862 it is sure of, and on the 294
it is unsure of (83.3%) the rule fires on 6 (5 right). Roles stay useful for binding parties in questions, not for kind.

## Cross-references (spot check, 60 contracts)
"document" references are mostly real related documents (Research Plan, Quality Agreement, Separation Agreement, the
Plan). Unresolved section references were often into other documents ("Section 7.09 of the Separation Agreement",
"Section 4043(c) of ERISA"): now kind "document-section"; "Section 7.e" is read as 7(e). The rest are parse misses
(numbering the lexer didn't accept).

## Next
- Router port (ask first; the router tree has other sessions' uncommitted changes): doccompile.py -> router/, contract
  map taggers retrained on headed units with the heading in the text (CUAD tuning contracts / ContractNLI train+dev;
  CUAD test stays sealed), per-answer clause types from compiled sections.
- Neuro-symbolic options for the IR / back end: see the summary given to the user 2026-10-03.

## Latency check for "the LLM proposes, rules check" vs a custom network (2026-10-03)
- LLM rule extraction (Fireworks gpt-oss-120b, effort low; 3 calls, scratchpad llm_speed.py): 6 compiled sections per
  call = 4.0-5.0 s (1.5-1.9k tokens in, 1.2-1.5k out: output-bound). A 47k-char contract (54 sections) = 9 calls: ~5 s
  if run side by side, ~40 s one after another; ~12k output tokens (~$0.01), against the account's 45k generated
  tokens/min shared with Tier 2. One question's 1-3 sections: ~1-2 s.
- Local network, DeBERTa-v3-xsmall size (the reader network's weights; a token tagger costs the same forward pass), this
  box's CPU (8 cores): 0.31 s per 512-token window with 4 threads; median contract (6.7k tokens) 4.4 s, 90th
  percentile (23k tokens) 14.7 s; 1 thread ~2.7x slower. One question's 1-3 sections: one window, ~0.3 s.

## Questions in a typed form (2026-10-03; the user: "User writes a question, we transform it into that typed language?")
- Clause type of a question (qtype_probe.py; 3,204 yes/no user-style questions of v5-v7 over 1,124 LEDGAR provisions,
  95 types): the LEDGAR section tagger on the question: top-1 41.7%, top-3 54.2%; a question classifier (CV by
  provision): top-1 43.4%, top-3 59.5%; with near-synonym headings merged (75 types) 47.2% / 62.8%. People ask about
  actions and things, not headings ("Can the Administrative Agent sell the collateral?" over an IP provision): the clause
  type is a soft hint for which sections to read, not the question's type.
- Pre-Tier 0's own question grammar (router/frames.question_frame = Frame(asks CAN/MUST/PROHIBITED/DOES/EXISTS/PROPERTY,
  actor, action, things, conditions ...); qparse_probe.py, 6,227 user-style questions of v5-v7): typed 65.9% with the
  base lexicon, 75.8% with each passage's parties added (as the live path does). Not typed: an unknown actor 1,182
  ("Can the Committee ...", first person "Am I allowed ..."), negated 90, two actions 58, other shapes ~150.
  Pre-Tier 0 answers 2.5-2.9% of these sets: the question side is mostly typed already; the contract side is matched by
  words, sentence by sentence.
