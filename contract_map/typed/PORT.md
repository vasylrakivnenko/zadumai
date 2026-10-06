# Porting the improved local engine to the router (2026-10-04) - NOT applied; needs the user's OK

> **Review notes (2026-10-06)** — discrepancies found in a review; the text below is unchanged unless noted.
>
> - The results table uses subsets (255 / 380 / 800 questions) without saying so; no script writes
>   `runs/final_config.txt` (it was written by hand). pt0_engine.patch still applies cleanly to the router (checked
>   2026-10-06 with `git apply --check`). Not applied.


What: pt0_engine.patch (10 files; `git apply --check` passes against the live tree as of 2026-10-04 02:30 UTC):
- router/frames.py: question typing (named parties "the Committee", filler words "also/still", the question's own named
  party over a lexicon thing, "No benefit ... shall be assigned by any Participant" read as a ban). Open sets: +7 right,
  0 new wrong, 0 lost.
- router/doccompile.py (new): the document compiler (page furniture, structure, sections, statements, parties, xrefs).
- router/located.py (new): the located reading for documents >= 3,000 chars (6 pieces by lemmas + embedder; Pre-Tier 0
  then the reader network on them; network yes at 0.90, or 0.951 with the retrained model).
- router/harness.py: the located stage after the reader network; with it on, Tier 0 NLI doesn't answer long documents
  (held-out long texts: 7/12 right reading them whole, 3/6 on located text).
- router/stages.py, router/admin.html, router/ui.html: switches "Tier 0 (located reading, long documents)" (key
  `located`, mask " l1") and "Reader network: the retrained model" (key `netft`, mask " f1"), both off by default.
- router/netreader.py: MODEL_DIR_FT (models/reader_net_ft1, or $READER_NET_FT), T_YES_FT 0.948.
- ask_ui.py: loads the retrained network and the two located readers; passes them by the switches.
- tests/test_located.py (new, 6 tests). Router suites with the patch: 502 + 11 (tier2) pass, 1 skipped.

Steps (after the user's OK):
  cd /root/projects/zadumai && git apply /root/zadumai_nli_proto/contract_map/typed/pt0_engine.patch
  mkdir -p legalbench_map/models/reader_net_ft1 && cp typed/models/ft1/{config.json,model.safetensors,tokenizer.json,tokenizer_config.json,train_args.json} legalbench_map/models/reader_net_ft1/
  (gitignored like models/reader_net; PROVENANCE: typed/RESULTS_NS.md "Idea 4")
  test on :8777, then restart zadum-router; both switches start off; turn them on in /admin.
Memory: +1 DeBERTa-xsmall (~280 MB) for the retrained model; spaCy en_core_web_sm is already in the venv.

Measured (runs/final_config.txt; same questions per row; live = today's switches):
| set | live | + located | + located + retrained |
| wc1 open (59) | 6 @ 83.3% | 8 @ 100% | 8 @ 100% |
| wc1 sealed (126, spent) | 10 @ 90.0% | 26 @ 100% | 22 @ 100% |
| NDA held-out (255) | 32 @ 90.6% | 53 @ 88.7% | 59 @ 93.2% |
| CUAD held-out (380) | 56 @ 85.7% | 91 @ 85.7% | 77 @ 89.6% |
| short held-out (800) | 164 @ 99.4% | 164 @ 99.4% | 192 @ 99.5% |
Latency: long documents ~3 s on the first question (compile + embed, cached per document), then ~0.3-0.6 s; Tier 0
NLI's 0.6-4 s on long documents is skipped with the located reading on. Short texts unchanged (~94 ms median).
