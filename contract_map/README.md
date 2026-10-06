# Contract map research (2026-10-02 to 2026-10-05)

_Code + write-ups are also in the zadumai repo (`contract_map/`, small result files only). The working copy with data, models and caches is `/root/zadumai_nli_proto/contract_map` on the server; scripts use its absolute paths._

Research for the router's "contract map": what kind of document a text is, what its sections are, and engines that
answer contract questions locally. Rewritten as an index on 2026-10-06 (review); the original first-hour notes on
FOLIO / CUAD / ACORD are at the bottom.

| folder / file | what | write-up |
|---|---|---|
| `bench/` | classifier benchmark, clause taxonomy, the document-kind router (the basis of the router's contract map, commit 56093f4) | bench/RESULTS.md |
| `build_models.py`, `eval_routing.py`, `llm_check.py`, `taxonomy.json`, `routed_rows.jsonl`, `any_tagger.py`, `clause_kind.py` | models and checks the live router's STATUS.md cites (`build_models.log`: the deployed encoder's measurements) | router STATUS.md |
| `doccompile/` | the document compiler: sections, defined terms, section tagging | doccompile/RESULTS.md |
| `cuadc/` | CUAD six-type clause compiler and its agreement with an LLM | cuadc/RESULTS.md |
| `cnli/` | selective ContractNLI engine (sealed result: 32.3% answered at 96.3%), label audit | cnli/RESULTS.md |
| `typed/` | typed pipeline + located reading; neuro-symbolic ideas 1-5; the NDA engine (XGBoost stacker, CNN `ndacnn/`, fine-tuned encoder `ndaenc/`, SEC NDAs `ndamore/`, Jev features); the Pre-Tier 0 port plan | typed/RESULTS.md (pipeline), **typed/RESULTS_NS.md** (main, newest), typed/PORT.md (port: `pt0_engine.patch`, not applied) |
| `typed/dev/`, `typed/base_router_snapshot/` | copies of the router code for development (dev/router/systemone.py is the Jev client used elsewhere) | |
| `data/`, `cuad_full/`, `CUAD_v1.zip` | public downloads, kept locally (FOLIO.owl, CUAD, ContractNLI and others under data/ext) | |

Most of the 6.4 GB is models and caches in `typed/` (CNN / encoder runs, fine-tuned reader checkpoints, the raw SEC
fetch); results that are unique are in `typed/feats/`, `typed/*/preds/`, `typed/runs/*.json`, `wc1.json`.

## Original notes (2026-10-02)
Data sources:
- FOLIO.owl  https://github.com/alea-institute/FOLIO/raw/main/FOLIO.owl  (SALI LMSS.owl has the same contract content)
- CUAD       https://github.com/TheAtticusProject/cuad/raw/main/data.zip  (kept as `CUAD_v1.zip`, `cuad_full/`, and
  `data/CUADv1.json`; `cuad_matrix.py` still opens `cuad/CUADv1.json`: point it at `data/CUADv1.json` to run it)
- ACORD      https://huggingface.co/datasets/theatticusproject/acord  (CC BY 4.0, not downloaded)
Scripts: parse.py / tree.py / sub.py / stats.py read the OWL (run from the folder holding FOLIO.owl);
cuad_matrix.py -> cuad_prevalence.txt (share of CUAD contracts of each type that have each of the 41 categories;
contract type guessed from the title, rough).
