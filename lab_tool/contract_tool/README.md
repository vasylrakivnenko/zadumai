# contract_tool v0

One command over a matter's documents folder, for an agent that should not read whole documents into its context.
Built 2026-10-05 after reading Harvey's LAB harness (harvey-labs @ a2b429e): its `read` returns the whole parsed document
and silently accepts tracked changes (pandoc default), and its `grep` searches raw files, so it finds nothing inside
.docx / .xlsx / .pptx. General purpose: it knows nothing about LAB's tasks or rubrics.

| action | arguments | returns |
|---|---|---|
| `outline` | (none) | every document: size, tracked-change counts, title |
| `outline` | `document` | its headings and clause numbers with ¶ ids |
| `find` | `query`, optional `document` | the 6 most relevant passages (≤ 2 per section / sheet), each with document, headings, clause number, ¶ id; tracked changes shown as `[-deleted-]{+inserted+}` |
| `show` | `document`, `query` = clause number (`4.2`) or ¶ range (`62-73`) | those paragraphs in full |
| `changes` | `document`, optional `query`, `offset` | the tracked changes clause by clause (author, date); `query` puts the relevant ones first |
| `compare` | `document` (older), `against` (newer), optional `query`, `offset` | the two versions aligned paragraph by paragraph: changed (word diff), added, removed |
| `ask` | = `find` within one document (no reader model in v0) | |

- Formats: .docx (paragraphs + table rows, tracked changes kept apart), .xlsx (rows named by the header row), .pptx,
  .eml (headers + body paragraphs), .txt / .md, .pdf (via pdftotext when installed).
- Search: BM25 (prefix-stemmed) + bge-small-en-v1.5 (CPU), reciprocal-rank fusion. Deleted text is searchable.
- Output capped at ~5,000 characters (~1,200 tokens) per call; `offset` pages `changes` / `compare`.
- Cache: parsed units and embeddings per file (sha1) in `contract_tool/cache/`.

```
dspy_venv/bin/python -m contract_tool.tool DOCS_DIR outline
dspy_venv/bin/python -m contract_tool.tool DOCS_DIR find --query "royalty rate" [--document NAME]
dspy_venv/bin/python -m contract_tool.tool DOCS_DIR changes --document NAME [--query Q] [--offset N]
dspy_venv/bin/python -m contract_tool.tool DOCS_DIR compare --document OLDER --against NEWER [--query Q]
dspy_venv/bin/python -m contract_tool.tool DOCS_DIR show --document NAME --query 4.2
```

Python: `ContractTool(docs_dir).call(action, query, document, against, offset)` returns the text an agent sees.
Evaluation: `eval_tool_v0.py` (results in `../RESULTS.md`).
