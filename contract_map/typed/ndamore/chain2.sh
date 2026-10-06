#!/bin/bash
# round 2 of the SEC NDAs (2026-10-04): after search2 -> select -> fetch -> filter -> label (IR, CNN data, Jev) -> teacher CNNs -> soft labels
cd /root/zadumai_nli_proto/contract_map/typed/ndamore
PY=/root/projects/zadumai/.venv/bin/python
# (the search has finished; a pgrep wait matched its own launcher)
echo "$(date -u +%H:%M) search done: $(wc -l < data/hits.jsonl) hits"
$PY - <<'PYEOF'
import json, re
h = [json.loads(l) for l in open("data/hits.jsonl")]
nda = re.compile(r"confidential|non-?disclosure|nondisclosure|secrecy|\bnda\b", re.I)
bad = re.compile(r"merger|purchase|employ|separation|severance|consult|option|credit|loan|lease|license|supply|settlement|plan of", re.I)
keep = {x["id"] for x in h if (nda.search((x["desc"] or "") + " " + x["id"].split(":")[1]) and not bad.search(x["desc"] or ""))
        or not (x["desc"] or "").strip() or re.fullmatch(r"(?i)\s*(ex(hibit)?[-\s.]*[\d.()a-z]+)\s*", x["desc"] or "")}
json.dump(sorted(keep), open("data/fetch_ids.json", "w")); print("selected", len(keep), "exhibits (incl. the 7,870 already fetched)")
PYEOF
$PY collect.py fetch > data/fetch2_log.txt 2>&1; tail -1 data/fetch2_log.txt
cp data/ndas.jsonl data/ndas_round1.jsonl
$PY collect.py filter
TOKENIZERS_PARALLELISM=false $PY label.py features 2>&1 | grep -v -i "warn\|longer than\|indexing" | tail -3
$PY label.py cnn > data/cnn_log2.txt 2>&1; tail -1 data/cnn_log2.txt
/root/zadumai_nli_proto/venv_xgb/bin/python label.py teach 2>&1 | grep -v -i warn | tail -1
echo "$(date -u +%H:%M) chain2 done"
