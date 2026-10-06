#!/bin/sh
cd /root/zadumai_nli_proto/contract_map/typed
PY=/root/projects/zadumai/.venv/bin/python
# 3 shards of nda_more by document, then the compiler extras for them
for k in 0 1 2; do (SHARD=$k nice $PY -c "
import sys, os, json
sys.argv = ['extract.py', 'nda_more', '--tag', 'more$k']
import extract as X
rows = X.SETS['nda_more']()
docs = sorted({r['id'].split('/')[0] for r in rows})
keep = set(docs[$k::3])
X.SETS['nda_more'] = lambda: [r for r in rows if r['id'].split('/')[0] in keep]
X.main()
" > logs/x_more_$k.log 2>&1) & done
wait
echo MORE_DONE > logs/more_done.log
