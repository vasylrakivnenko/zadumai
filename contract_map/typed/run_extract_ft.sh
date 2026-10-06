#!/bin/sh
# the same features with the fine-tuned network (idea 4): tag "ft"
cd /root/zadumai_nli_proto/contract_map/typed
PY=/root/projects/zadumai/.venv/bin/python
M=/root/zadumai_nli_proto/contract_map/typed/models/ft1
(nice $PY extract.py nda_tune --tag ft --model $M >> logs/xft_nda_tune.log 2>&1; nice $PY extract.py nda_ho --tag ft --model $M >> logs/xft_nda_ho.log 2>&1) &
(nice $PY extract.py cuad_tune --tag ft --model $M >> logs/xft_cuad_tune.log 2>&1; nice $PY extract.py cuad_ho --tag ft --model $M >> logs/xft_cuad_ho.log 2>&1) &
(nice $PY extract.py wc1_open --tag ft --model $M >> logs/xft_wc1_open.log 2>&1; nice $PY extract.py wc1_sealed --tag ft --model $M >> logs/xft_wc1_sealed.log 2>&1; nice $PY extract.py short_tune --tag ft --model $M --n 1500 >> logs/xft_short_tune.log 2>&1; nice $PY extract.py short_ho --tag ft --model $M >> logs/xft_short_ho.log 2>&1) &
wait
echo FT_DONE >> logs/xft_done.log
