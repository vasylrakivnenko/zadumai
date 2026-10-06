#!/bin/sh
# three lanes of feature extraction (resumable: finished rows are skipped)
cd /root/zadumai_nli_proto/contract_map/typed
PY=/root/projects/zadumai/.venv/bin/python
TAG=${1:-}
ARGS=${TAG:+--tag $TAG}
MODEL=${2:-}
M=${MODEL:+--model $MODEL}
(nice $PY extract.py nda_tune $ARGS $M >> logs/x_nda_tune$TAG.log 2>&1; nice $PY extract.py nda_ho $ARGS $M >> logs/x_nda_ho$TAG.log 2>&1) &
(nice $PY extract.py cuad_tune $ARGS $M >> logs/x_cuad_tune$TAG.log 2>&1; nice $PY extract.py cuad_ho $ARGS $M >> logs/x_cuad_ho$TAG.log 2>&1) &
(nice $PY extract.py wc1_open $ARGS $M >> logs/x_wc1_open$TAG.log 2>&1; nice $PY extract.py wc1_sealed $ARGS $M >> logs/x_wc1_sealed$TAG.log 2>&1; nice $PY extract.py short_tune $ARGS $M >> logs/x_short_tune$TAG.log 2>&1; nice $PY extract.py short_ho $ARGS $M >> logs/x_short_ho$TAG.log 2>&1) &
wait
