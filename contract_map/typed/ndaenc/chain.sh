#!/bin/bash
# cross-fitting folds 0-3, then the model on all training NDAs (predicts nda_ho only)
cd /root/zadumai_nli_proto/contract_map/typed/ndaenc
for f in 0 1 2 3 -1; do
  nice /root/projects/zadumai/.venv/bin/python train.py --fold $f > runs/log_$f.txt 2>&1 || exit 1
done
