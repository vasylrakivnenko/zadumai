#!/bin/bash
# the replay batch (2026-10-05): two models at a time, 4 requests each
cd /root/zadumai_nli_proto/lab_tool; PY=/root/projects/zadumai/.venv/bin/python
M1=accounts/fireworks/models/kimi-k3; M2=accounts/fireworks/models/deepseek-v4p1-flash; M3=accounts/fireworks/models/gpt-oss-120b
for mode in guided neutral; do
  MODE=$mode $PY run_replay.py A $M1 120 & MODE=$mode $PY run_replay.py A $M2 120; wait
  MODE=$mode $PY run_replay.py A $M3 120
done
MODE=guided $PY run_replay.py B $M1 20 & MODE=guided $PY run_replay.py B $M2 20; wait
MODE=guided $PY run_replay.py B $M3 20
echo "all done"
