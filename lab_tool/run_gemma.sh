#!/bin/bash
# Gemma 4 on the replay benchmark (2026-10-05, the user: "Let's try Gemma models too"); Gemini API, one run at a time
cd /root/zadumai_nli_proto/lab_tool; PY=/root/projects/zadumai/.venv/bin/python
for m in gemma-4-26b-a4b-it gemma-4-31b-it; do
  MODE=guided $PY run_replay.py A $m 120; MODE=neutral $PY run_replay.py A $m 120; MODE=guided $PY run_replay.py B $m 20
done
echo "all done"
