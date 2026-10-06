#!/bin/bash
# the tool-caller benchmark for one model (2026-10-05, Muse Glimmer): held-out test splits, plain vs GEPA instructions
# usage: run_bench.sh MODEL LABEL   (writes data/out_{A,B}_<LABEL>_guided_test_*.jsonl, appends to muse/PROGRESS.md)
cd /root/zadumai_nli_proto/lab_tool; PY=/root/projects/zadumai/.venv/bin/python; M=$1; L=$2
SPLIT=test TAG=test_plain MODE=guided LABEL=$L $PY run_replay.py A "$M" 0 | tail -1
SPLIT=test TAG=test_gepa_native MODE=guided LABEL=$L INSTR_FILE=data/gepa_A_instruction_native.txt $PY run_replay.py A "$M" 0 | tail -1
SPLIT=test TAG=test_plain MODE=guided LABEL=$L $PY run_replay.py B "$M" 40 | tail -1
SPLIT=test TAG=test_gepa MODE=guided LABEL=$L INSTR_FILE=data/gepa_B_instruction.txt $PY run_replay.py B "$M" 40 | tail -1
echo "- $(date -u +%H:%M) UTC benchmark done for $L" >> muse/PROGRESS.md
