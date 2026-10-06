#!/bin/bash
# Gemma 4 E4B for the LAB pilot (2026-10-06): vLLM (pulls torch cu130: the pod is on a CUDA 13 host), OpenAI-compatible
# chat API with Gemma's tool-call parser; served under a Fireworks-style name so LAB's Fireworks adapter (and our patches:
# context cap, output cap, redraws, usage log) drive it through FIREWORKS_API_BASE.
cd /workspace
pip install -q --break-system-packages vllm hf_transfer > pip.log 2>&1; echo "pip exit $?" >> pip.log
export HF_HUB_ENABLE_HF_TRANSFER=1
nohup vllm serve google/gemma-4-E4B-it --served-model-name accounts/fireworks/models/gemma-4-e4b --enable-auto-tool-choice \
  --tool-call-parser gemma4 --max-model-len 131072 --gpu-memory-utilization 0.92 --port 8000 > vllm.log 2>&1 < /dev/null &
echo started
