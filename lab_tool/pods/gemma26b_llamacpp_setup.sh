#!/bin/bash
# Gemma 4 26B-A4B (Google's QAT Q4_0 GGUF) on a 4090 with llama.cpp for the LAB pilot (2026-10-06). OpenAI-compatible
# chat API with tool calls (--jinja), 131k unified KV for 4 slots, served as a Fireworks-style name (LAB's Fireworks adapter).
cd /workspace; export PATH=/usr/local/cuda/bin:$PATH
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake build-essential git wget > apt.log 2>&1) 
mkdir -p models
(for i in $(seq 20); do wget -q -c --read-timeout=30 --tries=3 -O models/gemma-4-26B_q4_0-it.gguf \
   https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf && break; sleep 5; done; echo "download exit $?" > download.done) &
git clone -q https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > ../cmake.log 2>&1 \
  && cmake --build build -j "$(nproc)" --target llama-server > ../make.log 2>&1; echo "build exit $?" > ../build.done
wait
cd /workspace/llama.cpp/build/bin && setsid nohup ./llama-server -m /workspace/models/gemma-4-26B_q4_0-it.gguf --alias accounts/fireworks/models/gemma-4-26b \
  --jinja -c 131072 -np 4 --kv-unified -fa on -ctk q8_0 -ctv q8_0 --cache-ram 0 --ctx-checkpoints 0 -ngl 999 --host 127.0.0.1 --port 8090 \
  > /workspace/server.log 2>&1 < /dev/null &
echo started
