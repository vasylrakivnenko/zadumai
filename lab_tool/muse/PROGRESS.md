# Muse Glimmer 30B as a tool caller: progress log (checkpoints)
2026-10-05 (user: "Go. Report in the progress. Save checkpoints please.")
- 19:30 UTC Fireworks: Muse Glimmer 30B is in the catalog (not serverless) -> on-demand deployment, official shape
  "Muse Glimmer 30B BF16 1x B300" (accounts/fireworks/deploymentShapes/w49lad6x/versions/v82l0wmx). Deployment name in
  fw_deployment.txt (deleted with the deployment, 2026-10-05). DELETE it when done.
- 19:30 UTC Runpod pod ebcoma2fs64i2g (69.145.85.72 14366) created for the 4-bit track (zadum-muse-q4). DELETE it when done.
- 19:31 UTC llama.cpp c250304 (Oct 5) building for sm_89 (nvcc needs PATH=/usr/local/cuda/bin); Q4_K_M download running.
- Benchmark: muse/run_bench.sh MODEL LABEL (held-out test splits: A 235 moments from 32 General tasks; B 40 training-split
  Contracts tasks; plain one-line instruction vs Gemma's GEPA instructions); scoring: dspy_venv muse/score_muse.py LABEL.
  Reference, Gemma 4 E4B: A plain 77.7% balanced, A GEPA 90.9% (decisions; 67.4% native); B plain lift +20.5, GEPA +27.2.
- 19:33 UTC Fireworks deployment READY (~4 min); model string in muse/fw_model.txt
- 19:34 UTC Fireworks returns HTTP 400 on native tool calls (it can't parse Muse's ATEM <atem:invoke> format); fixed: run_replay renders Meta's chat template with the tools, calls /completions, parses ATEM blocks (call_atem). Smoke test OK. Full BF16 benchmark launched.
- 19:42 UTC benchmark done for muse-bf16
- BF16 (Fireworks, ATEM parsing) RESULTS, held-out: A plain 47.1% balanced (uses 15.6% / holds off 78.6%); A with
  Gemma's GEPA instruction 85.2% (74.3 / 96.0, native calls, no JSON-text issue); B plain lift +26.5 (5 queries);
  B GEPA lift +34.8 (12 queries) = best so far. Fireworks deployment DELETED after the run.
- 19:46 UTC Q4_K_M served by llama.cpp c250304 on the 4090 (16.3 GB; -c 32768 -np 4, q8_0 KV); tunnel 18081.
- 20:52 UTC benchmark done for muse-q4
- Q4_K_M (llama.cpp on the 4090) RESULTS, held-out (deployment format, Gemma's GEPA instructions transplanted, no re-tuning):
  A plain 45.9% balanced (uses 9.2 / holds off 82.5); A GEPA 83.8% (71.6 / 96.0, native calls); B plain lift +28.1
  (6 queries); B GEPA lift +32.9 (12 queries). Within 1-2 points of BF16 (A: 235 moments, so +-3 points is noise).
  Speed: ~66 min for 550 requests at 4 parallel slots (A with the 2,500-word GEPA instruction: 34 min) vs ~8 min on the
  Fireworks B300; Gemma 4 E4B (vLLM, same 4090) ran the same A set in ~3.5 min.
- 20:55 UTC Runpod pod ebcoma2fs64i2g DELETED (API 204, then 404); no tunnels left on 18080 / 18081. Fireworks deployment
  was deleted at 19:42. Muse track closed.
