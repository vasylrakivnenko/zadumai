"""Exact cost of the NDA pipeline test (2026-10-06): agent usage logs (every draw, cached tokens included) + judge log."""
import glob, json, sys
DS = (0.30, 0.006, 1.20); KIMI = (3.00, 0.30, 15.00); P = "/root/zadumai_nli_proto/lab_tool/runs/pilot"
def cost(rows, pr): return sum(((u["prompt"] - u["cached"]) * pr[0] + u["cached"] * pr[1] + u["completion"] * pr[2]) / 1e6 for u in rows)
tot = 0
for f in sorted(glob.glob(f"{P}/{sys.argv[1] if len(sys.argv) > 1 else 'pilot-*non-disclosure-agreement-first-turn-redline'}.usage.jsonl")):
    U = [json.loads(l) for l in open(f)]; c = cost(U, DS); tot += c
    print(f"{f.split('/')[-1][:-12]}: {len(U)} calls, prompt {sum(u['prompt'] for u in U) / 1e6:.2f}M (cached {sum(u['cached'] for u in U) / 1e6:.2f}M), output {sum(u['completion'] for u in U) / 1e3:.0f}k, truncated {sum(u['finish'] == 'length' for u in U)} -> ${c:.2f}")
JF = sys.argv[2] if len(sys.argv) > 2 else f"{P}/judge_usage.jsonl"
J = [json.loads(l) for l in open(JF)] if glob.glob(JF) else []
jc = cost(J, KIMI); print(f"judge: {len(J)} calls -> ${jc:.2f}\ntotal: ${tot + jc:.2f}")
