"""plain3 (3 layers, kernels 13-9-5) + pyramid (DPCNN), queued after sched_k.py; the user: "Why not at least 3 decreasing / shrinking?" (2026-10-04; the user: "Should we also try with bigger kernel? Maybe 12?" -> 13, odd;
and the learning curve before collecting more NDAs). 4 runs at a time, 2 threads each. usage: python sched_k.py"""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
jobs = [(["--variant", v], f) for v in ("pyramid", "plain3") for f in (0, 1, 2, 3, 4, -1)]
running = []
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or print(time.strftime("%H:%M:%S"), n, "exit", p.returncode, flush=True)]
    while jobs and len(running) < 4:
        x, f = jobs.pop(0); n = "_".join(x).replace("--", "") + f"_f{f}"
        cmd = ["nice", "-n", "5", PY, "train.py", "--fold", str(f), "--threads", "2", "--epochs", "12", "--ch", "96"] + x
        running.append((n, subprocess.Popen(cmd, cwd=HERE, stdout=open(f"{HERE}/runs/p_{n}.txt", "w"), stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(10)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
