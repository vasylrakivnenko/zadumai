"""Kernel widths + learning curve (2026-10-04; the user: "Should we also try with bigger kernel? Maybe 12?" -> 13, odd;
and the learning curve before collecting more NDAs). 4 runs at a time, 2 threads each. usage: python sched_k.py"""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
jobs = [(["--variant", "rising", "--k", "13"], f) for f in (0, 1, 2, 3, 4, -1)]
jobs += [(["--variant", "rising", "--frac", str(fr)], f) for fr in (0.75, 0.5, 0.25) for f in (0, 1, 2, 3, 4)]
jobs += [(["--variant", "plain", "--k", "13"], f) for f in (0, 1, 2, 3, 4, -1)]
jobs += [(["--variant", "rising", "--k", "5"], f) for f in (0, 1, 2, 3, 4, -1)]
running = []
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or print(time.strftime("%H:%M:%S"), n, "exit", p.returncode, flush=True)]
    while jobs and len(running) < 4:
        x, f = jobs.pop(0); n = "_".join(x).replace("--", "") + f"_f{f}"
        cmd = ["nice", "-n", "5", PY, "train.py", "--fold", str(f), "--threads", "2", "--epochs", "12", "--ch", "96"] + x
        running.append((n, subprocess.Popen(cmd, cwd=HERE, stdout=open(f"{HERE}/runs/k_{n}.txt", "w"), stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(10)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
