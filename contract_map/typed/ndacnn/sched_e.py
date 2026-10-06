"""Early round for the per-clause kernels (2026-10-04; the user: start per clause, then kernels over the whole pooled /
processed NDA): folds 0 and 1 only, width 5; the pyramid batch is paused meanwhile (runs/paused_pids.txt).
usage: python sched_e.py"""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
V = [["--variant", "clause", "--k", "5"], ["--variant", "clause", "--k", "5", "--across", "4"], ["--variant", "clausetok", "--k", "5"]]
jobs = [(x, f) for f in (0, 1) for x in V]
running = []
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or print(time.strftime("%H:%M:%S"), n, "exit", p.returncode, flush=True)]
    while jobs and len(running) < 4:
        x, f = jobs.pop(0); n = "_".join(x).replace("--", "") + f"_f{f}"
        cmd = ["nice", "-n", "5", PY, "train.py", "--fold", str(f), "--threads", "2", "--epochs", "12", "--ch", "96"] + x
        running.append((n, subprocess.Popen(cmd, cwd=HERE, stdout=open(f"{HERE}/runs/e_{n}.txt", "w"), stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(10)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
