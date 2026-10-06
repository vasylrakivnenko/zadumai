"""The teacher's CNNs with their weights saved (2026-10-04, step 2): rising width 5 and 3, 128-d vectors, 5 folds + all
training NDAs; 4 runs at a time, 2 threads each. usage: python sched_save.py"""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
jobs = [(["--variant", "rising", "--k", k, "--dim", "128", "--save"], f) for k in ("5", "3") for f in (0, 1, 2, 3, 4, -1)]
running = []
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or print(time.strftime("%H:%M:%S"), n, "exit", p.returncode, flush=True)]
    while jobs and len(running) < 4:
        x, f = jobs.pop(0); n = "_".join(x).replace("--", "") + f"_f{f}"
        cmd = ["nice", "-n", "5", PY, "train.py", "--fold", str(f), "--threads", "2", "--epochs", "12", "--ch", "96"] + x
        running.append((n, subprocess.Popen(cmd, cwd=HERE, stdout=open(f"{HERE}/runs/save_{n}.txt", "w"), stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(10)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
