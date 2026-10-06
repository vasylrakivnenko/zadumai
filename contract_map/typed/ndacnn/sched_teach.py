"""CNN with teachers (2026-10-04; the user: "Let's go! 1, see results and then go for 2"): the rising stack + Jev's
answers on the training NDAs (an extra head / mixed into the targets) and / or the symbolic detectors' flags (a head),
5 folds + the model on all training NDAs; 4 runs at a time, 2 threads each. usage: python sched_teach.py"""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
VARIANTS = [["--jev", "head"], ["--jev", "mix"], ["--regex", "0.3"], ["--jev", "mix", "--regex", "0.3"]]
jobs = [(x, f) for x in VARIANTS for f in (0, 1, 2, 3, 4, -1)]
running = []
while jobs or running:
    running = [(n, p) for n, p in running if p.poll() is None or print(time.strftime("%H:%M:%S"), n, "exit", p.returncode, flush=True)]
    while jobs and len(running) < 4:
        x, f = jobs.pop(0); n = "_".join(x) + f"_f{f}"
        cmd = ["nice", "-n", "5", PY, "train.py", "--variant", "rising", "--fold", str(f), "--threads", "2", "--epochs", "12", "--ch", "96"] + x
        running.append((n, subprocess.Popen(cmd, cwd=HERE, stdout=open(f"{HERE}/runs/teach_{n}.txt", "w"), stderr=subprocess.STDOUT)))
        print(time.strftime("%H:%M:%S"), "start", n, flush=True)
    time.sleep(10)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
