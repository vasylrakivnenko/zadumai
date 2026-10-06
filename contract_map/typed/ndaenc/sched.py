"""Runs the remaining encoder trainings side by side on the 8 cores (2026-10-04; fold 0 was started alone with 6
threads). Every 15 s: start the next job while there are free cores (at most 3 at a time), and share the free cores
among the running jobs (train.py reads runs/<name>/threads every 25 steps)."""
import os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
CORES, MAX_RUN = 8, 3
queue = [("f1", 1), ("f2", 2), ("f3", 3), ("full", -1)]
running = {}


def f0_alive():
    return subprocess.run(["pgrep", "-f", "[t]rain.py --fold 0$"], capture_output=True).returncode == 0


while queue or running:
    for n, p in list(running.items()):
        if p.poll() is not None:
            print(time.strftime("%H:%M:%S"), n, "exit", p.returncode, flush=True); del running[n]
    free = CORES - (6 if f0_alive() else 0)
    while queue and len(running) < MAX_RUN and free // (len(running) + 1) >= 2:
        n, f = queue.pop(0); os.makedirs(f"{HERE}/runs/{n}", exist_ok=True)
        share = free // (len(running) + 1)
        open(f"{HERE}/runs/{n}/threads", "w").write(str(share))
        running[n] = subprocess.Popen(["nice", PY, "train.py", "--fold", str(f), "--threads", str(share)], cwd=HERE,
                                      stdout=open(f"{HERE}/runs/log_{f}.txt", "w"), stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "start", n, "threads", share, flush=True)
    if running:
        names = sorted(running); base, extra = divmod(free, len(names))
        for k, n in enumerate(names):
            open(f"{HERE}/runs/{n}/threads", "w").write(str(max(1, base + (k < extra))))
    time.sleep(15)
print(time.strftime("%H:%M:%S"), "all done", flush=True)
