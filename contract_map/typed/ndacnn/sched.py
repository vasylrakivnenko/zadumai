"""Runs the CNN variants on the 8 cores (2026-10-04; the user: "When the training ends, run all these variants and then
report"). Waits until the encoder's 4 folds (../ndaenc) are done (its last run then keeps 4 cores, the CNN 2 runs x 2 threads; all 8
when it ends; 2026-10-04 the user: use the CPU), then: stage 1 = the five kernel stacks x (5 folds +
the model on all training NDAs for held-out); stage 2 = the stack with the best out-of-fold accuracy + the statement
graph, and + the graph + two slots (rule / exception), same runs. 4 runs at a time, 2 threads each.
usage: python sched.py  (log: runs/sched_log.txt; each run: runs/<name>_f<F>.txt)"""
import glob, json, os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/root/projects/zadumai/.venv/bin/python"
PAR, THREADS, EPOCHS, CH = 4, 2, 12, 96
FOLDS = [0, 1, 2, 3, 4, -1]
os.makedirs(f"{HERE}/runs", exist_ok=True)


def say(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def encoder_busy():
    return subprocess.run(["pgrep", "-f", "[n]daenc.*train.py|[t]rain.py --fold"], capture_output=True).returncode == 0 and \
        any(not os.path.exists(f"{HERE}/../ndaenc/preds/{n}") for n in ("f1_train.jsonl", "f2_train.jsonl", "f3_train.jsonl", "full_ho.jsonl"))


def name(v, extra):
    return v + ("+graph" if "--graph" in extra else "") + ("+2slot" if "2" in extra else "")


ENC = f"{HERE}/../ndaenc"


def enc_full_alive():
    return subprocess.run(["pgrep", "-f", "[t]rain.py --fold -1"], capture_output=True).returncode == 0


def run(jobs):
    running = []
    while jobs or running:
        running = [(n, p) for n, p in running if p.poll() is None or say(n, "exit", p.returncode)]
        par = PAR
        if enc_full_alive():  # the encoder's last run keeps 4 cores, the CNN gets the other 4
            open(f"{ENC}/runs/full/threads", "w").write("4"); par = 2
        while jobs and len(running) < par:
            v, extra, f = jobs.pop(0); n = f"{name(v, extra)}_f{f}"
            cmd = [PY, "train.py", "--variant", v, "--fold", str(f), "--threads", str(THREADS), "--epochs", str(EPOCHS), "--ch", str(CH)] + extra
            running.append((n, subprocess.Popen(["nice", "-n", "5"] + cmd, cwd=HERE, stdout=open(f"{HERE}/runs/{n}.txt", "w"), stderr=subprocess.STDOUT)))
            say("start", n)
        time.sleep(10)


def oof_acc(n):
    right = tot = 0
    gold = {}
    import pickle
    for d in pickle.load(open(f"{HERE}/data/train.pkl", "rb")):
        for k, y in enumerate(d["y"].tolist()): gold[(d["doc"], k)] = y
    QIDS = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13", "nda-15",
            "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]
    for f in range(5):
        for l in open(f"{HERE}/preds/{n}_f{f}.jsonl"):
            r = json.loads(l); d, q = r["id"].split("/")
            right += int(max(range(3), key=lambda k: r["p"][k]) == gold[(d, QIDS.index(q))]); tot += 1
    return right / tot


while not all(os.path.exists(f"{ENC}/preds/{n}") for n in ("f1_train.jsonl", "f2_train.jsonl", "f3_train.jsonl")): time.sleep(20)
say("encoder folds done (its full model may still run on 4 cores); stage 1")
STACKS = ["plain", "parallel", "rising", "falling", "updown"]
run([(v, [], f) for v in STACKS for f in FOLDS])
acc = {v: oof_acc(v) for v in STACKS}
say("stage 1 out-of-fold:", json.dumps({k: round(x, 4) for k, x in acc.items()}))
best = max(acc, key=acc.get)
say("stage 2 on", best)
run([(best, ["--graph"], f) for f in FOLDS] + [(best, ["--graph", "--slots", "2"], f) for f in FOLDS])
for n in (f"{best}+graph", f"{best}+graph+2slot"): say(n, "out-of-fold", round(oof_acc(n), 4))
say("all done")
