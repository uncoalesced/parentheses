"""
Overnight watchdog for one train.py run (documentation.md 2026-10-07).

    python scripts/overnight_driver.py --name parentheses-0.9-40M -- <train.py args>

- Runs train.py with stdout/stderr appended to out/<name>.log / .err.
- On a non-zero exit, resumes from the newest checkpoint in --out-dir, up to
  --max-restarts times. On a CUDA OOM, halves --batch-size and doubles
  --grad-accum first (same tokens per optimizer step).
- Every 60 s writes out/overnight_status.txt: stage, step, last loss, ETA. It
  reads only the last few KB of the log for that -- a status file, not a tail.

Python rather than the brief's overnight_driver.sh: this machine's Git Bash
has no grep/python on PATH for detached jobs, and tasklist parsing is fragile.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_RE = re.compile(rb"step\s+(\d+) \| loss ([\d.]+)")


def arg_value(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def set_arg(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag, str(value)]


def newest_ckpt(out_dir):
    best, best_step = None, -1
    for p in glob.glob(os.path.join(out_dir, "step_*.pt")):
        m = re.search(r"step_(\d+)", os.path.basename(p))
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best


def last_step(log_path):
    try:
        with open(log_path, "rb") as f:
            f.seek(max(0, os.path.getsize(log_path) - 4096))
            hits = STEP_RE.findall(f.read())
        return (int(hits[-1][0]), float(hits[-1][1])) if hits else (None, None)
    except OSError:
        return None, None


def write_status(path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(f"{k}: {v}\n" for k, v in kw.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--max-restarts", type=int, default=3)
    p.add_argument("--status", default=os.path.join(ROOT, "out", "overnight_status.txt"))
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    targs = [x for x in a.train_args if x != "--"]
    out_dir = arg_value(targs, "--out-dir")
    max_steps = int(arg_value(targs, "--max-steps"))
    log, err = (os.path.join(ROOT, "out", f"{a.name}.{e}") for e in ("log", "err"))
    restarts = 0
    while True:
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), *targs]
        with open(log, "ab") as lo, open(err, "ab") as le:
            lo.write(f"\n[driver] {time.strftime('%H:%M:%S')} launch: {' '.join(cmd)}\n".encode())
            lo.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lo, stderr=le,
                                    # unbuffered: a redirected stdout is block-buffered, which
                                    # left the log (and so the status file) minutes stale
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
            t0, s0 = time.time(), None
            while proc.poll() is None:
                time.sleep(60)
                step, loss = last_step(log)
                if step is not None and s0 is None:
                    t0, s0 = time.time(), step
                rate = (step - s0) / (time.time() - t0) if step and s0 is not None and step > s0 else None
                eta = time.strftime("%H:%M", time.localtime(time.time() + (max_steps - step) / rate)) if rate else "?"
                write_status(a.status, stage=f"G main run {a.name}", step=f"{step}/{max_steps}",
                             loss=loss, steps_per_s=f"{rate:.3f}" if rate else "?", eta=eta,
                             restarts=restarts, updated=time.strftime("%H:%M:%S"))
        rc = proc.returncode
        if rc == 0:
            step, loss = last_step(log)
            write_status(a.status, stage=f"G main run {a.name} DONE", step=step, loss=loss,
                         restarts=restarts, updated=time.strftime("%H:%M:%S"))
            return 0
        restarts += 1
        with open(err, "rb") as f:
            f.seek(max(0, os.path.getsize(err) - 8192))
            tail = f.read()
        oom = b"OutOfMemory" in tail or b"out of memory" in tail
        if restarts > a.max_restarts:
            write_status(a.status, stage=f"G main run {a.name} FAILED rc={rc}", restarts=restarts - 1,
                         updated=time.strftime("%H:%M:%S"))
            return rc
        if oom:
            bs, ga = int(arg_value(targs, "--batch-size")), int(arg_value(targs, "--grad-accum", 1))
            # --mix PATH ROWS [BOUNDARIES]: halve each ROWS too (min 1), and only halve the
            # batch if a main-data row is left, else train.py rejects the args and every
            # restart fails the same way
            mix_at = [i + 2 for i, x in enumerate(targs) if x == "--mix"]
            new_mix = [max(1, int(targs[i]) // 2) for i in mix_at]
            if bs > 1 and bs // 2 > sum(new_mix):
                set_arg(targs, "--batch-size", bs // 2)
                set_arg(targs, "--grad-accum", ga * 2)
                for i, r in zip(mix_at, new_mix):
                    targs[i] = str(r)
        ck = newest_ckpt(out_dir)
        if ck:
            set_arg(targs, "--resume-from", ck)
        with open(log, "ab") as lo:
            lo.write(f"[driver] exit {rc} (oom={oom}); restart {restarts} from {ck}\n".encode())


if __name__ == "__main__":
    sys.exit(main())
