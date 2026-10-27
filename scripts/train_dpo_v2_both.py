#!/usr/bin/env python3
"""
scripts/train_dpo_v2_both.py
Orchestrates DPO v2 training on:
  1. Candidate 1 (B_dpo_v2): from raw terminal SFT (step_1999_final.pt)
  2. Candidate 2 (B_dpo_ag_v2): from SWA-3 soup (step_2000_sft_notrunc_swa3.pt)
Manages out/locks/GPU.lock, logs training output, and triggers evaluations.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"E:\Parentheses\parentheses-0.9")
PY_EXE = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
LOCK_FILE = REPO_ROOT / "out" / "locks" / "GPU.lock"
PAIRS_FILE = str(REPO_ROOT / "out" / "dpo_pairs_v2_900.jsonl")

CANDIDATES = [
    {
        "name": "B_dpo_v2",
        "policy": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-sft-notrunc" / "step_1999_final.pt"),
        "out_dir": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-raw"),
        "log_file": str(REPO_ROOT / "out" / "dpo_v2_raw_train.log"),
    },
    {
        "name": "B_dpo_ag_v2",
        "policy": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-sft-notrunc" / "step_2000_sft_notrunc_swa3.pt"),
        "out_dir": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-swa3"),
        "log_file": str(REPO_ROOT / "out" / "dpo_v2_swa3_train.log"),
    },
]


def acquire_lock(name: str):
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    msg = f"antigravity {name} (started {time.strftime('%Y-%m-%d %H:%M:%S')})\n"
    LOCK_FILE.write_text(msg, encoding="utf-8")
    print(f"[LOCK] Acquired GPU.lock for {name}")


def release_lock():
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
            print("[LOCK] Released GPU.lock")
        except Exception as e:
            print(f"[WARN] Failed to release lock: {e}")


def run_dpo(cand: dict):
    acquire_lock(cand["name"])
    try:
        cmd = [
            PY_EXE,
            str(REPO_ROOT / "scripts" / "train_dpo_130m_v3.py"),
            "--policy", cand["policy"],
            "--pairs", PAIRS_FILE,
            "--out-dir", cand["out_dir"],
            "--tokenizer", str(REPO_ROOT / "data" / "tokenizer" / "v3_32k"),
            "--steps", "100",
            "--batch-size", "4",
            "--lr", "5e-7",
            "--beta", "0.1",
            "--sft-coef", "0.3",
            "--max-len", "1024",
            "--device", "cuda",
            "--ckpt-every", "50",
            "--full-eval-every", "50",
            "--log-every", "10",
        ]
        print(f"\n{'='*70}\n[TRAIN] Launching {cand['name']}\nPolicy: {cand['policy']}\nOutDir: {cand['out_dir']}\n{'='*70}")
        t0 = time.time()
        with open(cand["log_file"], "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8"
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
            proc.wait()
        dt = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"{cand['name']} failed with exit code {proc.returncode}")
        print(f"[OK] {cand['name']} finished in {dt:.1f}s")
    finally:
        release_lock()


def main():
    print(f"[START] DPO v2 Candidate Pipeline on {PAIRS_FILE}")
    for cand in CANDIDATES:
        run_dpo(cand)
    print("\n[COMPLETE] Both candidates trained successfully!")


if __name__ == "__main__":
    main()
