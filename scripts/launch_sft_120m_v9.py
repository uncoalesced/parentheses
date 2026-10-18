#!/usr/bin/env python3
"""Pre-flight check and launcher for Parentheses 0.9-120M SFT Run 9 (P1.4d data revision).

Fresh 250-step pass from the Session 5 base checkpoint (step_9536_final.pt) on the
rebuilt data/processed_sft_120m_v3 dataset (34,177,306 tokens, 0% train/val leakage,
0 anchor templates). This is the pinned v5/v6/v7 recipe (warmup 20, decay 9636 ->
min-lr 5e-6, 250 steps) applied to the P1.4d build.

P1.4d = P1.4c data (12,000 raw drills, letter+digit product models, -anium minerals)
back at the 2x honesty_unknown upsample (16,748 docs) after the v8 continuation run
showed that the 3x upsample + 500 resumed steps degraded quality (abstain 5/10, 1
global failure, degenerate identity text). Fresh 250 steps on P1.4b produced the best
model so far (v7: abstain 7/10, 0 global failures); P1.4d keeps v7's recipe shape
while widening the drill name/shape coverage. Runs under scripts/overnight_driver.py
(crash resume x3, status file).

Architecture: Recurrent Selective Linear Attention SSM (Mamba-2 / RWKV family).
NOT a Transformer.

Usage:
  .venv\\Scripts\\python.exe scripts\\launch_sft_120m_v9.py --dry-run
  .venv\\Scripts\\python.exe scripts\\launch_sft_120m_v9.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_NAME = "parentheses-0.9-120M-sft-v9"
DATA_DIR = "data/processed_sft_120m_v3"
OUT_DIR = f"checkpoints/{RUN_NAME}"
BASE_CKPT = "checkpoints/parentheses-0.9-120M/step_9536_final.pt"
MAX_STEPS = 9786


def preflight_check():
    print("[preflight] Checking prerequisites for SFT Run 9 (P1.4d honesty data)...")

    base_ckpt = REPO_ROOT / BASE_CKPT
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint missing: {base_ckpt}")
    print(f"[preflight] Found base checkpoint: {base_ckpt} ({base_ckpt.stat().st_size / 1e6:.1f} MB)")

    for name in ["train.bin", "train.mask.bin", "train_boundaries.npy", "manifest.json"]:
        p = REPO_ROOT / DATA_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"SFT v3 dataset file missing: {p}")
        print(f"[preflight] Found dataset file: {name} ({p.stat().st_size / 1e6:.2f} MB)")

    manifest = (REPO_ROOT / DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    if "honesty_unknown" not in manifest:
        raise RuntimeError("Dataset manifest lacks honesty_unknown category (stale pre-P1.4 build?)")
    if "e51bf45b" not in manifest:
        raise RuntimeError("Dataset manifest is not the P1.4d rebuild (train_sha256 e51bf45b... expected)")
    print("[preflight] Dataset manifest carries P1.4d honesty categories and expected sha256.")

    for name in ["train.bin", "train_boundaries.npy"]:
        p = REPO_ROOT / "data" / "processed_v5" / name
        if not p.exists():
            raise FileNotFoundError(f"Replay file missing: {p}")
    print("[preflight] Found replay data: data/processed_v5/train.bin & train_boundaries.npy")

    tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
    if not (tok_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"Tokenizer missing in {tok_dir}")
    print(f"[preflight] Found tokenizer: {tok_dir}")

    out_dir = REPO_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] Target output directory ready: {out_dir}")
    print("[preflight] All preflight checks passed successfully.")


def get_train_cmd():
    return [
        sys.executable, "-u", "train.py",
        "--preset", "parentheses-0.9-120M",
        "--data", f"{DATA_DIR}/train.bin",
        "--sft-spans", f"{DATA_DIR}/train.mask.bin",
        "--boundaries", f"{DATA_DIR}/train_boundaries.npy",
        "--mix", "data/processed_v5/train.bin", "1", "data/processed_v5/train_boundaries.npy",
        "--tokenizer", "data/tokenizer/v3_32k",
        "--resume-from", BASE_CKPT,
        "--out-dir", OUT_DIR,
        "--batch-size", "8",
        "--grad-accum", "4",
        "--ce-rows", "4096",
        "--grad-checkpointing",
        "--lr", "2.5e-5",
        "--warmup-steps", "20",
        "--decay-start", "9636",
        "--min-lr", "5e-6",
        "--weight-decay", "0.05",
        "--seed", "20261018",
        "--max-steps", str(MAX_STEPS),
        "--ckpt-every", "50",
        "--log-every", "5",
    ]


def get_driver_cmd():
    train_flags = get_train_cmd()[3:]
    return [sys.executable, "-u", "scripts/overnight_driver.py", "--name", RUN_NAME, "--"] + train_flags


def main():
    preflight_check()
    cmd = get_driver_cmd()
    print("\n[launch] Driver command:")
    print(" ".join(cmd))

    if "--dry-run" in sys.argv:
        print("\n[launch] Dry run requested. Exiting without launching training.")
        return 0

    lock_file = REPO_ROOT / "out" / "locks" / "GPU.lock"
    status_file = REPO_ROOT / "out" / "overnight_status.txt"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    if lock_file.exists():
        raise RuntimeError(f"GPU lock exists at {lock_file}: {lock_file.read_text(encoding='utf-8')}")

    print("\n[launch] Acquiring GPU lock...")
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lock_file.write_text(f"opencode sft-v9-train {now_str}\n", encoding="utf-8")

    try:
        print("[launch] Launching SFT Run 9 on GPU (driver with crash resume)...")
        status_file.write_text(
            f"stage: SFT Run 9 (P1.4d honesty data, 34.2M tokens)\nstep: 9536\nstatus: RUNNING\nupdated: {now_str}\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
        print(f"[launch] Started driver process PID {proc.pid}")
        rc = proc.wait()
        if rc == 0:
            status_file.write_text(
                f"stage: SFT Run 9 DONE (step {MAX_STEPS})\nstep: {MAX_STEPS}\nstatus: DONE\n"
                f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
        else:
            status_file.write_text(
                f"stage: SFT Run 9 FAILED (rc={rc})\nstatus: FAILED\n"
                f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
        return rc
    finally:
        if lock_file.exists():
            print("[launch] Releasing GPU lock...")
            try:
                lock_file.unlink()
            except Exception as e:
                print(f"[launch] Warning: failed to unlink GPU lock: {e}")


if __name__ == "__main__":
    sys.exit(main())
