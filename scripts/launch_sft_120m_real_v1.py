#!/usr/bin/env python3
"""Pre-flight check and launcher for Parentheses 0.9-120M SFT Run Real-v1 (Conversational + Reasoning Recovery).

2,500-step deep pass from the clean Session 5 base checkpoint (step_9536_final.pt) on the
new data/processed_sft_real_v1 dataset:
  - 35% Math & step-by-step reasoning (Orca-Math, MetaMathQA, synthetic arithmetic CoT)
  - 30% Multi-turn conversation & persona (SmolTalk, Parentheses 0.9 identity by Joel Anthony)
  - 20% Native Dravidian instruction following & math (Aya, IndicQA, Dravidian CoT)
  - 15% Pretraining replay from data/processed_v5/train.bin (full causal loss mask)

Architecture: Recurrent Selective Linear Attention SSM (Mamba-2 / RWKV family).
Linear O(T) training via chunked state space duality, constant O(1) inference updates.
NOT a Transformer.

Usage:
  .venv\\Scripts\\python.exe scripts\\launch_sft_120m_real_v1.py --dry-run
  .venv\\Scripts\\python.exe scripts\\launch_sft_120m_real_v1.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_NAME = "parentheses-0.9-120M-sft-real-v1"
DATA_DIR = "data/processed_sft_real_v1"
OUT_DIR = f"checkpoints/{RUN_NAME}"
BASE_CKPT = "checkpoints/parentheses-0.9-120M/step_9536_final.pt"
START_STEP = 9536
NUM_STEPS = 2500
MAX_STEPS = START_STEP + NUM_STEPS  # 12036


def preflight_check():
    print("=" * 60)
    print("[preflight] Pre-flight verification for SFT Real-v1...")
    print("=" * 60)

    base_ckpt = REPO_ROOT / BASE_CKPT
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint missing: {base_ckpt}")
    print(f"[preflight] Found clean base checkpoint: {base_ckpt} ({base_ckpt.stat().st_size / 1e6:.1f} MB)")

    for name in ["train.bin", "train.mask.bin", "train_boundaries.npy", "manifest.json"]:
        p = REPO_ROOT / DATA_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"SFT real-v1 dataset file missing: {p}")
        print(f"[preflight] Found dataset file: {name} ({p.stat().st_size / 1e6:.2f} MB)")

    manifest_raw = (REPO_ROOT / DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_raw)
    if manifest.get("version") != "sft_real_v1":
        raise RuntimeError(f"Unexpected manifest version: {manifest.get('version')}")
    if manifest.get("esg_count", 1) != 0:
        raise RuntimeError("ESG tokens detected in dataset! Strict exclusion policy violated.")
    print(f"[preflight] Dataset manifest verified: {manifest['total_documents']:,} docs, {manifest['total_tokens']:,} tokens, zero ESG.")

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
    print(f"[preflight] Output directory ready: {out_dir}")
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
        "--lr", "3.5e-5",
        "--warmup-steps", "50",
        "--decay-start", "11500",
        "--min-lr", "5e-6",
        "--weight-decay", "0.05",
        "--seed", "20261020",
        "--max-steps", str(MAX_STEPS),
        "--ckpt-every", "250",
        "--log-every", "10",
    ]


def get_driver_cmd():
    train_flags = get_train_cmd()[3:]
    return [sys.executable, "-u", "scripts/overnight_driver.py", "--name", RUN_NAME, "--"] + train_flags


def main():
    parser = argparse.ArgumentParser(description="Launcher for Parentheses 0.9-120M SFT Real-v1")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    preflight_check()

    driver_cmd = get_driver_cmd()
    print("\n[launcher] Training command via overnight_driver:")
    print(" ".join(driver_cmd))

    if args.dry_run:
        print("\n[launcher] Dry-run complete. Exiting without launching training.")
        return 0

    print("\n[launcher] Launching SFT Real-v1 under overnight_driver...")
    log_file = REPO_ROOT / "out" / f"{RUN_NAME}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- SFT Real-v1 Run Launched at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.flush()
        proc = subprocess.Popen(
            driver_cmd,
            cwd=str(REPO_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    print(f"[launcher] Launched background PID: {proc.pid}")
    print(f"[launcher] Logging to: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
