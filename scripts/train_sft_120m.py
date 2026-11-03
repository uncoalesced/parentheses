#!/usr/bin/env python3
"""Launcher for the 2,000-step SFT Sprint on Parentheses 0.9-120M.

Embeds structured <think> reasoning traces, STEM problem-solving,
0% refusal open data, and preserves native Dravidian fluency.

Usage:
    python scripts/train_sft_120m.py --data-dir data/processed_sft_stem_open_v1 --batch-size 4 --grad-accum 4 --max-steps 2000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Parentheses 0.9-120M SFT Sprint")
    parser.add_argument("--data-dir", default="data/processed_sft_stem_open_v1", help="Path to tokenized SFT directory")
    parser.add_argument("--resume-from", default="checkpoints/parentheses-0.9-120M-sft-real-v1/step_12035_final.pt", help="Base checkpoint to resume")
    parser.add_argument("--out-dir", default="checkpoints/parentheses-0.9-120M-sft-stem-open", help="Output checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=2000, help="Additional steps to train")
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak SFT learning rate")
    parser.add_argument("--min-lr", type=float, default=5e-6, help="Floor learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Matrix weight decay")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir
    train_bin = data_dir / "train.bin"
    mask_bin = data_dir / "train.mask.bin"
    bnd_npy = data_dir / "train_boundaries.npy"
    resume_ckpt = REPO_ROOT / args.resume_from

    print("=" * 70)
    print("PARENTHESES 0.9-120M SFT VALIDATION SPRINT")
    print("=" * 70)

    if not resume_ckpt.exists():
        sys.exit(f"[ERROR] Resume checkpoint not found: {resume_ckpt}")
    if not train_bin.exists() or not mask_bin.exists() or not bnd_npy.exists():
        sys.exit(f"[ERROR] Dataset files missing in {data_dir}")

    # Inspect starting step from checkpoint
    import torch
    ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    start_step = ckpt.get("step", 0)
    target_step = start_step + args.max_steps
    decay_start = start_step + int(args.max_steps * 0.75)

    print(f"Resume Checkpoint: {resume_ckpt.name} (Step {start_step:,})")
    print(f"Dataset Bin      : {train_bin} ({train_bin.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Steps to Run     : {args.max_steps:,} (from {start_step:,} to {target_step:,})")
    print(f"Batch Size       : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} docs/step")
    print(f"Learning Rate    : {args.lr:.1e} -> {args.min_lr:.1e} (cooldown starts at step {decay_start:,})")
    print(f"Output Directory : {args.out_dir}")

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "train.py"),
        "--preset", "parentheses-0.9-120M",
        "--data", str(train_bin),
        "--sft-spans", str(mask_bin),
        "--boundaries", str(bnd_npy),
        "--tokenizer", str(REPO_ROOT / "data" / "tokenizer" / "v3_32k"),
        "--resume-from", str(resume_ckpt),
        "--out-dir", str(REPO_ROOT / args.out_dir),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--ce-rows", "4096",
        "--grad-checkpointing",
        "--lr", str(args.lr),
        "--warmup-steps", "30",
        "--decay-start", str(decay_start),
        "--min-lr", str(args.min_lr),
        "--weight-decay", str(args.weight_decay),
        "--seed", str(args.seed),
        "--max-steps", str(target_step),
        "--ckpt-every", "500",
        "--log-every", "10",
    ]

    # Pretrain replay mix if available
    v5_train = REPO_ROOT / "data" / "processed_v5" / "train.bin"
    v5_bnd = REPO_ROOT / "data" / "processed_v5" / "train_boundaries.npy"
    if v5_train.exists() and v5_bnd.exists():
        cmd.extend(["--mix", str(v5_train), "1", str(v5_bnd)])
        print(f"Pretrain Replay  : Enabled (1 row/micro-batch from {v5_train.name})")

    print("\nCommand to execute:")
    print(" ".join(cmd))

    if args.dry_run:
        print("\n[DRY-RUN] Exiting without running.")
        return 0

    out_path = REPO_ROOT / args.out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = REPO_ROOT / "out" / "sft_120m_stem_open_sprint.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLaunching training in background, streaming to: {log_path}...")
    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n--- SFT Sprint Launched at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f_log.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=f_log, stderr=subprocess.STDOUT)

    print(f"[SUCCESS] Launched background training process PID: {proc.pid}")
    print(f"Monitor with: Get-Content -Path '{log_path}' -Wait -Tail 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
