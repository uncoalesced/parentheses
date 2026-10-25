#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/archive_checkpoints_130m.py

Continuous Checkpoint Archiver & Disk Protection Daemon for Parentheses 0.9-130M-v3:
  - Source: E:\\Parentheses\\parentheses-0.9\\checkpoints\\parentheses-0.9-130M-v3
  - Archive: I:\\parentheses\\checkpoints\\parentheses-0.9-130M-v3
  - Strategy:
      1. Every new checkpoint (step_*.pt) is copied to expansion storage (I: drive, 425+ GB free).
      2. File integrity is verified (exact file size matching).
      3. On fast NVMe (E:), retains the latest K checkpoints (default K=3) for instant resumption.
      4. Older verified checkpoints on E: are pruned to keep E: drive usage under 5.5 GB total.
      5. Every 2,500-step milestone checkpoint is permanently preserved on I:.
      6. Telemetry logged to out/checkpoint_archiver.log.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

SRC_DEFAULT = Path("E:/Parentheses/parentheses-0.9/checkpoints/parentheses-0.9-130M-v3")
DST_DEFAULT = Path("I:/parentheses/checkpoints/parentheses-0.9-130M-v3")
LOG_DEFAULT = Path("E:/Parentheses/parentheses-0.9/out/checkpoint_archiver.log")


def log(msg: str, log_file: Path | None = None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def get_step_from_name(name: str) -> int:
    m = re.search(r"step_(\d+)", name)
    return int(m.group(1)) if m else -1


def run_archive_cycle(src_dir: Path, dst_dir: Path, keep_local: int = 3, log_file: Path | None = None) -> int:
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find candidate checkpoint files in src_dir (ignoring SWA / model soup derivatives)
    all_files = [f for f in src_dir.glob("step_*.pt") if f.is_file() and "_swa" not in f.name]
    if not all_files:
        return 0

    now = time.time()
    src_files = []
    for f in all_files:
        try:
            st = f.stat()
            file_age = now - st.st_mtime
            # Partial-Write Guard:
            # Checkpoint must be >= 1.70 GB and modified at least 15s ago,
            # or at least 60s old if smaller. This ensures train.py (PID 6216)
            # has completely finished streaming torch.save() to disk before
            # the file is mirrored or counted toward the keep_local threshold.
            if st.st_size >= 1_700_000_000 and file_age >= 15.0:
                src_files.append(f)
            elif file_age >= 60.0 and st.st_size >= 100_000_000:
                src_files.append(f)
            else:
                log(f"[guard] Actively writing or incomplete: {f.name} ({st.st_size / 1e6:.1f} MB, age {file_age:.1f}s) - held until complete.", log_file)
        except Exception as e:
            log(f"[warn] Could not stat {f.name}: {e}", log_file)

    if not src_files:
        return 0

    # Sort files by step number ascending
    src_files.sort(key=lambda f: get_step_from_name(f.name))

    archived_count = 0
    # 1. Mirror to destination if not already present or incomplete
    for src_f in src_files:
        dst_f = dst_dir / src_f.name
        src_size = src_f.stat().st_size

        needs_copy = False
        if not dst_f.exists():
            needs_copy = True
        elif dst_f.stat().st_size != src_size:
            log(f"[sync] Size mismatch for {src_f.name} (src: {src_size}, dst: {dst_f.stat().st_size}). Re-copying...", log_file)
            needs_copy = True

        if needs_copy:
            log(f"[copy] Copying {src_f.name} ({src_size / 1e9:.2f} GB) to {dst_dir}...", log_file)
            shutil.copy2(src_f, dst_f)
            # Verify copy
            if dst_f.exists() and dst_f.stat().st_size == src_size:
                log(f"[verified] {src_f.name} safely mirrored on I: drive.", log_file)
                archived_count += 1
            else:
                log(f"[error] Verification failed for {src_f.name} on I: drive!", log_file)

    # 2. Prune older checkpoints on E: keeping only the latest `keep_local` fully-verified checkpoints
    if len(src_files) > keep_local:
        to_prune = src_files[:-keep_local]
        for prune_f in to_prune:
            dst_f = dst_dir / prune_f.name
            # Strict safety check: ONLY delete if target exists on I: and has identical non-zero size
            if dst_f.exists() and dst_f.stat().st_size == prune_f.stat().st_size and dst_f.stat().st_size > 500_000_000:
                log(f"[prune] Rotating {prune_f.name} from E: (safely preserved on I:). Reclaiming {prune_f.stat().st_size / 1e9:.2f} GB...", log_file)
                try:
                    prune_f.unlink()
                    log(f"[reclaimed] Successfully deleted {prune_f.name} from E: drive.", log_file)
                except Exception as e:
                    log(f"[warn] Could not remove {prune_f.name}: {e}", log_file)
            else:
                log(f"[skip-prune] Safety abort: {prune_f.name} not fully verified on I: drive; skipping deletion.", log_file)

    return archived_count


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Archiver Daemon for Parentheses 0.9-130M-v3")
    parser.add_argument("--src", type=str, default=str(SRC_DEFAULT), help="Source directory on E:")
    parser.add_argument("--dst", type=str, default=str(DST_DEFAULT), help="Archive directory on I:")
    parser.add_argument("--log", type=str, default=str(LOG_DEFAULT), help="Log file")
    parser.add_argument("--keep-local", type=int, default=3, help="Number of latest checkpoints to keep on E:")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    log_file = Path(args.log)

    log(f"=== Checkpoint Archiver Started ===", log_file)
    log(f"  Source:     {src_dir}", log_file)
    log(f"  Archive:    {dst_dir}", log_file)
    log(f"  Keep Local: {args.keep_local} latest checkpoints", log_file)

    if args.once:
        run_archive_cycle(src_dir, dst_dir, keep_local=args.keep_local, log_file=log_file)
        log("Single pass completed.", log_file)
        return

    while True:
        try:
            run_archive_cycle(src_dir, dst_dir, keep_local=args.keep_local, log_file=log_file)
        except Exception as e:
            log(f"[error] Exception in archive loop: {e}", log_file)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
