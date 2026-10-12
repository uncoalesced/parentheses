#!/usr/bin/env python3
"""Session 2 Pre-flight Verification Script for Parentheses 0.9-120M.

Verifies all hardware, locks, dataset integrity, tokenizer, and resume checkpoint
before kicking off Session 2 (Steps 1,908 -> 3,816 / 240M Tokens).
"""

import os
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def check_mark(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"

def main():
    print("=" * 65)
    print("PARENTHESES 0.9-120M: SESSION 2 PRE-FLIGHT VERIFICATION MATRIX")
    print("=" * 65)
    all_ok = True

    # 1. Checkpoint verification
    ckpt_path = REPO_ROOT / "checkpoints" / "parentheses-0.9-120M" / "step_1907_final.pt"
    ckpt_ok = ckpt_path.is_file() and ckpt_path.stat().st_size > 1_000_000_000
    print(f"{check_mark(ckpt_ok)} Resume Checkpoint: {ckpt_path.name}")
    if ckpt_ok:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        has_model = "model" in ckpt
        has_opt = "optimizer" in ckpt and ckpt["optimizer"] is not None
        step = ckpt.get("step", -1)
        opt_ok = has_model and has_opt and step == 1907
        print(f"       -> Step: {step}, Optimizer Momentum Intact: {has_opt}")
        if not opt_ok:
            all_ok = False
    else:
        all_ok = False

    # 2. Tokenizer verification
    tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
    tok_ok = (tok_dir / "tokenizer.json").is_file() and (tok_dir / "meta.json").is_file()
    print(f"{check_mark(tok_ok)} Tokenizer v3_32k: {tok_dir}")
    if not tok_ok:
        all_ok = False

    # 3. Training data & boundaries
    train_bin = REPO_ROOT / "data" / "processed_v5" / "train.bin"
    train_bound = REPO_ROOT / "data" / "processed_v5" / "train_boundaries.npy"
    data_ok = train_bin.is_file() and train_bound.is_file()
    if data_ok:
        tokens = train_bin.stat().st_size // 2
        print(f"{check_mark(data_ok)} Corpus v5 train.bin: {tokens:,} tokens on disk")
    else:
        print(f"{check_mark(data_ok)} Corpus v5 train.bin / boundaries missing on E:")
        all_ok = False

    # 4. GPU Lock check
    lock_file = REPO_ROOT / "out" / "locks" / "GPU.lock"
    lock_free = not lock_file.is_file()
    print(f"{check_mark(lock_free)} GPU Lock Status: {'Free' if lock_free else 'LOCKED by ' + lock_file.read_text().strip()}")
    if not lock_free:
        all_ok = False

    # 5. CUDA Hardware check
    cuda_avail = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    print(f"{check_mark(cuda_avail)} CUDA Accelerator: {cuda_name}")
    if not cuda_avail:
        all_ok = False

    # 6. Drive E & F Storage headroom
    import shutil
    e_free_gb = shutil.disk_usage("E:").free / (1024**3)
    f_free_gb = shutil.disk_usage("F:").free / (1024**3) if os.path.exists("F:") else 0
    print(f"[INFO] Storage Headroom: Drive E: {e_free_gb:.2f} GB Free | Drive F: {f_free_gb:.2f} GB Free")
    # 13 checkpoints every 150 steps + final + SFT-trial final, 1.37 GB each ~= 20.6 GB
    disk_ok = e_free_gb >= 22
    print(f"{check_mark(disk_ok)} Drive E: >= 22 GB free for session-2 + SFT-trial checkpoints")
    all_ok &= disk_ok

    ready_ok = (REPO_ROOT / "out" / "locks" / "V5_DATA_READY").is_file()
    print(f"{check_mark(ready_ok)} out/locks/V5_DATA_READY present")
    all_ok &= ready_ok

    # launch_120M_session.ps1 resumes from the newest-by-mtime checkpoint
    pts = sorted((REPO_ROOT / "checkpoints" / "parentheses-0.9-120M").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    newest_ok = bool(pts) and pts[-1].name == "step_1907_final.pt"
    print(f"{check_mark(newest_ok)} Newest checkpoint is step_1907_final.pt ({pts[-1].name if pts else 'none'})")
    all_ok &= newest_ok

    # 7. NPU Byte Filter artifact check
    npu_filter_path = REPO_ROOT / "checkpoints" / "npu_byte_filter" / "byte_filter.pt"
    npu_ok = npu_filter_path.is_file()
    print(f"{check_mark(npu_ok)} NPU ByteFilterCNN Artifact: {npu_filter_path.name if npu_ok else 'MISSING'}")

    print("=" * 65)
    if all_ok:
        print("RESULT: ALL CRITICAL GATES PASSED -> READY FOR SESSION 2 LAUNCH")
    else:
        print("RESULT: PRE-FLIGHT BLOCKED ON ONE OR MORE FAILED GATES")
    print("=" * 65)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
