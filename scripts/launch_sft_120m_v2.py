#!/usr/bin/env python3
"""Pre-flight check and launcher for Parentheses 0.9-120M SFT Run 2.

Trains on the clean data/processed_sft_120m_v2 dataset (32.2M tokens, 0% train/val leakage,
genuine Bespoke-Stratos reasoning, 100% attested low-resource vocabulary, diverse abstention)
starting from the Session 5 base checkpoint step_9536_final.pt.

Architecture: Recurrent Selective Linear Attention SSM (Mamba-2 / RWKV family).
NOT a Transformer.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def preflight_check():
    print("[preflight] Checking prerequisites for SFT Run 2...")
    
    # 1. Base checkpoint
    base_ckpt = REPO_ROOT / "checkpoints" / "parentheses-0.9-120M" / "step_9536_final.pt"
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint missing: {base_ckpt}")
    print(f"[preflight] Found base checkpoint: {base_ckpt} ({base_ckpt.stat().st_size / 1e6:.1f} MB)")
    
    # 2. Dataset files
    sft_dir = REPO_ROOT / "data" / "processed_sft_120m_v2"
    for name in ["train.bin", "train.mask.bin", "train_boundaries.npy", "manifest.json"]:
        p = sft_dir / name
        if not p.exists():
            raise FileNotFoundError(f"SFT v2 dataset file missing: {p}")
        print(f"[preflight] Found dataset file: {name} ({p.stat().st_size / 1e6:.2f} MB)")
        
    # 3. Pretraining replay files
    replay_data = REPO_ROOT / "data" / "processed_v5" / "train.bin"
    replay_bound = REPO_ROOT / "data" / "processed_v5" / "train_boundaries.npy"
    if not replay_data.exists() or not replay_bound.exists():
        raise FileNotFoundError(f"Replay files missing in data/processed_v5/")
    print(f"[preflight] Found replay data: {replay_data} & {replay_bound}")
    
    # 4. Tokenizer
    tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
    if not (tok_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"Tokenizer missing in {tok_dir}")
    print(f"[preflight] Found tokenizer: {tok_dir}")
    
    # 5. Output directory
    out_dir = REPO_ROOT / "checkpoints" / "parentheses-0.9-120M-sft-v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] Target output directory ready: {out_dir}")
    print("[preflight] All preflight checks passed successfully.")

def get_train_cmd():
    py_exe = sys.executable
    cmd = [
        py_exe, "-u", "train.py",
        "--preset", "parentheses-0.9-120M",
        "--data", "data/processed_sft_120m_v2/train.bin",
        "--sft-spans", "data/processed_sft_120m_v2/train.mask.bin",
        "--boundaries", "data/processed_sft_120m_v2/train_boundaries.npy",
        "--mix", "data/processed_v5/train.bin", "1", "data/processed_v5/train_boundaries.npy",
        "--tokenizer", "data/tokenizer/v3_32k",
        "--resume-from", "checkpoints/parentheses-0.9-120M/step_9536_final.pt",
        "--out-dir", "checkpoints/parentheses-0.9-120M-sft-v2",
        "--batch-size", "8",
        "--grad-accum", "4",
        "--ce-rows", "4096",
        "--grad-checkpointing",
        "--lr", "2.5e-5",
        "--warmup-steps", "20",
        "--decay-start", "9636",
        "--min-lr", "5e-6",
        "--weight-decay", "0.05",
        "--seed", "20261017",
        "--max-steps", "9786",
        "--ckpt-every", "50",
        "--log-every", "5",
    ]
    return cmd

def main():
    preflight_check()
    cmd = get_train_cmd()
    print("\n[launch] Launch command:")
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
    import time
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lock_file.write_text(f"antigravity sft-v2-train {now_str}\n", encoding="utf-8")
    
    try:
        print("[launch] Launching SFT Run 2 on GPU...")
        status_file.write_text(f"stage: SFT Run 2 (32.2M tokens)\nstep: 9536\nstatus: RUNNING\nupdated: {now_str}\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
        print(f"[launch] Started training process PID {proc.pid}")
        rc = proc.wait()
        if rc == 0:
            status_file.write_text(f"stage: SFT Run 2 DONE (step 9786)\nstep: 9786\nstatus: DONE\nupdated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        else:
            status_file.write_text(f"stage: SFT Run 2 FAILED (rc={rc})\nstatus: FAILED\nupdated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
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
