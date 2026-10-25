#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/watchdog_gate_19000.py

Automated Watchdog Gate for Step 19,000 Landing (Project Tanzinite / slab_v3):
  1. Polls E:/Parentheses/parentheses-0.9/checkpoints/parentheses-0.9-130M-v3/step_19000.pt
  2. Waits for full file write (1,780,875,377 bytes)
  3. Audits raw point weights on CPU (val loss, PPL, slab_v3 gates)
  4. Computes SWA-3 model soup (step_18000 + step_18500 + step_19000 -> step_19000_swa3.pt)
  5. Audits SWA-3 smoothed weights on CPU
  6. Updates overnight_status.txt
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Strictly CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

REPO_ROOT = Path("E:/Parentheses/parentheses-0.9")
E_CKPT_DIR = REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3"
I_CKPT_DIR = Path("I:/parentheses/checkpoints/parentheses-0.9-130M-v3")
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

TARGET_CKPT = E_CKPT_DIR / "step_19000.pt"
TARGET_SWA3 = I_CKPT_DIR / "step_19000_swa3.pt"
STATUS_FILE = REPO_ROOT / "out" / "overnight_status.txt"

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Watchdog Gate 19,000 armed.", flush=True)
print(f"Waiting for {TARGET_CKPT}...", flush=True)

# 1. Wait for checkpoint to appear and finish writing
while True:
    if TARGET_CKPT.exists():
        size1 = TARGET_CKPT.stat().st_size
        if size1 >= 1780000000:
            time.sleep(3)
            size2 = TARGET_CKPT.stat().st_size
            if size1 == size2:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint landed! Size: {size2:,} bytes.", flush=True)
                break
    time.sleep(3)

time.sleep(2)

# 2. Run CPU Audit on raw step_19000.pt
print("Running CPU audit on raw step_19000.pt...", flush=True)
cmd_raw = [
    str(PYTHON_EXE),
    str(REPO_ROOT / "scripts" / "eval_130m_audit.py"),
    "--ckpt", str(TARGET_CKPT),
    "--eval-loss",
    "--batches", "10"
]
res_raw = subprocess.run(cmd_raw, capture_output=True, text=True, cwd=str(REPO_ROOT))
print("Raw Audit Output:", flush=True)
print(res_raw.stdout, flush=True)

# 3. Run SWA-3 merge
print("Running SWA-3 merge for steps 18000 + 18500 + 19000...", flush=True)
ckpt_18000 = I_CKPT_DIR / "step_18000.pt"
if not ckpt_18000.exists():
    ckpt_18000 = E_CKPT_DIR / "step_18000.pt"

ckpt_18500 = I_CKPT_DIR / "step_18500.pt"
if not ckpt_18500.exists():
    ckpt_18500 = E_CKPT_DIR / "step_18500.pt"

cmd_swa = [
    str(PYTHON_EXE),
    str(REPO_ROOT / "scripts" / "merge_swa_checkpoints.py"),
    "--ckpts", str(ckpt_18000), str(ckpt_18500), str(TARGET_CKPT),
    "--out", str(TARGET_SWA3)
]
res_swa = subprocess.run(cmd_swa, capture_output=True, text=True, cwd=str(REPO_ROOT))
print("SWA Merge Output:", flush=True)
print(res_swa.stdout, flush=True)

# 4. Run CPU Audit on SWA-3 checkpoint
print("Running CPU audit on step_19000_swa3.pt...", flush=True)
cmd_swa_eval = [
    str(PYTHON_EXE),
    str(REPO_ROOT / "scripts" / "eval_130m_audit.py"),
    "--ckpt", str(TARGET_SWA3),
    "--eval-loss",
    "--batches", "10"
]
res_swa_eval = subprocess.run(cmd_swa_eval, capture_output=True, text=True, cwd=str(REPO_ROOT))
print("SWA Audit Output:", flush=True)
print(res_swa_eval.stdout, flush=True)

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 19,000 audit and SWA-3 merge complete.", flush=True)
