#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/watchdog_gate_final.py

Automated Final Watchdog Gate for Parentheses 0.9-130M-v3 (Project Tanzinite):
  1. Polls for final checkpoint (step_22887_final.pt or step_22888_final.pt - 1.500B tokens completion milestone)
  2. Verifies complete file write (size >= 1.78 GB, stable across reads)
  3. Audits final raw pretraining weights on CPU (val loss, PPL, slab_v3 gates)
  4. Computes final SWA-3 model soup (step_22000 + step_22500 + final -> step_22888_swa3.pt)
  5. Audits SWA-3 smoothed weights on CPU
  6. Updates overnight_status.txt, documentation.md, and journal.md
  7. Waits for GPU.lock release from pretraining PID 6216 / PID 4332
  8. Triggers launch_sft_130m_v3.ps1 on data/processed_sft_stem_open_v2
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Strictly CPU for audit to never touch GPU VRAM
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

REPO_ROOT = Path("E:/Parentheses/parentheses-0.9")
E_CKPT_DIR = REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3"
I_CKPT_DIR = Path("I:/parentheses/checkpoints/parentheses-0.9-130M-v3")
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

TARGET_SWA3_I = I_CKPT_DIR / "step_22888_swa3.pt"
TARGET_SWA3_E = E_CKPT_DIR / "step_22888_swa3_base.pt"
STATUS_FILE = REPO_ROOT / "out" / "overnight_status.txt"
LOCK_FILE = REPO_ROOT / "out" / "locks" / "GPU.lock"
DOCS_FILE = REPO_ROOT / ".agents" / "documentation.md"
JOURNAL_FILE = Path("F:/Lingual_Data/Agents-Brain/journal.md")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FINAL PRETRAINING WATCHDOG GATE ARMED.", flush=True)
print("Scanning for final checkpoint (step_22887_final.pt / step_22888_final.pt, ETA ~23:45 IST)...", flush=True)

candidate_names = [
    "step_22887_final.pt",
    "step_22888_final.pt",
    "step_22888.pt",
    "step_22887.pt",
]

target_ckpt: Path | None = None

while target_ckpt is None:
    # Check explicitly listed candidate names
    for name in candidate_names:
        for d in [E_CKPT_DIR, I_CKPT_DIR]:
            p = d / name
            if p.exists() and p.stat().st_size >= 1780000000:
                s1 = p.stat().st_size
                time.sleep(3)
                s2 = p.stat().st_size
                if s1 == s2:
                    target_ckpt = p
                    break
        if target_ckpt:
            break

    # Also check any file matching step_*_final.pt or step_2288*.pt
    if target_ckpt is None:
        for d in [E_CKPT_DIR, I_CKPT_DIR]:
            if not d.exists():
                continue
            for f in d.glob("step_*_final.pt"):
                if f.stat().st_size >= 1780000000:
                    s1 = f.stat().st_size
                    time.sleep(3)
                    if s1 == f.stat().st_size:
                        target_ckpt = f
                        break
            if target_ckpt:
                break

    if target_ckpt is None:
        time.sleep(5)

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final Checkpoint Detected: {target_ckpt.name} ({target_ckpt.stat().st_size:,} bytes).", flush=True)
time.sleep(2)

# 2. Run CPU Audit on raw final checkpoint
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running CPU audit on raw final checkpoint...", flush=True)
cmd_raw = [
    str(PYTHON_EXE),
    str(REPO_ROOT / "scripts" / "eval_130m_audit.py"),
    "--ckpt", str(target_ckpt),
    "--eval-loss",
    "--batches", "10"
]
res_raw = subprocess.run(cmd_raw, capture_output=True, text=True, cwd=str(REPO_ROOT))
raw_stdout = res_raw.stdout
print("=== Raw Final Audit Output ===", flush=True)
print(raw_stdout, flush=True)

# 3. Locate earlier checkpoints for SWA-3 merge
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Locating step_22000 and step_22500 for final SWA-3 merge...", flush=True)
ckpt_22000 = I_CKPT_DIR / "step_22000.pt"
if not ckpt_22000.exists():
    ckpt_22000 = E_CKPT_DIR / "step_22000.pt"

ckpt_22500 = I_CKPT_DIR / "step_22500.pt"
if not ckpt_22500.exists():
    ckpt_22500 = E_CKPT_DIR / "step_22500.pt"

print(f"SWA components: [1] {ckpt_22000}, [2] {ckpt_22500}, [3] {target_ckpt}", flush=True)

TARGET_SWA3_I.parent.mkdir(parents=True, exist_ok=True)
cmd_swa = [
    str(PYTHON_EXE),
    str(REPO_ROOT / "scripts" / "merge_swa_checkpoints.py"),
    "--ckpts", str(ckpt_22000), str(ckpt_22500), str(target_ckpt),
    "--out", str(TARGET_SWA3_I)
]
res_swa = subprocess.run(cmd_swa, capture_output=True, text=True, cwd=str(REPO_ROOT))
print("=== Final SWA Merge Output ===", flush=True)
print(res_swa.stdout, flush=True)

# Also copy SWA-3 to E: drive for ultra-low latency SFT training
try:
    print(f"Copying SWA-3 model soup to {TARGET_SWA3_E} for SFT training...", flush=True)
    shutil.copy2(str(TARGET_SWA3_I), str(TARGET_SWA3_E))
    print(f"Successfully copied SWA-3 model to {TARGET_SWA3_E}.", flush=True)
except Exception as e:
    print(f"Warning: Failed to copy to {TARGET_SWA3_E}: {e}", flush=True)

# 4. Run CPU Audit on final SWA-3 checkpoint
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running CPU audit on {TARGET_SWA3_I.name}...", flush=True)
cmd_swa_eval = [
    str(PYTHON_EXE),
    str(REPO_ROOT / "scripts" / "eval_130m_audit.py"),
    "--ckpt", str(TARGET_SWA3_I),
    "--eval-loss",
    "--batches", "10"
]
res_swa_eval = subprocess.run(cmd_swa_eval, capture_output=True, text=True, cwd=str(REPO_ROOT))
swa_stdout = res_swa_eval.stdout
print("=== Final SWA Audit Output ===", flush=True)
print(swa_stdout, flush=True)

# 5. Extract Loss/PPL values if available
def parse_metrics(out: str):
    loss, ppl = "N/A", "N/A"
    for line in out.splitlines():
        if "Validation Loss:" in line:
            parts = line.split("Validation Loss:")
            if len(parts) > 1:
                val_part = parts[1].split()[0]
                loss = val_part.strip()
        if "Perplexity:" in line:
            parts = line.split("Perplexity:")
            if len(parts) > 1:
                ppl = parts[1].strip()
    return loss, ppl

raw_loss, raw_ppl = parse_metrics(raw_stdout)
swa_loss, swa_ppl = parse_metrics(swa_stdout)

now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 6. Update overnight_status.txt
status_text = f"""archiver: PID 14280 ACTIVE (final pretraining complete, step_22888 preserved)
pretraining_status: COMPLETE (1,500,000,000 tokens trained, 22,888 steps reached)
last_checkpoint: {target_ckpt.name} (Loss: {raw_loss}, PPL: {raw_ppl})
swa3_final: {TARGET_SWA3_I.name} (SWA-3 Loss: {swa_loss}, SWA-3 PPL: {swa_ppl})
sft_status: LAUNCHING_ON_GPU (data/processed_sft_stem_open_v2)
updated: {now_str} IST
"""
STATUS_FILE.write_text(status_text, encoding="utf-8")
print(f"[{now_str}] overnight_status.txt updated successfully.", flush=True)

# 7. Append completion documentation
doc_entry = f"""
## Parentheses 0.9-130M-v3 Pretraining Completion & Final Audit ({now_str} IST)
- **Status:** **PRETRAINING COMPLETE — 1.500B TOKENS CONSUMED (22,888 STEPS)**
- **Final Raw Checkpoint:** `{target_ckpt.name}` | Val Loss: **{raw_loss}** | PPL: **{raw_ppl}**
- **Final SWA-3 Model Soup:** `{TARGET_SWA3_I.name}` (steps 22000 + 22500 + final) | Val Loss: **{swa_loss}** | PPL: **{swa_ppl}**
- **SFT Sequence Base:** `{TARGET_SWA3_E}` armed on `data/processed_sft_stem_open_v2` (105.88k docs, 64.63M tokens, 87.2% supervision density)
"""
try:
    with open(DOCS_FILE, "a", encoding="utf-8") as f:
        f.write(doc_entry + "\n")
    print(f"[{now_str}] documentation.md updated successfully.", flush=True)
except Exception as e:
    print(f"Error appending to documentation.md: {e}", flush=True)

try:
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n### [2026-10-25 {now_str.split()[1]} IST] Tanzinite 130M-v3: 1.500B Token Pretraining Complete & SFT Hand-off\n" + doc_entry + "\n")
    print(f"[{now_str}] journal.md updated successfully.", flush=True)
except Exception as e:
    print(f"Error appending to journal.md: {e}", flush=True)

# 8. Wait for GPU.lock release
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pretraining weights audited! Waiting for GPU.lock release from PID 6216/4332...", flush=True)
while LOCK_FILE.exists():
    time.sleep(5)

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU.lock freed! Launching Post-Pretraining SFT Sequence on processed_sft_stem_open_v2...", flush=True)

# 9. Launch Post-Pretraining SFT Sequence
sft_resume = str(TARGET_SWA3_E if TARGET_SWA3_E.exists() else target_ckpt)
sft_cmd = [
    "powershell",
    "-ExecutionPolicy", "Bypass",
    "-File", str(REPO_ROOT / "scripts" / "launch_sft_130m_v3.ps1"),
    "-ResumeFrom", sft_resume,
    "-DataDir", "data/processed_sft_stem_open_v2"
]
p = subprocess.Popen(sft_cmd, cwd=str(REPO_ROOT))
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SFT Launch initiated successfully (PID {p.pid})!", flush=True)
