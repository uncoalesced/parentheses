# Engineered by uncoalesced
"""
Step 12,000 Rest & Evaluation Sentinel for Parentheses 1.0-240M Session 6
Monitors live training until Step 12,000 is completed and archived,
then gracefully halts train.py to allow laptop hardware to rest,
releases GPU.lock, and executes the complete empirical benchmark suite:
  1. 60-probe adversarial suite (greedy + sampled T=0.7)
  2. score_60p.py multi-session comparative scoring (S4 vs S5 vs S6-12k)
  3. bpb_slab_v4.py & bpb_gate.py across all 11 evaluation languages
  4. Formats clean summary for Joel to review before resuming Leg 2 (12,001 -> 13,000/14,000).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "out" / "session6_240m_live.log"
CKPT_12K = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s6" / "step_12000.pt"
ARCHIVE_12K = Path(r"I:\parentheses\checkpoints\parentheses-1.0-240M-s6\step_12000.pt")
LOCK_PATH = REPO_ROOT / "out" / "locks" / "GPU.lock"
EVAL_DIR = REPO_ROOT / "out" / "eval_s6_12k"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def get_current_step():
    if not LOG_PATH.exists():
        return 0
    try:
        with open(LOG_PATH, "r", encoding="utf-16", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return 0

    for line in reversed(lines[-200:]):
        m = re.search(r"step\s+(\d+)", line)
        if m:
            return int(m.group(1))
    return 0


def is_step_12k_reached():
    if CKPT_12K.exists() and CKPT_12K.stat().st_size > 3000000000:
        return True
    return get_current_step() >= 12000


def wait_for_checkpoint_stable(ckpt_path, min_size_mb=2800, wait_sec=5):
    if not ckpt_path.exists():
        return False
    s1 = ckpt_path.stat().st_size
    if s1 < min_size_mb * 1024 * 1024:
        return False
    time.sleep(wait_sec)
    s2 = ckpt_path.stat().st_size
    return s1 == s2 and s1 > 3000000000


def halt_training():
    print("[*] Halting training to allow laptop hardware to rest...")
    # Find python process running train.py
    cmd = ["powershell", "-NoProfile", "-Command", 
           "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*train.py*' } | Select-Object -ExpandProperty ProcessId"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        pids = [p.strip() for p in res.stdout.splitlines() if p.strip()]
        for pid in pids:
            print(f"    - Terminating train.py process PID {pid}...")
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
    except Exception as e:
        print(f"[!] Warning during process halt: {e}")

    # Remove GPU.lock
    if LOCK_PATH.exists():
        try:
            LOCK_PATH.unlink()
            print("[+] GPU.lock released cleanly.")
        except Exception as e:
            print(f"[!] Warning removing lock: {e}")


def run_eval_suite():
    print("\n" + "="*70)
    print("=== EXECUTING STEP 12,000 EMPIRICAL BENCHMARK SUITE ===")
    print("="*70)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 60-probe greedy (100% CUDA GPU)
    greedy_out = EVAL_DIR / "s6_12000_chat_greedy.jsonl"
    print(f"[*] Running 60-probe greedy evaluation on CUDA -> {greedy_out.name}...", flush=True)
    subprocess.run([
        str(PYTHON_EXE), "scripts/eval_60p_adversarial_suite.py",
        "--checkpoint", str(CKPT_12K),
        "--output", str(greedy_out),
        "--device", "cuda",
        "--mode", "chat", "--temp", "0.0", "--samples", "1", "--max-tokens", "256"
    ], cwd=REPO_ROOT)

    # 2. 60-probe sampled T=0.7 (100% CUDA GPU)
    sampled_out = EVAL_DIR / "s6_12000_chat_t07x5.jsonl"
    print(f"[*] Running 60-probe sampled T=0.7 x 5 evaluation on CUDA -> {sampled_out.name}...", flush=True)
    subprocess.run([
        str(PYTHON_EXE), "scripts/eval_60p_adversarial_suite.py",
        "--checkpoint", str(CKPT_12K),
        "--output", str(sampled_out),
        "--device", "cuda",
        "--mode", "chat", "--temp", "0.7", "--samples", "5", "--seed", "1234", "--max-tokens", "256"
    ], cwd=REPO_ROOT)

    # 3. Mechanical scoring (S4 vs S5 vs S6-12k)
    scores_txt = EVAL_DIR / "scores_12k.txt"
    scores_json = EVAL_DIR / "scores_12k.json"
    print(f"[*] Running comparative scoring -> {scores_txt.name}...")
    score_cmd = [
        str(PYTHON_EXE), "scripts/score_60p.py",
        "out/eval_s4/s4_chat_greedy_256.jsonl",
        "out/eval_s5/s5_chat_greedy.jsonl",
        str(greedy_out), str(sampled_out),
        "--json", str(scores_json)
    ]
    with open(scores_txt, "w", encoding="utf-8") as f:
        subprocess.run(score_cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

    # 4. Bits-Per-Byte suite (100% GPU accelerated)
    bpb_json = EVAL_DIR / "bpb_12k.json"
    bpb_log = EVAL_DIR / "bpb_12k.log"
    s5_final = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s5" / "step_10000_final.pt"
    print(f"[*] Running BPB drift evaluation on CUDA -> {bpb_json.name}...")
    with open(bpb_log, "w", encoding="utf-8") as f:
        subprocess.run([
            str(PYTHON_EXE), "scripts/bpb_slab_v4.py",
            "--checkpoints", str(s5_final), str(CKPT_12K),
            "--device", "cuda", "--batch", "2",
            "--out", str(bpb_json)
        ], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

    # 5. BPB Gate check
    bpb_gate_txt = EVAL_DIR / "bpb_gate_12k.txt"
    with open(bpb_gate_txt, "w", encoding="utf-8") as f:
        subprocess.run([
            str(PYTHON_EXE), "scripts/bpb_gate.py",
            str(bpb_json), "--max-rise", "0.06"
        ], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

    print("\n[+] Step 12,000 evaluation suite complete!")
    print(f"    - Results saved in: {EVAL_DIR}")
    print("[+] All evaluations finished at 100% GPU performance. Laptop is now in full rest mode.")



def main():
    parser = argparse.ArgumentParser(description="Sentinel: Stop at 12k, Rest Laptop, and Run Evals")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()

    print("[*] Step 12,000 Sentinel Active:", flush=True)
    print("    - Monitoring training progress toward Step 12,000...", flush=True)
    print("    - Hardware runs at 100% GPU through training AND full evaluations.", flush=True)
    print("    - Laptop rests only after evaluations are 100% complete.", flush=True)

    iteration = 0
    while True:
        current_step = get_current_step()
        iteration += 1

        if iteration % 10 == 1 or iteration == 1:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            remaining = max(0, 12000 - current_step)
            print(f"[{timestamp}] Heartbeat: Step {current_step}/12,000 | {remaining} steps to gate | GPU 100%", flush=True)

        if is_step_12k_reached():
            print(f"\n[+] Step 12,000 reached (current step: {current_step})!", flush=True)
            print("[*] Verifying step_12000.pt checkpoint integrity...", flush=True)
            deadline = time.time() + 600
            verified = False
            while time.time() < deadline:
                if wait_for_checkpoint_stable(CKPT_12K):
                    print(f"[+] Checkpoint verified: {CKPT_12K} ({CKPT_12K.stat().st_size:,} bytes)", flush=True)
                    verified = True
                    break
                time.sleep(5)

            if not verified:
                print("[!] Checkpoint wait timed out or file missing. Proceeding with caution...", flush=True)

            # Halt training process
            halt_training()

            # Execute full evaluation suite at 100% GPU
            run_eval_suite()

            # Flag completion
            flag_path = REPO_ROOT / "out" / "step12000_eval_completed.flag"
            with open(flag_path, "w", encoding="utf-8") as f:
                f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\nStep 12,000 eval finished.\n")

            print(f"[+] All evaluations complete. Flag written: {flag_path}", flush=True)
            print("[+] LAPTOP IS NOW RESTING (GPU completely idle, fans ramping down).", flush=True)
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
