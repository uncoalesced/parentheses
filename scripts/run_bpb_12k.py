# Engineered by uncoalesced
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(r"E:\Parentheses\parentheses-0.9")
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
S5 = str(REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s5" / "step_10000_final.pt")
S6_12K = str(REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s6" / "step_12000.pt")
OUT_JSON = str(REPO_ROOT / "out" / "eval_s6_12k" / "bpb_12k.json")
OUT_LOG = str(REPO_ROOT / "out" / "eval_s6_12k" / "bpb_12k.log")
OUT_GATE = str(REPO_ROOT / "out" / "eval_s6_12k" / "bpb_gate_12k.txt")

print("[*] Running bpb_slab_v4 on CUDA...")
with open(OUT_LOG, "w", encoding="utf-8") as f:
    subprocess.run([
        PYTHON, "scripts/bpb_slab_v4.py",
        "--checkpoints", S5, S6_12K,
        "--device", "cuda", "--batch", "2",
        "--out", OUT_JSON
    ], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

print("[*] Running bpb_gate...")
with open(OUT_GATE, "w", encoding="utf-8") as f:
    subprocess.run([
        PYTHON, "scripts/bpb_gate.py",
        OUT_JSON, "--max-rise", "0.06"
    ], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

print("[+] Completed bpb run and gate check successfully.")
