#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/build_s7_sft_240m.py
Aggregates Hermes curated Tier 1 JSONL shards and the canonical self-identity dataset,
then invokes the 4,096-token First-Fit-Decreasing (FFD) packer (pack_sft_4096_v4.py)
to produce:
  - data/processed_sft_240m_v4_apache/train.bin
  - data/processed_sft_240m_v4_apache/train.mask.bin
  - data/processed_sft_240m_v4_apache/train_boundaries.npy
  - data/processed_sft_240m_v4_apache/train_boundaries_flat.npy
  - data/processed_sft_240m_v4_apache/manifest.json
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = Path(r"E:\Parentheses\data\curated_s7_packin")
IDENTITY_FILE = REPO_ROOT / "data" / "curated" / "parentheses_identity_sft_v1.jsonl"
OUT_DIR = REPO_ROOT / "data" / "processed_sft_240m_v4_apache"
TOKENIZER_DIR = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
PACKER_SCRIPT = REPO_ROOT / "scripts" / "pack_sft_4096_v4.py"

def main():
    print("=== PARENTHESES 1.0-240M: SESSION 7 SFT CORPUS BUILDER ===")
    
    if not CURATED_DIR.exists():
        print(f"[FAIL] Curated directory does not exist: {CURATED_DIR}")
        sys.exit(1)
        
    jsonl_files = list(CURATED_DIR.glob("*.jsonl"))
    # Filter out empty or quarantine files
    clean_files = []
    for f in jsonl_files:
        if f.name.startswith("rejected_") or f.name.startswith("quarantine") or f.name.startswith("refusal_quarantine") or f.stat().st_size == 0:
            continue
        clean_files.append(f)
        
    print(f"[*] Found {len(clean_files)} curated JSONL datasets in {CURATED_DIR}")
    
    if IDENTITY_FILE.exists():
        print(f"[*] Adding canonical self-identity dataset: {IDENTITY_FILE}")
        clean_files.append(IDENTITY_FILE)
    else:
        print(f"[WARN] Canonical identity dataset not found at {IDENTITY_FILE}")

    for f in clean_files:
        print(f"  + {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        
    cmd = [
        sys.executable,
        str(PACKER_SCRIPT),
        "--tokenizer", str(TOKENIZER_DIR / "tokenizer.json"),
        "--inputs"
    ] + [str(p) for p in clean_files] + [
        "--out-dir", str(OUT_DIR),
        "--seq-len", "4096",
        "--seed", "42"
    ]
    
    print("\n[*] Invoking FFD 4,096-token Document-Aware Packer...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd)
    
    if res.returncode == 0:
        print(f"\n[SUCCESS] SFT Corpus successfully packed to {OUT_DIR}")
    else:
        print(f"\n[FAIL] Packer failed with exit code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
