#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Download 7 Curated Permissive STEM, Code & Reasoning Datasets to 'Joel's Passport' (F:).
100% Apache-2.0 / MIT / ODC-By Commercial-Permissive Compliant.

Target Location: F:\Lingual_Data\raw\hf_permissive_stem_code\
Available Space: ~148 GB | Planned Download: ~32 GB | Headroom: > 115 GB

Datasets:
  1. bespokelabs/Bespoke-Stratos-17k           (120 MB, Apache-2.0, explicit <think> traces)
  2. microsoft/orca-math-word-problems-200k   (80 MB, MIT, step-by-step math reasoning)
  3. meta-math/MetaMathQA                     (377 MB, MIT, 395k math derivations)
  4. AI-MO/NuminaMath-CoT                     (1.18 GB, Apache-2.0, Olympiad & competition math)
  5. HuggingFaceTB/smoltalk                   (3.96 GB, Apache-2.0, instruction & chat)
  6. HuggingFaceTB/smollm-corpus (python-edu) (600 MB, Apache-2.0, educational Python code)
  7. HuggingFaceTB/smollm-corpus (textbooks)  (25 GB, Apache-2.0/ODC-By, Cosmopedia-v2 + FineWeb-Edu top tier)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Enable high-speed Rust-based hf_transfer for maximum download speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

DEST_BASE = Path(r"F:\Lingual_Data\raw\hf_permissive_stem_code")

DATASETS = [
    {
        "id": "bespokelabs/Bespoke-Stratos-17k",
        "name": "Bespoke-Stratos-17k",
        "license": "Apache-2.0",
        "role": "Explicit <think> ... </think> reasoning traces (STEM, logic, code)",
        "patterns": None,
    },
    {
        "id": "microsoft/orca-math-word-problems-200k",
        "name": "orca-math-word-problems-200k",
        "license": "MIT",
        "role": "200k step-by-step math word problems",
        "patterns": None,
    },
    {
        "id": "meta-math/MetaMathQA",
        "name": "MetaMathQA",
        "license": "MIT",
        "role": "395k math reasoning questions with forward/backward derivations",
        "patterns": None,
    },
    {
        "id": "AI-MO/NuminaMath-CoT",
        "name": "NuminaMath-CoT",
        "license": "Apache-2.0",
        "role": "Olympiad and competition math with full chain-of-thought derivations",
        "patterns": None,
    },
    {
        "id": "HuggingFaceTB/smoltalk",
        "name": "smoltalk",
        "license": "Apache-2.0",
        "role": "Curated multi-turn instruction, everyday reasoning, and factuality",
        "patterns": ["data/*", "*.json", "README.md"],
    },
    {
        "id": "HuggingFaceTB/smollm-corpus",
        "name": "smollm-corpus-python-edu",
        "license": "Apache-2.0",
        "role": "Curated educational Python code, algorithms, and unit tests",
        "patterns": ["python-edu/*", "README.md"],
    },
    {
        "id": "HuggingFaceTB/smollm-corpus",
        "name": "smollm-corpus-textbooks-edu",
        "license": "Apache-2.0 / ODC-By",
        "role": "Cosmopedia-v2 synthetic textbooks & FineWeb-Edu deduplicated top-tier sample",
        "patterns": [
            "cosmopedia-v2/train-0000[0-9]-*.parquet",
            "cosmopedia-v2/train-0001[0-4]-*.parquet",
            "fineweb-edu-dedup/train-0000[0-9]-*.parquet",
            "README.md",
        ],
    },
]


def download_all(dry_run: bool = False):
    print("=" * 70)
    print("Parentheses 1.0-240M: Hugging Face Permissive Sourcing to Joel's Passport (F:)")
    print(f"Target Directory: {DEST_BASE}")
    print(f"Time (UTC): {datetime.utcnow().isoformat()}")
    print("=" * 70)

    DEST_BASE.mkdir(parents=True, exist_ok=True)

    manifest_records = []

    for idx, ds in enumerate(DATASETS, start=1):
        target_dir = DEST_BASE / ds["name"]
        print(f"\n[{idx}/7] Preparing: {ds['id']} -> {ds['name']}")
        print(f"      License: {ds['license']} | Role: {ds['role']}")

        if dry_run:
            print(f"      [Dry-Run] Target: {target_dir}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        success = False
        for attempt in range(1, 6):
            try:
                snapshot_download(
                    repo_id=ds["id"],
                    repo_type="dataset",
                    local_dir=str(target_dir),
                    allow_patterns=ds["patterns"],
                    max_workers=2,
                    resume_download=True,
                    token=True,
                )
                success = True
                break
            except Exception as e:
                print(f"      [RETRY {attempt}/5] Encountered error on {ds['id']}: {e}. Retrying in 10s...")
                time.sleep(10)

        if not success:
            print(f"      [ERROR] Failed to download {ds['id']} after 5 attempts.")
            continue

        elapsed = time.time() - t0
        print(f"      [DONE] Downloaded in {elapsed:.1f}s -> {target_dir}")
        manifest_records.append({
            "repo_id": ds["id"],
            "local_dir": str(target_dir),
            "license": ds["license"],
            "role": ds["role"],
            "downloaded_at": datetime.utcnow().isoformat(),
        })

    # Write overall manifest.json
    manifest = {
        "version": "hf_permissive_stem_code_v1",
        "license_compliance": "Apache-2.0-Permissive-Only",
        "created_at": datetime.utcnow().isoformat(),
        "storage_target": str(DEST_BASE),
        "drive_label": "Joel's Passport (F:)",
        "datasets": manifest_records,
        "strict_exclusions": ["Camel-AI", "SciQ", "esg", "CC-BY-NC", "Llama-Community"],
    }
    manifest_path = DEST_BASE / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[+] All downloads finished! Manifest written to: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned datasets without downloading")
    args = parser.parse_args()
    download_all(dry_run=args.dry_run)
