"""Verification script to prove:
1. Token id 11 (<esg>) has exactly 0 occurrences in data/processed_v3/train.bin (read sequentially in chunks).
2. Low-resource Dravidian training sources are strictly tcy, kff, kxv, nit (and NO esg).

Usage:
    python scripts/verify_corpus_esg_and_sources.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    train_bin = REPO_ROOT / "data" / "processed_v3" / "train.bin"
    manifest_path = REPO_ROOT / "data" / "processed_v3" / "manifest.json"

    print("=== VERIFY ESG EXCLUSION & LOW-RES SOURCES ===")
    assert train_bin.exists(), f"Missing train.bin at {train_bin}"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    # 1. Sequential chunked scan of train.bin
    print(f"[1] Scanning {train_bin} sequentially for token id 11 (<esg>)...")
    t0 = time.time()
    chunk_tokens = 32 * 1024 * 1024  # 32M uint16 tokens = 64 MB per chunk
    total_tokens = 0
    count_11 = 0

    with open(train_bin, "rb") as f:
        while True:
            chunk = f.read(chunk_tokens * 2)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=np.uint16)
            total_tokens += len(arr)
            count_11 += int((arr == 11).sum())

    scan_time = time.time() - t0
    print(f"  Scanned: {total_tokens:,} tokens in {scan_time:.2f}s ({total_tokens / scan_time / 1e6:.1f} Mtok/s)")
    print(f"  Count of token ID 11 (<esg>): {count_11}")
    assert count_11 == 0, f"FAILED: Found {count_11} occurrences of <esg> in train.bin!"
    print("  RESULT: Zero <esg> tokens confirmed in training corpus.")

    # 2. Check low-resource sources in manifest
    print("\n[2] Checking manifest sources and language splits...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    expected_lowres = {"tcy", "kff", "kxv", "nit"}
    langs = set(m["languages"].keys())
    major_and_eng = {"kan", "tam", "tel", "mal", "eng"}
    actual_lowres = langs - major_and_eng

    print(f"  Languages in manifest: {sorted(langs)}")
    print(f"  Low-resource languages: {sorted(actual_lowres)}")
    assert actual_lowres == expected_lowres, f"Unexpected low-res languages: {actual_lowres} != {expected_lowres}"
    assert "esg" not in langs, "CRITICAL: 'esg' found in manifest languages!"

    # Check ESG exclusion notice in manifest
    esg_note = m.get("esg", "")
    print(f"  Manifest ESG field: {esg_note!r}")
    assert "EXCLUDED" in esg_note, f"Manifest missing explicit EXCLUDED note for esg: {esg_note}"

    # Check sources string
    lowres_sources = m.get("sources", {}).get("tcy/kff/kxv/nit", "")
    print(f"  Low-res sources field: {lowres_sources!r}")
    assert "lowres" in lowres_sources.lower(), f"Unexpected low-res source path: {lowres_sources}"

    print("\nALL CORPUS AND SOURCE CHECKS PASSED PERFECTLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
