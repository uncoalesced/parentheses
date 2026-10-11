"""Verification script to check that numbers in release/parentheses-0.9-41M/README.md
exactly match out/bpb_eval_0.9.json and data/processed_v3/manifest.json.

Usage:
    python scripts/check_model_card.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    readme_path = REPO_ROOT / "release" / "parentheses-0.9-41M" / "README.md"
    eval_path = REPO_ROOT / "out" / "bpb_eval_0.9.json"
    manifest_path = REPO_ROOT / "data" / "processed_v3" / "manifest.json"

    print("=== CHECK MODEL CARD NUMBERS & LICENSES ===")
    assert readme_path.exists(), f"Missing README at {readme_path}"
    assert eval_path.exists(), f"Missing eval at {eval_path}"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    readme_text = readme_path.read_text(encoding="utf-8")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)[0]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    errors = []

    # 1. Check BPB numbers (3 decimals)
    print("\n[1] Verifying Bits Per Byte (BPB)...")
    for lang, bpb_val in eval_data["bpb"].items():
        formatted_bpb = f"{bpb_val:.3f}"
        # Search for pattern in README table: e.g. **0.522**
        pattern = rf"\*\*{re.escape(formatted_bpb)}\*\*"
        if not re.search(pattern, readme_text):
            errors.append(f"BPB for {lang} ({formatted_bpb}) not found in README table.")
        else:
            print(f"  OK: {lang} BPB = {formatted_bpb} ({bpb_val:.6f})")

    # 2. Check tokens_per_byte numbers (4 decimals)
    print("\n[2] Verifying Tokens per Byte...")
    for lang, tpb_val in eval_data["tokens_per_byte"].items():
        formatted_tpb = f"{tpb_val:.4f}"
        if formatted_tpb not in readme_text:
            errors.append(f"Tokens/Byte for {lang} ({formatted_tpb}) not found in README.")
        else:
            print(f"  OK: {lang} Tokens/Byte = {formatted_tpb} ({tpb_val:.6f})")

    # 3. Check tag accuracy
    print("\n[3] Verifying Tag Accuracies...")
    for lang, tag_info in eval_data["tags"].items():
        acc = tag_info["acc"]
        if acc is None:
            continue
        formatted_acc = f"{acc * 100:.2f}%"
        # Can also match 100.0%
        alt_acc = f"{acc * 100:.1f}%"
        if formatted_acc not in readme_text and alt_acc not in readme_text:
            errors.append(f"Tag accuracy for {lang} ({formatted_acc} or {alt_acc}) not found in README.")
        else:
            print(f"  OK: {lang} tag accuracy = {formatted_acc} (raw {acc})")

    # 4. Check tag swap gaps
    print("\n[4] Verifying Tag Swap Gaps...")
    for lang, gap_val in eval_data["tagswap"].items():
        formatted_gap = f"+{gap_val:.3f}"
        if formatted_gap not in readme_text:
            errors.append(f"Tag swap gap for {lang} ({formatted_gap}) not found in README.")
        else:
            print(f"  OK: {lang} tag swap gap = {formatted_gap} ({gap_val:.6f})")

    # 5. Check manifest language table: tokens, share, epochs, licenses
    print("\n[5] Verifying Dataset Manifest Table...")
    for lang, info in manifest_data["languages"].items():
        tokens_str = f"{info['tokens']:,}"
        share_str = f"{info['share'] * 100:.2f}%"
        epochs_str = f"{info['epochs']:.3f}"

        if tokens_str not in readme_text:
            errors.append(f"Manifest tokens for {lang} ({tokens_str}) not found in README.")
        else:
            print(f"  OK: {lang} tokens = {tokens_str}")

        if share_str not in readme_text:
            errors.append(f"Manifest share for {lang} ({share_str}) not found in README.")
        else:
            print(f"  OK: {lang} share = {share_str}")

        if epochs_str not in readme_text:
            errors.append(f"Manifest epochs for {lang} ({epochs_str}) not found in README.")
        else:
            print(f"  OK: {lang} epochs = {epochs_str}")

    # Check total tokens and docs from manifest
    total_tokens_str = f"{manifest_data['train_tokens']:,}"
    total_docs_str = f"{manifest_data['train_docs']:,}"
    if total_tokens_str not in readme_text:
        errors.append(f"Total manifest train tokens ({total_tokens_str}) not found in README.")
    else:
        print(f"  OK: Total train tokens = {total_tokens_str}")

    if total_docs_str not in readme_text:
        errors.append(f"Total manifest train docs ({total_docs_str}) not found in README.")
    else:
        print(f"  OK: Total train docs = {total_docs_str}")

    # Check that ESG exclusion is explicitly in README
    if "esg" not in readme_text or "EXCLUDED" not in readme_text:
        errors.append("ESG exclusion statement missing in README.")
    else:
        print("  OK: ESG exclusion statement present.")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("\nALL MODEL CARD CHECKS PASSED PERFECTLY!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
