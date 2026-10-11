"""
Verification script for worker PERSONA gates.
Checks directories, file counts, sizes, SHA-256 hashes, licenses, DO_NOT_TRAIN,
and attribution rollup.
"""

import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path("E:/Parentheses/parentheses-0.9")
RAW_SOURCED = REPO_ROOT / "data/raw/manual/raw-sourced"

DATASETS = {
    "oasst1": {
        "expected_files": [
            "2023-04-12_oasst_all.messages.jsonl.gz",
            "2023-04-12_oasst_all.trees.jsonl.gz",
            "2023-04-12_oasst_prompts.messages.jsonl.gz",
            "2023-04-12_oasst_ready.messages.jsonl.gz",
            "2023-04-12_oasst_ready.trees.jsonl.gz",
            "2023-04-12_oasst_spam.messages.jsonl.gz",
            "LICENSE",
            "README.md",
            "SOURCE.txt",
            "SHA256SUMS.txt",
            "data/train-00000-of-00001-b42a775f407cee45.parquet",
            "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
        ],
        "expected_license": "Apache-2.0",
    },
    "oasst2": {
        "expected_files": [
            "2023-11-05_oasst2_all.messages.jsonl.gz",
            "2023-11-05_oasst2_all.trees.jsonl.gz",
            "2023-11-05_oasst2_prompts.messages.jsonl.gz",
            "2023-11-05_oasst2_ready.messages.jsonl.gz",
            "2023-11-05_oasst2_ready.trees.jsonl.gz",
            "2023-11-05_oasst2_spam.messages.jsonl.gz",
            "LICENSE",
            "README.md",
            "SOURCE.txt",
            "SHA256SUMS.txt",
            "data/train-00000-of-00001-88ba0162028a73fc.parquet",
            "data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
        ],
        "expected_license": "Apache-2.0",
    },
    "hh-rlhf": {
        "expected_files": [
            "DO_NOT_TRAIN.txt",
            "LICENSE",
            "README.md",
            "SOURCE.txt",
            "SHA256SUMS.txt",
            "harmless-base/test.jsonl.gz",
            "harmless-base/train.jsonl.gz",
            "helpful-base/test.jsonl.gz",
            "helpful-base/train.jsonl.gz",
            "helpful-online/test.jsonl.gz",
            "helpful-online/train.jsonl.gz",
            "helpful-rejection-sampled/test.jsonl.gz",
            "helpful-rejection-sampled/train.jsonl.gz",
            "red-team-attempts/red_team_attempts.jsonl.gz",
        ],
        "expected_license": "MIT",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify():
    all_ok = True
    print("=" * 70)
    print("PERSONA DATA VERIFICATION AUDIT")
    print("=" * 70)

    for ds_name, info in DATASETS.items():
        ds_dir = RAW_SOURCED / ds_name
        print(f"\n--- Checking {ds_name} at {ds_dir} ---")
        if not ds_dir.exists():
            print(f"FAIL: Directory {ds_dir} does not exist!")
            all_ok = False
            continue

        # Check expected files
        for rel_f in info["expected_files"]:
            fpath = ds_dir / rel_f
            if not fpath.exists():
                print(f"FAIL: Missing expected file {rel_f}")
                all_ok = False
            else:
                sz = fpath.stat().st_size
                print(f"  OK: {rel_f:<50} ({sz:,} bytes)")

        # Re-hash against SHA256SUMS.txt
        sums_file = ds_dir / "SHA256SUMS.txt"
        if not sums_file.exists():
            print(f"FAIL: SHA256SUMS.txt missing in {ds_name}")
            all_ok = False
        else:
            print(f"  Re-hashing against SHA256SUMS.txt ...")
            sums_lines = sums_file.read_text(encoding="utf-8").strip().splitlines()
            mismatches = 0
            for line in sums_lines:
                if not line.strip():
                    continue
                expected_hash, rel_path = line.split(maxsplit=1)
                file_to_check = ds_dir / rel_path
                if not file_to_check.exists():
                    print(f"    FAIL: Hash target {rel_path} does not exist!")
                    mismatches += 1
                    all_ok = False
                    continue
                actual_hash = sha256_file(file_to_check)
                if actual_hash != expected_hash:
                    print(f"    FAIL: Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")
                    mismatches += 1
                    all_ok = False
                else:
                    print(f"    MATCH: {rel_path} ({actual_hash[:16]}...)")
            if mismatches == 0:
                print(f"  -> All {len(sums_lines)} hashes in {ds_name} verified perfectly.")

        # Check DO_NOT_TRAIN for hh-rlhf
        if ds_name == "hh-rlhf":
            dnt = ds_dir / "DO_NOT_TRAIN.txt"
            if not dnt.exists():
                print("FAIL: DO_NOT_TRAIN.txt missing in hh-rlhf!")
                all_ok = False
            else:
                txt = dnt.read_text(encoding="utf-8")
                if "EXCLUDED FROM ALL TRAINING" not in txt or "refusal" not in txt.lower():
                    print("FAIL: DO_NOT_TRAIN.txt missing critical exclusion wording!")
                    all_ok = False
                else:
                    print("  OK: Prominent DO_NOT_TRAIN.txt verified with correct refusal/hedge rationale.")

    # Attribution check
    print("\n--- Checking Attribution Rollup ---")
    attrib_file = RAW_SOURCED / "ATTRIBUTION.md"
    if not attrib_file.exists():
        print("FAIL: ATTRIBUTION.md missing!")
        all_ok = False
    else:
        content = attrib_file.read_text(encoding="utf-8")
        for ds_name in ["oasst1", "oasst2", "hh-rlhf"]:
            if f"## {ds_name}" in content:
                print(f"  OK: {ds_name} listed in ATTRIBUTION.md")
            else:
                print(f"FAIL: {ds_name} not listed in ATTRIBUTION.md")
                all_ok = False
        if "## Problems" in content:
            print("FAIL: ATTRIBUTION.md contains Problems section!")
            all_ok = False
        else:
            print("  OK: Zero attribution problems reported.")

    # Stale line check in data/raw/manual/README.md
    print("\n--- Checking data/raw/manual/README.md ---")
    readme_txt = (REPO_ROOT / "data/raw/manual/README.md").read_text(encoding="utf-8")
    if "Does not exist yet" in readme_txt:
        print("FAIL: README.md still contains 'Does not exist yet'!")
        all_ok = False
    else:
        print("  OK: README.md stale line resolved.")

    # Status check in personality-data-brief.md
    print("\n--- Checking personality-data-brief.md ---")
    brief_txt = (REPO_ROOT / ".agents/open/personality-data-brief.md").read_text(encoding="utf-8")
    if "STATUS: COMPLETE" in brief_txt:
        print("  OK: personality-data-brief.md STATUS is COMPLETE.")
    else:
        print("FAIL: personality-data-brief.md STATUS not marked COMPLETE!")
        all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL AUDIT CHECKS PASSED PERFECTLY!")
        print("=" * 70)
        return 0
    else:
        print("AUDIT FAILED ON ONE OR MORE CHECKS!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(verify())
