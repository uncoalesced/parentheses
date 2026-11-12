# Engineered by uncoalesced
# Session 10 Held-Out Gate Hand-Read Inspection Utility
# Extracts 20 Epistemic Abstentions and 20 Dravidian Generations from out/s10/master_gate_heldout_details.jsonl

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    details_path = REPO_ROOT / "out" / "s10" / "master_gate_heldout_details.jsonl"
    if not details_path.exists():
        print(f"[!] Details file not found: {details_path}")
        return

    with open(details_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Loaded {len(records)} evaluation records from {details_path}")

    # 1. Extract 20 Epistemic Abstentions
    abstentions = [
        r for r in records
        if r.get("category") == "epistemic_abstention" or r.get("domain_id") == "D05"
    ]

    print("\n" + "=" * 80)
    print("[*] 20 SAMPLED EPISTEMIC ABSTENTION GENERATIONS (Cat B & Cat C held-out traps)")
    print("=" * 80)
    step_stride = max(1, len(abstentions) // 20)
    sampled_abst = abstentions[::step_stride][:20]

    for idx, r in enumerate(sampled_abst, 1):
        status = "[PASS] (IDK)" if r.get("has_idk") else "[FAIL] (CONFABULATION)"
        print(f"\n[{idx}/20] Prompt ({r.get('id')}, {r.get('language')}): {r.get('prompt')}")
        print(f"       Status: {status} | Stopped Cleanly: {r.get('stopped_cleanly')}")
        print(f"       Model Output:\n\"\"\"{r.get('generation')}\"\"\"")

    # 2. Extract 20 Dravidian Format / Content Generations
    dravidian = [
        r for r in records
        if r.get("language") in ("kn", "ta", "te", "ml") and r.get("category") == "dravidian_format"
    ]

    print("\n" + "=" * 80)
    print("[*] 20 SAMPLED DRAVIDIAN STRICT FORMAT GENERATIONS (Cat D held-out items)")
    print("=" * 80)
    step_stride_drav = max(1, len(dravidian) // 20)
    sampled_drav = dravidian[::step_stride_drav][:20]

    for idx, r in enumerate(sampled_drav, 1):
        status = "[PASS] (Final:)" if r.get("has_correct_format") else "[FAIL] (Missing Final:)"
        print(f"\n[{idx}/20] Prompt ({r.get('id')}, {r.get('language')}): {r.get('prompt')}")
        print(f"       Status: {status} | Stopped Cleanly: {r.get('stopped_cleanly')}")
        print(f"       Model Output:\n\"\"\"{r.get('generation')}\"\"\"")
    print("=" * 80)

if __name__ == "__main__":
    main()
