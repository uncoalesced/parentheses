"""
Offline Dataset Cleaner via NPU Byte Filter.
Cleans raw JSONL or text files from F:\Lingual_Data before tokenization,
filtering out mojibake, malformed UTF-8, boilerplate markup, and script mismatches.
Writes rejected samples with audit telemetry to out/filter_rejected.jsonl.

Usage:
    python scripts/clean_corpus_npu.py --input F:/Lingual_Data/data/processed_v3/text/kan.jsonl --output data/cleaned/kan.jsonl --target-lang kan
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.npu_byte_filter import NpuByteFilter


def clean_file(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    target_lang: str = "any",
    quality_threshold: float = 0.50,
    max_lines: int = -1,
):
    filt = NpuByteFilter(quality_threshold=quality_threshold)
    print(f"[clean_corpus] Initialized NpuByteFilter (Active Provider: {filt.active_provider})")
    print(f"[clean_corpus] Processing: {input_path} -> {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    rejected_count = 0
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(audit_path, "a", encoding="utf-8") as faudit:

        for idx, line in enumerate(fin):
            if max_lines > 0 and idx >= max_lines:
                break

            line_str = line.strip()
            if not line_str:
                continue

            # Handle JSONL format {"text": "..."} vs raw text
            if line_str.startswith("{") and line_str.endswith("}"):
                try:
                    obj = json.loads(line_str)
                    text_content = obj.get("text", "")
                except Exception:
                    text_content = line_str
            else:
                text_content = line_str

            # Run byte filter
            res = filt.filter_text(text_content)
            pred_lang = res["lang"]
            q_score = res["quality"]
            keep = res["keep"]

            # Script/Language consistency check
            if target_lang != "any" and target_lang not in ("lowres", "all"):
                # For specific target language, ensure it matches target or lowres dialect
                if pred_lang not in (target_lang, "lowres") and pred_lang != "eng":
                    keep = False
                    res["reject_reason"] = f"target_mismatch(expected={target_lang}, got={pred_lang})"

            if keep:
                fout.write(line + "\n")
                kept_count += 1
            else:
                rejected_count += 1
                rejection_record = {
                    "file": str(input_path.name),
                    "line": idx + 1,
                    "pred_lang": pred_lang,
                    "quality": round(q_score, 4),
                    "confidence": round(res["confidence"], 4),
                    "reason": res.get("reject_reason", "low_quality_or_noise"),
                    "snippet": text_content[:120],
                }
                faudit.write(json.dumps(rejection_record, ensure_ascii=False) + "\n")

            if (idx + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 0.001)
                print(f"[clean_corpus] {idx + 1:,} lines | Kept: {kept_count:,} | Rejected: {rejected_count:,} ({rate:.1f} lines/s)")

    elapsed = time.time() - t0
    total = kept_count + rejected_count
    print(f"\n[clean_corpus] Completed in {elapsed:.1f}s ({total / max(elapsed, 0.001):.1f} lines/s)")
    print(f"[clean_corpus] Total: {total:,} | Kept: {kept_count:,} ({100.0 * kept_count / max(total, 1):.1f}%) | Rejected: {rejected_count:,} ({100.0 * rejected_count / max(total, 1):.1f}%)")
    print(f"[clean_corpus] Rejection audit saved to: {audit_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input raw text or JSONL file")
    parser.add_argument("--output", required=True, help="Output cleaned file")
    parser.add_argument("--audit", default="out/filter_rejected.jsonl", help="Path to rejection audit log")
    parser.add_argument("--target-lang", default="any", help="Target language code (kan, tam, tel, mal, eng, etc.)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Minimum quality score to keep")
    parser.add_argument("--max-lines", type=int, default=-1, help="Max lines to process (for quick smoke tests)")
    args = parser.parse_args()

    clean_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        audit_path=ROOT / args.audit,
        target_lang=args.target_lang,
        quality_threshold=args.threshold,
        max_lines=args.max_lines,
    )


if __name__ == "__main__":
    main()
