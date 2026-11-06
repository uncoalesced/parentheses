"""
scripts/scrub_s6_devanagari.py
Parentheses 1.0 (240M) - S6 Devanagari Clean Extraction & Scrubbing Pipeline.
Extracts:
  1. 3.0M tokens of verified morphological Sanskrit from I:\parentheses\raw\sangraha_san\verified\san\data-0.parquet
  2. 4.0M tokens of non-overlapping clean Hindi prose from F:\Lingual_Data\raw\hf\ai4bharat_Sangraha_verified\verified\hin\data-0.parquet (skipping first 20,000 rows used in v5)
Outputs to:
  I:\parentheses\curated_s6_devanagari\san_clean.jsonl
  I:\parentheses\curated_s6_devanagari\hin_clean.jsonl
"""

import os
import re
import sys
import json
import time
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(r"I:\parentheses\curated_s6_devanagari")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOK_PATH = ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
tok = Tokenizer.from_file(str(TOK_PATH))

MODERN_IA = set(["है", "हैं", "था", "थी", "थे", "में", "को", "ने", "से", "का", "की", "के", "और", "किया", "आहे", "आहेत", "मध्ये", "ला", "चा", "ची", "चे", "आणि"])
SANSKRIT_PARTICLES = ["इति", "च", "एव", "अपि", "भवति", "अस्ति", "तस्य"]

def scrub_sanskrit(target_tokens=3_000_000):
    san_pq = Path(r"I:\parentheses\raw\sangraha_san\verified\san\data-0.parquet")
    out_file = OUT_DIR / "san_clean.jsonl"
    print(f"\n[1/2] Scrubbing Sanskrit from {san_pq.name} -> {out_file.name} (target: {target_tokens:,} tokens)...")
    
    pf = pq.ParquetFile(str(san_pq))
    total_tok = 0
    total_docs = 0
    t0 = time.time()
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            for text in table["text"].to_pylist():
                if not text:
                    continue
                s = text.strip()
                words = s.split()
                if len(words) < 10:
                    continue
                
                # Filter 1: Reject modern IA words > 0.3%
                ia_count = sum(1 for w in words if w in MODERN_IA)
                if (ia_count / len(words)) > 0.003:
                    continue
                
                # Filter 2: Require Sanskrit morphological markers >= 2
                markers = s.count("ः") + s.count("ऽ") + len(re.findall(r"[मत्]्(?:\s|$|[.,।!?])", s))
                for p in SANSKRIT_PARTICLES:
                    markers += s.count(p)
                if markers < 2:
                    continue
                
                # Filter 3: Token length bounds 100 <= len <= 1024
                enc = tok.encode(s, add_special_tokens=False).ids
                if not (100 <= len(enc) <= 1024):
                    continue
                
                record = {
                    "lang": "san",
                    "tag_id": 20,
                    "tokens": len(enc),
                    "text": s
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_tok += len(enc)
                total_docs += 1
                
                if total_tok >= target_tokens:
                    break
            if total_tok >= target_tokens:
                break
                
    elapsed = time.time() - t0
    print(f"[+] Sanskrit complete: {total_docs:,} docs, {total_tok:,} tokens in {elapsed:.1f}s")
    return total_docs, total_tok

def scrub_hindi(target_tokens=4_000_000, skip_rows=20_000):
    hin_pq = Path(r"F:\Lingual_Data\raw\hf\ai4bharat_Sangraha_verified\verified\hin\data-0.parquet")
    out_file = OUT_DIR / "hin_clean.jsonl"
    print(f"\n[2/2] Scrubbing Hindi from {hin_pq.name} (skipping first {skip_rows:,} rows) -> {out_file.name} (target: {target_tokens:,} tokens)...")
    
    pf = pq.ParquetFile(str(hin_pq))
    total_tok = 0
    total_docs = 0
    current_row = 0
    t0 = time.time()
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            for text in table["text"].to_pylist():
                current_row += 1
                if current_row <= skip_rows:
                    continue
                if not text:
                    continue
                s = text.strip()
                if len(s) < 40:
                    continue
                # Terminal punctuation check
                if not any(s.endswith(p) for p in [".", "।", "!", "?"]):
                    continue
                
                # Length bounds 100 <= len <= 1024
                enc = tok.encode(s, add_special_tokens=False).ids
                if not (100 <= len(enc) <= 1024):
                    continue
                
                record = {
                    "lang": "hin",
                    "tag_id": 19,
                    "tokens": len(enc),
                    "text": s
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_tok += len(enc)
                total_docs += 1
                
                if total_tok >= target_tokens:
                    break
            if total_tok >= target_tokens:
                break
                
    elapsed = time.time() - t0
    print(f"[+] Hindi complete: {total_docs:,} docs, {total_tok:,} tokens in {elapsed:.1f}s")
    return total_docs, total_tok

if __name__ == "__main__":
    print("=== Parentheses 1.0 (240M) - S6 Devanagari Clean Scrubber ===")
    s_docs, s_toks = scrub_sanskrit(3_000_000)
    h_docs, h_toks = scrub_hindi(4_000_000, 20_000)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "san_docs": s_docs,
        "san_tokens": s_toks,
        "hin_docs": h_docs,
        "hin_tokens": h_toks,
        "san_file": str(OUT_DIR / "san_clean.jsonl"),
        "hin_file": str(OUT_DIR / "hin_clean.jsonl"),
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] All done! Manifest written to {OUT_DIR / 'manifest.json'}")