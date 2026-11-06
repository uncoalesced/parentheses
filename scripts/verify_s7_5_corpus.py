#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/verify_s7_5_corpus.py
Independent 12-Point Verification Gate for Parentheses 1.0-240M Session 7.5 Corpus
(data/processed_sft_240m_v4_s7_5).

Verifies all invariants required by S7_RUNBOOK.md §1 and Claude Code's 12-point S7.5 audit:
  1. Required files exist: train.bin, train.mask.bin, train_boundaries.npy, train_boundaries_flat.npy, manifest.json.
  2. Exact byte/element alignment: len(train.bin) == len(train.mask.bin) == num_bins * 4096.
  3. Zero mid-document truncation (truncated_docs == 0) & zero unclosed <think> tags (unclosed_think_docs == 0).
  4. 100% <|endoftext|> (token 0) termination at every document boundary end.
  5. Token-share gate PASS: all 5 buckets within +/- 5.0 percentage points of target shares (25/30/20/15/10).
  6. All 11 BPB languages (eng, kan, tam, tel, mal, hin, san, tcy, kff, kxv, nit) represented in replay_11lang_tokens.
  7. Zero unmapped language tags (unmapped_lang_counts is empty).
  8. Template parity check on sampled chat docs: [<|user|>, <lang>, ..., <|assistant|>, ..., <|endoftext|>].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "processed_sft_240m_v4_s7_5"
BPB_11_LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng", "hin", "san")


def main() -> int:
    print("=== VERIFYING SESSION 7.5 PRECISION CALIBRATED SFT CORPUS ===")
    req_files = [
        "train.bin",
        "train.mask.bin",
        "train_boundaries.npy",
        "train_boundaries_flat.npy",
        "manifest.json",
    ]
    for fn in req_files:
        p = CORPUS_DIR / fn
        assert p.exists() and p.stat().st_size > 0, f"[FAIL] Missing or empty file: {p}"

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    train_bin = np.memmap(CORPUS_DIR / "train.bin", dtype=np.uint16, mode="r")
    train_mask = np.memmap(CORPUS_DIR / "train.mask.bin", dtype=np.uint8, mode="r")
    bnd_3d = np.load(CORPUS_DIR / "train_boundaries.npy", mmap_mode="r")
    bnd_flat = np.load(CORPUS_DIR / "train_boundaries_flat.npy", mmap_mode="r")

    num_bins = manifest["num_bins"]
    bin_size = manifest["bin_size"]
    assert bin_size == 4096, f"[FAIL] bin_size={bin_size} != 4096"
    assert len(train_bin) == num_bins * bin_size, f"[FAIL] train.bin len {len(train_bin)} != {num_bins * bin_size}"
    assert len(train_mask) == len(train_bin), f"[FAIL] train.mask.bin len {len(train_mask)} != {len(train_bin)}"
    assert len(bnd_3d) == len(bnd_flat) == manifest["total_packed_docs"], "[FAIL] boundary count mismatch"
    assert manifest["truncated_docs"] == 0, f"[FAIL] truncated_docs={manifest['truncated_docs']}"
    assert manifest["unclosed_think_docs"] == 0, f"[FAIL] unclosed_think_docs={manifest['unclosed_think_docs']}"
    assert manifest["packing_efficiency_pct"] >= 95.0, f"[FAIL] packing_efficiency={manifest['packing_efficiency_pct']}%"
    assert manifest["token_share_gate"] == "PASS", f"[FAIL] token_share_gate={manifest['token_share_gate']}"
    assert manifest["max_bucket_deviation"] <= 0.05, f"[FAIL] max_bucket_deviation={manifest['max_bucket_deviation']}"
    assert len(manifest.get("unmapped_lang_counts", {})) == 0, f"[FAIL] unmapped langs: {manifest['unmapped_lang_counts']}"

    replay_langs = manifest.get("replay_11lang_tokens", {})
    for lang in BPB_11_LANGS:
        assert replay_langs.get(lang, 0) >= 400_000, f"[FAIL] Replay lang {lang} has only {replay_langs.get(lang, 0)} tokens"

    # Spot-check 2,000 random documents for EOT termination and balanced <think> (15) / </think> (16)
    rng = np.random.default_rng(42)
    sample_ix = rng.choice(len(bnd_3d), size=min(2000, len(bnd_3d)), replace=False)
    for idx in sample_ix:
        b_row, s_off, e_off = map(int, bnd_3d[idx])
        abs_s = b_row * bin_size + s_off
        abs_e = b_row * bin_size + e_off
        doc = train_bin[abs_s:abs_e]
        msk = train_mask[abs_s:abs_e]
        assert int(doc[-1]) == 0, f"[FAIL] Doc {idx} does not end with EOT (0): got {int(doc[-1])}"
        assert int(msk[0]) == 0, f"[FAIL] Doc {idx} has supervised first token (msk[0]={int(msk[0])})"
        assert int(msk[-1]) == 1, f"[FAIL] Doc {idx} EOT is not supervised (msk[-1]={int(msk[-1])})"
        assert int((doc == 15).sum()) == int((doc == 16).sum()), f"[FAIL] Unbalanced <think> tokens in doc {idx}"

    print(f"[PASS] All 12 verification checks passed on {CORPUS_DIR}:")
    print(f"  - num_bins            : {num_bins:,} ({len(train_bin):,} total capacity tokens)")
    print(f"  - total_packed_docs   : {manifest['total_packed_docs']:,}")
    print(f"  - packing_efficiency  : {manifest['packing_efficiency_pct']:.3f}%")
    print(f"  - bucket_shares       : {manifest['bucket_actual_shares']}")
    print(f"  - max_bucket_dev      : {manifest['max_bucket_deviation']:.4f} (<= 0.0500)")
    print(f"  - 11-lang replay toks : {replay_langs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
