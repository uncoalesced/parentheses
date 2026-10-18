#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/check_sft_anchor.py

Decode documents from a processed SFT split (.bin + boundaries) and grep for the
purged synthetic anchor family ("Examining the key principles behind this
statement, ..." and its translations). Any hit means the anchor purge has not
reached the data and the corpus must not be trained on.

Usage:
  .venv\\Scripts\\python.exe scripts\\check_sft_anchor.py \\
      --bin out\\sft_v3_dryrun\\dryrun_train.bin \\
      --boundaries out\\sft_v3_dryrun\\dryrun_train_boundaries.npy \\
      --docs 100
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from features.free_think import LANG_PRIMING, LEGACY_ARTIFACT_PHRASES

EXTRA_PATTERNS = (
    "Examining the key principles",
    "key principles behind this statement",
)

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _priming_cues() -> set[str]:
    cues = set()
    for template in LANG_PRIMING.values():
        body = template.replace("{prompt}", "").strip(" ,.;:\n\t")
        if body:
            cues.add(_norm(body))
    return cues


def main() -> int:
    p = argparse.ArgumentParser(description="Grep an SFT split for purged anchor phrases")
    p.add_argument("--bin", required=True, help="uint16 token .bin to decode")
    p.add_argument("--boundaries", required=True, help="int64 doc-start offsets .npy")
    p.add_argument("--tokenizer", default=str(REPO / "data" / "tokenizer" / "v3_32k"))
    p.add_argument("--docs", type=int, default=100, help="how many documents to decode (0 = all)")
    args = p.parse_args()

    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    data = np.fromfile(args.bin, dtype=np.uint16)
    bnd = np.load(args.boundaries)
    n = len(bnd) if args.docs <= 0 else min(args.docs, len(bnd))

    cues = _priming_cues()
    legacy = [p for p in (_norm(p) for p in LEGACY_ARTIFACT_PHRASES) if p.strip(",.") not in cues]
    cue_patterns = [p for p in (_norm(p) for p in LEGACY_ARTIFACT_PHRASES) if p.strip(",.") in cues]

    hits = 0
    cue_docs = 0
    for i in range(n):
        s = int(bnd[i])
        e = int(bnd[i + 1]) if i + 1 < len(bnd) else len(data)
        text = _norm(tok.decode(data[s:e].tolist(), skip_special_tokens=False))
        low = text.lower()
        for phrase in legacy + list(EXTRA_PATTERNS):
            if phrase.lower() in low:
                hits += 1
                print(f"[hit] doc {i}: {phrase[:60]!r}")
        if any(cue in low for cue in cue_patterns):
            cue_docs += 1

    print(f"[check_sft_anchor] decoded {n:,} of {len(bnd):,} docs from {args.bin}: {hits} anchor hits")
    print(f"[check_sft_anchor] {cue_docs:,} docs contain a designed LANG_PRIMING continuation cue (expected in low-resource freethink docs)")
    if hits:
        print("[check_sft_anchor] FAIL: anchor phrases present in SFT data.")
        return 1
    print("[check_sft_anchor] PASS: 0 anchor hits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
