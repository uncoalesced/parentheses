#!/usr/bin/env python3
# Engineered by uncoalesced
"""
tests/test_pack_sft_4096_v4.py
Unit and property tests for 4096-token FFD document-aware binary packer (scripts/pack_sft_4096_v4.py).
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pack_sft_4096_v4 import (
    MaxSegmentTree,
    resolve_special_token_ids,
    encode_and_validate_document,
    ffd_pack_documents,
    EOT_ID,
    USER_ID,
    ASSISTANT_ID,
    THINK_OPEN_ID,
    THINK_CLOSE_ID,
)

TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"


def test_max_segment_tree_properties():
    """Verify O(log M) segment tree maintains exact capacities and first-fit guarantees."""
    tree = MaxSegmentTree(num_bins=4, bin_capacity=100)

    # Allocating 60 in bin 0
    b0 = tree.find_and_allocate(60)
    assert b0 == 0

    # Allocating 50: cannot fit in bin 0 (rem 40), fits in bin 1
    b1 = tree.find_and_allocate(50)
    assert b1 == 1

    # Allocating 30: fits in bin 0 (rem 40 -> 10)
    b0_second = tree.find_and_allocate(30)
    assert b0_second == 0

    # Allocating 15: cannot fit in bin 0 (rem 10), fits in bin 1 (rem 50 -> 35)
    b1_second = tree.find_and_allocate(15)
    assert b1_second == 1

    # Allocating 105: exceeds total capacity -> returns -1
    b_overflow = tree.find_and_allocate(105)
    assert b_overflow == -1


def test_special_token_resolution():
    """Verify that special token IDs dynamically match v3_32k ground truth."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    id_map = resolve_special_token_ids(tok)

    assert id_map["eot"] == 0
    assert id_map["user"] == 13
    assert id_map["assistant"] == 14
    assert id_map["think_open"] == 15
    assert id_map["think_close"] == 16


def test_encode_and_validate_document_quarantine():
    """Verify that malformed documents are correctly quarantined rather than corrupted."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    id_map = resolve_special_token_ids(tok)

    # 1. Unclosed string tag
    status, ids, mask = encode_and_validate_document(
        tok,
        prompt="Tell me something.",
        response="<think>I am thinking... No closure tag.",
        token_id_map=id_map,
    )
    assert status == "unclosed_think_string"
    assert ids is None and mask is None

    # 2. Exceeds max_seq_len
    long_prompt = "word " * 4100
    status, ids, mask = encode_and_validate_document(
        tok,
        prompt=long_prompt,
        response="Answer.",
        max_seq_len=4096,
        token_id_map=id_map,
    )
    assert status == "exceeds_4096"
    assert ids is None and mask is None

    # 3. Valid document passes with correct structure and mask
    status, ids, mask = encode_and_validate_document(
        tok,
        prompt="Solve 2 + 2",
        response="<think>2+2=4</think>\n4",
        lang="kan",
        max_seq_len=4096,
        token_id_map=id_map,
    )
    assert status == "ok"
    assert ids is not None and mask is not None
    assert len(ids) == len(mask)

    # Structure checks
    assert ids[0] == id_map["user"]
    assert ids[1] == 2  # <kan> id is 2
    assert id_map["assistant"] in ids
    assert ids[-1] == id_map["eot"]

    asst_idx = int(np.where(ids == id_map["assistant"])[0][0])
    # Prompt tokens (up to and including <|assistant|>) are masked to 0
    assert np.all(mask[: asst_idx + 1] == 0)
    # Assistant response + terminal EOT are supervised (mask == 1)
    assert np.all(mask[asst_idx + 1 :] == 1)


def test_ffd_packing_invariants():
    """Verify that bin packing preserves document atomicity, zero mid-doc truncation, and 100% EOT termination."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    id_map = resolve_special_token_ids(tok)

    docs = []
    # Create 50 synthetic documents of varying lengths (100 to 800 tokens)
    for i in range(50):
        prompt = f"Synthetic prompt number {i} with some extra padding text " * (i % 8 + 1)
        resp = f"<think>Reasoning step {i} detailed trace</think>\nResult {i}." * (i % 12 + 1)
        status, ids, mask = encode_and_validate_document(
            tok, prompt, resp, lang="eng", token_id_map=id_map
        )
        assert status == "ok"
        docs.append((ids, mask))

    bin_size = 4096
    packed_ids, packed_mask, b_3d, b_flat, stats = ffd_pack_documents(
        docs, bin_size=bin_size, seed=42, eot_id=id_map["eot"]
    )

    assert packed_ids.shape[1] == bin_size
    assert packed_mask.shape[1] == bin_size
    assert len(b_3d) == len(docs)
    assert len(b_flat) == len(docs)
    assert stats["packing_efficiency_pct"] > 70.0

    # Invariant: Every document in packed_ids terminates with EOT
    for bin_idx, start, end in b_3d:
        assert packed_ids[bin_idx, end - 1] == id_map["eot"]
        # Assistant tokens + terminal EOT must have mask 1
        assert packed_mask[bin_idx, end - 1] == 1

    # Invariant: Any trailing slack space must be padded with EOT and mask 0
    for row in range(packed_ids.shape[0]):
        row_docs = [b for b in b_3d if b[0] == row]
        last_end = max(b[2] for b in row_docs)
        if last_end < bin_size:
            assert np.all(packed_ids[row, last_end:] == id_map["eot"])
            assert np.all(packed_mask[row, last_end:] == 0)

    # Invariant: Flat boundaries match 3D boundaries
    for i, (bin_idx, start, end) in enumerate(b_3d):
        assert b_flat[i] == bin_idx * bin_size + start


def test_packer_cli_end_to_end():
    """Verify CLI packer execution with sample JSONL and output files."""
    script_path = REPO_ROOT / "scripts" / "pack_sft_4096_v4.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_jsonl = Path(tmp_dir) / "sample_sft.jsonl"
        out_dir = Path(tmp_dir) / "packed_out"

        with open(input_jsonl, "w", encoding="utf-8") as f:
            for i in range(25):
                row = {
                    "lang": "kan",
                    "prompt": f"ಪ್ರಶ್ನೆ ಸಂಖ್ಯೆ {i}",
                    "response": f"<think>ಚಿಂತನೆ ವಿವರ {i}</think>\nಉತ್ತರ {i}.",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        cmd = [
            sys.executable,
            str(script_path),
            "--tokenizer",
            str(TOKENIZER_PATH),
            "--inputs",
            str(input_jsonl),
            "--out-dir",
            str(out_dir),
            "--seq-len",
            "1024",  # smaller test bin size
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stats = json.loads(res.stdout)

        assert stats["total_packed_docs"] == 25
        assert (out_dir / "train.bin").exists()
        assert (out_dir / "train.mask.bin").exists()
        assert (out_dir / "train_boundaries.npy").exists()
        assert (out_dir / "train_boundaries_flat.npy").exists()
        assert (out_dir / "manifest.json").exists()

        # Check binary file sizes
        num_bins = stats["num_bins"]
        assert (out_dir / "train.bin").stat().st_size == num_bins * 1024 * 2
        assert (out_dir / "train.mask.bin").stat().st_size == num_bins * 1024 * 1
