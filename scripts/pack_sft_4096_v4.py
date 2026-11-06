#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/pack_sft_4096_v4.py
Parentheses 1.0 (240M / slab_v4) 4,096-Token First-Fit-Decreasing (FFD) Document-Aware Packer.
Invariants enforced:
  1. Zero mid-document truncation: every packed document fits completely inside a 4,096-token bin.
  2. Zero unclosed <think> tags: every reasoning document is verified to contain matching </think>.
  3. 100% <|endoftext|> termination: every packed document ends with EOT (token 0), triggering
     the slab_v4 tri-channel recurrent boundary reset (alpha_t=0, conv=0, mu=1) on the next token.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tokenizers import Tokenizer

EOT_ID = 0
USER_ID = 13
ASSISTANT_ID = 14
THINK_OPEN_ID = 15
THINK_CLOSE_ID = 16
IGNORE_INDEX = -100

LANG_TO_ID = {
    "kan": 2, "kn": 2,
    "tam": 3, "ta": 3,
    "tel": 4, "te": 4,
    "mal": 5, "ml": 5,
    "eng": 10, "en": 10,
}


def resolve_special_token_ids(tok: Tokenizer) -> Dict[str, int]:
    """Dynamically resolve special token IDs from tokenizer to ensure 100% vocabulary alignment."""
    def _get(tok_str: str, default: int) -> int:
        tid = tok.token_to_id(tok_str)
        return tid if tid is not None else default

    return {
        "eot": _get("<|endoftext|>", EOT_ID),
        "user": _get("<|user|>", USER_ID),
        "assistant": _get("<|assistant|>", ASSISTANT_ID),
        "think_open": _get("<think>", THINK_OPEN_ID),
        "think_close": _get("</think>", THINK_CLOSE_ID),
    }


class MaxSegmentTree:
    """
    O(log M) Segment Tree for First-Fit-Decreasing (FFD) bin packing across M bins.
    Finds the lowest-index bin with remaining capacity >= doc_len.
    """
    def __init__(self, num_bins: int, bin_capacity: int) -> None:
        self.n = 1
        while self.n < num_bins:
            self.n <<= 1
        self.capacity = bin_capacity
        self.tree = [bin_capacity] * (2 * self.n)

    def find_and_allocate(self, doc_len: int) -> int:
        idx = 1
        if self.tree[idx] < doc_len:
            return -1
        while idx < self.n:
            left = 2 * idx
            if self.tree[left] >= doc_len:
                idx = left
            else:
                idx = left + 1
        bin_idx = idx - self.n
        self.tree[idx] -= doc_len
        idx >>= 1
        while idx >= 1:
            self.tree[idx] = max(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx >>= 1
        return bin_idx


def encode_and_validate_document(
    tok: Tokenizer,
    prompt: str,
    response: str,
    lang: str = "eng",
    max_seq_len: int = 4096,
    token_id_map: Dict[str, int] | None = None,
) -> Tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Encode a single SFT document into (input_ids, target_mask) where:
      - input_ids: [L] uint16 token sequence: <|user|><lang> prompt <|assistant|> response <|endoftext|>
      - target_mask: [L] uint8 mask (1 where loss is active on assistant response + EOT, 0 on prompt)
    Returns (status, input_ids, loss_mask).
    """
    if token_id_map is None:
        token_id_map = resolve_special_token_ids(tok)

    eot_id = token_id_map["eot"]
    user_id = token_id_map["user"]
    asst_id = token_id_map["assistant"]
    th_open = token_id_map["think_open"]
    th_close = token_id_map["think_close"]

    open_c = response.count("<think>")
    close_c = response.count("</think>")
    if open_c != close_c:
        return "unclosed_think_string", None, None

    lang_id = LANG_TO_ID.get(lang.lower(), 10)
    p_ids = tok.encode(prompt.strip()).ids
    r_ids = tok.encode(response.strip()).ids

    full_ids = [user_id, lang_id] + p_ids + [asst_id] + r_ids + [eot_id]
    L = len(full_ids)
    if L > max_seq_len:
        return "exceeds_4096", None, None

    # Verify token-level <think> and </think> balance
    if full_ids.count(th_open) != full_ids.count(th_close):
        return "unclosed_think_token", None, None

    prompt_len = 2 + len(p_ids) + 1  # <|user|> + <lang> + prompt + <|assistant|>
    ids_arr = np.array(full_ids, dtype=np.uint16)
    mask_arr = np.zeros(L, dtype=np.uint8)
    mask_arr[prompt_len:] = 1  # Supervise assistant response tokens + terminal <|endoftext|>
    return "ok", ids_arr, mask_arr


def ffd_pack_documents(
    docs: List[Tuple[np.ndarray, np.ndarray]],
    bin_size: int = 4096,
    seed: int = 42,
    eot_id: int = EOT_ID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Pack variable-length (ids, mask) documents into exact `bin_size` (4096) rows using FFD.
    Returns:
      - packed_ids:       [num_bins, bin_size] uint16
      - packed_mask:      [num_bins, bin_size] uint8 (1 = supervise target, 0 = ignore / -100)
      - boundaries_3d:    [total_packed_docs, 3] int32 (bin_idx, start_offset, end_offset)
      - boundaries_flat:  [total_packed_docs] int64 flat 1D token offsets
      - stats:            summary metrics dictionary
    """
    if not docs:
        return (
            np.zeros((0, bin_size), dtype=np.uint16),
            np.zeros((0, bin_size), dtype=np.uint8),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            {"num_bins": 0, "bin_size": bin_size, "total_packed_docs": 0, "packing_efficiency_pct": 0.0},
        )

    # Sort descending by length for First-Fit-Decreasing optimality
    order = sorted(range(len(docs)), key=lambda i: len(docs[i][0]), reverse=True)
    max_possible_bins = len(docs)
    seg_tree = MaxSegmentTree(max_possible_bins, bin_size)

    bins_ids: List[List[np.ndarray]] = []
    bins_masks: List[List[np.ndarray]] = []
    bins_used_len: List[int] = []

    for doc_idx in order:
        ids_arr, mask_arr = docs[doc_idx]
        d_len = len(ids_arr)
        b_idx = seg_tree.find_and_allocate(d_len)
        if b_idx >= len(bins_ids):
            bins_ids.append([ids_arr])
            bins_masks.append([mask_arr])
            bins_used_len.append(d_len)
        else:
            bins_ids[b_idx].append(ids_arr)
            bins_masks[b_idx].append(mask_arr)
            bins_used_len[b_idx] += d_len

    num_bins = len(bins_ids)
    # Shuffle bin order deterministically so batches contain uniform length mixtures
    rng = np.random.default_rng(seed)
    bin_perm = rng.permutation(num_bins)

    packed_ids = np.full((num_bins, bin_size), eot_id, dtype=np.uint16)
    packed_mask = np.zeros((num_bins, bin_size), dtype=np.uint8)
    boundaries_3d: List[Tuple[int, int, int]] = []
    boundaries_flat: List[int] = []

    total_doc_tokens = 0
    total_supervised_tokens = 0

    for out_row, b_idx in enumerate(bin_perm):
        # Shuffle documents inside each bin to eliminate descending-length ordering bias within a 4096 window
        doc_list_ids = bins_ids[b_idx]
        doc_list_masks = bins_masks[b_idx]
        inner_perm = rng.permutation(len(doc_list_ids))

        cursor = 0
        for j in inner_perm:
            d_ids = doc_list_ids[j]
            d_msk = doc_list_masks[j]
            L = len(d_ids)
            packed_ids[out_row, cursor : cursor + L] = d_ids
            packed_mask[out_row, cursor : cursor + L] = d_msk
            boundaries_3d.append((out_row, cursor, cursor + L))
            boundaries_flat.append(out_row * bin_size + cursor)
            cursor += L
            total_doc_tokens += L
            total_supervised_tokens += int(d_msk.sum())

        # Any trailing slack [cursor : bin_size] remains eot_id (0) with packed_mask = 0 (ignored)

    total_capacity = num_bins * bin_size
    stats = {
        "num_bins": num_bins,
        "bin_size": bin_size,
        "total_packed_docs": len(boundaries_3d),
        "total_doc_tokens": total_doc_tokens,
        "total_supervised_tokens": total_supervised_tokens,
        "total_capacity_tokens": total_capacity,
        "packing_efficiency_pct": round(total_doc_tokens / max(1, total_capacity) * 100.0, 3),
        "truncated_docs": 0,
        "unclosed_think_docs": 0,
    }
    return (
        packed_ids,
        packed_mask,
        np.array(boundaries_3d, dtype=np.int32),
        np.array(boundaries_flat, dtype=np.int64),
        stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="4096-Token FFD Document-Aware SFT Packer (slab_v4)")
    parser.add_argument("--tokenizer", required=True, help="Path to v3_32k tokenizer.json")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input SFT JSONL files (including abstention JSONL)")
    parser.add_argument("--out-dir", required=True, help="Output directory for train.bin, train.mask.bin, manifest.json")
    parser.add_argument("--seq-len", type=int, default=4096, help="Packing bin length (default: 4096)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed (default: 42)")
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    token_id_map = resolve_special_token_ids(tok)

    valid_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    quarantined_counts = {"exceeds_4096": 0, "unclosed_think_string": 0, "unclosed_think_token": 0}

    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", row.get("instruction", row.get("question", "")))
                response = row.get("response", row.get("output", row.get("chosen", row.get("answer", ""))))
                lang = row.get("lang", row.get("language", "eng"))
                status, ids_arr, mask_arr = encode_and_validate_document(
                    tok, prompt, response, lang=lang, max_seq_len=args.seq_len, token_id_map=token_id_map
                )
                if status == "ok" and ids_arr is not None and mask_arr is not None:
                    valid_docs.append((ids_arr, mask_arr))
                else:
                    quarantined_counts[status] = quarantined_counts.get(status, 0) + 1

    packed_ids, packed_mask, boundaries_3d, boundaries_flat, stats = ffd_pack_documents(
        valid_docs, bin_size=args.seq_len, seed=args.seed, eot_id=token_id_map["eot"]
    )
    stats["quarantined"] = quarantined_counts

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packed_ids.tofile(out_dir / "train.bin")
    packed_mask.tofile(out_dir / "train.mask.bin")
    np.save(out_dir / "train_boundaries.npy", boundaries_3d)
    np.save(out_dir / "train_boundaries_flat.npy", boundaries_flat)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(stats, mf, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
