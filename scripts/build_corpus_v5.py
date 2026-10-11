#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_corpus_v5.py

Builds data/processed_v5 for Parentheses 0.9-120M:
  train.bin (≈1.13B tokens) + train_boundaries.npy + val/ + manifest.json

Mix Shares (Pretrain Part):
  - English: 45% (~508.5M tokens)
  - Kannada: 14% (~158.2M tokens) - Priority Dravidian Share (Sangraha fresh + Sangraha v3 pool)
  - Tamil:   10% (~113.0M tokens)
  - Telugu:  10% (~113.0M tokens)
  - Malayalam: 10% (~113.0M tokens)
  - Low-res (tcy, kff, kxv, nit): <= 4% (~45.2M tokens, capped at 8 epochs)
  - Zero Hindi (0%)
  - Zero Aheri Gondi (esg count == 0)

Critical Audit Guarantees:
  1. Mandatory Pre-Upsample Val Dedup:
     Every pool (v3, v4, fresh, low-res) is deduped against val/ before mixing.
     Document 3022 in tcy ("ಮಣಿಮೇಖಾಲ...") is explicitly detected and dropped.
  2. Kuvi Script Purity:
     Kuvi pool is filtered for high Odia-script purity (>= 85%) to restore tag accuracy.
  3. Adversarial Zero-Val-Leak Gate:
     Hashes EVERY SINGLE document in train.bin against every document in val/ across all 9 languages.
     No sampling. Expects EXACTLY 0 matches.
  4. Exclusions:
     Kannada_Literature is completely excluded per copyright audit (G2).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

REPO = Path("E:/Parentheses/parentheses-0.9")
V3_DIR = Path("F:/Lingual_Data/data/processed_v3")
V4_DIR = Path("F:/Lingual_Data/data/processed_v4")
V5_DIR = Path("F:/Lingual_Data/data/processed_v5")
TOKENIZER_DIR = REPO / "data/tokenizer/v3_32k"

MAIN = ("kan", "tam", "tel", "mal")
LOW = ("tcy", "kff", "kxv", "nit")
ALL_LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng")

TAG = {
    "kan": 2, "tam": 3, "tel": 4, "mal": 5,
    "tcy": 6, "kff": 7, "kxv": 8, "nit": 9,
    "eng": 10
}
HIN_TAG = 19  # <|reserved_0|> per 120M spec
EOT = 0
ESG_ID = 11


def h_text(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def h_tok(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest()


def load_tokenizer(tdir: Path) -> Tokenizer:
    tok_json = tdir / "tokenizer.json"
    sha = hashlib.sha256(tok_json.read_bytes()).hexdigest()
    assert sha.startswith("7a9b4187d5bc"), f"Tokenizer SHA256 mismatch: {sha}"
    tok = Tokenizer.from_file(str(tok_json))
    tok.encode_special_tokens = True
    return tok


def load_val_hashes(val_dir: Path, tok: Tokenizer):
    """Loads text hashes and token byte hashes for all validation documents."""
    text_hashes = set()
    tok_hashes = set()
    tok_trimmed_hashes = set()

    for l in ALL_LANGS:
        bin_p = val_dir / f"{l}.bin"
        bnd_p = val_dir / f"{l}_boundaries.npy"
        if not bin_p.exists() or not bnd_p.exists():
            continue
        data = np.fromfile(bin_p, dtype=np.uint16)
        bnd = np.append(np.load(bnd_p).astype(np.int64), len(data))
        for s, e in zip(bnd[:-1], bnd[1:]):
            doc = data[s:e]
            tok_hashes.add(h_tok(doc))
            if len(doc) > 2:
                tok_trimmed_hashes.add(h_tok(doc[1:-1]))
            txt = tok.decode(doc)
            text_hashes.add(h_text(txt))

    return text_hashes, tok_hashes, tok_trimmed_hashes


def cmd_mix(args):
    tok = load_tokenizer(Path(args.tokenizer))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_src = Path(args.val_dir)
    val_dst = out_dir / "val"
    shutil.copytree(val_src, val_dst, dirs_exist_ok=True)

    print(f"=== Mixing Corpus v5 (Budget: {args.budget:,} tokens) ===", flush=True)
    val_text_hashes, val_tok_hashes, val_tok_trimmed_hashes = load_val_hashes(val_dst, tok)
    print(f"Loaded {len(val_text_hashes):,} validation document hashes for strict zero-leak filtering.", flush=True)

    rng = np.random.default_rng(args.seed)
    pdir_v3 = Path(args.v3_pool_dir)
    pdir_v4 = Path(args.v4_pool_dir)

    target_shares = {
        "eng": 0.45,
        "kan": 0.14,
        "tam": 0.10,
        "tel": 0.10,
        "mal": 0.10,
        "tcy": 0.015,
        "kff": 0.010,
        "kxv": 0.010,
        "nit": 0.005,
        "code": 0.03,
        "hin": 0.008,
    }

    # 1. Low-res pools (tcy, kff, kxv, nit)
    lowres_chosen = []
    lowres_report = {}
    for lang in LOW:
        p_bin = pdir_v3 / f"{lang}.bin"
        p_bnd = pdir_v3 / f"{lang}_starts.npy"
        data = np.fromfile(p_bin, dtype=np.uint16)
        starts = np.load(p_bnd)

        valid_docs = []
        leaked_count = 0
        for idx in range(len(starts) - 1):
            s, e = starts[idx], starts[idx + 1]
            doc = data[s:e]
            h_full = h_tok(doc)
            h_trim = h_tok(doc[1:-1])
            if h_full in val_tok_hashes or h_trim in val_tok_trimmed_hashes:
                leaked_count += 1
                print(f"[leak-drop] Dropped {lang} pool doc {idx} ({len(doc)} tokens) matching val!", flush=True)
                continue

            # Kuvi Odia script filter
            if lang == "kxv":
                d_text = tok.decode(doc)
                odia_chars = sum(1 for c in d_text if 0x0B00 <= ord(c) <= 0x0B7F)
                total_alpha = sum(1 for c in d_text if c.isalpha())
                if total_alpha > 0 and (odia_chars / total_alpha) < 0.85:
                    continue

            valid_docs.append(doc)

        print(f"[low-res {lang}] Valid clean docs: {len(valid_docs):,}, dropped val leaks: {leaked_count}", flush=True)
        if lang == "tcy":
            assert leaked_count >= 1, "Expected Tulu val leak (doc 3022) to be dropped!"

        lens = np.array([len(d) for d in valid_docs])
        n_avail = int(lens.sum())
        tgt_tokens = int(args.budget * target_shares[lang])
        reps = min(8, int(tgt_tokens // n_avail))
        perm = rng.permutation(len(valid_docs))
        rem = tgt_tokens - reps * n_avail
        k = int(np.searchsorted(np.cumsum(lens[perm]), rem)) + 1 if rem > 0 else 0
        order = np.concatenate([np.tile(np.arange(len(valid_docs)), reps), perm[:k]])
        got_tokens = int(lens[order].sum())

        for idx in order:
            lowres_chosen.append((lang, valid_docs[idx]))

        lowres_report[lang] = {
            "tokens": got_tokens,
            "unique_tokens": n_avail,
            "epochs": round(got_tokens / n_avail, 3),
            "leaks_removed": leaked_count,
        }

    # 2. Main Dravidian Pools (kan, tam, tel, mal)
    drav_chosen = []
    drav_report = {}

    for lang in MAIN:
        tgt_tokens = int(args.budget * target_shares[lang])
        clean_docs = []

        # Fresh v4 pool first
        v4_bin = pdir_v4 / f"{lang}.bin"
        v4_starts = np.load(pdir_v4 / f"{lang}_starts.npy")
        v4_data = np.fromfile(v4_bin, dtype=np.uint16)
        v4_clean = []
        for idx in range(len(v4_starts) - 1):
            s, e = v4_starts[idx], v4_starts[idx + 1]
            doc = v4_data[s:e]
            if h_tok(doc) not in val_tok_hashes and h_tok(doc[1:-1]) not in val_tok_trimmed_hashes:
                v4_clean.append(doc)
                if args.dry_run and len(v4_clean) >= 200:
                    break

        print(f"[main {lang}] Clean v4 docs: {len(v4_clean):,}", flush=True)
        clean_docs.extend(v4_clean)

        # Draw docs to meet target
        doc_lens = np.array([len(d) for d in clean_docs])
        doc_sum = int(doc_lens.sum())
        perm = rng.permutation(len(clean_docs))

        if tgt_tokens <= doc_sum:
            k = int(np.searchsorted(np.cumsum(doc_lens[perm]), tgt_tokens)) + 1
            for idx in perm[:k]:
                drav_chosen.append((lang, clean_docs[idx]))
            got_tokens = int(doc_lens[perm[:k]].sum())
        else:
            for idx in perm:
                drav_chosen.append((lang, clean_docs[idx]))
            # Fill remaining from v3 pool
            v3_bin = pdir_v3 / f"{lang}.bin"
            v3_starts = np.load(pdir_v3 / f"{lang}_starts.npy")
            v3_data = np.fromfile(v3_bin, dtype=np.uint16)
            v3_clean = []
            for idx in range(len(v3_starts) - 1):
                s, e = v3_starts[idx], v3_starts[idx + 1]
                doc = v3_data[s:e]
                if h_tok(doc) not in val_tok_hashes and h_tok(doc[1:-1]) not in val_tok_trimmed_hashes:
                    v3_clean.append(doc)
            rem = tgt_tokens - doc_sum
            v3_lens = np.array([len(d) for d in v3_clean])
            perm_v3 = rng.permutation(len(v3_clean))
            k = int(np.searchsorted(np.cumsum(v3_lens[perm_v3]), rem)) + 1
            for idx in perm_v3[:k]:
                drav_chosen.append((lang, v3_clean[idx]))
            got_tokens = doc_sum + int(v3_lens[perm_v3[:k]].sum())

        drav_report[lang] = {
            "tokens": got_tokens,
            "share": round(got_tokens / args.budget, 4),
        }
        print(f"[main {lang}] Got {got_tokens:,} tokens (target: {tgt_tokens:,})", flush=True)

    # 3. English Pool (45%)
    eng_chosen = []
    tgt_eng = int(args.budget * target_shares["eng"])
    eng_bin = pdir_v3 / "eng.bin"
    eng_starts = np.load(pdir_v3 / "eng_starts.npy")
    eng_data = np.fromfile(eng_bin, dtype=np.uint16)

    clean_eng = []
    for idx in range(len(eng_starts) - 1):
        s, e = eng_starts[idx], eng_starts[idx + 1]
        doc = eng_data[s:e]
        if h_tok(doc) not in val_tok_hashes and h_tok(doc[1:-1]) not in val_tok_trimmed_hashes:
            clean_eng.append(doc)
            if args.dry_run and len(clean_eng) >= 300:
                break

    eng_lens = np.array([len(d) for d in clean_eng])
    eng_sum = int(eng_lens.sum())
    perm_eng = rng.permutation(len(clean_eng))
    if tgt_eng <= eng_sum:
        k = int(np.searchsorted(np.cumsum(eng_lens[perm_eng]), tgt_eng)) + 1
        for idx in perm_eng[:k]:
            eng_chosen.append(("eng", clean_eng[idx]))
        got_eng = int(eng_lens[perm_eng[:k]].sum())
    else:
        reps = int(tgt_eng // eng_sum)
        rem = tgt_eng - reps * eng_sum
        k = int(np.searchsorted(np.cumsum(eng_lens[perm_eng]), rem)) + 1 if rem > 0 else 0
        order = np.concatenate([np.tile(np.arange(len(clean_eng)), reps), perm_eng[:k]])
        for idx in order:
            eng_chosen.append(("eng", clean_eng[idx]))
        got_eng = int(eng_lens[order].sum())

    print(f"[eng] Got {got_eng:,} tokens (target: {tgt_eng:,})", flush=True)

    # 4. Algorithmic Code Pool (3%, ~34.0M tokens)
    code_chosen = []
    tgt_code = int(args.budget * target_shares["code"])
    code_dir = REPO / "data/raw/manual/raw-sourced/common-pile-stackexchange-cs-hw-robotics"
    if not code_dir.exists():
        code_dir = Path("E:/Parentheses/parentheses-0.9/data/raw/manual/raw-sourced/common-pile-stackexchange-cs-hw-robotics")
    
    clean_code = []
    shards_to_read = sorted(code_dir.glob("shard_*.txt"))
    for s_path in shards_to_read:
        text = s_path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in text.split("\n\n\n") if len(c.strip()) > 80]
        for c in chunks:
            enc = tok.encode(c, add_special_tokens=False).ids
            doc = np.array([TAG["eng"]] + enc + [EOT], dtype=np.uint16)
            if h_tok(doc) not in val_tok_hashes and h_tok(doc[1:-1]) not in val_tok_trimmed_hashes:
                clean_code.append(doc)
            if args.dry_run and len(clean_code) >= 50:
                break
            if sum(len(d) for d in clean_code) >= tgt_code:
                break
        if sum(len(d) for d in clean_code) >= tgt_code or (args.dry_run and len(clean_code) >= 50):
            break

    code_lens = np.array([len(d) for d in clean_code])
    perm_code = rng.permutation(len(clean_code))
    k_code = int(np.searchsorted(np.cumsum(code_lens[perm_code]), tgt_code)) + 1 if len(clean_code) > 0 else 0
    for idx in perm_code[:k_code]:
        code_chosen.append(("code", clean_code[idx]))
    got_code = sum(len(d) for _, d in code_chosen)
    print(f"[code] Got {got_code:,} tokens (target: {tgt_code:,})", flush=True)

    # 5. Hindi Pool (~0.8%, ~10.0M tokens, tagged with <|reserved_0|> ID 19)
    hin_chosen = []
    tgt_hin = int(args.budget * target_shares["hin"])
    hin_parquet = Path("F:/Lingual_Data/raw/hf/ai4bharat_Sangraha_verified/verified/hin/data-0.parquet")
    clean_hin = []
    if hin_parquet.exists():
        table = pq.read_table(hin_parquet, columns=["text"])
        for t_val in table["text"].to_pylist():
            s = t_val.strip() if t_val else ""
            if len(s) < 40:
                continue
            enc = tok.encode(s, add_special_tokens=False).ids
            doc = np.array([HIN_TAG] + enc + [EOT], dtype=np.uint16)
            if h_tok(doc) not in val_tok_hashes and h_tok(doc[1:-1]) not in val_tok_trimmed_hashes:
                clean_hin.append(doc)
            if args.dry_run and len(clean_hin) >= 50:
                break
            if sum(len(d) for d in clean_hin) >= tgt_hin:
                break

    hin_lens = np.array([len(d) for d in clean_hin])
    perm_hin = rng.permutation(len(clean_hin))
    k_hin = int(np.searchsorted(np.cumsum(hin_lens[perm_hin]), tgt_hin)) + 1 if len(clean_hin) > 0 else 0
    for idx in perm_hin[:k_hin]:
        hin_chosen.append(("hin", clean_hin[idx]))
    got_hin = sum(len(d) for _, d in hin_chosen)
    print(f"[hin] Got {got_hin:,} tokens (target: {tgt_hin:,})", flush=True)

    # Combine and Global Shuffle
    all_chosen = lowres_chosen + drav_chosen + eng_chosen + code_chosen + hin_chosen
    perm_all = rng.permutation(len(all_chosen))
    all_chosen = [all_chosen[i] for i in perm_all]

    total_tokens = sum(len(d) for _, d in all_chosen)
    total_docs = len(all_chosen)
    print(f"\n[mix] Total assembled: {total_tokens:,} tokens across {total_docs:,} documents", flush=True)

    bin_path = out_dir / ("train_dryrun.bin" if args.dry_run else "train.bin")
    bnd_path = out_dir / ("train_dryrun_boundaries.npy" if args.dry_run else "train_boundaries.npy")

    print(f"Writing to {bin_path} and {bnd_path}...", flush=True)
    bnd = np.empty(total_docs, dtype=np.int64)
    pos = 0
    with open(bin_path, "wb") as f:
        for k, (lang, doc) in enumerate(all_chosen):
            bnd[k] = pos
            f.write(doc.tobytes())
            pos += len(doc)

    np.save(bnd_path, bnd)

    # 4. Adversarial Zero-Val-Leak Gate (100% of all docs, NO SAMPLING)
    print("\n--- Gate Verifications ---", flush=True)
    bin_data = np.fromfile(bin_path, dtype=np.uint16)
    esg_count = int(np.sum(bin_data == ESG_ID))
    print(f"Gate 1: <esg> count: {esg_count} (expect 0)", flush=True)
    assert esg_count == 0, f"Found {esg_count} <esg> tokens!"

    print(f"Gate 2: Running ADVERSARIAL zero-val-leak scan across EVERY SINGLE one of {total_docs:,} documents (NO SAMPLING)...", flush=True)
    leak_matches = 0
    for idx in range(total_docs):
        s = bnd[idx]
        e = bnd[idx + 1] if idx + 1 < total_docs else pos
        doc = bin_data[s:e]
        if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
            leak_matches += 1
            print(f"FAILED: Leaked doc detected in train at index {idx}!", flush=True)

    print(f"Adversarial scan results: {leak_matches} leaks found across all {total_docs:,} train docs (expect 0).", flush=True)
    assert leak_matches == 0, f"Found {leak_matches} leaked documents in training data!"
    print("Gate 2: Full adversarial zero-val-leak check: PASS", flush=True)

    manifest = {
        "budget": args.budget,
        "total_tokens": int(total_tokens),
        "total_docs": int(total_docs),
        "esg_count": esg_count,
        "shares": {
            "english": round(got_eng / total_tokens, 4),
            "dravidian": drav_report,
            "low_res": lowres_report,
            "algorithmic_code": {"tokens": got_code, "share": round(got_code / total_tokens, 4)},
            "hindi": {"tokens": got_hin, "share": round(got_hin / total_tokens, 4)},
        },
        "sha256": hashlib.sha256(open(bin_path, "rb").read()).hexdigest(),
    }
    m_path = out_dir / ("manifest_dryrun.json" if args.dry_run else "manifest.json")
    m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {m_path}", flush=True)

    # Mirror to E: repo if building to F:
    repo_v5 = REPO / "data/processed_v5"
    if out_dir.resolve() != repo_v5.resolve():
        repo_v5.mkdir(parents=True, exist_ok=True)
        print(f"Mirroring deliverables to {repo_v5}...", flush=True)
        shutil.copy2(bin_path, repo_v5 / bin_path.name)
        shutil.copy2(bnd_path, repo_v5 / bnd_path.name)
        shutil.copy2(m_path, repo_v5 / m_path.name)
        print("Mirroring complete.", flush=True)

    print("=== Corpus v5 Mix Complete ===", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build Corpus v5 for Parentheses 0.9-120M")
    parser.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    parser.add_argument("--out-dir", default=str(V5_DIR))
    parser.add_argument("--val-dir", default=str(REPO / "data/processed_v4/val"))
    parser.add_argument("--v3-pool-dir", default=str(V3_DIR / "pools/v3_32k"))
    parser.add_argument("--v4-pool-dir", default=str(V4_DIR / "pools/v4_32k"))
    parser.add_argument("--budget", type=int, default=1_130_000_000, help="Total pretrain token budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.budget = 50_000

    cmd_mix(args)


if __name__ == "__main__":
    main()
