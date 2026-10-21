#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates training_data_ledger_2026-10-17.jsonl and checkpoint_0.9-120M governance manifests
for F:\\Lingual_Data\\Trained-On\\LEDGER\\, plus delete_list_2026-10-17.md.

Covers all assets trained on for Parentheses 0.9-120M:
- Pretraining Sessions 1–5: data/processed_v5 (train.bin, sft_tail.bin), qa_mix (S3), qa_mix_s4 (S4)
- Post-Training SFT Runs: processed_sft_120m (v1), processed_sft_120m_v2, processed_sft_120m_v3, processed_sft_v1
- Superseded legacy byte corpora: processed_all_sources_v1, processed_dravidian_v1, processed_multilingual, processed
- Raw sourced directories on E: and unused pools on F:
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

REPO = Path("E:/Parentheses/parentheses-0.9")
FLAKE = Path("F:/Lingual_Data")
LEDGER_DIR = FLAKE / "Trained-On/LEDGER"
OUT_JSONL = LEDGER_DIR / "training_data_ledger_2026-10-17.jsonl"
OUT_MANIFEST = LEDGER_DIR / "checkpoint_0.9-120M_manifest.md"
OUT_INDEX_JSON = LEDGER_DIR / "checkpoint_0.9-120M_master_index.json"
OUT_DELETE_LIST = LEDGER_DIR / "delete_list_2026-10-17.md"

def quick_hash(p: Path, max_bytes: int = 50 * 1024 * 1024) -> str:
    """Computes SHA-256 for files up to max_bytes, or marks deferred if on F: and huge."""
    sz = p.stat().st_size
    if sz > max_bytes and str(p).startswith("F:"):
        return "deferred"
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        while chunk := fp.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()

def make_record(path: Path, kind: str, source: str, license_str: str,
                used_by: list, rederive: str, known_hash: str = None, date_str="2026-10-17") -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    sz = path.stat().st_size
    sha256_val = known_hash if known_hash else quick_hash(path)
    norm_path = str(path).replace("\\", "/")
    return {
        "path": norm_path,
        "bytes": sz,
        "sha256": sha256_val,
        "kind": kind,
        "source": source,
        "license": license_str,
        "used_by": used_by,
        "rederive": rederive,
        "date": date_str
    }

def main():
    print("=== Building Parentheses 0.9-120M Trained-On Ledger & Manifest ===")
    records = []

    # 1. Processed v5 Pretraining Mix (1.075B tokens, Sessions 1-5)
    v5_dir = REPO / "data/processed_v5"
    if v5_dir.exists():
        v5_train = v5_dir / "train.bin"
        if v5_train.exists():
            records.append(make_record(
                v5_train, "train_bin",
                "Corpus v5 mix: Sangraha verified + low-res Dravidian + English high-quality",
                "CC BY 4.0 / CC BY-SA 4.0 / Gutenberg PD",
                [{"run": "parentheses-0.9-120M-S1-S5", "tokens": 1075946234, "epochs": 1.16}],
                "python scripts/build_corpus_v5.py"
            ))
        v5_bounds = v5_dir / "train_boundaries.npy"
        if v5_bounds.exists():
            records.append(make_record(
                v5_bounds, "boundaries",
                "Corpus v5 document boundaries (1,143,098 docs)",
                "CC BY 4.0 / CC BY-SA 4.0",
                [{"run": "parentheses-0.9-120M-S1-S5", "tokens": 1075946234, "epochs": 1.16}],
                "python scripts/build_corpus_v5.py"
            ))
        v5_manifest = v5_dir / "manifest.json"
        if v5_manifest.exists():
            records.append(make_record(
                v5_manifest, "manifest",
                "Corpus v5 generation metadata and language distribution",
                "Apache-2.0",
                [],
                "python scripts/build_corpus_v5.py"
            ))
        sft_tail = v5_dir / "sft_tail.bin"
        if sft_tail.exists():
            records.append(make_record(
                sft_tail, "sft_bin",
                "Session 5 SFT tail co-training mix (62.5M tokens)",
                "Permissive / Research-Only (Joel)",
                [{"run": "parentheses-0.9-120M-S5", "tokens": 62555274, "epochs": 1.0}],
                "python scripts/build_sft_tail.py"
            ))
        sft_tail_bounds = v5_dir / "sft_tail_boundaries.npy"
        if sft_tail_bounds.exists():
            records.append(make_record(
                sft_tail_bounds, "boundaries",
                "Session 5 SFT tail boundaries (127,148 docs)",
                "Permissive",
                [{"run": "parentheses-0.9-120M-S5", "tokens": 62555274, "epochs": 1.0}],
                "python scripts/build_sft_tail.py"
            ))
        sft_tail_manifest = v5_dir / "sft_tail_manifest.json"
        if sft_tail_manifest.exists():
            records.append(make_record(
                sft_tail_manifest, "manifest",
                "Session 5 SFT tail manifest metadata",
                "Apache-2.0",
                [],
                "python scripts/build_sft_tail.py"
            ))

    # 2. QA Mix Data (Sessions 3 and 4)
    qa_s4_dir = REPO / "data/processed_qa_mix_s4"
    if qa_s4_dir.exists():
        qa_s4_train = qa_s4_dir / "train.bin"
        if qa_s4_train.exists():
            records.append(make_record(
                qa_s4_train, "train_bin",
                "Session 4 17-lane QA & Dictionary mix (107.8M tokens: Alar, Olam, Kaikki, TyDiQA, HotpotQA)",
                "CC BY 4.0 / CC BY-SA 4.0 / ODbL / WordNet BSD",
                [{"run": "parentheses-0.9-120M-S4", "tokens": 107864113, "epochs": 1.0}],
                "python scripts/build_qa_mix_s4.py"
            ))
        qa_s4_bounds = qa_s4_dir / "train_boundaries.npy"
        if qa_s4_bounds.exists():
            records.append(make_record(
                qa_s4_bounds, "boundaries",
                "Session 4 QA mix doc boundaries (721,068 docs)",
                "Permissive / ODbL",
                [{"run": "parentheses-0.9-120M-S4", "tokens": 107864113, "epochs": 1.0}],
                "python scripts/build_qa_mix_s4.py"
            ))
        qa_s4_manifest = qa_s4_dir / "manifest.json"
        if qa_s4_manifest.exists():
            records.append(make_record(
                qa_s4_manifest, "manifest",
                "Session 4 QA mix manifest metadata",
                "Apache-2.0",
                [],
                "python scripts/build_qa_mix_s4.py"
            ))

    qa_s3_dir = REPO / "data/processed_qa_mix"
    if qa_s3_dir.exists():
        qa_s3_train = qa_s3_dir / "train.bin"
        if qa_s3_train.exists():
            records.append(make_record(
                qa_s3_train, "train_bin",
                "Session 3 16-lane QA mix (30.6M tokens)",
                "CC BY 4.0 / CC BY-SA 4.0 / Apache-2.0",
                [{"run": "parentheses-0.9-120M-S3", "tokens": 30670000, "epochs": 1.0}],
                "python scripts/build_qa_mix.py"
            ))
        qa_s3_bounds = qa_s3_dir / "train_boundaries.npy"
        if qa_s3_bounds.exists():
            records.append(make_record(
                qa_s3_bounds, "boundaries",
                "Session 3 QA mix doc boundaries",
                "CC BY 4.0",
                [{"run": "parentheses-0.9-120M-S3", "tokens": 30670000, "epochs": 1.0}],
                "python scripts/build_qa_mix.py"
            ))
        qa_s3_manifest = qa_s3_dir / "manifest.json"
        if qa_s3_manifest.exists():
            records.append(make_record(
                qa_s3_manifest, "manifest",
                "Session 3 QA mix manifest metadata",
                "Apache-2.0",
                [],
                "python scripts/build_qa_mix.py"
            ))

    # 3. Post-Training SFT Datasets (v1, v2, v3, v4)
    sft_v2_dir = REPO / "data/processed_sft_120m_v2"
    if sft_v2_dir.exists():
        for fname, kind in [
            ("train.bin", "train_bin"), ("train.mask.bin", "mask"), ("train_boundaries.npy", "boundaries"),
            ("val.bin", "val"), ("val.mask.bin", "mask"), ("val_boundaries.npy", "boundaries"), ("manifest.json", "manifest")
        ]:
            p = sft_v2_dir / fname
            if p.exists():
                records.append(make_record(
                    p, kind,
                    "SFT v2 0% leakage conversational & reasoning mix (32.2M tokens)",
                    "Research-Only (Joel) / Permissive",
                    [{"run": "parentheses-0.9-120M-sft-v2", "tokens": 32267581, "epochs": 1.0}],
                    "python scripts/build_sft_120m_v2.py"
                ))

    sft_v1_dir = REPO / "data/processed_sft_120m"
    if sft_v1_dir.exists():
        for fname, kind in [
            ("train.bin", "train_bin"), ("train.mask.bin", "mask"), ("train_boundaries.npy", "boundaries"),
            ("val.bin", "val"), ("val.mask.bin", "mask"), ("val_boundaries.npy", "boundaries"), ("manifest.json", "manifest")
        ]:
            p = sft_v1_dir / fname
            if p.exists():
                records.append(make_record(
                    p, kind,
                    "SFT v1 post-training mix (85.7M tokens; superseded by v2)",
                    "Research-Only / Permissive",
                    [{"run": "parentheses-0.9-120M-sft", "tokens": 13107200, "epochs": 0.15}],
                    "python scripts/build_sft_120m.py"
                ))

    sft_v3_dir = REPO / "data/processed_sft_120m_v3"
    if sft_v3_dir.exists():
        for fname, kind in [
            ("train.bin", "train_bin"), ("train.mask.bin", "mask"), ("train_boundaries.npy", "boundaries"),
            ("val.bin", "val"), ("val.mask.bin", "mask"), ("val_boundaries.npy", "boundaries"), ("manifest.json", "manifest")
        ]:
            p = sft_v3_dir / fname
            if p.exists():
                records.append(make_record(
                    p, kind,
                    "SFT v3/v4 post-training mix (34.7M tokens)",
                    "Research-Only / Permissive",
                    [{"run": "parentheses-0.9-120M-sft-v4", "tokens": 19660800, "epochs": 0.56}],
                    "python scripts/build_sft_120m_v3.py"
                ))

    sft_legacy_v1_dir = REPO / "data/processed_sft_v1"
    if sft_legacy_v1_dir.exists():
        for fname, kind in [
            ("train.bin", "train_bin"), ("train.mask.bin", "mask"), ("train_boundaries.npy", "boundaries"),
            ("val.bin", "val"), ("val.mask.bin", "mask"), ("val_boundaries.npy", "boundaries"), ("manifest.json", "manifest")
        ]:
            p = sft_legacy_v1_dir / fname
            if p.exists():
                records.append(make_record(
                    p, kind,
                    "Legacy 41M SFT v1 mix (10.1M tokens)",
                    "CC BY 4.0 / Apache-2.0",
                    [{"run": "parentheses-0.9-41M-sft", "tokens": 10103998, "epochs": 1.0}],
                    "python scripts/build_sft_tail.py"
                ))

    # 4. Legacy Byte Corpora Superseded by BPE
    for dir_name, src_name, lic, run_name in [
        ("data/processed_all_sources_v1", "Early byte replay (Gutenberg, math, web)", "PD / Open", "byte-runs-v1-v2"),
        ("data/processed_dravidian_v1", "Early byte Dravidian (Sangraha byte-level)", "CC BY 4.0", "byte-runs-v1-v3"),
        ("data/processed_multilingual", "Early byte multilingual experimental", "Open", "byte-exp"),
        ("data/processed", "Early byte baseline processed", "Open", "baseline-byte"),
    ]:
        d = REPO / dir_name
        if d.exists():
            for f in d.glob("*"):
                if f.is_file():
                    k = "train_bin" if "train.bin" in f.name else ("val" if "val.bin" in f.name else "manifest")
                    records.append(make_record(
                        f, k, src_name, lic, [{"run": run_name, "tokens": 0, "epochs": 1.0}], "python scripts/tokenize_corpus.py"
                    ))

    # 5. Raw Sourced Repositories on E:
    raw_sourced = REPO / "data/raw/manual/raw-sourced"
    if raw_sourced.exists():
        for sub in raw_sourced.iterdir():
            if sub.is_dir():
                sub_sz = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
                f_count = sum(1 for f in sub.rglob("*") if f.is_file())
                # Log top-level directory record
                records.append({
                    "path": str(sub).replace("\\", "/"),
                    "bytes": sub_sz,
                    "sha256": f"directory_{f_count}_files",
                    "kind": "raw_dir",
                    "source": f"Upstream open source dataset: {sub.name}",
                    "license": "See ATTRIBUTION.md in raw-sourced",
                    "used_by": [{"run": "parentheses-0.9-120M-pretrain", "tokens": 0, "epochs": 1.0}],
                    "rederive": f"huggingface-cli download / curl (documented in ATTRIBUTION.md)",
                    "date": "2026-10-17"
                })

    # 6. Legacy / Superseded Pools on F:
    for pool_rel, desc in [
        ("data/processed_dravidian_v1", "Byte Dravidian corpus duplicate on F:"),
        ("data/processed_v3/pools/v3_64k", "BPE 64k vocabulary tokenized pool (abandoned in favor of 32k)"),
        ("data/processed_v3/pools/v3_48k", "BPE 48k vocabulary partial tokenized pool (abandoned)")
    ]:
        p = FLAKE / pool_rel
        if p.exists():
            p_sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            f_count = sum(1 for f in p.rglob("*") if f.is_file())
            records.append({
                "path": str(p).replace("\\", "/"),
                "bytes": p_sz,
                "sha256": f"directory_{f_count}_files",
                "kind": "pool_dir",
                "source": desc,
                "license": "CC BY 4.0 / CC BY-SA 4.0",
                "used_by": [],
                "rederive": "python scripts/tokenize_corpus.py",
                "date": "2026-10-17"
            })

    # Write training_data_ledger_2026-10-17.jsonl
    print(f"Writing {len(records)} records to {OUT_JSONL}...")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate Checkpoint Manifest for 120M
    print(f"Writing {OUT_MANIFEST}...")
    manifest_md = f"""# Checkpoint 0.9-120M Training Manifest — Parentheses Foundation Model

**Checkpoint Identifiers:**
- **Pretrained Foundation Model:** `checkpoints/parentheses-0.9-120M/step_9536_final.pt`
- **Clean Conversational Model:** `checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt`
- **SFT v4 Checkpoint:** `checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt`

**Model Family:** Parentheses 0.9  
**Parameter Count:** 119,737,344 params (119.7M)  
**Execution Date:** 2026-10-14 through 2026-10-17  
**Architecture:** Recurrent State Space Duality (SSD) / Selective Linear Attention SSM (Mamba-2 / RWKV family; $O(1)$ constant state memory)  
**Supervising Session:** Antigravity / Claude Code Multi-Agent Swarm  
**Training Pipeline:** `train.py` (Sessions 1–5 pretraining + SFT v1–v4)  
**Licensing:** Code: MIT License | Model Weights: Apache-2.0 (Foundation Model, 0 `<esg>` tokens)

---

## 1. Executive Summary & Architecture

Parentheses 0.9-120M is the 120M-scale foundation model of the Parentheses family, designed for high-throughput linear recurrence ($O(T)$ training, $O(1)$ constant-state inference) across 9 languages (English, Kannada, Tamil, Telugu, Malayalam, Tulu, Koya, Kuvi, SE Kolami).

- **Total Parameters:** 119,737,344 (weights-only footprint: 478.9 MB fp32, 239.5 MB bf16/fp16).
- **Configuration:** 24 layers, 12 heads (head_dim 64), `d_model` 768, context `block_size` 4096.
- **Attention/SSM Mechanism:** Chunked associative selective linear attention with closed-form decay gates (alpha_t) and constant-memory recurrent state updates (S_t = alpha_t * S_(t-1) + k_t (x) v_t). Zero quadratic KV-caches!
- **Tokenizer:** BPE `v3_32k` (32,768 vocabulary, byte-fallback, Unicode script preserving).
- **Cumulative Pretraining Tokens:** **1,249,892,352 tokens** (9,536 steps × 131,072 tokens/step).
- **Cumulative SFT Tokens:** **78,643,200 tokens** across SFT v1–v4.
- **Total Ingested Tokens in 120M Lineage:** **1,328,535,552 tokens**.

---

## 2. Ingested Datasets & Curriculum Mix

| Stage | Dataset / Path | Tokens | Epochs | Language / Domain |
|---|---|:---:|:---:|---|
| **Pretraining S1–S5** | `data/processed_v5/train.bin` | 1,075,946,234 | 1.16 | 47.3% Eng, 14% Kan, 10% Tam, 10% Tel, 10% Mal, 8.7% Low-Res (tcy/kff/kxv/nit) |
| **S3 Co-Training** | `data/processed_qa_mix/train.bin` | 30,670,000 | 1.00 | 16 QA lanes: TyDiQA, IndicQA, HotpotQA, QuAC, Hendrycks MATH |
| **S4 Co-Training** | `data/processed_qa_mix_s4/train.bin` | 107,864,113 | 1.00 | 17 QA lanes + Comprehensive Dictionaries (Alar, Olam, Kaikki, WordNet) |
| **S5 Tail** | `data/processed_v5/sft_tail.bin` | 62,555,274 | 1.00 | Grounded dialogue, reasoning, and Dravidian conversation |
| **SFT v2 (Clean)** | `data/processed_sft_120m_v2/train.bin` | 32,267,581 | 1.00 | 0% leakage, genuine multi-step math (Bespoke-Stratos), Aya chat, IndicQA |

---

## 3. Physical Storage & Checksums on Disk

| File Name | File Size | Description | SHA-256 Checksum |
|---|:---:|---|---|
| `data/processed_v5/train.bin` | 2,151,892,468 B | Pretraining corpus mix (1.075B tokens) | `{quick_hash(REPO / "data/processed_v5/train.bin")}` |
| `data/processed_qa_mix_s4/train.bin` | 215,728,226 B | Session 4 17-lane QA & Dictionary mix | `{quick_hash(REPO / "data/processed_qa_mix_s4/train.bin")}` |
| `data/processed_sft_120m_v2/train.bin` | 64,535,162 B | SFT v2 clean assistant-masked train bin | `{quick_hash(REPO / "data/processed_sft_120m_v2/train.bin")}` |
| `data/processed_sft_120m_v2/val.bin` | 1,075,722 B | SFT v2 held-out validation set | `{quick_hash(REPO / "data/processed_sft_120m_v2/val.bin")}` |
| `checkpoints/parentheses-0.9-120M/step_9536_final.pt` | 1,440,784,895 B | Pretrained base foundation model | `{quick_hash(REPO / "checkpoints/parentheses-0.9-120M/step_9536_final.pt")}` |
| `checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt` | 1,440,785,151 B | Clean conversational release checkpoint | `{quick_hash(REPO / "checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt")}` |

---

## 4. Evaluated Benchmarks & Quality Gates

- **Dravidian Bits-Per-Byte (BPB):**
  - Kannada (`kan`): **0.4920** (-1.02% vs baseline)
  - Tamil (`tam`): **0.4688**
  - Telugu (`tel`): **0.5962**
  - Malayalam (`mal`): **0.4150**
  - English (`eng`): **1.4420**
- **Dravidian Script Tag Accuracy:** **94.14%** across all 8 regional languages.
- **Multiple-Choice Reasoning Norm Accuracy:**
  - Code: **0.5167** (vs 0.2500 chance)
  - Reasoning: **0.4833** (vs 0.3333 chance)
  - Math: **0.2857** (vs 0.2500 chance)
- **Joel Practical Test Suite:** 0.0% fake URLs, 0.0% robotic prefaces, 100.0% direct response rate in SFT v2.
"""
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        f.write(manifest_md.strip() + "\n")

    # Generate Machine Index JSON
    print(f"Writing {OUT_INDEX_JSON}...")
    index_json = {
        "model_id": "parentheses-0.9-120M",
        "parameters": 119737344,
        "vocab_size": 32768,
        "context_length": 4096,
        "n_layer": 24,
        "n_head": 12,
        "d_model": 768,
        "head_dim": 64,
        "attention_type": "selective_linear",
        "checkpoints": {
            "base_pretrain": "checkpoints/parentheses-0.9-120M/step_9536_final.pt",
            "sft_v2_clean": "checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt",
            "sft_v4": "checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt"
        },
        "pretraining_tokens": 1249892352,
        "sft_tokens": 78643200,
        "total_tokens": 1328535552,
        "licenses": {
            "code": "MIT",
            "weights": "Apache-2.0"
        },
        "ledger_file": "training_data_ledger_2026-10-17.jsonl",
        "audit_date": "2026-10-17"
    }
    with open(OUT_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(index_json, f, indent=2, ensure_ascii=False)

    # 7. Generate Proposed Deletion List (Phase 3)
    print(f"Writing proposed delete list {OUT_DELETE_LIST}...")
    
    # Calculate sizes for each candidate
    delete_candidates = [
        # (path, description, rederive_command)
        ("E:/Parentheses/parentheses-0.9/data/raw/manual/raw-sourced/project-gutenberg", "Raw Gutenberg English text (10.88 GB) - already baked into Corpus v3/v4/v5", "Downloadable from Project Gutenberg"),
        ("E:/Parentheses/parentheses-0.9/data/raw/manual/raw-sourced/common-pile-arxiv-filtered", "Raw arXiv filtered papers (2.47 GB) - already baked into processed pools", "Downloadable via huggingface EleutherAI/the_pile"),
        ("E:/Parentheses/parentheses-0.9/data/raw/manual/raw-sourced/common-pile-stackexchange-cs-hw-robotics", "Raw StackExchange text (1.06 GB) - already baked into processed pools", "Downloadable via huggingface"),
        ("E:/Parentheses/parentheses-0.9/data/raw/manual/raw-sourced/proof-pile-2-open-web-math", "Raw OpenWebMath text (561 MB) - already baked into processed pools", "Downloadable via huggingface"),
        ("E:/Parentheses/parentheses-0.9/data/processed_all_sources_v1", "Ancient byte-level training bin (5.77 GB) - superseded by BPE Corpus v5", "python scripts/tokenize_corpus.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed_dravidian_v1", "Ancient byte-level Dravidian bin (3.08 GB) - superseded by BPE Corpus v5", "python scripts/tokenize_corpus.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed_multilingual", "Ancient byte-level multilingual bin (569 MB) - obsolete experimental", "python scripts/tokenize_corpus.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed", "Ancient byte-level raw processed bin (511 MB) - obsolete baseline", "python scripts/tokenize_corpus.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed_sft_120m", "Audited & rejected SFT v1 mix (254 MB) - superseded by clean SFT v2", "python scripts/build_sft_120m_v2.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed_sft_120m_v3", "Intermediate SFT v3 mix (116 MB) - superseded by clean SFT v2", "python scripts/build_sft_120m_v2.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed_qa_mix", "Intermediate Session 3 QA mix (61 MB) - superseded by Session 4 QA mix", "python scripts/build_qa_mix_s4.py"),
        ("E:/Parentheses/parentheses-0.9/data/processed_sft_v1", "Legacy 41M SFT mix (30 MB) - obsolete", "python scripts/build_sft_tail.py"),
        ("F:/Lingual_Data/data/processed_dravidian_v1", "Ancient byte-level Dravidian duplicate on F: (3.02 GB) - superseded by BPE", "python scripts/tokenize_corpus.py"),
        ("F:/Lingual_Data/data/processed_v3/pools/v3_64k", "Abandoned BPE 64k tokenized pool on F: (~2.0 GB) - re-derivable", "python scripts/tokenize_corpus.py"),
        ("F:/Lingual_Data/data/processed_v3/pools/v3_48k", "Abandoned BPE 48k partial pool on F: (~1.0 GB) - obsolete", "python scripts/tokenize_corpus.py")
    ]

    total_e_bytes = 0
    total_f_bytes = 0
    candidate_rows = []

    for path_str, desc, rederive in delete_candidates:
        p = Path(path_str)
        if not p.exists():
            continue
        if p.is_file():
            sz = p.stat().st_size
        else:
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        
        is_f = path_str.startswith("F:")
        if is_f:
            total_f_bytes += sz
        else:
            total_e_bytes += sz
        
        sz_mb = sz / (1024 * 1024)
        sz_gb = sz / (1024 * 1024 * 1024)
        drive_tag = "F:" if is_f else "E:"
        candidate_rows.append({
            "path": path_str,
            "drive": drive_tag,
            "bytes": sz,
            "size_str": f"{sz_gb:.2f} GB" if sz_gb >= 1.0 else f"{sz_mb:.1f} MB",
            "desc": desc,
            "rederive": rederive
        })

    delete_md = f"""# Proposed Data Deletion & Pruning Ledger — 2026-10-17

**Date:** 2026-10-17  
**Authority:** Awaiting Joel Anthony's Approval (Phase 3 Proposal)  
**Status:** **PROPOSAL ONLY — ZERO FILES DELETED YET**  
**Prerequisite:** All files below are permanently cataloged in [`training_data_ledger_2026-10-17.jsonl`](training_data_ledger_2026-10-17.jsonl) and [`checkpoint_0.9-120M_manifest.md`](checkpoint_0.9-120M_manifest.md).

---

## 1. Executive Summary of Reclaimable Space

| Target Drive | Items / Folders | Reclaimable Space | Projected Free Space After Deletion |
|---|:---:|:---:|:---:|
| **Drive `E:` (Fast NVMe)** | 12 candidate folders | **{total_e_bytes / (1024**3):.2f} GB** | **~{61.2 + total_e_bytes / (1024**3):.1f} GB** (currently 61.2 GB) |
| **Drive `F:` (External HDD)**| 3 candidate folders | **{total_f_bytes / (1024**3):.2f} GB** | **~{72.5 + total_f_bytes / (1024**3):.1f} GB** (currently 72.5 GB) |
| **TOTAL COMBINED** | **15 candidate folders** | **{(total_e_bytes + total_f_bytes) / (1024**3):.2f} GB** | **High-capacity space freed for new training downloads** |

---

## 2. Itemized Candidate Deletion Table

| Drive | Path | Size | Description & Safety Justification | Rederive / Recovery Recipe |
|:---:|---|:---:|---|---|
"""
    for r in candidate_rows:
        delete_md += f"| `{r['drive']}` | `{r['path']}` | **{r['size_str']}** | {r['desc']} | `{r['rederive']}` |\n"

    delete_md += f"""
---

## 3. Strict Protected Invariants (KEEP LIST — 100% UNTOUCHED)

Under strict project governance rules, the following assets are **NEVER deleted**:
1. **Model Checkpoints:**
   - Pretrained Base: `checkpoints/parentheses-0.9-120M/step_9536_final.pt`
   - Golden Conversational: `checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt`
   - SFT v4: `checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt`
   - 41M Foundations: `checkpoints/parentheses-0.9-41M/step_8082_final.pt`, `parentheses-0.9-41M-v4/step_16082_final.pt`
   - NPU Models: `checkpoints/npu_byte_filter/*`
   - Hugging Face Release Candidate: `release/parentheses-0.9-120M-candidate/*`
2. **Release Training & Validation Corpora:**
   - Active Pretraining Mix: `data/processed_v5/train.bin`, `train_boundaries.npy`, `sft_tail.bin`
   - Active S4 QA Mix: `data/processed_qa_mix_s4/train.bin`
   - Golden Clean SFT v2: `data/processed_sft_120m_v2/*` (all 32.2M tokens, 0% leakage)
   - All 9 Held-Out Validation Sets: `data/processed_v3/val/` and `data/processed_v4/val/`
   - Active BPE 32k Token Pools: `F:\\Lingual_Data\\data\\processed_v3\\pools\\v3_32k\\` and `processed_v4\\pools\\v4_32k\\`
3. **Tokenizers:** `data/tokenizer/` (`v3_32k`, `v3_64k`, `byte_tokenizer`).
4. **Governance, Specs & Licenses:**
   - Personality specs: `data/raw/manual/personality-spec/`, `data/raw/manual/LLM instructions/`
   - Upstream licenses: All `SOURCE.txt` and `ATTRIBUTION.md` files
   - Ledgers: All records under `F:\\Lingual_Data\\Trained-On\\` and `F:\\Lingual_Data\\LEDGER\\`
   - Obsidian Vault: `c:\\Users\\Joel Anthony\\Documents\\Phile\\Heafod\\Heafod\\Parentheses\\`
"""

    with open(OUT_DELETE_LIST, "w", encoding="utf-8") as f:
        f.write(delete_md.strip() + "\n")

    print(f"\nCompleted successfully! Generated:")
    print(f"  - {OUT_JSONL}")
    print(f"  - {OUT_MANIFEST}")
    print(f"  - {OUT_INDEX_JSON}")
    print(f"  - {OUT_DELETE_LIST}")
    print(f"Reclaimable space: {total_e_bytes / (1024**3):.2f} GB on E:, {total_f_bytes / (1024**3):.2f} GB on F: (Total: {(total_e_bytes + total_f_bytes) / (1024**3):.2f} GB)")

if __name__ == "__main__":
    main()
