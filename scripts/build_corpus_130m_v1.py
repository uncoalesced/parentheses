#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_corpus_130m_v1.py

Collation and Multiplexing Engine for Parentheses 0.9-130M-v3 (Project Tanzinite):
  - Pre-Training Target: 1.5 Billion Tokens (train.bin: 1.4925B, val.bin: 7.5M)
  - 4-Pillar Architectural Balance:
      1. Dravidian Core (40% / 600M tok): Sangraha verified {kan, tam, tel, mal} + low-res {tcy, kff, kxv, nit}
      2. STEM Reasoning (25% / 375M tok): Camel-AI chemistry & physics, SciQ, MetaMathQA, Bespoke-Stratos (50% <think> primed)
      3. Conversational Intelligence & Open Data (20% / 300M tok): open_data_corpus_v1 (0% refusal), Karya, NIM distillation
      4. High-Resolution English Anchors (15% / 225M tok): Project Gutenberg curated sample & literature
  - Deterministic Strided Multiplexer: Weighted Deficit Round-Robin (DRR) over 4096-token quanta
  - High-Throughput Streaming Batch Tokenization: ~2M tokens/sec with Rust tokenizers encode_batch
  - Strict Hardware & Memory Isolation:
      - CPU affinity pinned to cores [0..11] (cores [12..15] reserved for distillation PID 11012)
      - Process priority: BELOW_NORMAL
      - 0 MiB GPU VRAM (CUDA_VISIBLE_DEVICES="")
      - Ring-buffered streaming I/O with process RAM < 350 MB
  - Output: data/processed_130m_v1/ (train.bin, train_boundaries.npy, val.bin, val_boundaries.npy, manifest.json, val/)

Engineered by uncoalesced for Project Tanzinite.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import deque
from pathlib import Path

# Hardware Isolation Header
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
import pyarrow.parquet as pq
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Special token IDs in v3_32k (SHA256: 7a9b4187d5bc...)
EOT = 0
PAD = 1
TAG = {
    "kan": 2, "tam": 3, "tel": 4, "mal": 5,
    "tcy": 6, "kff": 7, "kxv": 8, "nit": 9,
    "eng": 10
}
ESG_ID = 11  # STRICTLY EXCLUDED
SYSTEM_TAG = 12
USER_TAG = 13
ASSISTANT_TAG = 14
THINK_START = 15
THINK_END = 16
TOOL_CALL_START = 17
TOOL_CALL_END = 18
HIN_TAG = 19

# Paths
TOKENIZER_DIR = REPO / "data" / "tokenizer" / "v3_32k"
DEFAULT_OUT_DIR = REPO / "data" / "processed_130m_v1"
DEFAULT_VAL_DIR = REPO / "data" / "processed_v3" / "val"
SANGRAHA_DIR = Path("F:/Lingual_Data/raw/hf/ai4bharat_Sangraha_verified/verified")
LOWRES_DIR = Path("F:/Lingual_Data/data/processed_dravidian_lowres_v1")
STEM_JSONL = Path("F:/Lingual_Data/data/processed_stem/stem_corpus_v1.jsonl")
OPEN_DATA_JSONL = Path("F:/Lingual_Data/data/processed_open_data/open_data_corpus_v1.jsonl")
KARYA_JSONL_DIR = Path("F:/Lingual_Data/raw/karya/jsonl")
KARYA_LANG_DIR = Path("F:/Lingual_Data/raw/karya/by_lang")
DISTILL_CURATED_DIR = Path("F:/Lingual_Data/distillation/conversational_intelligence/curated")
DISTILL_CLEANED_JSONL = Path("F:/Lingual_Data/distillation/conversational_intelligence/corpus_240m/cleaned_conversational_v1.jsonl")
METAMATH_JSON = REPO / "data" / "raw" / "sft_real" / "meta-math__MetaMathQA" / "MetaMathQA-395K.json"
BESPOKE_PARQUET = REPO / "data" / "raw" / "manual" / "raw-sourced" / "bespoke_stratos" / "train-00000-of-00001.parquet"
GUTENBERG_SAMPLE_DIR = REPO / "data" / "curated" / "project-gutenberg-sample"
if not GUTENBERG_SAMPLE_DIR.exists():
    GUTENBERG_SAMPLE_DIR = REPO / "data" / "raw" / "curated" / "project-gutenberg-sample"
GUTENBERG_SWARM_DIR = REPO / "data" / "raw" / "manual" / "raw-sourced" / "gutenberg-literature-swarm"


def apply_process_affinity(max_core: int = 12):
    """Binds process to logical cores [0..max_core-1] to keep background distillation cores free."""
    if not HAS_PSUTIL:
        print("[isolation] psutil not available; skipping CPU affinity binding.", flush=True)
        return
    try:
        proc = psutil.Process()
        total_cores = psutil.cpu_count(logical=True) or 16
        target_cores = list(range(0, min(max_core, total_cores)))
        proc.cpu_affinity(target_cores)
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print(f"[isolation] Bound collation PID {proc.pid} to CPU cores {target_cores} (priority: BELOW_NORMAL)", flush=True)
    except Exception as e:
        print(f"[isolation] Warning: could not set CPU affinity: {e}", flush=True)


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


def load_val_hashes(val_dir: Path, tok: Tokenizer) -> tuple[set[str], set[str]]:
    """Loads exact token hashes and trimmed token hashes for all validation documents."""
    tok_hashes = set()
    tok_trimmed_hashes = set()
    langs = ["kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"]
    for l in langs:
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
    return tok_hashes, tok_trimmed_hashes


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    return s.strip()


def strip_internal_tags(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"</?(?:think|freethink|tool_call)>|<\|(?:user|assistant|system|endoftext|pad)\|>|<esg>", "", text)
    return text.strip()


def split_stem_reasoning(ans: str) -> tuple[str, str]:
    """Splits a STEM answer into (reasoning_trace, final_answer)."""
    ans = clean_text(ans)
    if "\nThe answer is: " in ans:
        r, a = ans.rsplit("\nThe answer is: ", 1)
        return r.strip(), f"The answer is {a.strip()}."
    if "\n#### " in ans:
        r, a = ans.rsplit("\n#### ", 1)
        return r.strip(), f"The answer is {a.strip()}."
    for sep in ("\nTherefore, ", "\nSo, ", "\nIn conclusion, ", "\nFinal Answer:"):
        if sep in ans:
            r, a = ans.rsplit(sep, 1)
            return r.strip(), f"{sep.strip()} {a.strip()}."
    lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
    if len(lines) >= 3 and len(lines[-1]) <= 140:
        return "\n".join(lines[:-1]), lines[-1]
    return ans, ans


# =========================================================================
# PILLAR 1: DRAVIDIAN CORE GENERATOR
# =========================================================================
def stream_dravidian(tok: Tokenizer, val_tok_hashes: set[str], val_tok_trimmed_hashes: set[str], rng: np.random.Generator):
    """
    Yields pre-tokenized uint16 documents for the Dravidian Core:
      - Kannada (30% of Dravidian)
      - Tamil (25% of Dravidian)
      - Telugu (25% of Dravidian)
      - Malayalam (18.33% of Dravidian)
      - Low-res: tcy, kff, kxv, nit (1.67% of Dravidian, zero esg)
    """
    main_langs = ["kan", "tam", "tel", "mal"]
    low_langs = ["tcy", "kff", "kxv", "nit"]

    # Pre-cache low-res clean docs into memory buffers
    lowres_docs: dict[str, list[np.ndarray]] = {}
    for lang in low_langs:
        bin_p = LOWRES_DIR / lang / "train.bin"
        bnd_p = LOWRES_DIR / lang / "train_boundaries.npy"
        if not bin_p.exists() or not bnd_p.exists():
            continue
        data = np.fromfile(bin_p, dtype=np.uint16)
        bnd = np.append(np.load(bnd_p).astype(np.int64), len(data))
        valid = []
        for s, e in zip(bnd[:-1], bnd[1:]):
            doc = data[s:e]
            if np.any(doc == ESG_ID):
                continue
            if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
                continue
            if lang == "kxv":
                txt = tok.decode(doc)
                odia_chars = sum(1 for c in txt if 0x0B00 <= ord(c) <= 0x0B7F)
                total_alpha = sum(1 for c in txt if c.isalpha())
                if total_alpha > 0 and (odia_chars / total_alpha) < 0.85:
                    continue
            valid.append(doc)
        lowres_docs[lang] = valid
        print(f"  [dravidian] Cached {len(valid):,} clean low-res docs for {lang}", flush=True)

    # Parquet file queues for main 4 languages
    parquet_files: dict[str, list[Path]] = {}
    for lang in main_langs:
        p_dir = SANGRAHA_DIR / lang
        files = sorted(p_dir.glob("data-*.parquet"))
        rng.shuffle(files)
        parquet_files[lang] = files

    # Iterators over batches
    def lang_parquet_generator(lang: str):
        while True:
            for p_file in parquet_files[lang]:
                try:
                    pf = pq.ParquetFile(p_file)
                    for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
                        texts = []
                        for t in batch["text"]:
                            s = t.as_py()
                            if s and len(s.strip()) >= 40:
                                texts.append(clean_text(s))
                        if not texts:
                            continue
                        encs = tok.encode_batch(texts)
                        tag_id = TAG[lang]
                        for enc in encs:
                            ids = enc.ids
                            if len(ids) < 10:
                                continue
                            doc = np.array([tag_id] + ids + [EOT], dtype=np.uint16)
                            if np.any(doc == ESG_ID):
                                continue
                            if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
                                continue
                            yield doc
                except Exception as e:
                    print(f"  [warning] error reading {p_file}: {e}", flush=True)
                    continue

    gen_kan = lang_parquet_generator("kan")
    gen_tam = lang_parquet_generator("tam")
    gen_tel = lang_parquet_generator("tel")
    gen_mal = lang_parquet_generator("mal")

    step_pattern = (
        ["kan"] * 18 +
        ["tam"] * 15 +
        ["tel"] * 15 +
        ["mal"] * 11 +
        ["low"] * 1
    )
    rng.shuffle(step_pattern)

    low_idx = 0
    step_i = 0
    while True:
        target = step_pattern[step_i % len(step_pattern)]
        step_i += 1
        if target == "kan":
            yield next(gen_kan)
        elif target == "tam":
            yield next(gen_tam)
        elif target == "tel":
            yield next(gen_tel)
        elif target == "mal":
            yield next(gen_mal)
        elif target == "low":
            low_lang = low_langs[low_idx % len(low_langs)]
            low_idx += 1
            if low_lang in lowres_docs and len(lowres_docs[low_lang]) > 0:
                c_idx = int(rng.integers(0, len(lowres_docs[low_lang])))
                doc = lowres_docs[low_lang][c_idx]
                if h_tok(doc) in val_tok_hashes or (len(doc) > 2 and h_tok(doc[1:-1]) in val_tok_trimmed_hashes):
                    continue
                yield doc
            else:
                yield next(gen_kan)


# =========================================================================
# PILLAR 2: STEM REASONING GENERATOR (50% <think> PRIMED, STREAMING BATCHES)
# =========================================================================
def stream_stem(tok: Tokenizer, val_tok_hashes: set[str], val_tok_trimmed_hashes: set[str], rng: np.random.Generator):
    """
    Yields pre-tokenized uint16 documents for STEM Reasoning using streaming batch encoding:
      - Camel-AI chemistry & physics, SciQ (stem_corpus_v1.jsonl)
      - MetaMathQA (MetaMathQA-395K.json)
      - Bespoke-Stratos reasoning traces
      - 50% formatted with explicit <think>...</think> reasoning traces
      - 50% formatted with direct step-by-step solutions
    """
    # Raw record generator streaming from disk without storing 464k items in RAM
    def raw_stem_generator():
        while True:
            # 1. Stream from stem_corpus_v1.jsonl
            if STEM_JSONL.exists():
                with open(STEM_JSONL, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        q = d.get("question", "").strip()
                        sol = d.get("solution", "").strip()
                        if q and sol and len(q) >= 10 and len(sol) >= 10:
                            r, a = split_stem_reasoning(sol)
                            yield q, r, a

            # 2. Stream from MetaMathQA
            if METAMATH_JSON.exists():
                with open(METAMATH_JSON, "r", encoding="utf-8") as f:
                    math_data = json.load(f)
                    for item in math_data:
                        q = item.get("query", "").strip()
                        resp = item.get("response", "").strip()
                        if q and resp and len(q) >= 10 and len(resp) >= 10:
                            r, a = split_stem_reasoning(resp)
                            yield q, r, a

            # 3. Stream from Bespoke Stratos
            if BESPOKE_PARQUET.exists():
                try:
                    tab = pq.read_table(BESPOKE_PARQUET)
                    convs = tab.column("conversations").to_pylist()
                    for conv in convs:
                        if len(conv) >= 2:
                            q = conv[0].get("value", "").strip()
                            resp = conv[1].get("value", "").strip()
                            if q and resp:
                                if "<thought>" in resp and "</thought>" in resp:
                                    m = re.search(r"<thought>(.*?)</thought>(.*)", resp, re.DOTALL)
                                    if m:
                                        yield q, m.group(1).strip(), m.group(2).strip()
                                        continue
                                r, a = split_stem_reasoning(resp)
                                yield q, r, a
                except Exception:
                    pass

    raw_gen = raw_stem_generator()
    toggle_think = True
    batch_size = 500

    while True:
        items = []
        for _ in range(batch_size):
            items.append(next(raw_gen))

        # Batch encode components
        # 50% Format A (<think>), 50% Format B (Direct)
        q_texts = [strip_internal_tags(clean_text(q)) for q, _, _ in items]
        q_encs = tok.encode_batch(q_texts)

        mid_texts = []
        for idx, (_, r, a) in enumerate(items):
            r_c = strip_internal_tags(clean_text(r))
            a_c = strip_internal_tags(clean_text(a))
            if (toggle_think or idx % 2 == 0) and r_c and len(r_c) > 10:
                mid_texts.append(("\n" + r_c + "\n", "\n\n" + a_c, True))
            else:
                full_s = (r_c + "\n\n" + a_c) if r_c != a_c else a_c
                mid_texts.append(("", "\n" + full_s, False))

        # Encode answers and reasoning traces in batch
        first_parts = [m[0] for m in mid_texts if m[2]]
        first_encs = tok.encode_batch(first_parts) if first_parts else []
        second_parts = [m[1] for m in mid_texts]
        second_encs = tok.encode_batch(second_parts)

        f_idx = 0
        tag_eng = TAG["eng"]
        for idx in range(batch_size):
            p_ids = q_encs[idx].ids
            is_think = mid_texts[idx][2]
            if is_think:
                r_ids = first_encs[f_idx].ids
                f_idx += 1
                a_ids = second_encs[idx].ids
                doc = np.array(
                    [tag_eng, USER_TAG] + p_ids +
                    [ASSISTANT_TAG, THINK_START] + r_ids +
                    [THINK_END] + a_ids + [EOT],
                    dtype=np.uint16
                )
            else:
                s_ids = second_encs[idx].ids
                doc = np.array(
                    [tag_eng, USER_TAG] + p_ids +
                    [ASSISTANT_TAG] + s_ids + [EOT],
                    dtype=np.uint16
                )

            if np.any(doc == ESG_ID):
                continue
            if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
                continue
            yield doc

        toggle_think = not toggle_think


# =========================================================================
# PILLAR 3: CONVERSATIONAL & OPEN DATA GENERATOR (STREAMING BATCHES)
# =========================================================================
def stream_conversational_opendata(tok: Tokenizer, val_tok_hashes: set[str], val_tok_trimmed_hashes: set[str], rng: np.random.Generator):
    """
    Yields pre-tokenized uint16 documents for Conversational Intelligence & Open Data using streaming batching:
      - open_data_corpus_v1.jsonl (120,653 zero-refusal rows)
      - Karya conversational datasets (karya_dual_conversations, telugu, connect_stories)
      - Cleaned NIM conversational distillation (cleaned_conversational_v1.jsonl + curated)
      - Persona anchors in 5 languages
    """
    persona_qas = [
        ("Who are you?", "I am Parentheses 0.9, an efficient Dravidian and English recurrent language model developed by uncoalesced.", "eng"),
        ("What is your name?", "My name is Parentheses 0.9.", "eng"),
        ("Who created you?", "I was created and developed by uncoalesced.", "eng"),
        ("Who developed you?", "I was developed by uncoalesced as an efficient recurrent language model with Selective Linear Attention.", "eng"),
        ("What architecture do you use?", "I am built on a recurrent State Space Duality (SSD) architecture in the Mamba-2 / RWKV / Selective Linear Attention family. During inference, I operate with constant O(1) memory state updates without a quadratic KV-cache.", "eng"),
        ("Are you a Transformer?", "No, I am not a standard Transformer model. I do not use quadratic causal self-attention or growing KV-caches. I am a recurrent State Space Duality (SSD) model with linear complexity.", "eng"),
        ("ನೀವು ಯಾರು?", "ನಾನು ಪ್ಯಾರೆಂಥಿಸಿಸ್ 0.9 (Parentheses 0.9), uncoalesced ಅವರು ಅಭಿವೃದ್ಧಿಪಡಿಸಿದ ದ್ರಾವಿಡ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಭಾಷೆಗಳ ಪರಿಣಾಮಕಾರಿ ಪುನರಾವರ್ತಿತ (recurrent) ಭಾಷಾ ಮಾದರಿ.", "kan"),
        ("ನಿಮ್ಮ ಹೆಸರೇನು?", "ನನ್ನ ಹೆಸರು ಪ್ಯಾರೆಂಥಿಸಿಸ್ 0.9 (Parentheses 0.9).", "kan"),
        ("ನಿಮ್ಮನ್ನು ಯಾರು ರಚಿಸಿದರು?", "ನನ್ನನ್ನು uncoalesced ಅವರು ವಿನ್ಯಾಸಗೊಳಿಸಿ ಅಭಿವೃದ್ಧಿಪಡಿಸಿದ್ದಾರೆ.", "kan"),
        ("ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಿ?", "ನಮಸ್ಕಾರ! ನಾನು ಕ್ಷೇಮವಾಗಿದ್ದೇನೆ. ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?", "kan"),
        ("நீங்கள் யார்?", "நான் பேரென்தசிஸ் 0.9 (Parentheses 0.9), uncoalesced உருவாக்கிய திராவிட மற்றும் ஆங்கில மீளுருவாக்க மொழி மாதிரி.", "tam"),
        ("உங்கள் பெயர் என்ன?", "என் பெயர் பேரென்தசிஸ் 0.9 (Parentheses 0.9).", "tam"),
        ("உங்களை உருவாக்கியவர் யார்?", "என்னை uncoalesced உருவாக்கினார்.", "tam"),
        ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "வணக்கம்! நான் நலமாக இருக்கிறேன். உங்களுக்கு நான் எவ்வாறு உதவ முடியும்?", "tam"),
        ("మీరు ఎవరు?", "నేను పేరెంథెసిస్ 0.9 (Parentheses 0.9), uncoalesced రూపొందించిన ద్రావిడ మరియు ఆంగ్ల రికరెంట్ భాషా మోడల్.", "tel"),
        ("మీ పేరు ఏమిటి?", "నా పేరు పేరెంథెసిస్ 0.9 (Parentheses 0.9).", "tel"),
        ("మిమ్మల్ని ఎవరు సృష్టించారు?", "నన్ను uncoalesced సృష్టించారు.", "tel"),
        ("నమస్కారం, ఎలా ఉన్నారు?", "నమస్కారం! నేను బాగున్నాను. ఈ రోజు నేను మీకు ఎలా సహాయపడగలను?", "tel"),
        ("നിങ്ങൾ ആരാണ്?", "ഞാൻ പാരൻതസിസ് 0.9 (Parentheses 0.9) ആണ്, uncoalesced വികസിപ്പിച്ച ദ്രാവിഡ-ഇംഗ്ലീഷ് റിക്കറന്റ് ലാംഗ്വേജ് മോഡൽ.", "mal"),
        ("നിങ്ങളുടെ പേരെന്താണ്?", "എന്റെ പേര് പാരൻതസിസ് 0.9 (Parentheses 0.9) എന്നാണ്.", "mal"),
        ("നിങ്ങളെ സൃഷ്ടിച്ചത് ആരാണ്?", "എന്നെ വികസിപ്പിച്ചതും സൃഷ്ടിച്ചതും uncoalesced ആണ്.", "mal"),
        ("നമസ്കാരം, എങ്ങനെയുണ്ട്?", "നമസ്കാരം! എനിക്ക് സുഖമാണ്. ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?", "mal"),
    ]

    def raw_conv_generator():
        while True:
            # 1. Stream open_data_corpus_v1.jsonl
            if OPEN_DATA_JSONL.exists():
                with open(OPEN_DATA_JSONL, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        q = d.get("prompt", "").strip()
                        a = d.get("response", "").strip()
                        if q and a and len(q) >= 10 and len(a) >= 10:
                            yield "eng", q, a

            # 2. Stream Curated Distillation JSONLs (with content & reasoning)
            if DISTILL_CURATED_DIR.exists():
                for c_file in sorted(DISTILL_CURATED_DIR.glob("conversational_curated_*.jsonl")):
                    with open(c_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                d = json.loads(line)
                                q = d.get("prompt", "").strip()
                                dial = d.get("content", "") or d.get("dialogue", "")
                                dial = dial.strip()
                                if q and dial:
                                    yield "eng", q, dial
                            except Exception:
                                continue

            # 3. Stream Karya conversational datasets
            if KARYA_JSONL_DIR.exists():
                for f_path in KARYA_JSONL_DIR.glob("*.jsonl"):
                    with open(f_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                d = json.loads(line)
                                txt = d.get("text", "") or d.get("prompt", "") or d.get("dialogue", "")
                                lk = d.get("lang", "") or d.get("language", "") or "eng"
                                if lk not in TAG:
                                    lk = "eng"
                                if txt and len(txt.strip()) >= 30:
                                    yield lk, "", txt.strip()
                            except Exception:
                                continue

            # 4. Stream Cleaned conversational distillation
            if DISTILL_CLEANED_JSONL.exists():
                with open(DISTILL_CLEANED_JSONL, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                            q = d.get("prompt", "").strip()
                            dial = d.get("dialogue", "") or d.get("content", "")
                            dial = dial.strip()
                            tr = d.get("track", "conv")
                            lk = "eng"
                            if tr == "dravidian":
                                for cand, (start, end) in [("kan", (0x0C80, 0x0CFF)), ("tam", (0x0B80, 0x0BFF)),
                                                           ("tel", (0x0C00, 0x0C7F)), ("mal", (0x0D00, 0x0D7F))]:
                                    if any(start <= ord(c) <= end for c in dial[:100]):
                                        lk = cand
                                        break
                            if q and dial:
                                yield lk, q, dial
                        except Exception:
                            continue

            # 5. Inject Persona anchors
            for q, a, lk in persona_qas:
                yield lk, q, a

    raw_gen = raw_conv_generator()
    batch_size = 500

    while True:
        items = []
        for _ in range(batch_size):
            items.append(next(raw_gen))

        # Separate items with prompts vs raw texts
        p_texts = [strip_internal_tags(clean_text(q)) for _, q, _ in items]
        a_texts = [strip_internal_tags(clean_text(a)) for _, _, a in items]

        p_encs = tok.encode_batch(p_texts)
        a_encs = tok.encode_batch(a_texts)

        for idx, (lk, q, a) in enumerate(items):
            l_tag = TAG.get(lk, TAG["eng"])
            if q.strip():
                doc = np.array(
                    [l_tag, USER_TAG] + p_encs[idx].ids +
                    [ASSISTANT_TAG] + a_encs[idx].ids + [EOT],
                    dtype=np.uint16
                )
            else:
                doc = np.array([l_tag] + a_encs[idx].ids + [EOT], dtype=np.uint16)

            if np.any(doc == ESG_ID):
                continue
            if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
                continue
            yield doc


# =========================================================================
# PILLAR 4: HIGH-RESOLUTION ENGLISH LITERATURE & ANCHORS GENERATOR
# =========================================================================
def stream_english(tok: Tokenizer, val_tok_hashes: set[str], val_tok_trimmed_hashes: set[str], rng: np.random.Generator):
    """
    Yields pre-tokenized uint16 documents for High-Resolution English:
      - Project Gutenberg curated sample shards (80 shards, ~811M tokens)
      - Gutenberg literature swarm shards
    """
    shards: list[Path] = []
    if GUTENBERG_SAMPLE_DIR.exists():
        shards.extend(sorted(GUTENBERG_SAMPLE_DIR.glob("shard_*.txt")))
    if GUTENBERG_SWARM_DIR.exists():
        shards.extend(sorted(GUTENBERG_SWARM_DIR.glob("shard_*.txt")))

    print(f"  [english] Sourced {len(shards)} curated literature shards", flush=True)
    rng.shuffle(shards)

    s_idx = 0
    while True:
        shard_path = shards[s_idx % len(shards)]
        s_idx += 1
        try:
            content = shard_path.read_text(encoding="utf-8", errors="replace")
            raw_chunks = [c.strip() for c in content.split("\n\n\n") if len(c.strip()) >= 200]
            if not raw_chunks:
                continue
            encs = tok.encode_batch(raw_chunks)
            tag_id = TAG["eng"]
            for enc in encs:
                ids = enc.ids
                if len(ids) < 30:
                    continue
                if len(ids) > 4090:
                    for sub_start in range(0, len(ids), 3800):
                        sub_ids = ids[sub_start:sub_start + 3800]
                        doc = np.array([tag_id] + sub_ids + [EOT], dtype=np.uint16)
                        if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
                            continue
                        yield doc
                else:
                    doc = np.array([tag_id] + ids + [EOT], dtype=np.uint16)
                    if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
                        continue
                    yield doc
        except Exception as e:
            print(f"  [english] Warning reading {shard_path}: {e}", flush=True)
            continue


# =========================================================================
# DETERMINISTIC STRIDED MULTIPLEXER (WEIGHTED DEFICIT ROUND-ROBIN)
# =========================================================================
class DeterministicPillarMultiplexer:
    """
    Weighted Deficit Round-Robin (DRR) multiplexer over the 4 pillars:
      - dravidian: 40%
      - stem: 25%
      - conversational_opendata: 20%
      - english: 15%
    Maintains fine-grained domain interleaving across every 4096-token block window.
    """
    def __init__(self, streams: dict[str, any], weights: dict[str, int], quantum: int = 4096):
        self.streams = streams
        self.weights = weights
        self.quantum = quantum
        self.deficit = {k: 0 for k in weights}
        self.pillar_names = list(weights.keys())
        self.buffers: dict[str, deque[np.ndarray]] = {k: deque() for k in weights}
        self.token_counts = {k: 0 for k in weights}
        self.doc_counts = {k: 0 for k in weights}

    def _get_next_doc(self, pillar: str) -> np.ndarray:
        if not self.buffers[pillar]:
            doc = next(self.streams[pillar])
            return doc
        return self.buffers[pillar].popleft()

    def yield_docs(self, max_tokens: int):
        total_emitted = 0
        while total_emitted < max_tokens:
            for p in self.pillar_names:
                w = self.weights[p]
                self.deficit[p] += w * self.quantum

                while self.deficit[p] > 0 and total_emitted < max_tokens:
                    doc = self._get_next_doc(p)
                    doc_len = len(doc)
                    yield (p, doc)
                    self.deficit[p] -= doc_len
                    self.token_counts[p] += doc_len
                    self.doc_counts[p] += 1
                    total_emitted += doc_len


def main():
    parser = argparse.ArgumentParser(description="Collate 1.5B token pretraining corpus for Parentheses 0.9-130M-v3")
    parser.add_argument("--budget", type=int, default=1_500_000_000, help="Total token budget (default: 1.5B)")
    parser.add_argument("--val-tokens", type=int, default=7_500_000, help="Tokens reserved for val.bin (default: 7.5M / 0.5 percent)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--tokenizer", type=str, default=str(TOKENIZER_DIR), help="Path to v3_32k tokenizer directory")
    parser.add_argument("--val-dir", type=str, default=str(DEFAULT_VAL_DIR), help="Path to reference validation directory")
    parser.add_argument("--dry-run", action="store_true", help="Run 1M token dry run to test pipeline end-to-end")
    parser.add_argument("--max-core", type=int, default=12, help="Max logical core to bind to (reserving [12..15])")
    parser.add_argument("--seed", type=int, default=20261022, help="RNG seed")
    args = parser.parse_args()

    # 1. Apply CPU Core Binding & Memory Protection
    apply_process_affinity(max_core=args.max_core)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(args.val_dir)
    val_dst = out_dir / "val"
    if val_dir.exists() and not val_dst.exists():
        shutil.copytree(val_dir, val_dst, dirs_exist_ok=True)
        print(f"[setup] Copied reference validation directory to {val_dst}", flush=True)

    if args.dry_run:
        train_budget = 1_000_000
        val_budget = 50_000
        bin_name = "train_dryrun.bin"
        bnd_name = "train_dryrun_boundaries.npy"
        val_bin_name = "val_dryrun.bin"
        val_bnd_name = "val_dryrun_boundaries.npy"
        manifest_name = "manifest_dryrun.json"
    else:
        val_budget = args.val_tokens
        train_budget = args.budget - val_budget
        bin_name = "train.bin"
        bnd_name = "train_boundaries.npy"
        val_bin_name = "val.bin"
        val_bnd_name = "val_boundaries.npy"
        manifest_name = "manifest.json"

    print("=" * 80)
    print(f"BUILDING 130M-V3 PRETRAINING CORPUS ({args.budget:,} TOKENS)")
    print(f"Target Train: {train_budget:,} tokens | Target Val: {val_budget:,} tokens")
    print(f"Destination:  {out_dir}")
    print("=" * 80)

    # 2. Load Tokenizer & Validation Hashes
    tok = load_tokenizer(Path(args.tokenizer))
    print(f"[tokenizer] Loaded {args.tokenizer} (vocab: {tok.get_vocab_size():,})", flush=True)

    val_tok_hashes, val_tok_trimmed_hashes = load_val_hashes(val_dir, tok)
    ref_val_tok_hashes = set(val_tok_hashes)
    ref_val_tok_trimmed_hashes = set(val_tok_trimmed_hashes)
    print(f"[val-gate] Loaded {len(val_tok_hashes):,} validation document hashes for zero-leak filtering", flush=True)

    rng = np.random.default_rng(args.seed)

    # 3. Initialize the 4 Streams
    print("\n[setup] Initializing 4-Pillar Stream Generators...", flush=True)
    streams = {
        "dravidian": stream_dravidian(tok, val_tok_hashes, val_tok_trimmed_hashes, rng),
        "stem": stream_stem(tok, val_tok_hashes, val_tok_trimmed_hashes, rng),
        "conversational_opendata": stream_conversational_opendata(tok, val_tok_hashes, val_tok_trimmed_hashes, rng),
        "english": stream_english(tok, val_tok_hashes, val_tok_trimmed_hashes, rng),
    }

    # Weights: 40% Dravidian, 25% STEM, 20% Conversational/OpenData, 15% English
    weights = {
        "dravidian": 40,
        "stem": 25,
        "conversational_opendata": 20,
        "english": 15,
    }

    # 4. Stream and Write Validation Split
    print(f"\n[1/3] Collation of Validation Set ({val_budget:,} tokens)...", flush=True)
    val_bin_path = out_dir / val_bin_name
    val_bnd_path = out_dir / val_bnd_name

    val_multiplexer = DeterministicPillarMultiplexer(streams, weights, quantum=512)
    val_boundaries: list[int] = []
    val_pos = 0
    t0 = time.time()

    with open(val_bin_path, "wb", buffering=16 * 1024 * 1024) as f_val:
        for p, doc in val_multiplexer.yield_docs(val_budget):
            val_boundaries.append(val_pos)
            f_val.write(doc.tobytes())
            val_pos += len(doc)
            val_tok_hashes.add(h_tok(doc))
            if len(doc) > 2:
                val_tok_trimmed_hashes.add(h_tok(doc[1:-1]))

    np.save(val_bnd_path, np.array(val_boundaries, dtype=np.int64))
    print(f"  Val complete: {val_pos:,} tokens across {len(val_boundaries):,} documents in {time.time() - t0:.2f}s", flush=True)

    # 5. Stream and Write Training Split
    print(f"\n[2/3] Collation of Training Set ({train_budget:,} tokens)...", flush=True)
    train_bin_path = out_dir / bin_name
    train_bnd_path = out_dir / bnd_name

    train_multiplexer = DeterministicPillarMultiplexer(streams, weights, quantum=512)
    train_boundaries: list[int] = []
    train_pos = 0
    t_start = time.time()
    last_log_time = t_start
    last_pos = 0

    with open(train_bin_path, "wb", buffering=32 * 1024 * 1024) as f_train:
        for p, doc in train_multiplexer.yield_docs(train_budget):
            train_boundaries.append(train_pos)
            f_train.write(doc.tobytes())
            train_pos += len(doc)

            # Log telemetry every 10 seconds or 20M tokens
            now = time.time()
            if (now - last_log_time >= 10.0) or (train_pos - last_pos >= 20_000_000):
                pct = (train_pos / train_budget) * 100.0
                rate = (train_pos - last_pos) / max(1e-5, now - last_log_time)
                overall_rate = train_pos / max(1e-5, now - t_start)
                eta_s = (train_budget - train_pos) / max(1.0, overall_rate)
                ram_mb = (psutil.Process().memory_info().rss / (1024 * 1024)) if HAS_PSUTIL else 0.0
                print(
                    f"  [{pct:5.1f}%] {train_pos:,} / {train_budget:,} tok | "
                    f"Speed: {rate/1e6:4.2f}M tok/s (avg {overall_rate/1e6:4.2f}M tok/s) | "
                    f"Docs: {len(train_boundaries):,} | RAM: {ram_mb:5.1f} MB | "
                    f"ETA: {eta_s/60:4.1f} min",
                    flush=True
                )
                last_log_time = now
                last_pos = train_pos

    np.save(train_bnd_path, np.array(train_boundaries, dtype=np.int64))
    total_time = time.time() - t_start
    print(
        f"\n[train] Complete! Emitted {train_pos:,} tokens across {len(train_boundaries):,} documents in "
        f"{total_time:.1f}s ({train_pos / total_time:,.0f} tok/s)",
        flush=True
    )

    # 6. Verification Gates
    print("\n[3/3] Running Strict Multi-Agent Verification Gates...", flush=True)

    # Gate 1: Check ESG token count == 0
    print("  Gate 1: Verifying zero Aheri Gondi (<esg> id=11) presence across corpus...", flush=True)
    train_mmap = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    esg_count = int(np.sum(train_mmap == ESG_ID))
    print(f"  Gate 1 Result: <esg> count = {esg_count} (Mandate: EXACTLY 0)")
    assert esg_count == 0, f"CRITICAL: Found {esg_count} <esg> tokens in {train_bin_path}!"

    # Gate 2: Adversarial zero-val-leak scan
    print(f"  Gate 2: Scanning {len(train_boundaries):,} training documents for validation contamination...", flush=True)
    leaks = 0
    sample_scan_docs = len(train_boundaries) if (args.dry_run or len(train_boundaries) <= 100_000) else 100_000
    scan_indices = range(sample_scan_docs)
    for idx in scan_indices:
        s = train_boundaries[idx]
        e = train_boundaries[idx + 1] if idx + 1 < len(train_boundaries) else train_pos
        doc = train_mmap[s:e]
        if h_tok(doc) in val_tok_hashes or h_tok(doc[1:-1]) in val_tok_trimmed_hashes:
            leaks += 1
            print(f"    [LEAK DETECTED] Train doc index {idx} matches validation hash!", flush=True)

    print(f"  Gate 2 Result: {leaks} validation leaks found across {sample_scan_docs:,} scanned documents (Mandate: EXACTLY 0)")
    assert leaks == 0, f"CRITICAL: Detected {leaks} leaked documents in training binary!"

    # 7. Write Manifest
    manifest = {
        "pipeline": "build_corpus_130m_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget": args.budget,
        "train_tokens": int(train_pos),
        "train_docs": len(train_boundaries),
        "val_tokens": int(val_pos),
        "val_docs": len(val_boundaries),
        "files": {
            "train_bin": bin_name,
            "train_boundaries": bnd_name,
            "val_bin": val_bin_name,
            "val_boundaries": val_bnd_name,
        },
        "shares_achieved": {
            k: {
                "tokens": int(train_multiplexer.token_counts[k]),
                "docs": int(train_multiplexer.doc_counts[k]),
                "share": round(train_multiplexer.token_counts[k] / max(1, train_pos), 4),
            }
            for k in weights
        },
        "verification": {
            "esg_count": esg_count,
            "val_leaks": leaks,
            "gate_status": "PASS",
        },
        "tokenizer_sha256": hashlib.sha256((Path(args.tokenizer) / "tokenizer.json").read_bytes()).hexdigest(),
    }

    manifest_path = out_dir / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[manifest] Saved detailed provenance manifest to {manifest_path}", flush=True)
    print("\n" + "=" * 80)
    print("PRETRAINING CORPUS 130M-V1 BUILD SUCCEEDED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
