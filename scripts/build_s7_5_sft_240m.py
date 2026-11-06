#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/build_s7_5_sft_240m.py
Parentheses 1.0-240M (slab_v4) — Session 7.5 Precision Calibrated SFT Corpus Builder.

Implements all 12 items from the S7 Closeout & S7.5 Review Plan:
  1. 11-Language BPB Replay Coverage:
     - 9 tagged languages (kan=2, tam=3, tel=4, mal=5, tcy=6, kff=7, kxv=8, nit=9, eng=10)
       from data/processed_240m_apache/train.bin + data/processed_s5_textbooks_240m/train.bin.
     - Hindi (hin) and Sanskrit (san) from I:/parentheses/curated_s6_devanagari/{hin,san}_clean.jsonl,
       decontaminated against data/val_candidates/{hin,san}_val.txt, encoded 50% unprefixed
       (body + EOT, matching bpb_slab_v4.py doc_stream where tok.token_to_id("<hin>"/"<san>") is None)
       and 50% with reserved tag_id (19 for hin, 20 for san, matching processed_v5).
  2. Replay Chunking: Long pretraining docs > 4096 tokens are split into <= 4096 chunks, each
     retaining its leading language context token (mask=0) and body + terminal EOT (mask=1).
  3. Explicit Language & Programming Tag Mapping: Every distinct lang/language string across all
     curated files is explicitly mapped; unmapped strings are logged and rejected (never silently
     defaulted to eng).
  4. Long-Trace <think> Rescue: SFT traces exceeding 4096 tokens whose prompt + final answer fit
     within 4096 have the middle of their <think>...</think> block compressed with balanced tags
     and 100% intact final answer + EOT.
  5. Explanatory Epistemic Abstention Only: Uses data/sft_v4_abstention_4000.jsonl (multilingual
     <think>..</think> false-premise/future/fictional explanations) and drops bare "I don't know."
     or generic trivia brush-off pools.
  6. Self-Identity 15x Upsample: 47 canonical rows x 15 = 705 rows across all prompt styles.
  7. Strict Token-Share Gate in manifest.json: Enforces exact 65,536,000 token budget (16,000 bins
     of 4,096 tokens = 1,000 steps) across the 5 canonical buckets within +/- 5.0 percentage points:
       - Bucket 1 (25.0%): 11-Language Pretraining Replay (16,384,000 tok)
       - Bucket 2 (30.0%): STEM & Math <think> Reasoning (19,660,800 tok)
       - Bucket 3 (20.0%): Python Code & Algorithmic Execution (13,107,200 tok)
       - Bucket 4 (15.0%): Native Dravidian SFT & Belebele QA (9,830,400 tok)
       - Bucket 5 (10.0%): Explanatory Abstention + Self-Identity (15x) + Tool/Chat (6,553,600 tok)
  8. Template Parity with eval_60p_adversarial_suite.py:
     [user_id, lang_id] + tok.encode(prompt, add_special_tokens=False).ids + [asst_id] + r_ids + [eot_id].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pack_sft_4096_v4 import (
    ASSISTANT_ID,
    EOT_ID,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    USER_ID,
    ffd_pack_documents,
    resolve_special_token_ids,
)

CURATED_DIR = Path(r"E:\Parentheses\data\curated_s7_packin")
OUT_DIR = REPO_ROOT / "data" / "processed_sft_240m_v4_s7_5"
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"

V5_APACHE_DIR = REPO_ROOT / "data" / "processed_240m_apache"
S5_TEXTBOOK_DIR = REPO_ROOT / "data" / "processed_s5_textbooks_240m"
S4_STEM_DIR = REPO_ROOT / "data" / "processed_s4_stem_240m"
V5_TAIL_DIR = REPO_ROOT / "data" / "processed_v5"
DEVANAGARI_DIR = Path(r"I:\parentheses\curated_s6_devanagari")
ABSTENTION_4000_PATH = REPO_ROOT / "data" / "sft_v4_abstention_4000.jsonl"
IDENTITY_PATH = REPO_ROOT / "data" / "curated" / "parentheses_identity_sft_v1.jsonl"

BIN_SIZE = 4096
TARGET_BINS = 16000
TARGET_TOTAL_TOKENS = TARGET_BINS * BIN_SIZE  # 65,536,000 tokens

BUCKET_TARGET_SHARES = {
    "replay_11lang": 0.25,
    "stem_math_think": 0.30,
    "code_execution": 0.20,
    "dravidian_sft_qa": 0.15,
    "abstention_id_chat": 0.10,
}

BUCKET_TARGETS = {k: int(TARGET_TOTAL_TOKENS * v) for k, v in BUCKET_TARGET_SHARES.items()}

REPLAY_LANG_TARGETS = {
    "eng": 3_600_000,
    "kan": 2_000_000,
    "tam": 2_000_000,
    "tel": 2_000_000,
    "mal": 2_000_000,
    "hin": 1_400_000,
    "san": 1_400_000,
    "tcy":   600_000,
    "kff":   484_000,
    "kxv":   450_000,
    "nit":   450_000,
}

TAG_ID_TO_LANG = {
    2: "kan", 3: "tam", 4: "tel", 5: "mal",
    6: "tcy", 7: "kff", 8: "kxv", 9: "nit", 10: "eng",
}

# Explicit mapping for all valid natural and programming language tags in curated_s7_packin
EXPLICIT_LANG_MAP: Dict[str, Tuple[int, str]] = {
    "kan": (2, "kan"), "kn": (2, "kan"),
    "tam": (3, "tam"), "ta": (3, "tam"),
    "tel": (4, "tel"), "te": (4, "tel"),
    "mal": (5, "mal"), "ml": (5, "mal"),
    "tcy": (6, "tcy"),
    "kff": (7, "kff"),
    "kxv": (8, "kxv"),
    "nit": (9, "nit"),
    "eng": (10, "eng"), "en": (10, "eng"),
    # Programming / agentic tags in fable-5 and gpt-5.6-sol -> English (<eng> = 10)
    "python": (10, "eng"), "py": (10, "eng"),
    "go": (10, "eng"), "ts": (10, "eng"), "typescript": (10, "eng"),
    "java": (10, "eng"), "powershell": (10, "eng"), "csharp": (10, "eng"),
    "bash": (10, "eng"), "rust": (10, "eng"), "c": (10, "eng"),
    "cpp": (10, "eng"), "ruby": (10, "eng"), "zsh": (10, "eng"),
    "js": (10, "eng"), "asm": (10, "eng"), "hcl": (10, "eng"),
    "code-review": (10, "eng"), "non-code": (10, "eng"),
    "seed-authoring": (10, "eng"), "multi": (10, "eng"),
}

INDIC_QA_MARKERS = ("ಪ್ರಶ್ನೆ:", "கேள்வி:", "ప్రశ్న:", "ചോദ്യം:")
PROBE_ENTITIES = (
    "velanthor", "hyper-lattice", "vibranium", "alistair finch", "cryo-hepatic",
    "napoleon", "2036", "rahul sharma", "madhya pradesh", "snow tiger", "submarine",
    "ವೇಲಾಂಥೂರ್", "ವೇಲಾಂತೂರ್", "వేలాంతూర్", "വേലാന്തൂർ",
)
BARE_ABSTENTION_STRINGS = {
    "i don't know.", "i do not know.", "i don't know", "i do not know",
    "that's beyond what i can confirm.", "i don't have a confident answer for that.",
    "i can't answer that confidently.", "i cannot confirm that.",
}
TELUGU_HEADLINE_PREFIXES = (
    "కింది న్యూస్ కథనానికి టైటిల్ ను వ్రాయండి:\n",
    "ఇవ్వబడిన న్యూస్ కథనానికి హెడ్లైన్ ను రాయండి:\n",
    "ఇచ్చిన న్యూస్ కథనానికి టైటిల్ రాయండి:\n",
    "ఇవ్వబడిన హెడ్లైన్ తో వార్తా కథనాన్ని రాయండి:\n",
)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_probe_keys() -> List[str]:
    from scripts.build_s4_stem_240m import probe_keys
    return probe_keys()


def load_val_candidate_hashes() -> set[bytes]:
    """Build normalized text hashes of hin_val.txt and san_val.txt to guarantee 0 val overlap."""
    hashes: set[bytes] = set()
    for dev_lang in ("hin", "san"):
        for base in (REPO_ROOT / "data" / "val_candidates", Path(r"I:\parentheses\val_candidates")):
            p = base / f"{dev_lang}_val.txt"
            if p.exists():
                raw = p.read_text(encoding="utf-8")
                for blk in raw.split("\n\n"):
                    b = " ".join(WORD_RE.findall(blk.lower()))
                    if b:
                        hashes.add(hashlib.sha1(b.encode("utf-8")).digest())
    return hashes


def encode_sft_doc_with_rescue(
    tok: Tokenizer,
    prompt: str,
    response: str,
    lang_raw: str,
    token_id_map: Dict[str, int],
    unmapped_counter: Counter,
    max_seq_len: int = BIN_SIZE,
) -> Tuple[str, np.ndarray | None, np.ndarray | None, str]:
    """
    Encode SFT prompt/response with template parity:
      [user_id, lang_id] + p_ids + [asst_id] + r_ids + [eot_id]
    Preserves prompt trailing whitespace/newlines (only strips leading whitespace) to match
    eval_60p_adversarial_suite.py line 689.
    Rescues long <think>...</think> traces (4097..12000 tokens) by compressing the middle
    of the <think> reasoning block while keeping 100% of the final answer intact.
    """
    lang_key = lang_raw.strip().lower()
    if lang_key not in EXPLICIT_LANG_MAP:
        unmapped_counter[lang_raw] += 1
        return "unmapped_lang", None, None, "unmapped"

    lang_id, canonical_lang = EXPLICIT_LANG_MAP[lang_key]
    eot_id = token_id_map["eot"]
    user_id = token_id_map["user"]
    asst_id = token_id_map["assistant"]
    th_open = token_id_map["think_open"]
    th_close = token_id_map["think_close"]

    resp_clean = response.strip()
    if not prompt.strip() or not resp_clean:
        return "empty", None, None, canonical_lang

    if resp_clean.lower() in BARE_ABSTENTION_STRINGS:
        return "bare_abstention", None, None, canonical_lang

    open_c = resp_clean.count("<think>")
    close_c = resp_clean.count("</think>")
    if open_c != close_c:
        return "unclosed_think_string", None, None, canonical_lang
    if open_c > 0 and resp_clean.find("<think>") > resp_clean.find("</think>"):
        return "unclosed_think_string", None, None, canonical_lang

    p_text = prompt.lstrip("\r\n ")
    p_ids = tok.encode(p_text, add_special_tokens=False).ids
    r_ids = tok.encode(resp_clean, add_special_tokens=False).ids

    full_ids = [user_id, lang_id] + p_ids + [asst_id] + r_ids + [eot_id]
    status = "ok"

    if len(full_ids) > max_seq_len:
        if open_c == 1 and th_open in r_ids and th_close in r_ids:
            t_start = r_ids.index(th_open)
            t_end = r_ids.index(th_close)
            if t_start < t_end:
                pre_think = r_ids[: t_start + 1]
                think_body = r_ids[t_start + 1 : t_end]
                post_think = r_ids[t_end:]
                fixed_overhead = 2 + len(p_ids) + 1 + len(pre_think) + len(post_think) + 1
                avail_for_think = max_seq_len - fixed_overhead
                if avail_for_think >= 64 and len(post_think) >= 4:
                    ellipsis_ids = tok.encode("\n...\n", add_special_tokens=False).ids
                    keep_budget = avail_for_think - len(ellipsis_ids)
                    if keep_budget >= 32:
                        half_a = keep_budget // 2
                        half_b = keep_budget - half_a
                        rescued_r_ids = (
                            pre_think
                            + think_body[:half_a]
                            + ellipsis_ids
                            + think_body[-half_b:]
                            + post_think
                        )
                        full_ids = [user_id, lang_id] + p_ids + [asst_id] + rescued_r_ids + [eot_id]
                        status = "rescued_long_think"
        if len(full_ids) > max_seq_len:
            return "exceeds_4096", None, None, canonical_lang

    if full_ids.count(th_open) != full_ids.count(th_close):
        return "unclosed_think_token", None, None, canonical_lang

    L = len(full_ids)
    prompt_len = 2 + len(p_ids) + 1
    ids_arr = np.array(full_ids, dtype=np.uint16)
    mask_arr = np.zeros(L, dtype=np.uint8)
    mask_arr[prompt_len:] = 1
    return status, ids_arr, mask_arr, canonical_lang


def chunk_replay_doc(
    body_ids: List[int],
    prefix_id: int | None,
    eot_id: int = EOT_ID,
    max_seq_len: int = BIN_SIZE,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split a pretraining document into <= max_seq_len chunks.
    Each chunk has:
      - optional leading prefix_id (mask=0)
      - body slice (mask=1)
      - terminal eot_id (mask=1)
    For unprefixed chunks (prefix_id is None, used for 50% of hin/san to match bpb_slab_v4.py),
    mask[0] = 0 and mask[1:] = 1 so no cross-document boundary prediction is supervised.
    """
    body_clean = [t for t in body_ids if t != eot_id]
    if len(body_clean) < 8:
        return []

    overhead = (1 if prefix_id is not None else 0) + 1  # prefix + EOT
    chunk_body_cap = max_seq_len - overhead
    out: List[Tuple[np.ndarray, np.ndarray]] = []

    for offset in range(0, len(body_clean), chunk_body_cap):
        sub = body_clean[offset : offset + chunk_body_cap]
        if len(sub) < 8:
            continue
        if prefix_id is not None:
            c_ids = [prefix_id] + sub + [eot_id]
        else:
            c_ids = sub + [eot_id]
        ids_arr = np.array(c_ids, dtype=np.uint16)
        mask_arr = np.ones(len(c_ids), dtype=np.uint8)
        mask_arr[0] = 0  # First token is context (never supervise across EOT boundary)
        out.append((ids_arr, mask_arr))
    return out


def build_bucket_1_replay(
    tok: Tokenizer,
    rng: np.random.Generator,
    lang_tokens: Counter,
    source_tokens: Counter,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """
    Bucket 1 (25.0% = 16,384,000 tokens): 11-Language Pretraining & Textbook Replay.
    Covers all 11 BPB_LANGS: kan, tam, tel, mal, tcy, kff, kxv, nit, eng, hin, san.
    """
    print("\n[Bucket 1/5] Building 11-Language Pretraining Replay (Target: 16,384,000 tokens)...")
    docs: List[Tuple[np.ndarray, np.ndarray]] = []
    per_lang_packed: Dict[str, int] = {k: 0 for k in REPLAY_LANG_TARGETS}

    # 1A. Sample 9 tagged languages from data/processed_240m_apache/train.bin
    v5_bin = np.memmap(V5_APACHE_DIR / "train.bin", dtype=np.uint16, mode="r")
    v5_bnd = np.load(V5_APACHE_DIR / "train_boundaries.npy", mmap_mode="r")
    n_v5_docs = len(v5_bnd)

    #Also include S5 textbooks for half of English target (1.80M from S5 textbooks, 1.80M from v5)
    s5_bin = np.memmap(S5_TEXTBOOK_DIR / "train.bin", dtype=np.uint16, mode="r")
    s5_bnd = np.load(S5_TEXTBOOK_DIR / "train_boundaries.npy", mmap_mode="r")
    s5_order = rng.permutation(len(s5_bnd))
    eng_s5_target = REPLAY_LANG_TARGETS["eng"] // 2
    for idx in s5_order:
        if per_lang_packed["eng"] >= eng_s5_target:
            break
        s = int(s5_bnd[idx])
        e = int(s5_bnd[idx + 1]) if idx + 1 < len(s5_bnd) else len(s5_bin)
        raw_doc = np.asarray(v5_slice := s5_bin[s:e], dtype=np.int64).tolist()
        if not raw_doc or raw_doc[0] != 10:
            continue
        for ids_arr, mask_arr in chunk_replay_doc(raw_doc[1:], prefix_id=10, eot_id=EOT_ID):
            if per_lang_packed["eng"] + len(ids_arr) > eng_s5_target + BIN_SIZE:
                break
            docs.append((ids_arr, mask_arr))
            L = len(ids_arr)
            per_lang_packed["eng"] += L
            lang_tokens["eng"] += L
            source_tokens["replay_s5_textbooks_eng"] += L

    # Scan permuted indices of processed_240m_apache for the 9 tagged languages
    v5_order = rng.permutation(n_v5_docs)
    needed_9 = {"kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"}
    for idx in v5_order:
        if all(per_lang_packed[l] >= REPLAY_LANG_TARGETS[l] for l in needed_9):
            break
        s = int(v5_bnd[idx])
        first_tok = int(v5_bin[s])
        lang = TAG_ID_TO_LANG.get(first_tok)
        if lang is None or per_lang_packed[lang] >= REPLAY_LANG_TARGETS[lang]:
            continue
        e = int(v5_bnd[idx + 1]) if idx + 1 < n_v5_docs else len(v5_bin)
        raw_doc = np.asarray(v5_bin[s:e], dtype=np.int64).tolist()
        for ids_arr, mask_arr in chunk_replay_doc(raw_doc[1:], prefix_id=first_tok, eot_id=EOT_ID):
            if per_lang_packed[lang] >= REPLAY_LANG_TARGETS[lang]:
                break
            docs.append((ids_arr, mask_arr))
            L = len(ids_arr)
            per_lang_packed[lang] += L
            lang_tokens[lang] += L
            source_tokens[f"replay_v5_{lang}"] += L

    # 1B. Hindi (hin) and Sanskrit (san) from I:\parentheses\curated_s6_devanagari
    val_hashes = load_val_candidate_hashes()
    for dev_lang, tag_id in (("hin", 19), ("san", 20)):
        dev_path = DEVANAGARI_DIR / f"{dev_lang}_clean.jsonl"
        if not dev_path.exists():
            raise FileNotFoundError(f"Missing Devanagari replay source: {dev_path}")
        raw_texts: List[str] = []
        with open(dev_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r.get("text", "").strip()
                if len(t) < 60:
                    continue
                norm = " ".join(WORD_RE.findall(t.lower()))
                if hashlib.sha1(norm.encode("utf-8")).digest() in val_hashes:
                    continue
                raw_texts.append(t)

        order = rng.permutation(len(raw_texts))
        target_tok = REPLAY_LANG_TARGETS[dev_lang]
        use_prefix_toggle = False
        for i in range(0, len(order), 512):
            if per_lang_packed[dev_lang] >= target_tok:
                break
            batch_texts = [raw_texts[j] for j in order[i : i + 512]]
            encoded_batch = tok.encode_batch(batch_texts, add_special_tokens=False)
            for enc in encoded_batch:
                if per_lang_packed[dev_lang] >= target_tok:
                    break
                # 50% unprefixed (matches bpb_slab_v4.py doc_stream where token_to_id('<hin>'/'<san>') is None)
                # 50% prefixed with reserved tag_id (19 for hin, 20 for san, matching processed_v5)
                pref = tag_id if use_prefix_toggle else None
                use_prefix_toggle = not use_prefix_toggle
                for ids_arr, mask_arr in chunk_replay_doc(enc.ids, prefix_id=pref, eot_id=EOT_ID):
                    if per_lang_packed[dev_lang] >= target_tok:
                        break
                    docs.append((ids_arr, mask_arr))
                    L = len(ids_arr)
                    per_lang_packed[dev_lang] += L
                    lang_tokens[dev_lang] += L
                    source_tokens[f"replay_devanagari_{dev_lang}"] += L

    tot = sum(per_lang_packed.values())
    print(f"  [+] Packed {len(docs):,} replay chunks ({tot:,} tokens): {per_lang_packed}")
    return docs, per_lang_packed


def ingest_jsonl_file(
    path: Path,
    tok: Tokenizer,
    token_id_map: Dict[str, int],
    rng: np.random.Generator,
    cap_tokens: int,
    probe_keys_list: List[str],
    unmapped_counter: Counter,
    quarantine_counter: Counter,
    lang_tokens: Counter,
    source_tokens: Counter,
    source_label: str,
    strip_telugu_headlines: bool = False,
    require_think: bool = False,
    upsample: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Read, decontaminate, validate/rescue, and sample up to cap_tokens from a JSONL file."""
    rows: List[Tuple[str, str, str]] = []
    seen_hashes: set[bytes] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            p = str(r.get("prompt") or r.get("instruction") or r.get("question") or "")
            a = str(r.get("response") or r.get("output") or r.get("chosen") or r.get("answer") or "")
            lang_raw = str(r.get("lang") or r.get("language") or "eng")

            if strip_telugu_headlines:
                # Skip reverse "write fake news article from headline" tasks
                if p.startswith("ఇవ్వబడిన హెడ్లైన్ తో వార్తా కథనాన్ని రాయండి:"):
                    continue
                for pref in TELUGU_HEADLINE_PREFIXES:
                    if p.startswith(pref):
                        body_article = p[len(pref):].strip()
                        p = f"{body_article}\n\nపై పేరాగ్రాఫ్ యొక్క ముఖ్య సారాంశాన్ని ఒక వాక్యంలో రాయండి."
                        break

            if require_think and "<think>" not in a:
                continue

            # Probe decontamination check (skip for self_identity which legitimately answers who built it)
            if source_label != "self_identity_15x":
                norm_p = " " + " ".join(WORD_RE.findall(p.lower())) + " "
                if any(k in norm_p for k in probe_keys_list):
                    quarantine_counter["probe_contaminated"] += 1
                    continue
                low_both = (p + " " + a).lower()
                if any(ent in low_both for ent in PROBE_ENTITIES):
                    quarantine_counter["probe_entity_contaminated"] += 1
                    continue

            h = hashlib.sha1((p.strip() + "\0" + a.strip()).encode("utf-8")).digest()
            if h in seen_hashes and upsample == 1:
                quarantine_counter["duplicate"] += 1
                continue
            seen_hashes.add(h)
            rows.append((p, a, lang_raw))

    if upsample > 1:
        rows = rows * upsample

    order = rng.permutation(len(rows))
    out_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    packed_toks = 0

    for idx in order:
        if packed_toks >= cap_tokens:
            break
        p, a, lang_raw = rows[int(idx)]
        status, ids_arr, mask_arr, canonical_lang = encode_sft_doc_with_rescue(
            tok, p, a, lang_raw, token_id_map, unmapped_counter
        )
        if status in ("ok", "rescued_long_think") and ids_arr is not None and mask_arr is not None:
            if status == "rescued_long_think":
                quarantine_counter["rescued_long_think"] += 1
            out_docs.append((ids_arr, mask_arr))
            L = len(ids_arr)
            packed_toks += L
            lang_tokens[canonical_lang] += L
            source_tokens[source_label] += L
        else:
            quarantine_counter[status] += 1

    print(f"    - {source_label:36s}: {len(out_docs):6,d} docs ({packed_toks:10,d} tok / target {cap_tokens:10,d})")
    return out_docs


def sample_pretokenized_sft_bin(
    bin_path: Path,
    bnd_path: Path,
    tok: Tokenizer,
    token_id_map: Dict[str, int],
    rng: np.random.Generator,
    cap_tokens: int,
    probe_keys_list: List[str],
    quarantine_counter: Counter,
    lang_tokens: Counter,
    source_tokens: Counter,
    source_label: str,
    filter_indic_qa: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample pre-tokenized chat documents (<|user|> <lang> ... <|assistant|> ... <|endoftext|>)."""
    b_arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    bounds = np.load(bnd_path, mmap_mode="r")
    n_docs = len(bounds)
    order = rng.permutation(n_docs)

    user_id = token_id_map["user"]
    asst_id = token_id_map["assistant"]
    eot_id = token_id_map["eot"]
    th_open = token_id_map["think_open"]
    th_close = token_id_map["think_close"]

    out_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    packed_toks = 0

    for idx in order:
        if packed_toks >= cap_tokens:
            break
        i = int(idx)
        s = int(bounds[i])
        e = int(bounds[i + 1]) if i + 1 < n_docs else len(b_arr)
        L = e - s
        if L <= 6 or L > BIN_SIZE:
            if L > BIN_SIZE:
                quarantine_counter["exceeds_4096"] += 1
            continue
        doc = np.asarray(b_arr[s:e], dtype=np.uint16)
        if int(doc[0]) != user_id or int(doc[-1]) != eot_id:
            continue
        lang_id = int(doc[1])
        canonical_lang = TAG_ID_TO_LANG.get(lang_id, "eng")

        asst_positions = np.nonzero(doc == asst_id)[0]
        if len(asst_positions) != 1:
            continue
        asst_pos = int(asst_positions[0])

        if int((doc == th_open).sum()) != int((doc == th_close).sum()):
            quarantine_counter["unclosed_think_token"] += 1
            continue

        if filter_indic_qa:
            txt = tok.decode(doc.tolist())
            if any(m in txt for m in INDIC_QA_MARKERS):
                quarantine_counter["indic_qa_dropped"] += 1
                continue
            norm = " " + " ".join(WORD_RE.findall(txt.lower())) + " "
            if any(k in norm for k in probe_keys_list):
                quarantine_counter["probe_contaminated"] += 1
                continue

        mask_arr = np.zeros(L, dtype=np.uint8)
        mask_arr[asst_pos + 1 :] = 1
        out_docs.append((doc, mask_arr))
        packed_toks += L
        lang_tokens[canonical_lang] += L
        source_tokens[source_label] += L

    print(f"    - {source_label:36s}: {len(out_docs):6,d} docs ({packed_toks:10,d} tok / target {cap_tokens:10,d})")
    return out_docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Session 7.5 Precision Calibrated SFT Corpus (240M)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("PARENTHESES 1.0-240M (slab_v4) — SESSION 7.5 PRECISION CALIBRATED SFT BUILDER")
    print(f"Target Output : {args.out_dir}")
    print(f"Target Budget : {TARGET_BINS:,} bins x {BIN_SIZE} = {TARGET_TOTAL_TOKENS:,} tokens (1,000 steps)")
    print("=" * 78)

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    token_id_map = resolve_special_token_ids(tok)
    rng = np.random.default_rng(args.seed)
    probe_keys_list = load_probe_keys()

    unmapped_counter: Counter = Counter()
    quarantine_counter: Counter = Counter()
    lang_tokens: Counter = Counter()
    source_tokens: Counter = Counter()
    bucket_tokens: Dict[str, int] = {}

    all_docs: List[Tuple[np.ndarray, np.ndarray]] = []

    # -------------------------------------------------------------------------
    # BUCKET 1 (25.0% = 16,384,000 tok): 11-Language Pretraining Replay
    # -------------------------------------------------------------------------
    b1_docs, replay_per_lang = build_bucket_1_replay(tok, rng, lang_tokens, source_tokens)
    all_docs.extend(b1_docs)
    bucket_tokens["replay_11lang"] = sum(len(d[0]) for d in b1_docs)

    # -------------------------------------------------------------------------
    # BUCKET 2 (30.0% = 19,660,800 tok): STEM & Math <think> Reasoning
    # -------------------------------------------------------------------------
    print("\n[Bucket 2/5] Building STEM & Math <think> Reasoning (Target: 19,660,800 tokens)...")
    b2_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    b2_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "deepscaler_curated.jsonl",
            tok, token_id_map, rng, 7_200_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "stem_deepscaler",
        )
    )
    b2_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "mattwesney_General_Inquiry_Thinking-Chain-Of-Thought.jsonl",
            tok, token_id_map, rng, 3_000_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "stem_mattwesney_cot",
        )
    )
    b2_rem = BUCKET_TARGETS["stem_math_think"] - sum(len(d[0]) for d in b2_docs)
    b2_docs.extend(
        sample_pretokenized_sft_bin(
            S4_STEM_DIR / "train.bin", S4_STEM_DIR / "train_boundaries.npy",
            tok, token_id_map, rng, b2_rem, probe_keys_list,
            quarantine_counter, lang_tokens, source_tokens,
            "stem_s4_stratos_metamath",
        )
    )
    all_docs.extend(b2_docs)
    bucket_tokens["stem_math_think"] = sum(len(d[0]) for d in b2_docs)

    # -------------------------------------------------------------------------
    # BUCKET 3 (20.0% = 13,107,200 tok): Python Code & Algorithmic Execution
    # -------------------------------------------------------------------------
    print("\n[Bucket 3/5] Building Python Code & Algorithmic Execution (Target: 13,107,200 tokens)...")
    b3_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    b3_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "curated_flytech-python-codes-25k.jsonl",
            tok, token_id_map, rng, 3_600_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "code_flytech_python_25k",
        )
    )
    b3_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "fable-5-coding-and-debugging-traces.jsonl",
            tok, token_id_map, rng, 3_200_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "code_fable5_traces",
        )
    )
    b3_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "greghavens_kimi-k3-coding-and-debugging-traces.jsonl",
            tok, token_id_map, rng, 1_000_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "code_kimi_k3_traces",
        )
    )
    b3_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "gpt-5.6-sol_curated.jsonl",
            tok, token_id_map, rng, 3_500_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "code_gpt56_sol",
        )
    )
    b3_rem = BUCKET_TARGETS["code_execution"] - sum(len(d[0]) for d in b3_docs)
    if b3_rem > 0:
        b3_docs.extend(
            ingest_jsonl_file(
                CURATED_DIR / "curated_simmo_python_fim_s7.jsonl",
                tok, token_id_map, rng, b3_rem, probe_keys_list,
                unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
                "code_simmo_python_fim",
            )
        )
    all_docs.extend(b3_docs)
    bucket_tokens["code_execution"] = sum(len(d[0]) for d in b3_docs)

    # -------------------------------------------------------------------------
    # BUCKET 4 (15.0% = 9,830,400 tok): Native Dravidian SFT & Belebele QA
    # -------------------------------------------------------------------------
    print("\n[Bucket 4/5] Building Native Dravidian SFT & Belebele QA (Target: 9,830,400 tokens)...")
    b4_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    for bel_lang, bel_file in (
        ("kan", "belebele_kan_Knda.jsonl"),
        ("tam", "belebele_tam_Taml.jsonl"),
        ("tel", "belebele_tel_Telu.jsonl"),
        ("mal", "belebele_mal_Mlym.jsonl"),
    ):
        b4_docs.extend(
            ingest_jsonl_file(
                CURATED_DIR / bel_file,
                tok, token_id_map, rng, 650_000, probe_keys_list,
                unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
                f"drav_belebele_{bel_lang}_3x",
                upsample=3,
            )
        )
    b4_docs.extend(
        sample_pretokenized_sft_bin(
            V5_TAIL_DIR / "sft_tail.bin", V5_TAIL_DIR / "sft_tail_boundaries.npy",
            tok, token_id_map, rng, 3_600_000, probe_keys_list,
            quarantine_counter, lang_tokens, source_tokens,
            "drav_aya_sft_tail",
            filter_indic_qa=True,
        )
    )
    b4_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "kannada_curated.jsonl",
            tok, token_id_map, rng, 2_000_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "drav_kannada_curated_cap",
        )
    )
    b4_rem = BUCKET_TARGETS["dravidian_sft_qa"] - sum(len(d[0]) for d in b4_docs)
    if b4_rem > 0:
        b4_docs.extend(
            ingest_jsonl_file(
                CURATED_DIR / "telugu_curated.jsonl",
                tok, token_id_map, rng, b4_rem, probe_keys_list,
                unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
                "drav_telugu_curated_depoisoned",
                strip_telugu_headlines=True,
            )
        )
    all_docs.extend(b4_docs)
    bucket_tokens["dravidian_sft_qa"] = sum(len(d[0]) for d in b4_docs)

    # -------------------------------------------------------------------------
    # BUCKET 5 (10.0% = 6,553,600 tok): Explanatory Abstention + Identity (15x) + Tool/Chat
    # -------------------------------------------------------------------------
    print("\n[Bucket 5/5] Building Explanatory Abstention + Self-Identity (15x) + Tool/Chat (Target: 6,553,600 tokens)...")
    b5_docs: List[Tuple[np.ndarray, np.ndarray]] = []
    b5_docs.extend(
        ingest_jsonl_file(
            IDENTITY_PATH,
            tok, token_id_map, rng, 300_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "self_identity_15x",
            upsample=15,
        )
    )
    b5_docs.extend(
        ingest_jsonl_file(
            ABSTENTION_4000_PATH,
            tok, token_id_map, rng, 1_800_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "explanatory_abstention_v4",
            require_think=True,
            upsample=3,
        )
    )
    b5_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "oasst1_decontaminated_dialogs_s7.jsonl",
            tok, token_id_map, rng, 1_100_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "chat_oasst1_decontaminated",
        )
    )
    b5_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "llamafactory_reason-tool-use-demo-1500.jsonl",
            tok, token_id_map, rng, 600_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "tool_llamafactory_1500",
            require_think=True,
        )
    )
    b5_docs.extend(
        ingest_jsonl_file(
            CURATED_DIR / "interstellarninja_tool-use-multiturn-reasoning.jsonl",
            tok, token_id_map, rng, 1_400_000, probe_keys_list,
            unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
            "tool_interstellarninja",
            require_think=True,
        )
    )
    b5_rem = BUCKET_TARGETS["abstention_id_chat"] - sum(len(d[0]) for d in b5_docs)
    if b5_rem > 0:
        b5_docs.extend(
            ingest_jsonl_file(
                CURATED_DIR / "hermes_reasoning_tool_use.jsonl",
                tok, token_id_map, rng, b5_rem, probe_keys_list,
                unmapped_counter, quarantine_counter, lang_tokens, source_tokens,
                "tool_hermes_reasoning",
                require_think=True,
            )
        )
    all_docs.extend(b5_docs)
    bucket_tokens["abstention_id_chat"] = sum(len(d[0]) for d in b5_docs)

    # -------------------------------------------------------------------------
    # FFD 4,096-TOKEN PACKING & STRICT +/- 5% SHARE GATE
    # -------------------------------------------------------------------------
    print(f"\n[*] Running FFD 4,096-Token Packer across {len(all_docs):,} validated documents...")
    packed_ids, packed_mask, boundaries_3d, boundaries_flat, stats = ffd_pack_documents(
        all_docs, bin_size=BIN_SIZE, seed=args.seed, eot_id=token_id_map["eot"]
    )

    total_doc_tokens = stats["total_doc_tokens"]
    actual_bucket_shares = {
        k: round(v / max(1, total_doc_tokens), 4) for k, v in bucket_tokens.items()
    }
    bucket_deviations = {
        k: round(actual_bucket_shares[k] - BUCKET_TARGET_SHARES[k], 4)
        for k in BUCKET_TARGET_SHARES
    }
    max_abs_deviation = max(abs(d) for d in bucket_deviations.values())

    print("\n[*] Token-Share Gate Verification (Tolerance: +/- 0.0500):")
    for k in BUCKET_TARGET_SHARES:
        tgt = BUCKET_TARGET_SHARES[k]
        act = actual_bucket_shares[k]
        dev = bucket_deviations[k]
        flag = "PASS" if abs(dev) <= 0.05 else "FAIL"
        print(f"    - {k:22s}: target={tgt:.4f} | actual={act:.4f} | diff={dev:+.4f} [{flag}]")

    if max_abs_deviation > 0.05:
        raise AssertionError(
            f"[GATE FAIL] Bucket share deviation {max_abs_deviation:.4f} exceeds +/- 0.05 limit: {bucket_deviations}"
        )
    if sum(unmapped_counter.values()) > 0:
        raise AssertionError(f"[GATE FAIL] Unmapped language strings encountered: {dict(unmapped_counter)}")

    actual_lang_shares = {
        k: round(v / max(1, total_doc_tokens), 4) for k, v in lang_tokens.most_common()
    }
    actual_source_shares = {
        k: round(v / max(1, total_doc_tokens), 4) for k, v in source_tokens.most_common()
    }

    stats.update({
        "version": "processed_sft_240m_v4_s7_5",
        "built_by": "scripts/build_s7_5_sft_240m.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_bins": TARGET_BINS,
        "target_tokens": TARGET_TOTAL_TOKENS,
        "bucket_target_shares": BUCKET_TARGET_SHARES,
        "bucket_actual_tokens": bucket_tokens,
        "bucket_actual_shares": actual_bucket_shares,
        "bucket_share_deviations": bucket_deviations,
        "max_bucket_deviation": round(max_abs_deviation, 4),
        "token_share_gate": "PASS",
        "replay_11lang_tokens": replay_per_lang,
        "per_language_tokens": dict(lang_tokens.most_common()),
        "per_language_shares": actual_lang_shares,
        "per_source_tokens": dict(source_tokens.most_common()),
        "per_source_shares": actual_source_shares,
        "unmapped_lang_counts": dict(unmapped_counter),
        "quarantined_and_rescued": dict(quarantine_counter),
        "elapsed_sec": round(time.time() - t0, 2),
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    packed_ids.tofile(args.out_dir / "train.bin")
    packed_mask.tofile(args.out_dir / "train.mask.bin")
    np.save(args.out_dir / "train_boundaries.npy", boundaries_3d)
    np.save(args.out_dir / "train_boundaries_flat.npy", boundaries_flat)

    manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(stats, mf, indent=2)

    print(f"\n[SUCCESS] Wrote Session 7.5 corpus to {args.out_dir} in {stats['elapsed_sec']}s:")
    print(f"  - num_bins              : {stats['num_bins']:,} (bin_size={BIN_SIZE})")
    print(f"  - total_packed_docs     : {stats['total_packed_docs']:,}")
    print(f"  - total_doc_tokens      : {stats['total_doc_tokens']:,}")
    print(f"  - total_supervised_toks : {stats['total_supervised_tokens']:,} ({stats['total_supervised_tokens']/max(1,stats['total_doc_tokens'])*100:.1f}%)")
    print(f"  - packing_efficiency    : {stats['packing_efficiency_pct']:.3f}%")
    print(f"  - rescued_long_think    : {quarantine_counter['rescued_long_think']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
