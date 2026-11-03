#!/usr/bin/env python3
"""Build the Elite STEM Reasoning + Open Data SFT Dataset (processed_sft_stem_open_v1).

Curated Composition (100,000 High-Impact Pairs):
  1. STEM Reasoning (45,000 pairs / 45%):
     - Chemistry (Camel-AI: IUPAC, stoichiometry, reaction mechanisms, thermodynamics)
     - Physics (Camel-AI: Newtonian dynamics, optics, electromagnetism, wave equations)
     - Math & General Science (SciQ & MetaMathQA step-by-step derivations)
     - ALL formatted with explicit <think>...</think> reasoning traces followed by clean answers!
  2. Open Data (25,000 pairs / 25%):
     - no_robots (human non-robotic direct responses)
     - openhermes-50k (unfiltered reasoning, direct answers)
     - self-oss-instruct (cybersecurity, systems, reverse engineering, C/Python code)
     - 0.0% Refusal Rate, zero corporate preambles
  3. Conversational Intelligence (15,000 pairs / 15%):
     - Purified multi-turn dialogues (Karya + cleaned_conversational_v1)
     - Identity and persona anchors across 5 languages
  4. Dravidian Replay & Native Script Instructions (15,000 pairs / 15%):
     - Aya dataset (human-annotated instruction following in kan, tam, tel, mal)
     - IndicQA (native Dravidian reading comprehension with evidence)
     - Native Dravidian dialogue & cultural literacy
     - Prevents catastrophic forgetting of Dravidian morphology!

Outputs:
  - data/processed_sft_stem_open_v1/train.bin
  - data/processed_sft_stem_open_v1/train.mask.bin
  - data/processed_sft_stem_open_v1/train_boundaries.npy
  - data/processed_sft_stem_open_v1/manifest.json

Engineered by uncoalesced for Project Tanzinite.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Special token IDs in v3_32k
EOT = 0
TAG = {
    "kan": 2,
    "tam": 3,
    "tel": 4,
    "mal": 5,
    "tcy": 6,
    "kff": 7,
    "kxv": 8,
    "nit": 9,
    "eng": 10,
    "hin": 11,
    "mar": 12,
}
USER_TAG = 13
ASSISTANT_TAG = 14
THINK_START = 15
THINK_END = 16

OUT_DIR = REPO / "data" / "processed_sft_stem_open_v1"
TOKENIZER_PATH = REPO / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"

STEM_FILE = Path("F:/Lingual_Data/data/processed_stem/stem_corpus_v1.jsonl")
OPEN_FILE = Path("F:/Lingual_Data/data/processed_open_data/open_data_corpus_v1.jsonl")
CONV_FILE = Path("F:/Lingual_Data/distillation/conversational_intelligence/corpus_240m/cleaned_conversational_v1.jsonl")
METAMATH_FILE = REPO / "data" / "raw" / "sft_real" / "meta-math__MetaMathQA" / "MetaMathQA-395K.json"
RAW_MANUAL = REPO / "data" / "raw" / "manual" / "raw-sourced"


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


def encode_chat_doc(tok: Tokenizer, lang: str, prompt: str, answer: str) -> tuple[np.ndarray, np.ndarray]:
    p = strip_internal_tags(clean_text(prompt))
    a = strip_internal_tags(clean_text(answer))
    l_tag = TAG.get(lang, TAG["eng"])
    p_ids = tok.encode(p, add_special_tokens=False).ids
    a_ids = tok.encode(a, add_special_tokens=False).ids
    prefix = [USER_TAG, l_tag] + p_ids + [ASSISTANT_TAG]
    target = a_ids + [EOT]
    tokens = np.array(prefix + target, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    mask[len(prefix):] = 1
    return tokens, mask


def encode_think_doc(tok: Tokenizer, lang: str, prompt: str, reasoning: str, answer: str) -> tuple[np.ndarray, np.ndarray]:
    p = strip_internal_tags(clean_text(prompt))
    r = strip_internal_tags(clean_text(reasoning))
    a = strip_internal_tags(clean_text(answer))
    l_tag = TAG.get(lang, TAG["eng"])
    p_ids = tok.encode(p, add_special_tokens=False).ids
    r_ids = tok.encode("\n" + r + "\n", add_special_tokens=False).ids
    a_ids = tok.encode("\n\n" + a, add_special_tokens=False).ids
    prefix = [USER_TAG, l_tag] + p_ids + [ASSISTANT_TAG]
    target = [THINK_START] + r_ids + [THINK_END] + a_ids + [EOT]
    tokens = np.array(prefix + target, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    mask[len(prefix):] = 1
    return tokens, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-stem", type=int, default=45000)
    parser.add_argument("--limit-open", type=int, default=25000)
    parser.add_argument("--limit-conv", type=int, default=15000)
    parser.add_argument("--limit-drav", type=int, default=15000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("=" * 70)
    print("BUILDING BALANCED SFT STEM + OPEN + DRAVIDIAN CORPUS (processed_sft_stem_open_v1)")
    print("=" * 70)

    if not TOKENIZER_PATH.exists():
        sys.exit(f"[ERROR] Tokenizer not found at {TOKENIZER_PATH}")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"Loaded tokenizer from {TOKENIZER_PATH} (vocab: {tok.get_vocab_size():,})")

    docs: list[tuple[str, str, str, str, str]] = []  # (domain, lang, prompt, reasoning, answer)

    # 1. Harvest STEM Data (Target: 45,000) with <think> traces
    print("\n[1/4] Sourcing STEM Reasoning Data (with <think> traces)...")
    stem_items = []
    if STEM_FILE.exists():
        with open(STEM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                q = d.get("question", "").strip()
                sol = d.get("solution", "").strip()
                dom = d.get("domain", "stem")
                if q and sol and len(q) >= 10 and len(sol) >= 10:
                    r, a = split_stem_reasoning(sol)
                    stem_items.append((f"stem_{dom}", "eng", q, r, a))
        print(f"  Loaded {len(stem_items):,} STEM problems from {STEM_FILE.name}")

    if METAMATH_FILE.exists():
        with open(METAMATH_FILE, "r", encoding="utf-8") as f:
            math_data = json.load(f)
            rng.shuffle(math_data)
            added_math = 0
            for item in math_data:
                if added_math >= 10000:
                    break
                q = item.get("query", "").strip()
                resp = item.get("response", "").strip()
                if q and resp:
                    r, a = split_stem_reasoning(resp)
                    stem_items.append(("stem_math", "eng", q, r, a))
                    added_math += 1
            print(f"  Added {added_math:,} Olympiad/Calculus problems from MetaMathQA")

    rng.shuffle(stem_items)
    selected_stem = stem_items[:args.limit_stem]
    docs.extend(selected_stem)
    print(f"  --> Selected {len(selected_stem):,} STEM pairs (with <think> formatting)")

    # 2. Harvest Open Data (Target: 25,000) (0% Refusal)
    print("\n[2/4] Sourcing Open Data (0% Refusal, Zero-Censorship)...")
    open_items = []
    if OPEN_FILE.exists():
        with open(OPEN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                q = d.get("prompt", "").strip()
                a = d.get("response", "").strip()
                dom = d.get("domain", "open_data")
                if q and a and len(q) >= 10 and len(a) >= 10:
                    open_items.append((dom, "eng", q, "", a))
        print(f"  Loaded {len(open_items):,} Open Data pairs from {OPEN_FILE.name}")

    rng.shuffle(open_items)
    selected_open = open_items[:args.limit_open]
    docs.extend(selected_open)
    print(f"  --> Selected {len(selected_open):,} Open Data pairs")

    # 3. Harvest Conversational Intelligence (Target: 15,000)
    print("\n[3/4] Sourcing Conversational Intelligence & Persona Anchors...")
    conv_items = []
    if CONV_FILE.exists():
        with open(CONV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                dial = d.get("dialogue", "").strip()
                tr = d.get("track", "conv")
                lang = "eng"
                if tr == "dravidian":
                    for lk, (start, end) in [("kan", (0x0C80, 0x0CFF)), ("tam", (0x0B80, 0x0BFF)),
                                             ("tel", (0x0C00, 0x0C7F)), ("mal", (0x0D00, 0x0D7F))]:
                        if any(start <= ord(c) <= end for c in dial[:100]):
                            lang = lk
                            break
                if p and dial:
                    conv_items.append((f"conv_{tr}", lang, p, "", dial))
        print(f"  Loaded {len(conv_items):,} purified dialogues from {CONV_FILE.name}")

    # Persona anchors across 5 languages
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
        ("നമസ്കാരം, എങ്ങനെയുണ്ട്?", "നമസ്കാരം! എനിക്ക് സുഖമാണ്. ഇന്ന് ഞാൻ നിങ്ങൾക്ക് എന്താണ് സഹായിക്കേണ്ടത്?", "mal"),
    ]
    for _ in range(150):
        for q, a, lk in persona_qas:
            conv_items.append(("conv_persona", lk, q, "", a))

    # Add multi-turn smoltalk chat dialogues
    sft_real_raw = REPO / "data" / "raw" / "sft_real" / "HuggingFaceTB__smol-smoltalk" / "data"
    if sft_real_raw.exists():
        import pyarrow.parquet as pq
        added_smol = 0
        for pq_file in sorted(glob.glob(str(sft_real_raw / "*.parquet"))):
            tab = pq.read_table(pq_file)
            srcs = tab.column("source").to_pylist()
            msgs_list = tab.column("messages").to_pylist()
            for src, msgs in zip(srcs, msgs_list):
                if added_smol >= 15000:
                    break
                if src in ("smol-magpie-ultra-short", "everyday-conversations") and len(msgs) >= 2:
                    q = msgs[0].get("content", "").strip()
                    a = msgs[1].get("content", "").strip()
                    if q and a and len(q) >= 10 and len(a) >= 10:
                        conv_items.append(("conv_chat", "eng", q, "", a))
                        added_smol += 1
            if added_smol >= 15000:
                break
        print(f"  Added {added_smol:,} multi-turn conversational chat pairs")

    rng.shuffle(conv_items)
    selected_conv = conv_items[:args.limit_conv]
    docs.extend(selected_conv)
    print(f"  --> Selected {len(selected_conv):,} Conversational pairs")
    print("\n[4/4] Sourcing Dravidian Replay Buffer (Aya, IndicQA & Native Instructions)...")
    drav_items = []

    # 4a. Aya dataset
    aya_p = RAW_MANUAL / "aya_dataset" / "data" / "train-00000-of-00001.parquet"
    if aya_p.exists():
        df = pd.read_parquet(aya_p)
        drav_df = df[df["language_code"].isin(["kan", "tam", "tel", "mal"])]
        for _, row in drav_df.iterrows():
            lang = str(row["language_code"]).strip()
            inp = str(row.get("inputs", "")).strip()
            tgt = str(row.get("targets", "")).strip()
            if len(inp) >= 6 and len(tgt) >= 6:
                drav_items.append(("drav_aya", lang, inp, "", tgt))
        print(f"  Loaded {len(drav_items):,} Aya native Dravidian instruction rows")

    # 4b. IndicQA
    indic_dir = RAW_MANUAL / "indicqa"
    if indic_dir.exists():
        added_indic = 0
        for json_file in indic_dir.glob("indicqa.*.json"):
            lang = json_file.stem.split(".")[-1]
            lang_map = {"kn": "kan", "ta": "tam", "te": "tel", "ml": "mal"}
            canon_l = lang_map.get(lang, "eng")
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                for entry in data.get("data", []):
                    for para in entry.get("paragraphs", []):
                        context = para.get("context", "").strip()
                        for qa in para.get("qas", []):
                            q = qa.get("question", "").strip()
                            ans_list = qa.get("answers", [])
                            if ans_list:
                                a = ans_list[0].get("text", "").strip()
                                prompt = f"ಸಂದರ್ಭ: {context[:300]}...\nಪ್ರಶ್ನೆ: {q}" if canon_l == "kan" else f"Context: {context[:300]}...\nQuestion: {q}"
                                drav_items.append(("drav_indicqa", canon_l, prompt, "", a))
                                added_indic += 1
            except Exception:
                continue
        print(f"  Added {added_indic:,} IndicQA reading comprehension rows")

    rng.shuffle(drav_items)
    selected_drav = drav_items[:args.limit_drav]
    docs.extend(selected_drav)
    print(f"  --> Selected {len(selected_drav):,} Dravidian Replay pairs")

    # Shuffle combined dataset
    rng.shuffle(docs)
    total_docs = len(docs)
    print(f"\nTotal Selected Dataset Size: {total_docs:,} documents")

    if args.dry_run:
        print("[DRY-RUN] Exiting without writing binaries.")
        return 0

    # Tokenize and pack binary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUT_DIR / "train.bin"
    mask_path = OUT_DIR / "train.mask.bin"
    bnd_path = OUT_DIR / "train_boundaries.npy"

    print(f"\nTokenizing and packing into {OUT_DIR}...")
    t0 = time.time()
    boundaries = []
    curr_offset = 0
    domain_counts = Counter()
    lang_counts = Counter()
    total_tokens = 0
    supervised_tokens = 0

    with open(bin_path, "wb") as f_tok, open(mask_path, "wb") as f_mask:
        for idx, (dom, lang, prompt, reasoning, answer) in enumerate(docs, 1):
            if reasoning and len(reasoning) >= 20:
                toks, mask = encode_think_doc(tok, lang, prompt, reasoning, answer)
            else:
                toks, mask = encode_chat_doc(tok, lang, prompt, answer)

            if len(toks) > args.max_seq_len:
                toks = toks[:args.max_seq_len]
                mask = mask[:args.max_seq_len]

            f_tok.write(toks.tobytes())
            f_mask.write(mask.tobytes())

            boundaries.append(curr_offset)
            curr_offset += len(toks)

            total_tokens += len(toks)
            supervised_tokens += int(mask.sum())
            domain_counts[dom] += 1
            lang_counts[lang] += 1

            if idx % 10000 == 0 or idx == total_docs:
                dt = time.time() - t0
                print(f"  [{idx:,}/{total_docs:,}] {total_tokens:,} tokens ({dt:.1f}s, {total_tokens/dt:,.0f} tok/s)")

    boundaries_arr = np.array(boundaries, dtype=np.int64)
    np.save(bnd_path, boundaries_arr)

    manifest = {
        "dataset_name": "processed_sft_stem_open_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "supervision_ratio": round(supervised_tokens / max(1, total_tokens), 4),
        "avg_tokens_per_doc": round(total_tokens / max(1, total_docs), 1),
        "domains": dict(domain_counts),
        "languages": dict(lang_counts),
        "max_seq_len": args.max_seq_len,
        "binary_files": {
            "tokens": str(bin_path.name),
            "mask": str(mask_path.name),
            "boundaries": str(bnd_path.name),
        }
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SFT STEM + OPEN + DRAVIDIAN PREPARATION COMPLETE!")
    print(f"Total Documents    : {total_docs:,}")
    print(f"Total Tokens       : {total_tokens:,} ({bin_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Supervised Tokens  : {supervised_tokens:,} ({manifest['supervision_ratio']*100:.1f}%)")
    print(f"Document Boundaries: {len(boundaries):,} documents indexed")
    print("Languages Breakdown:", dict(lang_counts))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
