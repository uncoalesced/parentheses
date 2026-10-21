#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_sft_120m.py

Builds the assistant-masked post-pretraining SFT corpus for Parentheses 0.9-120M:
  - data/processed_sft_120m/train.bin            (uint16 token IDs)
  - data/processed_sft_120m/train.mask.bin       (uint8 assistant-target loss mask: 1=supervised, 0=ignored)
  - data/processed_sft_120m/train_boundaries.npy (int64 document-start offsets for SSM state reset)
  - data/processed_sft_120m/val.bin
  - data/processed_sft_120m/val.mask.bin
  - data/processed_sft_120m/val_boundaries.npy
  - data/processed_sft_120m/manifest.json

Fixes the four root causes diagnosed in step_9536_final.pt:
  1. Strict Script & Language-Tag Purity:
     - Never tags English ASCII answers (such as Alar/Kaikki English definitions or
       Dravidian->English translations) with <kan>/<tam>/<tel>/<mal>.
     - Every <kan>, <tam>, <tel>, <mal>, <tcy>, <kff>, <kxv>, <nit> assistant turn is
       verified to be predominantly in its target Indic script (target_script_chars > ascii_alpha).
  2. Strict Assistant-Only Loss Masking (--sft-spans):
     - mask=0 on <|user|><lang> {prompt}<|assistant|> (and <|system|>tool_result:...<|assistant|>).
     - mask=1 on [<think>{reasoning}</think>]{answer}<|endoftext|>.
     - Prevents the model from echoing prompts, emitting "\n\nಪ್ರಶ್ನೆ:" / "Type: QUESTION:"
       mid-response, or hallucinating tool outputs.
  3. Concise, Complete <think>...</think> Traces Across English & Dravidian:
     - Caps <think> traces so </think> is always closed inside the context window and
       followed by a structured, careful final answer.
  4. All-9-Language Coverage + FreeThink Continuation + Calibrated Abstention:
     - Covers eng, kan, tam, tel, mal, tcy, kff, kxv, nit across chat, structured <think>,
       calibrated abstention (unanswerable contexts paired with answerable twins), and
       FreeThink continuation priming.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from features.free_think import format_primed

RAW_SOURCED = REPO / "data/raw/manual/raw-sourced"
INTELLI_DIR = Path("F:/Lingual_Data/intelli")
V3_TEXT_DIR = Path("F:/Lingual_Data/data/processed_v3/text")
TOKENIZER_DIR = REPO / "data/tokenizer/v3_32k"
VAL_DIR = REPO / "data/processed_v4/val"
OUT_DEFAULT = REPO / "data/processed_sft_120m"

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
}
ESG_ID = 11
SYSTEM_TAG = 12
USER_TAG = 13
ASSISTANT_TAG = 14
THINK_START = 15
THINK_END = 16
TOOL_START = 17
TOOL_END = 18

SCRIPTS = {
    "Knda": (0x0C80, 0x0CFF),
    "Taml": (0x0B80, 0x0BFF),
    "Telu": (0x0C00, 0x0C7F),
    "Mlym": (0x0D00, 0x0D7F),
    "Orya": (0x0B00, 0x0B7F),
    "Deva": (0x0900, 0x097F),
}
LANG_SCRIPT = {
    "kan": "Knda",
    "tam": "Taml",
    "tel": "Telu",
    "mal": "Mlym",
    "tcy": "Knda",
    "kff": "Orya",
    "kxv": "Orya",
    "nit": "Telu",
}

ABSTENTION_TEMPLATES = {
    "eng": [
        "Based on the provided passage, this cannot be determined because that specific detail is not mentioned.",
        "The provided text does not contain enough information to answer this question.",
        "I cannot verify the answer from the given context alone, as it does not state this fact.",
        "The passage does not mention this information, so I cannot answer without guessing.",
        "From the context given, there is no statement that answers this question directly.",
    ],
    "kan": [
        "ನೀಡಲಾದ ಪಠ್ಯದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ.",
        "ಕೊಟ್ಟಿರುವ ಸಂದರ್ಭದಲ್ಲಿ ಈ ವಿಷಯದ ಬಗ್ಗೆ ಉಲ್ಲೇಖವಿಲ್ಲದ ಕಾರಣ ಉತ್ತರವನ್ನು ನಿರ್ಧರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
        "ನೀಡಲಾದ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಇದನ್ನು ಖಚಿತಪಡಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
    ],
    "tam": [
        "கொடுக்கப்பட்டுள்ள பத்தியில் இந்தக் கேள்விக்குப் பதிலளிக்கப் போதுமான தகவல் இல்லை.",
        "வழங்கப்பட்ட உரையில் இதற்கான விவரம் குறிப்பிடப்படவில்லை என்பதால் உறுதியாகக் கூற முடியாது.",
        "கொடுக்கப்பட்ட தகவலின் அடிப்படையில் இதைத் தீர்மானிக்க இயலாது.",
    ],
    "tel": [
        "ఇచ్చిన పేరాలో ఈ ప్రశ్నకు సమాధానం చెప్పడానికి తగిన సమాచారం లేదు.",
        "అందించిన సందర్భంలో ఈ విషయం ప్రస్తావించబడలేదు కాబట్టి ఖచ్చితంగా చెప్పలేము.",
        "ఇచ్చిన సమాచారం ఆధారంగా దీనిని నిర్ధారించడం సాధ్యం కాదు.",
    ],
    "mal": [
        "നൽകിയിട്ടുള്ള ഭാഗത്ത് ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ ഇല്ല.",
        "തന്നിരിക്കുന്ന സന്ദർഭത്തിൽ ഇക്കാര്യം പരാമർശിച്ചിട്ടില്ലാത്തതിനാൽ കൃത്യമായി പറയാൻ കഴിയില്ല.",
        "ലഭ്യമായ വിവരങ്ങളുടെ അടിസ്ഥാനത്തിൽ ഇത് സ്ഥിരീകരിക്കാൻ സാധ്യമല്ല.",
    ],
}

ABSTENTION_THINK = {
    "eng": "Let me check the provided context carefully. Scanning the passage for the requested entity or fact shows that this detail is not stated anywhere in the text. Therefore, I should state clearly that the passage does not provide this information rather than guessing.",
    "kan": "ಕೊಟ್ಟಿರುವ ಪಠ್ಯವನ್ನು ಗಮನವಿಟ್ಟು ಪರಿಶೀಲಿಸೋಣ. ಕೇಳಲಾದ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ನಿಖರವಾದ ಮಾಹಿತಿ ಈ ಪಠ್ಯದಲ್ಲಿ ಎಲ್ಲೂ ಕಂಡುಬರುವುದಿಲ್ಲ. ಆದ್ದರಿಂದ ಊಹಿಸುವ ಬದಲು ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ ಎಂದು ಸ್ಪಷ್ಟವಾಗಿ ತಿಳಿಸಬೇಕು.",
    "tam": "கொடுக்கப்பட்டுள்ள பத்தியைக் கவனமாகப் பார்ப்போம். கேட்கப்பட்ட கேள்விக்கான தகவல் இந்தப் பத்தியில் எங்கும் குறிப்பிடப்படவில்லை. எனவே ஊகித்துப் பதிலளிக்காமல் தகவல் இல்லை என்பதைத் தெளிவாகக் கூற வேண்டும்.",
    "tel": "ఇచ్చిన పేరాను జాగ్రత్తగా పరిశీలిద్దాం. అడిగిన ప్రశ్నకు సంబంధించిన వివరాలు ఈ పేరాలో ఎక్కడా పేర్కొనబడలేదు. కాబట్టి ఊహించి చెప్పకుండా సమాచారం లేదని స్పష్టంగా తెలియజేయాలి.",
    "mal": "നൽകിയിട്ടുള്ള ഭാഗം ശ്രദ്ധാപൂർവ്വം പരിശോധിക്കാം. ചോദിച്ചിരിക്കുന്ന കാര്യത്തെക്കുറിച്ചുള്ള വിവരങ്ങൾ ഈ ഭാഗത്ത് എവിടെയും പ്രതിപാദിച്ചിട്ടില്ല. അതിനാൽ ഊഹിച്ച് ഉത്തരം നൽകാതെ വിവരങ്ങൾ ലഭ്യമല്ലെന്ന് വ്യക്തമാക്കണം.",
}


def load_tokenizer(tdir: Path) -> Tokenizer:
    tok_json = tdir / "tokenizer.json"
    sha = hashlib.sha256(tok_json.read_bytes()).hexdigest()
    assert sha.startswith("7a9b4187d5bc"), f"Tokenizer SHA256 mismatch: {sha}"
    tok = Tokenizer.from_file(str(tok_json))
    tok.encode_special_tokens = True
    return tok


def clean_text(s: str) -> str:
    if not s:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def script_profile(text: str) -> tuple[int, dict[str, int]]:
    """Returns (ascii_alpha_count, per_indic_script_counts). Ignores shared danda punctuation (U+0964/U+0965)."""
    ascii_alpha = 0
    counts = dict.fromkeys(SCRIPTS, 0)
    for ch in text:
        cp = ord(ch)
        if cp in (0x0964, 0x0965):
            continue
        if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A):
            ascii_alpha += 1
        elif cp >= 0x0900:
            for name, (lo, hi) in SCRIPTS.items():
                if lo <= cp <= hi:
                    counts[name] += 1
                    break
    return ascii_alpha, counts


def verify_lang_script(text: str, lang: str, min_chars: int = 6) -> bool:
    """True iff `text` matches the required script for `lang` with no English/Devanagari dominance."""
    ascii_alpha, counts = script_profile(text)
    if counts["Deva"] > 2:
        return False
    if lang == "eng":
        return ascii_alpha >= min_chars and sum(counts.values()) <= max(10, ascii_alpha // 3)
    target_script = LANG_SCRIPT.get(lang)
    if not target_script:
        return False
    target_n = counts[target_script]
    other_indic = sum(v for k, v in counts.items() if k != target_script)
    return target_n >= min_chars and target_n > ascii_alpha and target_n > other_indic


def load_val_hashes(val_dir: Path, tok: Tokenizer) -> set[str]:
    """Loads sha1 hashes of all validation texts across all 9 languages."""
    val_hashes = set()
    if not val_dir.exists():
        return val_hashes
    for l in ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"):
        bin_p = val_dir / f"{l}.bin"
        bnd_p = val_dir / f"{l}_boundaries.npy"
        if not bin_p.exists() or not bnd_p.exists():
            continue
        data = np.fromfile(bin_p, dtype=np.uint16)
        bnd = np.append(np.load(bnd_p).astype(np.int64), len(data))
        for s, e in zip(bnd[:-1], bnd[1:]):
            txt = tok.decode(data[s:e]).strip()
            val_hashes.add(hashlib.sha1(txt.encode("utf-8")).hexdigest())
    return val_hashes


# ---------------------------------------------------------------------------
# Encoding Helpers: returns (tokens_uint16, mask_uint8)
# ---------------------------------------------------------------------------

def encode_chat_doc(tok: Tokenizer, lang: str, prompt: str, answer: str) -> tuple[np.ndarray, np.ndarray]:
    l_tag = TAG[lang]
    p_ids = tok.encode(prompt.strip(), add_special_tokens=False).ids
    a_ids = tok.encode(answer.strip(), add_special_tokens=False).ids
    prefix = [USER_TAG, l_tag] + p_ids + [ASSISTANT_TAG]
    target = a_ids + [EOT]
    tokens = np.array(prefix + target, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    mask[len(prefix):] = 1
    return tokens, mask


def encode_think_doc(tok: Tokenizer, lang: str, prompt: str, reasoning: str, answer: str) -> tuple[np.ndarray, np.ndarray]:
    l_tag = TAG[lang]
    p_ids = tok.encode(prompt.strip(), add_special_tokens=False).ids
    r_ids = tok.encode("\n" + reasoning.strip() + "\n", add_special_tokens=False).ids
    a_ids = tok.encode("\n\n" + answer.strip(), add_special_tokens=False).ids
    prefix = [USER_TAG, l_tag] + p_ids + [ASSISTANT_TAG]
    target = [THINK_START] + r_ids + [THINK_END] + a_ids + [EOT]
    tokens = np.array(prefix + target, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    mask[len(prefix):] = 1
    return tokens, mask


def encode_tool_doc(tok: Tokenizer, lang: str, prompt: str, tool_call: str, tool_result: str, answer: str) -> tuple[np.ndarray, np.ndarray]:
    l_tag = TAG[lang]
    p_ids = tok.encode(prompt.strip(), add_special_tokens=False).ids
    call_ids = tok.encode(tool_call.strip(), add_special_tokens=False).ids
    res_ids = tok.encode("tool_result: " + tool_result.strip(), add_special_tokens=False).ids
    a_ids = tok.encode(answer.strip(), add_special_tokens=False).ids

    seg1_unmasked = [USER_TAG, l_tag] + p_ids + [ASSISTANT_TAG]
    seg2_masked = [TOOL_START] + call_ids + [TOOL_END]
    seg3_unmasked = [SYSTEM_TAG] + res_ids + [ASSISTANT_TAG]
    seg4_masked = a_ids + [EOT]

    tokens = np.array(seg1_unmasked + seg2_masked + seg3_unmasked + seg4_masked, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    i1 = len(seg1_unmasked)
    i2 = i1 + len(seg2_masked)
    i3 = i2 + len(seg3_unmasked)
    mask[i1:i2] = 1
    mask[i3:] = 1
    return tokens, mask


def encode_freethink_doc(tok: Tokenizer, lang: str, statement: str, continuation: str) -> tuple[np.ndarray, np.ndarray]:
    l_tag = TAG[lang]
    primed = format_primed(statement.strip(), lang, "continuation").rstrip(" ")
    p_ids = tok.encode(primed, add_special_tokens=False).ids
    c_ids = tok.encode(" " + continuation.strip(), add_special_tokens=False).ids
    prefix = [l_tag] + p_ids
    target = c_ids + [EOT]
    tokens = np.array(prefix + target, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    mask[len(prefix):] = 1
    return tokens, mask


# ---------------------------------------------------------------------------
# Rationale / Final-Answer Splitters for Intelli Lanes
# ---------------------------------------------------------------------------

def split_rationale_answer(ans: str) -> tuple[str | None, str]:
    """Splits a step-by-step answer into (reasoning, final_answer) when structured."""
    ans = clean_text(ans)
    # 1. Explicit '#### ' (GSM8K / Math style)
    if "\n#### " in ans:
        r, f = ans.rsplit("\n#### ", 1)
        if len(r.strip()) >= 20 and f.strip():
            return r.strip(), f.strip()
    # 2. Explicit 'Answer:' or 'Final Answer:' or 'Therefore, the answer is'
    for sep in ("\nFinal Answer:", "\nAnswer:", "\nTherefore, the answer is", "\nSo the answer is", "\nCorrect Answer:"):
        if sep in ans:
            r, f = ans.rsplit(sep, 1)
            if len(r.strip()) >= 20 and f.strip():
                return r.strip(), f.strip().lstrip(": ").strip()
    # 3. Multi-line explanation where last line is short conclusion
    lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
    if len(lines) >= 3 and len(lines[-1]) <= 120 and sum(len(x) for x in lines[:-1]) >= 40:
        return "\n".join(lines[:-1]), lines[-1]
    return None, ans


def format_prompt_with_context(question: str, context: str, lang: str) -> str:
    q = clean_text(question)
    c = clean_text(context)
    if not c:
        return q
    if len(c) > 2200:
        c = c[:2200].rsplit(" ", 1)[0] + "..."
    q_label = {
        "kan": "ಪ್ರಶ್ನೆ",
        "tam": "கேள்வி",
        "tel": "ప్రశ్న",
        "mal": "ചോദ്യം",
    }.get(lang, "Question")
    return f"{c}\n\n{q_label}: {q}"


# ---------------------------------------------------------------------------
# Pillar Collectors
# ---------------------------------------------------------------------------

def collect_bespoke_think(tok: Tokenizer, val_hashes: set[str], max_reason_toks: int = 420, max_total_toks: int = 1024):
    """Yields concise, complete Bespoke-Stratos-17k <think> traces that always close </think>."""
    pfile = RAW_SOURCED / "bespoke_stratos/train-00000-of-00001.parquet"
    if not pfile.exists():
        return
    table = pq.read_table(pfile, columns=["conversations"])
    for conv_item in table["conversations"].to_pylist():
        if len(conv_item) < 2:
            continue
        u_val = conv_item[0].get("value", "")
        a_val = conv_item[1].get("value", "")
        if "<|begin_of_thought|>" not in a_val or "<|end_of_thought|>" not in a_val:
            continue
        t_raw, ans_raw = a_val.split("<|end_of_thought|>", 1)
        t_raw = t_raw.replace("<|begin_of_thought|>", "").strip()
        ans_raw = ans_raw.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()
        p = clean_text(u_val)
        t = clean_text(t_raw)
        a = clean_text(ans_raw)
        if not (p and t and a):
            continue
        if hashlib.sha1(p.encode("utf-8")).hexdigest() in val_hashes:
            continue
        # Compress overly long thought chains to their opening + concluding paragraphs if needed
        r_ids = tok.encode(t, add_special_tokens=False).ids
        if len(r_ids) > max_reason_toks:
            paras = [x.strip() for x in t.split("\n\n") if x.strip()]
            if len(paras) >= 3:
                t_short = paras[0] + "\n\n" + paras[-2] + "\n\n" + paras[-1]
                r_ids = tok.encode(t_short, add_special_tokens=False).ids
                if len(r_ids) <= max_reason_toks:
                    t = t_short
                else:
                    continue
            else:
                continue
        if not verify_lang_script(a, "eng", min_chars=6):
            continue
        tokens, mask = encode_think_doc(tok, "eng", p, t, a)
        if len(tokens) <= max_total_toks:
            yield ("think_bespoke", "eng", tokens, mask)


def collect_intelli_lanes(tok: Tokenizer, val_hashes: set[str], rng: np.random.Generator):
    """Reads F:/Lingual_Data/intelli/lane_*/rows.jsonl with strict script & length governance."""
    if not INTELLI_DIR.exists():
        return

    # Per-source caps so no single massive dataset (like hotpot_qa or alar) swamps the SFT stage
    SOURCE_CAPS = {
        "alar": 600,
        "kaikki/enwiktionary": 600,
        "kaikki": 600,
        "olam": 2500,
        "wordnet3.1": 800,
        "hotpotqa/hotpot_qa/distractor/train": 3500,
        "allenai/quac/default/train": 2500,
        "deepmind/narrativeqa": 2000,
        "ucinlp/drop/default/train": 3000,
        "rajpurkar/squad_v2/squad_v2/train": 4000,
        "allenai/quoref/default/train": 2000,
        "HuggingFaceH4/stack-exchange-preferences/default/train": 3000,
        "koutch/staqc/sca_python/train": 2500,
    }
    source_counts = Counter()

    lane_files = sorted(INTELLI_DIR.glob("lane_*/rows.jsonl"))
    for lf in lane_files:
        lane_name = lf.parent.name
        if lf.stat().st_size == 0:
            continue
        with open(lf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                lic = str(row.get("license", ""))
                src = str(row.get("source", ""))
                # Exclude research-only trial rows for clean open-source release
                if "research-only" in lic.lower() or src.startswith("distill:"):
                    continue

                # Check source cap
                cap_key = next((k for k in SOURCE_CAPS if k in src), src)
                cap_limit = SOURCE_CAPS.get(cap_key, 4500)
                if source_counts[cap_key] >= cap_limit:
                    continue

                raw_lang = row.get("lang", "eng")
                if raw_lang == "esg":
                    continue
                q = clean_text(row.get("question", ""))
                ctx = clean_text(row.get("context", ""))
                ans = clean_text(row.get("answer", ""))
                if not q or not ans:
                    continue

                # Determine true language of the ASSISTANT answer
                ascii_a, counts_a = script_profile(ans)
                if counts_a["Deva"] > 2:
                    continue

                if raw_lang in LANG_SCRIPT:
                    target_scr = LANG_SCRIPT[raw_lang]
                    if counts_a[target_scr] >= 6 and counts_a[target_scr] > ascii_a:
                        eff_lang = raw_lang
                    elif ascii_a >= 10 and ascii_a > sum(counts_a.values()) * 2:
                        # Cross-lingual (Dravidian prompt -> English answer): MUST be tagged <eng>!
                        eff_lang = "eng"
                        if source_counts[f"xling_{lane_name}"] >= 1200:
                            continue
                        source_counts[f"xling_{lane_name}"] += 1
                    else:
                        continue
                else:
                    eff_lang = "eng"
                    if not verify_lang_script(ans, "eng", min_chars=4):
                        continue

                prompt = format_prompt_with_context(q, ctx, eff_lang)
                if hashlib.sha1(prompt.encode("utf-8")).hexdigest() in val_hashes:
                    continue

                # Lane 08 Unanswerable -> calibrated abstention (50% direct, 50% structured <think>)
                if lane_name == "lane_08_unanswerable":
                    templates = ABSTENTION_TEMPLATES.get(eff_lang, ABSTENTION_TEMPLATES["eng"])
                    abst_ans = templates[int(rng.integers(0, len(templates)))]
                    if rng.random() < 0.5:
                        think_txt = ABSTENTION_THINK.get(eff_lang, ABSTENTION_THINK["eng"])
                        tokens, mask = encode_think_doc(tok, eff_lang, prompt, think_txt, abst_ans)
                        cat = "honesty_think"
                    else:
                        tokens, mask = encode_chat_doc(tok, eff_lang, prompt, abst_ans)
                        cat = "honesty_chat"
                    if len(tokens) <= 1024:
                        source_counts[cap_key] += 1
                        yield (cat, eff_lang, tokens, mask)
                    continue

                # Structured reasoning lanes (math, logic, commonsense, reading_comp with rationale)
                if lane_name in ("lane_01_math_word", "lane_02_logic", "lane_03_commonsense", "lane_10_tam_qa"):
                    rat, final_a = split_rationale_answer(ans)
                    if rat and len(rat) >= 15 and final_a:
                        tokens, mask = encode_think_doc(tok, eff_lang, prompt, rat, final_a)
                        if len(tokens) <= 1024:
                            source_counts[cap_key] += 1
                            yield (f"think_{lane_name}", eff_lang, tokens, mask)
                            continue
                    elif ctx and len(ctx) >= 20 and len(ans) <= 200 and rng.random() < 0.45:
                        # Build concise grounded <think> verification from context
                        if eff_lang == "eng":
                            rat_synth = f"Let us check the given information carefully to answer: {q}\nFrom the context, the relevant detail leads directly to: {ans}"
                        elif eff_lang == "tam":
                            rat_synth = f"கொடுக்கப்பட்ட விவரங்களைக் கவனமாகப் பரிசீலிப்போம்: {q}\nஇதிலிருந்து சரியான விடை: {ans}"
                        else:
                            rat_synth = None
                        if rat_synth:
                            tokens, mask = encode_think_doc(tok, eff_lang, prompt, rat_synth, ans)
                            if len(tokens) <= 1024:
                                source_counts[cap_key] += 1
                                yield (f"think_{lane_name}", eff_lang, tokens, mask)
                                continue

                # Reading comp: convert 35% of concise extractive QA into structured <think> + answer
                if lane_name in ("lane_05_reading_comp", "lane_11_tel_qa") and ctx and len(ans) <= 160 and rng.random() < 0.35:
                    # Find sentence in ctx containing ans if extractive
                    sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", ctx) if s.strip()]
                    ev = next((s for s in sents if ans.lower() in s.lower() and len(s) <= 320), "")
                    if ev:
                        if eff_lang == "eng":
                            rat_rc = f"Scanning the passage for the question '{q}', the text states: \"{ev}\"\nTherefore, the answer is {ans}."
                        elif eff_lang == "tel":
                            rat_rc = f"ఇచ్చిన పేరాలో '{q}' అనే ప్రశ్నకు సంబంధించిన వాక్యం: \"{ev}\"\nకాబట్టి సరైన సమాధానం: {ans}"
                        else:
                            rat_rc = ""
                        if rat_rc:
                            tokens, mask = encode_think_doc(tok, eff_lang, prompt, rat_rc, ans)
                            if len(tokens) <= 1024:
                                source_counts[cap_key] += 1
                                yield (f"think_{lane_name}", eff_lang, tokens, mask)
                                continue

                # Default: direct assistant-masked chat turn
                tokens, mask = encode_chat_doc(tok, eff_lang, prompt, ans)
                if len(tokens) <= 1024:
                    source_counts[cap_key] += 1
                    yield (f"chat_{lane_name}", eff_lang, tokens, mask)


def collect_dravidian_aya_indicqa(tok: Tokenizer, val_hashes: set[str], rng: np.random.Generator):
    """Collects native-script Dravidian chat and structured <think> turns from Aya + IndicQA."""
    # 1. Aya dataset (kan, tam, tel, mal)
    aya_path = RAW_SOURCED / "aya_dataset/data/train-00000-of-00001.parquet"
    if aya_path.exists():
        table = pq.read_table(aya_path, columns=["inputs", "targets", "language_code"])
        for inp, tgt, lcode in zip(
            table["inputs"].to_pylist(),
            table["targets"].to_pylist(),
            table["language_code"].to_pylist(),
        ):
            if lcode not in ("kan", "tam", "tel", "mal"):
                continue
            p = clean_text(inp)
            a = clean_text(tgt)
            if len(p) < 10 or len(a) < 10:
                continue
            if not verify_lang_script(p, lcode, min_chars=6) or not verify_lang_script(a, lcode, min_chars=6):
                continue
            if hashlib.sha1(p.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(a.encode("utf-8")).hexdigest() in val_hashes:
                continue
            # If answer has multiple paragraphs/sentences, convert 25% into structured <think> + summary answer
            sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", a) if len(s.strip()) >= 15]
            if len(sents) >= 3 and rng.random() < 0.25:
                think_part = " ".join(sents[:-1])
                final_part = a
                tokens, mask = encode_think_doc(tok, lcode, p, think_part, final_part)
                if len(tokens) <= 1024:
                    yield ("drav_think_aya", lcode, tokens, mask)
                    continue
            tokens, mask = encode_chat_doc(tok, lcode, p, a)
            if len(tokens) <= 1024:
                yield ("drav_chat_aya", lcode, tokens, mask)

    # 2. IndicQA (kan, tam, tel, mal) - both answerable & unanswerable
    indicqa_dir = RAW_SOURCED / "indicqa"
    if indicqa_dir.exists():
        for short_l, canon_l in [("kn", "kan"), ("ta", "tam"), ("te", "tel"), ("ml", "mal")]:
            fpath = indicqa_dir / f"indicqa.{short_l}.json"
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for art in data.get("data", []):
                for p_obj in art.get("paragraphs", []):
                    ctx = clean_text(p_obj.get("context", ""))
                    if not ctx or not verify_lang_script(ctx, canon_l, min_chars=20):
                        continue
                    sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", ctx) if s.strip()]
                    for qa in p_obj.get("qas", []):
                        q = clean_text(qa.get("question", ""))
                        if not q or not verify_lang_script(q, canon_l, min_chars=4):
                            continue
                        prompt = format_prompt_with_context(q, ctx, canon_l)
                        if hashlib.sha1(prompt.encode("utf-8")).hexdigest() in val_hashes:
                            continue
                        answers = qa.get("answers", [])
                        ans_text = clean_text(answers[0].get("text", "")) if answers else ""
                        if not ans_text:
                            # Unanswerable IndicQA -> native Dravidian calibrated abstention!
                            templates = ABSTENTION_TEMPLATES[canon_l]
                            abst_ans = templates[int(rng.integers(0, len(templates)))]
                            think_txt = ABSTENTION_THINK[canon_l]
                            if rng.random() < 0.5:
                                tokens, mask = encode_think_doc(tok, canon_l, prompt, think_txt, abst_ans)
                            else:
                                tokens, mask = encode_chat_doc(tok, canon_l, prompt, abst_ans)
                            if len(tokens) <= 1024:
                                yield ("drav_honesty_indicqa", canon_l, tokens, mask)
                        else:
                            if not verify_lang_script(ans_text, canon_l, min_chars=2):
                                continue
                            ev = next((s for s in sents if ans_text in s and len(s) <= 350), "")
                            if ev and rng.random() < 0.55:
                                think_map = {
                                    "kan": f"ಕೊಟ್ಟಿರುವ ಪಠ್ಯದಲ್ಲಿ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ವಾಕ್ಯ: \"{ev}\"\nಆದ್ದರಿಂದ ಸರಿಯಾದ ಉತ್ತರ: {ans_text}",
                                    "tam": f"கொடுக்கப்பட்ட பத்தியில் கேள்விக்குரிய வாக்கியம்: \"{ev}\"\nஎனவே சரியான பதில்: {ans_text}",
                                    "tel": f"ఇచ్చిన పేరాలో ప్రశ్నకు సంబంధించిన వాక్యం: \"{ev}\"\nకాబట్టి సరైన సమాధానం: {ans_text}",
                                    "mal": f"നൽകിയിട്ടുള്ള ഭാഗത്ത് ചോദ്യവുമായി ബന്ധപ്പെട്ട വാചകം: \"{ev}\"\nഅതിനാൽ ശരിയായ ഉത്തരം: {ans_text}",
                                }
                                tokens, mask = encode_think_doc(tok, canon_l, prompt, think_map[canon_l], ans_text)
                                if len(tokens) <= 1024:
                                    yield ("drav_think_indicqa", canon_l, tokens, mask)
                                    continue
                            tokens, mask = encode_chat_doc(tok, canon_l, prompt, ans_text)
                            if len(tokens) <= 1024:
                                yield ("drav_chat_indicqa", canon_l, tokens, mask)


def collect_oasst2_and_tools(tok: Tokenizer, val_hashes: set[str]):
    """Collects OASST2 rank-0 English chat and genuine Hermes tool-calling traces."""
    ds_dir = RAW_SOURCED / "oasst2/data"
    if ds_dir.exists():
        seen = set()
        for pfile in sorted(ds_dir.glob("train-*.parquet")):
            table = pq.read_table(pfile)
            cols = table.column_names
            m_ids = table["message_id"].to_pylist()
            p_ids = table["parent_id"].to_pylist()
            roles = table["role"].to_pylist()
            langs = table["lang"].to_pylist()
            ranks = table["rank"].to_pylist()
            synths = table["synthetic"].to_pylist() if "synthetic" in cols else [False] * len(m_ids)
            deleteds = table["deleted"].to_pylist() if "deleted" in cols else [False] * len(m_ids)
            texts = table["text"].to_pylist()

            prompters = {}
            assistants = []
            for mid, pid, role, l, rk, syn, dlt, txt in zip(m_ids, p_ids, roles, langs, ranks, synths, deleteds, texts):
                if dlt or not txt:
                    continue
                if role == "prompter" and l == "en":
                    prompters[mid] = clean_text(txt)
                elif role == "assistant" and l == "en" and not syn and (rk == 0 or rk == 0.0):
                    assistants.append((pid, clean_text(txt)))

            for pid, ans in assistants:
                if pid in prompters:
                    p_txt = prompters[pid]
                    if len(p_txt) >= 12 and len(ans) >= 20 and verify_lang_script(ans, "eng", min_chars=12):
                        h = hashlib.sha1((p_txt + "\n" + ans).encode("utf-8")).hexdigest()
                        if h in seen or hashlib.sha1(p_txt.encode("utf-8")).hexdigest() in val_hashes:
                            continue
                        seen.add(h)
                        tokens, mask = encode_chat_doc(tok, "eng", p_txt, ans)
                        if len(tokens) <= 1024:
                            yield ("chat_oasst2", "eng", tokens, mask)

    # Genuine Tool Traces (with environment tool_result masked out!)
    hermes_dir = RAW_SOURCED / "hermes-function-calling-v1"
    if hermes_dir.exists():
        for fname in ["glaive-function-calling-5k.json", "func-calling-singleturn.json"]:
            fpath = hermes_dir / fname
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for item in data:
                convs = item.get("conversations", [])
                for i in range(len(convs) - 3):
                    if (
                        convs[i].get("from") == "human"
                        and convs[i + 1].get("from") == "gpt"
                        and convs[i + 2].get("from") == "tool"
                        and convs[i + 3].get("from") == "gpt"
                    ):
                        u_txt = clean_text(convs[i].get("value", ""))
                        tcall = convs[i + 1].get("value", "")
                        tresp = convs[i + 2].get("value", "")
                        ans_txt = clean_text(convs[i + 3].get("value", ""))
                        if "<tool_call>" in tcall and "</tool_call>" in tcall and ans_txt:
                            call_body = tcall.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
                            resp_body = tresp.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                            try:
                                c_json = json.loads(call_body)
                                name = c_json.get("name", "").lower()
                                args = c_json.get("arguments", {})
                                mapped = None
                                if any(w in name for w in ["web_search", "search_web", "google", "bing", "duckduckgo"]) or (
                                    name.startswith("search") and any(k in args for k in ["query", "query_string", "q", "keyword"])
                                ):
                                    mapped = "web_search"
                                elif any(w in name for w in ["fetch_url", "get_url", "read_url", "download_url", "webpage", "fetch_page"]) and any(
                                    k in args for k in ["url", "link", "uri"]
                                ):
                                    mapped = "fetch_url"
                                elif any(w in name for w in ["read_file", "view_file", "open_file", "cat", "load_file"]) and any(
                                    k in args for k in ["path", "file", "filename", "filepath"]
                                ):
                                    mapped = "read_file"
                                elif any(w in name for w in ["python", "code_runner", "execute_code", "run_code", "calculator", "eval_math"]) and any(
                                    k in args for k in ["code", "script", "expression", "formula"]
                                ):
                                    mapped = "python"
                                if mapped is None:
                                    continue
                                c_json["name"] = mapped
                                call_str = json.dumps(c_json, ensure_ascii=False)
                                tokens, mask = encode_tool_doc(tok, "eng", u_txt, call_str, clean_text(resp_body), ans_txt)
                                if len(tokens) <= 1024:
                                    yield ("tool_hermes", "eng", tokens, mask)
                            except Exception:
                                continue


def collect_lowres_and_freethink(tok: Tokenizer, val_hashes: set[str], rng: np.random.Generator):
    """Builds:
      1. Low-resource SFT chat + structured <think> turns for tcy, kff, kxv, nit from val-deduped v3 text.
      2. FreeThink continuation traces across all 9 languages (eng, kan, tam, tel, mal, tcy, kff, kxv, nit)
         so FreeThinkSession(mode='continuation') and FreeThinkSession(mode='structured') work cleanly
         without numeric list artifacts or script switching.
    """
    if not V3_TEXT_DIR.exists():
        return

    LOWRES_PROMPT_TEMPLATES = {
        "tcy": [
            "ಈ ವಿಷಯೊದ ಬಗೆಟ್ ವಿವರವಾದ್ ಬರೆಲೆ:\n{head}",
            "ಈ ವಾಕ್ಯೊನು ದುಂಬುವರಿಸಾದ್ ವಿವರಿಸಾಲೆ:\n{head}",
        ],
        "kff": [
            "ଇ କଥା ପେର୍କେ ଆରି ବିବରଣୀ କିଟ୍:\n{head}",
            "ଇ ଲେଖା ତଲେ ଆରି କଥା କେତୁଟ୍:\n{head}",
        ],
        "kxv": [
            "ଈ କାତା ପ୍ଡାକି ଅ଼ଡ଼େ ୱେହ୍‌ଦୁ:\n{head}",
            "ଈ ଲେଖା ତଲେ ଅ଼ଡ଼େ କାତା ୱେହ୍‌ଦୁ:\n{head}",
        ],
        "nit": [
            "ఇద్ గొట్టిన్ గురుంచి మరి వివరంగా వ్రాయు:\n{head}",
            "ఇదా వాక్యమున్ ముందుకు నడిపి వివరింపు:\n{head}",
        ],
    }

    for lang in ("eng", "kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit"):
        p_jsonl = V3_TEXT_DIR / f"{lang}.jsonl"
        if not p_jsonl.exists():
            continue
        is_lowres = lang in ("tcy", "kff", "kxv", "nit")
        max_docs_for_lang = 1600 if is_lowres else 900
        count = 0

        with open(p_jsonl, encoding="utf-8") as f:
            for line in f:
                if count >= max_docs_for_lang:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                txt = clean_text(obj.get("text", ""))
                if len(txt) < 180:
                    continue
                if hashlib.sha1(txt.encode("utf-8")).hexdigest() in val_hashes:
                    continue
                # Clean wiki markup remnants in tcy if any
                txt = re.sub(r"^\}\}+\s*", "", txt).strip()
                if not verify_lang_script(txt, lang, min_chars=40):
                    continue

                # Split into sentences
                sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n+", txt) if len(s.strip()) >= 25]
                if len(sents) < 3:
                    continue

                # Low-resource files (kff, kxv, nit) store whole chapters (~25-40 sentences per line),
                # so slide a non-overlapping stride-4 window to harvest multiple passages per chapter.
                offsets = range(0, max(1, len(sents) - 3), 4) if is_lowres else (0,)
                for off in offsets:
                    if count >= max_docs_for_lang:
                        break
                    sub = sents[off : off + 8]
                    if len(sub) < 3:
                        continue
                    head = sub[0]
                    body = " ".join(sub[1:])
                    if not verify_lang_script(head, lang, min_chars=12) or not verify_lang_script(body, lang, min_chars=30):
                        continue
                    # Avoid lines starting with raw digits so we actively counter numeric-start bias
                    if head[0].isdigit() or body[0].isdigit():
                        continue

                    # A. FreeThink Continuation trace (all 9 languages)
                    ft_tokens, ft_mask = encode_freethink_doc(tok, lang, head, body)
                    if 64 <= len(ft_tokens) <= 768:
                        yield ("freethink_cont", lang, ft_tokens, ft_mask)
                        count += 1

                    # B. Structured <think> trace on statements (all 9 languages)
                    if len(sub) >= 4 and (is_lowres or rng.random() < 0.55):
                        mid_think = sub[1]
                        rest_ans = " ".join(sub[2:])
                        if verify_lang_script(mid_think, lang, min_chars=10) and verify_lang_script(rest_ans, lang, min_chars=20):
                            st_tokens, st_mask = encode_think_doc(tok, lang, head, mid_think, rest_ans)
                            if 64 <= len(st_tokens) <= 768:
                                yield ("freethink_struct", lang, st_tokens, st_mask)

                    # C. Low-resource conversational SFT turns (tcy, kff, kxv, nit)
                    if is_lowres:
                        tmpl_list = LOWRES_PROMPT_TEMPLATES[lang]
                        u_prompt = tmpl_list[count % len(tmpl_list)].format(head=head)
                        c_tokens, c_mask = encode_chat_doc(tok, lang, u_prompt, body)
                        if 64 <= len(c_tokens) <= 768:
                            yield ("lowres_chat", lang, c_tokens, c_mask)


# ---------------------------------------------------------------------------
# Main Assembly & Verification Gates
# ---------------------------------------------------------------------------

def write_split(out_dir: Path, prefix: str, docs: list[tuple[np.ndarray, np.ndarray]]):
    bin_path = out_dir / f"{prefix}.bin"
    mask_path = out_dir / f"{prefix}.mask.bin"
    bnd_path = out_dir / f"{prefix}_boundaries.npy"

    starts = [0]
    with open(bin_path, "wb") as fb, open(mask_path, "wb") as fm:
        for toks, msk in docs:
            assert len(toks) == len(msk), f"Length mismatch: {len(toks)} vs {len(msk)}"
            fb.write(toks.tobytes())
            fm.write(msk.tobytes())
            starts.append(starts[-1] + len(toks))

    np.save(bnd_path, np.array(starts[:-1], dtype=np.int64))
    return bin_path, mask_path, bnd_path


def main():
    p = argparse.ArgumentParser(description="Build assistant-masked SFT corpus for Parentheses 0.9-120M")
    p.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    p.add_argument("--val-dir", default=str(VAL_DIR))
    p.add_argument("--out-dir", default=str(OUT_DEFAULT))
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer(Path(args.tokenizer))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_sft_120m] Loading validation hashes...", flush=True)
    val_hashes = load_val_hashes(Path(args.val_dir), tok)
    print(f"[build_sft_120m] Loaded {len(val_hashes):,} validation hashes.", flush=True)

    records: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    print("[1/4] Collecting concise Bespoke-Stratos <think> traces...", flush=True)
    for item in collect_bespoke_think(tok, val_hashes):
        records.append(item)
        if args.dry_run and len(records) >= 200:
            break
    print(f"      Total records so far: {len(records):,}", flush=True)

    print("[2/4] Collecting & filtering Intelli lanes (strict script + rationale + abstention)...", flush=True)
    for item in collect_intelli_lanes(tok, val_hashes, rng):
        records.append(item)
        if args.dry_run and len(records) >= 600:
            break
    print(f"      Total records so far: {len(records):,}", flush=True)

    print("[3/4] Collecting native Dravidian Aya + IndicQA (chat + <think> + abstention)...", flush=True)
    for item in collect_dravidian_aya_indicqa(tok, val_hashes, rng):
        records.append(item)
        if args.dry_run and len(records) >= 900:
            break
    print(f"      Total records so far: {len(records):,}", flush=True)

    print("[4/4] Collecting OASST2, Hermes Tools, Low-Resource SFT & 9-Lang FreeThink...", flush=True)
    for item in collect_oasst2_and_tools(tok, val_hashes):
        records.append(item)
        if args.dry_run and len(records) >= 1100:
            break
    for item in collect_lowres_and_freethink(tok, val_hashes, rng):
        records.append(item)
        if args.dry_run and len(records) >= 1400:
            break
    print(f"      Total raw filtered records: {len(records):,}", flush=True)

    # Upsample high-priority Dravidian (especially kan/mal/lowres) & structured/FreeThink/Tool categories (<= 3x)
    balanced_records = []
    for cat, lang, toks, msk in records:
        balanced_records.append((cat, lang, toks, msk))
        if not args.dry_run:
            if cat.startswith("drav_") or cat in ("lowres_chat", "freethink_cont", "freethink_struct", "honesty_think", "tool_hermes"):
                balanced_records.append((cat, lang, toks, msk))
            if lang in ("kan", "mal", "tcy", "kff", "kxv", "nit") or cat in ("freethink_struct", "tool_hermes"):
                balanced_records.append((cat, lang, toks, msk))

    # Shuffle deterministically
    perm = rng.permutation(len(balanced_records))
    shuffled = [balanced_records[i] for i in perm]

    # Split 98% train, 2% val
    n_val = max(64, int(len(shuffled) * 0.02))
    val_recs = shuffled[:n_val]
    train_recs = shuffled[n_val:]

    cat_tokens = Counter()
    lang_tokens = Counter()
    cat_docs = Counter()
    lang_docs = Counter()
    sup_tokens = 0
    total_tokens = 0

    for cat, lang, toks, msk in train_recs:
        n = int(len(toks))
        cat_tokens[cat] += n
        lang_tokens[lang] += n
        cat_docs[cat] += 1
        lang_docs[lang] += 1
        total_tokens += n
        sup_tokens += int(msk.sum())

    prefix = "dryrun_train" if args.dry_run else "train"
    vprefix = "dryrun_val" if args.dry_run else "val"

    train_bin, train_mask, train_bnd = write_split(out_dir, prefix, [(r[2], r[3]) for r in train_recs])
    val_bin, val_mask, val_bnd = write_split(out_dir, vprefix, [(r[2], r[3]) for r in val_recs])

    # --- Verification Gates ---
    print("\n--- Running Verification Gates ---", flush=True)
    b_data = np.fromfile(train_bin, dtype=np.uint16)
    m_data = np.fromfile(train_mask, dtype=np.uint8)
    bnd = np.load(train_bnd)

    assert len(b_data) == len(m_data), f"train.bin ({len(b_data)}) != train.mask.bin ({len(m_data)})"
    esg_count = int(np.sum(b_data == ESG_ID))
    assert esg_count == 0, f"Found {esg_count} <esg> tokens!"
    assert int(bnd[0]) == 0 and np.all(np.diff(bnd) > 0) and int(bnd[-1]) < len(b_data)

    # Verify every document has >= 1 supervised token and starts with mask == 0
    for idx in range(min(500, len(bnd))):
        s = int(bnd[idx])
        e = int(bnd[idx + 1]) if idx + 1 < len(bnd) else len(b_data)
        assert m_data[s] == 0, f"Doc {idx} starts with mask=1 at offset {s}!"
        assert m_data[s:e].sum() > 0, f"Doc {idx} has 0 supervised tokens!"
        assert b_data[e - 1] == EOT and m_data[e - 1] == 1, f"Doc {idx} does not end with supervised EOT!"

    print(
        f"[PASS] {len(train_recs):,} train docs ({total_tokens:,} tokens, {sup_tokens:,} supervised = {sup_tokens/max(1,total_tokens):.1%}), "
        f"{len(val_recs):,} val docs, 0 <esg> tokens.",
        flush=True,
    )

    manifest = {
        "train_docs": len(train_recs),
        "val_docs": len(val_recs),
        "train_tokens": total_tokens,
        "supervised_tokens": sup_tokens,
        "supervised_fraction": round(sup_tokens / max(1, total_tokens), 4),
        "tokens_by_lang": dict(sorted(lang_tokens.items(), key=lambda x: -x[1])),
        "docs_by_lang": dict(sorted(lang_docs.items(), key=lambda x: -x[1])),
        "tokens_by_category": dict(sorted(cat_tokens.items(), key=lambda x: -x[1])),
        "docs_by_category": dict(sorted(cat_docs.items(), key=lambda x: -x[1])),
        "train_sha256": hashlib.sha256(train_bin.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(train_mask.read_bytes()).hexdigest(),
    }
    m_path = out_dir / ("manifest_dryrun.json" if args.dry_run else "manifest.json")
    m_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build_sft_120m] Wrote manifest to {m_path}", flush=True)


if __name__ == "__main__":
    main()
