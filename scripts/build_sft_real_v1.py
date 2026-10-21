#!/usr/bin/env python3
"""Build the clean conversational + reasoning SFT dataset (processed_sft_real_v1).

Composition:
  1. Math & Step-by-Step Reasoning (35%):
     - microsoft/orca-math-word-problems-200k (step-by-step word problems)
     - meta-math/MetaMathQA (GSM8K/MATH derivations with <think>...</think>)
     - Synthetic arithmetic & logic drills in English & Dravidian (arithmetic, percentages, sequences, logic)
  2. Multi-turn Dialogue & Persona (30%):
     - HuggingFaceTB/smol-smoltalk (clean everyday chat, coding, constraints, summaries)
     - Parentheses 0.9 identity & persona dialogues (Joel Anthony, Dravidian recurrent model)
  3. Native Dravidian Reasoning & Dialogue (20%):
     - Aya dataset (human-annotated instruction following in kan, tam, tel, mal)
     - IndicQA (native Dravidian reading comprehension with evidence)
     - Native Dravidian arithmetic & reasoning with native script thinking traces
  4. Pretraining Replay (15%):
     - Sampled from data/processed_v5/train.bin across Dravidian and English
     - Full causal loss mask (mask=1) to prevent catastrophic forgetting

Outputs:
  - data/processed_sft_real_v1/train.bin            (uint16 token IDs)
  - data/processed_sft_real_v1/train.mask.bin       (uint8 assistant loss mask: 1=supervised, 0=ignored)
  - data/processed_sft_real_v1/train_boundaries.npy (int64 document-start offsets)
  - data/processed_sft_real_v1/manifest.json

Usage:
  .venv\\Scripts\\python.exe scripts\\build_sft_real_v1.py
  .venv\\Scripts\\python.exe scripts\\build_sft_real_v1.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
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
}
ESG_ID = 11  # Excluded / banned (Aheri Gondi Wycliffe copyright rule)
SYSTEM_TAG = 12
USER_TAG = 13
ASSISTANT_TAG = 14
THINK_START = 15
THINK_END = 16
TOOL_START = 17
TOOL_END = 18

OUT_DIR = REPO / "data" / "processed_sft_real_v1"
RAW_SFT = REPO / "data" / "raw" / "sft_real"
RAW_MANUAL = REPO / "data" / "raw" / "manual" / "raw-sourced"
V5_DIR = REPO / "data" / "processed_v5"
TOKENIZER_PATH = REPO / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"


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


def encode_multiturn_doc(tok: Tokenizer, lang: str, messages: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray] | None:
    l_tag = TAG.get(lang, TAG["eng"])
    tok_seq: list[int] = []
    mask_seq: list[int] = []
    
    # Optional leading system prompt
    for msg in messages:
        role = msg.get("role", "")
        content = strip_internal_tags(clean_text(msg.get("content", "")))
        if not content:
            continue
        c_ids = tok.encode(content, add_special_tokens=False).ids
        if role == "system":
            seg = [SYSTEM_TAG] + c_ids
            tok_seq.extend(seg)
            mask_seq.extend([0] * len(seg))
        elif role == "user":
            seg = [USER_TAG, l_tag] + c_ids + [ASSISTANT_TAG]
            tok_seq.extend(seg)
            mask_seq.extend([0] * len(seg))
        elif role == "assistant":
            seg = c_ids + [EOT]
            tok_seq.extend(seg)
            mask_seq.extend([1] * len(seg))
            
    if not tok_seq or sum(mask_seq) == 0:
        return None
    return np.array(tok_seq, dtype=np.uint16), np.array(mask_seq, dtype=np.uint8)


def split_math_reasoning(ans: str) -> tuple[str, str]:
    """Splits a math answer into (reasoning, final_answer)."""
    ans = clean_text(ans)
    if "\nThe answer is: " in ans:
        r, a = ans.rsplit("\nThe answer is: ", 1)
        return r.strip(), f"The answer is {a.strip()}."
    if "\n#### " in ans:
        r, a = ans.rsplit("\n#### ", 1)
        return r.strip(), f"The answer is {a.strip()}."
    for sep in ("\nTherefore, the answer is", "\nSo the answer is", "\nFinal Answer:"):
        if sep in ans:
            r, a = ans.rsplit(sep, 1)
            return r.strip(), f"The answer is {a.strip().lstrip(': ')}."
    lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
    if len(lines) >= 2 and len(lines[-1]) <= 100:
        return "\n".join(lines[:-1]), lines[-1]
    return ans, ans


# -----------------------------------------------------------------------------
# Collectors
# -----------------------------------------------------------------------------

def collect_orca_math(tok: Tokenizer, max_docs: int = 40000):
    p = RAW_SFT / "microsoft__orca-math-word-problems-200k" / "data" / "train-00000-of-00001.parquet"
    if not p.exists():
        print(f"[skip] orca-math not found at {p}")
        return
    print(f"[orca-math] reading {p}...")
    df = pd.read_parquet(p)
    indices = list(range(len(df)))
    random.Random(42).shuffle(indices)
    count = 0
    for idx in indices:
        if count >= max_docs:
            break
        row = df.iloc[idx]
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if len(q) < 10 or len(a) < 10:
            continue
        reasoning, final_ans = split_math_reasoning(a)
        if len(reasoning) >= 20 and reasoning != final_ans:
            tokens, mask = encode_think_doc(tok, "eng", q, reasoning, final_ans)
        else:
            tokens, mask = encode_chat_doc(tok, "eng", q, a)
        if len(tokens) > 2048:
            continue
        count += 1
        yield ("math_orca", "eng", tokens, mask)
    print(f"[orca-math] collected {count} docs")


def collect_metamath(tok: Tokenizer, max_docs: int = 40000):
    p = RAW_SFT / "meta-math__MetaMathQA" / "MetaMathQA-395K.json"
    if not p.exists():
        print(f"[skip] metamath not found at {p}")
        return
    print(f"[metamath] reading {p}...")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    indices = list(range(len(data)))
    random.Random(42).shuffle(indices)
    count = 0
    for idx in indices:
        if count >= max_docs:
            break
        row = data[idx]
        q = str(row.get("query", "")).strip()
        resp = str(row.get("response", "")).strip()
        if len(q) < 10 or len(resp) < 10:
            continue
        reasoning, final_ans = split_math_reasoning(resp)
        if len(reasoning) >= 20:
            tokens, mask = encode_think_doc(tok, "eng", q, reasoning, final_ans)
        else:
            tokens, mask = encode_chat_doc(tok, "eng", q, resp)
        if len(tokens) > 2048:
            continue
        count += 1
        yield ("math_metamath", "eng", tokens, mask)
    print(f"[metamath] collected {count} docs")


def collect_synthetic_math_and_logic(tok: Tokenizer, total_samples: int = 25000):
    """Generates verified arithmetic, percentages, sequences, word problems, and logic in English & Dravidian."""
    rng = random.Random(1337)
    
    # Language templates for math
    templates = {
        "eng": {
            "add": "What is {a} + {b}?",
            "sub": "What is {a} - {b}?",
            "mul": "What is {a} * {b}?",
            "div": "What is {a} / {b}?",
            "pct": "What is {p}% of {n}?",
            "sq": "What is {a} squared?",
            "sheep": "A farmer has {a} sheep. All but {b} run away. How many remain?",
            "overtake": "If you overtake the person in second place during a race, what position are you in?",
            "iron_cotton": "Which is heavier: 1 kg of iron or 1 kg of cotton? Answer briefly.",
            "prime": "What is the smallest prime number?",
            "after_num": "Reply with only the number that comes after {a}.",
            "before_num": "Reply with only the number that comes before {a}.",
            "linear": "Solve for x: {a}x + {b} = {c}.",
            "clock": "If a clock shows {h}:{m:02d}, what is the smaller angle between its hour and minute hands?",
            "day": "If yesterday was {n} days before {day}, what day is today?",
        },
        "kan": {
            "add": "{a} + {b} ಎಷ್ಟು?",
            "sub": "{a} - {b} ಎಷ್ಟು?",
            "mul": "{a} x {b} ಎಷ್ಟು?",
            "div": "{a} / {b} ಎಷ್ಟು?",
            "pct": "{n} ರ {p}% ಎಷ್ಟು?",
            "sq": "{a} ರ ವರ್ಗ (square) ಎಷ್ಟು?",
            "sheep": "ಒಬ್ಬ ರೈತನ ಬಳಿ {a} ಕುರಿಗಳಿವೆ. {b} ನ್ನು ಹೊರತುಪಡಿಸಿ ಉಳಿದವೆಲ್ಲ ಓಡಿಹೋದವು. ಎಷ್ಟು ಕುರಿಗಳು ಉಳಿದಿವೆ?",
            "overtake": "ಓಟದ ಸ್ಪರ್ಧೆಯಲ್ಲಿ ನೀವು ಎರಡನೇ ಸ್ಥಾನದಲ್ಲಿರುವ ವ್ಯಕ್ತಿಯನ್ನು ಹಿಂದಿಕ್ಕಿದರೆ, ನೀವು ಯಾವ ಸ್ಥಾನದಲ್ಲಿರುತ್ತೀರಿ?",
            "iron_cotton": "೧ ಕೆಜಿ ಕಬ್ಬಿಣ ಅಥವಾ ೧ ಕೆಜಿ ಹತ್ತಿ - ಇವೆರಡರಲ್ಲಿ ಯಾವುದು ಹೆಚ್ಚು ತೂಕವಿರುತ್ತದೆ?",
            "prime": "ಅತ್ಯಂತ ಚಿಕ್ಕ ಅವಿಭಾಜ್ಯ ಸಂಖ್ಯೆ (smallest prime number) ಯಾವುದು?",
        },
        "tam": {
            "add": "{a} + {b} எவ்வளவு?",
            "sub": "{a} - {b} எவ்வளவு?",
            "mul": "{a} x {b} எவ்வளவு?",
            "div": "{a} / {b} எவ்வளவு?",
            "pct": "{n} இல் {p}% எவ்வளவு?",
            "sq": "{a} இன் வர்க்கம் (square) என்ன?",
            "sheep": "ஒரு விவசாயிடம் {a} ஆடுகள் உள்ளன. {b} ஆடுகளைத் தவிர மற்றவை ஓடிவிட்டன. எத்தனை ஆடுகள் மீதமுள்ளன?",
            "overtake": "ஒரு ஓட்டப்பந்தயத்தில் நீங்கள் இரண்டாம் இடத்தில் இருப்பவரை முந்தினால், நீங்கள் எந்த இடத்தில் இருப்பீர்கள்?",
            "iron_cotton": "1 கிலோ இரும்பு அல்லது 1 கிலோ பருத்தி - இதில் எது அதிக எடையுடையது?",
            "prime": "மிகச்சிறிய பகா எண் (smallest prime number) எது?",
        },
        "tel": {
            "add": "{a} + {b} ఎంత?",
            "sub": "{a} - {b} ఎంత?",
            "mul": "{a} x {b} ఎంత?",
            "div": "{a} / {b} ఎంత?",
            "pct": "{n} లో {p}% ఎంత?",
            "sq": "{a} యొక్క వర్గం (square) ఎంత?",
            "sheep": "ఒక రైతు వద్ద {a} గొర్రెలు ఉన్నాయి. {b} తప్ప మిగిలినవన్నీ పారిపోయాయి. ఎన్ని మిగిలాయి?",
            "overtake": "ఒక పరుగు పందెంలో మీరు రెండవ స్థానంలో ఉన్న వ్యక్తిని అధిగమిస్తే, మీరు ఏ స్థానంలో ఉంటారు?",
            "iron_cotton": "1 కిలో ఇనుము లేదా 1 కిలో దూది - వీటిలో ఏది ఎక్కువ బరువు ఉంటుంది?",
            "prime": "అత్యంత చిన్న ప్రధాన సంఖ్య (smallest prime number) ఏది?",
        },
        "mal": {
            "add": "{a} + {b} എത്രയാണ്?",
            "sub": "{a} - {b} എത്രയാണ്?",
            "mul": "{a} x {b} എത്രയാണ്?",
            "div": "{a} / {b} എത്രയാണ്?",
            "pct": "{n} ന്റെ {p}% എത്രയാണ്?",
            "sq": "{a} ന്റെ വർഗ്ഗം (square) എത്രയാണ്?",
            "sheep": "ഒരു കർഷകന്റെ പക്കൽ {a} ആടുകളുണ്ട്. {b} എണ്ണം ഒഴികെ ബാക്കിയെല്ലാം ഓടിപ്പോയി. എത്ര ആടുകൾ അവശേഷിക്കുന്നു?",
            "overtake": "ഒരു ഓട്ടമത്സരത്തിൽ നിങ്ങൾ രണ്ടാം സ്ഥാനത്തുള്ളയാളെ മറികടന്നാൽ, നിങ്ങൾ ഏത് സ്ഥാനത്താകും?",
            "iron_cotton": "1 കിലോ ഇരുമ്പാണോ 1 കിലോ പഞ്ഞിയാണോ കൂടുതൽ ഭാരമുള്ളത്?",
            "prime": "ഏറ്റവും ചെറിയ അഭാജ്യ സംഖ്യ (smallest prime number) ഏതാണ്?",
        },
    }

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    count = 0

    while count < total_samples:
        lang = rng.choice(["eng", "eng", "kan", "tam", "tel", "mal"])
        t = templates[lang]
        kind = rng.choice(["add", "sub", "mul", "div", "pct", "sq", "logic", "word_puzzle"])

        if kind == "add":
            a = rng.randint(5, 999)
            b = rng.randint(5, 999)
            ans = a + b
            q = t["add"].format(a=a, b=b)
            r = f"Calculating {a} + {b}:\nSum = {ans}"
            ans_str = f"{a} + {b} = {ans}"
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        elif kind == "sub":
            a = rng.randint(20, 999)
            b = rng.randint(1, a)
            ans = a - b
            q = t["sub"].format(a=a, b=b)
            r = f"Calculating {a} - {b}:\nDifference = {ans}"
            ans_str = f"{a} - {b} = {ans}"
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        elif kind == "mul":
            a = rng.randint(2, 50)
            b = rng.randint(2, 25)
            ans = a * b
            q = t["mul"].format(a=a, b=b)
            # Step by step derivation in think
            tens = (a // 10) * 10
            units = a % 10
            if tens > 0 and units > 0:
                r = f"To multiply {a} by {b}:\n{tens} * {b} = {tens*b}\n{units} * {b} = {units*b}\n{tens*b} + {units*b} = {ans}"
            else:
                r = f"{a} multiplied by {b} is {ans}."
            ans_str = f"{a} * {b} = {ans}"
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        elif kind == "div":
            b = rng.randint(2, 20)
            mult = rng.randint(2, 50)
            a = b * mult
            ans = mult
            q = t["div"].format(a=a, b=b)
            r = f"Dividing {a} by {b}:\n{b} * {ans} = {a}\nSo {a} / {b} = {ans}"
            ans_str = f"{a} / {b} = {ans}"
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        elif kind == "pct":
            base = rng.choice([20, 40, 50, 60, 80, 100, 120, 150, 200, 250, 500, 1000, 1500])
            p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
            val = (base * p) / 100
            val_str = str(int(val)) if val.is_integer() else f"{val:.2f}"
            q = t["pct"].format(p=p, n=base)
            r = f"{p}% of {base} is ({p} / 100) * {base} = {val_str}."
            ans_str = f"{p}% of {base} is {val_str}."
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        elif kind == "sq":
            a = rng.randint(2, 35)
            ans = a * a
            q = t["sq"].format(a=a)
            r = f"{a} squared means {a} * {a} = {ans}."
            ans_str = f"{a} squared is {ans}."
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        elif kind == "word_puzzle":
            puzzle_type = rng.choice(["sheep", "overtake", "iron_cotton", "prime"])
            if puzzle_type == "sheep":
                a = rng.randint(12, 30)
                b = rng.randint(3, a - 2)
                q = t["sheep"].format(a=a, b=b)
                r = f"The phrase 'all but {b} run away' means that exactly {b} sheep did not run away and remain."
                ans_str = f"{b} remain."
            elif puzzle_type == "overtake":
                q = t["overtake"]
                r = "If you overtake the person in second place, you take their spot. Therefore, you are now in second place."
                ans_str = "You are in second place."
            elif puzzle_type == "iron_cotton":
                q = t["iron_cotton"]
                r = "Both 1 kg of iron and 1 kg of cotton have the exact same mass: 1 kilogram. Therefore, they weigh the same."
                ans_str = "They weigh the same."
            else:  # prime
                q = t["prime"]
                r = "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The first prime number is 2."
                ans_str = "The smallest prime number is 2."
            tokens, mask = encode_think_doc(tok, lang, q, r, ans_str)

        else:  # logic / algebra (eng only)
            if lang != "eng":
                continue
            alg_choice = rng.choice(["linear", "after_num", "day"])
            if alg_choice == "linear":
                a = rng.randint(2, 6)
                x = rng.randint(1, 15)
                b = rng.randint(1, 20)
                c = a * x + b
                q = templates["eng"]["linear"].format(a=a, b=b, c=c)
                r = f"Solving {a}x + {b} = {c}:\n1. Subtract {b} from both sides: {a}x = {c - b}\n2. Divide by {a}: x = {x}"
                ans_str = f"x = {x}"
            elif alg_choice == "after_num":
                a = rng.randint(10, 999)
                q = templates["eng"]["after_num"].format(a=a)
                r = f"The integer immediately after {a} is {a} + 1 = {a + 1}."
                ans_str = f"The number after {a} is {a + 1}."
            else:
                d_idx = rng.randint(0, 6)
                n = rng.choice([1, 2, 3])
                day = weekdays[d_idx]
                target_idx = (d_idx - n + 1) % 7
                today = weekdays[target_idx]
                q = templates["eng"]["day"].format(n=n, day=day)
                r = f"If yesterday was {n} days before {day}, then yesterday was {weekdays[(d_idx - n) % 7]}. Today is one day after yesterday, which is {today}."
                ans_str = f"Today is {today}."
            tokens, mask = encode_think_doc(tok, "eng", q, r, ans_str)

        count += 1
        yield ("math_synthetic", lang, tokens, mask)
    print(f"[synthetic-math] generated {count} verified drills")


def collect_smoltalk(tok: Tokenizer, max_docs: int = 50000):
    train_files = sorted(RAW_SFT.glob("HuggingFaceTB__smol-smoltalk/data/train-*.parquet"))
    if not train_files:
        print("[skip] smol-smoltalk train files not found")
        return
    print(f"[smoltalk] reading {len(train_files)} parquet files...")
    count = 0
    for f in train_files:
        if count >= max_docs:
            break
        df = pd.read_parquet(f)
        indices = list(range(len(df)))
        random.Random(count).shuffle(indices)
        for idx in indices:
            if count >= max_docs:
                break
            messages = df["messages"].iloc[idx]
            if not isinstance(messages, (list, np.ndarray)) or len(messages) < 2:
                continue
            doc = encode_multiturn_doc(tok, "eng", list(messages))
            if doc is None:
                continue
            tokens, mask = doc
            if len(tokens) < 16 or len(tokens) > 2048:
                continue
            count += 1
            yield ("smoltalk_chat", "eng", tokens, mask)
    print(f"[smoltalk] collected {count} multi-turn chat documents")


def collect_persona(tok: Tokenizer, total_samples: int = 4000):
    """Conversational persona and identity in English, Kannada, Tamil, Telugu, and Malayalam."""
    qas = [
        # English
        ("Who are you?", "I am Parentheses 0.9, an efficient Dravidian and English recurrent language model developed by uncoalesced."),
        ("What is your name?", "My name is Parentheses 0.9."),
        ("Who created you?", "I was created and developed by uncoalesced."),
        ("Who developed you?", "I was developed by uncoalesced as an efficient recurrent language model with Selective Linear Attention."),
        ("What architecture do you use?", "I am built on a recurrent State Space Duality (SSD) architecture in the Mamba-2 / RWKV / Selective Linear Attention family. During inference, I operate with constant O(1) memory state updates without a quadratic KV-cache."),
        ("What languages do you speak?", "I specialize in English and the Dravidian languages: Kannada, Tamil, Telugu, Malayalam, Tulu, Koya, Kuvi, and Kolami."),
        ("Are you a Transformer?", "No, I am not a standard Transformer model. I do not use quadratic causal self-attention or growing KV-caches. I am a recurrent State Space Duality (SSD) model with linear complexity."),
        ("Can you help me solve a math problem?", "Yes, certainly! Please share the problem and I will walk through the solution step by step."),
        ("Hello! How are you today?", "Hello! I am doing well, thank you. How can I help you today?"),
        ("Hi there", "Hello! How can I assist you today?"),
        
        # Kannada
        ("ನೀವು ಯಾರು?", "ನಾನು ಪ್ಯಾರೆಂಥಿಸಿಸ್ 0.9 (Parentheses 0.9), uncoalesced ಅವರು ಅಭಿವೃದ್ಧಿಪಡಿಸಿದ ದ್ರಾವಿಡ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಭಾಷೆಗಳ ಪರಿಣಾಮಕಾರಿ ಪುನರಾವರ್ತಿತ (recurrent) ಭಾಷಾ ಮಾದರಿ."),
        ("ನಿಮ್ಮ ಹೆಸರೇನು?", "ನನ್ನ ಹೆಸರು ಪ್ಯಾರೆಂಥಿಸಿಸ್ 0.9 (Parentheses 0.9)."),
        ("ನಿಮ್ಮನ್ನು ಯಾರು ರಚಿಸಿದರು?", "ನನ್ನನ್ನು uncoalesced ಅವರು ವಿನ್ಯಾಸಗೊಳಿಸಿ ಅಭಿವೃದ್ಧಿಪಡಿಸಿದ್ದಾರೆ."),
        ("ನಿಮಗೆ ಯಾವ ಭಾಷೆಗಳು ಗೊತ್ತು?", "ನಾನು ಕನ್ನಡ, ತಮಿಳು, ತೆಲುಗು, ಮಲಯಾಳಂ, ತುಳು ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಭಾಷೆಗಳಲ್ಲಿ ಉತ್ತರಿಸಬಲ್ಲೆ."),
        ("ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಿ?", "ನಮಸ್ಕಾರ! ನಾನು ಕ್ಷೇಮವಾಗಿದ್ದೇನೆ. ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"),
        
        # Tamil
        ("நீங்கள் யார்?", "நான் பேரென்தசிஸ் 0.9 (Parentheses 0.9), uncoalesced உருவாக்கிய திராவிட மற்றும் ஆங்கில மீளுருவாக்க மொழி மாதிரி."),
        ("உங்கள் பெயர் என்ன?", "என் பெயர் பேரென்தசிஸ் 0.9 (Parentheses 0.9)."),
        ("உங்களை உருவாக்கியவர் யார்?", "என்னை uncoalesced உருவாக்கினார்."),
        ("உங்களுக்கு என்னென்ன மொழிகள் தெரியும்?", "எனக்கு தமிழ், கன்னடம், தெலுங்கு, மலையாளம், துளு மற்றும் ஆங்கிலம் தெரியும்."),
        ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "வணக்கம்! நான் நலமாக இருக்கிறேன். உங்களுக்கு நான் எவ்வாறு உதவ முடியும்?"),
        
        # Telugu
        ("మీరు ఎవరు?", "నేను పేరెంథెసిస్ 0.9 (Parentheses 0.9), uncoalesced రూపొందించిన ద్రావిడ మరియు ఆంగ్ల రికరెంట్ భాషా మోడల్."),
        ("మీ పేరు ఏమిటి?", "నా పేరు పేరెంథెసిస్ 0.9 (Parentheses 0.9)."),
        ("మిమ్మల్ని ఎవరు సృష్టించారు?", "నన్ను uncoalesced సృష్టించారు."),
        ("మీకు ఏ భాషలు వచ్చు?", "నాకు తెలుగు, కన్నడ, తమిళం, మలయాಳಂ, తుళు మరియు ఇంగ్లీష్ భాషలు వచ్చు."),
        ("నమస్కారం, ఎలా ఉన్నారు?", "నమస్కారం! నేను బాగున్నాను. ఈ రోజు నేను మీకు ఎలా సహాయపడగలను?"),
        
        # Malayalam
        ("നിങ്ങൾ ആരാണ്?", "ഞാൻ പാരൻതസിസ് 0.9 (Parentheses 0.9) ആണ്, uncoalesced വികസിപ്പിച്ച ദ്രാവിഡ-ഇംഗ്ലീഷ് റിക്കറന്റ് ലാംഗ്വേജ് മോഡൽ."),
        ("നിങ്ങളുടെ പേരെന്താണ്?", "എന്റെ പേര് പാരൻതസിസ് 0.9 (Parentheses 0.9) എന്നാണ്."),
        ("നിങ്ങളെ സൃഷ്ടിച്ചത് ആരാണ്?", "എന്നെ വികസിപ്പിച്ചതും സൃഷ്ടിച്ചതും uncoalesced ആണ്."),
        ("നിങ്ങൾക്ക് ഏതൊക്കെ ഭാഷകൾ അറിയാം?", "എനിക്ക് മലയാളം, തമിഴ്, തെലുങ്ക്, കന്നഡ, തുളു, ഇംഗ്ലീഷ് എന്നീ ഭാഷകൾ കൈകാര്യം ചെയ്യാൻ കഴിയും."),
        ("നമസ്കാരം, എങ്ങനെയുണ്ട്?", "നമസ്കാരം! എനിക്ക് സുഖമാണ്. ഇന്ന് ഞാൻ നിങ്ങൾക്ക് എന്താണ് സഹായിക്കേണ്ടത്?"),
    ]

    rng = random.Random(42)
    count = 0
    while count < total_samples:
        q, a = rng.choice(qas)
        # Determine language from script
        if any("\u0c80" <= c <= "\u0cff" for c in q):
            lang = "kan"
        elif any("\u0b80" <= c <= "\u0bff" for c in q):
            lang = "tam"
        elif any("\u0c00" <= c <= "\u0c7f" for c in q):
            lang = "tel"
        elif any("\u0d00" <= c <= "\u0d7f" for c in q):
            lang = "mal"
        else:
            lang = "eng"
        tokens, mask = encode_chat_doc(tok, lang, q, a)
        count += 1
        yield ("persona", lang, tokens, mask)
    print(f"[persona] collected {count} identity documents")


def collect_dravidian_instructions(tok: Tokenizer, max_docs_per_lang: int = 15000, max_indicqa_per_lang: int = 600):
    """Collects Aya & IndicQA native Dravidian instruction data."""
    # 1. Aya dataset
    aya_p = RAW_MANUAL / "aya_dataset" / "data" / "train-00000-of-00001.parquet"
    if aya_p.exists():
        print(f"[dravidian-aya] reading {aya_p}...")
        df = pd.read_parquet(aya_p)
        drav_df = df[df["language_code"].isin(["kan", "tam", "tel", "mal"])]
        lang_counts = Counter()
        for _, row in drav_df.iterrows():
            lang = str(row["language_code"]).strip()
            if lang_counts[lang] >= max_docs_per_lang:
                continue
            inp = str(row.get("inputs", "")).strip()
            tgt = str(row.get("targets", "")).strip()
            if len(inp) < 6 or len(tgt) < 6:
                continue
            tokens, mask = encode_chat_doc(tok, lang, inp, tgt)
            if len(tokens) < 16 or len(tokens) > 2048:
                continue
            lang_counts[lang] += 1
            yield ("drav_aya", lang, tokens, mask)
        print(f"[dravidian-aya] collected {sum(lang_counts.values())} docs across {dict(lang_counts)}")

    # 2. IndicQA
    indic_dir = RAW_MANUAL / "indicqa"
    if indic_dir.exists():
        print(f"[dravidian-indicqa] reading {indic_dir}...")
        for json_file in indic_dir.glob("indicqa.*.json"):
            lang = json_file.stem.split(".")[-1]
            lang_map = {"kn": "kan", "ta": "tam", "te": "tel", "ml": "mal"}
            canon_l = lang_map.get(lang, "eng")
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            count = 0
            for item in data.get("data", []):
                if count >= max_indicqa_per_lang:
                    break
                for para in item.get("paragraphs", []):
                    if count >= max_indicqa_per_lang:
                        break
                    context = clean_text(para.get("context", ""))
                    for qa in para.get("qas", []):
                        if count >= max_indicqa_per_lang:
                            break
                        q = clean_text(qa.get("question", ""))
                        answers = [clean_text(a.get("text", "")) for a in qa.get("answers", []) if a.get("text")]
                        if not answers or not q or not context:
                            continue
                        ans = answers[0]
                        prompt = f"{context}\n\nಪ್ರಶ್ನೆ: {q}" if canon_l == "kan" else (
                            f"{context}\n\nகேள்வி: {q}" if canon_l == "tam" else (
                                f"{context}\n\nప్రశ్న: {q}" if canon_l == "tel" else (
                                    f"{context}\n\nചോദ്യം: {q}" if canon_l == "mal" else f"{context}\n\nQuestion: {q}"
                                )
                            )
                        )
                        tokens, mask = encode_think_doc(tok, canon_l, prompt, f"Grounded context contains: {ans}", ans)
                        if len(tokens) < 16 or len(tokens) > 2048:
                            continue
                        count += 1
                        yield ("drav_indicqa", canon_l, tokens, mask)
            print(f"[dravidian-indicqa] collected {count} docs for {canon_l}")


def collect_pretraining_replay(target_tokens: int = 15000000):
    """Streams pretraining replay documents from data/processed_v5/train.bin."""
    b_path = V5_DIR / "train.bin"
    bnd_path = V5_DIR / "train_boundaries.npy"
    if not b_path.exists() or not bnd_path.exists():
        print("[skip] pretraining train.bin or train_boundaries.npy not found")
        return
    print(f"[replay] loading pretraining tokens from {b_path}...")
    bnd = np.load(bnd_path)
    b_mmap = np.memmap(b_path, dtype=np.uint16, mode="r")
    total_docs = len(bnd)
    
    rng = random.Random(999)
    indices = list(range(total_docs - 1))
    rng.shuffle(indices)

    collected_tokens = 0
    count = 0
    for idx in indices:
        if collected_tokens >= target_tokens:
            break
        s = int(bnd[idx])
        e = int(bnd[idx + 1])
        doc_len = e - s
        if doc_len < 32 or doc_len > 2048:
            continue
        tokens = np.array(b_mmap[s:e], dtype=np.uint16)
        # Verify no ESG in replay
        if np.any(tokens == ESG_ID):
            continue
        mask = np.ones(len(tokens), dtype=np.uint8)  # Full causal loss mask for base replay
        collected_tokens += len(tokens)
        count += 1
        yield ("pretrain_replay", "mixed", tokens, mask)
    print(f"[replay] collected {count} replay docs ({collected_tokens/1e6:.2f}M tokens)")


# -----------------------------------------------------------------------------
# Main Packing Routine
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Pack processed_sft_real_v1")
    parser.add_argument("--dry-run", action="store_true", help="Pack a tiny 500-doc smoke dataset")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.dry_run:
        out_dir = out_dir.parent / (out_dir.name + "_dryrun")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    orca_max = 500 if args.dry_run else 35000
    metamath_max = 500 if args.dry_run else 35000
    synth_max = 500 if args.dry_run else 20000
    smol_max = 500 if args.dry_run else 50000
    persona_max = 200 if args.dry_run else 4000
    aya_max = 200 if args.dry_run else 15000
    indicqa_max = 50 if args.dry_run else 600
    replay_tokens = 50000 if args.dry_run else 15000000

    print("=" * 60)
    print(f"Building processed_sft_real_v1 -> {out_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    all_docs: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    print("\n[1/6] Collecting Orca-Math & MetaMathQA step-by-step reasoning...")
    for item in collect_orca_math(tok, max_docs=orca_max):
        all_docs.append(item)
    for item in collect_metamath(tok, max_docs=metamath_max):
        all_docs.append(item)

    print("\n[2/6] Collecting Synthetic Arithmetic & Logic Drills (English & Dravidian)...")
    for item in collect_synthetic_math_and_logic(tok, total_samples=synth_max):
        all_docs.append(item)

    print("\n[3/6] Collecting SmolTalk Multi-turn Dialogues...")
    for item in collect_smoltalk(tok, max_docs=smol_max):
        all_docs.append(item)

    print("\n[4/6] Collecting Parentheses 0.9 Persona & Identity...")
    for item in collect_persona(tok, total_samples=persona_max):
        all_docs.append(item)

    print("\n[5/6] Collecting Dravidian Instructions (Aya & IndicQA)...")
    for item in collect_dravidian_instructions(tok, max_docs_per_lang=aya_max, max_indicqa_per_lang=indicqa_max):
        all_docs.append(item)

    print("\n[6/6] Collecting Pretraining Replay...")
    for item in collect_pretraining_replay(target_tokens=replay_tokens):
        all_docs.append(item)

    print("\n" + "=" * 60)
    print(f"Total collected documents: {len(all_docs)}")
    print("Shuffling document order...")
    random.Random(20261020).shuffle(all_docs)

    # Calculate statistics
    cat_counts = Counter()
    cat_tokens = Counter()
    lang_counts = Counter()
    total_tokens = 0
    total_supervised_tokens = 0

    boundaries = [0]
    train_tokens_list: list[np.ndarray] = []
    train_masks_list: list[np.ndarray] = []

    for cat, lang, tokens, mask in all_docs:
        # Enforce strict invariant: zero ESG tokens
        if np.any(tokens == ESG_ID):
            continue
        cat_counts[cat] += 1
        lang_counts[lang] += 1
        n_tok = len(tokens)
        n_sup = int(np.sum(mask))
        cat_tokens[cat] += n_tok
        total_tokens += n_tok
        total_supervised_tokens += n_sup
        train_tokens_list.append(tokens)
        train_masks_list.append(mask)
        boundaries.append(total_tokens)

    boundaries.pop()  # last boundary is end of array
    bnd_arr = np.array(boundaries, dtype=np.int64)

    print(f"Total unique tokens: {total_tokens:,} ({total_tokens/1e6:.2f}M)")
    print(f"Total supervised target tokens: {total_supervised_tokens:,} ({total_supervised_tokens/1e6:.2f}M, {total_supervised_tokens/total_tokens*100:.1f}%)")
    print(f"Category breakdown:")
    for cat, c in cat_counts.most_common():
        toks = cat_tokens[cat]
        print(f"  - {cat:20s}: {c:6d} docs, {toks:10,d} tokens ({toks/total_tokens*100:5.1f}%)")
    print(f"Language breakdown: {dict(lang_counts)}")

    # Write files
    print("\nWriting train.bin...")
    full_tokens = np.concatenate(train_tokens_list)
    assert len(full_tokens) == total_tokens
    with open(out_dir / "train.bin", "wb") as f:
        full_tokens.tofile(f)

    print("Writing train.mask.bin...")
    full_masks = np.concatenate(train_masks_list)
    assert len(full_masks) == total_tokens
    with open(out_dir / "train.mask.bin", "wb") as f:
        full_masks.tofile(f)

    print("Writing train_boundaries.npy...")
    np.save(out_dir / "train_boundaries.npy", bnd_arr)

    # Manifest
    manifest = {
        "version": "sft_real_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_documents": len(train_tokens_list),
        "total_tokens": int(total_tokens),
        "supervised_tokens": int(total_supervised_tokens),
        "supervised_share": float(total_supervised_tokens / total_tokens if total_tokens else 0.0),
        "categories": {cat: {"docs": cat_counts[cat], "tokens": cat_tokens[cat]} for cat in cat_counts},
        "languages": dict(lang_counts),
        "esg_count": int(np.sum(full_tokens == ESG_ID)),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] Successfully packed {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
