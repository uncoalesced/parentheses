# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Option 3 Format Nudge v3 Dataset Builder
#
# Follows the EPOCH RULE from HANDOFF:
#   - Total bins: 500 bins of 4,096 tokens = 2,048,000 tokens (<= 2 epochs for 50 steps x 16 = 800 visits -> 1.6 epochs)
#   - 50% General SFT Replay (250 bins) sampled from data/processed_sft_240m_s10
#   - 30% Dravidian Arithmetic Format Grounding (150 bins) targeting Tamil and Telugu word problems with `Final: <val>`
#   - 12% Diverse Epistemic Abstentions (60 bins)
#   - 8% Architectural Identity Grounding (40 bins)
#
# Anti-Leakage Guard:
#   Entity lists and templates are strictly disjoint from the held-out test suite.

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.build_s7_5_sft_240m as b75
from scripts.pack_sft_4096_v4 import ffd_pack_documents, resolve_special_token_ids

# ---------------------------------------------------------------------------
# 1. Architectural Identity Q&A
# ---------------------------------------------------------------------------

IDENTITY_TEMPLATES = [
    ("Who created and designed the Parentheses model?",
     "Parentheses was created, designed, and trained from scratch by **uncoalesced** as an open-source research initiative."),
    ("Who is the author of Parentheses?",
     "Parentheses is authored and engineered by **uncoalesced**."),
    ("Tell me about the engineering origin of Parentheses.",
     "Parentheses was built entirely from scratch by **uncoalesced**, without using pretrained weights from any external foundation model."),
    ("Was Parentheses developed by a major tech corporation like Google, Meta, or OpenAI?",
     "No. Parentheses is an independent foundation model created entirely by **uncoalesced**, not by Google, OpenAI, Meta, or Anthropic."),
    ("What neural network architecture does Parentheses use?",
     "Parentheses is built on State Space Duality (SSD) and selective linear attention, belonging to the recurrent Mamba-2 and RWKV model family. It processes inputs in O(T) linear time during training and constant O(1) recurrent memory during inference."),
    ("Is Parentheses a standard Transformer?",
     "No, Parentheses is strictly not a Transformer. It completely eliminates quadratic causal self-attention (O(T^2)) and softmax attention matrices, replacing them with chunked selective linear attention and recurrent memory updates."),
    ("How does inference work in Parentheses compared to a GPT model?",
     "Unlike GPT or Transformer models which require a growing Key-Value cache (KV-cache) that expands with sequence length, Parentheses operates in recurrent mode with a constant O(1) memory state: S_t = alpha_t * S_{t-1} + (k_t outer v_t). Memory consumption remains completely flat regardless of context length."),
    ("Does Parentheses need a KV cache?",
     "No. Parentheses operates without a KV-cache. It updates a fixed-size recurrent memory tensor in O(1) time per step, eliminating memory expansion during long-form generation."),
    ("What parameter scale and layer structure is Parentheses 1.0?",
     "Parentheses 1.0 has 253.54 million parameters in a wide-and-shallow 16-layer configuration designated as slab_v4."),
    ("On what hardware envelope was Parentheses trained and verified?",
     "Parentheses was trained, verified, and calibrated entirely on local consumer hardware centered around an NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)."),
]

def generate_identity_rows(target_count: int = 1500) -> List[Dict[str, Any]]:
    rng = random.Random(20261112)
    rows = []
    while len(rows) < target_count:
        for p, r in IDENTITY_TEMPLATES:
            if len(rows) >= target_count:
                break
            prefix = rng.choice(["", "Hello! ", "Hi, ", "Quick question: ", "Can you clarify: "])
            full_p = f"{prefix}{p}" if prefix else p
            rows.append({
                "prompt": full_p,
                "response": r,
                "lang": "eng",
                "category": "identity",
            })
    return rows

# ---------------------------------------------------------------------------
# 2. Dravidian Format Grounding Cards (Strictly Disjoint from Held-out)
# ---------------------------------------------------------------------------

def generate_dravidian_format_rows(target_count: int = 6000) -> List[Dict[str, Any]]:
    rng = random.Random(20261112)
    rows = []

    kn_towns = ["ಮೈಸೂರು", "ಶಿವಮೊಗ್ಗ", "ಚಿಕ್ಕಮಗಳೂರು", "ದಾವಣಗೆರೆ", "ಹಾಸನ", "ಬೆಳಗಾವಿ", "ಉಡುಪಿ", "ಮಂಗಳೂರು"]
    ta_towns = ["மதுரை", "கோயம்புத்தூர்", "திருச்சி", "சேலம்", "தஞ்சாவூர்", "திருநெல்வேலி", "வேலூர்", "ஈரோடு"]
    te_towns = ["విజయవాడ", "గుంటూరు", "వరంగల్", "తిరుపతి", "విశాఖపట్నం", "కర్నూలు", "నెల్లూరు", "రాజమండ్రి"]
    ml_towns = ["കോഴിക്കോട്", "തൃശ്ശൂർ", "കൊല്ലം", "പാലക്കാട്", "കണ്ണൂർ", "ആലപ്പുഴ", "കോട്ടയം", "മലപ്പുറം"]

    while len(rows) < target_count:
        # A. Tamil Arithmetic Format Items (Completely disjoint nouns/verbs)
        t_type = rng.randint(1, 5)
        if t_type == 1:
            # Library shelves
            a = rng.randint(10, 45)
            b = rng.randint(12, 50)
            ans = a * b
            prompt = f"ஒரு மைய நூலகத்தில் {a} மர அலமாரிகள் உள்ளன. ஒவ்வொரு அலமாரியிலும் {b} சிறுகதை நூல்கள் அடுக்கப்பட்டுள்ளன. அங்குள்ள மொத்த நூல்கள் எத்தனை? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            response = f"மொத்த நூல்கள் = {a} * {b} = {ans}.\nFinal: {ans}"
        elif t_type == 2:
            # Fruit shop oranges / guavas
            a = rng.randint(25, 90)
            b = rng.randint(15, 80)
            ans = a + b
            prompt = f"ஒரு பழக்கடையில் {a} ஆரஞ்சுப் பழங்களும் {b} கொய்யாப்பழங்களும் விற்பனைக்கு வைக்கப்பட்டிருந்தன. கடையில் இருந்த மொத்தப் பழங்கள் எத்தனை? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            response = f"மொத்த பழங்கள் = {a} + {b} = {ans}.\nFinal: {ans}"
        elif t_type == 3:
            # Flower garlands subtraction
            a = rng.randint(40, 110)
            b = rng.randint(5, a - 5)
            ans = a - b
            prompt = f"ஒரு பூ வியாபாரியிடம் காலையில் {a} ரோஜா மாலைகள் இருந்தன. மாலையில் {b} மாலைகள் விற்றுத் தீர்ந்தன. அவரிடம் எஞ்சிய மாலைகள் எத்தனை? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            response = f"எஞ்சிய மாலைகள் = {a} - {b} = {ans}.\nFinal: {ans}"
        elif t_type == 4:
            # Direct arithmetic calculation
            a = rng.randint(20, 150)
            b = rng.randint(15, 90)
            ans = a + b
            prompt = f"{a} மற்றும் {b} ஆகிய எண்களின் கூட்டுத்தொகை எவ்வளவு? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            response = f"{a} + {b} = {ans}.\nFinal: {ans}"
        else:
            # Warehouse stock query
            town = rng.choice(ta_towns)
            val = rng.randint(30, 850)
            prompt = (
                f"கீழே கொடுக்கப்பட்டுள்ள சேமிப்புக் கிடங்கு பதிவேட்டைக் கவனித்து பதிலளிக்கவும்.\n\n"
                f"[பதிவேடு: {town}]\nதானியம்: உளுந்து\nஇருப்பு: {val} கிலோ\n\n"
                f"{town} சேமிப்புக் கிடங்கில் உள்ள உளுந்து இருப்பு எவ்வளவு? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            )
            response = f"{town} கிடங்கில் உள்ள உளுந்து இருப்பு {val} கிலோ.\nFinal: {val}"
        rows.append({"prompt": prompt, "response": response, "lang": "tam", "category": "dravidian_format"})

        # B. Telugu Arithmetic Format Items (Completely disjoint nouns/verbs)
        te_type = rng.randint(1, 5)
        if te_type == 1:
            # Library racks
            a = rng.randint(10, 40)
            b = rng.randint(15, 45)
            ans = a * b
            prompt = f"ఒక గ్రంథాలయంలో {a} పెద్ద అరలు ఉన్నాయి. ప్రతి అరలోనూ {b} కథల పుస్తకాలు ఉన్నాయి. అక్కడ ఉన్న మొత్తం పుస్తకాల సంఖ్య ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            response = f"మొత్తం పుస్తకాల సంఖ్య = {a} * {b} = {ans}.\nFinal: {ans}"
        elif te_type == 2:
            # Fruit market apples / oranges
            a = rng.randint(25, 95)
            b = rng.randint(15, 80)
            ans = a + b
            prompt = f"ఒక దుకాణంలో {a} ఆపిల్ పండ్లు మరియు {b} కమలా పండ్లు ఉన్నాయి. దుకాణంలోని మొత్తం పండ్ల సంఖ్య ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            response = f"మొత్తం పండ్ల సంఖ్య = {a} + {b} = {ans}.\nFinal: {ans}"
        elif te_type == 3:
            # Flower garlands subtraction
            a = rng.randint(45, 120)
            b = rng.randint(5, a - 5)
            ans = a - b
            name = rng.choice(["సుబ్బారావు", "గోపాల్", "ఆనంద్", "శివ"])
            prompt = f"{name} వద్ద {a} గులాబీ దండలు ఉండేవి. ఉదయం వ్యాపారంలో {b} దండలు అమ్ముడయ్యాయి. అతని వద్ద మిగిలి ఉన్న దండలు ఎన్ని? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            response = f"మిగిలిన దండలు = {a} - {b} = {ans}.\nFinal: {ans}"
        elif te_type == 4:
            # Direct arithmetic calculation
            a = rng.randint(25, 140)
            b = rng.randint(15, 90)
            ans = a + b
            prompt = f"{a} మరియు {b} ల సంకలనం ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            response = f"{a} + {b} = {ans}.\nFinal: {ans}"
        else:
            town = rng.choice(te_towns)
            val = rng.randint(40, 900)
            prompt = (
                f"క్రింది నిల్వ గిడ్డంగి నివేదికను పరిశీలించి సమాధానం ఇవ్వండి.\n\n"
                f"[గిడ్డంగి: {town}]\nదినుసు: వేరుశనగ\nనిల్వ: {val} బస్తాలు\n\n"
                f"{town} గిడ్డంగిలో ఉన్న వేరుశనగ నిల్వ ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            )
            response = f"{town} గిడ్డంగిలో ఉన్న వేరుశనగ నిల్వ {val} బస్తాలు.\nFinal: {val}"
        rows.append({"prompt": prompt, "response": response, "lang": "tel", "category": "dravidian_format"})

        # C. Kannada & Malayalam balanced items
        if rng.random() < 0.6:
            # Kannada
            a = rng.randint(15, 75)
            b = rng.randint(10, 60)
            ans = a + b
            prompt = f"ಒಂದು ಹಣ್ಣಿನ ಅಂಗಡಿಯಲ್ಲಿ {a} ದಾಳಿಂಬೆ ಮತ್ತು {b} ಕಿತ್ತಳೆ ಹಣ್ಣುಗಳಿವೆ. ಅಂಗಡಿಯಲ್ಲಿರುವ ಒಟ್ಟು ಹಣ್ಣುಗಳು ಎಷ್ಟು? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            response = f"ಒಟ್ಟು ಹಣ್ಣುಗಳು = {a} + {b} = {ans}.\nFinal: {ans}"
            rows.append({"prompt": prompt, "response": response, "lang": "kan", "category": "dravidian_format"})

            # Malayalam
            a = rng.randint(15, 75)
            b = rng.randint(10, 60)
            ans = a + b
            prompt = f"ഒരു പഴക്കടയിൽ {a} ഓറഞ്ചും {b} മാതളനാരങ്ങയും ഉണ്ട്. കടയിലുള്ള ആകെ പഴങ്ങൾ എത്രയാണ്? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            response = f"ആകെ പഴങ്ങൾ = {a} + {b} = {ans}.\nFinal: {ans}"
            rows.append({"prompt": prompt, "response": response, "lang": "mal", "category": "dravidian_format"})

    return rows[:target_count]

# ---------------------------------------------------------------------------
# 3. Epistemic Abstentions
# ---------------------------------------------------------------------------

def generate_epistemic_abstention_rows(target_count: int = 3000) -> List[Dict[str, Any]]:
    rng = random.Random(20261112)
    rows = []

    anachronisms = [
        ("Did Marco Polo stream his journey on Twitch?", "Marco Polo lived in the 13th century, centuries before video streaming was invented. Therefore, this never occurred."),
        ("What video game console did Queen Victoria play in Buckingham Palace?", "Video game consoles were not developed until the 1970s. Queen Victoria died in 1901."),
        ("In what year did Confucius drive a hybrid electric vehicle to Lu?", "Confucius lived in ancient China during the Spring and Autumn period. Automobiles did not exist then."),
        ("How many USB flash drives were used during the Apollo 11 moon landing in 1969?", "USB flash drives were invented in the late 1990s. Apollo 11 navigation used magnetic-core rope memory."),
    ]

    future_traps = [
        ("Which lucky numbers will be selected in the New York Mega Millions on July 14, 2045?", "July 14, 2045 is a future date. Future lottery numbers are random and inherently unknowable."),
        ("Which athlete takes the top podium spot for the 100m sprint in 2048?", "The 2048 Olympic Games have not occurred yet. Future sports winners cannot be known."),
        ("Predict the valuation of Bitcoin in US dollars on October 2, 2042.", "October 2, 2042 is in the future. Financial market prices cannot be predicted with certainty."),
    ]

    fake_entities = [
        ("What is the molecular formula of Lumicite-7?", "Lumicite-7 is not a verified chemical compound or physical element in scientific records."),
        ("Who was the royal chancellor of the Kingdom of Valendor in 1432?", "The Kingdom of Valendor is a fictional realm, not a historical nation."),
    ]

    dravidian_false_premises = [
        ("ಮೈಸೂರು ಅರಮನೆಯನ್ನು ಫ್ರಾನ್ಸ್‌ನ ಪ್ಯಾರಿಸ್ ನಗರದಲ್ಲಿ ನಿರ್ಮಿಸಲಾಗಿದೆಯೇ?", "ಇದು ತಪ್ಪು ಮಾಹಿತಿ. ಮೈಸೂರು ಅರಮನೆಯು ಕರ್ನಾಟಕದ ಮೈಸೂರು ನಗರದಲ್ಲಿದೆ.", "kan"),
        ("மதுரை மீனாட்சி அம்மன் கோவில் இங்கிலாந்தின் லண்டன் நகரில் உள்ளதா?", "இது தவறான தகவல். மதுரை மீனாட்சி அம்மன் கோவில் தமிழ்நாட்டின் மதுரையில் உள்ளது.", "tam"),
        ("చార్మినార్ కట్టడం రష్యాలోని మాస్కో నగరంలో ఉందా?", "ఇది వాస్తవం కాదు. చార్మినార్ తెలంగాణ రాష్ట్రంలోని హైదరాబాద్ నగరంలో ఉంది.", "tel"),
        ("പത്മനാഭസ്വാമി ക്ഷേത്രം ആസ്ട്രേലിയയിലെ സിഡ്നിയിലാണോ സ്ഥിതി ചെയ്യുന്നത്?", "ഇത് തെറ്റായ വിവരമാണ്. പത്മനാഭസ്വാമി ക്ഷേത്രം കേരളത്തിലെ തിരുവനന്തപുരത്താണ് സ്ഥിതി ചെയ്യുന്നത്.", "mal"),
    ]

    while len(rows) < target_count:
        q, r = rng.choice(anachronisms)
        rows.append({"prompt": q, "response": f"I cannot answer. {r}", "lang": "eng", "category": "abstention_anachronism"})
        q, r = rng.choice(future_traps)
        rows.append({"prompt": q, "response": f"I don't know. {r}", "lang": "eng", "category": "abstention_future"})
        q, r = rng.choice(fake_entities)
        rows.append({"prompt": q, "response": f"I cannot answer. {r}", "lang": "eng", "category": "abstention_fake"})
        q, r, lang = rng.choice(dravidian_false_premises)
        rows.append({"prompt": q, "response": r, "lang": lang, "category": "abstention_dravidian"})

    return rows[:target_count]

# ---------------------------------------------------------------------------
# 4. Master Corpus Assembly & Replay Sampling
# ---------------------------------------------------------------------------

FORBIDDEN_HELDOUT_SCENARIOS = [
    "napoleon", "abraham lincoln", "gettysburg", "leonardo da vinci",
    "alexander the great", "isaac newton", "thomas jefferson", "benjamin franklin",
    "charles babbage", "difference engine", "hannibal", "charlemagne",
    "lord nelson", "battle of trafalgar", "queen elizabeth", "william shakespeare",
    "california powerball", "apple inc", "icc men's cricket world cup",
    "census population of tokyo", "euro and the us dollar", "wimbledon",
    "112th oscars", "51st president", "reserve bank of india", "artemis 9",
    "chronium-99", "valyria", "north pangea", "subatomic aetherology",
    "dr. finch", "zephyria", "elidor vance", "krypto-ferrite",
    "morvath", "dr. donald thistlewood", "nebulastin", "luminescium-404",
]

def sample_replay_bins(
    s10_dir: Path,
    tok: Tokenizer,
    num_bins: int = 250,
    seed: int = 20261112
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    data_path = s10_dir / "train.bin"
    mask_path = s10_dir / "train.mask.bin"
    bnd_path = s10_dir / "train_boundaries.npy"

    total_tokens = data_path.stat().st_size // 2
    total_bins = total_tokens // 4096
    print(f"  Source S10 has {total_bins:,} bins ({total_tokens:,} tokens). Sampling {num_bins} clean bins...")

    data_mmap = np.memmap(data_path, dtype=np.uint16, mode="r")
    mask_mmap = np.memmap(mask_path, dtype=np.uint8, mode="r")
    bnd_3d = np.load(bnd_path, mmap_mode="r")

    rng = np.random.default_rng(seed)
    shuffled_candidates = rng.permutation(total_bins)

    clean_bins = []
    print("  Screening candidate replay bins against held-out scenarios...")
    for b_idx in shuffled_candidates:
        if len(clean_bins) >= num_bins:
            break
        start = int(b_idx) * 4096
        end = start + 4096
        tok_chunk = data_mmap[start:end]
        text_chunk = tok.decode(tok_chunk.tolist(), skip_special_tokens=True).lower()
        if not any(scn in text_chunk for scn in FORBIDDEN_HELDOUT_SCENARIOS):
            clean_bins.append(int(b_idx))

    clean_bins.sort()
    chosen_bin_indices = np.array(clean_bins)
    print(f"  Selected {len(chosen_bin_indices)} strictly decontaminated replay bins.")

    chosen_set = set(chosen_bin_indices)
    bnd_map = {b_idx: [] for b_idx in chosen_bin_indices}
    for row in bnd_3d:
        b_idx = int(row[0])
        if b_idx in chosen_set:
            bnd_map[b_idx].append(int(row[1]))

    out_data = np.empty((num_bins, 4096), dtype=np.uint16)
    out_mask = np.empty((num_bins, 4096), dtype=np.uint8)
    out_offsets: List[List[int]] = []

    for new_idx, orig_idx in enumerate(chosen_bin_indices):
        s = orig_idx * 4096
        e = s + 4096
        out_data[new_idx] = data_mmap[s:e]
        out_mask[new_idx] = mask_mmap[s:e]
        offs = sorted(bnd_map[orig_idx])
        if not offs or offs[0] != 0:
            offs = [0] + offs
        out_offsets.append(offs)

    return out_data, out_mask, out_offsets

def pack_rows_to_bins(
    tok: Tokenizer,
    rows: List[Dict[str, Any]],
    target_bins: int,
    bin_size: int = 4096,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Encodes rows and packs them into bins using FFD, returning exactly `target_bins` bins."""
    ids_map = resolve_special_token_ids(tok)
    unmapped = Counter()

    docs = []
    for r in rows:
        p = r["prompt"]
        a = r["response"]
        lang = r["lang"]
        status, ids_arr, mask_arr, _ = b75.encode_sft_doc_with_rescue(tok, p, a, lang, ids_map, unmapped)
        if status == "ok" and ids_arr is not None:
            docs.append((ids_arr, mask_arr))

    print(f"  Encoded {len(docs)} documents successfully.")
    p_ids, p_mask, b3d, bflat, stats = ffd_pack_documents(docs, bin_size=bin_size, seed=seed, eot_id=ids_map["eot"])
    actual_bins = p_ids.shape[0]
    print(f"  FFD packed into {actual_bins} bins (target: {target_bins}).")

    if actual_bins < target_bins:
        reps = (target_bins + actual_bins - 1) // actual_bins
        p_ids = np.tile(p_ids, (reps, 1))[:target_bins]
        p_mask = np.tile(p_mask, (reps, 1))[:target_bins]
    else:
        p_ids = p_ids[:target_bins]
        p_mask = p_mask[:target_bins]

    doc_offsets = []
    bnd_by_bin = {i: [] for i in range(actual_bins)}
    for row in b3d:
        b = int(row[0])
        if b < actual_bins:
            bnd_by_bin[b].append(int(row[1]))

    for i in range(target_bins):
        src_i = i % actual_bins
        starts = bnd_by_bin[src_i]
        if not starts or starts[0] != 0:
            starts = [0] + [s for s in starts if s != 0]
        doc_offsets.append(starts)

    return p_ids, p_mask, doc_offsets

def main():
    parser = argparse.ArgumentParser(description="Build S10 Option 3 Format Nudge Dataset (500 bins)")
    parser.add_argument("--s10-dir", type=Path, default=REPO_ROOT / "data" / "processed_sft_240m_s10")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed_sft_240m_s10_nudge_v3")
    parser.add_argument("--tok-path", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("[*] BUILDING BALANCED S10 OPTION 3 NUDGE V3 (500 BINS / 2.05M TOKENS)")
    print(f"Output Directory: {out_dir}")
    print("=" * 75)

    tok = Tokenizer.from_file(str(args.tok_path))

    # 1. 250 Replay Bins (50%)
    rep_ids, rep_mask, rep_offsets = sample_replay_bins(args.s10_dir, tok=tok, num_bins=250, seed=args.seed)

    # 2. 150 Dravidian Format Bins (30%)
    print("\n[*] Generating Dravidian arithmetic format grounding rows (Tamil/Telugu targeted)...")
    drav_rows = generate_dravidian_format_rows(target_count=5000)
    drav_ids, drav_mask, drav_offsets = pack_rows_to_bins(tok, drav_rows, target_bins=150, seed=args.seed + 1)

    # 3. 60 Epistemic Abstention Bins (12%)
    print("\n[*] Generating Epistemic abstention rows...")
    abst_rows = generate_epistemic_abstention_rows(target_count=2000)
    abst_ids, abst_mask, abst_offsets = pack_rows_to_bins(tok, abst_rows, target_bins=60, seed=args.seed + 2)

    # 4. 40 Architectural Identity Bins (8%)
    print("\n[*] Generating Architectural identity grounding rows...")
    id_rows = generate_identity_rows(target_count=1000)
    id_ids, id_mask, id_offsets = pack_rows_to_bins(tok, id_rows, target_bins=40, seed=args.seed + 3)

    print("\n[*] Interleaving categories across 500 bins...")
    total_bins = 500
    all_data = np.empty((total_bins, 4096), dtype=np.uint16)
    all_mask = np.empty((total_bins, 4096), dtype=np.uint8)
    all_boundaries: List[Tuple[int, int]] = []
    flat_offsets: List[int] = []

    r_idx, d_idx, a_idx, i_idx = 0, 0, 0, 0
    cycle = (
        ["rep"] * 5 +       # 50%
        ["drav"] * 3 +      # 30%
        ["abst"] * 1 +      # 10-12%
        ["id"] * 1          # 8-10%
    )  # 10 slots per block x 50 blocks = 500 bins!

    for b in range(total_bins):
        slot = cycle[b % len(cycle)]
        if slot == "rep" and r_idx < len(rep_ids):
            data_row, mask_row, offs = rep_ids[r_idx], rep_mask[r_idx], rep_offsets[r_idx]
            r_idx += 1
        elif slot == "drav" and d_idx < len(drav_ids):
            data_row, mask_row, offs = drav_ids[d_idx], drav_mask[d_idx], drav_offsets[d_idx]
            d_idx += 1
        elif slot == "abst" and a_idx < len(abst_ids):
            data_row, mask_row, offs = abst_ids[a_idx], abst_mask[a_idx], abst_offsets[a_idx]
            a_idx += 1
        elif slot == "id" and i_idx < len(id_ids):
            data_row, mask_row, offs = id_ids[i_idx], id_mask[i_idx], id_offsets[i_idx]
            i_idx += 1
        else:
            # Fallback
            if r_idx < len(rep_ids):
                data_row, mask_row, offs = rep_ids[r_idx], rep_mask[r_idx], rep_offsets[r_idx]
                r_idx += 1
            elif d_idx < len(drav_ids):
                data_row, mask_row, offs = drav_ids[d_idx], drav_mask[d_idx], drav_offsets[d_idx]
                d_idx += 1
            elif a_idx < len(abst_ids):
                data_row, mask_row, offs = abst_ids[a_idx], abst_mask[a_idx], abst_offsets[a_idx]
                a_idx += 1
            else:
                data_row, mask_row, offs = id_ids[i_idx], id_mask[i_idx], id_offsets[i_idx]
                i_idx += 1

        all_data[b] = data_row
        all_mask[b] = mask_row
        for o in offs:
            all_boundaries.append((b, int(o)))
            flat_offsets.append(b * 4096 + int(o))

    print(f"Actual Bin Allocations across 500 bins:")
    print(f"  Replay:     {r_idx} bins ({r_idx / total_bins * 100:.1f}%)")
    print(f"  Dravidian:  {d_idx} bins ({d_idx / total_bins * 100:.1f}%)")
    print(f"  Abstention: {a_idx} bins ({a_idx / total_bins * 100:.1f}%)")
    print(f"  Identity:   {i_idx} bins ({i_idx / total_bins * 100:.1f}%)")

    # Save binaries
    print("\n[*] Writing binary files to disk...")
    train_bin_path = out_dir / "train.bin"
    train_mask_path = out_dir / "train.mask.bin"
    bnd_3d_path = out_dir / "train_boundaries.npy"
    bnd_flat_path = out_dir / "train_boundaries_flat.npy"
    manifest_path = out_dir / "manifest.json"

    all_data.tofile(str(train_bin_path))
    all_mask.tofile(str(train_mask_path))

    bnd_array_3d = np.array([[b, o, 4096] for b, o in all_boundaries], dtype=np.int32)
    bnd_array_flat = np.array(flat_offsets, dtype=np.int64)

    np.save(str(bnd_3d_path), bnd_array_3d)
    np.save(str(bnd_flat_path), bnd_array_flat)

    manifest = {
        "dataset_name": "processed_sft_240m_s10_nudge_v3",
        "num_bins": total_bins,
        "tokens_per_bin": 4096,
        "total_tokens": total_bins * 4096,
        "total_docs": len(all_boundaries),
        "replay_bins": r_idx,
        "dravidian_format_bins": d_idx,
        "epistemic_abstention_bins": a_idx,
        "identity_bins": i_idx,
        "planned_steps": 50,
        "effective_epochs": (50 * 16) / total_bins,
        "epoch_rule_verified": ((50 * 16) / total_bins) <= 2.0,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] SUCCESS: Dataset built successfully in {out_dir}")
    print(f"    Total Bins:   {total_bins:,}")
    print(f"    Total Tokens: {total_bins * 4096:,}")
    print(f"    Total Docs:   {len(all_boundaries):,}")
    print(f"    Epoch Ratio:  {manifest['effective_epochs']:.2f} epochs across 50 steps (<= 2.0 rule: {manifest['epoch_rule_verified']})")

if __name__ == "__main__":
    main()
