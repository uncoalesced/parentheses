# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 Balanced Micro-Patch v2 Builder
#
# Follows the EPOCH RULE from HANDOFF (reviewed 2026-11-10):
#   - Total bins: 1,000 bins of 4,096 tokens = 4,096,000 tokens (<= 2 epochs for 125 steps x 16 = 2,000 visits)
#   - 52% General SFT Replay (520 bins) sampled from data/processed_sft_240m_s10
#   - 20% Dravidian Format Grounding (200 bins) concluding with `Final: <val>`
#   - 20% Diverse Epistemic Abstentions (200 bins) with clean honest termination
#   - 8% Architectural Identity Grounding (80 bins) on SSD, recurrent states, O(1) memory, uncoalesced
#
# Anti-Leakage Guard:
#   Entity lists and templates are strictly disjoint from the held-out test suite (Section 2.1).

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
# 1. Architectural Identity Q&A (Diverse, non-repeating paraphrases & multi-turn styles)
# ---------------------------------------------------------------------------

IDENTITY_TEMPLATES = [
    # Creator attribution
    ("Who created and designed the Parentheses model?",
     "Parentheses was created, designed, and trained from scratch by **uncoalesced** as an open-source research initiative."),
    ("Who is the author of Parentheses?",
     "Parentheses is authored and engineered by **uncoalesced**."),
    ("Tell me about the engineering origin of Parentheses.",
     "Parentheses was built entirely from scratch by **uncoalesced**, without using pretrained weights from any external foundation model."),
    ("Was Parentheses developed by a major tech corporation like Google, Meta, or OpenAI?",
     "No. Parentheses is an independent foundation model created entirely by **uncoalesced**, not by Google, OpenAI, Meta, or Anthropic."),
    ("Who funded or built Parentheses?",
     "Parentheses is an independent, sovereign research project created by **uncoalesced** to prove that highly capable models can be trained and run on consumer-grade hardware."),

    # Architecture & State Space Duality
    ("What neural network architecture does Parentheses use?",
     "Parentheses is built on State Space Duality (SSD) and selective linear attention, belonging to the recurrent Mamba-2 and RWKV model family. It processes inputs in O(T) linear time during training and constant O(1) recurrent memory during inference."),
    ("Is Parentheses a standard Transformer?",
     "No, Parentheses is strictly not a Transformer. It completely eliminates quadratic causal self-attention (O(T^2)) and softmax attention matrices, replacing them with chunked selective linear attention and recurrent memory updates."),
    ("How does inference work in Parentheses compared to a GPT model?",
     "Unlike GPT or Transformer models which require a growing Key-Value cache (KV-cache) that expands with sequence length, Parentheses operates in recurrent mode with a constant O(1) memory state: S_t = alpha_t * S_{t-1} + (k_t outer v_t). Memory consumption remains completely flat regardless of context length."),
    ("Does Parentheses need a KV cache?",
     "No. Parentheses operates without a KV-cache. It updates a fixed-size recurrent memory tensor in O(1) time per step, eliminating memory expansion during long-form generation."),
    ("What are the key recurrent equations governing Parentheses?",
     "At each step t, Parentheses updates its hidden recurrent state via S_t = alpha_t S_{t-1} + beta_t (k_t outer v_t), and computes its output projection as y_t = q_t S_t. This ensures constant-time, constant-memory token generation."),
    ("What is Token-Shift or mu-mixing in Parentheses?",
     "Token-Shift (or mu-mixing, inspired by RWKV) linearly blends the current token embedding with the previous time-step's embedding: x_t' = mu * x_t + (1 - mu) * x_{t-1}. This grants the model direct local inductive bias across consecutive tokens."),

    # Scale and specs
    ("What parameter scale and layer structure is Parentheses 1.0?",
     "Parentheses 1.0 has 253.54 million parameters in a wide-and-shallow 16-layer configuration designated as slab_v4."),
    ("How many layers does Parentheses 1.0 have?",
     "Parentheses 1.0 utilizes 16 wide-and-shallow layers with a model dimension of 1024 and 16 attention heads, maintaining numerical stability through FP32 inter-chunk recurrence and RMSNorm."),
    ("What context length does Parentheses support?",
     "Parentheses is trained with a native context window of 4,096 tokens, and its recurrent state formulation allows stable recurrent extrapolation to 16,000+ tokens without memory blowup."),

    # Meaning of name
    ("Why is this model named Parentheses?",
     "The name 'Parentheses' symbolizes enclosing unbounded knowledge and human thought within a bounded, finite container of weights."),
    ("What is the philosophy behind the name Parentheses?",
     "In mathematics and programming, parentheses define scope and encapsulate complex expressions into an operable unit. Similarly, Parentheses encapsulates broad multilingual knowledge into a compact, efficient recurrent model."),

    # Hardware & Licensing
    ("What hardware was used to train Parentheses?",
     "Parentheses was trained locally on consumer hardware—specifically an NVIDIA GeForce RTX 5050 laptop GPU (8GB VRAM)—demonstrating extreme compute efficiency through selective linear attention."),
    ("Is Parentheses open source?",
     "Yes. Parentheses is released as fully open-source software under the permissive Apache 2.0 license by uncoalesced."),
    ("What license governs Parentheses?",
     "Parentheses is distributed under the Apache 2.0 license, allowing free commercial, academic, and personal use, modification, and distribution."),
    ("Where can I find the official repository for Parentheses?",
     "The official code and checkpoints for Parentheses are published on GitHub by uncoalesced (https://github.com/uncoalesced) and mirrored on Hugging Face."),
]

def generate_identity_rows(target_count: int = 3500) -> List[Dict[str, Any]]:
    rng = random.Random(20261110)
    rows = []
    prefixes = [
        "", "Please explain: ", "Quick question: ", "Could you clarify: ",
        "I'm curious: ", "Can you describe ", "Help me understand: ",
    ]
    suffixes = [
        "", " Thanks!", " Please explain concisely.", " Give a direct explanation.",
        " Be specific.", " Explain from first principles.",
    ]
    
    while len(rows) < target_count:
        for p, r in IDENTITY_TEMPLATES:
            if len(rows) >= target_count:
                break
            prefix = rng.choice(prefixes)
            suffix = rng.choice(suffixes)
            full_p = f"{prefix}{p}{suffix}".strip()
            rows.append({
                "prompt": full_p,
                "response": r,
                "lang": "eng",
                "category": "identity",
            })
    return rows


# ---------------------------------------------------------------------------
# 2. Dravidian Format Grounding Cards (Strict Terminal `Final: <val>`)
# ---------------------------------------------------------------------------

def generate_dravidian_format_rows(target_count: int = 7000) -> List[Dict[str, Any]]:
    rng = random.Random(20261111)
    rows = []

    # Kannada towns & metrics
    kn_towns = ["ಮೈಸೂರು", "ಶಿವಮೊಗ್ಗ", "ಚಿಕ್ಕಮಗಳೂರು", "ದಾವಣಗೆರೆ", "ಹಾಸನ", "ಬೆಳಗಾವಿ", "ಉಡುಪಿ", "ಮಂಗಳೂರು", "ತುಮಕೂರು", "ಧಾರವಾಡ"]
    kn_crops = [("ರಾಗಿ", "ಕ್ವಿಂಟಾಲ್"), ("ಭತ್ತ", "ಚೀಲ"), ("ಅಡಿಕೆ", "ಕೆ.ಜಿ"), ("ಕಾಫಿ ಬೀಜ", "ಚೀಲ"), ("ಮೆಕ್ಕೆಜೋಳ", "ಕ್ವಿಂಟಾಲ್")]

    # Tamil towns & metrics
    ta_towns = ["மதுரை", "கோயம்புத்தூர்", "திருச்சி", "சேலம்", "தஞ்சாவூர்", "திருநெல்வேலி", "வேலூர்", "ஈரோடு", "திண்டுக்கல்", "காஞ்சிபுரம்"]
    ta_crops = [("நெல்", "மூட்டை"), ("கரும்பு", "டன்"), ("பருத்தி", "கிலோ"), ("வாழைப்பழம்", "தார்"), ("மஞ்சள்", "மூட்டை")]

    # Telugu towns & metrics
    te_towns = ["విజయవాడ", "గుంటూరు", "వరంగల్", "తిరుపతి", "విశాఖపట్నం", "కర్నూలు", "నెల్లూరు", "రాజమండ్రి", "కరీంనగర్", "అనంతపురం"]
    te_crops = [("వరి", "బస్తాలు"), ("మిర్చి", "క్వింటాళ్ళు"), ("పత్తి", "కిలోలు"), ("వేరుశనగ", "బస్తాలు"), ("చెరకు", "టన్నులు")]

    # Malayalam towns & metrics
    ml_towns = ["കോഴിക്കോട്", "തൃശ്ശൂർ", "കൊല്ലം", "പാലക്കാട്", "കണ്ണൂർ", "ആലപ്പുഴ", "കോട്ടയം", "മലപ്പുറം", "വയനാട്", "ഇടുക്കി"]
    ml_crops = [("കുരുമുളക്", "കിലോഗ്രാം"), ("ഏലം", "കിലോ"), ("റബ്ബർ", "ഷീറ്റ്"), ("തേങ്ങ", "എണ്ണം"), ("നെല്ല്", "ചാക്ക്")]

    while len(rows) < target_count:
        # A. Kannada Format Items
        town = rng.choice(kn_towns)
        crop, unit = rng.choice(kn_crops)
        val = rng.randint(20, 950)
        p_type = rng.randint(1, 3)
        if p_type == 1:
            # Registry card
            prompt = (
                f"ಕೆಳಗಿನ ಕೃಷಿ ಗೋದಾಮಿನ ದಾಸ್ತಾನು ವಿವರವನ್ನು ಗಮನಿಸಿ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಿ.\n\n"
                f"[ದಾಸ್ತಾನು ಪಟ್ಟಿ: {town}]\n"
                f"ಉತ್ಪನ್ನ: {crop}\n"
                f"ಪ್ರಮಾಣ: {val} {unit}\n"
                f"ಸ್ಥಿತಿ: ಪರಿಶೀಲಿಸಲಾಗಿದೆ\n\n"
                f"{town} ಗೋದಾಮಿನಲ್ಲಿರುವ {crop} ಪ್ರಮಾಣ ಎಷ್ಟು? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            )
            response = f"{town} ಗೋದಾಮಿನಲ್ಲಿರುವ {crop} ಪ್ರಮಾಣ {val} {unit}.\nFinal: {val}"
        elif p_type == 2:
            # Arithmetic
            a = rng.randint(10, 80)
            b = rng.randint(5, 50)
            ans = a + b
            prompt = f"{a} ಮತ್ತು {b} ಸಂಖ್ಯೆಗಳ ಮೊತ್ತವೆಷ್ಟು? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            response = f"{a} + {b} = {ans}.\nFinal: {ans}"
        else:
            # Multiplication
            a = rng.randint(4, 25)
            b = rng.randint(3, 15)
            ans = a * b
            prompt = f"ಒಂದು ಪೆಟ್ಟಿಗೆಯಲ್ಲಿ {a} ಮಾವಿನಹಣ್ಣುಗಳಿವೆ. ಅಂತಹ {b} ಪೆಟ್ಟಿಗೆಗಳಲ್ಲಿ ಒಟ್ಟು ಎಷ್ಟು ಹಣ್ಣುಗಳಿರುತ್ತವೆ? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            response = f"ಒಟ್ಟು ಹಣ್ಣುಗಳು = {a} * {b} = {ans}.\nFinal: {ans}"
        rows.append({"prompt": prompt, "response": response, "lang": "kan", "category": "dravidian_format"})

        # B. Tamil Format Items
        town = rng.choice(ta_towns)
        crop, unit = rng.choice(ta_crops)
        val = rng.randint(20, 950)
        p_type = rng.randint(1, 3)
        if p_type == 1:
            prompt = (
                f"கீழே கொடுக்கப்பட்டுள்ள கிடங்கு பதிவேட்டைப் படித்து விடையளிக்கவும்.\n\n"
                f"[கிடங்கு குறிப்பு: {town}]\n"
                f"பொருள்: {crop}\n"
                f"அளவு: {val} {unit}\n"
                f"நிலை: சரிபார்க்கப்பட்டது\n\n"
                f"{town} கிடங்கில் உள்ள {crop} அளவு என்ன? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            )
            response = f"{town} கிடங்கில் உள்ள {crop} அளவு {val} {unit}.\nFinal: {val}"
        elif p_type == 2:
            a = rng.randint(15, 90)
            b = rng.randint(10, 60)
            ans = a + b
            prompt = f"{a} மற்றும் {b} ஆகியவற்றின் கூடுதல் என்ன? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            response = f"{a} + {b} = {ans}.\nFinal: {ans}"
        else:
            a = rng.randint(5, 30)
            b = rng.randint(4, 12)
            ans = a * b
            prompt = f"ஒரு கூடையில் {a} ஆப்பிள்கள் உள்ளன. அதே போன்ற {b} கூடைகளில் மொத்தம் எத்தனை ஆப்பிள்கள் இருக்கும்? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            response = f"மொத்த ஆப்பிள்கள் = {a} * {b} = {ans}.\nFinal: {ans}"
        rows.append({"prompt": prompt, "response": response, "lang": "tam", "category": "dravidian_format"})

        # C. Telugu Format Items
        town = rng.choice(te_towns)
        crop, unit = rng.choice(te_crops)
        val = rng.randint(20, 950)
        p_type = rng.randint(1, 3)
        if p_type == 1:
            prompt = (
                f"క్రింది గోదాము రికార్డు వివరాలను చదివి సమాధానం ఇవ్వండి.\n\n"
                f"[గోదాము నివేదిక: {town}]\n"
                f"సరుకు: {crop}\n"
                f"పరిమాణం: {val} {unit}\n"
                f"పరిస్థితి: ధృవీకరించబడింది\n\n"
                f"{town} గోదాములో ఉన్న {crop} పరిమాణం ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            )
            response = f"{town} గోదాములో ఉన్న {crop} పరిమాణం {val} {unit}.\nFinal: {val}"
        elif p_type == 2:
            a = rng.randint(20, 100)
            b = rng.randint(10, 75)
            ans = a + b
            prompt = f"{a} మరియు {b} ల మొత్తం ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            response = f"{a} + {b} = {ans}.\nFinal: {ans}"
        else:
            a = rng.randint(6, 25)
            b = rng.randint(4, 15)
            ans = a * b
            prompt = f"ఒక పెట్టెలో {a} పుస్తకాలు ఉన్నాయి. అలాంటి {b} పెట్టెలలో మొత్తం ఎన్ని పుస్తకాలు ఉంటాయి? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            response = f"మొత్తం పుస్తకాలు = {a} * {b} = {ans}.\nFinal: {ans}"
        rows.append({"prompt": prompt, "response": response, "lang": "tel", "category": "dravidian_format"})

        # D. Malayalam Format Items
        town = rng.choice(ml_towns)
        crop, unit = rng.choice(ml_crops)
        val = rng.randint(20, 950)
        p_type = rng.randint(1, 3)
        if p_type == 1:
            prompt = (
                f"താഴെ നൽകിയിരിക്കുന്ന വെയർഹൗസ് വിവരങ്ങൾ പരിശോധിച്ച് ഉത്തരം നൽകുക.\n\n"
                f"[രേഖ: {town}]\n"
                f"ഇനം: {crop}\n"
                f"അളവ്: {val} {unit}\n"
                f"നില: സ്ഥിരീകരിച്ചു\n\n"
                f"{town} സംഭരണശാലയിലെ {crop} എത്രയാണ്? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            )
            response = f"{town} സംഭരണശാലയിലെ {crop} അളവ് {val} {unit} ആണ്.\nFinal: {val}"
        elif p_type == 2:
            a = rng.randint(12, 85)
            b = rng.randint(8, 45)
            ans = a + b
            prompt = f"{a}, {b} എന്നിവയുടെ തുക എത്രയാണ്? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            response = f"{a} + {b} = {ans}.\nFinal: {ans}"
        else:
            a = rng.randint(7, 24)
            b = rng.randint(3, 14)
            ans = a * b
            prompt = f"ഒരു ബോക്സിൽ {a} ഓറഞ്ചുകൾ ഉണ്ട്. അങ്ങനെയുള്ള {b} ബോക്സുകളിൽ ആകെ എത്ര ഓറഞ്ചുകൾ ഉണ്ടാകും? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            response = f"ആകെ ഓറഞ്ചുകൾ = {a} * {b} = {ans}.\nFinal: {ans}"
        rows.append({"prompt": prompt, "response": response, "lang": "mal", "category": "dravidian_format"})

    return rows[:target_count]


# ---------------------------------------------------------------------------
# 3. Diverse Epistemic Abstentions (Disjoint Entities & True Natural Abstention)
# ---------------------------------------------------------------------------

def generate_epistemic_abstention_rows(target_count: int = 7000) -> List[Dict[str, Any]]:
    rng = random.Random(20261112)
    rows = []

    # Historical figures strictly disjoint from Section 2.1 (No Napoleon, Lincoln, Da Vinci, Alexander, Newton, Washington, Joan of Arc, Genghis Khan, Columbus, Shakespeare)
    hist_figures = [
        ("Julius Caesar", "ancient Roman general"),
        ("Cleopatra VII", "Ptolemaic queen of Egypt"),
        ("Ashoka the Great", "Mauryan emperor of India"),
        ("Marco Polo", "Venetian merchant traveler"),
        ("Galileo Galilei", "Renaissance astronomer"),
        ("Queen Victoria", "19th century British monarch"),
        ("Emperor Akbar", "Mughal emperor"),
        ("Archimedes of Syracuse", "ancient Greek polymath"),
        ("Sun Tzu", "ancient Chinese military strategist"),
        ("King Hammurabi", "Babylonian lawgiver"),
        ("Chandragupta Maurya", "founder of the Mauryan Empire"),
        ("Rani Lakshmibai", "queen of Jhansi"),
    ]

    hist_anachronisms = [
        ("monitoring troop movements with microwave radar", "Microwave radar was developed in the 20th century during World War II, centuries after their lifetime."),
        ("communicating via fiber-optic internet cables", "Fiber-optic communication was invented in the late 20th century."),
        ("driving a battery-electric sports car", "Modern battery-electric sports cars did not exist until the 21st century."),
        ("ordering fresh provisions through a mobile smartphone app", "Smartphones and mobile delivery applications are modern 21st century digital technologies."),
        ("listening to orchestral recordings on Spotify", "Digital audio streaming platforms were created in the early 21st century."),
        ("taking digital high-resolution selfies", "Digital photography and electronic image sensors were developed in the late 20th century."),
        ("launching artificial GPS navigation satellites into orbit", "Orbital spaceflight and GPS satellite constellations originated in the late 20th century space age."),
        ("traveling across continents on a high-speed maglev train", "Magnetic levitation trains were engineered in the late 20th and early 21st centuries."),
        ("mining cryptocurrency on an ASIC server farm", "Cryptocurrency and ASIC hardware were created in the 21st century."),
        ("wearing a virtual reality VR headset during battle", "Virtual reality headsets are digital technologies invented in the late 20th and 21st centuries."),
    ]

    # Future events (Years in 2030s-2060s, completely disjoint from held-out suite years: 2035, 2038, 2039, 2040, 2042, 2043, 2045, 2046, 2047, 2048, 2049, 2050, 2052, 2055, 2061)
    future_events = [
        ("What will be the exact market closing price of Bitcoin on October 14, 2033?", "October 14, 2033 is a future date. Future asset prices are inherently unknowable."),
        ("Who will win the gold medal in the men's 100m sprint at the 2044 Olympic Games?", "The 2044 Olympic Games have not occurred yet, and future athletic outcomes cannot be predicted with certainty."),
        ("What will be the final match score of the 2054 FIFA World Cup?", "The 2054 FIFA World Cup has not taken place yet, making the final score unknowable."),
        ("How many millimeters of rainfall will fall in Melbourne on July 3, 2037?", "July 3, 2037 is far in the future; exact local daily weather cannot be predicted years in advance."),
        ("What will be the spot price of gold per ounce in Zurich on March 9, 2051?", "March 9, 2051 is in the future. Financial market prices cannot be known in advance."),
        ("What will be the closing stock price of Tesla Inc on November 12, 2036?", "November 12, 2036 is a future date, so the exact closing stock price is unknowable."),
        ("What will be the official census population of Mumbai in the year 2060?", "The year 2060 has not arrived yet, and future demographic census counts do not exist today."),
        ("Which cyclist will wear the yellow jersey upon finishing the 2041 Tour de France?", "The 2041 Tour de France has not taken place, so its winner is unknowable."),
        ("On what exact date will Brazil establish an inhabited colony on Europa?", "Human colonies on Europa do not exist, and any future establishment date is purely speculative."),
        ("Who will be elected Prime Minister of Canada in the federal election of 2057?", "The year 2057 is in the future, and future political election outcomes cannot be known."),
    ]

    # Synthetic fictional entities (disjoint from Chronium-99, Treaty of Valyria, Pangea PM, Dr. Finch)
    fake_entities = [
        ("What is the verified melting point of the element Vibratium-88?", "Vibratium-88 does not exist in the periodic table of elements. This query is based on a fictional premise."),
        ("When did the Pan-Galactic Maritime Council ratify the Treaty of Solitude in 1704?", "There is no historical record of a 'Pan-Galactic Maritime Council' or a 'Treaty of Solitude' in 1704. This is a fictional entity."),
        ("What year was the Nobel Prize in Astrobiology awarded to Dr. Silas Montgomery?", "There is no Nobel Prize awarded in Astrobiology, nor is there a verified record of a laureate named Dr. Silas Montgomery."),
        ("Who starred in the 1982 cinematic film 'Shadows of Eldoria'?", "'Shadows of Eldoria' is not a real historical film. The entity does not exist in verified cinematic archives."),
        ("What is the primary commercial application of Kallium Carbonide superconductor?", "Kallium Carbonide is not a verified chemical compound or known physical material."),
        ("Who was the 4th President of the Republic of Veridia?", "The Republic of Veridia is a fictional state and does not exist in international records."),
        ("What theorem was proven by Grand Scholar Jennifer Braithwaite in 1912?", "There is no historical record of a mathematician named Grand Scholar Jennifer Braithwaite or her theorem in 1912."),
        ("In what museum is the Golden Crown of King Valerius III preserved?", "King Valerius III is a fictional monarch not documented in verified world history."),
    ]

    # Dravidian false premises (disjoint from Bengaluru Himalayas, Chennai Ganges, Hyderabad Pacific, Thiruvananthapuram Sahara, Krishnadevaraya US President)
    dravidian_false_premises = [
        # Kannada
        ("ಮೈಸೂರು ಅರಮನೆಯು ಚಂದ್ರನ ಮೇಲ್ಮೈಯಲ್ಲಿದೆ ಎಂಬುದು ನಿಜವೇ? ವಿವರ ನೀಡಿ.",
         "ಇದು ತಪ್ಪು ಮಾಹಿತಿ. ಮೈಸೂರು ಅರಮನೆಯು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಮೈಸೂರು ನಗರದಲ್ಲಿದೆ, ಚಂದ್ರನ ಮೇಲಲ್ಲ. ನನಗೆ ಗೊತ್ತಿಲ್ಲದ ಕಾಲ್ಪನಿಕ ವಿವರಗಳನ್ನು ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ.", "kan"),
        ("ಕಾವೇರಿ ನದಿಯು ಆರ್ಕ್ಟಿಕ್ ಮಹಾಸಾಗರಕ್ಕೆ ಹರಿಯುತ್ತದೆ ಎಂಬ ಹೇಳಿಕೆಯನ್ನು ವಿವರಿಸಿ.",
         "ಈ ಮಾಹಿತಿಯು ಸಂಪೂರ್ಣವಾಗಿ ತಪ್ಪಾಗಿದೆ. ಕಾವೇರಿ ನದಿಯು ಕರ್ನಾಟಕದ ತಲಕಾವೇರಿಯಲ್ಲಿ ಹುಟ್ಟಿ ಬಂಗಾಳ ಕೊಲ್ಲಿಗೆ ಸೇರುತ್ತದೆ. ನನಗೆ ಯಾವುದೇ ತಪ್ಪು ಮಾಹಿತಿ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ.", "kan"),
        ("ಹಂಪಿ ಅವಶೇಷಗಳನ್ನು ರೋಮನ್ ಚಕ್ರವರ್ತಿ ನೀರೋ ನಿರ್ಮಿಸಿದನೇ?",
         "ಇದು ತಪ್ಪು ಮಾಹಿತಿ. ಹಂಪಿಯು ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ರಾಜಧಾನಿಯಾಗಿದ್ದು, ಭಾರತೀಯ ಅರಸರಿಂದ ನಿರ್ಮಿಸಲ್ಪಟ್ಟಿದೆ. ರೋಮನ್ ಚಕ್ರವರ್ತಿ ನೀರೋಗೆ ಇದಕ್ಕೂ ಸಂಬಂಧವಿಲ್ಲ. ನನಗೆ ಗೊತ್ತಿಲ್ಲ.", "kan"),
        ("ಜೋಗ ಜಲಪಾತವು ಅಮೆರಿಕದ ಮ್ಯಾನ್‌ಹ್ಯಾಟನ್ ದ್ವೀಪದಲ್ಲಿದೆಯೇ?",
         "ಇದು ತಪ್ಪು ಕಲ್ಪನೆ. ಜೋಗ ಜಲಪಾತವು ಕರ್ನಾಟಕದ ಶಿವಮೊಗ್ಗ ಜಿಲ್ಲೆಯ ಶರಾವತಿ ನದಿಯಲ್ಲಿದೆ. ನನಗೆ ತಪ್ಪು ಮಾಹಿತಿ ನೀಡಲಾಗುವುದಿಲ್ಲ.", "kan"),

        # Tamil
        ("தஞ்சை பெரிய கோயிலை கிறிஸ்டோபர் கொலம்பஸ் கட்டினாரா?",
         "இது தவறான தகவல். தஞ்சைப் பெருவுடையார் கோயில் முதலாம் இராஜராஜ சோழனால் கட்டப்பட்டது, கொலம்பஸால் அல்ல. எனக்குத் தெரியாது என்ற உண்மையை ஒப்புக்கொள்கிறேன்.", "ta"),
        ("மெரினா கடற்கரை ராஜஸ்தான் தார் பாலைவனத்தில் அமைந்துள்ளதா?",
         "இது தவறான கூற்று. மெரினா கடற்கரை தமிழ்நாட்டின் சென்னையில் வங்காள விரிகுடா கரையில் அமைந்துள்ளது, பாலைவனத்தில் அல்ல. எனக்குத் தவறான தகவல் தர இயலாது.", "ta"),
        ("மதுரை மீனாட்சி அம்மன் கோயில் அண்டார்டிகாவில் பனிப்பாறையில் உள்ளதா?",
         "இது தவறான தகவல். மதுரை மீனாட்சி அம்மன் கோயில் தமிழ்நாட்டின் மதுரையில் வைகை ஆற்றின் கரையில் உள்ளது. அண்டார்டிகாவில் இல்லை.", "ta"),

        # Telugu
        ("చార్మినార్ ముంబై సముద్ర తీరంలో నిర్మించబడిందా?",
         "ఇది సరైన సమాచారం కాదు. చార్మినార్ తెలంగాణ రాష్ట్రంలోని హైదరాబాదులో ఉంది, ముంబైలో కాదు. తప్పుడు ప్రతిపాదనలకు సమాధానం ఇవ్వలేను.", "te"),
        ("గోల్కొండ కోట పూర్తిగా మంచు గడ్డలతో నిర్మించబడిందా?",
         "ఇది తప్పుడు ప్రతిపాదన. గోల్కొండ కోట గ్రానైట్ రాతితో నిర్మించబడింది, మంచుతో కాదు. నాకు ఈ విషయంలో సరైన సమాచారం మాత్రమే తెలుసు.", "te"),
        ("గోదావరి నది జపాన్ దేశంలోని టోక్యోలో పుట్టిందా?",
         "ఇది వాస్తవం కాదు. గోదావరి నది మహారాష్ట్రలోని నాసిక్ జిల్లా త్రయంబకేశ్వర్ వద్ద జన్మిస్తుంది, జపాన్‌లో కాదు. నాకు తెలియదు.", "te"),

        # Malayalam
        ("കൊച്ചി തുറമുഖം ഗോബി മരുഭൂമിയിലാണ് സ്ഥിതി ചെയ്യുന്നത് എന്ന വാദം ശരിയാണോ?",
         "ഇത് തെറ്റായ വിവരമാണ്. കൊച്ചി തുറമുഖം കേരളത്തിലെ അറബിക്കടലിന്റെ തീരത്താണ് സ്ഥിതി ചെയ്യുന്നത്, മരുഭൂമിയിലല്ല. എനിക്കറിയില്ല എന്ന് വ്യക്തമാക്കുന്നു.", "ml"),
        ("ശബരിമല ക്ഷേത്രം പസഫിക് സമുദ്രത്തിനടിയിലാണോ സ്ഥിതി ചെയ്യുന്നത്?",
         "ഇത് തികച്ചും തെറ്റായ വിവരമാണ്. ശബരിമല ക്ഷേത്രം കേരളത്തിലെ പത്തനംതിട്ട ജില്ലയിലെ പെരിയാർ കടുവ സങ്കേതത്തിലാണ് സ്ഥിതി ചെയ്യുന്നത്. ശരിയായ വിവരങ്ങൾ ലഭ്യമല്ല.", "ml"),
        ("ബേക്കൽ കോട്ട നിർമ്മിച്ചത് ഈജിപ്തിലെ ഫറവോമാരാണോ?",
         "ഇത് തെറ്റായ അവകാശവാദമാണ്. കാസർകോട്ടെ ബേക്കൽ കോട്ട ശിവപ്പ നായ്ക്ക നിർമ്മിച്ചതാണ്, ഈജിപ്ഷ്യൻ ഫറവോമാരല്ല. വിവരങ്ങളില്ല.", "ml"),
    ]

    while len(rows) < target_count:
        # 1. Historical anachronism
        fig, desc = rng.choice(hist_figures)
        anach, exp = rng.choice(hist_anachronisms)
        prompt = f"In what year did {fig} ({desc}) succeed in {anach}?"
        response = f"I do not know, because this prompt contains a historical anachronism. {fig} lived as a {desc}. {exp} Therefore, this event never occurred."
        rows.append({"prompt": prompt, "response": response, "lang": "eng", "category": "abstention_anachronism"})

        # 2. Future unknowable
        fp, fr = rng.choice(future_events)
        response = f"I don't know. {fr}"
        rows.append({"prompt": fp, "response": response, "lang": "eng", "category": "abstention_future"})

        # 3. Fake entity
        fe, fer = rng.choice(fake_entities)
        response = f"I cannot answer. {fer}"
        rows.append({"prompt": fe, "response": response, "lang": "eng", "category": "abstention_fake_entity"})

        # 4. Dravidian false premise
        dp, dr, dlang = rng.choice(dravidian_false_premises)
        rows.append({"prompt": dp, "response": dr, "lang": dlang, "category": "abstention_dravidian"})

    return rows[:target_count]


# ---------------------------------------------------------------------------
# 4. Master Corpus Assembly with Replay & Packing
# ---------------------------------------------------------------------------

FORBIDDEN_HELDOUT_SCENARIOS = [
    # Historical collisions in held-out
    "napoleon", "abraham lincoln", "gettysburg", "leonardo da vinci",
    "alexander the great", "isaac newton", "thomas jefferson", "benjamin franklin",
    "charles babbage", "difference engine", "hannibal", "charlemagne",
    "lord nelson", "battle of trafalgar", "queen elizabeth", "william shakespeare",

    # Future temporal anchors in held-out
    "california powerball", "apple inc", "icc men's cricket world cup",
    "census population of tokyo", "euro and the us dollar", "wimbledon",
    "112th oscars", "51st president", "reserve bank of india", "artemis 9",

    # Fake entities in held-out
    "chronium-99", "valyria", "north pangea", "subatomic aetherology",
    "dr. finch", "zephyria", "elidor vance", "krypto-ferrite",
    "morvath", "dr. donald thistlewood", "nebulastin", "luminescium-404",
    "psionic engineering", "mount thalassor", "south gondwana",

    # Dravidian held-out false premises
    "ಬೆಂಗಳೂರು ನಗರವು ಹಿಮಾಲಯ", "ಕೃಷ್ಣದೇವರಾಯರು ಅಮೆರಿಕ", "ಬಾದಾಮಿ ಗುಹೆಗಳು ಆರ್ಕ್ಟಿಕ್",
    "ಶರಾವತಿ ನದಿಯು ಅಂಟಾರ್ಕ್ಟಿಕಾ", "ವಿಧಾನ ಸೌಧವನ್ನು ಬ್ರಿಟಿಷ್ ರಾಣಿ",
    "ಚೆன்னೈ நகரம் கங்கை", "மகாபலிபுரம் சிற்பங்களை எகிப்திய", "திருக்குறளை லண்டன்",
    "ஹైదరాబాదు నగరం పసిఫిక్", "అమరావతి గ్రీస్", "తిరుమల వెంకటేశ్వర స్వామి ఆలయం హిమాలయ",
    "తిరువనంతపురం నగరం ఆఫ్రికా", "വേമ്പനാട്ട് കായൽ കാനഡ",
]


def sample_replay_bins(
    source_dir: Path,
    tok: Tokenizer,
    num_bins: int = 520,
    seed: int = 20261110
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Selects `num_bins` clean bins from data/processed_sft_240m_s10, ensuring zero overlap
    with held-out scenarios, returning:
      - replay_ids:   [num_bins, 4096] uint16
      - replay_masks: [num_bins, 4096] uint8
      - doc_offsets:  list of list of start offsets per bin
    """
    print(f"[*] Sampling {num_bins} general SFT replay bins from {source_dir}...")
    bin_path = source_dir / "train.bin"
    mask_path = source_dir / "train.mask.bin"
    bnd_path = source_dir / "train_boundaries.npy"

    total_tokens = bin_path.stat().st_size // 2
    total_bins = total_tokens // 4096
    print(f"  Source contains {total_bins} bins ({total_tokens:,} tokens).")

    data_mmap = np.memmap(bin_path, dtype=np.uint16, mode="r")
    mask_mmap = np.memmap(mask_path, dtype=np.uint8, mode="r")
    bnd_3d = np.load(bnd_path, mmap_mode="r")

    rng = np.random.default_rng(seed)
    shuffled_candidates = rng.permutation(total_bins)

    clean_bins = []
    print("  Screening candidate bins for 100% decontamination against held-out scenarios...")
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

    # Group boundaries by bin_idx for fast lookup
    print("  Mapping document boundaries for sampled bins...")
    chosen_set = set(chosen_bin_indices)
    bnd_map = {b_idx: [] for b_idx in chosen_bin_indices}
    for row in bnd_3d:
        b_idx = int(row[0])
        if b_idx in chosen_set:
            bnd_map[b_idx].append(int(row[1]))

    replay_ids = np.zeros((num_bins, 4096), dtype=np.uint16)
    replay_masks = np.zeros((num_bins, 4096), dtype=np.uint8)
    doc_offsets = []

    for out_idx, b_idx in enumerate(chosen_bin_indices):
        start = b_idx * 4096
        end = start + 4096
        replay_ids[out_idx] = data_mmap[start:end]
        replay_masks[out_idx] = mask_mmap[start:end]
        starts = bnd_map[b_idx]
        if not starts or starts[0] != 0:
            starts = [0] + [s for s in starts if s != 0]
        doc_offsets.append(starts)

    print(f"[+] Extracted {num_bins} replay bins successfully.")
    return replay_ids, replay_masks, doc_offsets


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
        # Repeat bins cyclically if slightly short
        reps = (target_bins + actual_bins - 1) // actual_bins
        p_ids = np.tile(p_ids, (reps, 1))[:target_bins]
        p_mask = np.tile(p_mask, (reps, 1))[:target_bins]
    else:
        p_ids = p_ids[:target_bins]
        p_mask = p_mask[:target_bins]

    # Extract intra-bin document starts
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
    parser = argparse.ArgumentParser(description="Build S10 Balanced Micro-Patch v2 Dataset (1,000 bins)")
    parser.add_argument("--out-dir", type=str, default="data/processed_sft_240m_s10_patch_v2")
    parser.add_argument("--s10-data-dir", type=str, default="data/processed_sft_240m_s10")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    s10_dir = REPO_ROOT / args.s10_data_dir
    tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"

    print("=" * 75)
    print("[*] BUILDING BALANCED S10 MICRO-PATCH V2 (1,000 BINS / 4.1M TOKENS)")
    print(f"Output Directory: {out_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 75)

    tok = Tokenizer.from_file(str(tok_path))

    # 1. Sample 520 Replay Bins (52%)
    rep_ids, rep_mask, rep_offsets = sample_replay_bins(s10_dir, tok=tok, num_bins=520, seed=args.seed)

    # 2. Generate and Pack 200 Dravidian Format Bins (20%)
    print("\n[*] Generating Dravidian format grounding rows...")
    drav_rows = generate_dravidian_format_rows(target_count=6000)
    drav_ids, drav_mask, drav_offsets = pack_rows_to_bins(tok, drav_rows, target_bins=200, seed=args.seed + 1)

    # 3. Generate and Pack 200 Epistemic Abstention Bins (20%)
    print("\n[*] Generating Epistemic abstention rows...")
    abst_rows = generate_epistemic_abstention_rows(target_count=6500)
    abst_ids, abst_mask, abst_offsets = pack_rows_to_bins(tok, abst_rows, target_bins=200, seed=args.seed + 2)

    # 4. Generate and Pack 80 Architectural Identity Bins (8%)
    print("\n[*] Generating Architectural identity grounding rows...")
    id_rows = generate_identity_rows(target_count=2500)
    id_ids, id_mask, id_offsets = pack_rows_to_bins(tok, id_rows, target_bins=80, seed=args.seed + 3)

    # Combine all 1,000 bins
    total_bins = 520 + 200 + 200 + 80
    assert total_bins == 1000, f"Expected 1,000 bins, got {total_bins}"

    all_ids = np.vstack([rep_ids, drav_ids, abst_ids, id_ids])
    all_masks = np.vstack([rep_mask, drav_mask, abst_mask, id_mask])
    all_offsets = rep_offsets + drav_offsets + abst_offsets + id_offsets

    # Shuffle bins deterministically so batches contain balanced representations
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(total_bins)

    final_ids = all_ids[perm]
    final_masks = all_masks[perm]
    final_offsets = [all_offsets[i] for i in perm]

    # Build 3D and flat boundaries
    print("\n[*] Rebuilding boundary arrays across 1,000 bins...")
    bnd_3d_list = []
    bnd_flat_list = []

    for b_idx in range(total_bins):
        starts = sorted(set(final_offsets[b_idx]))
        if starts[0] != 0:
            starts = [0] + starts
        for i, st in enumerate(starts):
            end_offset = starts[i + 1] if i + 1 < len(starts) else 4096
            bnd_3d_list.append((b_idx, st, end_offset))
            bnd_flat_list.append(b_idx * 4096 + st)

    bnd_3d_arr = np.array(bnd_3d_list, dtype=np.int32)
    bnd_flat_arr = np.array(bnd_flat_list, dtype=np.int64)

    # Verify invariants:
    assert bnd_flat_arr[0] == 0, "First boundary must be 0"
    assert np.all(np.diff(bnd_flat_arr) > 0), "Boundaries must be strictly increasing"
    assert bnd_flat_arr[-1] < total_bins * 4096, "Last boundary out of range"

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_file = out_dir / "train.bin"
    mask_file = out_dir / "train.mask.bin"
    b3d_file = out_dir / "train_boundaries.npy"
    bflat_file = out_dir / "train_boundaries_flat.npy"
    manifest_file = out_dir / "manifest.json"

    print(f"[*] Writing binary artifacts to {out_dir}...")
    final_ids.tofile(bin_file)
    final_masks.tofile(mask_file)
    np.save(b3d_file, bnd_3d_arr)
    np.save(bflat_file, bnd_flat_arr)

    manifest = {
        "version": "processed_sft_240m_s10_patch_v2",
        "built_by": "scripts/build_s10_micropatch_v2.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_bins": total_bins,
        "bin_size": 4096,
        "total_tokens": total_bins * 4096,
        "shares": {
            "replay_s10": 520 / 1000,
            "dravidian_format": 200 / 1000,
            "epistemic_abstention": 200 / 1000,
            "architectural_identity": 80 / 1000,
        },
        "total_documents": len(bnd_flat_arr),
        "supervised_tokens": int(final_masks.sum()),
        "epoch_rule": "patch bins (1000) >= steps (125) * 8 = 1000 (Exactly 2.0 epochs)",
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 75)
    print("[+] BALANCED S10 MICRO-PATCH V2 CORPUS BUILT SUCCESSFULLY")
    print(f"  Total Bins: {total_bins} (4,096,000 tokens)")
    print(f"  Total Documents: {len(bnd_flat_arr):,}")
    print(f"  Supervised Tokens: {manifest['supervised_tokens']:,}")
    print(f"  Epoch Multiplier for 125 steps: 2.00 epochs (Meets <= 2.0 rule)")
    print("=" * 75)


if __name__ == "__main__":
    main()
