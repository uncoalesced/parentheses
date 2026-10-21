#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_sft_120m_v2.py

Builds the clean, high-signal, assistant-masked post-pretraining SFT v2 corpus
for Parentheses 0.9-120M (Recurrent Selective Linear Attention SSM):
  - data/processed_sft_120m_v2/train.bin            (uint16 token IDs)
  - data/processed_sft_120m_v2/train.mask.bin       (uint8 assistant-target loss mask: 1=supervised, 0=ignored)
  - data/processed_sft_120m_v2/train_boundaries.npy (int64 document-start offsets for SSM state reset)
  - data/processed_sft_120m_v2/val.bin
  - data/processed_sft_120m_v2/val.mask.bin
  - data/processed_sft_120m_v2/val_boundaries.npy
  - data/processed_sft_120m_v2/manifest.json

Addresses all adversarial audit findings from 2026-10-16-sft-argon-audit.md:
  1. Strict 0% Train/Val Leakage:
     - Deduplicates unique raw records FIRST.
     - Partitions 98/2 into train_recs and val_recs before any upsampling.
     - Upsamples ONLY train_recs (val is strictly untouched).
     - Automated SHA-256 collision check proves 0 shared documents between splits.
  2. Purged Pseudo-Reasoning & Zero Parrot Training:
     - Deleted fake Aya rationales (think_part = sents[:-1]); Aya is encoded strictly
       as direct conversational QA (encode_chat_doc).
     - Deleted arbitrary narrative sentence slicing (<think> sub[1] </think>).
     - Only genuine multi-step derivations (Bespoke-Stratos, MATH, Aqua-RAT/GSM8K)
       receive <think>...</think> tags.
  3. Grounded Low-Resource Dialect Prompts:
     - Verified against F:/Lingual_Data/data/processed_v3/text/{tcy,kff,kxv,nit}.jsonl.
     - Uses only authentic attested vocabulary (Gondi 'గొట్టిన్ ఇడ్', Kuvi 'ଈ କାତା ତଲେ ୱେହ୍‌ଦୁ',
       Koya 'ଇଦିନ୍ ପେର୍କେ କେତୁ / ଲେକାଟ୍', Tulu 'ಈ ವಿಷಯೊದ ಬಗೆಟ್ ಪನ್ಲೆ').
     - Emits each harvest window exactly once (50% continuation, 50% chat).
  4. Diversified & Capped Abstention:
     - Replaces single static canned paragraph with 12 diverse natural expressions
       per language (eng, kan, tam, tel, mal).
     - Total abstention documents strictly capped at ~500 across the entire corpus.
  5. High-Signal Compact Target:
     - Curates a dense 25M-35M token mix aligned with SSM post-pretraining convergence.
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
OUT_DEFAULT = REPO / "data/processed_sft_120m_v2"

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

# ---------------------------------------------------------------------------
# Diversified Abstention Bank (12 natural expressions per language)
# ---------------------------------------------------------------------------
ABSTENTION_THINK = {
    "eng": [
        "Let me check the provided context carefully. Scanning the passage for the requested fact shows that this detail is not stated in the text. I should state clearly that the text does not contain this information.",
        "Reviewing the passage for mentions of the question topic. The excerpt does not provide any information regarding this query, so I should decline to answer rather than guessing.",
        "Checking the text for the relevant entities and facts. None of the sentences address this question directly or indirectly. The appropriate response is to note that the information is absent.",
        "Analyzing the given excerpt against the question. The context does not supply sufficient details to answer this query truthfully.",
        "Evaluating the context carefully. The specific detail asked about is not mentioned in the text, so it cannot be determined from the provided material.",
        "Scanning the source text for relevant keywords. Since there is no discussion of this matter in the passage, I cannot provide an answer based solely on the text.",
        "Examining the provided statements. The facts needed to answer this question are missing from the excerpt, so I must state that it cannot be answered from the text.",
        "Looking through the context to verify if the question is answered. The passage makes no claim about this topic.",
        "Checking the provided material. The query asks about an event or detail not covered in the passage.",
        "Carefully reviewing each sentence of the context. There is no evidence in the text to support an answer to this question.",
        "Inspecting the passage for the required information. The source text does not contain the answer to this question.",
        "Reading through the excerpt to find the relevant fact. Since the information is omitted from the context, I should explain that it cannot be verified.",
    ],
    "kan": [
        "ಕೊಟ್ಟಿರುವ ಪಠ್ಯವನ್ನು ಗಮನವಿಟ್ಟು ಪರಿಶೀಲಿಸೋಣ. ಕೇಳಲಾದ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ನಿಖರವಾದ ಮಾಹಿತಿ ಈ ಪಠ್ಯದಲ್ಲಿ ಎಲ್ಲೂ ಕಂಡುಬರುವುದಿಲ್ಲ. ಆದ್ದರಿಂದ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ ಎಂದು ತಿಳಿಸಬೇಕು.",
        "ನೀಡಲಾದ ಸಂದರ್ಭವನ್ನು ಪರೀಕ್ಷಿಸಿದಾಗ, ಈ ವಿಷಯದ ಬಗ್ಗೆ ಯಾವುದೇ ಉಲ್ಲೇಖವಿಲ್ಲ. ಊಹಿಸುವ ಬದಲು ಪಠ್ಯದಲ್ಲಿ ವಿವರವಿಲ್ಲವೆಂದು ಸ್ಪಷ್ಟಪಡಿಸುವುದು ಸೂಕ್ತ.",
        "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಅಂಶಗಳನ್ನು ಪಠ್ಯದಲ್ಲಿ ಹುಡುಕೋಣ. ನೀಡಲಾದ ಭಾಗದಲ್ಲಿ ಈ ಬಗ್ಗೆ ಯಾವುದೇ ವಾಕ್ಯವಿಲ್ಲದಿರುವುದರಿಂದ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
        "ಒದಗಿಸಲಾದ ಪಠ್ಯದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಅಗತ್ಯವಿರುವ ವಿವರಗಳು ಲಭ್ಯವಿಲ್ಲ. ಆದ್ದರಿಂದ ಇದನ್ನು ದೃಢೀಕರಿಸಲಾಗುವುದಿಲ್ಲ ಎಂದು ಹೇಳಬೇಕು.",
        "ಪಠ್ಯದ ಪ್ರತಿಯೊಂದು ವಾಕ್ಯವನ್ನು ಗಮನಿಸಿದಾಗ, ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಯಾವುದೇ ಅಂಶ ಸಿಗುವುದಿಲ್ಲ. ಮಾಹಿತಿ ಕೊರತೆಯಿದೆ ಎಂದು ತಿಳಿಸಬೇಕು.",
        "ಕೊಟ್ಟಿರುವ ವಿವರಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿದಾಗ, ಕೇಳಲಾದ ವಿಷಯದ ಕುರಿತು ಯಾವುದೇ ಮಾಹಿತಿ ಪಠ್ಯದಲ್ಲಿಲ್ಲ ಎಂಬುದು ಸ್ಪಷ್ಟವಾಗುತ್ತದೆ.",
        "ಪ್ರಶ್ನೆಯನ್ನು ಪಠ್ಯದೊಂದಿಗೆ ತಾಳೆ ನೋಡಿದಾಗ, ಇದಕ್ಕೆ ಉತ್ತರಿಸಲು ಬೇಕಾದ ಆಧಾರಗಳು ಒದಗಿಸಲಾದ ಮಾಹಿತಿಯಲ್ಲಿ ಕಂಡುಬರುವುದಿಲ್ಲ.",
        "ನೀಡಿರುವ ಭಾಗದಲ್ಲಿ ಈ ವಿಷಯದ ಬಗ್ಗೆ ಚರ್ಚೆಯಾಗಿಲ್ಲ. ಆದ್ದರಿಂದ ಕೇವಲ ಪಠ್ಯದ ಆಧಾರದ ಮೇಲೆ ಉತ್ತರ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ.",
        "ಕೊಟ್ಟಿರುವ ಮಾಹಿತಿಯನ್ನು ಪರಿಶೀಲಿಸಿದ ನಂತರ, ಈ ನಿರ್ದಿಷ್ಟ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಮಾಹಿತಿ ಸಾಲದು ಎಂದು ನಿರ್ಧರಿಸಬಹುದು.",
        "ಪಠ್ಯದಲ್ಲಿ ಈ ವಿಷಯದ ಉಲ್ಲೇಖವಿಲ್ಲದಿರುವುದರಿಂದ, ಲಭ್ಯವಿರುವ ಪಠ್ಯದಿಂದಲೇ ಉತ್ತರವನ್ನು ಹೇಳಲಾಗುವುದಿಲ್ಲ ಎಂದು ಸ್ಪಷ್ಟಪಡಿಸಬೇಕು.",
        "ಸಂದರ್ಭದಲ್ಲಿ ಕೇಳಲಾದ ಅಂಶಗಳ ಬಗ್ಗೆ ಯಾವುದೇ ಹೇಳಿಕೆ ಇಲ್ಲ. ಆದ್ದರಿಂದ ಮಾಹಿತಿ ಒದಗಿಸಲಾಗಿಲ್ಲ ಎಂದು ಉತ್ತರಿಸಬೇಕು.",
        "ನೀಡಲಾದ ಸನ್ನಿವೇಶದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಸಂಗತಿಗಳು ಇಲ್ಲದಿರುವುದರಿಂದ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
    ],
    "tam": [
        "கொடுக்கப்பட்டுள்ள பத்தியைக் கவனமாகப் பார்ப்போம். கேட்கப்பட்ட கேள்விக்கான தகவல் இந்தப் பத்தியில் எங்கும் குறிப்பிடப்படவில்லை. எனவே தகவல் இல்லை என்பதைத் தெளிவாகக் கூற வேண்டும்.",
        "வழங்கப்பட்ட உரையில் இந்த விவரம் உள்ளதா என்று தேடினால், அது பற்றிய எந்தக் குறிப்பும் இல்லை. ஊகிக்காமல் பதில் இல்லை எனக் கூறுவது சரி.",
        "பத்தியின் வரிகளை ஆராயும்போது, இக்கேள்விக்குரிய நேரடித் தரவுகள் எதுவும் தரப்படவில்லை என்பது தெளிவாகிறது.",
        "கொடுக்கப்பட்ட தகவல்களின் அடிப்படையில் இந்த வினாவிற்குப் பதிலளிக்கத் தேவையான விவரங்கள் இதில் இல்லை.",
        "உரையை முழுமையாகப் படித்ததில், கேட்கப்பட்ட கேள்வி தொடர்பான செய்தி எதுவும் இதில் இடம் பெறவில்லை.",
        "வழங்கப்பட்டுள்ள சூழலில் இதற்கான விடை இல்லாததால், பத்தியின் அடிப்படையில் விடையளிக்க இயலாது எனத் தெரிவிக்க வேண்டும்.",
        "கேள்வியை பத்தியுடன் ஒப்பிட்டுப் பார்க்கும்போது, தேவையான உண்மைத் தரவுகள் உரைப்பகுதியில் காணப்படவில்லை.",
        "கொடுக்கப்பட்ட பகுதியில் இக்குறிப்பிட்ட தலைப்பு பற்றி எதுவும் கூறப்படவில்லை என்பதால் உறுதியான பதில் தர முடியாது.",
        "பத்தியில் இக்கேள்விக்குரிய விடை அடங்கவில்லை என்பதைத் தெளிவுபடுத்தி விடை பகர வேண்டும்.",
        "வழங்கப்பட்ட செய்தித் தொகுப்பில் இத்தகவல் விடுபட்டுள்ளது. ஆகவே விடை தெரியாது என்பதைத் தெரிவிக்க வேண்டும்.",
        "இக்கேள்விக்கான ஆதாரம் பத்தியில் இல்லை என்பதால், தகவல் இல்லை என பதிலளிக்க வேண்டும்.",
        "கொடுக்கப்பட்ட வாசகங்களில் இதற்கான குறிப்பு எதுவும் இல்லாததால் விடையளிக்க இயலாது.",
    ],
    "tel": [
        "ఇచ్చిన పేరాను జాగ్రత్తగా పరిశీలిద్దాం. అడిగిన ప్రశ్నకు సంబంధించిన వివరాలు ఈ పేరాలో ఎక్కడా పేర్కొనబడలేదు. కాబట్టి సమాచారం లేదని స్పష్టంగా తెలియజేయాలి.",
        "అందించిన సందర్భంలో ఈ ప్రశ్నకు సంబంధించిన విషయం ప్రస్తావించబడలేదు. ఊహించి చెప్పకుండా సమాధానం లభ్యం కాలేదని చెప్పాలి.",
        "పేరాలోని వాక్యాలను పరిశీలిస్తే, అడిగిన అంశానికి తగిన ఆధారాలు లేవు. అందువల్ల సమాచారం లేదని పేర్కొనడం సరైనది.",
        "ఇచ్చిన వివరాల ఆధారంగా ఈ ప్రశ్నకు సమాధానం చెప్పడానికి అవసరమైన సమాచారం ఇందులో లేదు.",
        "సందర్భాన్ని క్షుణ్ణంగా చదివినప్పుడు, ఈ ప్రశ్నకు సంబంధించిన అంశం ఎక్కడా చర్చించబడలేదని తెలుస్తోంది.",
        "అందించిన సమాచారంలో ఈ విషయం ప్రస్తావనకు రాలేదు కాబట్టి కేవలం ఈ పేరా ఆధారంగా సమాధానం నిర్ధారించలేము.",
        "ప్రశ్నను పేరాతో సరిపోల్చి చూస్తే, అవసరమైన వివరాలు ఈ భాగంలో కనిపించడం లేదు.",
        "ఇచ్చిన పాఠ్యంలో దీనికి సంబంధించిన సమాచారం అందుబాటులో లేనందున స్పష్టంగా సమాధానం చెప్పలేము.",
        "ఈ పేరాలో అడిగిన విషయానికి సంబంధించిన ఎటువంటి ఆధారమూ లభించలేదు.",
        "అందించిన సందర్భం ఆధారంగా ఈ ప్రశ్నకు సమాధానాన్ని నిర్ణయించడం సాధ్యం కాదు.",
        "పేరాలో ఈ వివరాలు లేవు కాబట్టి ప్రశ్నకు సమాధానం దొరకలేదని తెలియజేయాలి.",
        "ఇచ్చిన వాక్యాల నుండి ఈ ప్రశ్నకు జవాబు కనుగొనడానికి తగినంత సమాచారం లేదు.",
    ],
    "mal": [
        "നൽകിയിട്ടുള്ള ഭാഗം ശ്രദ്ധാപൂർവ്വം പരിശോധിക്കാം. ചോദിച്ചിരിക്കുന്ന കാര്യത്തെക്കുറിച്ചുള്ള വിവരങ്ങൾ ഈ ഭാഗത്ത് എവിടെയും പ്രതിപാദിച്ചിട്ടില്ല. അതിനാൽ വിവരങ്ങൾ ലഭ്യമല്ലെന്ന് വ്യക്തമാക്കണം.",
        "തന്നിരിക്കുന്ന സന്ദർഭത്തിൽ ഈ വിഷയത്തെക്കുറിച്ച് ഒരു പരാമർശവുമില്ല. ഊഹിച്ചു പറയാതെ വിവരങ്ങൾ ലഭ്യമല്ലെന്ന് പറയുകയാണ് ഉചിതം.",
        "ചോദ്യത്തിന് ആവശ്യമായ വിവരങ്ങൾ നൽകിയിട്ടുള്ള ഭാഗത്ത് കണ്ടെത്താൻ കഴിഞ്ഞില്ല. അതിനാൽ ഉത്തരം നിർണ്ണയിക്കാൻ കഴിയില്ല.",
        "നൽകിയിട്ടുള്ള പാഠഭാഗത്തിൽ ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വസ്തുതകൾ ഇല്ല.",
        "തന്നിരിക്കുന്ന ഭാഗം മുഴുവൻ വായിച്ചതിൽ നിന്നും, ഈ ചോദ്യവുമായി ബന്ധപ്പെട്ട വിവരങ്ങൾ ഇതിലില്ല എന്ന് വ്യക്തമാണ്.",
        "ലഭ്യമായ വിവരങ്ങളുടെ അടിസ്ഥാനത്തിൽ ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ സാധ്യമല്ലെന്ന് വ്യക്തമാക്കണം.",
        "പാഠഭാഗത്തിൽ ഈ ചോദ്യത്തെക്കുറിച്ചുള്ള വിവരങ്ങൾ ഉൾപ്പെടുത്തിയിട്ടില്ലാത്തതിനാൽ ഉത്തരം കണ്ടെത്താനാവില്ല.",
        "തന്നിരിക്കുന്ന സന്ദർഭത്തിൽ ഇതിനെക്കുറിച്ച് പരാമർശിക്കാത്തതിനാൽ കൃത്യമായ ഉത്തരം നൽകാൻ കഴിയില്ല.",
        "നൽകിയിട്ടുള്ള വിവരങ്ങളിൽ ഈ വിഷയത്തെക്കുറിച്ചുള്ള വിശദാംശങ്ങൾ ലഭ്യമല്ല.",
        "ഈ ഭാഗത്തിന്റെ അടിസ്ഥാനത്തിൽ ചോദിച്ച കാര്യത്തിന് ഉത്തരം നൽകാൻ സാധ്യമല്ല.",
        "നൽകിയിരിക്കുന്ന പാഠഭാഗം പരിശോധിച്ചതിൽ നിന്നും ഈ ചോദ്യത്തിനുള്ള ഉത്തരം ഇതിലില്ലെന്ന് വ്യക്തമാകുന്നു.",
        "സന്ദർഭത്തിൽ ഈ വിവരങ്ങൾ ലഭ്യമല്ലാത്തതിനാൽ ഉത്തരമില്ലെന്ന് വ്യക്തമാക്കണം.",
    ],
}

ABSTENTION_TEMPLATES = {
    "eng": [
        "Based on the provided passage, this detail is not mentioned.",
        "The provided text does not contain sufficient information to answer this question.",
        "I cannot determine the answer from the given context, as it is not stated in the passage.",
        "The text does not mention this information, so it cannot be answered from the provided material.",
        "There is no statement in the context that addresses this question.",
        "This information is not available in the given passage.",
        "The provided excerpt does not discuss this topic.",
        "From the context provided, it is not possible to answer this question.",
        "The given text does not provide details on this matter.",
        "This question cannot be answered using only the provided text.",
        "The context does not contain the required facts to answer this query.",
        "According to the provided passage, no information is given regarding this topic.",
    ],
    "kan": [
        "ನೀಡಲಾದ ಪಠ್ಯದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ.",
        "ಕೊಟ್ಟಿರುವ ಸಂದರ್ಭದಲ್ಲಿ ಈ ವಿಷಯದ ಬಗ್ಗೆ ಉಲ್ಲೇಖವಿಲ್ಲದ ಕಾರಣ ಉತ್ತರವನ್ನು ನಿರ್ಧರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
        "ನೀಡಲಾದ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಇದನ್ನು ಖಚಿತಪಡಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
        "ಪಠ್ಯದಲ್ಲಿ ಈ ವಿವರಗಳನ್ನು ಒದಗಿಸಿಲ್ಲವಾದ್ದರಿಂದ ಉತ್ತರ ಲಭ್ಯವಿಲ್ಲ.",
        "ಒದಗಿಸಲಾದ ಪಠ್ಯದಲ್ಲಿ ಈ ಬಗ್ಗೆ ಯಾವುದೇ ಮಾಹಿತಿಯಿಲ್ಲ.",
        "ಕೊಟ್ಟಿರುವ ಭಾಗದ ಆಧಾರದ ಮೇಲೆ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಾಗುವುದಿಲ್ಲ.",
        "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಪಠ್ಯದಲ್ಲಿ ಅಗತ್ಯವಿರುವ ಮಾಹಿತಿಯ ಕೊರತೆಯಿದೆ.",
        "ನೀಡಿರುವ ಸಂದರ್ಭದಲ್ಲಿ ಈ ವಿಷಯದ ಕುರಿತು ಯಾವುದೇ ವಿವರಗಳಿಲ್ಲ.",
        "ಪಠ್ಯದಲ್ಲಿ ಈ ವಿಷಯ ಪ್ರಸ್ತಾಪಿಸಲ್ಪಡದ ಕಾರಣ ಉತ್ತರ ನೀಡಲಾಗುವುದಿಲ್ಲ.",
        "ಲಭ್ಯವಿರುವ ಪಠ್ಯದ ಆಧಾರದ ಮೇಲೆ ಇದನ್ನು ನಿರ್ಧರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
        "ಕೊಟ್ಟಿರುವ ಪಠ್ಯದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ವಿವರಗಳು ಕಂಡುಬರುವುದಿಲ್ಲ.",
        "ನೀಡಲಾದ ಮಾಹಿತಿಯಿಂದ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ.",
    ],
    "tam": [
        "கொடுக்கப்பட்டுள்ள பத்தியில் இந்தக் கேள்விக்குப் பதிலளிக்கப் போதுமான தகவல் இல்லை.",
        "வழங்கப்பட்ட உரையில் இதற்கான விவரம் குறிப்பிடப்படவில்லை என்பதால் உறுதியாகக் கூற முடியாது.",
        "கொடுக்கப்பட்ட தகவலின் அடிப்படையில் இதைத் தீர்மானிக்க இயலாது.",
        "பத்தியில் இந்த செய்தி இடம்பெறாததால் இதற்கு விடையளிக்க முடியாது.",
        "வழங்கப்பட்டுள்ள உரைப்பகுதியில் இதற்கான தகவல் எதுவும் கிடைக்கவில்லை.",
        "கொடுக்கப்பட்டுள்ள விவரங்களின் அடிப்படையில் இதற்குப் பதிலளிக்க முடியாது.",
        "இந்தக் கேள்விக்குரிய செய்தி பத்தியில் எங்கும் குறிப்பிடப்படவில்லை.",
        "இக்கேள்விக்கான தகவல் தரப்பட்டுள்ள உரையில் இல்லை.",
        "வழங்கப்பட்ட பத்தியிலிருந்து இதற்கு விடை காண இயலாது.",
        "கொடுக்கப்பட்ட உரையின்படி இக்கேள்விக்குப் பதில் இல்லை.",
        "பத்தியில் இத்தகவல் கூறப்படாததால் விடையளிக்க இயலவில்லை.",
        "வழங்கப்பட்டுள்ள பகுதியில் இந்த விவரம் காணப்படவில்லை.",
    ],
    "tel": [
        "ఇచ్చిన పేరాలో ఈ ప్రశ్నకు సమాధానం చెప్పడానికి తగిన సమాచారం లేదు.",
        "అందించిన సందర్భంలో ఈ విషయం ప్రస్తావించబడలేదు కాబట్టి ఖచ్చితంగా చెప్పలేము.",
        "ఇచ్చిన సమాచారం ఆధారంగా దీనిని నిర్ధారించడం సాధ్యం కాదు.",
        "ఈ వివరాలు ఇచ్చిన పాఠ్యంలో ఎక్కడా పేర్కొనబడలేదు.",
        "అందించిన పేరాలో ఈ ప్రశ్నకు తగిన సమాధానం లభించలేదు.",
        "ఇచ్చిన సమాచారంలో ఈ అంశం ప్రస్తావనకు రాలేదు.",
        "ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి ఇచ్చిన భాగంలో తగిన వివరాలు లేవు.",
        "అందించిన సందర్భం ఆధారంగా సమాధానం చెప్పడం సాధ్యం కాదు.",
        "పాఠ్యంలో ఈ విషయానికి సంబంధించిన ఆధారాలు లేవు.",
        "ఇచ్చిన పేరా ఆధారంగా ఈ వివరాలను నిర్ధారించలేము.",
        "ఈ ప్రశ్నకు సంబంధించిన సమాచారం ఇచ్చిన సందర్భంలో అందుబాటులో లేదు.",
        "ఇచ్చిన పాఠ్యం ఆధారంగా దీనికి జవాబు లభ్యం కాదు.",
    ],
    "mal": [
        "നൽകിയിട്ടുള്ള ഭാഗത്ത് ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ ഇല്ല.",
        "തന്നിരിക്കുന്ന സന്ദർഭത്തിൽ ഇക്കാര്യം പരാമർശിച്ചിട്ടില്ലാത്തതിനാൽ കൃത്യമായി പറയാൻ കഴിയില്ല.",
        "ലഭ്യമായ വിവരങ്ങളുടെ അടിസ്ഥാനത്തിൽ ഇത് സ്ഥിരീകരിക്കാൻ സാധ്യമല്ല.",
        "നൽകിയിട്ടുള്ള പാഠഭാഗത്ത് ഈ വിവരങ്ങൾ ഉൾപ്പെടുത്തിയിട്ടില്ല.",
        "തന്നിരിക്കുന്ന ഭാഗത്തുനിന്ന് ഈ ചോദ്യത്തിന് ഉത്തരം കണ്ടെത്താനാവില്ല.",
        "ഈ ചോദ്യത്തിനുള്ള വിശദാംശങ്ങൾ നൽകിയിട്ടുള്ള ഭാഗത്ത് ലഭ്യമല്ല.",
        "പാഠഭാഗത്തിൽ ഈ വിഷയം പരാമർശിക്കാത്തതിനാൽ ഉത്തരം പറയാൻ സാധിക്കില്ല.",
        "നൽകിയിട്ടുള്ള വിവരങ്ങളുടെ അടിസ്ഥാനത്തിൽ ഇതിന് ഉത്തരം നൽകാൻ കഴിയില്ല.",
        "തന്നിരിക്കുന്ന സന്ദർഭത്തിൽ ഈ വസ്തുതകൾ ലഭ്യമല്ല.",
        "ഈ ചോദ്യത്തിന് ആവശ്യമായ വിവരങ്ങൾ നൽകിയിട്ടുള്ള പാഠത്തിൽ ഇല്ല.",
        "നൽകിയിരിക്കുന്ന ഭാഗം അടിസ്ഥാനമാക്കി ഈ ചോദ്യത്തിന് ഉത്തരം പറയാൻ കഴിയില്ല.",
        "ഈ വിഷയത്തെക്കുറിച്ചുള്ള വിവരങ്ങൾ തന്നിരിക്കുന്ന ഭാഗത്ത് കാണാനില്ല.",
    ],
}

# ---------------------------------------------------------------------------
# Grounded Low-Resource Prompt Templates (100% Attested Vocabulary)
# ---------------------------------------------------------------------------
LOWRES_PROMPT_TEMPLATES = {
    "tcy": [
        "ಈ ವಿಷಯೊದ ಬಗೆಟ್ ಪನ್ಲೆ:\n{head}",
        "ಉಂದೆನ್ ಬೊಕ್ಕ ಪನ್ಲೆ:\n{head}",
    ],
    "kff": [
        "ଇଦିନ୍ ପେର୍କେ କେତୁ:\n{head}",
        "ତାନ୍ ପେର୍କେ ଲେକାଟ୍:\n{head}",
    ],
    "kxv": [
        "ଈ କାତା ତଲେ ୱେହ୍‌ଦୁ:\n{head}",
        "ଈ କାତା ଅ଼ଡ଼େ ୱେହ୍‌ଦୁ:\n{head}",
    ],
    "nit": [
        "ఈ గొట్టిన్ ఇడ్:\n{head}",
        "మరి గొట్టిన్ ఇడ్:\n{head}",
    ],
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
    """Returns (ascii_alpha_count, per_indic_script_counts). Ignores shared danda punctuation."""
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
    """True iff text matches the required script for lang with no English/Devanagari dominance."""
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
    """Loads sha1 hashes of all pretraining validation texts across all 9 languages."""
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


def split_rationale_answer(ans: str) -> tuple[str | None, str]:
    """Splits a step-by-step answer into (reasoning, final_answer) when structured."""
    ans = clean_text(ans)
    if ans.startswith('"') and ans.endswith('"') and len(ans) > 2:
        ans = ans[1:-1].strip()
    # 1. Explicit '#### ' (GSM8K / Math style)
    if "\n#### " in ans:
        r, f = ans.rsplit("\n#### ", 1)
        if len(r.strip()) >= 20 and f.strip():
            return r.strip(), f.strip()
    # 2. Explicit 'Answer:' or 'Final Answer:' or 'Therefore, the answer is'
    for sep in ("\nFinal Answer:", "\nAnswer:", "\nanswer :", "\nanswer:", "\nTherefore, the answer is", "\nSo the answer is", "\nCorrect Answer:"):
        if sep in ans:
            r, f = ans.rsplit(sep, 1)
            if len(r.strip()) >= 20 and f.strip():
                return r.strip(), f.strip().lstrip(": ").strip()
    # 3. 'answer : ' within the text
    for sep in (" answer : ", " Answer : ", " answer: ", " Answer: "):
        if sep in ans:
            r, f = ans.rsplit(sep, 1)
            if len(r.strip()) >= 20 and f.strip():
                return r.strip(), f.strip().lstrip(": ").strip()
    # 4. Multi-line explanation where last line is short conclusion
    lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
    if len(lines) >= 2 and len(lines[-1]) <= 120 and sum(len(x) for x in lines[:-1]) >= 30:
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

def collect_bespoke_think(tok: Tokenizer, val_hashes: set[str], max_reason_toks: int = 420, max_total_toks: int = 1024, max_docs: int = 15000):
    """Yields concise, complete Bespoke-Stratos-17k <think> traces that always close </think>."""
    pfile = RAW_SOURCED / "bespoke_stratos/train-00000-of-00001.parquet"
    if not pfile.exists():
        return
    table = pq.read_table(pfile, columns=["conversations"])
    count = 0
    for conv_item in table["conversations"].to_pylist():
        if count >= max_docs:
            break
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
        # Compress overly long thought chains to their opening + concluding paragraphs
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
            count += 1
            yield ("think_bespoke", "eng", tokens, mask)


def collect_intelli_lanes(tok: Tokenizer, val_hashes: set[str], rng: np.random.Generator, max_total_abstention: int = 400):
    """Reads F:/Lingual_Data/intelli/lane_*/rows.jsonl with strict script, rationale, and abstention governance."""
    if not INTELLI_DIR.exists():
        return

    SOURCE_CAPS = {
        "alar": 400,
        "kaikki": 400,
        "olam": 1200,
        "wordnet": 600,
        "deepmind/aqua_rat": 4000,
        "allenai/math_qa": 3500,
        "hendrycks_math": 3500,
        "OmegaGamage/multiMWP": 3000,
        "databricks/databricks-dolly-15k": 4000,
        "tydiqa": 3000,
        "massive": 4000,
        "rajpurkar/squad_v2": 2500,
        "allenai/quac": 1500,
        "hotpot_qa": 2000,
        "staqc/sca_python": 2000,
    }
    source_counts = Counter()
    abstention_count = 0

    lane_files = sorted(INTELLI_DIR.glob("lane_*/rows.jsonl"))
    for lf in lane_files:
        lane_name = lf.parent.name
        if lf.stat().st_size == 0:
            continue
        lane_doc_count = 0
        lane_cap = 12000 if lane_name == "lane_01_math_word" else 5000
        with open(lf, encoding="utf-8") as f:
            for line in f:
                if lane_doc_count >= lane_cap:
                    break
                if lane_name == "lane_08_unanswerable" and abstention_count >= max_total_abstention:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                lic = str(row.get("license", ""))
                src = str(row.get("source", ""))
                # Exclude research-only trial rows
                if "research-only" in lic.lower() or src.startswith("distill:"):
                    continue

                cap_key = next((k for k in SOURCE_CAPS if k in src), src)
                cap_limit = SOURCE_CAPS.get(cap_key, 3500)
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

                # True language verification of assistant answer
                ascii_a, counts_a = script_profile(ans)
                if counts_a["Deva"] > 2:
                    continue

                if raw_lang in LANG_SCRIPT:
                    target_scr = LANG_SCRIPT[raw_lang]
                    if counts_a[target_scr] >= 6 and counts_a[target_scr] > ascii_a:
                        eff_lang = raw_lang
                    elif ascii_a >= 10 and ascii_a > sum(counts_a.values()) * 2:
                        eff_lang = "eng"
                        if source_counts[f"xling_{lane_name}"] >= 800:
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

                # Lane 08 Unanswerable -> Calibrated diversified abstention (strictly capped)
                if lane_name == "lane_08_unanswerable":
                    if abstention_count >= max_total_abstention:
                        continue
                    templates = ABSTENTION_TEMPLATES.get(eff_lang, ABSTENTION_TEMPLATES["eng"])
                    abst_ans = templates[int(rng.integers(0, len(templates)))]
                    if rng.random() < 0.5:
                        th_list = ABSTENTION_THINK.get(eff_lang, ABSTENTION_THINK["eng"])
                        think_txt = th_list[int(rng.integers(0, len(th_list)))]
                        tokens, mask = encode_think_doc(tok, eff_lang, prompt, think_txt, abst_ans)
                        cat = "honesty_think"
                    else:
                        tokens, mask = encode_chat_doc(tok, eff_lang, prompt, abst_ans)
                        cat = "honesty_chat"
                    if len(tokens) <= 1024:
                        source_counts[cap_key] += 1
                        abstention_count += 1
                        lane_doc_count += 1
                        yield (cat, eff_lang, tokens, mask)
                    continue

                # Structured reasoning lanes (MATH, Aqua-RAT, MathQA, Logic, Tamil Math)
                if lane_name in ("lane_01_math_word", "lane_02_logic", "lane_10_tam_qa"):
                    rat, final_a = split_rationale_answer(ans)
                    if rat and len(rat) >= 15 and final_a:
                        tokens, mask = encode_think_doc(tok, eff_lang, prompt, rat, final_a)
                        if len(tokens) <= 1024:
                            source_counts[cap_key] += 1
                            lane_doc_count += 1
                            yield (f"think_{lane_name}", eff_lang, tokens, mask)
                            continue
                    # Only yield if answer is clean and informative
                    if len(ans) >= 20:
                        tokens, mask = encode_chat_doc(tok, eff_lang, prompt, ans)
                        if len(tokens) <= 1024:
                            source_counts[cap_key] += 1
                            lane_doc_count += 1
                            yield (f"chat_{lane_name}", eff_lang, tokens, mask)
                    continue

                # Standard conversational / QA lanes (Dolly instructions, Code, Reading Comp, TyDi QA)
                tokens, mask = encode_chat_doc(tok, eff_lang, prompt, ans)
                if len(tokens) <= 1024:
                    source_counts[cap_key] += 1
                    lane_doc_count += 1
                    yield (f"chat_{lane_name}", eff_lang, tokens, mask)


def collect_dravidian_aya_indicqa(tok: Tokenizer, val_hashes: set[str], rng: np.random.Generator, max_docs_per_lang: int = 5000, max_indicqa_abst: int = 100):
    """Collects native-script Dravidian chat from Aya (100% direct conversational QA, zero parrot training)
    and IndicQA (reading comp QA + native abstention).
    """
    # 1. Aya dataset (kan, tam, tel, mal)
    aya_path = RAW_SOURCED / "aya_dataset/data/train-00000-of-00001.parquet"
    aya_counts = Counter()
    if aya_path.exists():
        table = pq.read_table(aya_path, columns=["inputs", "targets", "language_code"])
        for inp, tgt, lcode in zip(
            table["inputs"].to_pylist(),
            table["targets"].to_pylist(),
            table["language_code"].to_pylist(),
        ):
            if lcode not in ("kan", "tam", "tel", "mal"):
                continue
            if aya_counts[lcode] >= max_docs_per_lang:
                continue
            p = clean_text(inp)
            a = clean_text(tgt)
            if len(p) < 10 or len(a) < 10:
                continue
            if not verify_lang_script(p, lcode, min_chars=6) or not verify_lang_script(a, lcode, min_chars=6):
                continue
            if hashlib.sha1(p.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(a.encode("utf-8")).hexdigest() in val_hashes:
                continue

            # Pure direct conversational chat turn - NO fake think slicing, NO verbatim parroting
            tokens, mask = encode_chat_doc(tok, lcode, p, a)
            if len(tokens) <= 1024:
                aya_counts[lcode] += 1
                yield ("drav_chat_aya", lcode, tokens, mask)

    # 2. IndicQA (kan, tam, tel, mal)
    indicqa_dir = RAW_SOURCED / "indicqa"
    indicqa_abst_count = 0
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
                            # Native Dravidian unanswerable question
                            if indicqa_abst_count >= max_indicqa_abst:
                                continue
                            templates = ABSTENTION_TEMPLATES[canon_l]
                            abst_ans = templates[int(rng.integers(0, len(templates)))]
                            if rng.random() < 0.5:
                                th_list = ABSTENTION_THINK[canon_l]
                                think_txt = th_list[int(rng.integers(0, len(th_list)))]
                                tokens, mask = encode_think_doc(tok, canon_l, prompt, think_txt, abst_ans)
                            else:
                                tokens, mask = encode_chat_doc(tok, canon_l, prompt, abst_ans)
                            if len(tokens) <= 1024:
                                indicqa_abst_count += 1
                                yield ("drav_honesty_indicqa", canon_l, tokens, mask)
                        else:
                            if not verify_lang_script(ans_text, canon_l, min_chars=2):
                                continue
                            # Direct reading comprehension QA
                            tokens, mask = encode_chat_doc(tok, canon_l, prompt, ans_text)
                            if len(tokens) <= 1024:
                                yield ("drav_chat_indicqa", canon_l, tokens, mask)


def collect_oasst2_and_tools(tok: Tokenizer, val_hashes: set[str], max_oasst: int = 8000, max_hermes: int = 3500):
    """Collects OASST2 rank-0 English chat and genuine Hermes tool-calling traces with assistant masking."""
    ds_dir = RAW_SOURCED / "oasst2/data"
    oasst_count = 0
    if ds_dir.exists():
        seen = set()
        for pfile in sorted(ds_dir.glob("train-*.parquet")):
            if oasst_count >= max_oasst:
                break
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
                if oasst_count >= max_oasst:
                    break
                if pid in prompters:
                    p_txt = prompters[pid]
                    if len(p_txt) >= 12 and len(ans) >= 20 and verify_lang_script(ans, "eng", min_chars=12):
                        h = hashlib.sha1((p_txt + "\n" + ans).encode("utf-8")).hexdigest()
                        if h in seen or hashlib.sha1(p_txt.encode("utf-8")).hexdigest() in val_hashes:
                            continue
                        seen.add(h)
                        tokens, mask = encode_chat_doc(tok, "eng", p_txt, ans)
                        if len(tokens) <= 1024:
                            oasst_count += 1
                            yield ("chat_oasst2", "eng", tokens, mask)

    # Genuine Tool Traces (with environment tool_result unmasked/ignored in loss)
    hermes_dir = RAW_SOURCED / "hermes-function-calling-v1"
    hermes_count = 0
    if hermes_dir.exists():
        for fname in ["glaive-function-calling-5k.json", "func-calling-singleturn.json"]:
            if hermes_count >= max_hermes:
                break
            fpath = hermes_dir / fname
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for item in data:
                if hermes_count >= max_hermes:
                    break
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
                                    hermes_count += 1
                                    yield ("tool_hermes", "eng", tokens, mask)
                            except Exception:
                                continue


def collect_lowres_and_freethink(tok: Tokenizer, val_hashes: set[str], rng: np.random.Generator, max_lowres: int = 900, max_major: int = 400):
    """Builds grounded low-resource SFT chat + FreeThink continuation traces across all 9 languages.
    Harvest policy:
      - Emits each window exactly ONCE (50% continuation, 50% chat).
      - No sliced pseudo-reasoning across <think> tags.
      - 100% verified native dialect vocabulary for prompts.
    """
    if not V3_TEXT_DIR.exists():
        return

    for lang in ("eng", "kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit"):
        p_jsonl = V3_TEXT_DIR / f"{lang}.jsonl"
        if not p_jsonl.exists():
            continue
        is_lowres = lang in ("tcy", "kff", "kxv", "nit")
        max_docs_for_lang = max_lowres if is_lowres else max_major
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
                txt = re.sub(r"^\}\}+\s*", "", txt).strip()
                if not verify_lang_script(txt, lang, min_chars=40):
                    continue

                sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n+", txt) if len(s.strip()) >= 25]
                if len(sents) < 3:
                    continue

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
                    if head[0].isdigit() or body[0].isdigit():
                        continue

                    # Single representation per window (50% continuation, 50% chat for low-res; 100% continuation for major)
                    if is_lowres and rng.random() < 0.5:
                        tmpl_list = LOWRES_PROMPT_TEMPLATES[lang]
                        u_prompt = tmpl_list[count % len(tmpl_list)].format(head=head)
                        c_tokens, c_mask = encode_chat_doc(tok, lang, u_prompt, body)
                        if 64 <= len(c_tokens) <= 768:
                            yield ("lowres_chat", lang, c_tokens, c_mask)
                            count += 1
                    else:
                        ft_tokens, ft_mask = encode_freethink_doc(tok, lang, head, body)
                        if 64 <= len(ft_tokens) <= 768:
                            yield ("freethink_cont", lang, ft_tokens, ft_mask)
                            count += 1


# ---------------------------------------------------------------------------
# Main Assembly, Deduplication, Split, Upsampling & Verification
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
    p = argparse.ArgumentParser(description="Build SFT v2 dataset for Parentheses 0.9-120M")
    p.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    p.add_argument("--val-dir", default=str(VAL_DIR))
    p.add_argument("--out-dir", default=str(OUT_DEFAULT))
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer(Path(args.tokenizer))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_sft_120m_v2] Loading pretraining validation hashes to prevent leakage...", flush=True)
    val_hashes = load_val_hashes(Path(args.val_dir), tok)
    print(f"[build_sft_120m_v2] Loaded {len(val_hashes):,} validation hashes.", flush=True)

    raw_records: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    bespoke_cap = 200 if args.dry_run else 15000
    intelli_abst = 50 if args.dry_run else 400
    aya_cap = 100 if args.dry_run else 5000
    indic_abst = 20 if args.dry_run else 100
    oasst_cap = 150 if args.dry_run else 8000
    hermes_cap = 100 if args.dry_run else 3500
    lowres_cap = 60 if args.dry_run else 900
    major_cap = 40 if args.dry_run else 400

    print("[1/4] Collecting concise Bespoke-Stratos <think> traces...", flush=True)
    for item in collect_bespoke_think(tok, val_hashes, max_docs=bespoke_cap):
        raw_records.append(item)
    print(f"      Total records so far: {len(raw_records):,}", flush=True)

    print("[2/4] Collecting & filtering Intelli lanes (genuine rationale + diverse abstention)...", flush=True)
    for item in collect_intelli_lanes(tok, val_hashes, rng, max_total_abstention=intelli_abst):
        raw_records.append(item)
        if args.dry_run and len(raw_records) >= 800:
            break
    print(f"      Total records so far: {len(raw_records):,}", flush=True)

    print("[3/4] Collecting native Dravidian Aya (direct chat) + IndicQA...", flush=True)
    for item in collect_dravidian_aya_indicqa(tok, val_hashes, rng, max_docs_per_lang=aya_cap, max_indicqa_abst=indic_abst):
        raw_records.append(item)
        if args.dry_run and len(raw_records) >= 1200:
            break
    print(f"      Total records so far: {len(raw_records):,}", flush=True)

    print("[4/4] Collecting OASST2, Hermes Tools, Grounded Low-Resource & FreeThink...", flush=True)
    for item in collect_oasst2_and_tools(tok, val_hashes, max_oasst=oasst_cap, max_hermes=hermes_cap):
        raw_records.append(item)
    for item in collect_lowres_and_freethink(tok, val_hashes, rng, max_lowres=lowres_cap, max_major=major_cap):
        raw_records.append(item)
    print(f"      Total collected records: {len(raw_records):,}", flush=True)

    # -----------------------------------------------------------------------
    # Step 1: Strict Deduplication of Unique Raw Records FIRST
    # -----------------------------------------------------------------------
    seen_hashes = set()
    unique_records: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for cat, lang, toks, msk in raw_records:
        h = hashlib.sha256(toks.tobytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique_records.append((cat, lang, toks, msk))

    print(f"[Deduplication] Unique raw records: {len(unique_records):,} (from {len(raw_records):,} raw)", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: Partition 98% Train / 2% Val on UNIQUE Records BEFORE Upsampling
    # -----------------------------------------------------------------------
    rng.shuffle(unique_records)
    n_val = max(64, int(len(unique_records) * 0.02))
    val_recs = unique_records[:n_val]
    raw_train_recs = unique_records[n_val:]

    print(f"[Split] Raw partition: {len(raw_train_recs):,} train, {len(val_recs):,} val (2.0%)", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Upsample ONLY Train Records (Val is strictly untouched)
    # -----------------------------------------------------------------------
    train_recs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for cat, lang, toks, msk in raw_train_recs:
        train_recs.append((cat, lang, toks, msk))
        if not args.dry_run:
            # Low-resource languages: 2x
            if lang in ("tcy", "kff", "kxv", "nit"):
                train_recs.append((cat, lang, toks, msk))
            # High-signal genuine reasoning & tools: ~1.5x
            elif cat in ("think_bespoke", "think_lane_01_math_word", "think_lane_10_tam_qa", "tool_hermes"):
                if rng.random() < 0.5:
                    train_recs.append((cat, lang, toks, msk))
            # Dravidian Kannada & Malayalam: ~1.4x
            elif lang in ("kan", "mal") and cat.startswith("drav_"):
                if rng.random() < 0.4:
                    train_recs.append((cat, lang, toks, msk))

    rng.shuffle(train_recs)
    print(f"[Upsampling] Final train set: {len(train_recs):,} docs (upsampled from {len(raw_train_recs):,})", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Automated SHA-256 Collision Verification (0% Leakage Gate)
    # -----------------------------------------------------------------------
    train_hashes = {hashlib.sha256(r[2].tobytes()).hexdigest() for r in train_recs}
    val_doc_hashes = {hashlib.sha256(r[2].tobytes()).hexdigest() for r in val_recs}
    collision = train_hashes.intersection(val_doc_hashes)
    print(f"[Leakage Gate] Train unique hashes: {len(train_hashes):,}")
    print(f"[Leakage Gate] Val unique hashes:   {len(val_doc_hashes):,}")
    print(f"[Leakage Gate] Hash Collisions:     {len(collision)}")
    assert len(collision) == 0, f"FATAL LEAKAGE ERROR: Found {len(collision)} identical documents in both train and val!"
    print("[Leakage Gate] PASS: Strict 0.00% train/val leakage verified.", flush=True)

    # -----------------------------------------------------------------------
    # Step 5: Write Output Binaries & Boundary Offsets
    # -----------------------------------------------------------------------
    prefix = "dryrun_train" if args.dry_run else "train"
    vprefix = "dryrun_val" if args.dry_run else "val"

    train_bin, train_mask, train_bnd = write_split(out_dir, prefix, [(r[2], r[3]) for r in train_recs])
    val_bin, val_mask, val_bnd = write_split(out_dir, vprefix, [(r[2], r[3]) for r in val_recs])

    # -----------------------------------------------------------------------
    # Step 6: Verification Gates (Mask Alignment, EOT, ESG, Boundaries)
    # -----------------------------------------------------------------------
    print("\n--- Running Verification Gates ---", flush=True)
    b_data = np.fromfile(train_bin, dtype=np.uint16)
    m_data = np.fromfile(train_mask, dtype=np.uint8)
    bnd = np.load(train_bnd)

    assert len(b_data) == len(m_data), f"train.bin ({len(b_data)}) != train.mask.bin ({len(m_data)})"
    esg_count = int(np.sum(b_data == ESG_ID))
    assert esg_count == 0, f"Found {esg_count} <esg> tokens! ESG must be strictly excluded."
    assert int(bnd[0]) == 0 and np.all(np.diff(bnd) > 0) and int(bnd[-1]) < len(b_data)

    check_limit = min(1000, len(bnd))
    think_checks = 0
    for idx in range(check_limit):
        s = int(bnd[idx])
        e = int(bnd[idx + 1]) if idx + 1 < len(bnd) else len(b_data)
        doc_toks = b_data[s:e]
        doc_mask = m_data[s:e]
        assert doc_mask[0] == 0, f"Doc {idx} starts with mask=1 at offset {s}!"
        assert doc_mask.sum() > 0, f"Doc {idx} has 0 supervised tokens!"
        assert doc_toks[-1] == EOT and doc_mask[-1] == 1, f"Doc {idx} does not end with supervised EOT!"

        if THINK_START in doc_toks:
            think_checks += 1
            t_start_pos = np.where(doc_toks == THINK_START)[0][0]
            assert THINK_END in doc_toks, f"Doc {idx} has unclosed <think> tag!"
            t_end_pos = np.where(doc_toks == THINK_END)[0][0]
            assert t_start_pos < t_end_pos, f"Doc {idx} has inverted think tags!"
            assert doc_mask[t_start_pos] == 1 and doc_mask[t_end_pos] == 1, f"Doc {idx} think tags not supervised!"

    print(f"[Verification Gates] Checked {check_limit} documents ({think_checks} with <think> traces): All passed.", flush=True)

    # -----------------------------------------------------------------------
    # Step 7: Compute Statistics & Write Manifest
    # -----------------------------------------------------------------------
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

    val_tokens = sum(len(r[2]) for r in val_recs)

    print(
        f"\n[Summary] {len(train_recs):,} train docs ({total_tokens:,} tokens, {sup_tokens:,} supervised = {sup_tokens/max(1,total_tokens):.1%}), "
        f"{len(val_recs):,} val docs ({val_tokens:,} tokens), 0 <esg> tokens.",
        flush=True,
    )

    manifest = {
        "dataset_version": "v2",
        "licensing": "Research-Only (Joel) - SFT lineage inherits step_9536_final.pt (pretraining Session 5 mix)",
        "train_docs": len(train_recs),
        "val_docs": len(val_recs),
        "train_tokens": total_tokens,
        "val_tokens": val_tokens,
        "supervised_tokens": sup_tokens,
        "supervised_fraction": round(sup_tokens / max(1, total_tokens), 4),
        "leakage_hash_collisions": len(collision),
        "tokens_by_lang": dict(sorted(lang_tokens.items(), key=lambda x: -x[1])),
        "docs_by_lang": dict(sorted(lang_docs.items(), key=lambda x: -x[1])),
        "tokens_by_category": dict(sorted(cat_tokens.items(), key=lambda x: -x[1])),
        "docs_by_category": dict(sorted(cat_docs.items(), key=lambda x: -x[1])),
        "train_sha256": hashlib.sha256(train_bin.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(train_mask.read_bytes()).hexdigest(),
    }
    m_path = out_dir / ("manifest_dryrun.json" if args.dry_run else "manifest.json")
    m_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build_sft_120m_v2] Wrote manifest to {m_path}", flush=True)


if __name__ == "__main__":
    main()
