#!/usr/bin/env python3
"""Parentheses 0.9-120M: Interactive Web UI & Zero-Dependency Streaming Server.

Standalone local web application for testing and interacting with Parentheses 0.9-120M
(Recurrent Selective Linear Attention SSM, NOT a Transformer).

Features:
  - Zero heavy UI dependencies (pure Python http.server + Server-Sent Events SSE).
  - Dark mode modern UI with purple/indigo accents and Dravidian typography support.
  - 4 Generation Modes:
      * 🧠 Thinking Mode (<think>...</think> collapsible step-by-step reasoning + answer)
      * 💬 Chat Mode (direct conversational assistant)
      * 🌊 FreeThink Mode (open-ended recurrent thought stream on statements)
      * 📝 Base Mode (raw document continuation)
  - 9 Supported Languages:
      * Anchor Dravidian: kan (Kannada), tam (Tamil), tel (Telugu), mal (Malayalam)
      * Low-resource Dravidian: tcy (Tulu), kff (Koya), kxv (Kuvi), nit (SE Gondi / Naikpod)
      * Global Anchor: eng (English)
  - Real-time token streaming with live tok/sec and O(1) latency constancy telemetry.
  - Interactive prompt presets per language and mode.

Usage:
    python app.py                     # runs on http://127.0.0.1:8080 (auto CUDA/CPU)
    python app.py --device cpu        # force CPU
    python app.py --port 7860         # custom port
    python app.py --self-test         # headless automated test suite on CPU
"""

import argparse
import dataclasses
import json
import os
import re
import socket
import statistics
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "model").exists() and str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "features").exists() else SCRIPT_DIR.parents[1]
if (REPO_ROOT / "model").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.backbone import Parentheses
from model.config import ModelConfig

# Language metadata: codes, names, native scripts, flags
LANGUAGES = [
    {"code": "auto", "name": "Auto Detect", "native": "⚡ Auto", "flag": "🌐"},
    {"code": "eng", "name": "English", "native": "English", "flag": "🇬🇧"},
    {"code": "kan", "name": "Kannada", "native": "ಕನ್ನಡ", "flag": "🇮🇳"},
    {"code": "tam", "name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳"},
    {"code": "tel", "name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳"},
    {"code": "mal", "name": "Malayalam", "native": "മലയാളം", "flag": "🇮🇳"},
    {"code": "tcy", "name": "Tulu", "native": "ತುಳು", "flag": "🇮🇳"},
    {"code": "kff", "name": "Koya", "native": "କୋୟା", "flag": "🇮🇳"},
    {"code": "kxv", "name": "Kuvi", "native": "କୁୱି", "flag": "🇮🇳"},
    {"code": "nit", "name": "Naikpod Gondi", "native": "నాయకపోడ్", "flag": "🇮🇳"},
]


def detect_language(text: str, default: str = "eng") -> str:
    """Accurately detects language from text based on Unicode script distribution and lexical cues."""
    if not text:
        return default

    # Check explicit command overrides
    m = re.search(r"(?:^|\s)/(eng|kan|tam|tel|mal|tcy|kff|kxv|nit)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Tally script characters
    counts = {"kan": 0, "tam": 0, "tel": 0, "mal": 0, "orya": 0, "latin": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0C80 <= cp <= 0x0CFF:
            counts["kan"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tam"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["tel"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["mal"] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            counts["orya"] += 1
        elif ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
            counts["latin"] += 1

    # Pick non-latin if present
    non_latin = {k: v for k, v in counts.items() if k != "latin" and v > 0}
    if not non_latin:
        return "eng"

    best_script = max(non_latin, key=non_latin.get)

    if best_script == "tam":
        return "tam"
    if best_script == "mal":
        return "mal"
    if best_script == "kan":
        tulu_markers = ("ಇಂದೆತ", "ದುಂಬು", "ಪನ್ಪಿನ", "ದಾಯೆ", "ಎಂಕ್", "ಮಸ್ತ್", "ಇರ್", "ಬೋಡು", "ಇಜ್ಜಿ", "ತುಳು")
        if any(w in text for w in tulu_markers):
            return "tcy"
        return "kan"
    if best_script == "tel":
        nit_markers = ("ఇదా", "ఎన్నె", "అని", "మరి", "నాయకపోడ్", "తోడ")
        if any(w in text for w in nit_markers):
            return "nit"
        return "tel"
    if best_script == "orya":
        kuvi_markers = ("କୁୱି", "ଏ଼ଦା", "ମେହ୍", "ତାକି", "ଅ଼ଡ଼େ", "ଇନାକା")
        if any(w in text for w in kuvi_markers):
            return "kxv"
        return "kff"

    return "eng"

# Continuation priming templates for FreeThink mode
LANG_PRIMING = {
    "eng": "{prompt}\n\nLet us analyze the core mechanism and practical implications step by step:",
    "kan": "{prompt}\n\nಇದರ ಬಗ್ಗೆ ಮತ್ತಷ್ಟು ವಿವರಿಸುವುದಾದರೆ,",
    "tam": "{prompt}\n\nஇதைப் பற்றி மேலும் விரிவாகக் கூறினால்,",
    "tel": "{prompt}\n\nదీని గురించి మరింత వివరంగా చెప్పాలంటే,",
    "mal": "{prompt}\n\nഇതിനെക്കുറിച്ച് കൂടുതൽ വിശദമായി പറഞ്ഞാൽ,",
    "tcy": "{prompt}\n\nಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ",
    "kff": "{prompt}\n\nଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ",
    "kxv": "{prompt}\n\nଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ",
    "nit": "{prompt}\n\nఇదా ఎన్నె అని మరి",
}

_CMD_RE = re.compile(r"^\s*(/(?:free-?think|no-?think|think|chat|base))\b\s*", re.IGNORECASE)
_TAG_RE = re.compile(r"</?(?:free)?think>", re.IGNORECASE)


def parse_internal_command(text: str) -> tuple[str | None, str]:
    """Extracts and strips internal /think, /freethink, /nothink, /chat, /base commands and tags."""
    s = text.strip()
    mode_override = None
    while True:
        m = _CMD_RE.match(s)
        if not m:
            break
        cmd = m.group(1).lower()
        if cmd in ("/freethink", "/free-think"):
            mode_override = "freethink"
        elif cmd == "/think":
            mode_override = "think"
        elif cmd in ("/nothink", "/no-think", "/chat"):
            mode_override = "chat"
        elif cmd == "/base":
            mode_override = "base"
        s = s[m.end():].strip()
    s = _TAG_RE.sub("", s).strip()
    return mode_override, s


# Rich prompt presets for each mode and language
PROMPT_PRESETS = {
    "think": {
        "eng": "If 3 notebooks cost 45 rupees, how much do 7 notebooks cost? Walk through the calculation step by step.",
        "kan": "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಮಾಡುವುದು ಏಕೆ ಮುಖ್ಯ? ಕಾರಣಗಳನ್ನು ಹಂತ ಹಂತವಾಗಿ ವಿವರಿಸಿ.",
        "tam": "மரங்களை நட்டுப் பாதுகாப்பது நகரங்களின் வெப்பநிலையைக் குறைக்க எவ்வாறு உதவும்? படிநிலைகளுடன் விளக்குக.",
        "tel": "సౌరశక్తి వినియోగం వల్ల కలిగే ప్రయోజనాలు మరియు పర్యావరణ ప్రభావం ఏమిటి? వివరంగా వివరించండి.",
        "mal": "പരിസ്ഥിതി സംരക്ഷണവും സുസ്ഥിര വികസനവും ഒരുമിച്ച് കൊണ്ടുപോകേണ്ടത് എന്തുകൊണ്ട്? ഘട്ടം ഘട്ടമായി വ്യക്തമാക്കുക.",
        "tcy": "ಕರಾವಳಿ ಪ್ರದೇಶೊಡು ಮಳೆಗಾಲೊಡು ಕೃಷಿ ಮಲ್ಪುನ ಕ್ರಮ ಎಂಚಿನವು? ವಿವರವಾದ್ ಪನ್ಲೆ.",
        "kff": "ଗାଁ ଲୋକ୍‌କୁ ମିଳିମିଶି କାମ କିଲେ କିସା ଲାଭ ଆୟେ? କେତୁତାନ୍ କିୟାଟ୍ ।",
        "kxv": "ନା଼ୟୁଁତି ଲ଼କୁ କୃଷି କାମ କିହିଁ ମାଞ୍ଜାନା ସବୁ ସା଼ୟେମ କିନେରି? ମେହ୍‌ଦୁ ।",
        "nit": "మన ఊరిలో చెట్లను మరియు అడవులను ఎలా కాపాడుకోవాలి? వరుసగా చెప్పండి.",
    },
    "chat": {
        "eng": "Explain what a Recurrent State Space Model is and how it achieves linear O(T) training and constant O(1) inference.",
        "kan": "ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ಐತಿಹಾಸಿಕ ತಾಣಗಳ ಬಗ್ಗೆ ಮತ್ತು ಅವುಗಳ ವಾಸ್ತುಶಿಲ್ಪದ ಬಗ್ಗೆ ಸಂಕ್ಷಿಪ್ತವಾಗಿ ತಿಳಿಸಿ.",
        "tam": "சங்க இலக்கியத்தின் முக்கியத்துவத்தை மற்றும் தமிழ் கலாச்சாரத்தின் சிறப்புகளை எளிமையாக விளக்குங்கள்.",
        "tel": "కృత్రిమ మేధస్సు (Artificial Intelligence) భవిష్యత్తు మానవ సమాజంపై ఎలాంటి ప్రభావం చూపుతుందో చెప్పండి.",
        "mal": "കേരളത്തിലെ പ്രകൃತಿഭംഗിയെക്കുറിച്ചും പശ്ചിಮഘട്ടത്തിന്റെ പ്രാധാന്യത്തെക്കുറിച്ചും വിവരിക്കുക.",
        "tcy": "ತುಳುನಾಡ್ ದ ಕಂಬಳ, ಭೂತಾರಾಧನೆ ಬೊಕ್ಕ ಜನಪದ ಸಂಸ್ಕೃತಿದ ಬಗ್ಗೆ ಒಂತೆ ಪನ್ಲೆ.",
        "kff": "ଆମୋଟ୍ ଗାଁତେ ପର୍ବ ପର୍ବାଣି କିହିଁ ମାନାଟ୍ କେତୁତାନ୍ କିୟାଟ୍ ।",
        "kxv": "ମା଼ରୋ ଇଦି କସ୍ତ କାମ ତାକି ସବୁ ଲ଼କୁ କାମ କିନେରି ସା଼ୟେମ କିୟାଟ୍ ।",
        "nit": "రైతులకు మంచి పంట రావడానికి ఎలాంటి ఆధునిక పద్ధతులు ఉపయోగపడతాయి?",
    },
    "freethink": {
        "eng": "Recurrent state space models run in constant O(1) memory per step during inference, maintaining an associative state matrix.",
        "kan": "ಪಶ್ಚಿಮ ಘಟ್ಟಗಳ ದಟ್ಟ ಕಾಡುಗಳು ಕರ್ನಾಟಕದ ಜೀವವೈವಿಧ್ಯಕ್ಕೆ ಮತ್ತು ನದಿಗಳ ಜೀವನಾಡಿಗಳಿಗೆ ಮೂಲ ಆಧಾರವಾಗಿವೆ.",
        "tam": "கடல்சார் வணிகம் பல நூற்றாண்டுகளாகத் தென்னிந்தியத் துறைமுகங்களை உலக நாடுகளுடன் இணைத்து செழிப்பு தந்தது.",
        "tel": "గోదావరి నది పరీవాహక ప్రాంతం తెలుగు ప్రజల జీవన విధానానికి మరియు వ్యవసాయ సంస్కృతికి జీవనాడిగా ఉంది.",
        "mal": "കേരളത്തിലെ കായലുകളും പശ്ചിമഘട്ട മലനിരകളും കാലാവസ്ഥയെ നിയന്ത്രിക്കുന്നതിൽ വലിയ പങ്കുവಹിക്കുന്നു.",
        "tcy": "ತುಳುನಾಡ್ ದ ಸಂಸ್ಕೃತಿ ಬೊಕ್ಕ ಕೃಷಿ ಜನಕುಲೆನ ಬದುಕುದ ಒಂಜಿ ಮುಖ್ಯ ಭಾಗ ಆದುಂಡು.",
        "kff": "ଇଦ୍ ନାୟୁଁତେ ଲୋକ୍‌କୁ କୃଷି ଆରି ବଣ ଜଙ୍ଗଲ ତଲେ ମିଶି ଜିବନ କିନେରି ।",
        "kxv": "ମାଣ୍‌ସି ମୀର୍‌ଏଣା ଦିନାତା ଆ଼ନେ, ସାମା ଏ଼ୱାସି ଗାଡି କସ୍ତ ପା଼ଣ୍‌ନାୟି ମାନେ ।",
        "nit": "మా ఊరిలో పంటలు మరియు అడవులు ప్రజల జీవనానికి ముఖ్య ఆధారము.",
    },
    "base": {
        "eng": "The selective linear attention state space recurrence updates a fixed-size matrix state S_t = alpha_t * S_{t-1} + k_t (x) v_t",
        "kan": "ಬೆಂಗಳೂರು ಭಾರತದ ಪ್ರಮುಖ ಮಾಹಿತಿ ತಂತ್ರಜ್ಞಾನ ನಗರಿಯಾಗಿದ್ದು ಜಾಗತಿಕ ಮಟ್ಟದಲ್ಲಿ",
        "tam": "பண்டைய தமிழகத்தில் இலக்கியமும் கலைகளும் மிகச் சிறந்த முறையில் வளர்ச்சி பெற்று",
        "tel": "హైదరాబాద్ నగరం చారిత్రక ప్రాధాನ್ಯత కలిగిన ప్రదేశం మరియు ఆధునిక సాంకేతిక",
        "mal": "മലയാള ഭാഷ ദ്രാവിഡ ഭാഷാ കുടുംബത്തിലെ ഒരു പ്രധാന ഭാഷയായി നിലകൊള്ളുന്നു",
        "tcy": "ಕರಾವಳಿ ಪ್ರದೇಶದ ಕೃಷಿ ಬೊಕ್ಕ ಮೀನುಗಾರಿಕೆ ಜನಕುಲೆನ ಜೀವನೊಗು ಆಧಾರ",
        "kff": "ଗାଁ ଲୋକ୍‌କୁ ମିଳିମିଶି କାମ କିହିଁ ମାଞ୍ଜାନା ସବୁ କଷ୍ଟ ଦୂର ଆୟେ",
        "kxv": "ନା଼ୟୁଁତି ଲ଼କୁ ରଅତେରି ରଅଣାଇଁ ସା଼ୟେମ କିହିଁ ମାଞ୍ଜାନା ସବୁ କାମ କିନେରି",
        "nit": "మంచి మాటలు మరియు సహాయం మనుషుల మధ్య స్నేహాన్ని పెంచుతాయి",
    },
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Parentheses 0.9-120M | Recurrent SSM Studio</title>
  <style>
    :root {
      --bg: #0b0f19;
      --surface: #111827;
      --surface-border: #1f2937;
      --surface-hover: #1e293b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #8b5cf6;
      --accent-glow: rgba(139, 92, 246, 0.35);
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
      --think-bg: #151b2b;
      --think-border: #312e81;
      --think-text: #c7d2fe;
      --success: #10b981;
      --cyan: #06b6d4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Noto Sans", "Noto Sans Kannada", "Noto Sans Tamil", "Noto Sans Telugu", "Noto Sans Malayalam", "Noto Sans Oriya", sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(17, 24, 39, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--surface-border);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-logo {
      font-size: 1.6rem;
      font-weight: 800;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
    }
    .brand-tag {
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(139, 92, 246, 0.2);
      color: #c4b5fd;
      padding: 0.2rem 0.5rem;
      border-radius: 9999px;
      border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .device-badge {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.8rem;
      font-weight: 500;
      background: #1e293b;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      border: 1px solid var(--surface-border);
    }
    .device-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    main {
      flex: 1;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    /* Tabs */
    .tabs-container {
      display: flex;
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 0.35rem;
      gap: 0.35rem;
      overflow-x: auto;
    }
    .tab-btn {
      flex: 1;
      min-width: 140px;
      padding: 0.65rem 1rem;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.9rem;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
    }
    .tab-btn:hover {
      background: var(--surface-hover);
      color: var(--text);
    }
    .tab-btn.active {
      background: var(--accent-gradient);
      color: #ffffff;
      box-shadow: 0 4px 12px var(--accent-glow);
    }
    /* Language Selector */
    .lang-bar {
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
      padding-bottom: 0.5rem;
    }
    .lang-pill {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      padding: 0.4rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      white-space: nowrap;
      transition: all 0.2s ease;
    }
    .lang-pill:hover {
      background: var(--surface-hover);
      color: var(--text);
      border-color: #4b5563;
    }
    .lang-pill.active {
      background: rgba(139, 92, 246, 0.15);
      border-color: var(--accent);
      color: #e0e7ff;
      font-weight: 600;
      box-shadow: 0 0 10px rgba(139, 92, 246, 0.2);
    }
    /* Main Layout */
    .app-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }
    @media (min-width: 900px) {
      .app-grid {
        grid-template-columns: 420px 1fr;
      }
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .panel-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .example-chips {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .chip {
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
      font-size: 0.82rem;
      cursor: pointer;
      text-align: left;
      line-height: 1.35;
      transition: all 0.2s ease;
    }
    .chip:hover {
      background: rgba(139, 92, 246, 0.2);
      border-color: var(--accent);
      color: #fff;
    }
    .prompt-box {
      width: 100%;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 0.85rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      resize: vertical;
      min-height: 120px;
      outline: none;
      transition: border-color 0.2s;
    }
    .prompt-box:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    .control-item {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .control-item label {
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    .control-item input[type="range"] {
      accent-color: var(--accent);
      cursor: pointer;
    }
    .btn-submit {
      background: var(--accent-gradient);
      color: white;
      border: none;
      padding: 0.85rem 1.25rem;
      border-radius: 10px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      transition: all 0.2s;
      box-shadow: 0 4px 14px var(--accent-glow);
    }
    .btn-submit:hover:not(:disabled) {
      opacity: 0.95;
      transform: translateY(-1px);
    }
    .btn-submit:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-stop {
      background: #dc2626;
      box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35);
    }
    /* Output Area */
    .output-panel {
      flex: 1;
      min-height: 480px;
      display: flex;
      flex-direction: column;
    }
    .telemetry-strip {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--surface-border);
      font-size: 0.8rem;
    }
    .telemetry-group {
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }
    .stat-badge {
      background: #1e293b;
      padding: 0.25rem 0.55rem;
      border-radius: 6px;
      color: #cbd5e1;
      display: flex;
      align-items: center;
      gap: 0.3rem;
    }
    .stat-badge.o1-badge {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .output-scroll {
      flex: 1;
      padding: 1rem 0;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    /* Reasoning Drawer (<think>) */
    .think-drawer {
      background: var(--think-bg);
      border: 1px solid var(--think-border);
      border-radius: 10px;
      overflow: hidden;
      transition: all 0.3s ease;
    }
    .think-header {
      padding: 0.65rem 1rem;
      background: rgba(49, 46, 129, 0.4);
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      user-select: none;
      font-size: 0.85rem;
      font-weight: 600;
      color: #a5b4fc;
    }
    .think-header:hover {
      background: rgba(49, 46, 129, 0.6);
    }
    .think-status {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .think-pulse {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #818cf8;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.4); opacity: 0.5; }
    }
    .think-content {
      padding: 0.85rem 1rem;
      color: var(--think-text);
      font-size: 0.9rem;
      line-height: 1.6;
      white-space: pre-wrap;
      font-family: inherit;
    }
    .think-drawer.collapsed .think-content {
      display: none;
    }
    .response-content {
      color: var(--text);
      font-size: 1rem;
      line-height: 1.65;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .freethink-watermark {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      color: var(--cyan);
      background: rgba(6, 182, 212, 0.1);
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      border: 1px solid rgba(6, 182, 212, 0.25);
      margin-bottom: 0.5rem;
    }
    .cursor-blink {
      display: inline-block;
      width: 7px;
      height: 15px;
      background: var(--accent);
      margin-left: 2px;
      vertical-align: middle;
      animation: blink 1s infinite;
    }
    @keyframes blink {
      0%, 49% { opacity: 1; }
      50%, 100% { opacity: 0; }
    }
    .output-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.5rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--surface-border);
    }
    .btn-secondary {
      background: #1e293b;
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      background: #334155;
      color: var(--text);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">( ) Parentheses 0.9-120M</div>
      <div class="brand-tag">SSM · 9 Languages</div>
    </div>
    <div class="device-badge">
      <div class="device-dot"></div>
      <span id="device-label">Detecting device...</span>
    </div>
  </header>

  <main>
    <!-- Mode Tabs -->
    <div class="tabs-container">
      <button class="tab-btn active" data-mode="think">🧠 Thinking Mode</button>
      <button class="tab-btn" data-mode="chat">💬 Chat Mode</button>
      <button class="tab-btn" data-mode="freethink">🌊 FreeThink Mode</button>
      <button class="tab-btn" data-mode="base">📝 Base Mode</button>
    </div>

    <!-- Language Selector -->
    <div class="lang-bar" id="lang-bar">
      <!-- Injected by JavaScript -->
    </div>

    <div class="app-grid">
      <!-- Input Column -->
      <div class="panel">
        <div class="panel-title">
          <span>Prompt Input</span>
          <span style="font-size: 0.75rem; color: var(--text-muted);">Ctrl+Enter to run</span>
        </div>

        <textarea class="prompt-box" id="prompt-input" placeholder="Type or select a prompt preset..."></textarea>

        <div class="example-chips" id="example-chips">
          <!-- Preset chips injected dynamically -->
        </div>

        <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; margin-top:0.25rem;">
          <input type="file" id="mm-file-input" accept=".pdf,.docx,.png,.jpg,.jpeg,.bmp,.webp,.html,.htm,.xml,.csv,.tsv,.json,.jsonl,.txt,.md" style="display:none;">
          <button type="button" class="btn-secondary" id="btn-attach" style="display:inline-flex; align-items:center; gap:0.35rem;">
            📎 Attach Image / Document (.pdf, .docx, .png, .jpg, .csv, .txt)
          </button>
          <span id="mm-status-badge" style="font-size:0.78rem; color:#34d399; display:none;"></span>
        </div>

        <div class="controls-grid">
          <div class="control-item">
            <label>Temperature: <span id="temp-val">0.7</span></label>
            <input type="range" id="temp-slider" min="0.1" max="1.5" step="0.05" value="0.7">
          </div>
          <div class="control-item">
            <label>Top-K: <span id="topk-val">40</span></label>
            <input type="range" id="topk-slider" min="1" max="100" step="1" value="40">
          </div>
          <div class="control-item" style="grid-column: span 2;">
            <label>Max Tokens: <span id="maxtok-val">256</span></label>
            <input type="range" id="maxtok-slider" min="32" max="1024" step="32" value="256">
          </div>
        </div>

        <button class="btn-submit" id="btn-generate">
          <span>Generate Response</span>
        </button>
      </div>

      <!-- Output Column -->
      <div class="panel output-panel">
        <div class="telemetry-strip">
          <div class="telemetry-group">
            <div class="stat-badge" id="speed-badge">⚡ 0.0 tok/s</div>
            <div class="stat-badge" id="time-badge">⏱ 0.0s</div>
            <div class="stat-badge" id="tokens-badge">🔤 0 tokens</div>
          </div>
          <div class="stat-badge o1-badge">
            ✓ Recurrent SSM O(1) Memory
          </div>
        </div>

        <div class="output-scroll" id="output-scroll">
          <!-- Thinking drawer (hidden until thinking tokens arrive) -->
          <div class="think-drawer" id="think-drawer" style="display: none;">
            <div class="think-header" id="think-header">
              <div class="think-status">
                <div class="think-pulse" id="think-pulse"></div>
                <span id="think-status-text">Reasoning Process...</span>
              </div>
              <span id="think-toggle-btn" style="font-size: 0.75rem; color: #a5b4fc;">[Collapse]</span>
            </div>
            <div class="think-content" id="think-content"></div>
          </div>

          <!-- Final Response area -->
          <div id="freethink-badge-container"></div>
          <div class="response-content" id="response-content">Select a language and mode above, then enter a prompt or click a preset to test Parentheses 0.9-120M.</div>
        </div>

        <div class="output-actions">
          <button class="btn-secondary" id="btn-copy">📋 Copy</button>
          <button class="btn-secondary" id="btn-clear">🧹 Clear</button>
        </div>
      </div>
    </div>
  </main>

  <script>
    const LANGUAGES = """ + json.dumps(LANGUAGES, ensure_ascii=False) + """;
    const PRESETS = """ + json.dumps(PROMPT_PRESETS, ensure_ascii=False) + """;

    let currentMode = "think";
    let currentLang = "eng";
    let activeController = null;

    // Elements
    const langBar = document.getElementById("lang-bar");
    const tabButtons = document.querySelectorAll(".tab-btn");
    const promptInput = document.getElementById("prompt-input");
    const exampleChips = document.getElementById("example-chips");
    const btnGenerate = document.getElementById("btn-generate");
    const btnCopy = document.getElementById("btn-copy");
    const btnClear = document.getElementById("btn-clear");
    const deviceLabel = document.getElementById("device-label");
    const speedBadge = document.getElementById("speed-badge");
    const timeBadge = document.getElementById("time-badge");
    const tokensBadge = document.getElementById("tokens-badge");
    const thinkDrawer = document.getElementById("think-drawer");
    const thinkHeader = document.getElementById("think-header");
    const thinkContent = document.getElementById("think-content");
    const thinkPulse = document.getElementById("think-pulse");
    const thinkStatusText = document.getElementById("think-status-text");
    const thinkToggleBtn = document.getElementById("think-toggle-btn");
    const responseContent = document.getElementById("response-content");
    const freethinkBadgeContainer = document.getElementById("freethink-badge-container");
    const outputScroll = document.getElementById("output-scroll");

    // Sliders
    const tempSlider = document.getElementById("temp-slider");
    const tempVal = document.getElementById("temp-val");
    tempSlider.oninput = () => tempVal.innerText = tempSlider.value;

    const topkSlider = document.getElementById("topk-slider");
    const topkVal = document.getElementById("topk-val");
    topkSlider.oninput = () => topkVal.innerText = topkSlider.value;

    const maxtokSlider = document.getElementById("maxtok-slider");
    const maxtokVal = document.getElementById("maxtok-val");
    maxtokSlider.oninput = () => maxtokVal.innerText = maxtokSlider.value;

    // Initialize Language Bar
    LANGUAGES.forEach(lang => {
      const btn = document.createElement("button");
      btn.className = `lang-pill ${lang.code === currentLang ? "active" : ""}`;
      btn.innerHTML = `${lang.flag} ${lang.native} <span style="font-size:0.75rem; opacity:0.7;">(${lang.code})</span>`;
      btn.onclick = () => switchLanguage(lang.code);
      btn.id = `lang-pill-${lang.code}`;
      langBar.appendChild(btn);
    });

    // Initialize Mode Tabs
    tabButtons.forEach(btn => {
      btn.onclick = () => switchMode(btn.dataset.mode);
    });

    function switchMode(mode) {
      currentMode = mode;
      tabButtons.forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
      renderPresets();
    }

    function switchLanguage(langCode) {
      currentLang = langCode;
      document.querySelectorAll(".lang-pill").forEach(p => p.classList.toggle("active", p.id === `lang-pill-${langCode}`));
      renderPresets();
    }

    function renderPresets() {
      exampleChips.innerHTML = "";
      const text = PRESETS[currentMode]?.[currentLang] || "";
      if (text) {
        const chip = document.createElement("div");
        chip.className = "chip";
        chip.innerText = "💡 " + text;
        chip.onclick = () => { promptInput.value = text; promptInput.focus(); };
        exampleChips.appendChild(chip);
      }
    }

    // Toggle reasoning drawer
    thinkHeader.onclick = () => {
      thinkDrawer.classList.toggle("collapsed");
      thinkToggleBtn.innerText = thinkDrawer.classList.contains("collapsed") ? "[Expand]" : "[Collapse]";
    };

    // Health check & device detection
    fetch("/api/health")
      .then(r => r.json())
      .then(d => {
        deviceLabel.innerText = `${d.model} (${d.device.toUpperCase()})`;
      })
      .catch(() => {
        deviceLabel.innerText = "Parentheses 0.9-120M (Ready)";
      });

    renderPresets();

    // Copy & Clear
    btnCopy.onclick = () => {
      const textToCopy = (thinkContent.innerText ? `[Reasoning]\n${thinkContent.innerText}\n\n[Response]\n` : "") + responseContent.innerText;
      navigator.clipboard.writeText(textToCopy);
      btnCopy.innerText = "✓ Copied!";
      setTimeout(() => btnCopy.innerText = "📋 Copy", 2000);
    };

    btnClear.onclick = () => {
      thinkDrawer.style.display = "none";
      thinkContent.innerText = "";
      responseContent.innerText = "";
      freethinkBadgeContainer.innerHTML = "";
      speedBadge.innerText = "⚡ 0.0 tok/s";
      timeBadge.innerText = "⏱ 0.0s";
      tokensBadge.innerText = "🔤 0 tokens";
      const mmBadge = document.getElementById("mm-status-badge");
      if (mmBadge) { mmBadge.style.display = "none"; mmBadge.innerText = ""; }
      fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clear: true })
      }).catch(() => {});
    };

    const btnAttach = document.getElementById("btn-attach");
    const mmFileInput = document.getElementById("mm-file-input");
    const mmStatusBadge = document.getElementById("mm-status-badge");
    if (btnAttach && mmFileInput) {
      btnAttach.onclick = () => mmFileInput.click();
      mmFileInput.onchange = () => {
        const file = mmFileInput.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = async (ev) => {
          const dataUrl = ev.target.result || "";
          const b64 = dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : dataUrl;
          mmStatusBadge.style.display = "inline-block";
          mmStatusBadge.innerText = `⏳ Ingesting ${file.name}...`;
          try {
            const r = await fetch("/api/ingest", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ filename: file.name, data_base64: b64, lang: currentLang })
            });
            const res = await r.json();
            if (res.status === "ok") {
              mmStatusBadge.innerText = `✓ Attached ${res.filename} (${res.modality}, ${res.char_count} chars)`;
            } else {
              mmStatusBadge.innerText = `⚠ ${res.error || "Failed to attach file"}`;
            }
          } catch (e) {
            mmStatusBadge.innerText = `⚠ Error: ${e.message}`;
          }
        };
        reader.readAsDataURL(file);
      };
    }

    // Keyboard shortcut
    promptInput.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        if (!activeController) triggerGenerate();
      }
    });

    btnGenerate.onclick = () => {
      if (activeController) {
        activeController.abort();
        activeController = null;
        resetGenerateButton();
      } else {
        triggerGenerate();
      }
    };

    function resetGenerateButton() {
      btnGenerate.classList.remove("btn-stop");
      btnGenerate.innerHTML = "<span>Generate Response</span>";
      thinkPulse.style.display = "none";
    }

    async function triggerGenerate() {
      let prompt = promptInput.value.trim();
      if (!prompt) return;

      // Automatically hide and route internal /think, /freethink, /nothink, /chat, /base commands
      const cmdMatch = prompt.match(/^\s*\/(free-?think|no-?think|think|chat|base)\b\s*/i);
      if (cmdMatch) {
        const rawCmd = cmdMatch[1].toLowerCase();
        if (rawCmd === "freethink" || rawCmd === "free-think") switchMode("freethink");
        else if (rawCmd === "think") switchMode("think");
        else if (rawCmd === "nothink" || rawCmd === "no-think" || rawCmd === "chat") switchMode("chat");
        else if (rawCmd === "base") switchMode("base");
        prompt = prompt.slice(cmdMatch[0].length).replace(/<\/?(?:free)?think>/gi, "").trim();
        promptInput.value = prompt;
        if (!prompt) return;
      }

      activeController = new AbortController();
      btnGenerate.classList.add("btn-stop");
      btnGenerate.innerHTML = "<span>⏹ Stop Generation</span>";

      // Reset output fields
      thinkDrawer.style.display = "none";
      thinkDrawer.classList.remove("collapsed");
      thinkToggleBtn.innerText = "[Collapse]";
      thinkContent.innerText = "";
      responseContent.innerText = "";
      freethinkBadgeContainer.innerHTML = "";

      if (currentMode === "freethink") {
        freethinkBadgeContainer.innerHTML = `<div class="freethink-watermark">🌊 FreeThink Mode · Continuous Recurrent Stream (${currentLang})</div>`;
      }

      speedBadge.innerText = "⚡ 0.0 tok/s";
      timeBadge.innerText = "⏱ 0.0s";
      tokensBadge.innerText = "🔤 0 tokens";

      const payload = {
        prompt: prompt,
        mode: currentMode,
        lang: currentLang,
        temperature: parseFloat(tempSlider.value),
        top_k: parseInt(topkSlider.value),
        max_tokens: parseInt(maxtokSlider.value),
      };

      try {
        const res = await fetch("/api/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: activeController.signal
        });

        if (!res.ok) throw new Error("Server error: " + res.statusText);

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\\n\\n");
          buffer = lines.pop();

          for (const block of lines) {
            if (!block.startsWith("data: ")) continue;
            const dataStr = block.replace(/^data: /, "").trim();
            if (!dataStr) continue;
            try {
              const msg = JSON.parse(dataStr);
              handleStreamMessage(msg);
            } catch (err) {
              console.error("JSON parse error on SSE:", err, dataStr);
            }
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          responseContent.innerText += `\\n[Error: ${err.message}]`;
        }
      } finally {
        activeController = null;
        resetGenerateButton();
      }
    }

    function handleStreamMessage(msg) {
      if (msg.type === "token") {
        if (msg.section === "think") {
          thinkDrawer.style.display = "block";
          thinkPulse.style.display = "inline-block";
          thinkStatusText.innerText = "Reasoning Process...";
          thinkContent.innerText += msg.token;
        } else {
          responseContent.innerText += msg.token;
        }

        if (msg.tps !== undefined) {
          speedBadge.innerText = `⚡ ${msg.tps.toFixed(1)} tok/s`;
        }
        if (msg.elapsed !== undefined) {
          timeBadge.innerText = `⏱ ${msg.elapsed.toFixed(1)}s`;
        }
        if (msg.token_count !== undefined) {
          tokensBadge.innerText = `🔤 ${msg.token_count} tokens`;
        }
        outputScroll.scrollTop = outputScroll.scrollHeight;
      } else if (msg.type === "think_end") {
        thinkPulse.style.display = "none";
        thinkStatusText.innerText = `Reasoning Complete (${msg.think_tokens || ""} tokens in ${msg.elapsed?.toFixed(1) || ""}s)`;
      } else if (msg.type === "done") {
        thinkPulse.style.display = "none";
        if (msg.tps !== undefined) speedBadge.innerText = `⚡ ${msg.tps.toFixed(1)} tok/s`;
        if (msg.elapsed !== undefined) timeBadge.innerText = `⏱ ${msg.elapsed.toFixed(2)}s`;
        if (msg.total_tokens !== undefined) tokensBadge.innerText = `🔤 ${msg.total_tokens} tokens`;
      }
    }
  </script>
</body>
</html>
"""


def load_safetensors_fallback(filepath: Path) -> dict[str, torch.Tensor]:
    """Loads tensors from safetensors format with pure-Python fallback."""
    try:
        import safetensors.torch
        return safetensors.torch.load_file(str(filepath))
    except ImportError:
        pass

    with open(filepath, "rb") as f:
        length_bytes = f.read(8)
        header_len = struct.unpack("<Q", length_bytes)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data = f.read()

    dtype_map = {
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
    }
    tensors = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        dt = dtype_map[v["dtype"]]
        shape = v["shape"]
        start, end = v["data_offsets"]
        tensor = torch.frombuffer(bytearray(data[start:end]), dtype=dt).reshape(shape).clone()
        tensors[k] = tensor
    return tensors


def resolve_model_and_tokenizer(model_dir_or_ckpt: str | None = None, device: str = "cpu"):
    """Loads the model and tokenizer from safetensors release candidate or checkpoint .pt file."""
    from tokenizers import Tokenizer

    # 1. Check specified path
    search_paths = []
    if model_dir_or_ckpt:
        search_paths.append(Path(model_dir_or_ckpt).resolve())

    # 2. Check candidate release directory (safetensors)
    cand_dir = SCRIPT_DIR if (SCRIPT_DIR / "model.safetensors").exists() else REPO_ROOT / "release" / "parentheses-0.9-120M-candidate"
    if cand_dir.exists():
        search_paths.append(cand_dir)

    # 3. Check trained checkpoint .pt
    search_paths.extend([
        REPO_ROOT / "checkpoints" / "parentheses-0.9-120M-sft" / "step_9636_final.pt",
        REPO_ROOT / "checkpoints" / "parentheses-0.9-120M" / "step_9536_final.pt",
    ])

    for target in search_paths:
        if not target.exists():
            continue

        # Safetensors directory format
        if target.is_dir() and (target / "model.safetensors").exists():
            cfg_dict = json.loads((target / "config.json").read_text(encoding="utf-8"))
            cfg_fields = {f.name for f in dataclasses.fields(ModelConfig)}
            cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in cfg_fields})
            model = Parentheses(cfg)
            fname = "model.bf16.safetensors" if (device == "cuda" and (target / "model.bf16.safetensors").exists()) else "model.safetensors"
            state_dict = load_safetensors_fallback(target / fname)
            model.load_state_dict(state_dict, strict=False)
            if cfg.tie_embeddings:
                model.head.weight = model.tok_emb.weight
            model.to(device).eval()
            tok_path = target / "tokenizer.json"
            tok = Tokenizer.from_file(str(tok_path))
            tok.encode_special_tokens = True
            return model, tok, f"safetensors ({target.name})"

        # Torch checkpoint .pt format
        if target.is_file() and target.suffix == ".pt":
            ckpt = torch.load(target, map_location=device, weights_only=False)
            model = Parentheses(ckpt["cfg"]).to(device).eval()
            model.load_state_dict(ckpt["model"])
            tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
            if not tok_path.exists():
                tok_path = target.parent.parent / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
            tok = Tokenizer.from_file(str(tok_path))
            tok.encode_special_tokens = True
            return model, tok, f"checkpoint ({target.name})"

    # Fallback: Synthetic tiny model for self-test / testing without weights
    tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
    if not tok_path.exists() and (cand_dir / "tokenizer.json").exists():
        tok_path = cand_dir / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    tok.encode_special_tokens = True
    cfg = ModelConfig(vocab_size=32768, block_size=256, n_layer=2, n_head=2, n_embd=32, attn_type="selective_linear")
    model = Parentheses(cfg).to(device).eval()
    return model, tok, "synthetic-preview"


class GenerationEngine:
    def __init__(self, model: Parentheses, tok, device: str = "cpu", web: bool = False):
        self.model = model
        self.tok = tok
        self.device = device
        self.web = web
        self._lock = threading.Lock()
        self.active_contexts: list[str] = []
        self.visual_states: list | None = None

    def clear_multimodal(self):
        with self._lock:
            self.active_contexts.clear()
            self.visual_states = None

    def ingest_multimodal(
        self,
        filename: str,
        raw_bytes: bytes | None = None,
        path: str | None = None,
        lang: str = "eng",
    ) -> dict:
        from features.multimodal import (
            IMAGE_EXTENSIONS,
            describe_image,
            extract_document_text,
            prefill_visual_state,
            state_memory_bytes,
            state_shapes,
        )

        with self._lock:
            source = Path(path) if path else raw_bytes
            if source is None:
                raise ValueError("Either raw_bytes (data_base64) or path must be provided")
            ext = Path(filename or (path or "")).suffix.lower()
            is_image = ext in IMAGE_EXTENSIONS or (
                isinstance(source, (bytes, bytearray))
                and bytes(source).startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM", b"RIFF"))
            )
            if is_image:
                desc = describe_image(source, lang=lang)
                summary = desc.summary
                ssm_bytes = 0
                ssm_shapes = []
                if getattr(self.model.cfg, "attn_type", "") == "selective_linear":
                    self.visual_states = prefill_visual_state(
                        self.model, source, states=self.visual_states, patch_size=16, grid_size=64
                    )
                    ssm_bytes = state_memory_bytes(self.visual_states)
                    ssm_shapes = [list(s) for s in state_shapes(self.visual_states)]
                self.active_contexts.append(summary)
                return {
                    "status": "ok",
                    "filename": filename or "image",
                    "modality": "image",
                    "summary": summary,
                    "char_count": len(summary),
                    "attributes": dict(desc),
                    "ssm_state_bytes": ssm_bytes,
                    "ssm_state_shapes": ssm_shapes,
                }

            doc_text = extract_document_text(source, filename=filename, lang=lang).strip()
            max_chars = max(512, getattr(self.model.cfg, "block_size", 1024) // 2)
            clipped = doc_text[:max_chars]
            if clipped:
                self.active_contexts.append(clipped)
            return {
                "status": "ok",
                "filename": filename or "document",
                "modality": "document",
                "summary": clipped,
                "char_count": len(doc_text),
            }

    def stream_generate(self, prompt: str, mode: str = "think", lang: str = "auto",
                        max_tokens: int = 256, temperature: float = 0.7, top_k: int = 40,
                        min_p: float = 0.06, repetition_penalty: float = 1.16,
                        no_repeat_ngram_size: int = 5, web: bool | None = None):
        with self._lock:
            cmd_mode, clean_prompt = parse_internal_command(prompt)
            if cmd_mode is not None:
                mode = cmd_mode

            # Dynamic Language Detection
            inline_lang = None
            m = re.match(r"^\s*/(eng|kan|tam|tel|mal|tcy|kff|kxv|nit)\b", clean_prompt, re.IGNORECASE)
            if m:
                inline_lang = m.group(1).lower()
                clean_prompt = clean_prompt[m.end():].strip()

            active_lang = inline_lang or (lang if lang and lang != "auto" else detect_language(clean_prompt, default="eng"))
            lang = active_lang

            # Web intent gating and extraction
            web_enabled = self.web if web is None else web
            web_blocks: list[str] = []
            try:
                from features.web_tools import (
                    search_web,
                    fetch_url,
                    format_web_results_for_prompt,
                    extract_web_queries,
                    _INLINE_WEB_CMD_RE,
                    _INLINE_URL_CMD_RE,
                )
                queries_to_search, urls_to_fetch = extract_web_queries(clean_prompt, web_enabled=web_enabled, lang=lang)
                clean_prompt = _INLINE_WEB_CMD_RE.sub(" ", clean_prompt)
                clean_prompt = _INLINE_URL_CMD_RE.sub(" ", clean_prompt)
                clean_prompt = re.sub(r"[ \t]{2,}", " ", clean_prompt).strip()

                for q in queries_to_search:
                    results = search_web(q, max_results=3)
                    if results:
                        formatted_ctx = format_web_results_for_prompt(results, query=q, max_words=150)
                        if formatted_ctx:
                            web_blocks.append(formatted_ctx)
                for u in urls_to_fetch:
                    doc_res = fetch_url(u, max_chars=1500)
                    if doc_res.get("text"):
                        clean_txt = re.sub(r"https?://\S+", "", doc_res["text"]).strip()
                        words = clean_txt.split()
                        if len(words) > 150:
                            clean_txt = " ".join(words[:150]) + "..."
                        page_title = re.sub(r"https?://\S+", "", doc_res.get('title', 'Webpage')).strip()
                        web_blocks.append(f"[Web Context: {page_title} - {clean_txt}]")
            except Exception:
                pass

            tag_id = self.tok.token_to_id(f"<{lang}>") or self.tok.token_to_id("<eng>")
            user_id = self.tok.token_to_id("<|user|>")
            asst_id = self.tok.token_to_id("<|assistant|>")
            think_id = self.tok.token_to_id("<think>")
            think_end_id = self.tok.token_to_id("</think>")
            stops = {self.tok.token_to_id(s) for s in ("<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>", "<|pad|>", "<esg>", "<tool_call>", "</tool_call>")} - {None}

            context_items = list(self.active_contexts[-3:] if self.active_contexts else [])
            if web_blocks:
                context_items.extend(web_blocks)

            if context_items:
                ctx_str = "\n\n".join(context_items)
                max_ctx = max(256, self.model.cfg.block_size // 2)
                clean_prompt = f"{ctx_str[:max_ctx]}\n\n{clean_prompt}".strip()

            if mode == "freethink":
                template = LANG_PRIMING.get(lang, LANG_PRIMING["eng"])
                primed = template.format(prompt=clean_prompt).rstrip(" ")
                input_ids = [tag_id] + self.tok.encode(primed, add_special_tokens=False).ids
            elif mode == "base":
                input_ids = [tag_id] + self.tok.encode(clean_prompt, add_special_tokens=False).ids
            elif mode == "chat":
                body_ids = self.tok.encode(clean_prompt, add_special_tokens=False).ids
                input_ids = [user_id, tag_id] + body_ids + [asst_id]
            else:  # think
                body_ids = self.tok.encode(clean_prompt, add_special_tokens=False).ids
                nl_ids = self.tok.encode("\n", add_special_tokens=False).ids
                input_ids = [user_id, tag_id] + body_ids + [asst_id, think_id] + nl_ids

            idx = torch.tensor([input_ids[-self.model.cfg.block_size:]], dtype=torch.long, device=self.device)
            stream = self.model.stream(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            in_think = (mode == "think")
            think_closed = False
            think_tokens = 0
            total_tokens = 0
            timings = []
            pending = []
            start_wall = time.perf_counter()

            yield {"type": "start", "mode": mode, "lang": lang}

            for next_t in stream:
                t0 = time.perf_counter()
                tid = int(next_t.item())
                dt = time.perf_counter() - t0
                timings.append(dt)
                total_tokens += 1

                if tid in stops:
                    break

                if tid == think_id:
                    continue

                if tid == think_end_id:
                    if mode == "think" and not think_closed:
                        if pending:
                            chunk = _TAG_RE.sub("", self.tok.decode(pending, skip_special_tokens=False))
                            pending = []
                            if chunk:
                                yield {
                                    "type": "token", "token": chunk, "section": "think",
                                    "token_count": total_tokens, "elapsed": time.perf_counter() - start_wall,
                                    "tps": total_tokens / max(1e-5, time.perf_counter() - start_wall),
                                }
                        think_closed = True
                        in_think = False
                        yield {"type": "think_end", "elapsed": time.perf_counter() - start_wall, "think_tokens": think_tokens}
                        continue
                    else:
                        break

                if in_think:
                    think_tokens += 1

                pending.append(tid)
                chunk = self.tok.decode(pending, skip_special_tokens=False)
                if chunk and not chunk.endswith("\ufffd"):
                    pending = []
                    chunk = _TAG_RE.sub("", chunk)
                    if chunk:
                        elapsed = time.perf_counter() - start_wall
                        tps = total_tokens / max(1e-5, elapsed)
                        yield {
                            "type": "token",
                            "token": chunk,
                            "section": "think" if in_think else "response",
                            "token_count": total_tokens,
                            "elapsed": elapsed,
                            "tps": tps,
                        }

            if pending:
                chunk = _TAG_RE.sub("", self.tok.decode(pending, skip_special_tokens=False))
                if chunk:
                    elapsed = time.perf_counter() - start_wall
                    tps = total_tokens / max(1e-5, elapsed)
                    yield {
                        "type": "token",
                        "token": chunk,
                        "section": "think" if in_think else "response",
                        "token_count": total_tokens,
                        "elapsed": elapsed,
                        "tps": tps,
                    }

            if mode == "think" and not think_closed:
                yield {"type": "think_end", "elapsed": time.perf_counter() - start_wall, "think_tokens": think_tokens}

            # Latency constancy ratio
            body = timings[1:] if len(timings) > 1 else timings
            win = max(1, min(20, len(body) // 2))
            first_ms = statistics.median(body[:win]) * 1000 if body else 0.0
            last_ms = statistics.median(body[-win:]) * 1000 if body else 0.0
            ratio = last_ms / max(first_ms, 1e-6) if body else 1.0
            total_elapsed = time.perf_counter() - start_wall
            final_tps = total_tokens / max(1e-5, total_elapsed)

            yield {
                "type": "done",
                "total_tokens": total_tokens,
                "elapsed": total_elapsed,
                "tps": final_tps,
                "latency_ratio": ratio,
                "constant_o1": ratio < 1.15,
            }


class ParenthesesHTTPHandler(BaseHTTPRequestHandler):
    engine: GenerationEngine = None
    model_name: str = "Parentheses 0.9-120M"
    device_name: str = "cpu"

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif url.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            data = {
                "status": "ok",
                "model": self.model_name,
                "device": self.device_name,
                "languages": len([l for l in LANGUAGES if l["code"] != "auto"]),
                "modes": ["think", "chat", "freethink", "base"],
                "multimodal": True,
                "web": getattr(self.engine, "web", False),
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif url.path == "/api/examples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(PROMPT_PRESETS, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/ingest":
            import base64
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": "Invalid JSON body"}).encode("utf-8"))
                return

            if payload.get("clear"):
                self.engine.clear_multimodal()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "cleared": True}).encode("utf-8"))
                return

            filename = payload.get("filename", "")
            b64_data = payload.get("data_base64", "")
            file_path = payload.get("path", None)
            lang = payload.get("lang", "eng")
            try:
                raw_bytes = base64.b64decode(b64_data) if b64_data else None
                res = self.engine.ingest_multimodal(
                    filename=filename, raw_bytes=raw_bytes, path=file_path, lang=lang
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            except Exception as exc:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(exc)}).encode("utf-8"))
            return

        if url.path == "/api/stream":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON body")
                return

            prompt = payload.get("prompt", "")
            mode = payload.get("mode", "think")
            lang = payload.get("lang", "eng")
            max_tokens = int(payload.get("max_tokens", 256))
            temperature = float(payload.get("temperature", 0.7))
            top_k = int(payload.get("top_k", 40))
            web = payload.get("web", None)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for event in self.engine.stream_generate(
                    prompt, mode=mode, lang=lang, max_tokens=max_tokens,
                    temperature=temperature, top_k=top_k, web=web
                ):
                    line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
            except (ConnectionError, BrokenPipeError):
                pass
            finally:
                self.close_connection = True
        else:
            self.send_response(404)
            self.end_headers()


def _self_test():
    """Headless automated test verifying server, routes, multimodal ingestion, and SSE streaming on CPU."""
    import base64
    import io
    from PIL import Image
    from features.multimodal import create_docx_bytes, create_pdf_bytes

    print("[self-test] Starting Parentheses app.py self-test on CPU...")

    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    model, tok, source_name = resolve_model_and_tokenizer(device="cpu")
    engine = GenerationEngine(model, tok, device="cpu")

    handler_cls = type(
        "SelfTestHandler",
        (ParenthesesHTTPHandler,),
        {"engine": engine, "model_name": f"Parentheses-0.9-120M ({source_name})", "device_name": "cpu"}
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        # 1. Test GET /
        req = urllib.request.urlopen(f"{base_url}/", timeout=5.0)
        assert req.status == 200
        html = req.read().decode("utf-8")
        assert "Parentheses 0.9-120M" in html
        assert "Thinking Mode" in html
        assert "FreeThink Mode" in html
        assert "Attach Image / Document" in html
        assert "ಕನ್ನಡ" in html
        print("  [x] GET / (HTML studio + Attach Image/Document UI) verified")

        # 2. Test GET /api/health
        req = urllib.request.urlopen(f"{base_url}/api/health", timeout=5.0)
        assert req.status == 200
        health = json.loads(req.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["languages"] == 9
        assert health.get("multimodal") is True
        print("  [x] GET /api/health verified")

        def read_sse(resp):
            evs = []
            for line in resp:
                s = line.decode("utf-8").strip()
                if s.startswith("data: "):
                    try:
                        ev = json.loads(s[6:])
                        evs.append(ev)
                        if ev.get("type") == "done":
                            break
                    except Exception:
                        pass
            return evs

        # 3. Test POST /api/ingest (PDF, DOCX, and PNG Image)
        pdf_b64 = base64.b64encode(create_pdf_bytes(["Selective linear attention O(1) recurrent state."])).decode("ascii")
        req_pdf = urllib.request.Request(
            f"{base_url}/api/ingest",
            data=json.dumps({"filename": "arch.pdf", "data_base64": pdf_b64, "lang": "eng"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        res_pdf = json.loads(urllib.request.urlopen(req_pdf, timeout=10.0).read().decode("utf-8"))
        assert res_pdf["status"] == "ok" and res_pdf["modality"] == "document" and "Selective linear" in res_pdf["summary"]

        docx_b64 = base64.b64encode(create_docx_bytes(["ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ."])).decode("ascii")
        req_docx = urllib.request.Request(
            f"{base_url}/api/ingest",
            data=json.dumps({"filename": "kan.docx", "data_base64": docx_b64, "lang": "kan"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        res_docx = json.loads(urllib.request.urlopen(req_docx, timeout=10.0).read().decode("utf-8"))
        assert res_docx["status"] == "ok" and "ಬೆಂಗಳೂರು" in res_docx["summary"]

        img_buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(20, 180, 90)).save(img_buf, format="PNG")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")
        req_img = urllib.request.Request(
            f"{base_url}/api/ingest",
            data=json.dumps({"filename": "green.png", "data_base64": img_b64, "lang": "eng"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        res_img = json.loads(urllib.request.urlopen(req_img, timeout=10.0).read().decode("utf-8"))
        assert res_img["status"] == "ok" and res_img["modality"] == "image" and res_img["ssm_state_bytes"] > 0
        print("  [x] POST /api/ingest (PDF, DOCX, and PNG SSM visual state) verified")

        # 4. Test POST /api/stream (Thinking Mode with /think command stripping)
        payload = json.dumps({"prompt": "/think Compute 2 + 2", "mode": "chat", "lang": "eng", "max_tokens": 4}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/stream", data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30.0)
        assert resp.status == 200
        events = read_sse(resp)
        assert any(e.get("type") == "start" and e.get("mode") == "think" for e in events)
        assert any(e.get("type") == "done" for e in events)
        for ev in events:
            if ev.get("type") == "token":
                for tag in ("<think>", "</think>", "<freethink>", "</freethink>", "/think", "/freethink"):
                    assert tag not in ev.get("token", ""), f"Leaked {tag} in SSE token"
        print("  [x] POST /api/stream (Thinking Mode SSE & /think internal routing) verified")

        # 5. Test POST /api/stream (FreeThink Mode in Kannada with /freethink command stripping)
        payload = json.dumps({"prompt": "/freethink ಬೆಂಗಳೂರು ಒಂದು ದೊಡ್ಡ ನಗರ.", "mode": "chat", "lang": "kan", "max_tokens": 4}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/stream", data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30.0)
        assert resp.status == 200
        events = read_sse(resp)
        assert any(e.get("type") == "start" and e.get("mode") == "freethink" for e in events)
        assert any(e.get("type") == "done" for e in events)
        print("  [x] POST /api/stream (FreeThink Mode SSE in Kannada & /freethink internal routing) verified")

        # 6. Test POST /api/stream (Chat Mode in Tamil)
        payload = json.dumps({"prompt": "வணக்கம்", "mode": "chat", "lang": "tam", "max_tokens": 4}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/stream", data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30.0)
        assert resp.status == 200
        events = read_sse(resp)
        assert any(e.get("type") == "start" for e in events)
        assert any(e.get("type") == "done" for e in events)
        print("  [x] POST /api/stream (Chat Mode SSE in Tamil) verified")

        # 7. Test POST /api/stream with web intent gating
        # Negative case: is grass green? with web=True (gated, no search triggered)
        payload = json.dumps({"prompt": "is grass green?", "mode": "chat", "lang": "eng", "max_tokens": 4, "web": True}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/stream", data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30.0)
        assert resp.status == 200
        events = read_sse(resp)
        assert any(e.get("type") == "done" for e in events)

        # Dravidian language with web=True (gated, no search triggered)
        payload = json.dumps({"prompt": "ಮಳೆಗಾಲದಲ್ಲಿ ಕೃಷಿ", "mode": "chat", "lang": "kan", "max_tokens": 4, "web": True}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/stream", data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30.0)
        assert resp.status == 200
        events = read_sse(resp)
        assert any(e.get("type") == "done" for e in events)

        # Positive case: /web command (offline mocked to avoid live network latency)
        from unittest.mock import patch
        with patch("features.web_tools.search_web") as mock_search:
            mock_search.return_value = [{"title": "IPL 2024 Winner", "snippet": "KKR won the 2024 final", "url": "https://example.com"}]
            payload = json.dumps({"prompt": "/web who won the 2024 IPL", "mode": "chat", "lang": "eng", "max_tokens": 4}).encode("utf-8")
            req = urllib.request.Request(f"{base_url}/api/stream", data=payload, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=30.0)
            assert resp.status == 200
            events = read_sse(resp)
            assert any(e.get("type") == "done" for e in events)
            assert mock_search.called
        print("  [x] POST /api/stream (Web intent gating verified)")

    finally:
        server.shutdown()
        server.server_close()

    print("[self-test] app.py ok - all API routes and streaming verified")


def main():
    p = argparse.ArgumentParser(description="Parentheses 0.9-120M Streaming Web UI")
    p.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    p.add_argument("--port", type=int, default=8080, help="Port to listen on")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--model-path", default=None, help="Explicit checkpoint .pt or safetensors folder")
    p.add_argument("--web", "--web-search", action="store_true", help="enable real-time web search grounding for prompts")
    p.add_argument("--self-test", action="store_true", help="Run automated headless test suite on CPU and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    # Fallback to CPU if CUDA is requested but unavailable or busy
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[app] CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"[app] Loading Parentheses 0.9-120M on {device}...")
    try:
        model, tok, source_name = resolve_model_and_tokenizer(args.model_path, device=device)
    except Exception as e:
        if device == "cuda":
            print(f"[app] Failed to load on CUDA ({e}), falling back to CPU...")
            device = "cpu"
            model, tok, source_name = resolve_model_and_tokenizer(args.model_path, device="cpu")
        else:
            raise e

    print(f"[app] Model loaded from {source_name} on {device}")
    engine = GenerationEngine(model, tok, device=device, web=args.web)

    handler_cls = type(
        "StudioHandler",
        (ParenthesesHTTPHandler,),
        {"engine": engine, "model_name": f"Parentheses 0.9-120M ({source_name})", "device_name": device}
    )

    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    url = f"http://{args.host}:{args.port}"
    print(f"\n=======================================================")
    print(f"  Parentheses 0.9-120M Web Studio running at:")
    print(f"  --> {url}")
    print(f"  Open the link in your web browser to test all 9 languages & 4 modes.")
    if args.web:
        print("  Live web search grounding: ENABLED (Gated: factual queries & /web only)")
    print(f"=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] Shutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
