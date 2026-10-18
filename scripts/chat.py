"""
Interactive terminal chat: type a prompt, get a completion back, repeat.
For eyeballing quality by hand -- real-time token streaming with ANSI styling.

Supports 4 generation modes:
  - think      : Structured <think>...</think> reasoning (dim cyan) + answer (bold bright)
  - chat       : Direct assistant response (<|user|><lang>...<|assistant|>)
  - freethink  : Continuous open-ended FreeThink Mode stream
  - base       : Direct <lang> continuation

Engineered by uncoalesced

Usage:
    python scripts/chat.py --checkpoint checkpoints/parentheses-0.9-120M-sft/step_9636_final.pt --format think --lang eng
    python scripts/chat.py --checkpoint checkpoints/parentheses-0.9-120M-sft/step_9636_final.pt --format think --lang kan
    python scripts/chat.py --checkpoint release/parentheses-0.9-120M-candidate/model.safetensors --format chat
    python scripts/chat.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

if sys.platform == "win32":
    os.system("")  # Enables ANSI VT100 escape codes in Windows conhost / Windows Terminal
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate_dir in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) > 1 else SCRIPT_DIR]:
    if (candidate_dir / "model").exists() and str(candidate_dir) not in sys.path:
        sys.path.insert(0, str(candidate_dir))

try:
    from model import PRESETS, Parentheses
    from model.config import ModelConfig
except ImportError:
    from model.backbone import Parentheses
    from model.config import ModelConfig
    PRESETS = {}

# Self-contained fallbacks so chat.py runs standalone in release/ without extra repo dependencies
try:
    from eval_queries import STOP_TOKENS, looping
except ImportError:
    STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>")

    def looping(ids: list[int], window: int = 32, repeats: int = 4) -> bool:
        need = window * repeats
        if len(ids) < need:
            return False
        tail = ids[-need:]
        first = tail[:window]
        return all(tail[i * window:(i + 1) * window] == first for i in range(1, repeats))

try:
    from features.free_think import (
        INTERNAL_COMMANDS,
        INTERNAL_TAGS,
        LEGACY_ARTIFACT_PHRASES,
        parse_internal_command,
        strip_internal_tags,
    )
except ImportError:
    import re as _re

    INTERNAL_COMMANDS = ("/think", "/freethink", "/free-think", "/nothink", "/no-think", "/chat", "/base")
    INTERNAL_TAGS = ("<think>", "</think>", "<freethink>", "</freethink>")
    _CMD_RE = _re.compile(r"(?<!\w)/(?:freethink|free-think|nothink|no-think|think|chat|base)(?!\w)", _re.IGNORECASE)
    _TAG_RE = _re.compile(
        r"</?(?:freethink|free-think|think)>|<\|(?:user|assistant|system|endoftext|pad)\|>|<esg>|</?tool_call>",
        _re.IGNORECASE,
    )

    LEGACY_ARTIFACT_PHRASES = (
        "Examining the key principles behind this statement,",
        "Examining the key principles behind this statement.",
        "Examining the key principles behind this statement:",
        "Examining the key principles behind this statement",
        "ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ,",
        "ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ.",
        "ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ",
        "இதன் முக்கியக் கூறுகளை ஆராய்ந்தால்,",
        "இதன் முக்கியக் கூறுகளை ஆராய்ந்தால்.",
        "இதன் முக்கியக் கூறுகளை ஆராய்ந்தால்",
        "దీనిలోని ముఖ్య అంశాలను విశ్లేషిస్తే,",
        "దీనిలోని ముఖ్య అంశాలను విశ్లేషిస్తే.",
        "దీనిలోని ముఖ్య అంశాలను విశ్లేషిస్తే",
        "ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ,",
        "ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ.",
        "ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ",
        "ಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ,",
        "ಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ.",
        "ಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ",
        "ଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ,",
        "ଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ.",
        "ଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ",
        "ଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ,",
        "ଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ.",
        "ଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ",
        "ఇదా ఎన్నె అని మరి,",
        "ఇదా ఎన్నె అని మరి.",
        "ఇదా ఎన్నె అని మరి",
    )

    def parse_internal_command(prompt: str, default_mode: str | None = None) -> tuple[str, str | None]:
        if not prompt:
            return "", default_mode
        lower = prompt.lower()
        cmd = None
        if any(t in lower for t in ("/freethink", "/free-think", "<freethink>", "</freethink>")):
            cmd = "freethink"
        elif any(t in lower for t in ("/nothink", "/no-think")):
            cmd = "chat"
        elif any(t in lower for t in ("/think", "<think>", "</think>")):
            cmd = "think"
        elif "/chat" in lower:
            cmd = "chat"
        elif "/base" in lower:
            cmd = "base"
        cleaned = _CMD_RE.sub(" ", _TAG_RE.sub(" ", prompt))
        cleaned = _re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        return cleaned, (cmd or default_mode)

    def strip_internal_tags(text: str) -> str:
        if not text:
            return ""
        out = _CMD_RE.sub("", _TAG_RE.sub("", text))
        stripped = out.lstrip(" \t\r\n")
        for phrase in LEGACY_ARTIFACT_PHRASES:
            if stripped.startswith(phrase):
                return stripped[len(phrase):].lstrip(" \t\r\n,.:;")
        return out

STYLE_THINK = "\033[2;36m"   # Dim cyan
STYLE_ANSWER = "\033[0;1m"   # Bold / bright normal
STYLE_RESET = "\033[0m"      # Reset formatting


import re as _mm_re

_MM_CMD_RE = _mm_re.compile(
    r'(?:^|\s)/(?:image|img|doc|ingest)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    _mm_re.IGNORECASE,
)
_WEB_CMD_RE = _mm_re.compile(
    r'(?:^|\s)/(?:web|search)\s+(?:"([^"]+)"|\'([^\']+)\'|([^\n\r/]+))',
    _mm_re.IGNORECASE,
)
_URL_CMD_RE = _mm_re.compile(
    r'(?:^|\s)/(?:url|fetch)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    _mm_re.IGNORECASE,
)


# SFT v1-v3 never trained a <|system|> preamble (<|system|> only appears as
# "tool_result:" turns), but without an identity prompt the model invents
# facts instead of abstaining, so the identity prompt is ON by default and
# --no-system is the opt-out.
DEFAULT_SYSTEM_PROMPT = (
    "You are Parentheses 0.9, an efficient Dravidian and English recurrent language model "
    "developed by Joel Anthony. You answer questions directly, concisely, and accurately. "
    "If you do not know the answer to a factual question, say \"I don't know\" instead of "
    "inventing details."
)


# ---------------------------------------------------------------------------
# Deterministic math intercept: simple arithmetic is answered exactly with a
# whitelisted AST evaluator (never eval()) and the model is skipped.
# ---------------------------------------------------------------------------
import ast as _ast
import operator as _op
from fractions import Fraction as _Fraction

_NUM = r"\d+(?:\.\d+)?"
_MATH_PREFIX_RE = _mm_re.compile(
    r"^\s*(?:what\s+is|what's|whats|calculate|compute|evaluate|find)\s+", _mm_re.IGNORECASE
)
_MATH_SUFFIX_RE = _mm_re.compile(r"\s*(?:=\s*)?[\?\.!\s]*$")
_PERCENT_OF_RE = _mm_re.compile(rf"^({_NUM})\s*%\s*of\s*\$?({_NUM})$", _mm_re.IGNORECASE)
_SQUARED_RE = _mm_re.compile(rf"^(?:the\s+square\s+of\s+({_NUM})|({_NUM})\s+(squared|cubed))$", _mm_re.IGNORECASE)
_PCT_LIST = rf"((?:{_NUM}\s*%\s*(?:,\s*|\s+and\s+|\s+then\s+|\s+followed\s+by\s+)?)+)"
_DISCOUNT_RE = _mm_re.compile(
    rf"^(?:the\s+price\s+of\s+)?(?:an?\s+)?\$?({_NUM})(?:\s*(?:rs\.?|rupees?|dollars?|inr|usd))?"
    rf"(?:\s+(?:item|price|product|shirt|thing))?\s+(?:with|after|at|less)\s+"
    rf"(?:successive\s+|two\s+|three\s+)?(?:discounts?\s+of\s+)?{_PCT_LIST}\s*(?:successive\s+)?(?:discounts?|off)?$",
    _mm_re.IGNORECASE,
)
_DISCOUNT_ONLY_RE = _mm_re.compile(
    rf"^(?:the\s+)?(?:single\s+discount\s+equivalent\s+to\s+)?(?:successive\s+|combined\s+)?"
    rf"discounts?\s+of\s+{_PCT_LIST}$",
    _mm_re.IGNORECASE,
)
_COST_RE = _mm_re.compile(
    rf"^(?:if\s+)?(?:an?\s+|the\s+)?(?:product|item|shirt|thing|price)?\s*(?:costs?|priced\s+at)\s+₹?\$?({_NUM})\b",
    _mm_re.IGNORECASE,
)
_ANSWER_CLAUSE_RE = _mm_re.compile(
    r"[\s,.:;]*\b(?:answer|respond|reply)\b[^.!?\n]*[.!?]?\s*$",
    _mm_re.IGNORECASE,
)
_EXPR_CHARS_RE = _mm_re.compile(r"^[\d\.\s\+\-\*/\^\(\)x×÷]+$")
_BIN_OPS = {
    _ast.Add: _op.add,
    _ast.Sub: _op.sub,
    _ast.Mult: _op.mul,
    _ast.Div: _op.truediv,
    _ast.Pow: _op.pow,
}
_UNARY_OPS = {_ast.UAdd: _op.pos, _ast.USub: _op.neg}


def _safe_eval(expr: str) -> _Fraction:
    """Evaluate + - * / ** over numeric literals only (exact Fractions). Raises ValueError otherwise."""
    tree = _ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, _ast.Expression):
            return ev(node.body)
        if isinstance(node, _ast.Constant) and type(node.value) in (int, float):
            return _Fraction(str(node.value))
        if isinstance(node, _ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, _ast.Pow) and (right.denominator != 1 or abs(right) > 64 or abs(left) > 10**12):
                raise ValueError("exponent out of range")
            if isinstance(node.op, _ast.Div) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _Fraction(_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, _ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    return ev(tree)


def _fmt_num(v: _Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


def _pct_list(text: str) -> list[_Fraction]:
    return [_Fraction(x) for x in _mm_re.findall(_NUM, text)]


def solve_simple_math(prompt: str) -> str | None:
    """Return an exact answer for simple arithmetic prompts, else None (the model handles it)."""
    if not prompt:
        return None
    text = prompt.strip()
    if len(text) > 120 or "\n" in text:
        return None
    body = _MATH_PREFIX_RE.sub("", text)
    body = _ANSWER_CLAUSE_RE.sub("", body)
    body = _MATH_SUFFIX_RE.sub("", body).strip()
    body = _mm_re.sub(r"(?<=\d),(?=\d{3}\b)", "", body)  # 1,000 -> 1000
    if not body:
        return None

    m = _PERCENT_OF_RE.match(body)
    if m:
        v = _Fraction(m.group(1)) * _Fraction(m.group(2)) / 100
        return f"{m.group(1)}% of {m.group(2)} is {_fmt_num(v)}."

    m = _SQUARED_RE.match(body)
    if m:
        base = m.group(1) or m.group(2)
        word = (m.group(3) or "squared").lower()
        v = _Fraction(base) ** (2 if word == "squared" else 3)
        return f"{base} {word} is {_fmt_num(v)}."

    m = _DISCOUNT_RE.match(body)
    if m:
        pcts = _pct_list(m.group(2))
        if pcts and all(0 <= p <= 100 for p in pcts):
            start = _Fraction(m.group(1))
            price = start
            for p in pcts:
                price *= 1 - p / 100
            total_off = (1 - price / start) * 100 if start else _Fraction(0)
            return (f"After discounts of {', '.join(_fmt_num(p) + '%' for p in pcts)}, "
                    f"the price is {_fmt_num(price)} (a total discount of {_fmt_num(total_off)}%).")

    m = _DISCOUNT_ONLY_RE.match(body)
    if m:
        pcts = _pct_list(m.group(1))
        if len(pcts) >= 2 and all(0 <= p <= 100 for p in pcts):
            keep = _Fraction(1)
            for p in pcts:
                keep *= 1 - p / 100
            return (f"Successive discounts of {', '.join(_fmt_num(p) + '%' for p in pcts)} "
                    f"equal a single discount of {_fmt_num((1 - keep) * 100)}%.")

    m = _COST_RE.match(body)
    if m and _mm_re.search(r"\b(?:discount|off)\b", body, _mm_re.IGNORECASE):
        start = _Fraction(m.group(1))
        pcts = [_Fraction(x) for x in _mm_re.findall(rf"({_NUM})\s*%", body[m.end():])]
        if pcts and all(0 <= p <= 100 for p in pcts):
            price = start
            for p in pcts:
                price *= 1 - p / 100
            total_off = (1 - price / start) * 100 if start else _Fraction(0)
            return (f"After discounts of {', '.join(_fmt_num(p) + '%' for p in pcts)}, "
                    f"the price is {_fmt_num(price)} (a total discount of {_fmt_num(total_off)}%).")

    expr = _mm_re.sub(r"\s+plus\s+", " + ", body, flags=_mm_re.IGNORECASE)
    expr = _mm_re.sub(r"\s+minus\s+", " - ", expr, flags=_mm_re.IGNORECASE)
    expr = _mm_re.sub(r"\s+(?:times|multiplied\s+by)\s+", " * ", expr, flags=_mm_re.IGNORECASE)
    expr = _mm_re.sub(r"\s+(?:divided\s+by|over)\s+", " / ", expr, flags=_mm_re.IGNORECASE)
    expr = _mm_re.sub(r"\s+to\s+the\s+power\s+of\s+", " ^ ", expr, flags=_mm_re.IGNORECASE)
    if not _EXPR_CHARS_RE.match(expr):
        return None
    if not _mm_re.search(r"\d\s*[\+\-\*/\^x×÷]\s*[\(\d\-]", expr):
        return None  # a lone number is not a calculation
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = _mm_re.sub(r"(?<=[\d\)])\s*x\s*(?=[\d\(])", "*", expr)
    if "x" in expr:
        return None  # algebra (a variable named x) is not plain arithmetic
    try:
        v = _safe_eval(expr)
    except ZeroDivisionError:
        return "Division by zero is undefined."
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
        return None
    return f"{' '.join(body.split())} = {_fmt_num(v)}"


# ---------------------------------------------------------------------------
# Web context: 1-2 extracted sentences, no URLs or bare domains, so the 120M
# model is not primed to regurgitate link lists.
# ---------------------------------------------------------------------------
_URLISH_RE = _mm_re.compile(
    r"(?:https?://\S+|www\.\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|in|co|uk|info|news|ai)\b(?:/\S*)?)",
    _mm_re.IGNORECASE,
)
_SENT_SPLIT_RE = _mm_re.compile(r"(?<=[\.!\?])\s+")


def compact_web_snippet(results: list[dict], max_sentences: int = 2, max_words: int = 60) -> str:
    """First usable result snippet, cut to its first 1-2 sentences, with URLs and domains removed."""
    for r in results or []:
        raw = _mm_re.sub(r"\s+", " ", r.get("snippet") or r.get("text") or "").strip()
        sents = []
        # Split first so a URL that ends a sentence doesn't take its full stop with it.
        for s in _SENT_SPLIT_RE.split(raw):
            end = s[-1] if s and s[-1] in ".!?" else ""
            s = _URLISH_RE.sub("", s)
            s = _mm_re.sub(r"\s{2,}", " ", s).strip(" -|·.!?")
            if s:
                sents.append(s + (end or "."))
        if len(" ".join(sents).split()) < 4:
            continue
        sents = [s for s in sents if s.strip()][:max_sentences]
        words = " ".join(sents).split()
        out = " ".join(words[:max_words])
        return out + "..." if len(words) > max_words else out
    return ""


LANG_NAMES = {
    "eng": "English",
    "kan": "ಕನ್ನಡ (Kannada)",
    "tam": "தமிழ் (Tamil)",
    "tel": "తెలుగు (Telugu)",
    "mal": "മലയാളം (Malayalam)",
    "tcy": "ತುಳು (Tulu)",
    "kff": "କୋୟା (Koya)",
    "kxv": "କୁୱି (Kuvi)",
    "nit": "నాయకపోడ్ (SE Gondi)",
}


def detect_language(text: str, default: str = "eng") -> str:
    """Accurately detects language from text based on Unicode script distribution and lexical cues."""
    if not text:
        return default

    # Check explicit command overrides
    m = _mm_re.search(r"(?:^|\s)/(eng|kan|tam|tel|mal|tcy|kff|kxv|nit)\b", text, _mm_re.IGNORECASE)
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


_LANG_CMD_RE = _mm_re.compile(
    r'(?:^|\s)/(?:lang\s+(\w+)|(eng|kan|tam|tel|mal|tcy|kff|kxv|nit|auto))\b',
    _mm_re.IGNORECASE,
)


def _parse_multimodal_commands(prompt: str) -> tuple[str, list[str], list[str], list[str], list[str], str | None]:
    """Extract and strip inline `/image`, `/doc`, `/web`, `/url`, and language commands (`/kan`, `/tam`, etc.)."""
    if not prompt:
        return "", [], [], [], [], None
    images: list[str] = []
    docs: list[str] = []
    web_queries: list[str] = []
    web_urls: list[str] = []
    target_lang: str | None = None

    for match in _LANG_CMD_RE.finditer(prompt):
        val = (match.group(1) or match.group(2) or "").lower()
        if val in ("eng", "kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "auto"):
            target_lang = val

    for match in _MM_CMD_RE.finditer(prompt):
        full = match.group(0).strip().lower()
        target = match.group(1) or match.group(2) or match.group(3)
        if not target:
            continue
        if full.startswith(("/image", "/img")):
            images.append(target)
        else:
            docs.append(target)

    for match in _WEB_CMD_RE.finditer(prompt):
        q = match.group(1) or match.group(2) or match.group(3)
        if q and q.strip():
            web_queries.append(q.strip())

    for match in _URL_CMD_RE.finditer(prompt):
        u = match.group(1) or match.group(2) or match.group(3)
        if u and u.strip():
            web_urls.append(u.strip())

    cleaned = _LANG_CMD_RE.sub(" ", prompt)
    cleaned = _MM_CMD_RE.sub(" ", cleaned)
    cleaned = _WEB_CMD_RE.sub(" ", cleaned)
    cleaned = _URL_CMD_RE.sub(" ", cleaned)
    cleaned = _mm_re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, images, docs, web_queries, web_urls, target_lang


@torch.no_grad()
def reply(
    model: Parentheses,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    tok=None,
    lang: str = "auto",
    fmt: str = "think",
    stream: bool = True,
    images: list[str | Path] | None = None,
    docs: list[str | Path] | None = None,
    web: bool = False,
    system: str | None = None,
) -> str:
    """Generate completion with real-time streaming, ANSI styling, dynamic multilingual and web grounding.

    All `/think`, `/freethink`, `/nothink`, `/chat`, `/base`, `/image`, `/doc`,
    `/web`, `/url`, `/kan`, `/tam`, etc. commands and tags are handled strictly
    internally and automatically hidden from visible output.
    """
    prompt_no_mm, inline_imgs, inline_docs, inline_web_q, inline_web_u, inline_lang = _parse_multimodal_commands(prompt)
    if prompt_no_mm:
        prompt = prompt_no_mm
    clean_prompt, resolved_fmt = parse_internal_command(prompt, default_mode=fmt)
    if clean_prompt:
        prompt = clean_prompt
    if resolved_fmt in ("think", "chat", "freethink", "base"):
        fmt = resolved_fmt

    # Resolve active language dynamically: explicit inline > passed lang > auto-detect
    if inline_lang and inline_lang != "auto":
        active_lang = inline_lang
    elif lang and lang != "auto":
        active_lang = lang
    else:
        active_lang = detect_language(clean_prompt or prompt, default="eng")
    lang = active_lang

    # Simple arithmetic is answered exactly; the model never sees it.
    math_ans = solve_simple_math(prompt)
    if math_ans is not None:
        if stream:
            sys.stdout.write(STYLE_ANSWER + math_ans + STYLE_RESET + "\n")
            sys.stdout.flush()
        return math_ans

    all_imgs = list(images or []) + inline_imgs
    all_docs = list(docs or []) + inline_docs

    # Execute web search / URL fetch if requested
    web_blocks: list[str] = []
    if web or inline_web_q or inline_web_u:
        try:
            from features.web_tools import (
                search_web,
                fetch_url,
                should_trigger_web_search,
            )
            queries = list(inline_web_q)
            # Only search if explicit inline query OR if intent gating approves real-time lookup
            if web and not inline_web_q and prompt:
                if should_trigger_web_search(prompt, web_enabled=True, lang=lang):
                    queries.append(prompt)
            for q in queries:
                snippet = compact_web_snippet(search_web(q, max_results=3) or [])
                if snippet:
                    web_blocks.append(f"[Web Context: {snippet}]")
            for u in inline_web_u:
                doc_res = fetch_url(u, max_chars=1500)
                snippet = compact_web_snippet([doc_res]) if doc_res.get("title") != "Error" else ""
                if snippet:
                    web_blocks.append(f"[Web Context: {snippet}]")
        except Exception:
            pass

    context_blocks: list[str] = []
    if all_imgs or all_docs:
        try:
            from features.multimodal import describe_image, extract_document_text, prefill_visual_state
            vis_states = None
            for img_p in all_imgs:
                desc = describe_image(img_p, lang=lang)
                context_blocks.append(desc.summary)
                if getattr(model.cfg, "attn_type", "") == "selective_linear":
                    vis_states = prefill_visual_state(model, img_p, states=vis_states)
            for doc_p in all_docs:
                doc_txt = extract_document_text(doc_p, lang=lang).strip()
                if doc_txt:
                    max_chars = max(256, getattr(model.cfg, "block_size", 512) // 2)
                    context_blocks.append(doc_txt[:max_chars])
        except Exception:
            pass

    if web_blocks:
        context_blocks.extend(web_blocks)

    if context_blocks:
        grounded_ctx = "\n\n".join(context_blocks)
        prompt = f"{grounded_ctx}\n\n{prompt}".strip()

    if tok is None:
        if fmt == "freethink":
            from features.free_think import FreeThinkSession
            sess = FreeThinkSession(model, temperature=temperature, top_k=top_k, tokenizer=None, lang=lang)
            chunks = []
            if stream:
                sys.stdout.write(STYLE_THINK)
                sys.stdout.flush()
            for chunk in sess.start(prompt, max_tokens=max_new_tokens, force=True):
                clean_c = strip_internal_tags(chunk)
                if clean_c:
                    chunks.append(clean_c)
                    if stream:
                        sys.stdout.write(clean_c)
                        sys.stdout.flush()
            if stream:
                sys.stdout.write(STYLE_RESET + "\n")
                sys.stdout.flush()
            return "".join(chunks)

        byte_prompt = f"{system}\n\n{prompt}" if system else prompt
        ids = torch.tensor([list(byte_prompt.encode("utf-8"))], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
        full = bytes(out[0].tolist()).decode("utf-8", errors="replace")
        ans = strip_internal_tags(full[len(byte_prompt):])
        if stream:
            style = STYLE_THINK if fmt == "think" else STYLE_ANSWER
            sys.stdout.write(style + ans + STYLE_RESET + "\n")
            sys.stdout.flush()
        return ans

    if fmt == "freethink":
        try:
            from features.free_think import FreeThinkSession
            sess = FreeThinkSession(model, temperature=temperature, top_k=top_k, tokenizer=tok, lang=lang)
            gen = sess.start(prompt, max_tokens=max_new_tokens, force=True)
        except ImportError:
            priming_map = {
                "eng": f"{prompt}\n\nThinking about this further,",
                "kan": f"{prompt}\n\nಇದರ ಬಗ್ಗೆ ಮತ್ತಷ್ಟು ವಿವರಿಸುವುದಾದರೆ,",
                "tam": f"{prompt}\n\nஇதைப் பற்றி மேலும் விரிவாகக் கூறினால்,",
                "tel": f"{prompt}\n\nదీని గురించి మరింత వివరంగా చెప్పాలంటే,",
                "mal": f"{prompt}\n\nഇതിനെക്കുറിച്ച് കൂടുതൽ വിശദമായി പറഞ്ഞാൽ,",
            }
            cue = priming_map.get(lang, f"{prompt}\n\nThinking about this further,")
            tag_id = tok.token_to_id(f"<{lang}>") or tok.token_to_id("<eng>")
            in_ids = [tag_id] + tok.encode(cue, add_special_tokens=False).ids
            idx = torch.tensor([in_ids], dtype=torch.long, device=device)

            def _fallback_gen():
                p = []
                for nxt in model.stream(
                    idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                    min_p=0.06, repetition_penalty=1.12, no_repeat_ngram_size=4,
                ):
                    p.append(int(nxt.item()))
                    s = tok.decode(p, skip_special_tokens=False)
                    if s and not s.endswith("\ufffd"):
                        yield strip_internal_tags(s)
                        p = []
                if p:
                    yield strip_internal_tags(tok.decode(p, skip_special_tokens=False))

            gen = _fallback_gen()

        chunks = []
        if stream:
            sys.stdout.write(STYLE_THINK)
            sys.stdout.flush()
        for chunk in gen:
            clean_c = strip_internal_tags(chunk)
            if clean_c:
                chunks.append(clean_c)
                if stream:
                    sys.stdout.write(clean_c)
                    sys.stdout.flush()
        if stream:
            sys.stdout.write(STYLE_RESET + "\n")
            sys.stdout.flush()
        return "".join(chunks)

    tag_id = tok.token_to_id(f"<{lang}>") or tok.token_to_id("<eng>")
    user_id = tok.token_to_id("<|user|>")
    asst_id = tok.token_to_id("<|assistant|>")
    think_id = tok.token_to_id("<think>")
    think_end_id = tok.token_to_id("</think>")

    body = tok.encode(prompt.strip(), add_special_tokens=False).ids

    # Strip redundant leading language tag if user typed one in prompt
    lang_tags = {tok.token_to_id(f"<{l}>") for l in ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng", "esg", "hin")} - {None}
    if body and body[0] in lang_tags:
        tag_id = body[0]
        body = body[1:]

    nl_ids = tok.encode("\n", add_special_tokens=False).ids
    sys_ids: list[int] = []
    system_id = tok.token_to_id("<|system|>")
    if system and system_id is not None and fmt in ("think", "chat"):
        sys_ids = [system_id] + tok.encode(system.strip(), add_special_tokens=False).ids
    if fmt == "think":
        in_ids = sys_ids + [user_id, tag_id] + body + [asst_id, think_id] + nl_ids
    elif fmt == "chat":
        in_ids = sys_ids + [user_id, tag_id] + body + [asst_id]
    else:  # base
        in_ids = [tag_id] + body

    stops = {tok.token_to_id(s) for s in STOP_TOKENS + ("<|assistant|>", "<|pad|>", "<esg>")} - {None}
    idx = torch.tensor([in_ids], dtype=torch.long, device=device)

    collected: list[str] = []
    out_tokens: list[int] = []
    pending: list[int] = []

    in_think = (fmt == "think")
    filter_artifact = in_think
    strip_leading_punct = False
    artifact_buf = ""

    if stream:
        sys.stdout.write(STYLE_THINK if in_think else STYLE_ANSWER)
        sys.stdout.flush()

    for nxt in model.stream(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        min_p=0.06,
        repetition_penalty=1.12,
        no_repeat_ngram_size=4,
    ):
        t = int(nxt.item())
        if t in stops:
            if not out_tokens:
                continue
            break

        out_tokens.append(t)

        if t == think_id:
            in_think = True
            filter_artifact = True
            strip_leading_punct = False
            if stream:
                sys.stdout.write(STYLE_THINK)
                sys.stdout.flush()
            continue

        if t == think_end_id:
            if in_think:
                if filter_artifact and artifact_buf:
                    s = artifact_buf.lstrip(" \t\r\n")
                    for p in LEGACY_ARTIFACT_PHRASES:
                        if s.startswith(p):
                            artifact_buf = s[len(p):].lstrip(" \t\r\n,.:;")
                            break
                    if artifact_buf:
                        collected.append(artifact_buf)
                        if stream:
                            sys.stdout.write(artifact_buf)
                    artifact_buf = ""
                filter_artifact = False
                strip_leading_punct = False

                if pending:
                    chunk = strip_internal_tags(tok.decode(pending, skip_special_tokens=False))
                    if chunk:
                        collected.append(chunk)
                        if stream:
                            sys.stdout.write(chunk)
                    pending = []
                sep = "\n\n"
                collected.append(sep)
                if stream:
                    sys.stdout.write(STYLE_RESET + sep + STYLE_ANSWER)
                    sys.stdout.flush()
                in_think = False
            continue

        pending.append(t)
        chunk = tok.decode(pending, skip_special_tokens=False)
        if chunk and not chunk.endswith("\ufffd"):
            pending = []
            clean_c = strip_internal_tags(chunk)
            if not clean_c:
                continue

            if filter_artifact:
                artifact_buf += clean_c
                s = artifact_buf.lstrip(" \t\r\n")
                if any(p.startswith(s) and len(p) > len(s) for p in LEGACY_ARTIFACT_PHRASES):
                    continue
                matched_phrase = False
                for p in LEGACY_ARTIFACT_PHRASES:
                    if s.startswith(p):
                        artifact_buf = s[len(p):].lstrip(" \t\r\n,.:;")
                        matched_phrase = True
                        break
                if matched_phrase:
                    filter_artifact = False
                    strip_leading_punct = True
                    clean_c = artifact_buf
                    artifact_buf = ""
                elif any(p.startswith(s) for p in LEGACY_ARTIFACT_PHRASES):
                    continue
                else:
                    filter_artifact = False
                    clean_c = artifact_buf
                    artifact_buf = ""

            if strip_leading_punct and clean_c:
                clean_c = clean_c.lstrip(" \t\r\n,.:;")
                if clean_c:
                    strip_leading_punct = False

            if clean_c:
                collected.append(clean_c)
                if stream:
                    sys.stdout.write(clean_c)
                    sys.stdout.flush()

        if looping(out_tokens):
            break

    if filter_artifact and artifact_buf:
        s = artifact_buf.lstrip(" \t\r\n")
        for p in LEGACY_ARTIFACT_PHRASES:
            if s.startswith(p):
                artifact_buf = s[len(p):].lstrip(" \t\r\n,.:;")
                break
        if artifact_buf:
            collected.append(artifact_buf)
            if stream:
                sys.stdout.write(artifact_buf)
        artifact_buf = ""

    if pending:
        clean_c = strip_internal_tags(tok.decode(pending, skip_special_tokens=False))
        if clean_c:
            collected.append(clean_c)
            if stream:
                sys.stdout.write(clean_c)

    if stream:
        sys.stdout.write(STYLE_RESET + "\n")
        sys.stdout.flush()

    raw_ans = "".join(collected)
    return strip_internal_tags(raw_ans)


def load_model_and_tokenizer(ckpt_path: Path | str, device: str = "cpu"):
    """Load model and tokenizer from either a .pt checkpoint or release directory."""
    path = Path(ckpt_path)
    if not path.exists():
        if (SCRIPT_DIR / ckpt_path).exists():
            path = SCRIPT_DIR / ckpt_path
        elif (SCRIPT_DIR / "model.safetensors").exists():
            path = SCRIPT_DIR / "model.safetensors"
        elif (SCRIPT_DIR.parent / ckpt_path).exists():
            path = SCRIPT_DIR.parent / ckpt_path

    if path.is_dir() or path.suffix == ".safetensors":
        release_dir = path if path.is_dir() else path.parent
        import json
        from tokenizers import Tokenizer
        cfg_dict = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
        cfg = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 32768),
            block_size=cfg_dict.get("block_size", 4096),
            n_layer=cfg_dict.get("n_layer", 32),
            n_head=cfg_dict.get("n_head", 8),
            n_embd=cfg_dict.get("n_embd", 512),
            dropout=cfg_dict.get("dropout", 0.0),
            bias=cfg_dict.get("bias", False),
            tie_embeddings=cfg_dict.get("tie_embeddings", True),
            attn_type=cfg_dict.get("attn_type", "selective_linear"),
            slab_v2=cfg_dict.get("slab_v2", False),
        )
        model = Parentheses(cfg)
        st_file = release_dir / "model.safetensors"
        try:
            import safetensors.torch
            sd = safetensors.torch.load_file(str(st_file))
        except ImportError:
            import struct
            with open(st_file, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len).decode("utf-8"))
                raw = bytearray(f.read())
            dtype_map = {"F32": torch.float32, "BF16": torch.bfloat16}
            mv = memoryview(raw)
            sd = {}
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                s, e = v["data_offsets"]
                sd[k] = torch.frombuffer(mv[s:e], dtype=dtype_map[v["dtype"]]).reshape(v["shape"]).clone()

        model.load_state_dict(sd, strict=False)
        if cfg.tie_embeddings:
            model.head.weight = model.tok_emb.weight
        model.to(device).eval()
        tok = Tokenizer.from_file(str(release_dir / "tokenizer.json"))
        tok.encode_special_tokens = True
        return model, tok, cfg, 9636

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = Parentheses(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    try:
        from features.free_think import load_bpe_tokenizer
        tok = load_bpe_tokenizer(ckpt)
    except ImportError:
        from tokenizers import Tokenizer
        tok_p = Path(ckpt["tokenizer"]["path"]) / "tokenizer.json"
        tok = Tokenizer.from_file(str(tok_p))
        tok.encode_special_tokens = True
    return model, tok, cfg, ckpt.get("step", 0)


def build_arg_parser(default_ckpt: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parentheses 0.9 Interactive Chat")
    p.add_argument("--checkpoint", default=default_ckpt)
    p.add_argument("--lang", default="auto", help="language code: eng, kan, tam, tel, mal, tcy, kff, kxv, nit, or 'auto' (default: dynamic auto-detection per prompt)")
    p.add_argument("--format", "--mode", choices=["think", "chat", "freethink", "base"], default="chat",
                   help="prompt format: chat (direct conversational response, default), think (with  thinking traces), freethink (open-ended stream), base")
    p.add_argument("--image", nargs="*", default=[], help="attach image(s) (.png, .jpg, .bmp, .webp) for SSM visual state & grounding")
    p.add_argument("--doc", "--ingest", dest="doc", nargs="*", default=[], help="attach document(s) (.pdf, .docx, .html, .csv, .json, .txt)")
    p.add_argument("--web", "--web-search", action="store_true",
                   help="allow web grounding for clearly real-time questions (/web <query> always searches); "
                        "injects a 1-2 sentence URL-free snippet")
    p.add_argument("--system", nargs="?", const=DEFAULT_SYSTEM_PROMPT, default=DEFAULT_SYSTEM_PROMPT,
                   help="prepend a <|system|> identity/honesty prompt (ON by default). "
                        "Bare --system uses the built-in Parentheses prompt; --system \"...\" sets custom text.")
    p.add_argument("--no-system", action="store_const", const=None, dest="system",
                   help="disable the default <|system|> identity/honesty prompt.")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--self-test", action="store_true")
    return p


def main():
    default_ckpt = "checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt"
    if not Path(default_ckpt).exists():
        if Path("checkpoints/parentheses-0.9-120M-sft/step_9636_final.pt").exists():
            default_ckpt = "checkpoints/parentheses-0.9-120M-sft/step_9636_final.pt"
        elif (SCRIPT_DIR / "model.safetensors").exists():
            default_ckpt = str(SCRIPT_DIR / "model.safetensors")
        elif (SCRIPT_DIR / "release/parentheses-0.9-120M-candidate/model.safetensors").exists():
            default_ckpt = str(SCRIPT_DIR / "release/parentheses-0.9-120M-candidate/model.safetensors")

    p = build_arg_parser(default_ckpt)
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    model, tok, cfg, step = load_model_and_tokenizer(args.checkpoint, device=args.device)
    session_lang = args.lang
    session_images = list(args.image)
    session_docs = list(args.doc)
    print(f"[chat] Loaded {args.checkpoint} (step {step}, {model.num_params()/1e6:.1f}M params, "
          f"attn={cfg.attn_type}, lang={session_lang}, mode={args.format}) on {args.device}.")
    print("Multi-lingual auto-routing: ENABLED (type freely in English, Kannada, Tamil, Telugu, Malayalam, Tulu, Koya, Kuvi, Gondi).")
    if session_images or session_docs:
        print(f"[chat] Attached {len(session_images)} image(s) and {len(session_docs)} document(s).")
    if args.web:
        print("[chat] Web grounding: ENABLED for real-time questions only (or /web <query>).")
    if args.system:
        print("[chat] System prompt: ON (--no-system to disable).")
    else:
        print("[chat] System prompt: OFF (--system to enable).")
    print("Commands: /lang <code|auto>, /think, /freethink, /chat, /kan, /tam, /tel, /mal, /eng, /image <path>, /doc <path>, /web <query>, /url <url>; Empty line to quit\n")

    while True:
        try:
            prompt = input("\033[1;32m> \033[0m").strip().replace("\\n", "\n")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not prompt:
            break

        # Check inline session commands
        lower_p = prompt.lower().strip()
        if lower_p in ("/auto", "/lang auto"):
            session_lang = "auto"
            print("\033[1;36m[chat] Language mode switched to: AUTO (Dynamic per-prompt detection)\033[0m")
            continue
        elif lower_p.startswith("/lang "):
            parts = lower_p.split()
            if len(parts) >= 2 and (parts[1] in LANG_NAMES or parts[1] == "auto"):
                session_lang = parts[1]
                name = LANG_NAMES.get(session_lang, "Auto-detect")
                print(f"\033[1;36m[chat] Session default language switched to: {session_lang} ({name})\033[0m")
                continue

        cleaned_p, new_imgs, new_docs, new_web_q, new_web_u, inline_lang = _parse_multimodal_commands(prompt)
        if new_imgs:
            session_images.extend(new_imgs)
        if new_docs:
            session_docs.extend(new_docs)
        if not cleaned_p and (new_imgs or new_docs):
            print(f"[chat] Attached {len(new_imgs)} image(s), {len(new_docs)} document(s) to session context.")
            continue

        # Display dynamic auto-routing badge
        active_lang = inline_lang or (session_lang if session_lang != "auto" else detect_language(cleaned_p or prompt, default="eng"))
        lang_label = LANG_NAMES.get(active_lang, active_lang)
        print(f"\033[2;34m[Routing: {active_lang} ({lang_label})]\033[0m")

        reply(
            model,
            prompt,
            args.device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            tok=tok,
            lang=session_lang,
            fmt=args.format,
            stream=True,
            images=session_images,
            docs=session_docs,
            web=args.web,
            system=args.system,
        )


def _self_test():
    import tempfile
    from PIL import Image
    from features.multimodal import create_pdf_bytes

    # Deterministic math intercept: exact test-01.md arithmetic cases
    assert solve_simple_math("What is 17 × 6? Answer with the number only.") == "17 × 6 = 102"
    assert solve_simple_math("What is 144 divided by 12? Answer with the number only.") == "144 divided by 12 = 12"
    assert solve_simple_math("What is 15% of 80? Answer with the number only.") == "15% of 80 is 12."
    assert solve_simple_math("What is 7 squared? Answer with the number only.") == "7 squared is 49."
    assert solve_simple_math("What is 17 * 6?") == "17 * 6 = 102"
    assert solve_simple_math("144 / 12") == "144 / 12 = 12"
    assert solve_simple_math("7^2") == "7^2 = 49"
    assert solve_simple_math("1,000 + 1,500") == "1000 + 1500 = 2500"
    disc = solve_simple_math(
        "If a product costs ₹1,500 and receives a 20% discount followed by a 10% discount, what is its final price?"
    )
    assert disc is not None and "1080" in disc, disc
    assert solve_simple_math("Hello, how are you?") is None

    # --system is ON by default; --no-system is the opt-out
    _p = build_arg_parser("dummy.pt")
    assert _p.parse_args(["--self-test"]).system == DEFAULT_SYSTEM_PROMPT
    assert _p.parse_args(["--no-system"]).system is None
    assert _p.parse_args(["--system", "custom identity"]).system == "custom identity"

    torch.manual_seed(0)
    cfg = PRESETS.get("tiny-smoke", ModelConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2, n_embd=32, attn_type="selective_linear"))
    model = Parentheses(cfg).eval()
    out = reply(model, "12", "cpu", max_new_tokens=8, temperature=1.0, top_k=4, stream=False)
    assert len(out.encode("utf-8")) >= 1, out

    tok_dir = SCRIPT_DIR.parent / "data" / "tokenizer" / "v3_32k"
    if not (tok_dir / "tokenizer.json").exists():
        tok_dir = SCRIPT_DIR
    if (tok_dir / "tokenizer.json").exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        tok.encode_special_tokens = True
    else:
        from features.free_think import load_bpe_tokenizer
        tok = load_bpe_tokenizer({"tokenizer": {"path": str(tok_dir)}})
    bpe = Parentheses(ModelConfig(vocab_size=32768, block_size=256, n_layer=2, n_head=2, n_embd=32,
                                  attn_type="selective_linear")).eval()
    for fmt in ("chat", "think", "base", "freethink"):
        bpe_out = reply(bpe, "Hello world", "cpu", max_new_tokens=6, temperature=1.0, top_k=8,
                        tok=tok, lang="eng", fmt=fmt, stream=False)
        assert isinstance(bpe_out, str) and len(bpe_out) > 0, f"Failed for {fmt}"

    # Verify /image, /doc, /web, /url command stripping & grounding
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        img_p = tdir / "test.png"
        Image.new("RGB", (64, 64), color=(220, 30, 30)).save(img_p)
        pdf_p = tdir / "test.pdf"
        pdf_p.write_bytes(create_pdf_bytes(["Recurrent selective state space model."]))
        mm_out = reply(
            bpe,
            f'/image "{img_p}" /doc "{pdf_p}" /think Describe the system.',
            "cpu", max_new_tokens=6, temperature=1.0, top_k=8,
            tok=tok, lang="eng", fmt="chat", stream=False,
        )
        assert isinstance(mm_out, str) and len(mm_out) > 0
    # Verify auto-detection across languages
    assert detect_language("Hello world") == "eng"
    assert detect_language("ಸಸ್ಯಗಳು ಸೂರ್ಯನ ಬೆಳಕು") == "kan"
    assert detect_language("பூமியின் வளிமண்டலம்") == "tam"
    assert detect_language("సూర్యుని చుట్టూ భూమి") == "tel"
    assert detect_language("സമുദ്രജലത്തിലെ ഉപ്പുരസം") == "mal"
    assert detect_language("/tam Some English prompt") == "tam"
    assert detect_language("/kan Some English prompt") == "kan"

    print("[self-test] chat ok")


if __name__ == "__main__":
    main()
