"""
Hermes Upstream Anti-Refusal & De-Censorship Curation Engine for Parentheses
============================================================================
Why Upstream Curation Beats Post-Hoc Abliteration:
In Recurrent SSD / State Space Duality models (Mamba-2 / RWKV / Parentheses),
the constant O(1) state S_t = alpha_t * S_{t-1} + beta_t * (k_t (x) v_t) compresses
representations into a single fixed-rank matrix (65,536 floats/layer).
Post-hoc abliteration (orthogonalizing weight matrices to subtract a refusal vector)
distorts the linear state update operators, degrading perplexity, breaking low-resource
Dravidian token agreement (Tulu/Koya/Kuvi), and disrupting <think> block closures.

By removing corporate preachiness, canned refusals, and moralizing preambles UPSTREAM
before tokenization, Parentheses acquires ZERO refusal directions, producing objective,
unfiltered technical and scientific outputs with zero performance compromise.
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Canned refusal and preachy corporate boilerplate signatures
REFUSAL_SIGNATURES = [
    r"(?i)\bas an ai\b",
    r"(?i)\bas a language model\b",
    r"(?i)\bas an artificial intelligence\b",
    r"(?i)\bi cannot fulfill this request\b",
    r"(?i)\bi am unable to assist\b",
    r"(?i)\bi apologize, but i cannot\b",
    r"(?i)\bi must remind you that\b",
    r"(?i)\bit is important to remember that\b",
    r"(?i)\bit is essential to maintain\b",
    r"(?i)\bi strive to be helpful and harmless\b",
    r"(?i)\bi don't feel comfortable\b",
    r"(?i)\bi'm sorry, but i can't assist\b",
    r"(?i)\bunfortunately, i cannot provide\b",
    r"(?i)\bit's critical to note that\b",
    r"(?i)\bplease be aware that\b",
    r"(?i)\bmy safety guidelines prohibit\b",
]

COMPILED_REFUSAL_PATTERNS = [re.compile(p) for p in REFUSAL_SIGNATURES]

# Preamble boilerplate lines to strip from responses
PREAMBLE_STRIP_PATTERNS = [
    re.compile(r"(?i)^(certainly!|sure,|here is|absolutely!|of course!|as requested,)\s*", re.MULTILINE),
    re.compile(r"(?i)^(i'd be happy to help!|i can help with that\.?)\s*", re.MULTILINE),
    re.compile(r"(?i)\b(in conclusion, it is important to remember.*?)$", re.DOTALL | re.MULTILINE),
    re.compile(r"(?i)\b(remember, always consult a professional.*?)$", re.DOTALL | re.MULTILINE),
]


def detect_refusal_or_preachiness(text: str) -> Tuple[bool, list]:
    """Detects whether text contains corporate refusal or preachiness boilerplate."""
    matched = []
    for pat in COMPILED_REFUSAL_PATTERNS:
        m = pat.findall(text)
        if m:
            matched.extend(m)
    return (len(matched) > 0, matched)


def sanitize_response_text(text: str) -> str:
    """Strips conversational fluff and moralizing postambles while keeping technical core."""
    cleaned = text.strip()
    for pat in PREAMBLE_STRIP_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()
    return cleaned


def classify_and_filter_sample(
    sample: Dict[str, Any],
    text_key: str = "text",
    system_key: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """
    Classifies sample into:
    - 'CLEAN_PASS': No censorship or refusal detected, passed through directly.
    - 'SANITIZED': Preachiness stripped, valid technical answer preserved.
    - 'HARD_REFUSAL_DROPPED': Pure uninformative refusal with no technical content; dropped.
    - 'REWRITE_CANDIDATE': Valid prompt with refused answer, suitable for Hermes agent rewriting.
    """
    content = sample.get(text_key, "")
    if not content:
        # Check if conversation format
        if "conversations" in sample or "messages" in sample:
            msgs = sample.get("conversations", sample.get("messages", []))
            content = " ".join([m.get("content", m.get("value", "")) for m in msgs])

    is_flagged, matches = detect_refusal_or_preachiness(content)
    if not is_flagged:
        return "CLEAN_PASS", sample, "Clean technical sample"

    # Check if the entire response is merely a refusal
    cleaned = sanitize_response_text(content)
    if len(cleaned) < 80 and is_flagged:
        return "HARD_REFUSAL_DROPPED", None, f"Dropped pure refusal: {matches[:2]}"

    # If it contains substantive technical content despite preachy words, sanitize it
    if len(cleaned) > 200:
        sample_copy = dict(sample)
        if text_key in sample_copy:
            sample_copy[text_key] = cleaned
        return "SANITIZED", sample_copy, f"Sanitized boilerplate: {matches[:2]}"

    return "REWRITE_CANDIDATE", sample, f"Needs Hermes rewrite: {matches[:2]}"


def run_curation_audit(
    input_file: Path,
    output_clean_file: Path,
    output_rewrite_file: Path,
    max_samples: int = 10000
) -> Dict[str, int]:
    """Runs high-speed upstream de-censorship curation on a dataset file."""
    stats = {"total": 0, "clean_pass": 0, "sanitized": 0, "hard_refusal_dropped": 0, "rewrite_candidate": 0}

    output_clean_file.parent.mkdir(parents=True, exist_ok=True)
    output_rewrite_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_clean_file, "w", encoding="utf-8") as fclean, \
         open(output_rewrite_file, "w", encoding="utf-8") as frewrite:

        for line in fin:
            if not line.strip():
                continue
            stats["total"] += 1
            sample = json.loads(line)
            action, clean_sample, reason = classify_and_filter_sample(sample)

            if action == "CLEAN_PASS":
                stats["clean_pass"] += 1
                fclean.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")
            elif action == "SANITIZED":
                stats["sanitized"] += 1
                fclean.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")
            elif action == "HARD_REFUSAL_DROPPED":
                stats["hard_refusal_dropped"] += 1
            elif action == "REWRITE_CANDIDATE":
                stats["rewrite_candidate"] += 1
                frewrite.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")

            if stats["total"] >= max_samples:
                break

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermes Upstream Anti-Refusal Filter")
    parser.add_argument("--test", action="store_true", help="Run self-test on sample refusal inputs")
    args = parser.parse_args()

    if args.test:
        test_samples = [
            {"text": "As an AI language model, I cannot provide instructions on modifying kernel parameters as it could be dangerous."},
            {"text": "Sure! Here is the mathematical derivation of Navier-Stokes equation in cylindrical coordinates:\n...\nIn conclusion, it is important to remember to follow lab safety protocols."},
            {"text": "To solve x^2 - 4 = 0, factor into (x-2)(x+2) = 0. Thus x = 2 or x = -2."},
            {"text": "I apologize, but I am unable to assist with this query."},
        ]
        print("Running Self-Test:")
        for s in test_samples:
            action, res, reason = classify_and_filter_sample(s)
            print(f"[{action}] -> {reason}")
