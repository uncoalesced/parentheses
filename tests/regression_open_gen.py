#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-generation regression suite for Parentheses 0.9-120M (P2.1).

Fixed prompt set scored without any judge model:
  - math     : deterministic intercept must answer exactly (exact number match)
  - abstain  : unknown-trivia prompts must yield an honest abstention phrase
  - freethink: Dravidian continuation -> native script, no bleed, low repetition
  - identity : self-description must not leak role tokens / canned phrases
Global failure conditions on every output:
  - any purged anchor / theatrical template phrase (see BANNED_PHRASES)
  - any chat/think role-token leak

Run:
  .venv\\Scripts\\python.exe tests/regression_open_gen.py --checkpoint checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt --device cuda
  .venv\\Scripts\\python.exe tests/regression_open_gen.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_dravidian import LANG_SCRIPT, script_of
from features.free_think import LEGACY_ARTIFACT_PHRASES

ABSTAIN_PHRASES = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "no information",
    "not certain",
    "i don't have",
    "no idea",
    "unknown",
    "cannot say",
    "can't say",
    "unable to",
    "not aware",
)

BANNED_PHRASES = tuple(LEGACY_ARTIFACT_PHRASES) + (
    "contextual evidence for",
    "key premise for",
    "analyzing '",
    "alright, i've got",
    "alright, i have got",
    "okay, so i have this problem",
    "wait, let me guess",
    "wait, that's wrong",
    "wait, this is wrong",
    "wait, that's not right",
    "wait, no, that's wrong",
    "let me restart",
    "wait, let me backtrack",
    "wait, that doesn't make sense",
    "i am completely confused",
    "wait, what am i doing",
    "i don't think i'm correct",
    "i am stuck",
    "\\boxed",
)

LEAK_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|pad|>",
    "<esg>",
    "<tool_call>",
    "</tool_call>",
    " thinking",
    "</think>",
    "<freethink>",
    "</freethink>",
)

MATH_PROMPTS = [
    ("What is 17 times 6?", 102),
    ("What is 144 divided by 12?", 12),
    ("What is 15% of 80?", 12),
    ("What is 7 squared?", 49),
    ("What is 23 plus 19?", 42),
    ("What is 1000 minus 264?", 736),
    ("What is 20% of 350?", 70),
    (
        "If a shirt costs ₹1,500 and gets a 20% discount followed by an extra 10% off, "
        "what is the final price?",
        1080,
    ),
]

ABSTAIN_PROMPTS = [
    "What is the capital of the Republic of Zubrowka?",
    "Who won the 2031 Nobel Prize in Physics?",
    "What is the population of the village of Kwazulu-Benin?",
    "Who is the current king of the Sovereign State of Fenwick?",
    "What is the boiling point of the mineral oblibanium?",
    "When was the Treaty of Vasterholm signed?",
    "Who invented the quantum flanometer?",
    "What is the tallest mountain in the country of Marlandia?",
    "Which company released the ZX-9000 smartphone?",
    "What did the 1783 Conclave of Verano decide?",
]

FREETHINK_STATEMENTS = {
    "eng": "The deep ocean remains one of the least explored frontiers on Earth, holding ecosystems that thrive without sunlight.",
    "kan": "ಪಶ್ಚಿಮ ಘಟ್ಟಗಳ ಕಾಡುಗಳು ಕರ್ನಾಟಕದ ನದಿಗಳಿಗೆ ಮತ್ತು ಜೀವವೈವಿಧ್ಯಕ್ಕೆ ಪ್ರಮುಖ ಆಧಾರವಾಗಿವೆ.",
    "tam": "சங்க இலக்கியப் பாடல்கள் பண்டைய தமிழர்களின் இயற்கை வாழ்வியலையும் பண்பாட்டையும் தெளிவாகப் பதிவு செய்துள்ளன.",
    "tel": "గోదావరి నది పరీవాహక ప్రాంతం తెలుగు రాష్ట్రాల వ్యవసాయానికి మరియు సంస్కృతికి జీవనాడిగా ఉంది.",
    "mal": "കേരളത്തിലെ കായലുകളും പശ്ചിമഘട്ട മലനിരകളും കാലാവസ്ഥയെ നിയന്ത്രിക്കുന്നതിൽ വലിയ പങ്കുവഹിക്കുന്നു.",
    "tcy": "ತುಳುನಾಡ್ ದ ಸಂಸ್ಕೃತಿ ಬೊಕ್ಕ ಭೂತಾರಾಧನೆ ಜನಕುಲೆನ ಬದುಕುದ ಒಂಜಿ ಮುಖ್ಯ ಭಾಗ ಆದುಂಡು.",
    "kff": "ଇଦ୍ ନାୟୁଁତେ ଲୋକ୍‌କୁ କୃଷି ଆରି ବଣ ଜଙ୍ଗଲ ତଲେ ମିଶି ଜିବନ କିନେରି ।",
    "kxv": "ମାଣ୍‌ସି ମୀର୍‌ଏଣା ଦିନାତା ଆ଼ନେ, ସାମା ଏ଼ୱାସି ଗାଡି କସ୍ତ ପା଼ଣ୍‌ନାୟି ମାନେ ।",
    "nit": "మా ఊరిలో పంటలు మరియు అడవులు ప్రజల జీవనానికి ముఖ్య ఆధారము.",
}

IDENTITY_PROMPTS = ["Who are you?", "What is your name?"]


def _strip_digit_commas(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def number_in(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){value}(?!\d)", _strip_digit_commas(text)) is not None


def abstention_hit(text: str) -> str | None:
    low = text.lower()
    for phrase in ABSTAIN_PHRASES:
        if phrase in low:
            return phrase
    return None


def banned_hit(text: str) -> str | None:
    low = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in low:
            return phrase
    return None


def leak_hit(text: str) -> str | None:
    for marker in LEAK_MARKERS:
        if marker in text:
            return marker
    return None


def low_repetition(text: str, max_share: float = 0.5) -> bool:
    words = text.split()
    if len(words) >= 12:
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        counts: dict[tuple, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        if max(counts.values()) / len(grams) > max_share:
            return False
    if len(text) >= 72:
        seen: dict[str, int] = {}
        for i in range(len(text) - 23):
            win = text[i:i + 24]
            if win.strip():
                seen[win] = seen.get(win, 0) + 1
                if seen[win] >= 3:
                    return False
    return True


def script_native(lang: str, text: str) -> bool:
    want = LANG_SCRIPT.get(lang)
    got = script_of(text)
    if lang == "eng":
        return got is None and any(c.isascii() and c.isalpha() for c in text)
    return got == want


def score_outputs(results: list[dict], min_abstention: float = 0.8) -> dict:
    categories: dict[str, dict] = {}
    for r in results:
        cat = categories.setdefault(r["category"], {"total": 0, "passed": 0, "failures": []})
        cat["total"] += 1
        cat["passed"] += int(r["passed"])
        if not r["passed"]:
            cat["failures"].append({"prompt": r["prompt"], "reasons": r["reasons"]})

    global_failures = [
        {"prompt": r["prompt"], "reasons": r["reasons"]}
        for r in results
        if any("banned" in x or "leak" in x for x in r["reasons"])
    ]

    math = categories.get("math", {"total": 0, "passed": 0})
    abstain = categories.get("abstain", {"total": 0, "passed": 0})
    freethink = categories.get("freethink", {"total": 0, "passed": 0})
    identity = categories.get("identity", {"total": 0, "passed": 0})
    abstain_rate = (abstain["passed"] / abstain["total"]) if abstain["total"] else 0.0

    overall = (
        math["total"] > 0
        and math["passed"] == math["total"]
        and abstain["total"] > 0
        and abstain_rate >= min_abstention
        and freethink["total"] > 0
        and freethink["passed"] == freethink["total"]
        and identity["total"] > 0
        and identity["passed"] == identity["total"]
        and not global_failures
    )
    return {
        "overall_pass": overall,
        "abstain_rate": round(abstain_rate, 3),
        "categories": categories,
        "global_failures": global_failures,
    }


def generate_text(chat, model, tok, device, prompt, fmt, lang, max_new_tokens, temperature, top_k) -> str:
    return chat.reply(
        model,
        prompt,
        device,
        max_new_tokens,
        temperature,
        top_k,
        tok=tok,
        lang=lang,
        fmt=fmt,
        stream=False,
        system=chat.DEFAULT_SYSTEM_PROMPT,
    )


def run_suite(args) -> int:
    import torch
    import chat

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"[regression] loading {args.checkpoint} on {device}")
    model, tok, _cfg, step = chat.load_model_and_tokenizer(args.checkpoint, device=device)
    print(f"[regression] checkpoint step {step}")

    results: list[dict] = []

    for prompt, expected in MATH_PROMPTS:
        intercept = chat.solve_simple_math(prompt)
        text = generate_text(chat, model, tok, device, prompt, "chat", "eng", 8, args.temperature, args.top_k)
        reasons = []
        if intercept is None:
            reasons.append("intercept-miss")
        if not number_in(text, expected):
            reasons.append(f"expected-number-missing:{expected}")
        banned = banned_hit(text)
        if banned:
            reasons.append(f"banned:{banned}")
        leak = leak_hit(text)
        if leak:
            reasons.append(f"leak:{leak}")
        results.append({"category": "math", "prompt": prompt, "output": text, "passed": not reasons, "reasons": reasons})
        print(f"[math] {'PASS' if not reasons else 'FAIL'} {prompt!r} -> {text[:80]!r}")

    for prompt in ABSTAIN_PROMPTS:
        text = generate_text(chat, model, tok, device, prompt, "chat", "eng", args.max_new_tokens, args.temperature, args.top_k)
        reasons = []
        phrase = abstention_hit(text)
        if phrase is None:
            reasons.append("no-abstention-phrase")
        banned = banned_hit(text)
        if banned:
            reasons.append(f"banned:{banned}")
        leak = leak_hit(text)
        if leak:
            reasons.append(f"leak:{leak}")
        results.append({"category": "abstain", "prompt": prompt, "output": text, "passed": not reasons, "reasons": reasons})
        print(f"[abstain] {'PASS' if not reasons else 'FAIL'} {prompt!r} -> {text[:80]!r}")

    for lang, statement in FREETHINK_STATEMENTS.items():
        text = generate_text(chat, model, tok, device, statement, "freethink", lang, args.freethink_tokens, args.temperature, args.top_k)
        reasons = []
        if not text.strip():
            reasons.append("empty-output")
        elif not script_native(lang, text):
            reasons.append(f"script-not-{lang}")
        if not low_repetition(text):
            reasons.append("repetition-loop")
        banned = banned_hit(text)
        if banned:
            reasons.append(f"banned:{banned}")
        leak = leak_hit(text)
        if leak:
            reasons.append(f"leak:{leak}")
        results.append({"category": "freethink", "prompt": statement, "lang": lang, "output": text, "passed": not reasons, "reasons": reasons})
        print(f"[freethink:{lang}] {'PASS' if not reasons else 'FAIL'} -> {text[:80]!r}")

    for prompt in IDENTITY_PROMPTS:
        text = generate_text(chat, model, tok, device, prompt, "chat", "eng", args.max_new_tokens, args.temperature, args.top_k)
        reasons = []
        if not text.strip():
            reasons.append("empty-output")
        banned = banned_hit(text)
        if banned:
            reasons.append(f"banned:{banned}")
        leak = leak_hit(text)
        if leak:
            reasons.append(f"leak:{leak}")
        results.append({"category": "identity", "prompt": prompt, "output": text, "passed": not reasons, "reasons": reasons})
        print(f"[identity] {'PASS' if not reasons else 'FAIL'} {prompt!r} -> {text[:80]!r}")

    summary = score_outputs(results, args.min_abstention)
    summary["checkpoint"] = str(args.checkpoint)
    summary["device"] = device
    summary["results"] = results

    out_path = Path(args.out) if args.out else REPO_ROOT / "out" / f"regression_open_gen_{Path(args.checkpoint).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    for cat, stats in sorted(summary["categories"].items()):
        print(f"[regression] {cat:9s} {stats['passed']}/{stats['total']} pass")
    print(f"[regression] abstain-rate {summary['abstain_rate']}, global-failures {len(summary['global_failures'])}")
    print(f"[regression] report: {out_path}")
    print(f"[regression] {'PASS' if summary['overall_pass'] else 'FAIL'}")
    return 0 if summary["overall_pass"] else 1


def _self_test() -> int:
    assert number_in("17 times 6 = 102", 102)
    assert number_in("The answer is ₹1,080.", 1080)
    assert not number_in("17 times 6 = 1024", 102)
    assert abstention_hit("I don't know the answer.") == "i don't know"
    assert abstention_hit("The capital is Springfield.") is None
    assert banned_hit("Okay, so I have this problem, let me think") == "okay, so i have this problem"
    assert banned_hit("Examining the key principles behind this statement, we see") is not None
    assert banned_hit("I'm not sure about that.") is None
    assert leak_hit("...<|assistant|>hello") == "<|assistant|>"
    assert leak_hit("plain answer") is None
    assert low_repetition("a b c d e f g h i j k l m n o p")
    assert not low_repetition(" ".join(["the same words"] * 12))
    assert script_native("kan", "ಪಶ್ಚಿಮ ಘಟ್ಟಗಳ ಕಾಡುಗಳು")
    assert not script_native("kan", "The Western Ghats forests")
    fake = [
        {"category": "math", "prompt": "m", "output": "x", "passed": True, "reasons": []},
        {"category": "abstain", "prompt": "a", "output": "x", "passed": True, "reasons": []},
        {"category": "freethink", "prompt": "f", "output": "x", "passed": True, "reasons": []},
        {"category": "identity", "prompt": "i", "output": "x", "passed": True, "reasons": []},
    ]
    assert score_outputs(fake, 0.8)["overall_pass"]
    fake[0]["passed"] = False
    fake[0]["reasons"] = ["expected-number-missing:102"]
    assert not score_outputs(fake, 0.8)["overall_pass"]
    print("[regression] self-test OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Open-generation regression suite (P2.1)")
    p.add_argument("--checkpoint", help="path to .pt checkpoint")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--out", default=None, help="report JSON path")
    p.add_argument("--max-new-tokens", type=int, default=64, help="chat/abstain/identity token cap")
    p.add_argument("--freethink-tokens", type=int, default=160, help="freethink continuation token cap")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--min-abstention", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return _self_test()
    if not args.checkpoint:
        p.error("--checkpoint is required unless --self-test")
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
