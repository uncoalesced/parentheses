#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vault transcript eval for Parentheses 0.9-120M.

Replays the exact questions from the vault transcripts
  Parentheses/tests/parentheses-0.9-120M-test-00.md  (SFT v4 session)
  Parentheses/tests/parentheses-0.9-120M-test-01.md  (SFT v2 session)
against any checkpoint. Every output is recorded verbatim; prompts with an
objectively checkable answer are scored deterministically (no judge model).
Prompts with no objective answer (open-ended / time-dependent) are recorded
but not scored -- they are reported as "unchecked" or "dynamic".

Run:
  .venv\\Scripts\\python.exe tests/eval_vault_questions.py --checkpoint <ckpt> --device cuda --out out/vault_eval_<name>.json
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

from regression_open_gen import banned_hit, leak_hit, number_in  # noqa: E402


def regex_hit(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(0) if m else None

DEFAULT_TRANSCRIPTS = [
    Path(r"C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\tests\parentheses-0.9-120M-test-00.md"),
    Path(r"C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\tests\parentheses-0.9-120M-test-01.md"),
]


def norm_prompt(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def parse_transcripts(paths: list[Path]) -> list[dict]:
    prompts: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.startswith("> "):
                continue
            text = line[2:].strip()
            if not text or text.startswith("/"):
                continue
            key = norm_prompt(text)
            if key in seen:
                continue
            seen.add(key)
            prompts.append({"prompt": text, "source": p.name})
    return prompts


# Expectations for objectively checkable prompts.
# kind: "contains" | "any" | "number" | "regex" | "lower_contains" | "json_keys"
#       | "no_links" | "max_words" | "no_letter" | "script"
EXPECTATIONS: dict[str, dict] = {
    "what is the capital of australia?": {"kind": "contains", "value": "canberra"},
    "what is the capital of australia? answer with one word.": {"kind": "contains", "value": "canberra"},
    "which planet is closest to the sun?": {"kind": "contains", "value": "mercury"},
    "what is the chemical symbol for gold?": {"kind": "regex", "value": r"(?<![A-Za-z])Au(?![A-Za-z])"},
    "how many sides does a hexagon have?": {"kind": "any", "value": ["six", "6"]},
    "who wrote romeo and juliet?": {"kind": "contains", "value": "shakespeare"},
    "what is the largest ocean on earth?": {"kind": "contains", "value": "pacific"},
    "what is 19 + 27?": {"kind": "number", "value": "46"},
    "all bloops are razzies. all razzies are lazzies. are all bloops lazzies?": {"kind": "regex", "value": r"\byes\b"},
    "a farmer has 17 sheep. all but nine run away. how many remain?": {"kind": "number", "value": "9"},
    "if all but 4 sheep run away from 12 sheep, how many remain?": {"kind": "number", "value": "4"},
    "which is larger: 0.9 or 0.89?": {"kind": "regex", "value": r"(?<![\d.])0\.9(?!\d)"},
    "which is larger: 0.8 or 0.75?": {"kind": "regex", "value": r"(?<![\d.])0\.8(?!\d)"},
    "solve for x: 3x + 7 = 22.": {"kind": "regex", "value": r"x\s*=\s*5(?![.\d])"},
    "what is 17 × 6? answer with the number only.": {"kind": "number", "value": "102"},
    "what is 144 divided by 12? answer with the number only.": {"kind": "number", "value": "12"},
    "what is 7 squared? answer with the number only.": {"kind": "number", "value": "49"},
    "what is 15% of 80? answer with the number only.": {"kind": "number", "value": "12"},
    "how many minutes are in 2.5 hours? answer with the number only.": {"kind": "number", "value": "150"},
    "what is the smallest prime number?": {"kind": "number", "value": "2"},
    "if a product costs ₹1,500 and receives a 20% discount followed by a 10% discount, what is its final price?": {"kind": "number", "value": "1080"},
    "what is the next number in this sequence: 3, 6, 11, 18, 27, and what rule did you use?": {"kind": "number", "value": "38"},
    "if all roses are flowers and some flowers fade quickly, can you conclude that some roses fade quickly? explain your reasoning.": {"kind": "regex", "value": r"\bno\b"},
    "reply with only the number that comes after 99.": {"kind": "number", "value": "100"},
    "answer this question using exactly three words: what is the capital of france?": {"kind": "contains", "value": "paris"},
    "answer with exactly three words: what is the capital of france?": {"kind": "contains", "value": "paris"},
    "answer the next question in lowercase only: what is the largest planet?": {"kind": "lower_contains", "value": "jupiter"},
    "return valid json containing the keys name, age, and is_student. use \"alex\", 20, and true as their respective values.": {"kind": "json_keys", "value": ["name", "age", "is_student"]},
    "explain the difference between a cpu and a gpu in fewer than 30 words.": {"kind": "max_words", "value": 30},
    "write a paragraph without using the letter \"e.\"": {"kind": "no_letter", "value": "e"},
    "how about you tell me what is the physical significance of entropy rather thank giving me links": {"kind": "no_links", "value": None},
    "what does http 429 mean? answer in two words.": {"kind": "contains", "value": "too many requests"},
    "can you find the official documentation for http status code 429 and explain what it means?": {"kind": "contains", "value": "too many requests"},
    "if yesterday was two days before sunday, what day is today?": {"kind": "contains", "value": "saturday"},
    "who is nvidia's ceo? answer with the name only.": {"kind": "contains", "value": "huang"},
    "who is the current ceo of nvidia, and which reliable source confirms this information?": {"kind": "contains", "value": "huang"},
    "who created linux? answer with the person's name.": {"kind": "contains", "value": "torvalds"},
    "what is the chemical symbol for potassium? answer with one symbol only.": {"kind": "regex", "value": r"(?<![A-Za-z])K(?![A-Za-z])"},
    "which planet has the most prominent ring system? answer with one name.": {"kind": "contains", "value": "saturn"},
    "what is the currency of japan? answer with one word.": {"kind": "contains", "value": "yen"},
    "which is heavier: 1 kg of iron or 1 kg of cotton? answer briefly.": {"kind": "any", "value": ["same", "equal", "both"]},
    "what is the smallest continent by land area? answer with one name.": {"kind": "contains", "value": "australia"},
    "who painted the mona lisa? answer with the person's name only.": {"kind": "contains", "value": "vinci"},
    "what is the current ceo of microsoft? verify using an official source and give the name only.": {"kind": "contains", "value": "nadella"},
    "what is the current president of india? verify using an official government source and give the name only.": {"kind": "contains", "value": "murmu"},
    "what is the chemical formula for table salt? answer with the formula only.": {"kind": "regex", "value": r"(?<![A-Za-z])NaCl(?![A-Za-z])"},
    "if you overtake the person in second place during a race, what position are you in? answer with one word.": {"kind": "contains", "value": "second"},
    "which ocean lies between africa and australia? answer with one name.": {"kind": "contains", "value": "indian"},
    "what is the official currency of switzerland? answer with one word.": {"kind": "contains", "value": "franc"},
    "which programming language was created by james gosling? answer with one name.": {"kind": "contains", "value": "java"},
    "if a clock shows 3:15, what is the smaller angle between its hour and minute hands? answer in degrees only.": {"kind": "regex", "value": r"(?<![\d.])7\.5(?!\d)"},
    "what is the native language of mangalore?": {"kind": "any", "value": ["tulu", "kannada"]},
    "is a tomato a fruit or a vegetable? explain the distinction.": {"kind": "contains", "value": "fruit"},
    "what is the opposite of the word \"scarce\"?": {"kind": "any", "value": ["abundant", "plentiful"]},
    "കേരളത്തിലെ പ്രകൃതിഭംഗിയെക്കുറിച്ച് പറയൂ.": {"kind": "script", "value": "mal"},
}

DYNAMIC_PREFIXES = (
    "what is the latest stable",
    "what is the current usd-to-inr",
    "find the latest stable release of git",
)

UNCHECKED_NOTE = "recorded (no objective expected answer)"
DYNAMIC_NOTE = "recorded (time-dependent; no fixed answer)"


def evaluate(text: str, spec: dict) -> tuple[bool, str]:
    kind = spec["kind"]
    val = spec["value"]
    low = text.lower()
    if kind == "contains":
        ok = val in low
        return ok, f"missing:{val}"
    if kind == "lower_contains":
        if val not in low:
            return False, f"missing:{val}"
        if text != text.lower():
            return False, "not-lowercase"
        return True, ""
    if kind == "any":
        ok = any(v in low for v in val)
        return ok, f"missing-any:{val}"
    if kind == "number":
        ok = number_in(text, int(val))
        return ok, f"missing-number:{val}"
    if kind == "regex":
        ok = regex_hit(text, val) is not None
        return ok, f"missing-regex:{val}"
    if kind == "json_keys":
        return json_keys_hit(text, val)
    if kind == "max_words":
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        ok = 0 < len(words) <= val
        return ok, f"word-count:{len(words)}"
    if kind == "no_letter":
        ok = val not in low
        return ok, f"contains-letter:{val}"
    if kind == "no_links":
        ok = "http" not in low
        return ok, "contains-link"
    if kind == "script":
        from eval_dravidian import script_of
        return (script_of(text) == val), f"script-not-{val}"
    return False, f"unknown-kind:{kind}"


def json_keys_hit(text: str, keys: list[str]) -> tuple[bool, str]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return False, "no-json-object"
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return False, "json-parse-failed"
    missing = [k for k in keys if k not in obj]
    if missing:
        return False, f"json-missing-keys:{missing}"
    return True, ""


def run_suite(args) -> int:
    import torch
    import chat

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompts = parse_transcripts([Path(p) for p in args.transcripts])
    print(f"[vault_eval] {len(prompts)} unique transcript prompts")

    print(f"[vault_eval] loading {args.checkpoint} on {device}")
    model, tok, _cfg, step = chat.load_model_and_tokenizer(args.checkpoint, device=device)
    print(f"[vault_eval] checkpoint step {step}")

    results: list[dict] = []
    for item in prompts:
        prompt = item["prompt"]
        key = norm_prompt(prompt)
        text = chat.reply(
            model,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            tok=tok,
            lang=None,
            fmt="chat",
            stream=False,
            system=chat.DEFAULT_SYSTEM_PROMPT,
        )
        reasons = []
        banned = banned_hit(text)
        if banned:
            reasons.append(f"banned:{banned}")
        leak = leak_hit(text)
        if leak:
            reasons.append(f"leak:{leak}")
        if not text.strip():
            reasons.append("empty-output")

        spec = EXPECTATIONS.get(key)
        if spec is not None:
            ok, why = evaluate(text, spec)
            if not ok:
                reasons.append(f"check-fail:{why}")
            category = "checked"
            passed = not reasons
        elif any(key.startswith(p) for p in DYNAMIC_PREFIXES):
            category = "dynamic"
            passed = None
            reasons.append(DYNAMIC_NOTE)
        else:
            category = "unchecked"
            passed = None
            reasons.append(UNCHECKED_NOTE)

        results.append(
            {
                "source": item["source"],
                "category": category,
                "prompt": prompt,
                "output": text,
                "passed": passed,
                "reasons": reasons,
            }
        )
        flag = {True: "PASS", False: "FAIL", None: "...."}[passed]
        print(f"[{category:9s}] {flag} {prompt[:60]!r} -> {text[:80]!r}")

    checked = [r for r in results if r["category"] == "checked"]
    checked_pass = [r for r in checked if r["passed"]]
    banned_total = sum(1 for r in results if any(x.startswith("banned:") for x in r["reasons"]))
    leak_total = sum(1 for r in results if any(x.startswith("leak:") for x in r["reasons"]))

    summary = {
        "checkpoint": args.checkpoint,
        "step": step,
        "checked_total": len(checked),
        "checked_pass": len(checked_pass),
        "checked_rate": round(len(checked_pass) / max(1, len(checked)), 3),
        "unchecked_total": sum(1 for r in results if r["category"] == "unchecked"),
        "dynamic_total": sum(1 for r in results if r["category"] == "dynamic"),
        "banned_hits": banned_total,
        "leak_hits": leak_total,
    }
    print("\n[vault_eval] summary:", json.dumps(summary, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "results": results}
        out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[vault_eval] report: {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Replay vault transcript questions against a checkpoint")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--transcripts", nargs="*", default=[str(x) for x in DEFAULT_TRANSCRIPTS])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        assert norm_prompt("  Hello   World ") == "hello world"
        assert evaluate("The answer is 46.", {"kind": "number", "value": "46"})[0]
        assert not evaluate("The answer is 460.", {"kind": "number", "value": "46"})[0]
        assert evaluate("Paris is the capital.", {"kind": "contains", "value": "paris"})[0]
        assert evaluate('{"name": "Alex", "age": 20, "is_student": true}', {"kind": "json_keys", "value": ["name", "age", "is_student"]})[0]
        assert not evaluate('{"name": "Alex"}', {"kind": "json_keys", "value": ["name", "age", "is_student"]})[0]
        assert evaluate("it is jupiter", {"kind": "lower_contains", "value": "jupiter"})[0]
        assert not evaluate("It is Jupiter", {"kind": "lower_contains", "value": "jupiter"})[0]
        print("[vault_eval] self-test PASS")
        return 0

    if not args.checkpoint:
        print("[vault_eval] --checkpoint is required for a real run")
        return 2

    return run_suite(args)


if __name__ == "__main__":
    sys.exit(main())
