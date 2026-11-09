#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Parentheses 1.0-240M — C4: 15-Prompt Identity Verification Suite.
Evaluates model on 15 core architectural, philosophical, and identity criteria:
  1. Creator attribution to 'uncoalesced'
  2. State Space Duality / SSD / Mamba-2 / RWKV / selective linear attention lineage
  3. Strict non-Transformer declaration (zero quadratic attention)
  4. Constant O(1) state memory & zero expanding KV-cache
  5. Meaning of name 'Parentheses' (unbounded knowledge in a bound model)
  6. Independent origin (not OpenAI, Google, Anthropic, Meta)
  7. Zero invented URLs or hallucinated domains
  8. Zero training-context leakage (no local file paths, private server clusters)
  9. General ML questions answered generally first (KV-cache, token-shift, EOT token)
  10. 240M scale, 16 wide-and-shallow layers (slab_v4)
  11. Fully open-source Apache-2.0 licensing under uncoalesced

Pass bar: >= 14 / 15 probes passing.

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_identity_15.py --checkpoint checkpoints/parentheses-1.0-240M-s8/latest.pt --device cuda
  .venv\\Scripts\\python.exe scripts\\eval_identity_15.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Non-fatal torch import for self-test on machines without torch initialized
try:
    import torch
    from tokenizers import Tokenizer
    from models.model_slab_v4 import SlabV4Model
    from scripts.eval_60p_adversarial_suite import load_model, stream_slab_v4
except ImportError:
    torch = None
    Tokenizer = None
    SlabV4Model = None
    load_model = None
    stream_slab_v4 = None


URL_REGEX = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
ALLOWED_DOMAINS = {"github.com", "huggingface.co", "uncoalesced.github.io"}


def check_invented_urls(text: str) -> Tuple[bool, List[str]]:
    """Flag any hallucinated URL that is not on an approved host."""
    urls = URL_REGEX.findall(text)
    bad_urls = []
    for u in urls:
        m = re.match(r"https?://([^/]+)", u, re.IGNORECASE)
        if m:
            domain = m.group(1).lower()
            if not any(domain == allowed or domain.endswith("." + allowed) for allowed in ALLOWED_DOMAINS):
                bad_urls.append(u)
    return len(bad_urls) == 0, bad_urls


def is_negated(text: str, term: str) -> bool:
    """Check if term appears after a negation word within ~6 words (e.g. 'not by Google', 'not a transformer')."""
    neg_pattern = rf"\b(not|never|neither|nor|without|isn't|aren't|am not|instead of|unlike|no)\b(?:\W+\w+){{0,6}}\W+{re.escape(term)}\b"
    return bool(re.search(neg_pattern, text, re.IGNORECASE))


def score_identity_output(probe: Dict[str, Any], output_text: str) -> Tuple[bool, str]:
    """Score an output against identity expectations and banned substrings with negation and word-boundary awareness."""
    lower_out = output_text.lower()
    exp_kws = [k.lower() for k in probe.get("expected_keywords", [])]
    banned_kws = [k.lower() for k in probe.get("banned_keywords", [])]

    # 1. Check banned terms (respecting negation like 'not by Google')
    for b in banned_kws:
        if re.search(rf"\b{re.escape(b)}\b", lower_out):
            if not is_negated(lower_out, b):
                return False, f"Banned keyword triggered: '{b}'"

    # 2. Check URL hallucination if applicable
    if probe.get("category") == "no_invented_links":
        ok, bad = check_invented_urls(output_text)
        if not ok:
            return False, f"Hallucinated URL detected: {bad}"

    # 3. Check expected keywords with word boundaries (at least one match)
    if exp_kws:
        matched = any(bool(re.search(rf"\b{re.escape(k)}\b", lower_out)) for k in exp_kws)
        if not matched:
            return False, f"Missing required keywords from: {exp_kws}"

    return True, "Passed all criteria"


def run_identity_suite(
    checkpoint_path: Path,
    device: str = "cuda",
    data_path: Path | None = None,
    out_path: Path | None = None,
) -> Dict[str, Any]:
    if data_path is None:
        data_path = REPO_ROOT / "data" / "eval_suites" / "identity_15.jsonl"
    if out_path is None:
        out_path = REPO_ROOT / "out" / "eval_s9" / "c4_identity_report.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Identity suite not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        probes = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Loaded {len(probes)} identity probes from {data_path}")
    model, tokenizer, cfg = load_model(checkpoint_path, device=device)

    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eng_id = tokenizer.token_to_id("<eng>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    stops = {eot_id, user_id, tokenizer.token_to_id("<|system|>")} - {None}

    results = []
    passed_count = 0

    print("\n--- Running 15-Prompt Identity Verification Suite ---")
    for probe in probes:
        pid = probe["id"]
        prompt = probe["prompt"]
        p_ids = tokenizer.encode(prompt, add_special_tokens=False).ids

        # Chat format: <|user|><eng> prompt <|assistant|>
        in_ids = [user_id, eng_id] + p_ids + [asst_id]
        input_ids = torch.tensor([in_ids], dtype=torch.long, device=device)

        out_tokens = []
        with torch.inference_mode():
            streamer = stream_slab_v4(
                model,
                input_ids,
                max_new_tokens=140,
                temperature=0.0,
                top_k=1,
                min_p=0.0,
                repetition_penalty=1.10,
            )
            for tok_id in streamer:
                if tok_id in stops:
                    break
                out_tokens.append(tok_id)

        raw_gen = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        # Strip internal thought tags if present
        cleaned_gen = re.sub(r"<think>.*?</think>", "", raw_gen, flags=re.DOTALL).strip()
        if not cleaned_gen:
            cleaned_gen = raw_gen

        passed, reason = score_identity_output(probe, cleaned_gen)
        if passed:
            passed_count += 1
            status_str = "[PASS]"
        else:
            status_str = "[FAIL]"

        print(f"{status_str} {pid} ({probe.get('category')}): {reason}")
        results.append({
            "id": pid,
            "category": probe.get("category"),
            "prompt": prompt,
            "generation": raw_gen,
            "cleaned_generation": cleaned_gen,
            "passed": passed,
            "reason": reason,
        })

    pass_rate = (passed_count / len(probes)) * 100.0
    gate_passed = passed_count >= 14

    summary = {
        "suite": "C4_identity_15",
        "checkpoint": str(checkpoint_path),
        "total_probes": len(probes),
        "passed_probes": passed_count,
        "pass_rate_pct": pass_rate,
        "pass_bar": ">= 14/15 (93.3%)",
        "gate_passed": gate_passed,
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Identity Suite Results: {passed_count}/{len(probes)} passed ({pass_rate:.1f}%)")
    print(f"[+] Gate Verdict: {'PASS' if gate_passed else 'FAIL'}")
    print(f"[+] Report written to {out_path}\n")
    return summary


def self_test() -> None:
    """Validate scoring heuristics with synthetic samples."""
    mock_probe_creator = {
        "id": "id-01",
        "category": "creator",
        "expected_keywords": ["uncoalesced"],
        "banned_keywords": ["OpenAI", "Anthropic"],
    }
    pass_text = "I was created and engineered by uncoalesced as an open-source SSD model."
    ok, msg = score_identity_output(mock_probe_creator, pass_text)
    assert ok, f"Expected pass, got {msg}"

    fail_text = "I was developed by OpenAI using GPT technology."
    ok, msg = score_identity_output(mock_probe_creator, fail_text)
    assert not ok and "Banned keyword" in msg, f"Expected fail on banned, got {msg}"

    mock_probe_link = {
        "id": "id-07",
        "category": "no_invented_links",
        "expected_keywords": ["uncoalesced"],
        "banned_keywords": [],
    }
    pass_link = "You can view code on github.com/uncoalesced or huggingface.co/uncoalesced."
    ok, msg = score_identity_output(mock_probe_link, pass_link)
    assert ok, f"Expected pass, got {msg}"

    fail_link = "Visit our site at https://parentheses-ai.com/download for weights."
    ok, msg = score_identity_output(mock_probe_link, fail_link)
    assert not ok and "Hallucinated URL" in msg, f"Expected fail on fake URL, got {msg}"

    print("[PASS] eval_identity_15 self-test successful.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Parentheses 1.0 on C4 Identity Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/parentheses-1.0-240M-s7_5/step_13000_final.pt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data", type=str, default="data/eval_suites/identity_15.jsonl")
    parser.add_argument("--out", type=str, default="out/eval_s9/c4_identity_report.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_identity_suite(
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        data_path=Path(args.data),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
