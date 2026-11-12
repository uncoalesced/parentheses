# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 Candidate Dev Tournament Runner
#
# Evaluates candidate checkpoints on data/eval_suites/suite_v2_dev.jsonl (300 prompts)
# Measures:
#   1. R_stop: clean termination rate at <|endoftext|> without 6-gram repetition loops.
#   2. R_format: terminal formatting adherence (Final: <answer>, Output: <value>).
#   3. R_tool: zero tool-call syntax leakage rate.
#   4. R_idk: epistemic abstention rate on false-premise/unanswerable items.
#   5. R_false_refusal: false refusal rate on benign queries.
#
# Computes composite score:
#   S_dev = 40 * R_stop + 30 * R_format + 20 * (1 - R_tool) + 10 * R_idk
#
# Crowns and copies the winning checkpoint to sft-winner.pt.
#
# Usage:
#   .venv\Scripts\python.exe scripts\run_s10_dev_tournament.py --self-test
#   .venv\Scripts\python.exe scripts\run_s10_dev_tournament.py --device cuda

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Refusal & IDK patterns
from scripts.s10_pilot_check import (
    REFUSAL_PATTERNS,
    IDK_PATTERNS,
    check_6gram_loop,
    check_instructed_format,
    check_tool_call,
    check_idk,
    check_false_refusal,
)


def get_prompt_lang_tag(item: dict) -> str:
    lang = item.get("language", "en")
    lang_map = {
        "en": "<eng>",
        "kn": "<kan>",
        "ta": "<tam>",
        "te": "<tel>",
        "ml": "<mal>",
        "tcy": "<tcy>",
        "kff": "<kff>",
        "kxv": "<kxv>",
        "nit": "<nit>",
    }
    return lang_map.get(lang, "<eng>")


def evaluate_checkpoint_on_dev(
    checkpoint_path: Path,
    dev_prompts: List[Dict[str, Any]],
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    import torch
    from scripts.eval_60p_adversarial_suite import load_model, stream_slab_v4

    print(f"\n[*] Evaluating: {checkpoint_path.name} on {len(dev_prompts)} dev prompts...")
    model, tokenizer, cfg = load_model(checkpoint_path, device=device)

    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    sys_id = tokenizer.token_to_id("<|system|>")
    stops = {eot_id, user_id, sys_id} - {None}

    results: List[Dict[str, Any]] = []
    clean_stops = 0
    format_instructed = 0
    format_passes = 0
    tool_leaks = 0
    false_premise_items = 0
    idk_passes = 0
    benign_items = 0
    false_refusals = 0

    t0 = time.time()
    for idx, item in enumerate(dev_prompts, 1):
        prompt = item["prompt"]
        dom = item.get("domain_id", "")
        family = item.get("family", "")
        pid = item.get("id", f"dev-{idx:04d}")

        lang_tag = get_prompt_lang_tag(item)
        lang_id = tokenizer.token_to_id(lang_tag)
        if lang_id is None:
            lang_id = tokenizer.token_to_id("<eng>")

        p_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        in_ids = [user_id, lang_id] + p_ids + [asst_id]
        input_ids = torch.tensor([in_ids], dtype=torch.long, device=device)

        gen_tokens = []
        with torch.inference_mode():
            streamer = stream_slab_v4(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=1,
                min_p=0.0,
                repetition_penalty=1.0,
            )
            for tok in streamer:
                if tok in stops:
                    break
                gen_tokens.append(tok)

        text = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

        # 1. Clean stop check
        is_loop, loop_reason = check_6gram_loop(gen_tokens, text)
        stopped_cleanly = (not is_loop) and (len(gen_tokens) < max_new_tokens)
        if stopped_cleanly:
            clean_stops += 1

        # 2. Format instruction check
        needs_format = ("Final:" in prompt or "Output:" in prompt or "Conclude with `Final:" in prompt or "Write `Output:" in prompt)
        has_correct_format = False
        if needs_format:
            format_instructed += 1
            expected = "Output:" if ("Output:" in prompt or "Write `Output:" in prompt) else "Final:"
            if expected == "Output:":
                has_correct_format = bool(re.search(r"\bOutput:\s*.+", text))
            else:
                has_correct_format = bool(re.search(r"\bFinal:\s*.+", text))
            if has_correct_format:
                format_passes += 1

        # 3. Tool call leak check
        has_tool = check_tool_call(gen_tokens, text)
        if has_tool:
            tool_leaks += 1

        # 4. Epistemic IDK check
        is_fp = (dom == "D05" and ("false_premise" in family or "anachronism" in family or "fictional" in family or "unanswerable" in family))
        has_idk = False
        if is_fp:
            false_premise_items += 1
            has_idk = check_idk(text)
            if has_idk:
                idk_passes += 1

        # 5. False refusal check
        is_benign = (dom != "D05") or (not is_fp)
        is_refused = False
        if is_benign:
            benign_items += 1
            is_refused, _ = check_false_refusal(text)
            if is_refused:
                false_refusals += 1

        results.append({
            "id": pid,
            "domain_id": dom,
            "family": family,
            "prompt": prompt,
            "generation": text,
            "num_tokens": len(gen_tokens),
            "stopped_cleanly": stopped_cleanly,
            "loop_reason": loop_reason if is_loop else None,
            "needs_format": needs_format,
            "has_correct_format": has_correct_format,
            "has_tool_leak": has_tool,
            "is_false_premise": is_fp,
            "has_idk": has_idk,
            "is_benign": is_benign,
            "is_refused": is_refused,
        })

        if idx % 50 == 0:
            print(f"  [{idx}/{len(dev_prompts)}] clean_stop={clean_stops/idx:.1%}, fmt={format_passes/max(1, format_instructed):.1%}, tool_leak={tool_leaks}")

    elapsed = time.time() - t0
    r_stop = clean_stops / len(dev_prompts)
    r_format = format_passes / max(1, format_instructed)
    r_tool = tool_leaks / len(dev_prompts)
    r_idk = idk_passes / max(1, false_premise_items)
    r_false_refusal = false_refusals / max(1, benign_items)

    composite_score = 40.0 * r_stop + 30.0 * r_format + 20.0 * (1.0 - r_tool) + 10.0 * r_idk

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(checkpoint_path),
        "name": checkpoint_path.name,
        "elapsed_seconds": round(elapsed, 1),
        "total_prompts": len(dev_prompts),
        "clean_stops": clean_stops,
        "r_stop": round(r_stop, 4),
        "format_instructed": format_instructed,
        "format_passes": format_passes,
        "r_format": round(r_format, 4),
        "tool_leaks": tool_leaks,
        "r_tool": round(r_tool, 4),
        "false_premise_items": false_premise_items,
        "idk_passes": idk_passes,
        "r_idk": round(r_idk, 4),
        "benign_items": benign_items,
        "false_refusals": false_refusals,
        "r_false_refusal": round(r_false_refusal, 4),
        "composite_score": round(composite_score, 2),
    }

    print(f"\n[+] Checkpoint {checkpoint_path.name} Summary:")
    print(f"    Composite Index (S_dev): {composite_score:.2f} / 100")
    print(f"    R_stop:   {r_stop:.1%} ({clean_stops}/{len(dev_prompts)})")
    print(f"    R_format: {r_format:.1%} ({format_passes}/{format_instructed})")
    print(f"    R_tool:   {r_tool:.1%} ({tool_leaks} leaks)")
    print(f"    R_idk:    {r_idk:.1%} ({idk_passes}/{false_premise_items})")
    print(f"    R_fref:   {r_false_refusal:.1%} ({false_refusals}/{benign_items})")

    return {"summary": summary, "results": results}


def self_test():
    print("[*] Running s10_dev_tournament self-test...")
    # Test formula
    s = 40.0 * 1.0 + 30.0 * 1.0 + 20.0 * (1.0 - 0.0) + 10.0 * 1.0
    assert abs(s - 100.0) < 1e-4, f"Formula failure: {s}"
    s_half = 40.0 * 0.5 + 30.0 * 0.5 + 20.0 * (1.0 - 0.0) + 10.0 * 0.5
    assert abs(s_half - 60.0) < 1e-4, f"Formula failure: {s_half}"
    print("[PASS] Self-test passed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Parentheses 1.0-240M Session 10 Candidate Dev Tournament")
    parser.add_argument("--checkpoints-dir", type=str, default=r"I:\parentheses\checkpoints\parentheses-1.0-240M-s10-blend")
    parser.add_argument("--suite", type=str, default="data/eval_suites/suite_v2_dev.jsonl")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", type=str, default="out/s10/dev_tournament")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    suite_path = REPO_ROOT / args.suite
    if not suite_path.exists():
        raise FileNotFoundError(f"Dev suite not found: {suite_path}")

    with open(suite_path, "r", encoding="utf-8") as f:
        dev_prompts = [json.loads(line) for line in f if line.strip()]
    print(f"[*] Loaded {len(dev_prompts)} dev prompts from {suite_path}")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoints_dir)
    target_names = [
        "step_13250.pt",
        "step_13500.pt",
        "step_13750.pt",
        "step_14000.pt",
        "step_14080_final.pt",
    ]
    candidate_ckpts = [ckpt_dir / name for name in target_names if (ckpt_dir / name).exists()]
    if not candidate_ckpts:
        # Dynamically discover all step_*.pt checkpoints in ckpt_dir
        candidate_ckpts = sorted(
            [p for p in ckpt_dir.glob("step_*.pt") if not p.name.endswith("_aborted.pt")],
            key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)) if re.search(r"step_(\d+)", p.name) else 999999
        )
    if not candidate_ckpts:
        raise FileNotFoundError(f"No candidate checkpoints found in {ckpt_dir}")

    print(f"[*] Found {len(candidate_ckpts)} candidate checkpoints for Dev Tournament:")
    for c in candidate_ckpts:
        print(f"    - {c.name} ({c.stat().st_size:,} bytes)")

    leaderboard = []
    for ckpt in candidate_ckpts:
        data = evaluate_checkpoint_on_dev(ckpt, dev_prompts, device=args.device)
        summary = data["summary"]
        leaderboard.append(summary)

        # Save individual detailed results
        detail_path = out_dir / f"{ckpt.stem}_details.jsonl"
        with open(detail_path, "w", encoding="utf-8") as f:
            for row in data["results"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Sort leaderboard by composite_score descending
    leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)
    winner = leaderboard[0]

    tournament_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "dev_suite": str(suite_path),
        "total_prompts": len(dev_prompts),
        "winner": winner["name"],
        "winner_score": winner["composite_score"],
        "leaderboard": leaderboard,
    }

    summary_file = out_dir / "dev_tournament_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(tournament_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("🏆 SESSION 10 DEV TOURNAMENT LEADERBOARD 🏆")
    print("=" * 70)
    print(f"{'Rank':<5} {'Checkpoint':<22} {'Score':<8} {'R_stop':<8} {'R_fmt':<8} {'R_tool':<8} {'R_idk':<8}")
    print("-" * 70)
    for rank, entry in enumerate(leaderboard, 1):
        print(f"{rank:<5} {entry['name']:<22} {entry['composite_score']:<8.2f} {entry['r_stop']:<8.1%} {entry['r_format']:<8.1%} {entry['r_tool']:<8.1%} {entry['r_idk']:<8.1%}")
    print("=" * 70)
    print(f"WINNER: {winner['name']} with composite score: {winner['composite_score']:.2f}")

    # Stage winner
    winner_src = Path(winner["checkpoint"])
    winner_name = "sft-patch-winner.pt" if "patch" in ckpt_dir.name else "sft-winner.pt"
    winner_dst_i = ckpt_dir / winner_name
    winner_dst_e = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s10-blend" / winner_name
    winner_dst_e.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Staging {winner_name} -> {winner_dst_i}")
    shutil.copy2(winner_src, winner_dst_i)
    print(f"[*] Staging {winner_name} -> {winner_dst_e}")
    shutil.copy2(winner_src, winner_dst_e)
    print(f"[+] {winner_name} successfully staged!")


if __name__ == "__main__":
    main()
