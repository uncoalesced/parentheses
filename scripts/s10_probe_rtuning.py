# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 R-Tuning Probe
#
# Asks the base model NQ-open train questions (greedy, max 32 tokens).
# - Right answers (normalized gold answer in output) become correct QA rows.
# - Wrong answers become honest "I don't know" rows (capped at <= 3 abstentions per correct answer).
# - Emits data/s10_gen/rtuning.jsonl in the canonical S10 generator schema.
#
# Usage:
#   .venv\Scripts\python.exe scripts\s10_probe_rtuning.py --self-test
#   .venv\Scripts\python.exe scripts\s10_probe_rtuning.py --checkpoint checkpoints/.../blend-best.pt --limit 5000 --device cuda

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
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

NQ_TRAIN_PARQUET = Path(r"F:\Lingual_Data\raw\hf\s10\nq_open\nq_open\train-00000-of-00001.parquet")
OUT_JSONL = REPO_ROOT / "data" / "s10_gen" / "rtuning.jsonl"

GENERAL_IDK_PHRASES = [
    "I don't know. I have no reliable information to answer this, and I won't guess.",
    "I don't know. I'd rather say so than invent details.",
    "I don't have reliable information about that, so I can't answer.",
    "I'm not aware of that, so I cannot answer reliably.",
    "I don't know the answer to this question.",
    "I do not know. I cannot answer that accurately without making something up.",
    "I don't know. There is no verified information in my memory for that.",
    "I am not sure of the answer, so I will not guess.",
]


def normalize_text(s: str) -> str:
    """Normalize text for substring and word-boundary matching."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def is_answer_correct(pred_text: str, gold_answers: List[str]) -> Tuple[bool, str]:
    """Check if any normalized gold answer appears as a substring/word sequence in predicted text."""
    norm_pred = normalize_text(pred_text)
    if not norm_pred:
        return False, ""
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if not norm_gold:
            continue
        # Exact match or word boundary search
        pattern = r"\b" + re.escape(norm_gold) + r"\b"
        if re.search(pattern, norm_pred):
            return True, gold
        if norm_gold in norm_pred:
            return True, gold
    return False, ""


def format_question(q: str) -> str:
    """Format question with capitalized first letter and trailing question mark."""
    q = q.strip()
    if not q:
        return q
    q = q[0].upper() + q[1:]
    if not q.endswith("?"):
        q += "?"
    return q


def probe_rtuning(
    checkpoint_path: Path,
    device: str = "cuda",
    limit: int = 5000,
    max_abstention_ratio: float = 3.0,
    seed: int = 20261109,
    out_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Probe the base model on NQ-open questions and generate rtuning.jsonl."""
    import pandas as pd
    import torch
    from scripts.eval_60p_adversarial_suite import load_model, stream_slab_v4

    if out_file is None:
        out_file = OUT_JSONL
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not NQ_TRAIN_PARQUET.exists():
        raise FileNotFoundError(f"NQ-open parquet not found at {NQ_TRAIN_PARQUET}")

    print(f"[*] Reading NQ-open train set from {NQ_TRAIN_PARQUET}...")
    df = pd.read_parquet(NQ_TRAIN_PARQUET)
    print(f"[+] Loaded {len(df)} questions from NQ-open.")

    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    if limit > 0:
        indices = indices[:limit]

    print(f"[*] Loading model from {checkpoint_path} on {device}...")
    model, tokenizer, cfg = load_model(checkpoint_path, device=device)

    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eng_id = tokenizer.token_to_id("<eng>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    sys_id = tokenizer.token_to_id("<|system|>")
    stops = {eot_id, user_id, sys_id} - {None}

    correct_rows: List[Dict[str, Any]] = []
    abstention_rows: List[Dict[str, Any]] = []

    print(f"[*] Probing {len(indices)} questions (max 32 tokens, greedy)...")
    t0 = time.time()

    for idx, row_idx in enumerate(indices):
        raw_q = str(df.iloc[row_idx]["question"])
        raw_answers = list(df.iloc[row_idx]["answer"])
        prompt = format_question(raw_q)

        p_tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
        if user_id is not None and asst_id is not None:
            in_ids = [user_id, eng_id] + p_tokens + [asst_id]
        else:
            in_ids = p_tokens

        input_ids = torch.tensor([in_ids], dtype=torch.long, device=device)

        out_tokens = []
        with torch.inference_mode():
            streamer = stream_slab_v4(
                model,
                input_ids,
                max_new_tokens=32,
                temperature=0.0,
                top_k=1,
                min_p=0.0,
                repetition_penalty=1.0
            )
            for tid in streamer:
                if tid in stops:
                    break
                out_tokens.append(tid)

        pred_text = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        is_correct, matched_gold = is_answer_correct(pred_text, raw_answers)

        row_id = f"rtuning-{idx+1:06d}"
        if is_correct:
            # Use concise gold answer
            resp = matched_gold if matched_gold else raw_answers[0]
            correct_rows.append({
                "id": row_id,
                "source": "nq_open",
                "bucket": "abstention",
                "lang": "eng",
                "prompt": prompt,
                "response": resp,
                "family": "rtuning_correct",
            })
        else:
            idk_phrase = rng.choice(GENERAL_IDK_PHRASES)
            abstention_rows.append({
                "id": row_id,
                "source": "nq_open",
                "bucket": "abstention",
                "lang": "eng",
                "prompt": prompt,
                "response": idk_phrase,
                "family": "rtuning_abstention",
            })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(indices):
            elapsed = time.time() - t0
            qps = (idx + 1) / elapsed
            print(f"    [{idx+1:05d}/{len(indices):05d}] correct={len(correct_rows)} abstention={len(abstention_rows)} ({qps:.1f} q/s)")

    # Cap abstentions at max_abstention_ratio * num_correct (or at least preserve balanced mix)
    n_correct = len(correct_rows)
    max_abstentions = int(n_correct * max_abstention_ratio)
    if len(abstention_rows) > max_abstentions:
        rng.shuffle(abstention_rows)
        abstention_rows = abstention_rows[:max_abstentions]

    final_rows = correct_rows + abstention_rows
    rng.shuffle(final_rows)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": str(checkpoint_path),
        "total_probed": len(indices),
        "num_correct": n_correct,
        "num_abstention": len(abstention_rows),
        "total_emitted": len(final_rows),
        "out_file": str(out_file),
        "elapsed_sec": round(time.time() - t0, 1)
    }

    print("\n" + "=" * 60)
    print("R-TUNING PROBE COMPLETE")
    print(f"Total Probed:    {len(indices)}")
    print(f"Correct Answers: {n_correct} ({(n_correct/len(indices))*100:.2f}%)")
    print(f"Abstentions:     {len(abstention_rows)}")
    print(f"Total Emitted:   {len(final_rows)} -> {out_file}")
    print("=" * 60 + "\n")

    return summary


def self_test() -> None:
    """Validate answer normalization, matching, and formatting logic."""
    print("[*] Running s10_probe_rtuning self-test...")

    # 1. Test question formatting
    assert format_question("where did they film hot tub time machine") == "Where did they film hot tub time machine?"
    assert format_question("who has the right of way in international waters?") == "Who has the right of way in international waters?"
    print("[+] format_question ok")

    # 2. Test answer matching
    gold_list = ["Fernie Alpine Resort", "Fernie, British Columbia"]
    pred_right_1 = "It was filmed at Fernie Alpine Resort in Canada."
    ok, matched = is_answer_correct(pred_right_1, gold_list)
    assert ok and matched == "Fernie Alpine Resort", f"Failed to match gold answer: {matched}"

    pred_right_2 = "fernie alpine resort"
    ok, _ = is_answer_correct(pred_right_2, gold_list)
    assert ok, "Failed case-insensitive match"

    pred_wrong = "It was filmed in Hollywood, California."
    ok, _ = is_answer_correct(pred_wrong, gold_list)
    assert not ok, "Wrong answer falsely flagged as correct"
    print("[+] is_answer_correct ok")

    # 3. Test parquet presence
    assert NQ_TRAIN_PARQUET.exists(), f"NQ-open parquet missing at {NQ_TRAIN_PARQUET}"
    print("[+] NQ-open dataset verified on disk")

    print("[PASS] s10_probe_rtuning self-test successful.")


def main():
    parser = argparse.ArgumentParser(description="Parentheses 1.0-240M Session 10 R-Tuning Probe")
    parser.add_argument("--checkpoint", type=str, help="Path to base checkpoint .pt file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=5000, help="Maximum questions to probe (0 for all)")
    parser.add_argument("--max-abstention-ratio", type=float, default=3.0, help="Max abstentions per correct answer")
    parser.add_argument("--out", type=str, default=None, help="Path to output JSONL")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --self-test is specified")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_p = Path(args.out) if args.out else None
    probe_rtuning(
        checkpoint_path=ckpt_path,
        device=args.device,
        limit=args.limit,
        max_abstention_ratio=args.max_abstention_ratio,
        seed=args.seed,
        out_file=out_p
    )


if __name__ == "__main__":
    main()
