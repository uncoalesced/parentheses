#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/topup_s8_pairs.py
Tops up on-policy DPO pairs to >= 800 by sampling 8 candidates (T=0.85) on unpaired prompts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer
from scripts.train_dpo_v2_apache import (
    EOT_TOKEN_ID,
    THINK_OPEN,
    load_slab_v4_checkpoint,
    sample_candidates_recurrent,
    score_on_policy_candidate,
)

OUT_DIR = ROOT / "data" / "dpo_240m_apache"
PAIRS_FILE = OUT_DIR / "on_policy_pairs.jsonl"
PROMPTS_FILE = OUT_DIR / "prompts.jsonl"
TARGET_PAIRS = 805


def main() -> int:
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))
    existing_pairs = [json.loads(l) for l in PAIRS_FILE.open(encoding="utf-8") if l.strip()]
    print(f"[*] Loaded {len(existing_pairs)} existing pairs.")
    if len(existing_pairs) >= TARGET_PAIRS:
        print(f"[+] Already have {len(existing_pairs)} >= {TARGET_PAIRS} pairs. Nothing to do.")
        return 0

    prompts = [json.loads(l) for l in PROMPTS_FILE.open(encoding="utf-8") if l.strip()]
    paired_prompts = set(p["prompt"] for p in existing_pairs)

    # Prioritize math, qa, code, general, identity
    order = {"math": 1, "qa": 2, "code": 3, "general": 4, "identity": 5, "abstention": 6}
    unpaired = [p for p in prompts if p["prompt"] not in paired_prompts]
    unpaired.sort(key=lambda x: order.get(x.get("task_type", ""), 9))
    print(f"[*] Found {len(unpaired)} unpaired prompts. Need {TARGET_PAIRS - len(existing_pairs)} more pairs.")

    ckpt_path = ROOT / "checkpoints" / "parentheses-1.0-240M-s7_5" / "step_13000_final.pt"
    model, cfg = load_slab_v4_checkpoint(ckpt_path, device="cuda")
    model.eval()

    stop_ids = {
        tok.token_to_id("<|user|>"),
        tok.token_to_id("<|assistant|>"),
        tok.token_to_id("<|endoftext|>"),
    }

    current_pairs = len(existing_pairs)
    t_start = time.time()

    with PAIRS_FILE.open("a", encoding="utf-8") as out_f:
        for idx, item in enumerate(unpaired, start=1):
            if current_pairs >= TARGET_PAIRS:
                break

            prompt_text = item["prompt"]
            prompt_ids = tok.encode(prompt_text).ids
            task_type = item.get("task_type", "general")
            lang = item.get("lang", "en")
            gt_num = item.get("ground_truth_num", None)
            req_think = bool(item.get("requires_think", THINK_OPEN in prompt_text))
            req_abstain = bool(item.get("requires_abstention", False))

            all_gen = sample_candidates_recurrent(
                model=model,
                prompt_ids=prompt_ids,
                num_candidates=8,
                max_new_tokens=256,
                temperature=0.85,
                top_p=0.95,
                stop_ids=stop_ids,
                device="cuda",
            )

            candidates: List[Tuple[float, List[int], str]] = []
            for gen_ids in all_gen:
                gen_text = tok.decode([t for t in gen_ids if t != EOT_TOKEN_ID], skip_special_tokens=False)
                sc = score_on_policy_candidate(
                    text=gen_text,
                    token_ids=gen_ids,
                    task_type=task_type,
                    lang=lang,
                    ground_truth_num=gt_num,
                    requires_think=req_think,
                    requires_abstention=req_abstain,
                    ground_truth_text=item.get("ground_truth_text"),
                )
                candidates.append((sc, gen_ids, gen_text))

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_ids, best_txt = candidates[0]
            worst_sc, worst_ids, worst_txt = candidates[-1]

            if (best_sc - worst_sc) >= 0.35 and best_ids != worst_ids:
                record = {
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids,
                    "chosen": best_txt,
                    "chosen_ids": best_ids,
                    "rejected": worst_txt,
                    "rejected_ids": worst_ids,
                    "score_chosen": round(best_sc, 4),
                    "score_rejected": round(worst_sc, 4),
                    "lang": lang,
                    "task_type": task_type,
                    "license": "Apache-2.0",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                current_pairs += 1

            if idx % 10 == 0 or current_pairs >= TARGET_PAIRS:
                el = time.time() - t_start
                print(
                    f"[Top-Up] Tried {idx}/{len(unpaired)} prompts | Total pairs: {current_pairs}/{TARGET_PAIRS} | "
                    f"{el:.0f}s elapsed",
                    flush=True,
                )

    del model
    torch.cuda.empty_cache()
    print(f"[+] Top-up complete. Final pairs count: {current_pairs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
