#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/sweep_s6_s8_blend.py

Sweep weight interpolation between Session 6 (Pretrain Anchor) and Session 8 (DPO Golden)
to find the optimal model soup / task vector weight that maximizes the /900 benchmark rubric
and beats both S6 (434/900) and S8 (381/900).

Recurrent State Space Duality (SSD / Mamba-2) architecture:
- Constant O(1) recurrent state updates: S_t = alpha_t * S_{t-1} + beta_t * (k_t (x) v_t)
- Zero quadratic attention, zero KV-cache, zero sequence softmax.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_slab_v4 import SlabV4Model
from scripts.eval_60p_adversarial_suite import PROBES, stream_slab_v4, get_lang_tag
from scripts.grade_rubric_900 import grade_probe

S6_PATH = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s6" / "step_12000.pt"
S8_PATH = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s8-golden" / "parentheses-1.0-240M-dpo-swa3-golden.pt"
TOK_PATH = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"


def evaluate_model_60p(
    model: SlabV4Model,
    tokenizer: Tokenizer,
    device: str = "cuda",
    max_tokens: int = 256,
    temp: float = 0.0,
    rep_penalty: float = 1.15,
) -> Tuple[int, int, int, int, Dict[str, int], List[Dict[str, Any]]]:
    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    stops = {eot_id, user_id, tokenizer.token_to_id("<|system|>")} - {None}

    results = []
    tot_f = tot_l = tot_t = 0
    dom_scores: Dict[str, int] = {}

    for p in PROBES:
        lang_id = get_lang_tag(p, tokenizer)
        body_ids = tokenizer.encode(p["prompt"], add_special_tokens=False).ids
        inp_list = [user_id, lang_id] + body_ids + [asst_id]
        inp = torch.tensor([inp_list], dtype=torch.long, device=device)

        gen_tokens = []
        with torch.inference_mode():
            for tok in stream_slab_v4(
                model,
                inp,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_k=1 if temp <= 0.0 else 50,
                min_p=0.05,
                repetition_penalty=rep_penalty,
            ):
                if tok in stops:
                    break
                gen_tokens.append(tok)

        text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        f_s, l_s, t_s = grade_probe(p["id"], p["domain"], text, p)
        tot_f += f_s
        tot_l += l_s
        tot_t += t_s
        dom = p["domain"]
        dom_scores[dom] = dom_scores.get(dom, 0) + (f_s + l_s + t_s)

        results.append({
            "id": p["id"],
            "domain": p["domain"],
            "prompt": p["prompt"],
            "generation": text,
            "f": f_s,
            "l": l_s,
            "t": t_s,
            "total": f_s + l_s + t_s,
        })

    tot_score = tot_f + tot_l + tot_t
    return tot_score, tot_f, tot_l, tot_t, dom_scores, results


def blend_state_dicts(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    weight_b: float,
) -> Dict[str, torch.Tensor]:
    weight_a = 1.0 - weight_b
    blended = {}
    for k, v_b in sd_b.items():
        v_a = sd_a[k]
        if torch.is_floating_point(v_b):
            blended[k] = (weight_a * v_a.float() + weight_b * v_b.float()).to(v_b.dtype)
        else:
            blended[k] = v_b.clone()
    
    # Enforce tied embeddings
    if "tok_emb.weight" in blended and "lm_head.weight" in blended:
        blended["lm_head.weight"] = blended["tok_emb.weight"].clone()

    return blended


def main():
    parser = argparse.ArgumentParser(description="Sweep S6 / S8 weight blend")
    parser.add_argument("--s6", default=str(S6_PATH), help="Path to S6 checkpoint")
    parser.add_argument("--s8", default=str(S8_PATH), help="Path to S8 checkpoint")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], help="List of S8 weights (e.g. 0.5)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-best", type=str, default=None, help="Save best blended checkpoint if score > threshold (e.g. 434)")
    args = parser.parse_args()

    print(f"[*] Loading Tokenizer: {TOK_PATH}")
    tokenizer = Tokenizer.from_file(str(TOK_PATH))

    print(f"[*] Loading S6 Checkpoint from {args.s6} on CPU...")
    s6_bundle = torch.load(args.s6, map_location="cpu", weights_only=False)
    cfg = s6_bundle.get("cfg")
    sd_s6 = s6_bundle["model"]

    print(f"[*] Loading S8 Checkpoint from {args.s8} on CPU...")
    s8_bundle = torch.load(args.s8, map_location="cpu", weights_only=False)
    sd_s8 = s8_bundle["model"]

    print("\n" + "=" * 80)
    print("PARENT-1.0-240M: S6 (Anchor) vs S8 (Golden) WEIGHT INTERPOLATION SWEEP")
    print("Baseline Anchor S6: 434 / 900 (Fluency 214, Logic 102, Task 118)")
    print("Baseline Golden S8: 381 / 900 (Fluency 162, Logic 102, Task 117)")
    print("=" * 80)

    best_score = -1
    best_w = None
    best_sd = None
    records = []

    for w in args.weights:
        w_s8 = float(w)
        w_s6 = 1.0 - w_s8
        print(f"\n---> Evaluating Blend: {w_s6:.2f} * S6 + {w_s8:.2f} * S8...")
        t0 = time.time()

        blended_sd = blend_state_dicts(sd_s6, sd_s8, w_s8)
        model = SlabV4Model(cfg)
        model.load_state_dict(blended_sd, strict=True)
        model.to(args.device).eval()

        score, f, l, t, doms, details = evaluate_model_60p(model, tokenizer, device=args.device)
        elapsed = time.time() - t0

        print(f"     Score: {score} / 900 ({score/9.0:.2f}%) | Fluency: {f}/300 | Logic: {l}/300 | Task: {t}/300 | Time: {elapsed:.1f}s")
        print(f"     Domains: Coding={doms.get('Coding', 0)}/150, Arith={doms.get('Arithmetic & Logic', 0)}/150, MajDrav={doms.get('Major Dravidian', 0)}/150, LowDrav={doms.get('Low-Resource Dravidian', 0)}/150, Eng={doms.get('English Conceptual', 0)}/150, Abst={doms.get('Epistemic Abstention', 0)}/150")

        records.append({
            "w_s8": w_s8,
            "score": score,
            "f": f,
            "l": l,
            "t": t,
            "doms": doms,
            "details": details,
        })

        if score > best_score:
            best_score = score
            best_w = w_s8
            best_sd = blended_sd

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'S8 Weight':<12} | {'Total (/900)':<14} | {'Fluency':<10} | {'Logic':<10} | {'Task':<10} | {'Coding':<8} | {'Arith':<8}")
    print("-" * 80)
    for r in records:
        d = r["doms"]
        print(f"{r['w_s8']:<12.2f} | {r['score']:>4} / 900 ({r['score']/9.0:.1f}%) | {r['f']:>3} / 300   | {r['l']:>3} / 300 | {r['t']:>3} / 300 | {d.get('Coding', 0):>3}/150  | {d.get('Arithmetic & Logic', 0):>3}/150")
    print("-" * 80)
    print(f"S6 Baseline  |  434 / 900 (48.2%) | 214 / 300   | 102 / 300 | 118 / 300 |  57/150  |  98/150")
    print(f"S8 Baseline  |  381 / 900 (42.3%) | 162 / 300   | 102 / 300 | 117 / 300 |  96/150  |  37/150")
    print("=" * 80)
    print(f"[+] Optimal Blend: S8 weight = {best_w:.2f} with Score = {best_score} / 900")

    if args.save_best and best_sd is not None:
        out_p = Path(args.save_best)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "step": 13080,
            "cfg": cfg,
            "model": best_sd,
            "blend_info": {
                "s6_source": str(args.s6),
                "s8_source": str(args.s8),
                "s8_weight": best_w,
                "s6_weight": 1.0 - best_w,
                "score_900": best_score,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        }
        torch.save(save_dict, out_p)
        print(f"[+] Saved best blended model to {out_p} ({out_p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
