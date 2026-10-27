#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_dpo_130m_v3.py

Direct Preference Optimization (DPO) Calibrator for Parentheses 0.9-130M-v3 (Project Tanzinite):
  - Calibrated Honesty & Identity Alignment
  - Balanced 1:1 Preference Pairs: out/dpo_pairs_balanced.jsonl
  - Universal Segregated Differential Optimizer: Zero weight decay on alpha_proj, conv_w, and RMSNorm
  - SFT Regularization Anchor: --sft-coef 0.3 to prevent policy drift and collapse

Usage:
  .venv\\Scripts\\python.exe scripts\\train_dpo_130m_v3.py \\
      --policy checkpoints/parentheses-0.9-130M-v3-sft/step_2000_final.pt \\
      --pairs out/dpo_pairs_balanced.jsonl \\
      --out-dir checkpoints/parentheses-0.9-130M-v3-dpo-calibrated \\
      --steps 100 --lr 5e-7 --beta 0.1 --sft-coef 0.3 --batch-size 4 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Parentheses
from train import build_optimizer

EOT, ENG, USER, ASSISTANT = 0, 10, 13, 14

LANG_MAP = {
    "kan": 2, "kn": 2,
    "tam": 3, "ta": 3,
    "tel": 4, "te": 4,
    "mal": 5, "ml": 5,
    "eng": 10, "en": 10,
}


def detect_lang_id(prompt: str, lang: str | None = None) -> int:
    if lang and lang in LANG_MAP:
        return LANG_MAP[lang]
    if re.search(r"[ಀ-೿]", prompt):
        return 2  # kan
    if re.search(r"[஀-௿]", prompt):
        return 3  # tam
    if re.search(r"[ఀ-౿]", prompt):
        return 4  # tel
    if re.search(r"[ഀ-ൿ]", prompt):
        return 5  # mal
    return 10  # eng


def load_tok(path: str) -> tuple[Tokenizer, dict]:
    tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    sha = hashlib.sha256((Path(path) / "tokenizer.json").read_bytes()).hexdigest()
    return tok, {"path": path, "sha256": sha}


def load_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["cfg"], ckpt["model"], ckpt.get("step", 0)


def encode_pair(tok: Tokenizer, prompt: str, response: str, lang: str | None = None) -> tuple[list[int], int]:
    lang_id = detect_lang_id(prompt, lang)
    p_ids = tok.encode(prompt.strip()).ids
    r_ids = tok.encode(response.strip()).ids
    # Aligns with SFT & FreeThink prompt order: <|user|><lang> prompt <|assistant|> response <|endoftext|>
    ids = [USER, lang_id] + p_ids + [ASSISTANT] + r_ids + [EOT]
    response_start = 2 + len(p_ids) + 1
    return ids, response_start


def collate_batch(batch: list[tuple[list[int], int]], device: str,
                  max_len: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    resp_starts = []
    for ids, start in batch:
        if len(ids) > max_len:
            ids = ids[:max_len]
            start = min(start, max_len - 1)
        seqs.append(ids)
        resp_starts.append(start)
    T = max(len(s) for s in seqs)
    x = torch.zeros((len(seqs), T), dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (s, st) in enumerate(zip(seqs, resp_starts)):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, st:len(s)] = True
        mask[i, st] = True
    return x.to(device), mask.to(device)


def response_logps(model: Parentheses, x: torch.Tensor, mask: torch.Tensor,
                   device: str, needs_grad: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    ctx = torch.enable_grad() if needs_grad else torch.no_grad()
    with ctx, torch.autocast(device_type=device, dtype=torch.bfloat16,
                             enabled=(device == "cuda")):
        logits, _ = model(x)
    if not needs_grad:
        logits = logits.float()
    shift_logits = logits[:, :-1, :].float()
    shift_mask = mask[:, 1:]
    targets = x[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * shift_mask
    counts = shift_mask.sum(dim=1).clamp_min(1)
    mean_logp = tok_logp.sum(dim=1) / counts

    ce_loss = None
    if needs_grad:
        ce_loss = -tok_logp.sum() / shift_mask.sum().clamp_min(1)
    return mean_logp, ce_loss


@torch.no_grad()
def pair_diffs(model: Parentheses, encoded: list, device: str, max_len: int,
               batch_size: int = 8) -> torch.Tensor:
    """Chosen-minus-rejected mean response log-prob for every pair, in order (no grad, eval mode)."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i:i + batch_size]
        c_x, c_mask = collate_batch([e[0] for e in chunk], device, max_len=max_len)
        r_x, r_mask = collate_batch([e[1] for e in chunk], device, max_len=max_len)
        c, _ = response_logps(model, c_x, c_mask, device, needs_grad=False)
        r, _ = response_logps(model, r_x, r_mask, device, needs_grad=False)
        out.append((c - r).float().cpu())
    model.train(was_training)
    return torch.cat(out)


def whole_set_stats(pi_diff: torch.Tensor, ref_diff: torch.Tensor, beta: float) -> dict:
    """Margin / accuracy / DPO loss over all pairs; the per-step log line only sees one batch."""
    m = pi_diff - ref_diff
    return {"pairs": len(m), "margin": m.mean().item(), "median_margin": m.median().item(),
            "acc": (m > 0).float().mean().item(), "loss": -F.logsigmoid(beta * m).mean().item()}


def save_ckpt(path: Path, model, optimizer, cfg, step: int, tok_meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": cfg, "step": step, "tokenizer": tok_meta}, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Parentheses 0.9-130M-v3 Calibrated DPO Alignment")
    ap.add_argument("--policy", "--base-ckpt", dest="policy", default="checkpoints/parentheses-0.9-130M-v3-sft/step_1999_final.pt")
    ap.add_argument("--ref", default=None, help="Reference checkpoint (default: same as policy)")
    ap.add_argument("--pairs", default="out/dpo_pairs_balanced.jsonl")
    ap.add_argument("--out-dir", default="checkpoints/parentheses-0.9-130M-v3-dpo-calibrated")
    ap.add_argument("--tokenizer", default="data/tokenizer/v3_32k")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--sft-coef", "--sft-coeff", dest="sft_coef", type=float, default=0.3,
                    help="CE regularization coefficient on chosen responses")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--full-eval-every", type=int, default=None,
                    help="score margin/acc over ALL pairs every N steps (default --ckpt-every; 0 = only "
                         "before training and at the end). Written to <out-dir>/pair_eval.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("PARENTHESES 0.9-130M-V3 CALIBRATED DPO ALIGNMENT")
    print(f"Policy:    {args.policy}")
    print(f"Pairs:     {args.pairs}")
    print(f"Out Dir:   {args.out_dir}")
    print(f"Hyperparams: steps={args.steps}, lr={args.lr}, beta={args.beta}, sft_coef={args.sft_coef}")
    print("=" * 70)

    if not Path(args.policy).exists():
        # Fallback search in priority order: step_1999_final.pt, step_2000_final.pt, or newest *.pt
        cands = [
            Path("checkpoints/parentheses-0.9-130M-v3-sft/step_1999_final.pt"),
            Path("checkpoints/parentheses-0.9-130M-v3-sft/step_2000_final.pt"),
        ] + sorted(Path("checkpoints/parentheses-0.9-130M-v3-sft").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        found = next((c for c in cands if c.exists()), None)
        if found:
            args.policy = str(found)
            print(f"[notice] Policy fallback to: {args.policy}")
        else:
            sys.exit(f"[ABORT] Policy checkpoint not found: {args.policy}")

    ref_path = args.ref or args.policy
    tok, tok_meta = load_tok(args.tokenizer)

    cfg, policy_state, base_step = load_ckpt(args.policy, args.device)
    policy = Parentheses(cfg).to(args.device)
    policy.load_state_dict(policy_state)
    policy.train()

    # Frozen Reference Model: Loaded strictly in BF16 on GPU (or CPU) to conserve VRAM
    _, ref_state, _ = load_ckpt(ref_path, "cpu")
    ref_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    ref = Parentheses(cfg).to(dtype=ref_dtype, device=args.device)
    ref.load_state_dict({k: v.to(dtype=ref_dtype) if torch.is_floating_point(v) else v for k, v in ref_state.items()})
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # Universal segregated optimizer
    optimizer = build_optimizer(policy, lr=args.lr, weight_decay=args.weight_decay)

    # Load pairs
    with open(args.pairs, "r", encoding="utf-8") as f:
        raw_pairs = [json.loads(line) for line in f if line.strip()]
    print(f"[dpo] Loaded {len(raw_pairs)} preference pairs from {args.pairs}")

    encoded = []
    for row in raw_pairs:
        p = row.get("prompt", "")
        c = row.get("chosen", "")
        r = row.get("rejected", "")
        if not p or not c or not r:
            continue
        lang = row.get("lang")
        c_ids, c_start = encode_pair(tok, p, c, lang)
        r_ids, r_start = encode_pair(tok, p, r, lang)
        encoded.append(((c_ids, c_start), (r_ids, r_start)))

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(args.device, enabled=(args.device == "cuda"))

    # whole-set margins: the reference is frozen, so its diffs are scored once
    full_every = args.ckpt_every if args.full_eval_every is None else args.full_eval_every
    ref_all = pair_diffs(ref, encoded, args.device, args.max_len)
    pair_log = []

    def full_eval(step: int):
        s = {"step": step, **whole_set_stats(pair_diffs(policy, encoded, args.device, args.max_len),
                                             ref_all, args.beta)}
        pair_log.append(s)
        (out_p / "pair_eval.json").write_text(json.dumps(pair_log, indent=1), encoding="utf-8")
        print(f"[all-pairs] step {step:4d} | {s['pairs']} pairs | margin {s['margin']:+.4f} "
              f"(median {s['median_margin']:+.4f}) | acc {s['acc']*100:5.1f}% | dpo {s['loss']:.4f}")

    full_eval(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = [random.choice(encoded) for _ in range(args.batch_size)]
        c_batch = [item[0] for item in batch]
        r_batch = [item[1] for item in batch]

        c_x, c_mask = collate_batch(c_batch, args.device, max_len=args.max_len)
        r_x, r_mask = collate_batch(r_batch, args.device, max_len=args.max_len)

        optimizer.zero_grad(set_to_none=True)

        pi_c, sft_loss = response_logps(policy, c_x, c_mask, args.device, needs_grad=True)
        pi_r, _ = response_logps(policy, r_x, r_mask, args.device, needs_grad=True)

        with torch.no_grad():
            ref_c, _ = response_logps(ref, c_x, c_mask, args.device, needs_grad=False)
            ref_r, _ = response_logps(ref, r_x, r_mask, args.device, needs_grad=False)

        pi_diff = pi_c - pi_r
        ref_diff = ref_c - ref_r
        logits = args.beta * (pi_diff - ref_diff)
        dpo_loss = -F.logsigmoid(logits).mean()

        total_loss = dpo_loss
        if args.sft_coef > 0.0 and sft_loss is not None:
            total_loss = total_loss + args.sft_coef * sft_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            acc = (logits > 0).float().mean().item()
            margin = (pi_diff - ref_diff).mean().item()
            dt = time.time() - t0
            sft_str = f" | sft {sft_loss.item():.4f}" if sft_loss is not None else ""
            print(f"step {step:4d}/{args.steps} | dpo {dpo_loss.item():.4f}{sft_str} | "
                  f"batch({args.batch_size}) margin {margin:+.4f} acc {acc*100:5.1f}% | {dt:.1f}s")

        if full_every and step % full_every == 0 and step != args.steps:
            full_eval(step)

        if step % args.ckpt_every == 0:
            ckpt_path = out_p / f"step_{step:04d}.pt"
            save_ckpt(ckpt_path, policy, optimizer, cfg, base_step + step, tok_meta)
            print(f"[info] saved checkpoint: {ckpt_path}")

    full_eval(args.steps)
    final_path = out_p / f"step_{args.steps:04d}_final.pt"
    save_ckpt(final_path, policy, optimizer, cfg, base_step + args.steps, tok_meta)
    print(f"\n[info] Calibrated DPO completed! Saved final release candidate: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
