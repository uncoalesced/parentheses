#!/usr/bin/env python3
"""Direct Preference Optimization for Parentheses 0.9-120M (honesty calibration).

Policy and reference start from the same checkpoint; the reference is frozen.
Pairs are JSONL: {"prompt": ..., "chosen": ..., "rejected": ...}.

Loss (per pair, response-token log-prob sums normalized by length):
  -log sigmoid(beta * ((logp_pi_c - logp_ref_c) - (logp_pi_r - logp_ref_r)))

Usage:
  .venv\\Scripts\\python.exe scripts\\train_dpo_120m.py --policy checkpoints/.../step_9785_final.pt \\
      --pairs out/dpo_pairs_v1.jsonl --out-dir checkpoints/parentheses-0.9-120M-dpo-v1 \\
      --steps 150 --lr 1e-6 --beta 0.2 --batch-size 4 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def load_tok(path: str) -> tuple[Tokenizer, dict]:
    tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    sha = hashlib.sha256((Path(path) / "tokenizer.json").read_bytes()).hexdigest()
    return tok, {"path": path, "sha256": sha}


def load_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["cfg"], ckpt["model"], ckpt.get("step", 0)


def load_torch_state(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def encode_pair(tok: Tokenizer, prompt: str, response: str) -> tuple[list[int], int]:
    prefix = [USER, ENG] + tok.encode(prompt, add_special_tokens=False).ids + [ASSISTANT]
    resp = tok.encode(response, add_special_tokens=False).ids + [EOT]
    return prefix + resp, len(prefix)


def make_batch(tok: Tokenizer, rows: list[dict], max_len: int, device: str):
    seqs, resp_starts = [], []
    for r in rows:
        ids, start = encode_pair(tok, r["prompt"], r["response"])
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
                   device: str, needs_grad: bool) -> torch.Tensor:
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
    return tok_logp.sum(dim=1) / counts


def make_optimizer(model: torch.nn.Module, lr: float, wd: float):
    return build_optimizer(model, lr=lr, weight_decay=wd)


def save_ckpt(path: Path, model, optimizer, cfg, step: int, tok_meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": cfg, "step": step, "tokenizer": tok_meta}, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--ref", default=None, help="reference checkpoint (default: same as policy)")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tokenizer", default="data/tokenizer/v3_32k")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--sft-coef", type=float, default=0.0,
                    help="optional CE on chosen responses (0 = pure DPO)")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.policy or not args.pairs or not args.out_dir:
        ap.error("--policy, --pairs, and --out-dir are required unless --self-test is set")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    cfg, state, start_step = load_ckpt(args.policy, device)
    tok, tok_meta = load_tok(args.tokenizer)

    policy = Parentheses(cfg).to(device)
    policy.load_state_dict(state)
    policy.train()
    ref_state = state if not args.ref else load_ckpt(args.ref, device)[1]
    ref = Parentheses(cfg).to(device)
    ref.load_state_dict(ref_state)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    pairs = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not pairs:
        raise SystemExit("no pairs")
    print(f"[dpo] {len(pairs)} pairs; policy={args.policy} ref={'self' if not args.ref else args.ref}")
    print(f"[dpo] steps={args.steps} batch={args.batch_size} lr={args.lr} beta={args.beta} "
          f"sft_coef={args.sft_coef} device={device}")

    opt = make_optimizer(policy, args.lr, args.weight_decay)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)
    t0 = time.time()
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cursor = 0
    pos = 0
    ema = None
    for step in range(1, args.steps + 1):
        batch = []
        while len(batch) < args.batch_size:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch.append(pairs[order[cursor]])
            cursor += 1

        # chosen and rejected forwards are stacked: [c0..cB-1, r0..rB-1]
        rows_c = [{"prompt": p["prompt"], "response": p["chosen"]} for p in batch]
        rows_r = [{"prompt": p["prompt"], "response": p["rejected"]} for p in batch]
        x_c, m_c = make_batch(tok, rows_c, args.max_len, device)
        x_r, m_r = make_batch(tok, rows_r, args.max_len, device)

        pi_c = response_logps(policy, x_c, m_c, device, needs_grad=True)
        pi_r = response_logps(policy, x_r, m_r, device, needs_grad=True)
        with torch.no_grad():
            ref_c = response_logps(ref, x_c, m_c, device, needs_grad=False)
            ref_r = response_logps(ref, x_r, m_r, device, needs_grad=False)

        margin = (pi_c - ref_c) - (pi_r - ref_r)
        loss = -F.logsigmoid(args.beta * margin).mean()
        if args.sft_coef > 0:
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits, _ = policy(x_c)
            shift_logits = logits[:, :-1, :].float()
            shift_mask = m_c[:, 1:]
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                x_c[:, 1:].reshape(-1), ignore_index=0, reduction="none")
            ce = (ce * shift_mask.reshape(-1)).sum() / shift_mask.sum().clamp_min(1)
            loss = loss + args.sft_coef * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        opt.step()

        val = loss.item()
        ema = val if ema is None else 0.9 * ema + 0.1 * val
        acc = (margin > 0).float().mean().item()
        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"[dpo] step {step}/{args.steps} loss {val:.4f} ema {ema:.4f} "
                  f"acc {acc:.2f} beta_margin {args.beta * margin.mean().item():.4f} "
                  f"{(dt / step):.2f}s/step", flush=True)
        if step % args.ckpt_every == 0 and step < args.steps:
            save_ckpt(out_dir / f"step_{step:04d}.pt", policy, opt, cfg, step, tok_meta)
    final = out_dir / f"step_{args.steps:04d}_final.pt"
    save_ckpt(final, policy, opt, cfg, start_step + args.steps, tok_meta)
    print(f"[dpo] saved {final}")
    return 0


def self_test() -> int:
    from model.config import PRESETS
    torch.manual_seed(0)
    cfg = PRESETS["tiny-smoke"]
    model = Parentheses(cfg)
    x = torch.randint(1, cfg.vocab_size - 1, (2, 16))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 8:] = True
    lp = response_logps(model, x, mask, "cpu", needs_grad=True)
    assert lp.shape == (2,), lp.shape
    lp.mean().backward()
    assert any(p.grad is not None for p in model.parameters())
    opt = make_optimizer(model, 1e-4, 0.0)
    assert len(opt.param_groups) >= 2
    lp2 = response_logps(model, x, mask, "cpu", needs_grad=False)
    assert lp2.shape == (2,) and torch.isfinite(lp2).all()
    print("[dpo] self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
