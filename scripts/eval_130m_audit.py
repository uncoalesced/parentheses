#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_130m_audit.py

Fast, CPU-Safe Diagnostic Audit for Parentheses 0.9-130M-v3 (Project Tanzinite):
  1. Dynamic Write Selectivity:
     - Frobenius norm of block.attn.beta_proj.weight (w.norm().item()) across Layers 0-15
     - Effective write scale: 2 * sigmoid(beta_proj.bias) across Layers 0-15
  2. Token-Shift Dynamics (mu-mix):
     - Min, Max, Mean, Std of token_shift.mu across all channels and layers
  3. Value Residual Highway (V0):
     - Effective V0 mixing weight: sigmoid(v_residual_lamb) across Layers 1-15
  4. (Optional) Validation Loss & Perplexity on data/processed_130m_v1/val.bin

Usage:
  python scripts/eval_130m_audit.py --ckpt checkpoints/parentheses-0.9-130M-v3/step_1500.pt
  python scripts/eval_130m_audit.py --ckpt checkpoints/parentheses-0.9-130M-v3/step_2000.pt --eval-loss --batches 10
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Force CPU mode to guarantee zero VRAM collision with training run
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.backbone import Parentheses
from model.config import ModelConfig


def audit_checkpoint(ckpt_path: Path, eval_loss: bool = False, batches: int = 5, seq_len: int = 2048) -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", -1)
    cfg = ckpt.get("cfg", None) or ckpt.get("config", None)
    sd = ckpt["model"]

    # 1. Optimizer and Scaler Stats
    opt_sd = ckpt.get("optimizer", {})
    lrs = [g["lr"] for g in opt_sd.get("param_groups", [])] if opt_sd else []
    scaler_sd = ckpt.get("scaler", {})
    scale = scaler_sd.get("scale", None) if scaler_sd else None

    # 2. Beta Proj Dynamics (Selective Write)
    beta_weight_norms = []
    beta_bias_scales = []
    for layer_idx in range(16):
        w_key = f"blocks.{layer_idx}.attn.beta_proj.weight"
        b_key = f"blocks.{layer_idx}.attn.beta_proj.bias"
        if w_key in sd:
            beta_weight_norms.append(sd[w_key].norm().item())
        if b_key in sd:
            beta_bias_scales.append((2.0 * torch.sigmoid(sd[b_key])).mean().item())

    # 3. Token-Shift Mu Dynamics
    mu_tensors = [v for k, v in sd.items() if ".mu" in k]
    if mu_tensors:
        all_mu = torch.cat([m.flatten() for m in mu_tensors])
        mu_min = all_mu.min().item()
        mu_max = all_mu.max().item()
        mu_mean = all_mu.mean().item()
        mu_std = all_mu.std().item()
    else:
        mu_min, mu_max, mu_mean, mu_std = 0.0, 0.0, 0.0, 0.0

    # 4. Value Residual Highway (V0)
    v0_lambs = []
    for layer_idx in range(1, 16):
        v_key = f"blocks.{layer_idx}.attn.v_residual_lamb"
        if v_key in sd:
            v0_lambs.append(torch.sigmoid(sd[v_key]).mean().item())

    result = {
        "step": step,
        "checkpoint": str(ckpt_path),
        "file_size_gb": ckpt_path.stat().st_size / 1e9,
        "lrs": lrs,
        "loss_scale": scale,
        "beta_weight_norm_mean": float(np.mean(beta_weight_norms)) if beta_weight_norms else 0.0,
        "beta_bias_scale_mean": float(np.mean(beta_bias_scales)) if beta_bias_scales else 0.0,
        "mu_min": mu_min,
        "mu_max": mu_max,
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "v0_highway_mix_mean": float(np.mean(v0_lambs)) if v0_lambs else 0.0,
    }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  TANZINITE 130M-v3 AUDIT: Step {step:6d} ({ckpt_path.name})")
    print("=" * 70)
    print(f"  File Size:              {result['file_size_gb']:.2f} GB")
    print(f"  Learning Rates:         {lrs}")
    if scale is not None:
        print(f"  Loss Scale:             {scale}")
    print("-" * 70)
    print("  RECURRENT GATE TELEMETRY (slab_v3):")
    print(f"  - Dynamic Write Weight Norm: {result['beta_weight_norm_mean']:.4f}  (modulates write by x_t)")
    print(f"  - Static Write Scale Mean:   {result['beta_bias_scale_mean']:.4f}  (neutral baseline = 1.0000)")
    print(f"  - Token-Shift mu Range:      [{result['mu_min']:.4f}, {result['mu_max']:.4f}]  (init = 0.5000)")
    print(f"  - Token-Shift mu Mean/Std:   {result['mu_mean']:.4f} +/- {result['mu_std']:.4f}")
    print(f"  - V0 Value Highway Mix:      {result['v0_highway_mix_mean']:.4f}  (init = 0.5000)")
    print("=" * 70)

    # 5. Optional Validation Loss Evaluation
    if eval_loss and cfg is not None:
        val_bin = REPO_ROOT / "data/processed_130m_v1/val.bin"
        if val_bin.exists():
            print(f"\n[Eval] Running CPU validation loss on {batches} batches of seq_len={seq_len}...")
            model = Parentheses(cfg)
            model.load_state_dict(sd)
            model.eval()

            data = np.memmap(val_bin, dtype=np.uint16, mode="r")
            n_tokens = len(data)
            losses = []

            # Deterministic, evenly-spaced strided evaluation slices across all 4 pillars of val.bin
            max_start = n_tokens - (seq_len + 1)
            offsets = np.linspace(0, max_start, batches, dtype=int)
            with torch.no_grad():
                for idx in offsets:
                    chunk = torch.from_numpy(data[int(idx) : int(idx) + seq_len + 1].astype(np.int64))
                    x = chunk[:-1].unsqueeze(0)  # [1, seq_len]
                    y = chunk[1:].unsqueeze(0)   # [1, seq_len]

                    _, loss = model(x, targets=y, return_logits=False)
                    losses.append(loss.item())

            avg_loss = float(np.mean(losses))
            ppl = math.exp(min(avg_loss, 20.0))
            result["val_loss"] = avg_loss
            result["val_ppl"] = ppl
            print(f"  Validation Loss:        {avg_loss:.4f}")
            print(f"  Validation Perplexity:  {ppl:.2f}")
            print("=" * 70 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Tanzinite 130M-v3 Checkpoint Diagnostic Audit")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--eval-loss", action="store_true", help="Compute validation loss on CPU")
    parser.add_argument("--batches", type=int, default=5, help="Number of validation batches to evaluate")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for validation evaluation")
    args = parser.parse_args()

    audit_checkpoint(Path(args.ckpt), eval_loss=args.eval_loss, batches=args.batches, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
