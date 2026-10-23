#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/inspect_checkpoint_slab_v3.py

Zero-GPU CPU Inspection Probe for Parentheses 0.9-130M-v3 Checkpoints:
  - Audits slab_v3 Subsystems:
      1. Layer-0 V0 Value Residual Highway (v_residual_lamb)
      2. Beta-Gated Selective Write (beta_proj.weight norm & beta_proj.bias scale)
      3. Token-Shift Mu Mixing (token_shift.mu mean & range)
      4. Segregated Optimizer Parameter Groups (lr, weight decay)
  - Evaluates optional quick CPU validation loss on held-out data/processed_130m_v1/val.bin

Usage:
  python scripts/inspect_checkpoint_slab_v3.py checkpoints/parentheses-0.9-130M-v3/step_1500.pt
  python scripts/inspect_checkpoint_slab_v3.py checkpoints/parentheses-0.9-130M-v3/step_2000.pt --eval-val
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from model import Parentheses


def inspect_checkpoint(ckpt_path: Path, eval_val: bool = False, max_val_docs: int = 15):
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    print("=" * 75)
    print(f"SLAB_V3 ZERO-GPU AUDIT: {ckpt_path.name}")
    print(f"Path: {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.2f} GB)")
    print("=" * 75)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", 0)
    cfg = ckpt.get("cfg")
    m = ckpt.get("model", {})
    opt = ckpt.get("optimizer", {})

    print(f"Step: {step:,}")
    if cfg:
        print(f"Architecture: {cfg.n_layer} layers | {cfg.n_head} heads | d_model={cfg.n_embd} | slab_v3={getattr(cfg, 'slab_v3', False)}")

    # 1. Optimizer Groups
    if opt and "param_groups" in opt:
        print("\n[Optimizer Groups]")
        for g in opt["param_groups"]:
            name = g.get("name", "default")
            lr = g.get("lr")
            wd = g.get("weight_decay")
            print(f"  Group '{name:<18}': lr={lr:.2e} | weight_decay={wd}")

    # 2. V0 Value Highway (v_residual_lamb)
    v_keys = [k for k in m if "v_residual_lamb" in k]
    v_keys.sort(key=lambda k: int(k.split(".")[1]) if k.split(".")[1].isdigit() else 0)
    if v_keys:
        v_vals = [torch.sigmoid(m[k]).mean().item() for k in v_keys]
        v_mean = sum(v_vals) / len(v_vals)
        print("\n[1. Layer-0 V0 Value Residual Highway: sigma(lambda_v)]")
        print(f"  Overall Mean: {v_mean:.4f} | Min: {min(v_vals):.4f} (L{v_vals.index(min(v_vals))}) | Max: {max(v_vals):.4f} (L{v_vals.index(max(v_vals))})")
        per_layer_str = " ".join([f"L{i}:{v:.3f}" for i, v in enumerate(v_vals)])
        print(f"  Per-layer:    {per_layer_str}")

    # 3. Beta-Gated Selective Write (beta_proj.bias & beta_proj.weight)
    b_bias_keys = [k for k in m if "beta_proj.bias" in k]
    b_bias_keys.sort(key=lambda k: int(k.split(".")[1]) if k.split(".")[1].isdigit() else 0)
    b_weight_keys = [k for k in m if "beta_proj.weight" in k]
    b_weight_keys.sort(key=lambda k: int(k.split(".")[1]) if k.split(".")[1].isdigit() else 0)

    if b_bias_keys and b_weight_keys:
        b_vals = [(2.0 * torch.sigmoid(m[k])).mean().item() for k in b_bias_keys]
        w_norms = [m[k].norm().item() for k in b_weight_keys]
        b_mean = sum(b_vals) / len(b_vals)
        w_mean = sum(w_norms) / len(w_norms)
        print("\n[2. Beta-Gated Selective Write Subsystem]")
        print(f"  Static Bias Scale (2*sig(b_beta)): Mean={b_mean:.4f} | Min={min(b_vals):.4f} | Max={max(b_vals):.4f}")
        print(f"  Dynamic Weight Norm (||W_beta||_F): Mean={w_mean:.4f} | Min={min(w_norms):.4f} | Max={max(w_norms):.4f}")
        bias_str = " ".join([f"L{i}:{v:.3f}" for i, v in enumerate(b_vals)])
        norm_str = " ".join([f"L{i}:{v:.2f}" for i, v in enumerate(w_norms)])
        print(f"  Bias Per-Layer:  {bias_str}")
        print(f"  Norm Per-Layer:  {norm_str}")

    # 4. Token-Shift Mu Mixing (token_shift.mu)
    mu_keys = [k for k in m if "token_shift.mu" in k]
    mu_keys.sort(key=lambda k: int(k.split(".")[1]) if k.split(".")[1].isdigit() else 0)
    if mu_keys:
        mu_means = [m[k].mean().item() for k in mu_keys]
        mu_mins = [m[k].min().item() for k in mu_keys]
        mu_maxs = [m[k].max().item() for k in mu_keys]
        overall_mu = sum(mu_means) / len(mu_means)
        print("\n[3. Token-Shift Mu-Mixing Subsystem]")
        print(f"  Channel-wise Mu: Mean={overall_mu:.4f} | Min Across Layers={min(mu_mins):.4f} | Max Across Layers={max(mu_maxs):.4f}")
        mu_str = " ".join([f"L{i}:{v:.3f}" for i, v in enumerate(mu_means)])
        print(f"  Mu Per-Layer:   {mu_str}")

    # 5. Optional Quick Held-Out Validation Evaluation
    if eval_val:
        val_bin = REPO_ROOT / "data/processed_130m_v1/val.bin"
        val_bnd = REPO_ROOT / "data/processed_130m_v1/val_boundaries.npy"
        if val_bin.exists() and val_bnd.exists():
            print(f"\n[4. Fast CPU Held-Out Validation Loss Probe ({max_val_docs} docs)]")
            data = np.fromfile(val_bin, dtype=np.uint16)
            bnd = np.append(np.load(val_bnd), len(data))
            samples = []
            for s, e in zip(bnd[:-1], bnd[1:]):
                if 50 < e - s <= 256:
                    samples.append(data[s:e])
                if len(samples) >= max_val_docs:
                    break
            model = Parentheses(cfg).eval()
            model.load_state_dict(m)
            losses = []
            with torch.no_grad():
                for doc in samples:
                    x = torch.from_numpy(doc[:-1].astype(np.int64)).unsqueeze(0)
                    y = torch.from_numpy(doc[1:].astype(np.int64)).unsqueeze(0)
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            print(f"  Empirical Mean Val Loss: {mean_loss:.4f} (across {len(samples)} short docs)")
        else:
            print("\n[4. Validation file val.bin not found on E:, skipping val loss probe]")

    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect slab_v3 checkpoint parameters on CPU")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--eval-val", action="store_true", help="Run fast CPU validation loss probe")
    parser.add_argument("--max-val-docs", type=int, default=15, help="Number of docs to probe for val loss")
    args = parser.parse_args()

    inspect_checkpoint(Path(args.checkpoint), eval_val=args.eval_val, max_val_docs=args.max_val_docs)


if __name__ == "__main__":
    main()
