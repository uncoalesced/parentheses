#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/grow_model_slab_v4.py
Function-Preserving Net2WiderNet State Expansion: 130M-v3 -> 240M-v4.

Transfers trained weights from Parentheses 0.9-130M-v3 (148.38M params, 16 layers, d_model=768, 12 heads)
into Parentheses 1.0 (240M / slab_v4: 253.5M params, 16 layers, d_model=1024, 16 heads).

Mathematical Guarantees:
  1. Identical Depth: Exactly 16 layers preserved 1:1.
  2. Preserved Head Rank: d_head = 64 invariant (12 heads -> 16 heads).
  3. Function Preservation (Zero Residual Expansion):
     - tok_emb.weight[:, 768:1024] initialized to 0.0.
     - attn.proj.weight[768:1024, :] and [:, 768:1024] initialized to 0.0.
     - mlp.w2.weight[768:1024, :] and [:, 2048:2752] initialized to 0.0.
     - RMSNorm weights scaled by sqrt(768 / 1024) to maintain exact variance scaling.
  4. Prior Preservation:
     - Learned decay horizons log_alpha, beta-write biases, token-shift mu, and V0 highway
       lambdas for heads 0..11 are copied directly; heads 12..15 inherit diverse priors from 0..3 with jitter.
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.config import PRESETS, ModelConfig
from models.model_slab_v4 import SlabV4Config, SlabV4Model


@torch.no_grad()
def grow_130m_to_240m(
    src_path: str | Path,
    out_path: Optional[str | Path] = None,
    jitter: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[SlabV4Model, dict]:
    """
    Expand 130M-v3 checkpoint into 240M-v4 model with function-preserving initialization.
    """
    torch.manual_seed(seed)
    src_p = Path(src_path)
    if not src_p.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_p}")

    print(f"[grow] Loading source 130M-v3 checkpoint from {src_p}...")
    ckpt130 = torch.load(src_p, map_location="cpu", weights_only=False)
    sd130 = ckpt130["model"]
    cfg130 = ckpt130["cfg"]
    tok_meta = ckpt130.get("tokenizer", None)

    C1, C2 = 768, 1024
    H1, H2 = 12, 16
    Dh = 64
    assert C1 == H1 * Dh and C2 == H2 * Dh, "Head dim mismatch"
    norm_scale = math.sqrt(C1 / C2)

    # 240M target configuration
    d_cfg = SlabV4Config(
        vocab_size=getattr(cfg130, "vocab_size", 32768),
        block_size=4096,
        n_layer=16,
        n_head=16,
        n_embd=1024,
        chunk_size=64,
        tie_embeddings=True,
        bias=False,
        dropout=0.0,
        slab_v4=True,
    )

    print(f"[grow] Instantiating Parentheses 1.0 (240M) model architecture...")
    model240 = SlabV4Model(d_cfg)
    sd240 = model240.state_dict()

    # 1. Embeddings & Final Norm
    sd240["tok_emb.weight"].zero_()
    sd240["tok_emb.weight"][:, :C1] = sd130["tok_emb.weight"]

    sd240["ln_f.weight"].fill_(norm_scale)
    sd240["ln_f.weight"][:C1] = sd130["ln_f.weight"] * norm_scale

    # 2. Block-by-block parameter expansion
    for i in range(16):
        # Pre-norms
        sd240[f"blocks.{i}.ln1.weight"].fill_(norm_scale)
        sd240[f"blocks.{i}.ln1.weight"][:C1] = sd130[f"blocks.{i}.ln1.weight"] * norm_scale

        sd240[f"blocks.{i}.ln2.weight"].fill_(norm_scale)
        sd240[f"blocks.{i}.ln2.weight"][:C1] = sd130[f"blocks.{i}.ln2.weight"] * norm_scale

        # Token-Shift mu
        mu130 = sd130[f"blocks.{i}.attn.token_shift.mu"].squeeze()
        sd240[f"blocks.{i}.attn.mu"][:C1] = mu130
        sd240[f"blocks.{i}.attn.mu"][C1:] = mu130[:C2 - C1] + torch.randn(C2 - C1) * jitter

        # Conv1D weights & bias (Q, K, V parts of 768 -> 1024)
        cw130 = sd130[f"blocks.{i}.attn.conv_w"]   # [2304, 4]
        cb130 = sd130[f"blocks.{i}.attn.conv_b"]   # [2304]
        cw240 = sd240[f"blocks.{i}.attn.conv_weight"]  # [3072, 1, 4]
        cb240 = sd240[f"blocks.{i}.attn.conv_bias"]    # [3072]

        for p in range(3):
            src_slice = slice(p * C1, (p + 1) * C1)
            dst_slice = slice(p * C2, p * C2 + C1)
            extra_slice = slice(p * C2 + C1, (p + 1) * C2)

            # Conv weights
            cw240[dst_slice, 0, :] = cw130[src_slice]
            cw240[extra_slice, 0, :] = cw130[src_slice][:C2 - C1] + torch.randn(C2 - C1, 4) * jitter

            # Conv biases
            cb240[dst_slice] = cb130[src_slice]
            cb240[extra_slice] = cb130[src_slice][:C2 - C1] + torch.randn(C2 - C1) * jitter

        # QKV Projections
        qkv130 = sd130[f"blocks.{i}.attn.qkv.weight"]   # [2304, 768]
        qkv240 = sd240[f"blocks.{i}.attn.qkv.weight"]   # [3072, 1024]
        qkv240.normal_(0.0, 0.01)

        for p in range(3):
            src_r = slice(p * C1, (p + 1) * C1)
            dst_r = slice(p * C2, p * C2 + C1)
            extra_r = slice(p * C2 + C1, (p + 1) * C2)

            qkv240[dst_r, :C1] = qkv130[src_r, :C1]
            qkv240[extra_r, :C1] = qkv130[src_r, :C1][:C2 - C1, :] + torch.randn(C2 - C1, C1) * jitter

        # Recurrent alpha_proj (log_alpha)
        aw130 = sd130[f"blocks.{i}.attn.alpha_proj.weight"]  # [12, 768]
        ab130 = sd130[f"blocks.{i}.attn.alpha_proj.bias"]    # [12]
        aw240 = sd240[f"blocks.{i}.attn.alpha_proj.weight"]  # [16, 1024]
        ab240 = sd240[f"blocks.{i}.attn.alpha_proj.bias"]    # [16]

        aw240.normal_(0.0, 0.01)
        aw240[:H1, :C1] = aw130
        aw240[H1:, :C1] = aw130[:H2 - H1] + torch.randn(H2 - H1, C1) * jitter

        ab240[:H1] = ab130
        ab240[H1:] = ab130[:H2 - H1] + torch.randn(H2 - H1) * jitter

        # Selective write beta_proj
        bw130 = sd130[f"blocks.{i}.attn.beta_proj.weight"]   # [12, 768]
        bb130 = sd130[f"blocks.{i}.attn.beta_proj.bias"]     # [12]
        bw240 = sd240[f"blocks.{i}.attn.beta_proj.weight"]   # [16, 1024]
        bb240 = sd240[f"blocks.{i}.attn.beta_proj.bias"]     # [16]

        bw240.normal_(0.0, 0.01)
        bw240[:H1, :C1] = bw130
        bw240[H1:, :C1] = bw130[:H2 - H1] + torch.randn(H2 - H1, C1) * jitter

        bb240[:H1] = bb130
        bb240[H1:] = bb130[:H2 - H1] + torch.randn(H2 - H1) * jitter

        # Output RMSNorm y_norm
        yn130 = sd130[f"blocks.{i}.attn.y_norm"]  # [768] -> [12, 64]
        yn240 = sd240[f"blocks.{i}.attn.y_norm"]  # [16, 64]
        yn240[:H1] = yn130.view(H1, Dh)
        yn240[H1:] = yn130.view(H1, Dh)[:H2 - H1]

        # V0 Highway lambda (layers 1..15)
        if i > 0:
            vl130 = sd130[f"blocks.{i}.attn.v_residual_lamb"].squeeze()  # [12]
            vl240 = sd240[f"blocks.{i}.attn.v_residual_lamb"]            # [16]
            vl240[:H1] = vl130
            vl240[H1:] = vl130[:H2 - H1] + torch.randn(H2 - H1) * jitter

        # Output Gate
        gw130 = sd130[f"blocks.{i}.attn.gate.weight"]  # [768, 768]
        gw240 = sd240[f"blocks.{i}.attn.gate.weight"]  # [1024, 1024]
        gw240.normal_(0.0, 0.01)
        gw240[:C1, :C1] = gw130
        gw240[C1:, :C1] = gw130[:C2 - C1, :] + torch.randn(C2 - C1, C1) * jitter

        # Attention Out Projection: writes residual. Must be zero in new channels/heads!
        pw130 = sd130[f"blocks.{i}.attn.proj.weight"]  # [768, 768]
        pw240 = sd240[f"blocks.{i}.attn.proj.weight"]  # [1024, 1024]
        pw240.zero_()
        pw240[:C1, :C1] = pw130

        # SwiGLU MLP
        w1_130 = sd130[f"blocks.{i}.mlp.w1.weight"]  # [2048, 768]
        w3_130 = sd130[f"blocks.{i}.mlp.w3.weight"]  # [2048, 768]
        w2_130 = sd130[f"blocks.{i}.mlp.w2.weight"]  # [768, 2048]
        w1_240 = sd240[f"blocks.{i}.mlp.w1.weight"]  # [2752, 1024]
        w3_240 = sd240[f"blocks.{i}.mlp.w3.weight"]  # [2752, 1024]
        w2_240 = sd240[f"blocks.{i}.mlp.w2.weight"]  # [1024, 2752]

        hid1, hid2 = w1_130.shape[0], w1_240.shape[0]  # 2048, 2752
        w1_240.normal_(0.0, 0.01)
        w1_240[:hid1, :C1] = w1_130
        w1_240[hid1:, :C1] = w1_130[:hid2 - hid1, :] + torch.randn(hid2 - hid1, C1) * jitter

        w3_240.normal_(0.0, 0.01)
        w3_240[:hid1, :C1] = w3_130
        w3_240[hid1:, :C1] = w3_130[:hid2 - hid1, :] + torch.randn(hid2 - hid1, C1) * jitter

        w2_240.zero_()
        w2_240[:C1, :hid1] = w2_130

    model240.load_state_dict(sd240)
    print(f"[grow] Successfully transferred parameters into 240M model ({model240.num_params()/1e6:.2f}M params).")

    # Output packaging
    out_dict = {
        "model": model240.state_dict(),
        "cfg": PRESETS["parentheses-1.0-240M"],
        "step": 0,
        "tokenizer": tok_meta,
        "parent_checkpoint": str(src_p.name),
        "expansion_type": "Net2WiderNet_130M_to_240M",
    }

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_dict, out_p)
        print(f"[grow] Saved warm-started 240M checkpoint to: {out_p}")

    return model240, out_dict


def main():
    parser = argparse.ArgumentParser(description="Net2WiderNet 130M-v3 -> 240M-v4 Function-Preserving State Expansion")
    parser.add_argument(
        "--src",
        default="I:/parentheses/checkpoints/parentheses-0.9-130M-v3/step_22888_swa3_base.pt",
        help="Path to 130M-v3 source checkpoint",
    )
    parser.add_argument(
        "--out",
        default="checkpoints/parentheses-1.0-240M/step_0_grown.pt",
        help="Output path for warm-started 240M checkpoint",
    )
    parser.add_argument("--jitter", type=float, default=1e-3, help="Prior jitter for expanded heads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--check-parity", action="store_true", help="Run forward verification against 130M model")
    args = parser.parse_args()

    model240, out_dict = grow_130m_to_240m(
        src_path=args.src,
        out_path=args.out,
        jitter=args.jitter,
        seed=args.seed,
    )

    if args.check_parity:
        print("\n[grow] Running Forward Verification Check...")
        from model.backbone import Parentheses
        ckpt130 = torch.load(args.src, map_location="cpu", weights_only=False)
        model130 = Parentheses(ckpt130["cfg"])
        model130.load_state_dict(ckpt130["model"])
        model130.eval()
        model240.eval()

        dummy_tokens = torch.randint(10, 500, (1, 64))
        with torch.no_grad():
            out130, _ = model130(dummy_tokens)
            out240, _ = model240(dummy_tokens)

        # Compare top-1 predictions
        pred130 = out130.argmax(dim=-1)
        pred240 = out240.argmax(dim=-1)
        matching = (pred130 == pred240).float().mean().item()
        print(f"[grow] Top-1 token agreement on random probe: {matching * 100:.1f}%")
        max_diff = (out130 - out240).abs().max().item()
        mean_diff = (out130 - out240).abs().mean().item()
        print(f"[grow] Logits max diff: {max_diff:.4f}, mean diff: {mean_diff:.4f}")


if __name__ == "__main__":
    main()
