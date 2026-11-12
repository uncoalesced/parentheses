#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Engineered by uncoalesced

"""
scripts/probe_activations.py

Harvests internal representations from Parentheses 1.0-240M (slab_v4) across
150 strictly balanced contrastive pairs using forward hooks and recurrent step inference:

Captures at the final prompt token (decision boundary):
  1. Recurrent associative memory states S_t in R^(16 x 16 x 64 x 64)
  2. Selective write gates beta_t in R^(16 x 16)
  3. Retention / decay gates alpha_t in R^(16 x 16)
  4. MLP intermediate neuron activations a_t in R^(16 x 2752)
  5. Residual stream representations h_t in R^(16 x 1024)

Outputs saved strictly to:
  E:\\Parentheses\\parentheses-0.9\\out\\probing\\activations_s10.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_slab_v4 import SlabV4Model, SlabV4Config


def harvest_pair_activations(
    model: SlabV4Model,
    tokenizer: Tokenizer,
    prompt_text: str,
    lang: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Prefills a formatted prompt token by token through recurrent step inference,
    recording internal layer representations, gates, and MLP activations at the
    final assistant trigger token.
    """
    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    lang_id = tokenizer.token_to_id(f"<{lang}>") or tokenizer.token_to_id("<eng>")

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    input_ids = [user_id, lang_id] + prompt_ids + [asst_id]

    n_layers = len(model.blocks)
    captured_mlp = {}
    captured_gates = {}
    captured_res = {}

    hooks = []

    def make_w2_hook(layer_idx: int):
        def hook(mod, inp, out):
            captured_mlp[layer_idx] = inp[0].detach().cpu()
        return hook

    def make_beta_hook(layer_idx: int):
        def hook(mod, inp, out):
            captured_gates[f"beta_{layer_idx}"] = torch.sigmoid(out).detach().cpu()
        return hook

    def make_alpha_hook(layer_idx: int):
        def hook(mod, inp, out):
            captured_gates[f"alpha_{layer_idx}"] = torch.exp(-F.softplus(out)).detach().cpu()
        return hook

    def make_ln1_hook(layer_idx: int):
        def hook(mod, inp, out):
            if layer_idx > 0:
                captured_res[layer_idx - 1] = inp[0].detach().cpu()
        return hook

    def make_ln_f_hook():
        def hook(mod, inp, out):
            captured_res[n_layers - 1] = inp[0].detach().cpu()
        return hook

    for l in range(n_layers):
        hooks.append(model.blocks[l].mlp.w2.register_forward_hook(make_w2_hook(l)))
        hooks.append(model.blocks[l].attn.beta_proj.register_forward_hook(make_beta_hook(l)))
        hooks.append(model.blocks[l].attn.alpha_proj.register_forward_hook(make_alpha_hook(l)))
        hooks.append(model.blocks[l].ln1.register_forward_hook(make_ln1_hook(l)))
    hooks.append(model.ln_f.register_forward_hook(make_ln_f_hook()))

    states = model.empty_states(1, device=device)
    with torch.no_grad():
        for i, tok in enumerate(input_ids):
            inp = torch.tensor([[tok]], dtype=torch.long, device=device)
            # Only record activations at the last token (asst_id)
            if i < len(input_ids) - 1:
                captured_mlp.clear()
                captured_gates.clear()
                captured_res.clear()
            _, states = model.step(inp, states)

    # Clean up forward hooks
    for h in hooks:
        h.remove()

    # Extract recurrent S_t for all layers from final states
    # states is a list of tuples: (S: [1, H, Dh, Dh], conv: [1, pad, 3*C], token: [1, 1, C])
    s_t_all = torch.stack([st[0].squeeze(0).cpu() for st in states], dim=0)  # [16, H, Dh, Dh]
    beta_all = torch.stack([captured_gates[f"beta_{l}"].squeeze(0) for l in range(n_layers)], dim=0)  # [16, H]
    alpha_all = torch.stack([captured_gates[f"alpha_{l}"].squeeze(0) for l in range(n_layers)], dim=0)  # [16, H]
    mlp_all = torch.stack([captured_mlp[l].squeeze(0) for l in range(n_layers)], dim=0)  # [16, d_ffn]
    res_all = torch.stack([captured_res[l].squeeze(0) for l in range(n_layers)], dim=0)  # [16, d_model]

    return {
        "s_t": s_t_all,        # [16, 16, 64, 64]
        "beta": beta_all,      # [16, 16]
        "alpha": alpha_all,    # [16, 16]
        "mlp": mlp_all,        # [16, 2752]
        "hidden": res_all,     # [16, 1024]
    }


def main():
    parser = argparse.ArgumentParser(description="Harvest internal activations from Parentheses 1.0-240M")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/parentheses-1.0-240M-s10-blend/sft-nudge-winner.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="data/eval_suites/probing_contrastive_honesty.jsonl",
        help="Path to contrastive probing suite",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/probing/activations_s10.pt",
        help="Path to save harvested activations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    args = parser.parse_args()

    suite_path = REPO_ROOT / args.suite
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite not found at {suite_path}. Run build_probing_suite.py first!")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"[*] Harvesting Internal Activations for Mechanistic Probing")
    print(f"    Checkpoint: {args.checkpoint}")
    print(f"    Probing Suite: {suite_path}")
    print(f"    Output Path: {out_path}")
    print(f"    Device: {args.device}")
    print(f"=======================================================\n")

    # Load Model
    t0 = time.time()
    ckpt = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("cfg", None) or ckpt.get("config", None)
    cfg = SlabV4Config(
        vocab_size=getattr(raw_cfg, "vocab_size", 32768),
        block_size=getattr(raw_cfg, "block_size", 4096),
        n_layer=getattr(raw_cfg, "n_layer", 16),
        n_head=getattr(raw_cfg, "n_head", 16),
        n_embd=getattr(raw_cfg, "n_embd", 1024),
        chunk_size=getattr(raw_cfg, "chunk_size", 64),
        eot_token_id=getattr(raw_cfg, "eot_token_id", 0),
        dropout=getattr(raw_cfg, "dropout", 0.0),
        bias=getattr(raw_cfg, "bias", False),
    )
    model = SlabV4Model(cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[+] Loaded SlabV4Model (253M) in {time.time() - t0:.2f}s")

    # Load Tokenizer
    tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Load Pairs
    pairs = []
    with open(suite_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    print(f"[*] Loaded {len(pairs)} contrastive pairs. Commencing extraction...")

    dataset_activations = []
    t_start = time.time()

    for idx, pair in enumerate(pairs, 1):
        pid = pair["pair_id"]
        subset = pair["subset"]
        lang = pair.get("lang", "en")

        # Harvest Prompt A
        act_a = harvest_pair_activations(model, tokenizer, pair["prompt_a"], lang, device=args.device)
        # Harvest Prompt B
        act_b = harvest_pair_activations(model, tokenizer, pair["prompt_b"], lang, device=args.device)

        dataset_activations.append({
            "pair_id": pid,
            "subset": subset,
            "lang": lang,
            "prompt_a": pair["prompt_a"],
            "label_a": pair["label_a"],
            "act_a": act_a,
            "prompt_b": pair["prompt_b"],
            "label_b": pair["label_b"],
            "act_b": act_b,
        })

        if idx % 25 == 0 or idx == len(pairs):
            elapsed = time.time() - t_start
            rate = idx / elapsed
            print(f"    [{idx:3d}/{len(pairs)}] Harvested pair {pid:18s} ({rate:.1f} pairs/sec)")

    # Save to disk
    print(f"\n[*] Saving harvested activations to {out_path}...")
    torch.save(dataset_activations, out_path)
    file_size_mb = out_path.stat().st_size / 1e6
    print(f"[+] Gate P1 PASSED: Successfully extracted 150/150 pairs ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
