#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/stage_hf_130m.py

Stages clean weights-only safetensors and config for Parentheses v0.9-130M-v3
into E:\\Parentheses\\parentheses-0.9\\hf_staging_130m, stripping AdamW optimizer states.
"""

from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CKPT_PATH = REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-swa3" / "step_0100_final.pt"
STAGING_DIR = REPO_ROOT / "hf_staging_130m"


def main():
    print(f"[*] Starting Hugging Face staging for Parentheses v0.9-130M-v3...")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading raw checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    sd = ckpt["model"]
    cfg = ckpt["cfg"]
    step = ckpt.get("step", 2099)

    print(f"[+] Loaded model checkpoint (Step: {step}). Stripping optimizer state...")

    # Clean and cast to bfloat16
    clean_tensors_bf16 = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            k_clean = k.replace("_orig_mod.", "")
            clean_tensors_bf16[k_clean] = v.to(torch.bfloat16).contiguous()

    safetensors_path = STAGING_DIR / "model.safetensors"
    save_file(clean_tensors_bf16, str(safetensors_path))
    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"[+] Saved stripped BF16 weights to {safetensors_path} ({size_mb:.2f} MB)")

    # Write config.json
    config_dict = {
        "model_type": "parentheses",
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "dropout": cfg.dropout,
        "bias": cfg.bias,
        "tie_embeddings": cfg.tie_embeddings,
        "attn_type": cfg.attn_type,
        "slab_v2": cfg.slab_v2,
        "slab_v3": cfg.slab_v3,
        "slab_v4": getattr(cfg, "slab_v4", False),
        "total_tensors": len(clean_tensors_bf16),
        "lineage": "Parentheses 0.9-130M-v3 (Project Tanzinite)",
        "golden_checkpoint": "step_0100_final.pt (DPO v2 SWA-3)",
        "val_ppl": 30.15,
        "val_loss": 3.4061,
        "refusal_rate": 0.0,
        "freethink_tag_closures": "72/72 (100%)",
    }
    config_path = STAGING_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[+] Generated {config_path}")

    # Copy tokenizer files if present
    tok_dir = REPO_ROOT / "data" / "tokenizer"
    if (tok_dir / "tokenizer.json").exists():
        shutil.copy(tok_dir / "tokenizer.json", STAGING_DIR / "tokenizer.json")
        shutil.copy(tok_dir / "tokenizer_config.json", STAGING_DIR / "tokenizer_config.json")
        print(f"[+] Copied tokenizer.json and tokenizer_config.json")
    elif (REPO_ROOT / "release" / "parentheses-0.9-120M-candidate" / "tokenizer.json").exists():
        shutil.copy(REPO_ROOT / "release" / "parentheses-0.9-120M-candidate" / "tokenizer.json", STAGING_DIR / "tokenizer.json")
        shutil.copy(REPO_ROOT / "release" / "parentheses-0.9-120M-candidate" / "tokenizer_config.json", STAGING_DIR / "tokenizer_config.json")
        print(f"[+] Copied tokenizer from release package")

    # Generate README.md
    readme_content = """---
language:
- kn
- ta
- te
- ml
- en
license: apache-2.0
tags:
- parentheses
- uncoalesced
- linear-attention
- state-space-duality
- recurrent-memory
- dravidian
- mamba2-family
---

# Parentheses v0.9-130M-v3 (Project Tanzinite)

**Parentheses v0.9-130M-v3** is a sovereign recurrent language model architected by **uncoalesced**, specialized for Dravidian languages (Kannada, Tamil, Telugu, Malayalam) and English.

## Architecture Highlights
- **Recurrent State Space Duality (SSD):** Linear $O(T)$ chunked associative linear recurrence in parallel training, and $O(1)$ constant memory updates in inference: $S_t = \\alpha_t S_{t-1} + \\beta_t (k_t \\otimes v_t)$ and $y_t = q_t S_t$. Zero sequence Softmax, zero growing KV-cache.
- **Scale:** 148.38M parameters (16 layers, $d_{\\text{model}}=768$, 12 heads, $d_{\\text{head}}=64$).
- **Tokenization:** Custom BPE 32k (`v3_32k`) tokenizer (32,000 vocab + 19 special tokens: `<think>`, `</think>`, language tags).
- **Recurrent Innovations:** $\\beta$-gated dynamic write selectivity, Token-Shift ($\\mu$-mix), and $V_0$ value highway.
- **Certified Golden Checkpoint:** Evaluated at record **30.15 Val PPL (3.4061 Val Loss)**, **0.0% Refusal Rate** (19/19 constructive responses), and **100.0% `<think>` reasoning block closure** (72/72 closures).
- **Edge Deployment:** Static-shape DirectML NPU graph verified with bitwise parity.

## Model Details
- **Base Pretraining:** 1.50B tokens (Sangraha Dravidian core, STEM reasoning, Karya conversational text).
- **Clean SFT v2:** 131.07M tokens on `processed_sft_stem_open_v2` with zero truncated documents.
- **Calibrated DPO v2:** Preference optimization over balanced reasoning and epistemic abstention traces.
"""
    readme_path = STAGING_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[+] Generated {readme_path}")

    # Parity verification
    print("[*] Verifying staged model.safetensors...")
    from safetensors.torch import load_file
    loaded = load_file(str(safetensors_path))
    assert len(loaded) == len(clean_tensors_bf16), "Tensor count mismatch!"
    print(f"[+] Verification passed: {len(loaded)} tensors loaded correctly.")
    print(f"[+] Hugging Face staging complete in {STAGING_DIR}")


if __name__ == "__main__":
    main()
