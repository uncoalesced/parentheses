"""Export script for Parentheses 0.9-41M release artifacts.

Usage:
    python scripts/export_release.py

Creates release/parentheses-0.9-41M/ containing:
- model.safetensors (fp32 weights-only)
- model.bf16.safetensors (bf16 weights-only)
- model.pt (fp32 weights-only fallback)
- model.bf16.pt (bf16 weights-only fallback)
- config.json
- tokenizer.json
- tokenizer_config.json
- load_example.py
"""

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.backbone import Parentheses
from model.config import ModelConfig


def main():
    out_dir = REPO_ROOT / "release" / "parentheses-0.9-41M"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = REPO_ROOT / "checkpoints" / "parentheses-0.9-41M" / "step_8082_final.pt"
    print(f"[export] Loading source checkpoint: {ckpt_path} (CPU)...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg: ModelConfig = ckpt["cfg"]
    sd = ckpt["model"]

    # 1. config.json
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
        "params": sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor)),
    }
    cfg_file = out_dir / "config.json"
    cfg_file.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    print(f"[export] Wrote {cfg_file}")

    # 2. tokenizer.json
    tok_src = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
    tok_dst = out_dir / "tokenizer.json"
    shutil.copyfile(tok_src, tok_dst)
    print(f"[export] Copied tokenizer.json to {tok_dst}")

    # 3. tokenizer_config.json
    tok_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": cfg.block_size,
        "clean_up_tokenization_spaces": False,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": None,
        "added_tokens_decoder": {
            "0": {"content": "<|endoftext|>", "special": True},
            "1": {"content": "<|pad|>", "special": True},
            "2": {"content": "<kan>", "special": True},
            "3": {"content": "<tam>", "special": True},
            "4": {"content": "<tel>", "special": True},
            "5": {"content": "<mal>", "special": True},
            "6": {"content": "<tcy>", "special": True},
            "7": {"content": "<kff>", "special": True},
            "8": {"content": "<kxv>", "special": True},
            "9": {"content": "<nit>", "special": True},
            "10": {"content": "<eng>", "special": True},
            "11": {"content": "<esg>", "special": True},
            "12": {"content": "<|system|>", "special": True},
            "13": {"content": "<|user|>", "special": True},
            "14": {"content": "<|assistant|>", "special": True},
            "15": {"content": "<think>", "special": True},
            "16": {"content": "</think>", "special": True},
            "17": {"content": "<tool_call>", "special": True},
            "18": {"content": "</tool_call>", "special": True},
        },
    }
    tok_cfg_file = out_dir / "tokenizer_config.json"
    tok_cfg_file.write_text(json.dumps(tok_config, indent=2), encoding="utf-8")
    print(f"[export] Wrote {tok_cfg_file}")

    # 4. Instantiate model and load state dict
    model = Parentheses(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 5. Save safetensors (fp32 and bf16)
    try:
        import safetensors.torch

        has_safetensors = True
    except ImportError:
        has_safetensors = False

    if has_safetensors:
        print("[export] Saving model.safetensors (fp32)...")
        safetensors.torch.save_model(model, str(out_dir / "model.safetensors"))

        print("[export] Saving model.bf16.safetensors (bf16)...")
        model_bf16 = copy.deepcopy(model).to(torch.bfloat16)
        safetensors.torch.save_model(model_bf16, str(out_dir / "model.bf16.safetensors"))
    else:
        print("[export] WARNING: safetensors package not in current python environment!")

    # 6. Also save standard PyTorch weights-only checkpoints for universal fallback
    print("[export] Saving model.pt (fp32 weights-only)...")
    torch.save(model.state_dict(), str(out_dir / "model.pt"))
    print("[export] Saving model.bf16.pt (bf16 weights-only)...")
    model_bf16_sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save(model_bf16_sd, str(out_dir / "model.bf16.pt"))

    # 7. Verification: Gate test on 4 fixed prompts
    print("[export] Running gate verification on 4 fixed prompts...")
    prompts = [
        torch.tensor([[2]]),  # <kan>
        torch.tensor([[3]]),  # <tam>
        torch.tensor([[4]]),  # <tel>
        torch.tensor([[5]]),  # <mal>
    ]

    reloaded_model = Parentheses(cfg)
    if has_safetensors and (out_dir / "model.safetensors").exists():
        safetensors.torch.load_model(reloaded_model, str(out_dir / "model.safetensors"))
    else:
        reloaded_model.load_state_dict(torch.load(str(out_dir / "model.pt"), map_location="cpu", weights_only=True))
    reloaded_model.eval()

    diffs = []
    for i, p in enumerate(prompts):
        with torch.no_grad():
            orig_logits, _ = model(p)
            reloaded_logits, _ = reloaded_model(p)
            diff = (orig_logits - reloaded_logits).abs().max().item()
            diffs.append(diff)
            print(f"  Prompt {i+1} ({p.tolist()}): max abs diff = {diff:.6e}")

    max_diff = max(diffs)
    print(f"[export] Gate result: max abs diff overall = {max_diff:.6e} (must be < 1e-5)")
    assert max_diff < 1e-5, f"Gate failed! max diff {max_diff} >= 1e-5"
    print("[export] GATE PASSED!")


if __name__ == "__main__":
    main()
