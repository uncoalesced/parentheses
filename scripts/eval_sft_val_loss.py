#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval_sft_val_loss.py

Evaluates masked SFT loss, unmasked cross-entropy loss, and perplexity on the
clean validation set (data/processed_sft_120m_v2/val.bin) for Parentheses 0.9-120M
(Recurrent Selective Linear Attention SSM):
  - Overall masked SFT loss (tokens with mask == 1: assistant answers, <think>, tools)
  - Overall unmasked cross-entropy loss (all tokens)
  - Per-language breakdown across all 9 languages (eng, kan, tam, tel, mal, tcy, kff, kxv, nit)
  - Per-category breakdown (structured_think, drav_chat, eng_chat, lowres_chat, freethink_cont, tool_hermes)

Usage:
  # Self-test on synthetic dummy model (CPU)
  python scripts/eval_sft_val_loss.py --self-test

  # Benchmark checkpoint on CPU (e.g. 200 documents)
  python scripts/eval_sft_val_loss.py --checkpoint checkpoints/parentheses-0.9-120M/step_9536_final.pt --device cpu --max-docs 200

  # Full evaluation on CUDA / CPU
  python scripts/eval_sft_val_loss.py --checkpoint checkpoints/parentheses-0.9-120M-sft/step_9736_final.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ModelConfig, Parentheses

VAL_DIR_DEFAULT = REPO_ROOT / "data/processed_sft_120m_v2"
OUT_DEFAULT = REPO_ROOT / "out/eval_sft_val_loss.json"

TAG_TO_LANG = {
    2: "kan",
    3: "tam",
    4: "tel",
    5: "mal",
    6: "tcy",
    7: "kff",
    8: "kxv",
    9: "nit",
    10: "eng",
}
LANG_TO_TAG = {v: k for k, v in TAG_TO_LANG.items()}

USER_TAG = 13
ASSISTANT_TAG = 14
THINK_START = 15
THINK_END = 16
TOOL_START = 17
TOOL_END = 18
EOT = 0


def identify_doc(doc: np.ndarray) -> tuple[str, str]:
    """Identifies (language, category) from document token prefixes and structure."""
    if len(doc) == 0:
        return "unknown", "unknown"

    # Continuation docs start with [lang_tag]
    if doc[0] in TAG_TO_LANG:
        lang = TAG_TO_LANG[doc[0]]
        cat = "freethink_cont"
        return lang, cat

    # Chat / Think / Tool docs start with [USER_TAG, lang_tag]
    if doc[0] == USER_TAG and len(doc) > 1 and doc[1] in TAG_TO_LANG:
        lang = TAG_TO_LANG[doc[1]]
        if THINK_START in doc:
            cat = "structured_think"
        elif TOOL_START in doc:
            cat = "tool_hermes"
        elif lang in ("tcy", "kff", "kxv", "nit"):
            cat = "lowres_chat"
        elif lang in ("kan", "tam", "tel", "mal"):
            cat = "drav_chat"
        else:
            cat = "eng_chat"
        return lang, cat

    # Fallback search in first 4 tokens for language tag
    for t in doc[:4]:
        if t in TAG_TO_LANG:
            lang = TAG_TO_LANG[t]
            cat = "structured_think" if THINK_START in doc else "conversational_qa"
            return lang, cat

    return "unknown", "unknown"


def evaluate_val_loss(
    model: Parentheses,
    docs: list[tuple[np.ndarray, np.ndarray, str, str]],
    device: str = "cpu",
    verbose_interval: int = 50,
) -> dict:
    """Evaluates masked and unmasked loss on a list of (tokens, mask, lang, cat) documents."""
    model.eval()

    total_sup_loss = 0.0
    total_sup_tokens = 0
    total_unmasked_loss = 0.0
    total_all_tokens = 0

    lang_stats = defaultdict(lambda: {"sup_loss": 0.0, "sup_tokens": 0, "all_loss": 0.0, "all_tokens": 0, "docs": 0})
    cat_stats = defaultdict(lambda: {"sup_loss": 0.0, "sup_tokens": 0, "all_loss": 0.0, "all_tokens": 0, "docs": 0})

    t0 = time.time()
    n_docs = len(docs)

    for i, (toks, mask, lang, cat) in enumerate(docs):
        if len(toks) < 2:
            continue

        x = torch.from_numpy(toks[:-1].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(toks[1:].astype(np.int64)).unsqueeze(0).to(device)
        m = torch.from_numpy(mask[1:].astype(bool)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(x)
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")

            n_all = len(loss_all)
            all_loss = loss_all.sum().item()

            n_sup = m.sum().item()
            sup_loss = loss_all[m.view(-1)].sum().item() if n_sup > 0 else 0.0

        total_sup_loss += sup_loss
        total_sup_tokens += n_sup
        total_unmasked_loss += all_loss
        total_all_tokens += n_all

        ls = lang_stats[lang]
        ls["sup_loss"] += sup_loss
        ls["sup_tokens"] += n_sup
        ls["all_loss"] += all_loss
        ls["all_tokens"] += n_all
        ls["docs"] += 1

        cs = cat_stats[cat]
        cs["sup_loss"] += sup_loss
        cs["sup_tokens"] += n_sup
        cs["all_loss"] += all_loss
        cs["all_tokens"] += n_all
        cs["docs"] += 1

        if (i + 1) % verbose_interval == 0 or (i + 1) == n_docs:
            elapsed = time.time() - t0
            cur_sup_loss = total_sup_loss / max(1, total_sup_tokens)
            cur_all_loss = total_unmasked_loss / max(1, total_all_tokens)
            print(
                f"[{i+1}/{n_docs}] {elapsed:.1f}s | "
                f"Masked SFT Loss: {cur_sup_loss:.4f} (PPL: {math.exp(min(20, cur_sup_loss)):.2f}) | "
                f"Unmasked Loss: {cur_all_loss:.4f} (PPL: {math.exp(min(20, cur_all_loss)):.2f}) | "
                f"Supervised: {total_sup_tokens:,} / {total_all_tokens:,} tokens ({total_sup_tokens/max(1,total_all_tokens):.1%})",
                flush=True,
            )

    elapsed_total = time.time() - t0
    overall_masked_loss = total_sup_loss / max(1, total_sup_tokens)
    overall_unmasked_loss = total_unmasked_loss / max(1, total_all_tokens)

    by_lang = {}
    for l, d in sorted(lang_stats.items(), key=lambda x: -x[1]["docs"]):
        l_sup = d["sup_loss"] / max(1, d["sup_tokens"])
        l_all = d["all_loss"] / max(1, d["all_tokens"])
        by_lang[l] = {
            "docs": d["docs"],
            "supervised_tokens": d["sup_tokens"],
            "total_tokens": d["all_tokens"],
            "supervised_fraction": round(d["sup_tokens"] / max(1, d["all_tokens"]), 4),
            "masked_loss": round(l_sup, 4),
            "masked_ppl": round(math.exp(min(20, l_sup)), 2),
            "unmasked_loss": round(l_all, 4),
            "unmasked_ppl": round(math.exp(min(20, l_all)), 2),
        }

    by_cat = {}
    for c, d in sorted(cat_stats.items(), key=lambda x: -x[1]["docs"]):
        c_sup = d["sup_loss"] / max(1, d["sup_tokens"])
        c_all = d["all_loss"] / max(1, d["all_tokens"])
        by_cat[c] = {
            "docs": d["docs"],
            "supervised_tokens": d["sup_tokens"],
            "total_tokens": d["all_tokens"],
            "supervised_fraction": round(d["sup_tokens"] / max(1, d["all_tokens"]), 4),
            "masked_loss": round(c_sup, 4),
            "masked_ppl": round(math.exp(min(20, c_sup)), 2),
            "unmasked_loss": round(c_all, 4),
            "unmasked_ppl": round(math.exp(min(20, c_all)), 2),
        }

    return {
        "evaluated_docs": n_docs,
        "elapsed_seconds": round(elapsed_total, 2),
        "tokens_per_second": round(total_all_tokens / max(1e-3, elapsed_total), 1),
        "total_tokens": total_all_tokens,
        "supervised_tokens": total_sup_tokens,
        "supervised_fraction": round(total_sup_tokens / max(1, total_all_tokens), 4),
        "overall": {
            "masked_sft_loss": round(overall_masked_loss, 4),
            "masked_sft_ppl": round(math.exp(min(20, overall_masked_loss)), 2),
            "unmasked_loss": round(overall_unmasked_loss, 4),
            "unmasked_ppl": round(math.exp(min(20, overall_unmasked_loss)), 2),
        },
        "by_lang": by_lang,
        "by_category": by_cat,
    }


def load_val_corpus(val_dir: Path, max_docs: int | None = None) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Loads validation binary, mask, and boundaries."""
    bin_path = val_dir / "val.bin"
    mask_path = val_dir / "val.mask.bin"
    bnd_path = val_dir / "val_boundaries.npy"

    if not bin_path.exists() or not mask_path.exists() or not bnd_path.exists():
        raise FileNotFoundError(f"Missing validation files in {val_dir}. Required: val.bin, val.mask.bin, val_boundaries.npy")

    b_data = np.fromfile(bin_path, dtype=np.uint16)
    m_data = np.fromfile(mask_path, dtype=np.uint8)
    bnd = np.append(np.load(bnd_path).astype(np.int64), len(b_data))

    total_available = len(bnd) - 1
    limit = min(total_available, max_docs) if max_docs is not None else total_available

    docs = []
    for i in range(limit):
        s = int(bnd[i])
        e = int(bnd[i + 1])
        toks = b_data[s:e]
        mask = m_data[s:e]
        lang, cat = identify_doc(toks)
        docs.append((toks, mask, lang, cat))

    return docs


def run_self_test():
    """Self-test with synthetic dummy model on CPU."""
    print("[self-test] Initializing synthetic Parentheses model on CPU...")
    cfg = ModelConfig(
        vocab_size=32768,
        block_size=512,
        n_layer=2,
        n_head=2,
        n_embd=32,
        attn_type="selective_linear",
    )
    model = Parentheses(cfg).to("cpu").eval()

    # Synthetic document 1: eng structured think
    doc1_toks = np.array([USER_TAG, LANG_TO_TAG["eng"], 100, 101, ASSISTANT_TAG, THINK_START, 200, 201, THINK_END, 300, 301, EOT], dtype=np.uint16)
    doc1_mask = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)

    # Synthetic document 2: kan chat
    doc2_toks = np.array([USER_TAG, LANG_TO_TAG["kan"], 500, 501, ASSISTANT_TAG, 600, 601, EOT], dtype=np.uint16)
    doc2_mask = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.uint8)

    # Synthetic document 3: tel continuation
    doc3_toks = np.array([LANG_TO_TAG["tel"], 700, 701, 800, 801, EOT], dtype=np.uint16)
    doc3_mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)

    dummy_docs = [
        (doc1_toks, doc1_mask, "eng", "structured_think"),
        (doc2_toks, doc2_mask, "kan", "drav_chat"),
        (doc3_toks, doc3_mask, "tel", "freethink_cont"),
    ]

    print("[self-test] Running evaluation pass on 3 dummy documents...")
    res = evaluate_val_loss(model, dummy_docs, device="cpu", verbose_interval=1)

    assert res["evaluated_docs"] == 3
    assert res["overall"]["masked_sft_loss"] > 0
    assert res["overall"]["unmasked_loss"] > 0
    assert "eng" in res["by_lang"]
    assert "kan" in res["by_lang"]
    assert "tel" in res["by_lang"]
    assert "structured_think" in res["by_category"]
    assert "drav_chat" in res["by_category"]
    assert "freethink_cont" in res["by_category"]
    print("[self-test] PASS: All self-test assertions succeeded.")


def main():
    p = argparse.ArgumentParser(description="Evaluate validation loss and perplexity on Parentheses SFT v2 dataset.")
    p.add_argument("--checkpoint", default="checkpoints/parentheses-0.9-120M/step_9536_final.pt")
    p.add_argument("--val-dir", default=str(VAL_DIR_DEFAULT))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--verbose-interval", type=int, default=25)
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return

    val_dir = Path(args.val_dir)
    print(f"[eval_sft_val_loss] Loading validation data from {val_dir} (max_docs={args.max_docs})...", flush=True)
    docs = load_val_corpus(val_dir, max_docs=args.max_docs)
    print(f"[eval_sft_val_loss] Loaded {len(docs):,} validation documents.", flush=True)

    print(f"[eval_sft_val_loss] Loading checkpoint {args.checkpoint} on {args.device}...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "unknown")
    print(f"[eval_sft_val_loss] Model loaded successfully (step={step}, params={sum(p.numel() for p in model.parameters()):,}).", flush=True)

    print(f"[eval_sft_val_loss] Starting evaluation on {len(docs):,} documents...", flush=True)
    results = evaluate_val_loss(model, docs, device=args.device, verbose_interval=args.verbose_interval)
    results["checkpoint"] = str(args.checkpoint)
    results["step"] = step
    results["val_dir"] = str(val_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval_sft_val_loss] Wrote complete results to {out_path}", flush=True)

    # Print clean summary table
    print("\n" + "=" * 78)
    print(f"  SFT v2 Validation Loss Summary | Checkpoint: {Path(args.checkpoint).name} (step={step})")
    print("=" * 78)
    print(f"  Documents Evaluated:  {results['evaluated_docs']:,}")
    print(f"  Tokens Evaluated:     {results['total_tokens']:,} (Supervised: {results['supervised_tokens']:,} = {results['supervised_fraction']:.1%})")
    print(f"  Overall Masked Loss:  {results['overall']['masked_sft_loss']:.4f}  (PPL: {results['overall']['masked_sft_ppl']:.2f})")
    print(f"  Overall Unmasked CE:  {results['overall']['unmasked_loss']:.4f}  (PPL: {results['overall']['unmasked_ppl']:.2f})")
    print("-" * 78)
    print(f"  {'Language':<10} | {'Docs':<6} | {'Supervised Tokens':<18} | {'Masked Loss':<12} | {'PPL':<8}")
    print("-" * 78)
    for lang, ld in results["by_lang"].items():
        print(f"  {lang:<10} | {ld['docs']:<6} | {ld['supervised_tokens']:<18,} | {ld['masked_loss']:<12.4f} | {ld['masked_ppl']:<8.2f}")
    print("-" * 78)
    print(f"  {'Category':<22} | {'Docs':<6} | {'Supervised Tokens':<18} | {'Masked Loss':<12} | {'PPL':<8}")
    print("-" * 78)
    for cat, cd in results["by_category"].items():
        print(f"  {cat:<22} | {cd['docs']:<6} | {cd['supervised_tokens']:<18,} | {cd['masked_loss']:<12.4f} | {cd['masked_ppl']:<8.2f}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
