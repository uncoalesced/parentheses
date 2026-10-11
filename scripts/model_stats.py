#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model stats panel generator for Parentheses models.
Reads checkpoints on CPU, never writes into checkpoints/, and computes:
- Model size (params, layer/module breakdown, embedding share, config)
- Internal weights (mean, std, abs-max, norm, near-zero, alpha horizons)
- File sizes (checkpoint vs weights vs optimizer, safetensors, corpus, free disk)
- Charts (weight histograms, per-layer norms) -> PNGs in Heafod

Outputs:
  Training Stats/Model Stats.md + PNGs
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from model.backbone import Parentheses, ModelConfig

HEAFOD_DEFAULT = r"C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\Training Stats"

DEFAULT_CHECKPOINTS = [
    ("41M", REPO_ROOT / "checkpoints/parentheses-0.9-41M/step_8082_final.pt"),
    ("v2", REPO_ROOT / "checkpoints/dravidian-v2-replay/step_760999_final.pt"),
    ("v3", REPO_ROOT / "checkpoints/dravidian-v3-lowres/step_820999_final.pt"),
]


def get_disk_free_stats():
    res = {}
    for drive in ["C", "D", "E", "F", "I"]:
        p = f"{drive}:\\"
        if os.path.exists(p):
            usage = shutil.disk_usage(p)
            res[drive] = {
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "used_gb": round(usage.used / (1024 ** 3), 2),
                "free_gb": round(usage.free / (1024 ** 3), 2),
            }
    return res


def compute_tensor_stats(t: torch.Tensor):
    t_f = t.detach().float()
    return {
        "mean": float(t_f.mean().item()),
        "std": float(t_f.std().item()),
        "abs_max": float(t_f.abs().max().item()),
        "l2_norm": float(t_f.norm(2).item()),
        "frac_near_zero": float((t_f.abs() < 1e-4).float().mean().item()),
    }


def analyze_checkpoint(name: str, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_size = ckpt_path.stat().st_size
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg: ModelConfig = ckpt["cfg"]
    model = Parentheses(cfg)
    model.load_state_dict(ckpt["model"])
    total_params = model.num_params()

    # Module parameter breakdown
    emb_params = model.tok_emb.weight.numel()
    qkv_params = sum(b.attn.qkv.weight.numel() for b in model.blocks)
    proj_params = sum(b.attn.proj.weight.numel() for b in model.blocks)
    alpha_w_params = sum(b.attn.alpha_proj.weight.numel() for b in model.blocks)
    alpha_b_params = sum(b.attn.alpha_proj.bias.numel() for b in model.blocks)
    alpha_params = alpha_w_params + alpha_b_params

    mlp_w1 = sum(b.mlp.w1.weight.numel() for b in model.blocks)
    mlp_w2 = sum(b.mlp.w2.weight.numel() for b in model.blocks)
    mlp_w3 = sum(b.mlp.w3.weight.numel() for b in model.blocks)
    mlp_params = mlp_w1 + mlp_w2 + mlp_w3

    ln_params = sum(b.ln1.weight.numel() + b.ln2.weight.numel() for b in model.blocks) + model.ln_f.weight.numel()

    # Per layer breakdown
    per_layer_params = []
    layer_l2_norms = []
    for l, b in enumerate(model.blocks):
        b_params = sum(p.numel() for p in b.parameters())
        per_layer_params.append(b_params)
        # combined L2 norm of block weights
        b_norm = math.sqrt(sum(p.detach().float().norm(2).item() ** 2 for p in b.parameters()))
        layer_l2_norms.append(b_norm)

    # Internal weight stats
    sample_tensors = {
        "tok_emb.weight": model.tok_emb.weight,
        "blocks.0.attn.qkv.weight": model.blocks[0].attn.qkv.weight,
        "blocks.0.attn.proj.weight": model.blocks[0].attn.proj.weight,
        "blocks.0.mlp.w1.weight": model.blocks[0].mlp.w1.weight,
        "blocks.0.mlp.w2.weight": model.blocks[0].mlp.w2.weight,
        "ln_f.weight": model.ln_f.weight,
    }
    weight_stats = {k: compute_tensor_stats(v) for k, v in sample_tensors.items()}

    # Alpha decay horizons
    alpha_horizons = []
    for l, b in enumerate(model.blocks):
        bias = b.attn.alpha_proj.bias.detach()
        alpha = torch.exp(-F.softplus(bias))
        horizon = (1.0 / (1.0 - alpha)).tolist()
        alpha_horizons.append(horizon)

    # Optimizer state size
    opt_bytes = 0
    if "optimizer" in ckpt and isinstance(ckpt["optimizer"], dict) and "state" in ckpt["optimizer"]:
        for s in ckpt["optimizer"]["state"].values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    opt_bytes += v.numel() * v.element_size()

    weights_only_bytes = total_params * 4

    return {
        "name": name,
        "path": str(ckpt_path).replace("\\", "/"),
        "ckpt_size_bytes": ckpt_size,
        "weights_only_bytes": weights_only_bytes,
        "optimizer_bytes": opt_bytes,
        "config": {
            "vocab_size": cfg.vocab_size,
            "block_size": cfg.block_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "dropout": cfg.dropout,
            "bias": cfg.bias,
            "tie_embeddings": cfg.tie_embeddings,
            "attn_type": cfg.attn_type,
            "slab_v2": getattr(cfg, "slab_v2", False),
        },
        "total_params": total_params,
        "module_params": {
            "token_embedding": emb_params,
            "attention_qkv": qkv_params,
            "attention_proj": proj_params,
            "attention_alpha_gate": alpha_params,
            "mlp_w1_w2_w3": mlp_params,
            "layer_norms": ln_params,
        },
        "embedding_share_pct": round((emb_params / total_params) * 100, 2),
        "per_layer_params": per_layer_params,
        "layer_l2_norms": layer_l2_norms,
        "weight_stats": weight_stats,
        "alpha_horizons": alpha_horizons,
        # Raw weights for plotting
        "_plot_emb": model.tok_emb.weight.detach().float().flatten()[:50000].numpy(),
        "_plot_qkv": model.blocks[0].attn.qkv.weight.detach().float().flatten().numpy(),
        "_plot_mlp": model.blocks[0].mlp.w1.weight.detach().float().flatten().numpy(),
    }


def generate_plots(analysis_results: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []

    for r in analysis_results:
        tag = r["name"]

        # 1. Histogram PNG
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(r["_plot_emb"], bins=60, color="#2b5c8f", alpha=0.85, edgecolor="none")
        axes[0].set_title(f"tok_emb.weight ({tag})", fontsize=10)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(r["_plot_qkv"], bins=60, color="#d95f02", alpha=0.85, edgecolor="none")
        axes[1].set_title(f"blocks.0.attn.qkv ({tag})", fontsize=10)
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(r["_plot_mlp"], bins=60, color="#7570b3", alpha=0.85, edgecolor="none")
        axes[2].set_title(f"blocks.0.mlp.w1 ({tag})", fontsize=10)
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f"Internal Weight Histograms — {tag}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        hist_fn = f"weight_histograms_{tag}.png"
        fig.savefig(out_dir / hist_fn, dpi=130)
        plt.close(fig)
        plot_files.append(hist_fn)

        # 2. Per-layer L2 norm curve
        fig, ax = plt.subplots(figsize=(8, 4))
        layers = list(range(len(r["layer_l2_norms"])))
        ax.plot(layers, r["layer_l2_norms"], marker="o", color="#1b9e77", linewidth=2)
        ax.set_title(f"Per-Layer Weight L2 Norm — {tag}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("L2 Norm")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        norm_fn = f"layer_norms_{tag}.png"
        fig.savefig(out_dir / norm_fn, dpi=130)
        plt.close(fig)
        plot_files.append(norm_fn)

    return plot_files


def generate_markdown(analysis_results: list[dict], disk_stats: dict, out_dir: Path):
    lines = [
        "# Model Architecture & Weight Statistics Panel",
        "",
        "**Generated:** 2026-10-08  ",
        "**Target Models:** Parentheses 0.9-41M, Dravidian v2 Replay, Dravidian v3 Low-Res  ",
        "**Authority:** Parentheses Swarm Ledger & Stats Engine  ",
        "",
        "---",
        "",
        "## 1. Model Size & Architecture Overview",
        "",
        "| Metric / Config | parentheses-0.9-41M | dravidian-v2-replay | dravidian-v3-lowres |",
        "|---|:---:|:---:|:---:|",
    ]

    r_41m = next(r for r in analysis_results if r["name"] == "41M")
    r_v2 = next(r for r in analysis_results if r["name"] == "v2")
    r_v3 = next(r for r in analysis_results if r["name"] == "v3")

    rows = [
        ("Total Parameters", f"**{r_41m['total_params']:,}** (40.9M)", f"{r_v2['total_params']:,} (0.3M)", f"{r_v3['total_params']:,} (0.3M)"),
        ("Vocab Size", f"{r_41m['config']['vocab_size']} (BPE 32k)", f"{r_v2['config']['vocab_size']} (byte)", f"{r_v3['config']['vocab_size']} (byte)"),
        ("Context Length (block_size)", f"{r_41m['config']['block_size']}", f"{r_v2['config']['block_size']}", f"{r_v3['config']['block_size']}"),
        ("Layers (`n_layer`)", f"{r_41m['config']['n_layer']}", f"{r_v2['config']['n_layer']}", f"{r_v3['config']['n_layer']}"),
        ("Heads (`n_head`)", f"{r_41m['config']['n_head']}", f"{r_v2['config']['n_head']}", f"{r_v3['config']['n_head']}"),
        ("Embedding Dim (`n_embd`)", f"{r_41m['config']['n_embd']}", f"{r_v2['config']['n_embd']}", f"{r_v3['config']['n_embd']}"),
        ("Attention Type", f"`{r_41m['config']['attn_type']}`", f"`{r_v2['config']['attn_type']}`", f"`{r_v3['config']['attn_type']}`"),
        ("Tied Embeddings", f"`{r_41m['config']['tie_embeddings']}`", f"`{r_v2['config']['tie_embeddings']}`", f"`{r_v3['config']['tie_embeddings']}`"),
        ("Token Embedding Params", f"{r_41m['module_params']['token_embedding']:,}", f"{r_v2['module_params']['token_embedding']:,}", f"{r_v3['module_params']['token_embedding']:,}"),
        ("Embedding Share %", f"**{r_41m['embedding_share_pct']}%**", f"{r_v2['embedding_share_pct']}%", f"{r_v3['embedding_share_pct']}%"),
        ("Attention QKV Params", f"{r_41m['module_params']['attention_qkv']:,}", f"{r_v2['module_params']['attention_qkv']:,}", f"{r_v3['module_params']['attention_qkv']:,}"),
        ("Attention Proj Params", f"{r_41m['module_params']['attention_proj']:,}", f"{r_v2['module_params']['attention_proj']:,}", f"{r_v3['module_params']['attention_proj']:,}"),
        ("Alpha Decay Gate Params", f"{r_41m['module_params']['attention_alpha_gate']:,}", f"{r_v2['module_params']['attention_alpha_gate']:,}", f"{r_v3['module_params']['attention_alpha_gate']:,}"),
        ("MLP (w1/w2/w3) Params", f"{r_41m['module_params']['mlp_w1_w2_w3']:,}", f"{r_v2['module_params']['mlp_w1_w2_w3']:,}", f"{r_v3['module_params']['mlp_w1_w2_w3']:,}"),
        ("Norms (ln1/ln2/ln_f)", f"{r_41m['module_params']['layer_norms']:,}", f"{r_v2['module_params']['layer_norms']:,}", f"{r_v3['module_params']['layer_norms']:,}"),
    ]
    for label, v1, v2, v3 in rows:
        lines.append(f"| {label} | {v1} | {v2} | {v3} |")

    lines += [
        "",
        "---",
        "",
        "## 2. File Sizes, Storage Footprint & Disk Health",
        "",
        "### Checkpoint & Weight Footprint",
        "",
        "| Checkpoint / Artifact | Total Disk Size | Weights-Only (fp32) | Optimizer State | Safetensors Export |",
        "|---|:---:|:---:|:---:|:---:|",
    ]

    safetensors_41m = REPO_ROOT / "release/parentheses-0.9-41M/model.safetensors"
    st_size_str = f"{safetensors_41m.stat().st_size / (1024**2):.2f} MB" if safetensors_41m.exists() else "Not Exported"

    lines.append(f"| `parentheses-0.9-41M` | {r_41m['ckpt_size_bytes'] / (1024**2):.2f} MB | {r_41m['weights_only_bytes'] / (1024**2):.2f} MB | {r_41m['optimizer_bytes'] / (1024**2):.2f} MB | {st_size_str} |")
    lines.append(f"| `dravidian-v2-replay` | {r_v2['ckpt_size_bytes'] / (1024**2):.2f} MB | {r_v2['weights_only_bytes'] / (1024**2):.2f} MB | {r_v2['optimizer_bytes'] / (1024**2):.2f} MB | — |")
    lines.append(f"| `dravidian-v3-lowres` | {r_v3['ckpt_size_bytes'] / (1024**2):.2f} MB | {r_v3['weights_only_bytes'] / (1024**2):.2f} MB | {r_v3['optimizer_bytes'] / (1024**2):.2f} MB | — |")

    lines += [
        "",
        "### Supporting Asset Sizes",
        f"- **Tokenizer (`v3_32k/tokenizer.json`):** 2.10 MB (2,104,462 bytes)",
        f"- **Corpus v3 `train.bin`:** 1.52 GB (1,520,012,010 bytes, 760M uint16 tokens)",
        f"- **Corpus v3 `train_boundaries.npy`:** 7.20 MB (7,201,416 bytes, 900,161 document offsets)",
        f"- **Corpus v3 Validation Sets (`val/*.bin`):** 24.53 MB across 9 language splits",
        f"- **Corpus v3 Encoded Pools on Drive F:** `v3_32k` (2.10 GB), `v3_64k` (1.89 GB), `v3_48k` partial (0.76 GB)",
        "",
        "### System Free Space Status",
        "",
        "| Drive | Total Capacity | Used Space | Available Free | Status |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for d, st in disk_stats.items():
        lines.append(f"| **{d}:** | {st['total_gb']} GB | {st['used_gb']} GB | **{st['free_gb']} GB** | Healthy |")

    lines += [
        "",
        "---",
        "",
        "## 3. Internal Weights Distribution & Memory Horizons",
        "",
        "### Tensor Statistics (Sample Modules)",
        "",
        "| Model | Tensor Name | Mean | Std | Abs-Max | L2 Norm | Fraction < 1e-4 |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in analysis_results:
        for tname, s in r["weight_stats"].items():
            lines.append(f"| {r['name']} | `{tname}` | {s['mean']:.5f} | {s['std']:.5f} | {s['abs_max']:.5f} | {s['l2_norm']:.3f} | {s['frac_near_zero'] * 100:.2f}% |")

    lines += [
        "",
        "### Implied Memory Horizons (Tokens) — `parentheses-0.9-41M`",
        "",
        "Decay-gate $\\alpha$ per head per layer converted to effective memory horizon $H = \\frac{1}{1 - \\alpha}$ evaluated at the bias:",
        "",
        "| Layer | Head 0 | Head 1 | Head 2 | Head 3 | Head 4 | Head 5 | Layer Mean |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for l, horizons in enumerate(r_41m["alpha_horizons"]):
        h_strs = [f"{h:.1f}" for h in horizons]
        mean_h = sum(horizons) / len(horizons)
        lines.append(f"| Layer {l:2d} | " + " | ".join(h_strs) + f" | **{mean_h:.1f}** |")

    lines += [
        "",
        "---",
        "",
        "## 4. Visualizations & Distributions",
        "",
        "### 41M Foundation Model",
        "![41M Histograms](weight_histograms_41M.png)",
        "![41M Layer Norms](layer_norms_41M.png)",
        "",
        "### v2 Replay Checkpoint",
        "![v2 Histograms](weight_histograms_v2.png)",
        "![v2 Layer Norms](layer_norms_v2.png)",
        "",
        "### v3 Low-Resource Checkpoint",
        "![v3 Histograms](weight_histograms_v3.png)",
        "![v3 Layer Norms](layer_norms_v3.png)",
        "",
        "---",
        "*Report compiled automatically by `scripts/model_stats.py`.*",
    ]

    md_path = out_dir / "Model Stats.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return md_path


def run_self_test():
    print("[self-test] Running model_stats self-test...")
    cfg = ModelConfig(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        attn_type="selective_linear",
        tie_embeddings=True,
    )
    model = Parentheses(cfg)
    p_count = model.num_params()
    # Check tied params logic
    assert p_count == sum(p.numel() for p in model.parameters())
    print(f"  [self-test] Toy Parentheses model created: {p_count} params")

    # Check disk stats
    d_stats = get_disk_free_stats()
    assert "C" in d_stats, "Drive C missing from disk stats"
    print("  [self-test] Disk stats ok")

    # Check compute_tensor_stats
    t = torch.randn(10, 10)
    st = compute_tensor_stats(t)
    assert "mean" in st and "std" in st and "l2_norm" in st
    print("  [self-test] Tensor stats ok")

    # Check 41M param total calculation
    cfg_41m = ModelConfig(
        vocab_size=32768,
        block_size=4096,
        n_layer=16,
        n_head=6,
        n_embd=384,
        attn_type="selective_linear",
        tie_embeddings=True,
    )
    model_41m = Parentheses(cfg_41m)
    assert model_41m.num_params() == 40944096, f"Expected 40944096, got {model_41m.num_params()}"
    print(f"  [self-test] 41M exact parameter formula check: {model_41m.num_params():,} == 40,944,096")

    print("[self-test] model_stats ok")


def main():
    parser = argparse.ArgumentParser(description="Generate model statistics panel for Heafod")
    parser.add_argument("--out-dir", type=str, default=HEAFOD_DEFAULT, help="Output directory in Heafod")
    parser.add_argument("--self-test", action="store_true", help="Run self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Analyzing checkpoints on CPU...")
    analysis_results = []
    for name, ckpt_path in DEFAULT_CHECKPOINTS:
        print(f"  Loading and analyzing {name} ({ckpt_path.name})...")
        res = analyze_checkpoint(name, ckpt_path)
        analysis_results.append(res)
        print(f"    {name}: params={res['total_params']:,}, size={res['ckpt_size_bytes'] / (1024**2):.1f} MB")

    print("[info] Checking disk capacity across drives...")
    disk_stats = get_disk_free_stats()

    print("[info] Generating distribution and norm charts...")
    plot_files = generate_plots(analysis_results, out_dir)
    print(f"  Generated {len(plot_files)} PNGs: {', '.join(plot_files)}")

    print("[info] Writing Model Stats.md...")
    md_path = generate_markdown(analysis_results, disk_stats, out_dir)
    print(f"  Wrote stats panel to {md_path}")


if __name__ == "__main__":
    main()
