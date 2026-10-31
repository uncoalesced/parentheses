# Engineered by uncoalesced
"""
Parentheses 1.0-240M (slab_v4) - Session 1 Telemetry & Validation Digest Plotter
Parses JSONL/log telemetry, overlays periodic 20-window val.bin evaluations against
train loss EMA, audits WDDM throughput & SSD gates, and outputs Markdown + PNG.
"""

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Clean Session 1 Net2WiderNet Reference Baselines (Repaired train.bin + val.bin)
BASELINE_TRAIN_LOSS = 1.8274
BASELINE_VAL_LOSS = 3.2477
BASELINE_BETA_WRITE = 0.9784
BASELINE_V0_MIX = 0.5075
GRAD_CLIP_THRESHOLD = 1.0
GRAD_SPIKE_ALERT = 5.0


def first_valid(record: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in record and record[k] is not None:
            try:
                return float(record[k])
            except (ValueError, TypeError):
                continue
    return None


def parse_jsonl(jsonl_path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    if not jsonl_path.exists():
        return records

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = first_valid(data, ["step", "iter", "global_step"])
            loss = first_valid(data, ["loss", "train_loss", "ce_loss"])
            val_loss = first_valid(data, ["val_loss", "eval_loss"])

            if step is None:
                continue

            entry: Dict[str, float] = {
                "step": int(step),
                "loss": loss if loss is not None else float("nan"),
                "lr": first_valid(data, ["lr", "learning_rate"]) or 0.0,
                "grad_norm": first_valid(data, ["grad_norm", "gnorm", "grad_l2"]) or 0.0,
                "beta_write": first_valid(data, ["beta_write", "beta", "ssd_beta"]) or float("nan"),
                "v0_mix": first_valid(data, ["v0_mix", "v0", "lambda_v"]) or float("nan"),
                "tok_per_sec": first_valid(data, ["tok_per_sec", "tps", "tokens_per_sec"]) or float("nan"),
                "vram_gb": first_valid(data, ["vram_gb", "mem_gb", "reserved_gb"]) or float("nan"),
                "val_loss": val_loss if val_loss is not None else float("nan"),
                "val_ppl": first_valid(data, ["val_ppl", "eval_ppl"]) or float("nan"),
            }
            records.append(entry)

    # Merge records sharing the same step (e.g., if val_loss is logged on a separate JSONL line)
    merged: Dict[int, Dict[str, float]] = {}
    for r in records:
        s = r["step"]
        if s not in merged:
            merged[s] = r
        else:
            for k, v in r.items():
                if k != "step" and not (isinstance(v, float) and math.isnan(v)):
                    merged[s][k] = v

    return [merged[k] for k in sorted(merged.keys()) if not math.isnan(merged[k]["loss"])]


def compute_ema(values: List[float], alpha: float = 0.05) -> List[float]:
    ema: List[float] = []
    curr: Optional[float] = None
    for v in values:
        if math.isnan(v) or math.isinf(v):
            ema.append(curr if curr is not None else float("nan"))
            continue
        curr = v if curr is None else (alpha * v + (1.0 - alpha) * curr)
        ema.append(curr)
    return ema


def safe_stats(values: List[float]) -> Dict[str, float]:
    clean = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if not clean:
        return {k: float("nan") for k in ["first", "last", "min", "max", "mean", "median"]}
    return {
        "first": clean[0],
        "last": clean[-1],
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
    }


def render_plots(records: List[Dict[str, float]], out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    loss_ema = compute_ema(losses, alpha=0.05)
    gnorms = [r["grad_norm"] for r in records]
    betas = [r["beta_write"] for r in records]
    v0s = [r["v0_mix"] for r in records]

    val_steps = [r["step"] for r in records if not math.isnan(r["val_loss"])]
    val_losses = [r["val_loss"] for r in records if not math.isnan(r["val_loss"])]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=160)
    fig.suptitle(
        "Parentheses 1.0-240M (slab_v4) - Clean Session 1 Telemetry & Validation Digest",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Train Loss EMA + Periodic val.bin Loss
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, color="#4a90e2", alpha=0.30, linewidth=0.9, label="Raw Train Loss")
    ax1.plot(steps, loss_ema, color="#00e5ff", linewidth=2.0, label="Train Loss EMA (alpha=0.05)")
    if val_steps:
        ax1.plot(val_steps, val_losses, color="#ff5252", marker="o", markersize=5, linewidth=1.8, label="val.bin Loss (20-win)")
    ax1.axhline(BASELINE_VAL_LOSS, color="#ff8a80", linestyle="--", linewidth=1.0, label=f"Step 1 Val Baseline ({BASELINE_VAL_LOSS:.4f})")
    ax1.set_title("Cross-Entropy Loss: Train vs. True val.bin (4096 ctx)")
    ax1.set_xlabel("Optimizer Step (65,536 tok/step)")
    ax1.set_ylabel("Loss (nats)")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Gradient Norm
    ax2 = axes[0, 1]
    ax2.plot(steps, gnorms, color="#ffb74d", alpha=0.65, linewidth=1.0, label="Grad L2 Norm")
    ax2.plot(steps, compute_ema(gnorms, alpha=0.05), color="#ff9100", linewidth=1.8, label="Grad Norm EMA")
    ax2.axhline(GRAD_CLIP_THRESHOLD, color="#69f0ae", linestyle="--", linewidth=1.0, label=f"Clip Threshold ({GRAD_CLIP_THRESHOLD})")
    ax2.set_title("Gradient L2 Norm Stability")
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("L2 Norm")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Panel 3: SSD Write Gate (beta_write)
    ax3 = axes[1, 0]
    ax3.plot(steps, betas, color="#b388ff", alpha=0.6, linewidth=1.0, label="beta_write (16 heads mean)")
    ax3.plot(steps, compute_ema(betas, alpha=0.05), color="#7c4dff", linewidth=1.8, label="beta_write EMA")
    ax3.axhline(BASELINE_BETA_WRITE, color="#ff5252", linestyle="--", linewidth=1.0, label=f"Init Prior ({BASELINE_BETA_WRITE:.4f})")
    ax3.set_title("SSD Selective Write Gate (beta_write)")
    ax3.set_xlabel("Optimizer Step")
    ax3.set_ylabel("Gate Activation")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="best", fontsize=8.5)

    # Panel 4: V0 Residual Highway (v0_mix)
    ax4 = axes[1, 1]
    ax4.plot(steps, v0s, color="#69f0ae", alpha=0.6, linewidth=1.0, label="v0_mix (lambda_v)")
    ax4.plot(steps, compute_ema(v0s, alpha=0.05), color="#00e676", linewidth=1.8, label="v0_mix EMA")
    ax4.axhline(BASELINE_V0_MIX, color="#ff5252", linestyle="--", linewidth=1.0, label=f"Init Prior ({BASELINE_V0_MIX:.4f})")
    ax4.set_title("V0 Highway Residual Mixing (v0_mix)")
    ax4.set_xlabel("Optimizer Step")
    ax4.set_ylabel("Mixing Ratio")
    ax4.grid(True, alpha=0.2)
    ax4.legend(loc="best", fontsize=8.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.text(0.99, 0.005, "Engineered by uncoalesced", ha="right", fontsize=8, alpha=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    return True


def build_digest_report(
    records: List[Dict[str, float]],
    source_path: Path,
    out_md: Path,
    out_png: Path,
    plot_created: bool,
) -> str:
    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    loss_ema = compute_ema(losses, alpha=0.05)
    gnorms = [r["grad_norm"] for r in records]
    betas = [r["beta_write"] for r in records]
    v0s = [r["v0_mix"] for r in records]
    tps_list = [r["tok_per_sec"] for r in records]
    vram_list = [r["vram_gb"] for r in records]
    val_recs = [r for r in records if not math.isnan(r["val_loss"])]

    s_loss = safe_stats(losses)
    s_ema = safe_stats(loss_ema)
    s_gnorm = safe_stats(gnorms)
    s_beta = safe_stats(betas)
    s_v0 = safe_stats(v0s)
    s_tps = safe_stats(tps_list)
    s_vram = safe_stats(vram_list)
    s_val = safe_stats([r["val_loss"] for r in val_recs])

    total_steps = len(records)
    min_step, max_step = steps[0], steps[-1]
    tokens_processed = (max_step - min_step + 1) * 65536

    nan_steps = [r["step"] for r in records if math.isnan(r["loss"]) or math.isinf(r["loss"])]
    grad_spikes = [r["step"] for r in records if r["grad_norm"] > GRAD_SPIKE_ALERT]

    val_rows = []
    for vr in val_recs:
        vppl = math.exp(min(vr["val_loss"], 20.0))
        delta_v = vr["val_loss"] - BASELINE_VAL_LOSS
        val_rows.append(
            f"| `{vr['step']}` | `{vr['val_loss']:.4f}` | `{vppl:.2f}` | `{delta_v:+.4f}` | `{vr['lr']:.2e}` |"
        )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    report = f"""# Parentheses 1.0-240M (`slab_v4`) - Clean Session 1 Digest

- **Generated:** `{now_str}`
- **Telemetry Source:** `{source_path}`
- **Steps Logged:** `{min_step}` -> `{max_step}` (`{total_steps:,}` steps | `{tokens_processed / 1e6:.2f}M` BPE tokens)
- **Steady-State Reserved VRAM:** `{s_vram['median']:.3f} GB` (Max: `{s_vram['max']:.3f} GB`)
- **Median Throughput:** `{s_tps['median']:.0f} tok/s`
- **Diagnostic Plot:** `{out_png if plot_created else "Skipped"}`

---

## 1. Summary Metrics (`Repaired train.bin: 807,927,808 tokens`)

| Metric | Step 1 Baseline | Initial (`step {min_step}`) | Latest (`step {max_step}`) | Min | Max | Mean | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Loss (Raw)** | `{BASELINE_TRAIN_LOSS:.4f}` | `{s_loss['first']:.4f}` | `{s_loss['last']:.4f}` | `{s_loss['min']:.4f}` | `{s_loss['max']:.4f}` | `{s_loss['mean']:.4f}` | `{s_loss['last'] - s_loss['first']:+.4f}` |
| **Train Loss EMA** | `{BASELINE_TRAIN_LOSS:.4f}` | `{s_ema['first']:.4f}` | `{s_ema['last']:.4f}` | `{s_ema['min']:.4f}` | `{s_ema['max']:.4f}` | `{s_ema['mean']:.4f}` | `{s_ema['last'] - s_ema['first']:+.4f}` |
| **`val.bin` Loss (20-win)** | `{BASELINE_VAL_LOSS:.4f}` | `{s_val['first']:.4f}` | `{s_val['last']:.4f}` | `{s_val['min']:.4f}` | `{s_val['max']:.4f}` | `{s_val['mean']:.4f}` | `{s_val['last'] - s_val['first']:+.4f}` |
| **Grad L2 Norm** | `0.3968` | `{s_gnorm['first']:.4f}` | `{s_gnorm['last']:.4f}` | `{s_gnorm['min']:.4f}` | `{s_gnorm['max']:.4f}` | `{s_gnorm['mean']:.4f}` | `{s_gnorm['last'] - s_gnorm['first']:+.4f}` |
| **`beta_write` (SSD)** | `{BASELINE_BETA_WRITE:.4f}` | `{s_beta['first']:.4f}` | `{s_beta['last']:.4f}` | `{s_beta['min']:.4f}` | `{s_beta['max']:.4f}` | `{s_beta['mean']:.4f}` | `{s_beta['last'] - s_beta['first']:+.4f}` |
| **`v0_mix` (Highway)** | `{BASELINE_V0_MIX:.4f}` | `{s_v0['first']:.4f}` | `{s_v0['last']:.4f}` | `{s_v0['min']:.4f}` | `{s_v0['max']:.4f}` | `{s_v0['mean']:.4f}` | `{s_v0['last'] - s_v0['first']:+.4f}` |

---

## 2. True `val.bin` Perplexity Checkpoints (`81,920` Tokens / Evaluation)

| Step | `val_loss` (nats) | `val_ppl` | Delta vs Step 1 (`3.2477`) | Learning Rate |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(val_rows) if val_rows else "| *Pending next 250-step boundary* | - | - | - | - |"}

---
*Engineered by uncoalesced*
"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path(r"E:\Parentheses\parentheses-0.9\out\session1_240m_metrics.jsonl"))
    parser.add_argument("--out-md", type=Path, default=Path(r"E:\Parentheses\parentheses-0.9\out\session1_morning_digest.md"))
    parser.add_argument("--out-png", type=Path, default=Path(r"E:\Parentheses\parentheses-0.9\out\session1_morning_digest.png"))
    args = parser.parse_args()

    records = parse_jsonl(args.jsonl)
    if not records:
        print(f"[ALERT] No step records found in {args.jsonl}")
        return 1

    plot_ok = render_plots(records, args.out_png)
    report = build_digest_report(records, args.jsonl, args.out_md, args.out_png, plot_ok)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
