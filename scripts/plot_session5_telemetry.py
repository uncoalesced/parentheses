#!/usr/bin/env python3
"""
Statistical Plotting Engine for Parentheses 1.0-240M Session 5.
Generates publication-quality graphs for training loss, gradient norms,
recurrent state space dynamics, and hardware telemetry (GPU temp, power, VRAM).
Outputs to out/stats/ and copies to vault 'Training Stats/'.
"""

import os
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out"
STATS_DIR = OUT_DIR / "stats"
VAULT_STATS_DIR = Path(r"C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\Training Stats")

STATS_DIR.mkdir(parents=True, exist_ok=True)
if VAULT_STATS_DIR.exists():
    VAULT_STATS_DIR.mkdir(parents=True, exist_ok=True)


def plot_session5():
    # 1. First run exporter to refresh CSVs
    import export_session5_stats
    export_session5_stats.export_stats()

    metrics_csv = OUT_DIR / "session5_metrics.csv"
    telemetry_csv = OUT_DIR / "session5_telemetry.csv"

    if not metrics_csv.exists() or os.path.getsize(metrics_csv) == 0:
        print("[!] No metrics CSV found to plot.")
        return

    df_m = pd.read_csv(metrics_csv)
    df_t = pd.read_csv(telemetry_csv) if telemetry_csv.exists() else None

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # --- Plot 1: Loss & Gradient Norms ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)

    # Loss curve with rolling average
    steps = df_m["step"]
    loss = df_m["loss"]
    rolling_loss = loss.rolling(window=15, min_periods=1).mean()

    ax1.plot(steps, loss, color="#94a3b8", alpha=0.45, linewidth=1.0, label="Raw Batch Loss")
    ax1.plot(steps, rolling_loss, color="#2563eb", linewidth=2.0, label="15-Step Rolling Mean")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold", color="#1e293b")
    ax1.set_title("Parentheses 1.0-240M: Session 5 Convergence & Loss Trajectory (Steps 8,001 - 10,000)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Annotate min loss
    min_idx = loss.idxmin()
    min_step = steps.iloc[min_idx]
    min_val = loss.iloc[min_idx]
    ax1.annotate(
        f"Min: {min_val:.4f}\n(Step {min_step})",
        xy=(min_step, min_val),
        xytext=(min_step, min_val + 0.25),
        arrowprops=dict(facecolor="#dc2626", shrink=0.05, width=1.2, headwidth=6),
        fontsize=9, fontweight="bold", color="#dc2626",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="#dc2626", facecolor="#fef2f2", alpha=0.9)
    )

    # Gradient Norm
    gnorm = df_m["grad_norm"]
    rolling_gnorm = gnorm.rolling(window=15, min_periods=1).mean()
    ax2.plot(steps, gnorm, color="#cbd5e1", alpha=0.45, linewidth=1.0, label="Raw Grad Norm")
    ax2.plot(steps, rolling_gnorm, color="#059669", linewidth=2.0, label="15-Step Rolling Mean")
    ax2.axhline(2.0, color="#ef4444", linestyle=":", linewidth=1.5, label="Tripwire Limit (2.0)")
    ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold", color="#1e293b")
    ax2.set_ylabel("Gradient L2 Norm", fontsize=11, fontweight="bold", color="#1e293b")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    loss_png = STATS_DIR / "session5_loss_and_gradients.png"
    plt.savefig(loss_png, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {loss_png}")

    # --- Plot 2: Recurrent Dynamics & Throughput ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)

    # Recurrent SSD Gating
    ax1.plot(steps, df_m["beta_write"], color="#7c3aed", linewidth=2.0, label=r"$\sigma(b_\beta)$ (Recurrent Write Gate)")
    ax1.plot(steps, df_m["v0_mix"], color="#d97706", linewidth=2.0, label=r"$\sigma(\lambda_v)$ ($V_0$ Value Highway Mix)")
    ax1.set_ylabel("Gating Magnitude", fontsize=11, fontweight="bold", color="#1e293b")
    ax1.set_title("Parentheses 1.0-240M: Session 5 Recurrent SSD Gating & Hardware Throughput", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="center right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Throughput (tok/s)
    tps = df_m["tok_per_sec"]
    ax2.plot(steps, tps, color="#0284c7", alpha=0.4, linewidth=1.0)
    ax2.plot(steps, tps.rolling(15, min_periods=1).mean(), color="#0284c7", linewidth=2.0, label="Throughput (tok/s)")
    ax2.axhline(2850, color="#64748b", linestyle="--", linewidth=1.2, label="Nominal Target (2,850 tok/s)")
    ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold", color="#1e293b")
    ax2.set_ylabel("Tokens / Second", fontsize=11, fontweight="bold", color="#1e293b")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    dynamics_png = STATS_DIR / "session5_recurrent_dynamics.png"
    plt.savefig(dynamics_png, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {dynamics_png}")

    # --- Plot 3: Hardware & Thermal Telemetry ---
    if df_t is not None and not df_t.empty:
        df_t["dt"] = pd.to_datetime(df_t["timestamp"])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True, dpi=300)

        # Temperature
        ax1.plot(df_t["dt"], df_t["temp_c"], color="#e11d48", linewidth=1.8, label="GPU Temperature (°C)")
        ax1.axhline(82, color="#ef4444", linestyle=":", linewidth=1.5, label="Thermal Limit (82°C)")
        ax1.set_ylabel("Temperature (°C)", fontsize=10, fontweight="bold")
        ax1.set_title("NVIDIA RTX 5050 Laptop (8GB) Hardware Telemetry (Session 5)", fontsize=13, fontweight="bold", pad=12)
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Power Draw
        ax2.plot(df_t["dt"], df_t["power_w"], color="#f59e0b", linewidth=1.8, label="GPU Power Draw (Watts)")
        ax2.set_ylabel("Power (W)", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper left", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # VRAM
        ax3.plot(df_t["dt"], df_t["vram_used_mb"] / 1024.0, color="#6366f1", linewidth=1.8, label="VRAM Allocated (GB)")
        ax3.axhline(8.0, color="#64748b", linestyle="--", linewidth=1.2, label="Physical VRAM Cap (8.0 GB)")
        ax3.set_ylabel("VRAM (GB)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Time (IST)", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper left", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        hardware_png = STATS_DIR / "session5_hardware_telemetry.png"
        plt.savefig(hardware_png, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {hardware_png}")

        # Mirror to Vault Training Stats
        if VAULT_STATS_DIR.exists():
            for p in [loss_png, dynamics_png, hardware_png]:
                dst = VAULT_STATS_DIR / p.name
                shutil.copy2(p, dst)
                print(f"[+] Mirrored to Vault: {dst}")


if __name__ == "__main__":
    plot_session5()
