import os, sys, json, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("E:/Parentheses/parentheses-0.9")
OUT_DIR = ROOT / "out/swarm/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI, DejaVu Sans, Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def parse_log(path):
    steps, losses, lrs, elapseds = [], [], [], []
    pattern = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([\d\.]+)\s+\|\s+lr\s+([\d\.e\-\+]+)\s+\|\s+([\d\.]+)s elapsed")
    if not path.exists():
        return steps, losses, lrs, elapseds
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                elapseds.append(float(m.group(4)))
    return np.array(steps), np.array(losses), np.array(lrs), np.array(elapseds)

def smooth(vals, alpha=0.15):
    if len(vals) == 0:
        return vals
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return np.array(out)

def plot_training_loss():
    s1, l1, lr1, e1 = parse_log(ROOT / "out/parentheses-0.9-120M-run1-lr3e-4.log")
    s2, l2, lr2, e2 = parse_log(ROOT / "out/parentheses-0.9-120M.log")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Run 1 (Old)
    if len(s1) > 0:
        ax1.plot(s1, l1, color='#ef4444', alpha=0.25, label='Run 1 (Old lr=3e-4, warmup 300) raw')
        ax1.plot(s1, smooth(l1, 0.1), color='#dc2626', linewidth=2.0, label='Run 1 (Old) EMA smoothed (Stalled / Diverged)')
        ax1.axvline(x=s1[-1], color='#dc2626', linestyle=':', alpha=0.7)
        ax1.text(s1[-1] - 10, 3.45, f'Run 1 Killed @ Step {s1[-1]}\n(Representation Shock)', 
                 color='#b91c1c', fontsize=9, ha='right', weight='bold')

    # Run 2 (Active Session 1)
    if len(s2) > 0:
        ax1.plot(s2, l2, color='#3b82f6', alpha=0.22, label='Run 2 (lr=1e-4, warmup 500) raw')
        ax1.plot(s2, smooth(l2, 0.1), color='#1d4ed8', linewidth=2.2, label='Run 2 Session 1 EMA smoothed (Converged: 3.078)')
        ax1.axvline(x=s2[-1], color='#16a34a', linestyle='--', alpha=0.8)
        ax1.text(s2[-1] + 15, 3.12, f'Session 1 Clean Exit\n@ Step {s2[-1]} (16:25 IST)\nFinal Loss: 3.0783', 
                 color='#15803d', fontsize=9, ha='left', weight='bold')

    # Warmup marker
    ax1.axvline(x=500, color='#64748b', linestyle='-.', alpha=0.6)
    ax1.text(510, 3.48, 'Warmup Complete\n(Step 500, full lr=1e-4)', color='#475569', fontsize=8.5)

    ax1.set_ylabel('Cross-Entropy Training Loss', fontsize=11, weight='bold')
    ax1.set_title('Parentheses 0.9-120M — Session 1 Convergence Dynamics: Run 1 (lr=3e-4) vs Run 2 (lr=1e-4)', fontsize=13, weight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax1.set_ylim(2.85, 3.55)

    # Learning Rate subplot
    if len(s2) > 0:
        ax2.plot(s2, lr2, color='#0891b2', linewidth=2, label='Run 2 LR (1e-4 target)')
    if len(s1) > 0:
        ax2.plot(s1, lr1, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.7, label='Run 1 LR (3e-4 target)')
    ax2.set_ylabel('Learning Rate', fontsize=10, weight='bold')
    ax2.set_xlabel('Training Steps', fontsize=11, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='center right', frameon=True, framealpha=0.9)
    ax2.set_xlim(0, 1950)

    plt.tight_layout()
    chart1 = OUT_DIR / "training_loss_run1_vs_run2.png"
    plt.savefig(chart1)
    plt.close()
    print(f"Generated: {chart1}")

def plot_validation_trajectory():
    # Load all npu_eval export.json
    eval_dir = ROOT / "out/npu_eval"
    checkpoints = [
        ("Grown (Step 0)", "120M_step_grown", 0, "baseline"),
        ("Old Step 400", "120M_step_400", 400, "old"),
        ("Old Step 800", "120M_step_800", 800, "old"),
        ("Run 2 Step 150", "120M_lr1e-4_step_150", 150, "run2"),
        ("Run 2 Step 300", "120M_lr1e-4_step_300", 300, "run2"),
        ("Run 2 Step 600", "120M_lr1e-4_step_600", 600, "run2"),
        ("Probe 600 Cooled", "120M_probe600_cooled", 700, "probe"),
        ("Run 2 Step 1200", "120M_lr1e-4_step_1200", 1200, "run2"),
        ("Run 2 Step 1500", "120M_lr1e-4_step_1500", 1500, "run2"),
        ("Run 2 Step 1601", "120M_lr1e-4_step_1601", 1601, "run2"),
    ]

    data = {}
    for label, folder, step, kind in checkpoints:
        exp_path = eval_dir / folder / "export.json"
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            data[label] = {
                "step": step,
                "kind": kind,
                "losses": d.get("torch_fp32_loss", {})
            }

    if not data:
        print("No validation data found")
        return

    # Languages to plot
    core_langs = [("kan", "Kannada", "#2563eb"), 
                  ("tel", "Telugu", "#7c3aed"), 
                  ("tam", "Tamil", "#db2777"), 
                  ("mal", "Malayalam", "#059669")]
    low_res = [("tcy", "Tulu", "#d97706"), 
               ("kff", "Koya", "#ea580c"), 
               ("kxv", "Kuvi", "#16a34a"), 
               ("nit", "Kolami", "#0284c7"), 
               ("eng", "English", "#475569")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Core Dravidian
    run2_keys = [k for k, v in data.items() if v["kind"] in ("baseline", "run2")]
    run2_keys.sort(key=lambda k: data[k]["step"])
    steps_r2 = [data[k]["step"] for k in run2_keys]

    for code, name, color in core_langs:
        losses_r2 = [data[k]["losses"].get(code, np.nan) for k in run2_keys]
        ax1.plot(steps_r2, losses_r2, marker='o', linewidth=2, label=f"{name} ({code})", color=color)

    ax1.set_title("Core Dravidian Languages — Held-Out Val Loss", fontsize=12, weight='bold')
    ax1.set_xlabel("Training Steps", fontsize=10, weight='bold')
    ax1.set_ylabel("Validation Loss (torch fp32 cross-entropy)", fontsize=10, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='best', frameon=True)

    # Low-Resource Dravidian + English
    for code, name, color in low_res:
        losses_r2 = [data[k]["losses"].get(code, np.nan) for k in run2_keys]
        ax2.plot(steps_r2, losses_r2, marker='s', linewidth=2, label=f"{name} ({code})", color=color)

    ax2.set_title("Low-Resource Dravidian & English — Held-Out Val Loss", fontsize=12, weight='bold')
    ax2.set_xlabel("Training Steps", fontsize=10, weight='bold')
    ax2.set_ylabel("Validation Loss (torch fp32 cross-entropy)", fontsize=10, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='best', frameon=True)

    plt.suptitle("Parentheses 0.9-120M — Multi-Language Generalization Trajectory Across Session 1", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    chart2 = OUT_DIR / "validation_loss_by_language.png"
    plt.savefig(chart2)
    plt.close()
    print(f"Generated: {chart2}")

def plot_deltas_vs_grown():
    eval_dir = ROOT / "out/npu_eval"
    grown_file = eval_dir / "120M_step_grown" / "export.json"
    s1200_file = eval_dir / "120M_lr1e-4_step_1200" / "export.json"
    s1500_file = eval_dir / "120M_lr1e-4_step_1500" / "export.json"
    s1601_file = eval_dir / "120M_lr1e-4_step_1601" / "export.json"

    target_file = s1601_file if s1601_file.exists() else (s1500_file if s1500_file.exists() else s1200_file)
    target_name = "Step 1601 (Final)" if target_file == s1601_file else ("Step 1500" if target_file == s1500_file else "Step 1200")

    if not grown_file.exists() or not target_file.exists():
        print("Missing grown or target file for deltas")
        return

    with open(grown_file, "r", encoding="utf-8") as f:
        g = json.load(f)["torch_fp32_loss"]
    with open(target_file, "r", encoding="utf-8") as f:
        t = json.load(f)["torch_fp32_loss"]

    langs = ["kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"]
    names = ["Kannada", "Tamil", "Telugu", "Malayalam", "Tulu", "Koya", "Kuvi", "Kolami", "English"]
    
    deltas = [t.get(l, 0) - g.get(l, 0) for l in langs]
    colors = ['#16a34a' if d < 0 else '#dc2626' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, deltas, color=colors, width=0.55, edgecolor='#334155', linewidth=0.8)

    ax.axhline(0, color='#334155', linewidth=1)
    ax.set_ylabel("Delta vs Grown Baseline (Δ Loss = Step_Final - Step_0)", fontsize=10, weight='bold')
    ax.set_title(f"Parentheses 0.9-120M: Held-Out Loss Delta vs Grown Baseline ({target_name})\n(Green = Improved / Lower Loss; Negative is better)", fontsize=12, weight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    for bar, d in zip(bars, deltas):
        yval = bar.get_height()
        va = 'bottom' if yval >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (0.005 if yval >= 0 else -0.015), 
                f"{d:+.4f}", ha='center', va=va, fontsize=9, weight='bold')

    plt.tight_layout()
    chart3 = OUT_DIR / "language_improvement_delta.png"
    plt.savefig(chart3)
    plt.close()
    print(f"Generated: {chart3}")

def plot_speed_and_tokens():
    s2, l2, lr2, e2 = parse_log(ROOT / "out/parentheses-0.9-120M.log")
    if len(s2) < 2:
        return

    # batch_size=8, grad_accum=4, seq_len=4096 => 131,072 tokens/step
    tokens_per_step = 8 * 4 * 4096
    tokens = s2 * tokens_per_step / 1e6 # in Millions
    hours = e2 / 3600.0

    # calculate rolling steps per second
    dt = np.diff(e2)
    ds = np.diff(s2)
    st_per_s = np.where(dt > 0, ds / dt, 0.058)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(hours, tokens, color='#2563eb', linewidth=2.2)
    ax1.fill_between(hours, tokens, color='#3b82f6', alpha=0.15)
    ax1.set_xlabel('Wall-Clock GPU Training (Hours)', fontsize=10, weight='bold')
    ax1.set_ylabel('Cumulative Tokens Trained (Millions)', fontsize=10, weight='bold')
    ax1.set_title('Token Ingestion Velocity (RTX 5050 Laptop GPU)', fontsize=12, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.text(hours[-1]*0.6, tokens[-1]*0.4, f"Final Ingestion: {tokens[-1]:.1f}M tokens\n1,601 steps in {hours[-1]:.1f}h", 
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#93c5fd", lw=1))

    ax2.plot(s2[1:], st_per_s, color='#059669', alpha=0.3, label='Raw step/s')
    ax2.plot(s2[1:], smooth(st_per_s, 0.15), color='#047857', linewidth=2, label='Smoothed step/s (~0.059 st/s)')
    ax2.set_xlabel('Training Steps', fontsize=10, weight='bold')
    ax2.set_ylabel('Throughput (steps / second)', fontsize=10, weight='bold')
    ax2.set_title('Throughput Stability (Selective Linear Attention + Per-Block AC)', fontsize=12, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0.04, 0.08)

    plt.suptitle("Parentheses 0.9-120M Session 1 — Hardware Execution & Ingestion Profiles", fontsize=13, weight='bold', y=0.98)
    plt.tight_layout()
    chart4 = OUT_DIR / "throughput_and_tokens.png"
    plt.savefig(chart4)
    plt.close()
    print(f"Generated: {chart4}")

def main():
    plot_training_loss()
    plot_validation_trajectory()
    plot_deltas_vs_grown()
    plot_speed_and_tokens()
    print("All charts generated successfully.")

if __name__ == "__main__":
    main()
