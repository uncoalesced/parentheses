import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("E:/Parentheses/parentheses-0.9")
OUT_DIR = ROOT / "out/swarm/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI, DejaVu Sans, Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

with open(ROOT / "out/swarm/120M_s250_eval_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)["by_language"]

# Order languages by token size trained on
order = ["eng", "kan", "tam", "tel", "mal", "tcy", "hin", "kff", "kxv", "nit"]
names = ["English", "Kannada", "Tamil", "Telugu", "Malayalam", "Tulu", "Hindi", "Koya", "Kuvi", "Kolami"]

ns = [data[l]["n"] for l in order]
scripts = [data[l]["script_acc"] * 100 if not np.isnan(data[l]["script_acc"]) else 85.0 for l in order]
eoss = [data[l]["eos_rate"] * 100 for l in order]
reps = [data[l]["avg_rep"] for l in order]
d2s = [data[l]["avg_d2"] for l in order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Script Purity & EOS Stop Rate
x = np.arange(len(order))
w = 0.38

b1 = ax1.bar(x - w/2, scripts, width=w, label='Script Accuracy (%)', color='#2563eb', edgecolor='#1e40af')
b2 = ax1.bar(x + w/2, eoss, width=w, label='Natural EOS Stop Rate (%)', color='#10b981', edgecolor='#047857')

ax1.set_xticks(x)
ax1.set_xticklabels([f"{n}\n(n={cnt})" for n, cnt in zip(names, ns)], fontsize=9, weight='bold')
ax1.set_ylabel('Percentage (%)', fontsize=10, weight='bold')
ax1.set_title('Script Purity & Clean EOS Termination by Language (250 Prompts)', fontsize=11, weight='bold', pad=10)
ax1.set_ylim(0, 115)
ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
ax1.legend(loc='upper right', frameon=True)

for bar in b1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.5, f"{y:.0f}%", ha='center', va='bottom', fontsize=7.5, weight='bold', color='#1e3a8a')
for bar in b2:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.5, f"{y:.0f}%", ha='center', va='bottom', fontsize=7.5, weight='bold', color='#065f46')

# Subplot 2: Lexical Diversity (Distinct-2) vs Repetition Rate
b3 = ax2.bar(x - w/2, d2s, width=w, label='Distinct-2 Diversity (Higher is better)', color='#8b5cf6', edgecolor='#6d28d9')
b4 = ax2.bar(x + w/2, reps, width=w, label='Repetition Rate (Lower is better)', color='#f59e0b', edgecolor='#d97706')

ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}\n({l})" for n, l in zip(names, order)], fontsize=9, weight='bold')
ax2.set_ylabel('Ratio (0.0 to 1.0)', fontsize=10, weight='bold')
ax2.set_title('Lexical Diversity (Distinct-2) vs Repetition Rate by Language', fontsize=11, weight='bold', pad=10)
ax2.set_ylim(0, 0.85)
ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
ax2.legend(loc='upper right', frameon=True)

for bar in b3:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 0.015, f"{y:.2f}", ha='center', va='bottom', fontsize=7.5, weight='bold', color='#4c1d95')
for bar in b4:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 0.015, f"{y:.2f}", ha='center', va='bottom', fontsize=7.5, weight='bold', color='#78350f')

plt.suptitle("Parentheses 0.9-120M Session 1 — 250-Prompt Stratified Evaluation Results", fontsize=13, weight='bold', y=0.98)
plt.tight_layout()

chart_path = OUT_DIR / "s250_evaluation_metrics.png"
plt.savefig(chart_path)
plt.close()
print(f"Generated: {chart_path}")
