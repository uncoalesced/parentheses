"""
Draw the training-metric figures used by README.md and docs/ in the uncoalesced design language.

Graphite ground, Porcelain ink, one Sandy Brown accent, JetBrains Mono, circles and rounded squares,
no gradients. Text is converted to paths so the SVGs render the same on GitHub without the font.

Engineered by uncoalesced

Usage (from the repo root):
    python scripts/plot_docs_figures.py
    python scripts/plot_docs_figures.py --out docs/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch

GRAPHITE, PORCELAIN = "#333333", "#FDFFFC"
SANDY = "#F19143"
MUTED = "#9A9B99"
BORDER = "#4B4B4B"

ROOT = Path(__file__).resolve().parent.parent
MARK = ROOT / "assets" / "parentheses_mark.svg"
FONT_CANDIDATES = [
    Path(r"C:\Users\Joel Anthony\AppData\Local\Programs\cursor\resources\app\out\media\jetbrains-mono-regular.ttf"),
    Path(r"C:\Windows\Fonts\JetBrainsMono-Regular.ttf"),
]

plt.rcParams["svg.fonttype"] = "path"
for _f in FONT_CANDIDATES:
    if _f.exists():
        font_manager.fontManager.addfont(str(_f))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_f)).get_name()
        break
else:
    plt.rcParams["font.family"] = "monospace"


def mix(a: str, b: str, t: float) -> str:
    ra = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    rb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ra, rb))


SANDY_TINT = mix(SANDY, PORCELAIN, 0.35)
SANDY_SHADE = mix(SANDY, GRAPHITE, 0.35)


def draw_mark(fig, rect):
    """Draw the parentheses mosaic mark (circles + rounded squares) into a figure-relative rect."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 900)
    ax.set_ylim(900, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    svg = MARK.read_text(encoding="utf-8")
    for cx, cy, r, fill in re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)" fill="(#\w+)"', svg):
        ax.add_patch(Circle((float(cx), float(cy)), float(r), color=fill))
    for x, y, w, h, rx, fill in re.findall(
            r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" rx="([\d.]+)" fill="(#\w+)"', svg):
        ax.add_patch(FancyBboxPatch((float(x), float(y)), float(w), float(h),
                                    boxstyle=f"round,pad=0,rounding_size={float(rx)}", color=fill))


def frame(title: str, subtitle: str):
    fig = plt.figure(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(GRAPHITE)
    fig.text(0.115, 0.925, title, color=PORCELAIN, fontsize=17, weight="bold")
    fig.text(0.115, 0.875, subtitle, color=MUTED, fontsize=10.5)
    draw_mark(fig, [0.035, 0.855, 0.065, 0.11])
    fig.text(0.965, 0.03, "uncoalesced / parentheses", color=MUTED, fontsize=9, ha="right")
    return fig


def style(ax, grid_axis: str = "y"):
    ax.set_facecolor(GRAPHITE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=PORCELAIN, length=0, labelsize=11)
    ax.grid(axis=grid_axis, color=BORDER, lw=0.8)
    ax.set_axisbelow(True)


def ema(xs: list[float], alpha: float) -> list[float]:
    out, acc = [], None
    for x in xs:
        acc = x if acc is None else alpha * x + (1 - alpha) * acc
        out.append(acc)
    return out


def sft_loss_130m(log_path: Path, out: Path):
    pts = [(int(m.group(1)), float(m.group(2)))
           for m in re.finditer(r"^step\s+(\d+) \| loss ([\d.]+)", log_path.read_text(encoding="utf-8"), re.M)]
    steps, loss = zip(*pts)
    fig = frame("parentheses-0.9-130M: instruction tuning loss",
                "SFT v2, 2,000 steps, 131M tokens  ·  dots = logged steps, line = moving average")
    ax = fig.add_axes([0.08, 0.12, 0.87, 0.68])
    style(ax)
    ax.scatter(steps, loss, s=14, color=SANDY_SHADE, zorder=2)
    ax.plot(steps, ema(list(loss), 0.08), color=SANDY, lw=3, solid_capstyle="round", zorder=3)
    ax.set_xlabel("step", color=MUTED, fontsize=11)
    ax.set_ylabel("loss", color=MUTED, fontsize=11)
    fig.savefig(out, facecolor=GRAPHITE)
    plt.close(fig)


def bpb_130m(bpb_json: Path, out: Path):
    d = json.loads(bpb_json.read_text(encoding="utf-8"))["B_dpo_ag"]
    rows = [("kan", "Kannada", SANDY), ("tam", "Tamil", SANDY_TINT), ("tel", "Telugu", SANDY_TINT),
            ("mal", "Malayalam", SANDY_TINT), ("eng", "English", SANDY_TINT),
            ("tcy", "Tulu", SANDY_SHADE), ("nit", "Kolami", SANDY_SHADE),
            ("kff", "Koya", SANDY_SHADE), ("kxv", "Kuvi", SANDY_SHADE)]
    strong = {"kan", "tam", "tel", "mal", "eng"}
    fig = frame("parentheses-0.9-130M: bits per byte, by language",
                "held-out text, lower is better  ·  strong languages on top, low-resource below")
    ax = fig.add_axes([0.16, 0.12, 0.79, 0.68])
    style(ax, "x")
    ys = [0, 1, 2, 3, 4, 5.6, 6.6, 7.6, 8.6]
    for y, (code, _, col) in zip(ys, rows):
        v = d[code]
        ax.plot([0, v], [y, y], color=BORDER, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter([v], [y], s=520 if code == "kan" else 300, color=col, zorder=3)
        ax.text(v + 0.07, y, f"{v:.2f}", va="center", color=PORCELAIN if code == "kan" else MUTED,
                fontsize=12 if code == "kan" else 11, weight="bold" if code == "kan" else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n, _ in rows])
    for lbl, (code, _, _) in zip(ax.get_yticklabels(), rows):
        lbl.set_color(SANDY if code == "kan" else (PORCELAIN if code in strong else MUTED))
    ax.axhline(4.8, color=BORDER, lw=1, ls=(0, (3, 4)))
    ax.text(1.95, 4.8, "low-resource", color=MUTED, fontsize=9.5, ha="right", va="center",
            backgroundcolor=GRAPHITE)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(9.2, -0.6)
    fig.savefig(out, facecolor=GRAPHITE)
    plt.close(fig)


def tokenizer(out: Path):
    fig = frame("one token, how much Indic text?",
                "average bytes of Indic text per token  ·  each square = 1 byte")
    ax = fig.add_axes([0.06, 0.12, 0.9, 0.66])
    ax.set_facecolor(GRAPHITE)
    ax.axis("off")
    ax.set_xlim(-3.4, 13.0)
    ax.set_ylim(3.8, -0.6)
    ax.set_aspect("equal")
    rows = [(0.2, "GPT-style, 48k", 3.67, MUTED), (1.45, "ours, 48k", 10.76, SANDY),
            (2.7, "ours, 32k (shipped)", 9.94, SANDY_TINT)]
    for y, name, val, col in rows:
        ax.text(-3.3, y, name, color=MUTED if col == MUTED else PORCELAIN, fontsize=13, va="center")
        full, frac = int(val), val - int(val)
        for i in range(full + (1 if frac > 0 else 0)):
            w = 0.78 if i < full else 0.78 * frac
            ax.add_patch(FancyBboxPatch((i + 0.11, y - 0.39), w, 0.78,
                                        boxstyle="round,pad=0,rounding_size=0.09", color=col))
        ax.text(full + 1.3, y, f"{val:.2f}", color=PORCELAIN, fontsize=15, weight="bold", va="center")
    ax.text(-3.3, 3.55, "same vocab size, vowel signs and viramas kept attached: about 2.9x more Indic text per token",
            color=MUTED, fontsize=10.5)
    fig.savefig(out, facecolor=GRAPHITE)
    plt.close(fig)


def training_240m(metrics_jsonl: Path, out: Path):
    by_step: dict[int, float] = {}
    for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_step[int(r["step"])] = float(r["loss"])  # a resumed step overwrites the earlier one
    steps = sorted(by_step)
    loss = [by_step[s] for s in steps]
    fig = frame("parentheses-1.0-240M: pretraining so far",
                "train loss, moving average  ·  each band is one session on the same laptop GPU")
    ax = fig.add_axes([0.08, 0.18, 0.87, 0.60])
    style(ax)
    sessions = [(1, 2000, "S1"), (2001, 4000, "S2"), (4001, 6000, "S3"), (6001, 8000, "S4  stem + code")]
    for i, (a, b, name) in enumerate(sessions):
        if i % 2 == 0:
            ax.axvspan(a, b, color=mix(GRAPHITE, PORCELAIN, 0.05), lw=0, zorder=0)
        ax.text((a + b) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center",
                color=MUTED, fontsize=10.5)
    ax.plot(steps, ema(loss, 0.01), color=SANDY, lw=2.6, solid_capstyle="round", zorder=3)
    ax.scatter([steps[-1]], [ema(loss, 0.01)[-1]], s=160, color=SANDY, zorder=4)
    ax.set_xlim(0, max(steps) + 150)
    ax.set_xlabel("step", color=MUTED, fontsize=11)
    ax.set_ylabel("loss", color=MUTED, fontsize=11)
    ax.text(0.0, -0.2, "S4 switched to a STEM and code mix, so part of its drop is the data changing, not only learning",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    fig.savefig(out, facecolor=GRAPHITE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sft_loss_130m(ROOT / "out" / "sft_130m_v3_notrunc.log", out / "sft-loss-130m.svg")
    bpb_130m(ROOT / "out" / "cmp_130m_v3" / "bpb.json", out / "bpb-by-language-130m.svg")
    tokenizer(out / "tokenizer-bytes-per-token.svg")
    training_240m(ROOT / "out" / "session1_240m_metrics.jsonl", out / "training-240m-sessions.svg")
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
