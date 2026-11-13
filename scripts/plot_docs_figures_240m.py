#!/usr/bin/env python3
# Engineered by uncoalesced
"""Regenerate the 240M docs figures that are computed from the released weights.

  docs/figures/param-budget.svg     where the 253,544,176 parameters live
  docs/figures/head-timescales.svg  per-head memory half-life: initial value and measured on text

Both are hand-written SVG in the uncoalesced style (Graphite ground, Sandy Brown
accent, JetBrains Mono), matching the other diagrams in docs/figures/.

Usage:
    python scripts/plot_docs_figures_240m.py [--release release/parentheses-1.0-240M-tanzanite]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GRAPHITE, CARD, RULE = "#333333", "#292929", "#4B4B4B"
PORCELAIN, MUTED = "#FDFFFC", "#9A9B99"
SANDY, SANDY_SOFT, SANDY_DARK = "#F19143", "#F5B884", "#AF703D"
FONT = "'JetBrains Mono', 'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace"

# Mixed bilingual sample used to measure how fast each head forgets on real input.
SAMPLE_TEXT = (
    "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಬಹಳ ಮುಖ್ಯ. ಕೆರೆಗಳು ಮತ್ತು ಬಾವಿಗಳು ತುಂಬಿದರೆ ಬೇಸಿಗೆಯಲ್ಲಿ ನೀರಿನ ಕೊರತೆ ಕಡಿಮೆಯಾಗುತ್ತದೆ. "
    "Water conservation during the monsoon refills lakes and wells, which reduces shortages in summer. "
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n"
)


def text(x, y, s, size=13, fill=PORCELAIN, weight=400, anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">\n'
            f'<rect width="{w}" height="{h}" fill="{GRAPHITE}"/>\n' + "\n".join(body) + "\n</svg>\n")


def load_model(release: Path):
    import chat as chat_mod
    model, tok = chat_mod.load_model_and_tokenizer(release, "cpu")[:2]
    return model.eval(), tok


def param_budget(model, out: Path) -> None:
    groups = {"swiglu": 0, "qkv": 0, "emb": 0, "out": 0, "small": 0}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        if ".mlp." in name:
            groups["swiglu"] += n
        elif name.endswith("attn.qkv.weight"):
            groups["qkv"] += n
        elif name in ("tok_emb.weight", "head.weight"):
            groups["emb"] += n
        elif name.endswith(("attn.proj.weight", "attn.gate.weight")):
            groups["out"] += n
        else:
            groups["small"] += n
    total = sum(groups.values())
    assert total == 253_544_176, total

    legend = [
        ("swiglu", SANDY, "SwiGLU (W₁, W₃, W₂)  1024 → 2752 → 1024"),
        ("qkv", SANDY_DARK, "q, k, v projection  1024 → 3072"),
        ("out", PORCELAIN, "output proj + output gate"),
        ("emb", SANDY_SOFT, "token embedding (tied with the head)"),
        ("small", MUTED, "decay, write, conv, norms, μ, V0 gates"),
    ]
    cols, cell, gap = 32, 22.0, 6.0
    squares = []
    for key, color, _ in legend:
        squares += [color] * max(1, round(groups[key] / 1e6))
    rows = math.ceil(len(squares) / cols)
    grid_top = 100.0
    legend_top = grid_top + rows * (cell + gap) + 30
    h = int(legend_top + len(legend) * 30 + 30)

    body = [
        text(36, 46, f"Where the {total:,} parameters live", 20, weight=600),
        text(36, 72, "one square ≈ 1 M parameters · 16 layers · the tied embedding is counted once", 13, MUTED),
    ]
    for i, color in enumerate(squares):
        r, c = divmod(i, cols)
        body.append(f'<rect x="{40 + c * (cell + gap):.1f}" y="{grid_top + r * (cell + gap):.1f}" '
                    f'width="{cell}" height="{cell}" rx="5" fill="{color}"/>')
    for j, (key, color, label) in enumerate(legend):
        y = legend_top + j * 30
        body.append(f'<rect x="40.0" y="{y - 13:.1f}" width="16.0" height="16.0" rx="4" fill="{color}"/>')
        body.append(text(68, y, label))
        body.append(text(720, y, f"{groups[key]:,}", anchor="end"))
        body.append(text(810, y, f"{100 * groups[key] / total:.1f} %", fill=MUTED, anchor="end"))
    out.write_text(svg(980, h, body), encoding="utf-8")
    print(f"[figures] {out.name}: {groups} total={total:,}")


@torch.no_grad()
def head_timescales(model, tok, out: Path) -> None:
    H = model.cfg.n_head
    # Initial value set by SlabV4Attention._init_decay_bias: alpha = linspace(0.5, 0.999, H).
    init_alpha = torch.linspace(0.5, 0.999, H)

    # Measured: run real text through the parallel form and record log_alpha = -softplus(alpha_proj(x_ctrl))
    # for every layer, head and token.
    captured = []
    hooks = [blk.attn.alpha_proj.register_forward_hook(lambda m, i, o: captured.append(-F.softplus(o[0].float())))
             for blk in model.blocks]
    ids = tok.encode(SAMPLE_TEXT, add_special_tokens=False).ids
    T = (len(ids) // 64) * 64
    model(torch.tensor([ids[:T]]))
    for h_ in hooks:
        h_.remove()
    log_alpha = torch.stack(captured)                      # [layers, T, H]
    median_alpha = torch.exp(log_alpha.median(dim=1).values).median(dim=0).values  # median over tokens, then layers

    def half_life(a):
        return math.log(2) / max(-math.log(float(a)), 1e-12)

    init_hl = [half_life(a) for a in init_alpha]
    meas_hl = [half_life(a) for a in median_alpha]

    x0, x1, top, row_h = 230.0, 900.0, 112.0, 26.0
    lo, hi = -1, 5   # log10 axis range in tokens

    def xpos(v):
        return x0 + (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (x1 - x0)

    bottom = top + H * row_h
    h = int(bottom + 90)
    body = [
        text(36, 46, "Memory half-life of each head (slab_v4, 16 heads)", 20, weight=600),
        text(36, 72, "half-life = ln 2 / (−ln α): tokens until one write has decayed to half · log scale", 13, MUTED),
    ]
    for e in range(lo, hi + 1):
        x = xpos(10 ** e)
        body.append(f'<polyline points="{x:.1f},{top - 14:.1f} {x:.1f},{bottom:.1f}" fill="none" stroke="{RULE}" stroke-width="1"/>')
        body.append(text(x, bottom + 22, f"10^{e}", 12, MUTED, anchor="middle"))
    xb = xpos(4096)
    body.append(f'<polyline points="{xb:.1f},{top - 14:.1f} {xb:.1f},{bottom:.1f}" fill="none" stroke="{SANDY_SOFT}" '
                f'stroke-width="1" stroke-dasharray="4 4"/>')
    body.append(text(xb, top - 20, "4,096-token training window", 11, SANDY_SOFT, anchor="middle"))
    for i in range(H):
        y = top + i * row_h + row_h / 2
        body.append(text(36, y + 4, f"head {i:2d}", 13))
        body.append(text(118, y + 4, f"α₀ {float(init_alpha[i]):.3f}", 11, MUTED))
        xi, xm = xpos(init_hl[i]), xpos(meas_hl[i])
        body.append(f'<polyline points="{xi:.1f},{y:.1f} {xm:.1f},{y:.1f}" fill="none" stroke="{RULE}" stroke-width="2"/>')
        body.append(f'<circle cx="{xi:.1f}" cy="{y:.1f}" r="6.0" fill="none" stroke="{MUTED}" stroke-width="2"/>')
        body.append(f'<circle cx="{xm:.1f}" cy="{y:.1f}" r="7.0" fill="{SANDY}"/>')
    ly = bottom + 52
    body.append(f'<circle cx="46" cy="{ly - 4:.1f}" r="6" fill="none" stroke="{MUTED}" stroke-width="2"/>')
    body.append(text(60, ly, "initial value (α from 0.500 to 0.999)", 12, MUTED))
    body.append(f'<circle cx="400" cy="{ly - 4:.1f}" r="7" fill="{SANDY}"/>')
    body.append(text(414, ly, "released weights: median α on a short Kannada, English, Python sample", 12))
    out.write_text(svg(980, h, body), encoding="utf-8")
    print(f"[figures] {out.name}: init half-life {init_hl[0]:.1f}..{init_hl[-1]:,.0f}, "
          f"measured {min(meas_hl):.1f}..{max(meas_hl):,.0f} tokens")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default=str(ROOT / "release" / "parentheses-1.0-240M-tanzanite"))
    args = ap.parse_args()
    model, tok = load_model(Path(args.release))
    figs = ROOT / "docs" / "figures"
    param_budget(model, figs / "param-budget.svg")
    head_timescales(model, tok, figs / "head-timescales.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
