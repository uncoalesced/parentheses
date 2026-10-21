#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_sft_runs.py

Compares SFT evaluation runs on Parentheses 0.9-120M across training stages:
  - Base Pretraining (step 9536)
  - SFT Run 1 (step 9636)
  - SFT Run 2 Early (step 9550)
  - SFT Run 2 Final (step 9786)

Outputs formatted Markdown comparison tables across:
  1. Overall Masked SFT Loss, Unmasked CE Loss, and Perplexities
  2. Per-Category Masked SFT Loss & Perplexity
  3. Per-Language Masked SFT Loss & Perplexity (all 9 Dravidian & English languages)

Usage:
  python scripts/compare_sft_runs.py
  python scripts/compare_sft_runs.py --out out/sft_comparison_report.md
  python scripts/compare_sft_runs.py --runs Base=out/eval_sft_val_loss_step9536_baseline.json "SFT v2 (9550)"=out/eval_sft_v2_step9550.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CANDIDATE_RUNS = [
    ("Base (9536)", REPO_ROOT / "out/eval_sft_val_loss_step9536_baseline.json"),
    ("SFT v1 (9636)", REPO_ROOT / "out/eval_sft_v1_step9636.json"),
    ("SFT v2 (9550)", REPO_ROOT / "out/eval_sft_v2_step9550.json"),
    ("SFT v2 (9600)", REPO_ROOT / "out/eval_sft_v2_step9600.json"),
    ("SFT v2 (9650)", REPO_ROOT / "out/eval_sft_v2_step9650.json"),
    ("SFT v2 (9700)", REPO_ROOT / "out/eval_sft_v2_step9700.json"),
    ("SFT v2 (9750)", REPO_ROOT / "out/eval_sft_v2_step9750.json"),
    ("SFT v2 Final (9786)", [REPO_ROOT / "out/eval_sft_v2_step9786.json", REPO_ROOT / "out/eval_sft_v2_step9785_final.json"]),
]


def load_run_data(path: Path | list[Path] | tuple[Path, ...]) -> dict | None:
    if isinstance(path, (list, tuple)):
        for p in path:
            d = load_run_data(p)
            if d is not None:
                return d
        return None
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Warning] Failed to load {path}: {e}", file=sys.stderr)
        return None


def format_delta(current: float, base: float, lower_is_better: bool = True) -> str:
    diff = current - base
    if abs(diff) < 1e-4:
        return "0.00"
    sign = "+" if diff > 0 else ""
    # If lower is better (e.g. loss/perplexity):
    # negative diff is good (green in intuition)
    return f"{sign}{diff:.2f}"


def build_markdown_report(runs: list[tuple[str, dict]]) -> str:
    if not runs:
        return "# SFT Runs Comparison Report\n\nNo evaluation runs found.\n"

    base_name, base_data = runs[0]
    run_names = [name for name, _ in runs]

    md = []
    md.append("# Parentheses 0.9-120M: SFT Runs Comparison Report\n")
    md.append(f"Evaluated on clean SFT v2 validation split (`data/processed_sft_120m_v2/val.bin`).\n")

    # --- 1. Overall Metrics Table ---
    md.append("## 1. Overall Loss & Perplexity Progression\n")
    headers = ["Metric"] + run_names
    if len(runs) > 1:
        headers.append(f"Delta ({run_names[-1]} vs {base_name})")
    
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")

    metrics_def = [
        ("Masked SFT Loss", lambda d: d["overall"]["masked_sft_loss"], True, "{:.4f}"),
        ("Masked SFT Perplexity", lambda d: d["overall"]["masked_sft_ppl"], True, "{:.2f}"),
        ("Unmasked CE Loss", lambda d: d["overall"]["unmasked_loss"], True, "{:.4f}"),
        ("Unmasked CE Perplexity", lambda d: d["overall"]["unmasked_ppl"], True, "{:.2f}"),
        ("Supervised Fraction", lambda d: d.get("supervised_fraction", 0.0), False, "{:.1%}"),
        ("Evaluated Documents", lambda d: d["evaluated_docs"], False, "{:,}"),
        ("Evaluated Tokens", lambda d: d["total_tokens"], False, "{:,}"),
    ]

    for label, getter, lower_better, fmt in metrics_def:
        row = [f"**{label}**"]
        vals = []
        for _, data in runs:
            v = getter(data)
            vals.append(v)
            row.append(fmt.format(v))
        if len(runs) > 1:
            if isinstance(vals[-1], (int, float)) and isinstance(vals[0], (int, float)):
                d_str = format_delta(vals[-1], vals[0], lower_is_better=lower_better)
                row.append(f"**{d_str}**")
            else:
                row.append("-")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n---\n")

    # --- 2. Per-Category Masked SFT Loss Table ---
    md.append("## 2. Per-Category Masked SFT Loss & Perplexity\n")
    all_cats = set()
    for _, d in runs:
        all_cats.update(d.get("by_category", {}).keys())
    
    preferred_order = [
        "structured_think",
        "drav_chat",
        "eng_chat",
        "lowres_chat",
        "freethink_cont",
        "tool_hermes",
    ]
    sorted_cats = [c for c in preferred_order if c in all_cats] + sorted([c for c in all_cats if c not in preferred_order])

    cat_headers = ["Category"]
    for name in run_names:
        cat_headers.append(f"{name} Loss (PPL)")
    if len(runs) > 1:
        cat_headers.append("Loss Delta")

    md.append("| " + " | ".join(cat_headers) + " |")
    md.append("| " + " | ".join([":---"] + [":---:"] * (len(cat_headers) - 1)) + " |")

    for cat in sorted_cats:
        row = [f"`{cat}`"]
        cat_losses = []
        for _, d in runs:
            cdata = d.get("by_category", {}).get(cat)
            if cdata:
                loss = cdata["masked_loss"]
                ppl = cdata["masked_ppl"]
                cat_losses.append(loss)
                row.append(f"{loss:.4f} ({ppl:.2f})")
            else:
                cat_losses.append(None)
                row.append("—")
        if len(runs) > 1:
            if cat_losses[-1] is not None and cat_losses[0] is not None:
                d_str = format_delta(cat_losses[-1], cat_losses[0], lower_is_better=True)
                row.append(f"**{d_str}**")
            else:
                row.append("-")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n---\n")

    # --- 3. Per-Language Masked SFT Loss Table ---
    md.append("## 3. Per-Language Masked SFT Loss & Perplexity (All 9 Languages)\n")
    lang_order = ["eng", "tel", "tam", "kan", "mal", "kff", "nit", "tcy", "kxv"]
    all_langs = set()
    for _, d in runs:
        all_langs.update(d.get("by_lang", {}).keys())
    sorted_langs = [l for l in lang_order if l in all_langs] + sorted([l for l in all_langs if l not in lang_order])

    lang_headers = ["Language"]
    for name in run_names:
        lang_headers.append(f"{name} Loss (PPL)")
    if len(runs) > 1:
        lang_headers.append("Loss Delta")

    md.append("| " + " | ".join(lang_headers) + " |")
    md.append("| " + " | ".join([":---"] + [":---:"] * (len(lang_headers) - 1)) + " |")

    lang_names = {
        "eng": "English (`eng`)",
        "tel": "Telugu (`tel`)",
        "tam": "Tamil (`tam`)",
        "kan": "Kannada (`kan`)",
        "mal": "Malayalam (`mal`)",
        "kff": "Koya (`kff`)",
        "nit": "SE Gondi (`nit`)",
        "tcy": "Tulu (`tcy`)",
        "kxv": "Kuvi (`kxv`)",
    }

    for lang in sorted_langs:
        display_name = lang_names.get(lang, f"`{lang}`")
        row = [display_name]
        lang_losses = []
        for _, d in runs:
            ldata = d.get("by_lang", {}).get(lang)
            if ldata:
                loss = ldata["masked_loss"]
                ppl = ldata["masked_ppl"]
                lang_losses.append(loss)
                row.append(f"{loss:.4f} ({ppl:.2f})")
            else:
                lang_losses.append(None)
                row.append("—")
        if len(runs) > 1:
            if lang_losses[-1] is not None and lang_losses[0] is not None:
                d_str = format_delta(lang_losses[-1], lang_losses[0], lower_is_better=True)
                row.append(f"**{d_str}**")
            else:
                row.append("-")
        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md) + "\n"


def main():
    p = argparse.ArgumentParser(description="Compare SFT evaluation runs and output Markdown comparison tables.")
    p.add_argument("--runs", nargs="*", help="Optional custom run specs as 'Label=path/to/eval.json'")
    p.add_argument("--out", default=None, help="Optional output path for Markdown report")
    args = p.parse_args()

    active_runs: list[tuple[str, dict]] = []

    if args.runs:
        for item in args.runs:
            if "=" in item:
                lbl, path_s = item.split("=", 1)
            else:
                path_s = item
                lbl = Path(item).stem
            data = load_run_data(Path(path_s))
            if data:
                active_runs.append((lbl, data))
            else:
                print(f"[Notice] Skipping missing or unreadable run: {item}", file=sys.stderr)
    else:
        for lbl, path in DEFAULT_CANDIDATE_RUNS:
            data = load_run_data(path)
            if data:
                active_runs.append((lbl, data))

    if not active_runs:
        print("[Error] No evaluation JSON files found to compare.", file=sys.stderr)
        sys.exit(1)

    report_md = build_markdown_report(active_runs)
    print(report_md)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report_md, encoding="utf-8")
        print(f"[compare_sft_runs] Wrote comparison report to {out_p}")


if __name__ == "__main__":
    main()
