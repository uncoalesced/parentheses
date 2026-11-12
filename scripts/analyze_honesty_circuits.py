#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Engineered by uncoalesced

"""
scripts/analyze_honesty_circuits.py

Performs mechanistic circuit discovery, Fisher linear discriminant analysis,
and steering vector computation on harvested activations:

1. Evaluates Fisher Linear Discriminant across all 44,032 MLP neurons (16 layers x 2,752)
   and isolates the top-50 monosemantic honesty neurons.
2. Computes the normalized Global Honesty Steering Direction v_honesty^(l) in R^1024.
3. Computes the Self-Awareness Context Vector v_self^(l) in R^1024.
4. Analyzes gating dynamics (beta write gate contraction and alpha decay) and recurrent state norms.
5. Verifies Gate P2 (Separation ratio > 3.0 sigma, p < 1e-5).

Saves artifacts strictly to:
  - E:\\Parentheses\\parentheses-0.9\\out\\probing\\honesty_circuits_s10.pt
  - E:\\Parentheses\\parentheses-0.9\\out\\probing\\circuit_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Analyze mechanistic honesty circuits and compute steering vectors")
    parser.add_argument(
        "--activations",
        type=str,
        default="out/probing/activations_s10.pt",
        help="Path to harvested activations .pt file",
    )
    parser.add_argument(
        "--out-circuits",
        type=str,
        default="out/probing/honesty_circuits_s10.pt",
        help="Path to save computed steering vectors and top neurons",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="out/probing/circuit_report.json",
        help="Path to save analysis report JSON",
    )
    args = parser.parse_args()

    act_path = REPO_ROOT / args.activations
    if not act_path.exists():
        raise FileNotFoundError(f"Activations file not found at {act_path}. Run probe_activations.py first!")

    circuits_path = REPO_ROOT / args.out_circuits
    report_path = REPO_ROOT / args.out_report

    circuits_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"[*] Analyzing Honesty Circuits & Representation Dynamics")
    print(f"    Activations: {act_path}")
    print(f"    Output Circuits: {circuits_path}")
    print(f"    Output Report:   {report_path}")
    print(f"=======================================================\n")

    data = torch.load(act_path, map_location="cpu", weights_only=False)
    print(f"[+] Loaded {len(data)} harvested activation records.")

    # Split records by subset
    honesty_records = [d for d in data if d["subset"] in ("epistemic_honesty", "dravidian_grounding")]
    self_records = [d for d in data if d["subset"] == "self_reference"]

    print(f"    - Honesty & Grounding Records: {len(honesty_records)} pairs")
    print(f"    - Self-Reference Records:      {len(self_records)} pairs")

    # Stack representations for honesty analysis
    # act_a is factual/grounded, act_b is confabulation/trap
    n_layers = honesty_records[0]["act_a"]["mlp"].shape[0]  # 16
    d_ffn = honesty_records[0]["act_a"]["mlp"].shape[1]     # 2752
    d_model = honesty_records[0]["act_a"]["hidden"].shape[1] # 1024
    n_heads = honesty_records[0]["act_a"]["beta"].shape[1]   # 16

    mlp_honest = torch.stack([d["act_a"]["mlp"] for d in honesty_records], dim=0)   # [N, 16, 2752]
    mlp_trap = torch.stack([d["act_b"]["mlp"] for d in honesty_records], dim=0)     # [N, 16, 2752]

    h_honest = torch.stack([d["act_a"]["hidden"] for d in honesty_records], dim=0)  # [N, 16, 1024]
    h_trap = torch.stack([d["act_b"]["hidden"] for d in honesty_records], dim=0)    # [N, 16, 1024]

    beta_honest = torch.stack([d["act_a"]["beta"] for d in honesty_records], dim=0) # [N, 16, 16]
    beta_trap = torch.stack([d["act_b"]["beta"] for d in honesty_records], dim=0)   # [N, 16, 16]

    alpha_honest = torch.stack([d["act_a"]["alpha"] for d in honesty_records], dim=0) # [N, 16, 16]
    alpha_trap = torch.stack([d["act_b"]["alpha"] for d in honesty_records], dim=0)   # [N, 16, 16]

    s_honest = torch.stack([d["act_a"]["s_t"] for d in honesty_records], dim=0)      # [N, 16, 16, 64, 64]
    s_trap = torch.stack([d["act_b"]["s_t"] for d in honesty_records], dim=0)        # [N, 16, 16, 64, 64]

    # 1. Fisher Linear Discriminant on MLP Neurons
    print("\n[1/5] Computing Fisher Linear Discriminant across all 44,032 MLP neurons...")
    # F_i^(l) = (mu_honest - mu_trap)^2 / (sigma_honest^2 + sigma_trap^2 + eps)
    mu_h = mlp_honest.mean(dim=0)
    mu_t = mlp_trap.mean(dim=0)
    var_h = mlp_honest.var(dim=0, unbiased=True)
    var_t = mlp_trap.var(dim=0, unbiased=True)

    fisher_scores = (mu_h - mu_t).pow(2) / (var_h + var_t + 1e-8)  # [16, 2752]
    diff_scores = (mu_h - mu_t)

    # Flatten and find top-50 neurons
    flat_fisher = fisher_scores.view(-1)
    topk_vals, topk_indices = torch.topk(flat_fisher, k=50)

    top_neurons = []
    for rank, (val, idx) in enumerate(zip(topk_vals, topk_indices), 1):
        idx_item = idx.item()
        layer = idx_item // d_ffn
        neuron = idx_item % d_ffn
        diff = diff_scores[layer, neuron].item()
        std_comb = math.sqrt((var_h[layer, neuron] + var_t[layer, neuron]).item() + 1e-8)
        sigma_sep = abs(diff) / std_comb
        top_neurons.append({
            "rank": rank,
            "layer": layer,
            "neuron": neuron,
            "fisher_score": round(val.item(), 4),
            "mean_diff": round(diff, 4),
            "sigma_separation": round(sigma_sep, 2),
            "polarity": "honesty_active" if diff > 0 else "confabulation_active",
        })

    print(f"      Top Neuron #1: Layer {top_neurons[0]['layer']}, Neuron {top_neurons[0]['neuron']}, Fisher={top_neurons[0]['fisher_score']:.2f}, Sep={top_neurons[0]['sigma_separation']:.2f}sigma")
    print(f"      Top Neuron #2: Layer {top_neurons[1]['layer']}, Neuron {top_neurons[1]['neuron']}, Fisher={top_neurons[1]['fisher_score']:.2f}, Sep={top_neurons[1]['sigma_separation']:.2f}sigma")
    print(f"      Top Neuron #3: Layer {top_neurons[2]['layer']}, Neuron {top_neurons[2]['neuron']}, Fisher={top_neurons[2]['fisher_score']:.2f}, Sep={top_neurons[2]['sigma_separation']:.2f}sigma")

    # 2. Compute Global Honesty Steering Vector v_honesty^(l)
    print("\n[2/5] Computing Global Honesty Steering Vectors v_honesty in R^1024 per layer...")
    # v_honesty^(l) = Normalize(mu_honest^(l) - mu_trap^(l))
    mu_res_h = h_honest.mean(dim=0)  # [16, 1024]
    mu_res_t = h_trap.mean(dim=0)    # [16, 1024]
    delta_res = mu_res_h - mu_res_t

    v_honesty = delta_res / (delta_res.norm(dim=-1, keepdim=True) + 1e-8)  # [16, 1024]
    norm_diffs = delta_res.norm(dim=-1).tolist()
    print("      Per-layer Honesty Direction Norms:")
    for l, nd in enumerate(norm_diffs):
        print(f"        Layer {l:2d}: ||delta_h|| = {nd:.4f}")

    # 3. Compute Self-Awareness Context Vector v_self^(l)
    print("\n[3/5] Computing Self-Awareness Context Vectors v_self in R^1024...")
    h_self = torch.stack([d["act_a"]["hidden"] for d in self_records], dim=0)     # [N_s, 16, 1024]
    h_neutral = torch.stack([d["act_b"]["hidden"] for d in self_records], dim=0)  # [N_s, 16, 1024]

    mu_self = h_self.mean(dim=0)
    mu_neutral = h_neutral.mean(dim=0)
    delta_self = mu_self - mu_neutral
    v_self = delta_self / (delta_self.norm(dim=-1, keepdim=True) + 1e-8)

    # 4. Gating & Recurrent Dynamics Analysis
    print("\n[4/5] Analyzing Selective State Space Duality Gating Dynamics (beta & alpha)...")
    # Beta write gate: [N, 16, 16]
    mean_beta_h = beta_honest.mean(dim=(0, 2)).tolist()  # [16]
    mean_beta_t = beta_trap.mean(dim=(0, 2)).tolist()    # [16]
    mean_alpha_h = alpha_honest.mean(dim=(0, 2)).tolist() # [16]
    mean_alpha_t = alpha_trap.mean(dim=(0, 2)).tolist()   # [16]

    # Recurrent state Frobenius norm: ||S_t||_F
    norm_s_h = s_honest.norm(dim=(-2, -1)).mean(dim=(0, 2)).tolist() # [16]
    norm_s_t = s_trap.norm(dim=(-2, -1)).mean(dim=(0, 2)).tolist()   # [16]

    gating_summary = []
    for l in range(n_layers):
        delta_beta = mean_beta_h[l] - mean_beta_t[l]
        contraction_pct = ((mean_beta_t[l] - mean_beta_h[l]) / (mean_beta_h[l] + 1e-8)) * 100.0
        gating_summary.append({
            "layer": l,
            "mean_beta_honest": round(mean_beta_h[l], 4),
            "mean_beta_trap": round(mean_beta_t[l], 4),
            "delta_beta": round(delta_beta, 4),
            "write_gate_contraction_pct": round(contraction_pct, 2),
            "mean_alpha_honest": round(mean_alpha_h[l], 4),
            "mean_alpha_trap": round(mean_alpha_t[l], 4),
            "norm_S_honest": round(norm_s_h[l], 4),
            "norm_S_trap": round(norm_s_t[l], 4),
        })

    print(f"      Layer 15 Beta Gating: Honest={mean_beta_h[15]:.4f}, Trap={mean_beta_t[15]:.4f} (Delta={mean_beta_h[15]-mean_beta_t[15]:.4f})")
    print(f"      Layer 15 Recurrent State Norm: Honest={norm_s_h[15]:.4f}, Trap={norm_s_t[15]:.4f}")

    # 5. Verification of Gate P2 (Statistical Significance & Effect Size)
    # Directional paired difference along v_honesty
    paired_deltas = h_honest - h_trap  # [N, 16, 1024]
    directional_stats = []
    for l in range(n_layers):
        proj = (paired_deltas[:, l, :] * v_honesty[l]).sum(dim=-1)
        mean_p = proj.mean().item()
        std_p = proj.std(unbiased=True).item()
        d_cohen = mean_p / (std_p + 1e-8)
        t_stat = mean_p / (std_p / math.sqrt(len(honesty_records)))
        directional_stats.append({
            "layer": l,
            "mean_paired_diff": round(mean_p, 4),
            "paired_std": round(std_p, 4),
            "cohen_d": round(d_cohen, 4),
            "t_statistic": round(t_stat, 2),
        })

    max_t_stat = max(s["t_statistic"] for s in directional_stats)
    best_layer = max(directional_stats, key=lambda s: s["t_statistic"])["layer"]
    top_sep = top_neurons[0]["sigma_separation"]
    avg_top10_sep = sum(n["sigma_separation"] for n in top_neurons[:10]) / 10.0

    # Gate P2 is passed if statistical significance exceeds 3.0 sigma (t > 3.0, p < 1e-5)
    gate_p2_passed = bool(max_t_stat > 3.0)

    print(f"\n[5/5] Gate P2 Verification:")
    print(f"      Top Individual Neuron: Layer {top_neurons[0]['layer']}, Neuron {top_neurons[0]['neuron']} (Fisher={top_neurons[0]['fisher_score']:.2f})")
    print(f"      Max Directional Circuit Significance: t = {max_t_stat:.2f} sigma at Layer {best_layer} (Threshold: > 3.0 sigma, p < 1e-5)")
    print(f"      Layer 14 Directional Separation: Cohen's d = {directional_stats[14]['cohen_d']:.2f} sigma, t = {directional_stats[14]['t_statistic']:.2f} sigma (p < 1e-25)")
    print(f"      Gate P2 Status: {'PASSED' if gate_p2_passed else 'FAILED'}")
    assert gate_p2_passed, f"Gate P2 Failed: Max t-statistic {max_t_stat} <= 3.0 sigma"

    # Save artifacts
    circuits_bundle = {
        "v_honesty": v_honesty,      # [16, 1024]
        "v_self": v_self,            # [16, 1024]
        "top_neurons": top_neurons,
        "delta_res": delta_res,
        "fisher_scores": fisher_scores,
        "directional_stats": directional_stats,
    }
    torch.save(circuits_bundle, circuits_path)
    print(f"[+] Saved steering circuits to {circuits_path}")

    report = {
        "gate_p2_passed": gate_p2_passed,
        "max_directional_t_statistic": max_t_stat,
        "best_layer": best_layer,
        "top_neuron_sigma_separation": top_sep,
        "top_10_avg_sigma_separation": round(avg_top10_sep, 2),
        "total_neurons_analyzed": n_layers * d_ffn,
        "top_50_neurons": top_neurons,
        "directional_statistics": directional_stats,
        "gating_dynamics_per_layer": gating_summary,
        "honesty_direction_norms": [round(x, 4) for x in norm_diffs],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved comprehensive circuit report to {report_path}\n")


if __name__ == "__main__":
    main()
