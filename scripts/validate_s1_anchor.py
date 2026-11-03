# Engineered by uncoalesced
"""
Parentheses 1.0-240M (slab_v4) - Session 1 to Session 2 Anchor Validator
Audits head variance (0..11 vs 12..15), donor head symmetry breaking (12..15 vs 0..3),
zero-init residual/MLP wake-up (0..767 vs 768..1023), SSD recurrent gates,
and AdamW optimizer state health before launching Session 2.
"""

import argparse
import glob
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch.nn.functional as F

# Architectural Constants for Parentheses 1.0-240M (slab_v4)
D_MODEL = 1024
D_INHERITED = 768
N_HEADS = 16
N_INHERITED_HEADS = 12
N_NEW_HEADS = 4
D_HEAD = 64


def resolve_latest_checkpoint(ckpt_arg: str) -> Path:
    path = Path(ckpt_arg)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            path.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not locate checkpoint file at or inside: {ckpt_arg}")


def clean_state_dict(raw_ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = raw_ckpt.get("model", raw_ckpt.get("state_dict", raw_ckpt))
    ptr_cache: Dict[int, torch.Tensor] = {}
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            ptr = v.data_ptr()
            if ptr not in ptr_cache:
                ptr_cache[ptr] = v.detach().float().cpu()
            out[k.replace("_orig_mod.", "")] = ptr_cache[ptr]
    return out


def tensor_rms(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(t * t)).item())


def cosine_sim_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    norm_a = torch.linalg.norm(af)
    norm_b = torch.linalg.norm(bf)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(torch.dot(af, bf) / (norm_a * norm_b))


def extract_head_view(param: torch.Tensor, name: str) -> Optional[torch.Tensor]:
    """
    Reshapes a parameter tensor so dim 0 is N_HEADS (16) if it represents
    per-head weights, projections, or SSD recurrent gates.
    """
    shape = tuple(param.shape)
    if len(shape) == 0:
        return None

    # Explicit head dimension at axis 0: e.g., (16,), (16, 64), (16, 64, 64)
    if shape[0] == N_HEADS:
        return param.reshape(N_HEADS, -1)

    # Explicit head dimension at axis 1: e.g., (batch_or_layer, 16, ...)
    if len(shape) >= 2 and shape[1] == N_HEADS:
        perm = [1, 0] + list(range(2, len(shape)))
        return param.permute(*perm).contiguous().reshape(N_HEADS, -1)

    # 2D projection matrices where output dim (axis 0) is 1024 (Q, K, V, gate projections)
    if len(shape) == 2 and shape[0] == D_MODEL and any(
        tok in name.lower() for tok in ["q_proj", "k_proj", "v_proj", "wq", "wk", "wv", "beta", "gate", "in_proj"]
    ):
        return param.reshape(N_HEADS, D_HEAD, shape[1]).reshape(N_HEADS, -1)

    # 2D output projection matrices where input dim (axis 1) is 1024 (out_proj / wo)
    if len(shape) == 2 and shape[1] == D_MODEL and any(
        tok in name.lower() for tok in ["out_proj", "wo", "o_proj"]
    ):
        return param.permute(1, 0).contiguous().reshape(N_HEADS, D_HEAD, shape[0]).reshape(N_HEADS, -1)

    return None


def audit_checkpoint(
    s1_path: Path,
    init_path: Optional[Path],
) -> Tuple[str, bool]:
    print(f"[*] Loading Session 1 checkpoint: {s1_path}")
    s1_raw = torch.load(s1_path, map_location="cpu", weights_only=False)
    s1_state = clean_state_dict(s1_raw)
    step = s1_raw.get("step", s1_raw.get("iter", "unknown"))

    init_state: Optional[Dict[str, torch.Tensor]] = None
    if init_path and init_path.exists():
        print(f"[*] Loading baseline Net2WiderNet init checkpoint: {init_path}")
        init_raw = torch.load(init_path, map_location="cpu", weights_only=False)
        init_state = clean_state_dict(init_raw)

    # 1. NaN / Inf Integrity Scan (deduplicated by data_ptr for tied embeddings)
    nan_params: List[str] = []
    seen_ptrs: set[int] = set()
    total_params = 0
    for name, tensor in s1_state.items():
        ptr = tensor.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            total_params += tensor.numel()
        if not torch.isfinite(tensor).all():
            nan_params.append(name)

    # 2. Per-Head Variance & Symmetry Breaking (0..11 vs 12..15)
    head_metrics: List[Dict[str, Any]] = []
    ssd_gate_metrics: List[Dict[str, Any]] = []
    residual_slice_metrics: List[Dict[str, Any]] = []

    for name, param in sorted(s1_state.items()):
        lname = name.lower()
        head_tensor = extract_head_view(param, name)

        if head_tensor is not None:
            old_heads = head_tensor[:N_INHERITED_HEADS]  # (12, D)
            new_heads = head_tensor[N_INHERITED_HEADS:]  # (4, D)

            old_rms = tensor_rms(old_heads)
            new_rms = tensor_rms(new_heads)
            old_std = float(old_heads.std().item()) if old_heads.numel() > 1 else 0.0
            new_std = float(new_heads.std().item()) if new_heads.numel() > 1 else 0.0
            ratio = (new_rms / old_rms) if old_rms > 1e-12 else 0.0

            # Donor head cosine similarity: compare heads 12..15 against donor heads 0..3
            donor_cosines = [
                cosine_sim_flat(head_tensor[12 + i], head_tensor[i])
                for i in range(N_NEW_HEADS)
            ]
            mean_donor_cos = sum(donor_cosines) / len(donor_cosines)

            # Delta from init_grown.pt if available
            delta_old_rms = float("nan")
            delta_new_rms = float("nan")
            if init_state and name in init_state:
                init_ht = extract_head_view(init_state[name], name)
                if init_ht is not None and init_ht.shape == head_tensor.shape:
                    delta_old_rms = tensor_rms(old_heads - init_ht[:N_INHERITED_HEADS])
                    delta_new_rms = tensor_rms(new_heads - init_ht[N_INHERITED_HEADS:])

            entry = {
                "name": name,
                "shape": tuple(param.shape),
                "old_rms": old_rms,
                "new_rms": new_rms,
                "old_std": old_std,
                "new_std": new_std,
                "rms_ratio": ratio,
                "donor_cos": mean_donor_cos,
                "delta_old": delta_old_rms,
                "delta_new": delta_new_rms,
            }
            head_metrics.append(entry)

            if any(k in lname for k in ["beta", "mu", "shift", "lambda", "v0", "decay", "gate"]):
                ssd_gate_metrics.append(entry)

        # 3. Residual & MLP Expansion Slice Wake-Up (0..767 vs 768..1023)
        elif param.ndim == 2 and (param.shape[0] == D_MODEL or param.shape[1] == D_MODEL):
            if param.shape[0] == D_MODEL:
                base_slice = param[:D_INHERITED, :]
                exp_slice = param[D_INHERITED:, :]
                slice_axis = "dim0 (rows 768..1023)"
            else:
                base_slice = param[:, :D_INHERITED]
                exp_slice = param[:, D_INHERITED:]
                slice_axis = "dim1 (cols 768..1023)"

            base_rms = tensor_rms(base_slice)
            exp_rms = tensor_rms(exp_slice)
            ratio = (exp_rms / base_rms) if base_rms > 1e-12 else 0.0

            residual_slice_metrics.append({
                "name": name,
                "axis": slice_axis,
                "base_rms": base_rms,
                "exp_rms": exp_rms,
                "ratio": ratio,
            })

    # 4. Optimizer State Verification (AdamW exp_avg_sq health)
    opt_dict = s1_raw.get("optimizer", s1_raw.get("opt", None))
    opt_finite = True
    opt_populated_tensors = 0
    opt_zero_second_moment = 0

    if isinstance(opt_dict, dict) and "state" in opt_dict:
        for _, st in opt_dict["state"].items():
            if isinstance(st, dict) and "exp_avg_sq" in st:
                v = st["exp_avg_sq"]
                if torch.is_tensor(v):
                    opt_populated_tensors += 1
                    if not torch.isfinite(v).all():
                        opt_finite = False
                    if float(v.abs().max().item()) < 1e-16:
                        opt_zero_second_moment += 1

    # 5. Evaluate Readiness Gates for Session 2
    mean_rms_ratio = (
        sum(m["rms_ratio"] for m in head_metrics) / len(head_metrics)
        if head_metrics else 0.0
    )
    mean_donor_cos = (
        sum(m["donor_cos"] for m in head_metrics) / len(head_metrics)
        if head_metrics else 1.0
    )
    dead_exp_slices = [
        m["name"] for m in residual_slice_metrics if m["exp_rms"] < 1e-5
    ]

    gate_no_nan = (len(nan_params) == 0) and opt_finite
    gate_head_variance = 0.15 <= mean_rms_ratio <= 2.00
    gate_symmetry_broken = mean_donor_cos < 0.9995
    gate_residual_awake = len(dead_exp_slices) == 0
    gate_opt_ready = opt_populated_tensors > 0 and opt_zero_second_moment == 0

    all_passed = all([
        gate_no_nan,
        gate_head_variance,
        gate_symmetry_broken,
        gate_residual_awake,
        gate_opt_ready,
    ])

    # Build Markdown Report
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    verdict = "[PASS] READY FOR SESSION 2 (LR 3e-4 -> 4.5e-4)" if all_passed else "[WARN] REVIEW ANOMALIES BEFORE SESSION 2"

    head_rows = []
    for m in head_metrics[:40]:  # Cap table to keep digest scannable
        d_old = f"{m['delta_old']:.5f}" if not math.isnan(m["delta_old"]) else "N/A"
        d_new = f"{m['delta_new']:.5f}" if not math.isnan(m["delta_new"]) else "N/A"
        head_rows.append(
            f"| `{m['name']}` | `{m['old_rms']:.5f}` | `{m['new_rms']:.5f}` | "
            f"`{m['rms_ratio']:.3f}` | `{m['donor_cos']:.5f}` | `{d_old}` | `{d_new}` |"
        )

    ssd_rows = []
    for m in ssd_gate_metrics:
        ssd_rows.append(
            f"| `{m['name']}` | `{m['old_rms']:.5f}` | `{m['new_rms']:.5f}` | "
            f"`{m['old_std']:.5f}` | `{m['new_std']:.5f}` | `{m['donor_cos']:.5f}` |"
        )

    slice_rows = []
    for m in residual_slice_metrics[:32]:
        slice_rows.append(
            f"| `{m['name']}` | {m['axis']} | `{m['base_rms']:.5f}` | `{m['exp_rms']:.5f}` | `{m['ratio']:.3f}` |"
        )

    report = f"""# Parentheses 1.0-240M (`slab_v4`) - Session 1 Anchor Validation Report

- **Timestamp:** `{now_str}`
- **Session 1 Checkpoint:** `{s1_path}` (Step: `{step}`)
- **Baseline Init Checkpoint:** `{init_path if (init_path and init_path.exists()) else "Not provided"}`
- **Total Model Parameters Audited:** `{total_params:,}` (`{total_params / 1e6:.2f}M`)
- **Overall Gate Verdict:** **{verdict}**

---

## 1. Go / No-Go Gate Summary for Session 2

| Gate Check | Criterion | Measured Value | Status |
| :--- | :--- | :--- | :--- |
| **1. Numerical Integrity** | `0` NaN/Inf in weights + AdamW states | `{len(nan_params)}` corrupted tensors | `{"[PASS]" if gate_no_nan else "[FAIL]"}` |
| **2. Head RMS Ratio (`12..15` / `0..11`)** | Mean ratio within `[0.15, 2.00]` | `{mean_rms_ratio:.4f}` across `{len(head_metrics)}` tensors | `{"[PASS]" if gate_head_variance else "[WARN]"}` |
| **3. Donor Head Symmetry Breaking** | Mean Cosine(`12..15`, `0..3`) below `0.9995` | `{mean_donor_cos:.6f}` | `{"[PASS]" if gate_symmetry_broken else "[WARN]"}` |
| **4. Zero-Init Slice Wake-Up (`768..1023`)** | All expanded residual/MLP slices RMS above `1e-5` | `{len(dead_exp_slices)}` dead slices out of `{len(residual_slice_metrics)}` | `{"[PASS]" if gate_residual_awake else "[WARN]"}` |
| **5. AdamW `exp_avg_sq` Readiness** | Second-moment buffers populated & finite | `{opt_populated_tensors}` state tensors (`{opt_zero_second_moment}` dead) | `{"[PASS]" if gate_opt_ready else "[WARN]"}` |

---

## 2. SSD Recurrent Gates & Gating Priors (`Heads 0..11` vs `Heads 12..15`)

| Parameter Name | Old Heads RMS (`0..11`) | New Heads RMS (`12..15`) | Old Std | New Std | Donor Cosine (`12..15` vs `0..3`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(ssd_rows) if ssd_rows else "| *No explicit 16-head gate tensors matched filter* | - | - | - | - | - |"}

---

## 3. Per-Head Projection & Attention Variance (`Heads 0..11` vs `Heads 12..15`)

| Parameter Name | Old RMS (`0..11`) | New RMS (`12..15`) | Ratio (`New/Old`) | Donor Cos (`12..15` vs `0..3`) | Delta RMS Old | Delta RMS New |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(head_rows)}

---

## 4. Zero-Initialized Residual & MLP Expansion Slices (`0..767` vs `768..1023`)

| Parameter Name | Expanded Slice Axis | Inherited RMS (`0..767`) | Expanded RMS (`768..1023`) | Ratio (`Exp/Base`) |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(slice_rows)}

---
*Engineered by uncoalesced*
"""
    return report, all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Session 1 Anchor Checkpoint for Parentheses 1.0-240M")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=r"E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M-s1",
        help="Path to final Session 1 checkpoint (.pt) or directory containing it",
    )
    parser.add_argument(
        "--init-ckpt",
        type=Path,
        default=Path(r"E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M\init_grown.pt"),
        help="Path to initial Net2WiderNet checkpoint for weight delta comparison",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(r"E:\Parentheses\parentheses-0.9\out\session1_anchor_validation.md"),
        help="Output path for the Markdown validation report",
    )
    args = parser.parse_args()

    try:
        s1_ckpt_path = resolve_latest_checkpoint(args.ckpt)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1

    report_md, all_passed = audit_checkpoint(s1_ckpt_path, args.init_ckpt)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report_md, encoding="utf-8")

    print(f"[+] Wrote validation report to: {args.out_md}\n")
    print(report_md)
    return 0 if all_passed else 2


if __name__ == "__main__":
    sys.exit(main())
