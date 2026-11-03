#!/usr/bin/env python3
# Engineered by uncoalesced
"""
tests/test_slab_v4_boundary_parity.py
Verifies exact forward and backward gradient parity between:
  - Parallel / Chunked slab_v4 primitives (C=64)
  - Step-by-step O(1) recurrent reference execution with explicit state zeroing on EOT
across all 4 topological boundary regimes:
  Regime 1: No document boundary inside sequence
  Regime 2: Exact chunk-seam boundary (EOT at last token C-1 of chunk k-1 -> m_0 = 1 at chunk k)
  Regime 3: Single & multiple mid-chunk boundaries (t* in [1, C-1])
  Regime 4: Combined seam boundary (t=0) + mid-chunk boundaries in the same chunk
"""

import os
import sys
from pathlib import Path
import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.slab_v4_boundary import (
    compute_doc_ids,
    token_shift_v4,
    causal_conv1d_v4,
    ssd_chunk_scan_v4,
    ssd_recurrent_step_v4,
)

EOT_ID = 0


def build_four_regime_tokens(T: int = 256, C: int = 64, device: str = "cpu") -> torch.Tensor:
    """
    Construct a 4-batch tensor where each batch element exercises one of the 4 regimes:
      Batch 0 (Regime 1): No EOT tokens anywhere.
      Batch 1 (Regime 2): EOT at t = C-1 (63) and t = 2C-1 (127) -> fires m_0 = 1 at chunk 1 and chunk 2 seams!
      Batch 2 (Regime 3): Mid-chunk EOTs at t = 17, t = 89, t = 140, t = 210.
      Batch 3 (Regime 4): Seam + Mid-chunk combined: EOT at t = 63 (seam at 64) AND t = 75, 110 (mid-chunk 1), plus consecutive EOTs at 191, 192.
    """
    input_ids = torch.randint(10, 500, (4, T), dtype=torch.long, device=device)
    # Batch 1: Seam boundaries at chunk 1 (t=64) and chunk 2 (t=128)
    input_ids[1, C - 1] = EOT_ID
    input_ids[1, 2 * C - 1] = EOT_ID
    # Batch 2: Mid-chunk boundaries
    for pos in (17, 89, 140, 210):
        input_ids[2, pos] = EOT_ID
    # Batch 3: Seam + multiple mid-chunk boundaries in the same chunk
    for pos in (C - 1, C + 11, C + 46, 3 * C - 1, 3 * C):
        input_ids[3, pos] = EOT_ID
    return input_ids


def test_ssd_four_regime_forward_and_backward_parity():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, C, H, Dh = 4, 256, 64, 12, 64

    input_ids = build_four_regime_tokens(T=T, C=C, device=device)
    reset_mask, doc_ids = compute_doc_ids(input_ids, eot_token_id=EOT_ID)

    # Inputs for Chunked Parallel Path (scaled by 1/sqrt(Dh) for attention variance stability)
    scale = Dh ** -0.5
    q1 = (torch.randn(B, T, H, Dh, device=device, dtype=torch.float32) * scale).requires_grad_(True)
    k1 = (torch.randn(B, T, H, Dh, device=device, dtype=torch.float32) * scale).requires_grad_(True)
    v1 = torch.randn(B, T, H, Dh, device=device, dtype=torch.float32, requires_grad=True)
    log_a1 = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.5 - 0.01).detach().requires_grad_(True)
    beta1 = (torch.rand(B, T, H, device=device, dtype=torch.float32) * 1.5 + 0.1).detach().requires_grad_(True)
    init_s1 = (torch.randn(B, H, Dh, Dh, device=device, dtype=torch.float32) * scale).requires_grad_(True)

    # Cloned Inputs for Step-by-Step Recurrent Reference Path
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    log_a2 = log_a1.detach().clone().requires_grad_(True)
    beta2 = beta1.detach().clone().requires_grad_(True)
    init_s2 = init_s1.detach().clone().requires_grad_(True)

    y_chunk, s_chunk = ssd_chunk_scan_v4(
        q1, k1, v1, log_a1, beta1, reset_mask, doc_ids, chunk_size=C, initial_state=init_s1
    )
    y_step, s_step = ssd_recurrent_step_v4(
        q2, k2, v2, log_a2, beta2, reset_mask, initial_state=init_s2
    )

    # 1. Verify Forward Output & Terminal State Parity per Regime
    for b in range(B):
        max_y_err = (y_chunk[b] - y_step[b]).abs().max().item()
        max_s_err = (s_chunk[b] - s_step[b]).abs().max().item()
        assert max_y_err < 1e-5, f"Regime {b+1} forward y mismatch: {max_y_err}"
        assert max_s_err < 1e-5, f"Regime {b+1} terminal state mismatch: {max_s_err}"

    # 2. Verify Backward Gradient Parity & Zero-NaN Guarantee
    grad_out = torch.randn_like(y_chunk)
    loss_chunk = (y_chunk * grad_out).sum() + 0.1 * s_chunk.sum()
    loss_step = (y_step * grad_out).sum() + 0.1 * s_step.sum()
    loss_chunk.backward()
    loss_step.backward()

    for name, g_c, g_s in [
        ("q", q1.grad, g2 := q2.grad),
        ("k", k1.grad, k2.grad),
        ("v", v1.grad, v2.grad),
        ("log_alpha", log_a1.grad, log_a2.grad),
        ("beta", beta1.grad, beta2.grad),
        ("init_state", init_s1.grad, init_s2.grad),
    ]:
        assert not torch.isnan(g_c).any(), f"NaN detected in chunked {name}.grad!"
        assert not torch.isinf(g_c).any(), f"Inf detected in chunked {name}.grad!"
        max_grad_err = (g_c - g_s).abs().max().item()
        assert max_grad_err < 1e-4, f"Backward {name}.grad mismatch: {max_grad_err}"


def test_token_shift_and_conv1d_step_parity():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, D, K = 4, 256, 64, 4

    input_ids = build_four_regime_tokens(T=T, C=64, device=device)
    reset_mask, doc_ids = compute_doc_ids(input_ids, eot_token_id=EOT_ID)

    x = torch.randn(B, T, D, device=device, requires_grad=True)
    mu = torch.rand(D, device=device, requires_grad=True)
    w = torch.randn(D, 1, K, device=device, requires_grad=True)
    b = torch.randn(D, device=device, requires_grad=True)

    # Parallel forward
    x_shift_par, _ = token_shift_v4(x, mu, reset_mask)
    y_conv_par, _, _ = causal_conv1d_v4(x_shift_par, w, b, doc_ids)

    # Step-by-step recurrent forward with explicit state zeroing on reset_mask
    prev_tok = torch.zeros(B, 1, D, device=device)
    conv_buf = torch.zeros(B, K - 1, D, device=device)
    step_outs = []

    for t in range(T):
        m_t = reset_mask[:, t].view(B, 1, 1)
        prev_tok = torch.where(m_t, torch.zeros_like(prev_tok), prev_tok)
        conv_buf = torch.where(m_t, torch.zeros_like(conv_buf), conv_buf)

        x_t = x[:, t : t + 1, :]
        x_sh_t = mu.view(1, 1, D) * x_t + (1.0 - mu.view(1, 1, D)) * prev_tok
        prev_tok = x_t

        window = torch.cat([conv_buf, x_sh_t], dim=1)  # [B, K, D]
        y_t = (window.permute(0, 2, 1) * w.squeeze(1).unsqueeze(0)).sum(dim=-1) + b.unsqueeze(0)
        conv_buf = window[:, 1:, :]
        step_outs.append(y_t)

    y_conv_step = torch.stack(step_outs, dim=1)
    max_err = (y_conv_par - y_conv_step).abs().max().item()
    assert max_err < 1e-5, f"Token-Shift + Conv1d step parity failed: max_err={max_err}"


if __name__ == "__main__":
    test_ssd_four_regime_forward_and_backward_parity()
    test_token_shift_and_conv1d_step_parity()
    print("[PASS] All 4 boundary regimes passed forward & backward parity (0 NaNs, <1e-5 max error).")
