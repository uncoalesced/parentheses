#!/usr/bin/env python3
# Engineered by uncoalesced
"""
models/slab_v4_boundary.py
Parentheses 1.0 (240M / slab_v4) Tri-Channel Document Boundary Reset Primitives:
  1. compute_doc_ids: O(1) boolean reset mask m_t and cumulative document ID d_t
  2. token_shift_v4: Document-boundary-isolated Token-Shift (mu-mixing)
  3. causal_conv1d_v4: Document-boundary-isolated Depthwise Causal Conv1d (K=4)
  4. ssd_chunk_scan_v4: NaN-safe, seam-exact 4-regime Chunked SSD Associative Scan (C=64)
  5. ssd_recurrent_step_v4: Ground-truth O(1) recurrent step reference with EOT reset
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_doc_ids(
    input_ids: torch.Tensor,
    eot_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the document-start reset mask m_t and cumulative document ID d_t.
    A reset occurs at token t (m_t = 1) iff token t-1 was <|endoftext|>.
    Token t=0 at the start of a sequence has m_0 = 0 unless preceded by an external EOT flag.

    Args:
        input_ids: [B, T] integer token IDs
        eot_token_id: integer ID of <|endoftext|>

    Returns:
        reset_mask: [B, T] boolean tensor (True at t_EOT + 1)
        doc_ids:    [B, T] int64 cumulative document index d_t = sum_{tau=0}^t m_tau
    """
    B, T = input_ids.shape
    reset_mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
    if T > 1:
        reset_mask[:, 1:] = (input_ids[:, :-1] == eot_token_id)
    doc_ids = torch.cumsum(reset_mask.to(torch.int64), dim=1)
    return reset_mask, doc_ids


def token_shift_v4(
    x: torch.Tensor,
    mu: torch.Tensor,
    reset_mask: torch.Tensor,
    prev_token_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Channel-wise Token-Shift (mu-mixing) with strict document-boundary isolation.
    At m_t == True, forces mu_effective = 1 so x_tilde_t = x_t (zero bleed from x_{t-1}).

    Args:
        x:                [B, T, D] input activations
        mu:               [D] or [1, 1, D] learnable shift parameter
        reset_mask:       [B, T] boolean reset mask (True at document start t_EOT + 1)
        prev_token_state: Optional [B, 1, D] activation from previous chunk/step

    Returns:
        x_shifted:       [B, T, D] mixed activations
        next_prev_state: [B, 1, D] final token activation for stateful streaming
    """
    B, T, D = x.shape
    if prev_token_state is None:
        prev_token_state = torch.zeros((B, 1, D), dtype=x.dtype, device=x.device)

    x_prev = torch.cat([prev_token_state, x[:, :-1, :]], dim=1)  # [B, T, D]
    # Zero out x_prev wherever reset_mask is True (equivalent to forcing mu = 1 at m_t = 1)
    keep_prev = (~reset_mask).unsqueeze(-1).to(x.dtype)          # [B, T, 1]
    mu_view = mu.view(1, 1, D)

    x_shifted = mu_view * x + (1.0 - mu_view) * (x_prev * keep_prev)
    next_prev_state = x[:, -1:, :]
    return x_shifted, next_prev_state


def causal_conv1d_v4(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    doc_ids: torch.Tensor,
    conv_state: Optional[torch.Tensor] = None,
    state_doc_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Depthwise causal 1D convolution (kernel size K=4) with branchless document-ID tap masking.
    Tap k in {0, 1, 2, 3} contributes to output t iff d_t == d_{t-k}.

    Args:
        x:             [B, T, D] input tensor
        weight:        [D, 1, K] depthwise conv weights (K=4)
        bias:          Optional [D] conv bias
        doc_ids:       [B, T] int64 cumulative document IDs
        conv_state:    Optional [B, K-1, D] previous activation buffer
        state_doc_ids: Optional [B, K-1] int64 document IDs for conv_state positions

    Returns:
        out:            [B, T, D] convolved output
        next_conv_state:[B, K-1, D] updated conv buffer
        next_state_ids: [B, K-1] updated document IDs for conv buffer
    """
    B, T, D = x.shape
    K = weight.shape[-1]
    pad = K - 1

    if conv_state is None:
        conv_state = torch.zeros((B, pad, D), dtype=x.dtype, device=x.device)
    if state_doc_ids is None:
        # Default to doc_id 0 (if m_0 is True, doc_ids[:, 0] is 1, so state_doc_ids=0 is automatically masked)
        state_doc_ids = torch.zeros((B, pad), dtype=doc_ids.dtype, device=doc_ids.device)

    x_padded = torch.cat([conv_state, x], dim=1)                  # [B, T + K - 1, D]
    d_padded = torch.cat([state_doc_ids, doc_ids], dim=1)         # [B, T + K - 1]

    # Unfold sliding windows of length K along sequence dimension: [B, T, D, K]
    x_windows = x_padded.unfold(dimension=1, size=K, step=1)      # [B, T, D, K]
    d_windows = d_padded.unfold(dimension=1, size=K, step=1)      # [B, T, K]

    # Current token's document ID is the rightmost element of each window (index K-1)
    d_curr = d_windows[:, :, -1:].expand_as(d_windows)            # [B, T, K]
    tap_mask = (d_windows == d_curr).unsqueeze(2).to(x.dtype)     # [B, T, 1, K]

    w = weight.squeeze(1).view(1, 1, D, K)                        # [1, 1, D, K]
    out = (x_windows * tap_mask * w).sum(dim=-1)                  # [B, T, D]

    if bias is not None:
        out = out + bias.view(1, 1, D)

    next_conv_state = x_padded[:, -pad:, :]
    next_state_ids = d_padded[:, -pad:]
    return out, next_conv_state, next_state_ids


def ssd_chunk_scan_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    reset_mask: torch.Tensor,
    doc_ids: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NaN-safe, seam-exact Chunked State Space Duality (SSD) Associative Scan.
    Implements Section 9.1 of ARCHITECTURE.md:
      - Finite prefix sums P_t = cumsum(log_alpha_t) in FP32
      - Intra-chunk (CxC): L_{i,j} = exp(P_i - P_j) * I[i >= j] * I[d_i == d_j]
      - Chunk-to-state write (B_k): w_out(j) = exp(P_{C-1} - P_j) * I[d_{C-1} == d_j]
      - Inter-chunk decay (A_k):    A_k = exp(P_{C-1}) * I[d_{C-1} == d_0 - m_0]
      - State-to-chunk read (C_k):  w_in(i) = exp(P_i) * I[d_i == d_0 - m_0]

    Args:
        q, k, v:       [B, T, H, Dh] query, key, value tensors
        log_alpha:     [B, T, H] log-decay in (-inf, 0] (kept strictly finite)
        beta:          [B, T, H] selective write gate in (0, 2)
        reset_mask:    [B, T] boolean reset mask m_t
        doc_ids:       [B, T] int64 cumulative document IDs d_t
        chunk_size:    int chunk size C (default 64; T must be divisible by C)
        initial_state: Optional [B, H, Dh, Dh] incoming recurrent state S_{-1}

    Returns:
        y:           [B, T, H, Dh] output tensor
        final_state: [B, H, Dh, Dh] terminal associative state S_{N-1}
    """
    B, T, H, Dh = q.shape
    C = chunk_size
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size C={C}"
    N = T // C

    orig_dtype = q.dtype
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = (v * beta.unsqueeze(-1)).to(torch.float32)  # Fold beta write gate into v
    log_a_f = log_alpha.to(torch.float32)

    # Reshape into chunks: [B, N, C, ...]
    q_c = q_f.view(B, N, C, H, Dh).permute(0, 1, 3, 2, 4)        # [B, N, H, C, Dh]
    k_c = k_f.view(B, N, C, H, Dh).permute(0, 1, 3, 2, 4)        # [B, N, H, C, Dh]
    v_c = v_f.view(B, N, C, H, Dh).permute(0, 1, 3, 2, 4)        # [B, N, H, C, Dh]
    log_a_c = log_a_f.view(B, N, C, H).permute(0, 1, 3, 2)       # [B, N, H, C]

    d_c = doc_ids.view(B, N, C)                                  # [B, N, C]
    m_c = reset_mask.view(B, N, C)                               # [B, N, C]

    # Uncorrupted finite prefix sum within each chunk: P_t = sum_{tau=0}^t log_alpha_tau
    P = torch.cumsum(log_a_c, dim=-1)                            # [B, N, H, C]

    # 1. Intra-Chunk Lower-Triangular Matrix L_{i,j} [B, N, H, C, C]
    # Note: P_i - P_j excludes log_alpha_j, which is exact for both normal & reset tokens!
    P_diff = P.unsqueeze(-1) - P.unsqueeze(-2)                   # [B, N, H, C, C]
    tril_mask = torch.tril(torch.ones((C, C), dtype=torch.bool, device=q.device))
    same_doc_intra = (d_c.unsqueeze(-1) == d_c.unsqueeze(-2)).unsqueeze(2)  # [B, N, 1, C, C]
    valid_intra = tril_mask.view(1, 1, 1, C, C) & same_doc_intra

    # Mask P_diff before exp to prevent exp(>0) overflow for upper-triangular j > i entries
    P_diff_safe = P_diff.masked_fill(~valid_intra, -1e30)
    L = torch.exp(P_diff_safe) * valid_intra.to(torch.float32)   # [B, N, H, C, C]

    qk = torch.matmul(q_c, k_c.transpose(-1, -2))                # [B, N, H, C, C]
    y_intra = torch.matmul(qk * L, v_c)                          # [B, N, H, C, Dh]

    # 2. Seam-Safe Document Anchor d_prev = d_0 - m_0 for incoming state S_{k-1}
    d_0 = d_c[:, :, 0]                                           # [B, N]
    m_0 = m_c[:, :, 0].to(d_c.dtype)                             # [B, N]
    d_prev = d_0 - m_0                                           # [B, N]
    d_last = d_c[:, :, -1]                                       # [B, N]

    # 3. Chunk-to-State Write Weights w_out(j) = exp(P_{C-1} - P_j) * I[d_{C-1} == d_j]
    same_as_last = (d_c == d_last.unsqueeze(-1)).unsqueeze(2)    # [B, N, 1, C]
    P_out_diff = (P[:, :, :, -1:] - P).masked_fill(~same_as_last, -1e30)
    w_out = torch.exp(P_out_diff) * same_as_last.to(torch.float32)  # [B, N, H, C]

    # 4. Inter-Chunk State Decay Scalar A_k = exp(P_{C-1}) * I[d_{C-1} == d_0 - m_0]
    no_reset_in_chunk = (d_last == d_prev).unsqueeze(-1)         # [B, N, 1]
    A_k = torch.exp(P[:, :, :, -1]) * no_reset_in_chunk.to(torch.float32)  # [B, N, H]

    # 5. State-to-Chunk Read Weights w_in(i) = exp(P_i) * I[d_i == d_0 - m_0]
    same_as_prev = (d_c == d_prev.unsqueeze(-1)).unsqueeze(2)    # [B, N, 1, C]
    P_in_safe = P.masked_fill(~same_as_prev, -1e30)
    w_in = torch.exp(P_in_safe) * same_as_prev.to(torch.float32) # [B, N, H, C]

    # 6. Inter-Chunk FP32 Scan over N chunks
    if initial_state is None:
        S = torch.zeros((B, H, Dh, Dh), dtype=torch.float32, device=q.device)
    else:
        S = initial_state.to(torch.float32)

    k_weighted = k_c * w_out.unsqueeze(-1)                       # [B, N, H, C, Dh]
    U_chunk = torch.matmul(k_weighted.transpose(-1, -2), v_c)    # [B, N, H, Dh, Dh]

    y_inter_chunks = []
    for c_idx in range(N):
        q_in = q_c[:, c_idx] * w_in[:, c_idx].unsqueeze(-1)      # [B, H, C, Dh]
        y_inter_c = torch.matmul(q_in, S)                        # [B, H, C, Dh]
        y_inter_chunks.append(y_inter_c)
        S = A_k[:, c_idx].view(B, H, 1, 1) * S + U_chunk[:, c_idx]

    y_inter = torch.stack(y_inter_chunks, dim=1)                 # [B, N, H, C, Dh]
    y_total = (y_intra + y_inter).permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, Dh)
    return y_total.to(orig_dtype), S


def ssd_recurrent_step_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    reset_mask: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ground-truth step-by-step O(1) recurrent reference implementation.
    At reset_mask[:, t] == True, forces alpha_t = 0 so S_t = 0 * S_{t-1} + beta_t * (k_t ^T v_t).
    """
    B, T, H, Dh = q.shape
    orig_dtype = q.dtype
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    alpha_f = torch.exp(log_alpha.to(torch.float32))
    beta_f = beta.to(torch.float32)

    if initial_state is None:
        S = torch.zeros((B, H, Dh, Dh), dtype=torch.float32, device=q.device)
    else:
        S = initial_state.to(torch.float32)

    outputs = []
    for t in range(T):
        keep = (~reset_mask[:, t]).view(B, 1, 1, 1).to(torch.float32)
        a_t = alpha_f[:, t].view(B, H, 1, 1) * keep
        b_t = beta_f[:, t].view(B, H, 1, 1)
        kv_t = torch.matmul(k_f[:, t].unsqueeze(-1), v_f[:, t].unsqueeze(-2))  # [B, H, Dh, Dh]
        S = a_t * S + b_t * kv_t
        y_t = torch.matmul(q_f[:, t].unsqueeze(-2), S).squeeze(-2)             # [B, H, Dh]
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1).to(orig_dtype)
    return y, S
