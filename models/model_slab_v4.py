#!/usr/bin/env python3
# Engineered by uncoalesced
"""
models/model_slab_v4.py
Parentheses 1.0 (240M / slab_v4) Recurrent State Space Duality (SSD) Model.

Key Architectural Invariants:
  1. Wide-and-shallow aspect ratio: 16 layers, d_model=1024, 16 heads (rank 1024/layer), ~238M params.
  2. Single Entry-Point Document Reset Wiring: compute_doc_ids(input_ids, eot_token_id=0) computes
     document reset mask m_t and cumulative document index d_t once at the model entry point.
  3. Tri-Channel Boundary Reset Isolation:
       - Channel 1 (token_shift_v4): mu-mixing with reset isolation at m_t == True.
       - Channel 2 (causal_conv1d_v4): depthwise causal conv1d (K=4) with document-ID tap masking.
       - Channel 3 (ssd_chunk_scan_v4): NaN-safe, seam-exact 4-regime chunked SSD associative scan (C=64).
  4. Layer-0 V0 Value Residual Highway: Layer 0 value activations propagate directly across all
     layers via a gated highway to ensure gradient flow and associative stability.
  5. Exact Step Parity: O(1) step inference with associative state reset on EOT boundaries.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.slab_v4_boundary import (
    compute_doc_ids,
    token_shift_v4,
    causal_conv1d_v4,
    ssd_chunk_scan_v4,
    ssd_recurrent_step_v4,
)


@dataclass
class SlabV4Config:
    vocab_size: int = 32768        # v3_32k BPE tokenizer
    block_size: int = 4096        # Max sequence length (4,096 tokens)
    n_layer: int = 16             # Wide-and-shallow depth
    n_head: int = 16              # 16 heads (head_dim = 64)
    n_embd: int = 1024            # d_model = 1024 (rank 1024 per layer)
    chunk_size: int = 64          # Chunk size C for chunked SSD scan
    eot_token_id: int = 0         # <|endoftext|> token ID
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    ignore_index: int = -100

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, cfg: SlabV4Config):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SlabV4Attention(nn.Module):
    """
    Parentheses 1.0 (slab_v4) Selective Linear Attention Layer.
    Implements Token-Shift (mu-mix), Depthwise Causal Conv1d (K=4),
    Chunked SSD Scan (C=64), Beta-Gated Selective Write, and V0 Highway.
    """
    def __init__(self, cfg: SlabV4Config, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.chunk_size = cfg.chunk_size

        # Channel 1: Token-Shift (mu-mixing parameter)
        self.mu = nn.Parameter(torch.full((self.n_embd,), 0.5))

        # Projections: Q, K, V in one linear pass
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=cfg.bias)

        # Channel 2: Depthwise Causal Conv1d (K=4)
        self.conv_kernel_size = 4
        self.conv_weight = nn.Parameter(
            torch.randn(3 * self.n_embd, 1, self.conv_kernel_size) * (1.0 / math.sqrt(self.conv_kernel_size))
        )
        self.conv_bias = nn.Parameter(torch.zeros(3 * self.n_embd)) if cfg.bias else None

        # Gated Decay Rate: log_alpha = -softplus(alpha_proj(x_ctrl))
        self.alpha_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
        self._init_decay_bias()

        # Beta-Gated Selective Write: beta = 2.0 * sigmoid(beta_proj(x_ctrl))
        self.beta_proj = nn.Linear(self.n_embd, self.n_head, bias=False)

        # Layer-0 V0 Value Residual Highway
        if self.layer_idx > 0:
            self.v_residual_lamb = nn.Parameter(torch.zeros(self.n_head))

        # Per-head output RMSNorm and SiLU gate
        self.y_norm = nn.Parameter(torch.ones(self.n_head, self.head_dim))
        self.gate = nn.Linear(self.n_embd, self.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=cfg.bias)

    def _init_decay_bias(self) -> None:
        """Log-uniform spread initialization for decay rates (alpha in ~0.5 to 0.999)."""
        with torch.no_grad():
            r = torch.linspace(0.5, 0.999, self.n_head)
            target_log_alpha = torch.log(r)
            b = torch.log(torch.exp(-target_log_alpha) - 1.0)
            self.alpha_proj.bias.copy_(b)
            self.alpha_proj.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        reset_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward associative linear attention across sequence length T (must be multiple of C=64).

        Args:
            x:          [B, T, n_embd] input activations
            reset_mask: [B, T] boolean reset mask (True at document boundaries t_EOT + 1)
            doc_ids:    [B, T] int64 cumulative document indices
            v0:         Optional [B, T, n_head, head_dim] Layer-0 value tensor

        Returns:
            out:    [B, T, n_embd] layer output
            v0_out: [B, T, n_head, head_dim] outgoing value tensor for highway
        """
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim

        # Channel 1: Document-boundary isolated Token-Shift
        x_ctrl, _ = token_shift_v4(x, self.mu, reset_mask)

        # QKV linear projection
        u = self.qkv(x)  # [B, T, 3 * n_embd]

        # Channel 2: Document-boundary isolated Depthwise Causal Conv1d
        u_conv, _, _ = causal_conv1d_v4(u, self.conv_weight, self.conv_bias, doc_ids)

        qkv = u_conv.view(B, T, 3, H, Dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, T, H, Dh]

        # L2 normalize Q and K in float32 for associative stability
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)

        # V0 Value Residual Highway
        v0_out = v if self.layer_idx == 0 else v0
        if self.layer_idx > 0 and v0 is not None:
            mix = torch.sigmoid(self.v_residual_lamb).view(1, 1, H, 1).to(v.dtype)
            v = mix * v + (1.0 - mix) * v0

        # Beta-Gated Selective Write: [B, T, H]
        beta = 2.0 * torch.sigmoid(self.beta_proj(x_ctrl))

        # Decay gate: log_alpha <= 0 strictly: [B, T, H]
        log_alpha = -F.softplus(self.alpha_proj(x_ctrl))

        # Channel 3: NaN-safe, seam-exact 4-regime chunked SSD scan
        y, _ = ssd_chunk_scan_v4(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            beta=beta,
            reset_mask=reset_mask,
            doc_ids=doc_ids,
            chunk_size=self.chunk_size,
        )  # [B, T, H, Dh]

        # Per-head RMSNorm
        y_float = y.float()
        y_normed = (y_float * torch.rsqrt(y_float.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype)
        w_norm = self.y_norm.view(1, 1, H, Dh).to(y.dtype)
        y = (y_normed * w_norm).reshape(B, T, C)

        # Output gate + linear projection
        out = self.proj(y * F.silu(self.gate(x)))
        return out, v0_out

    def step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        is_boundary: bool = False,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Recurrent O(1) step for generation: one token through the layer.

        Args:
            x:           [B, n_embd] input token activations
            state:       Optional (S_prev, conv_state, token_state)
            is_boundary: True if a document boundary occurs before this token
            v0:          Optional [B, 1, n_head, head_dim] Layer-0 value vector

        Returns:
            out:       [B, n_embd] step output
            new_state: (S_next, conv_next, token_next)
            v0_out:    [B, 1, n_head, head_dim] Layer-0 value vector for highway
        """
        B, C = x.shape
        H, Dh = self.n_head, self.head_dim
        pad = self.conv_kernel_size - 1

        if state is None:
            S_prev = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=torch.float32)
            conv_state = torch.zeros(B, pad, 3 * C, device=x.device, dtype=x.dtype)
            token_state = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        else:
            S_prev, conv_state, token_state = state

        # 1. Token-Shift step with boundary reset
        if is_boundary:
            token_state = torch.zeros_like(token_state)
            conv_state = torch.zeros_like(conv_state)

        x_ctrl = self.mu.view(1, C) * x + (1.0 - self.mu.view(1, C)) * token_state.squeeze(1)
        token_next = x.unsqueeze(1)  # [B, 1, C]

        # 2. QKV projection + Causal Conv1d step
        u = self.qkv(x)  # [B, 3*C]
        window = torch.cat([conv_state, u.unsqueeze(1)], dim=1)  # [B, K, 3*C]
        w_conv = self.conv_weight.squeeze(1).unsqueeze(0)        # [1, 3*C, K]
        u_conv = (window.permute(0, 2, 1) * w_conv).sum(dim=-1)  # [B, 3*C]
        if self.conv_bias is not None:
            u_conv = u_conv + self.conv_bias.unsqueeze(0)
        conv_next = window[:, 1:, :]  # [B, pad, 3*C]

        qkv = u_conv.view(B, 1, 3, H, Dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, 1, H, Dh]

        # L2 norm
        q = F.normalize(q.float(), dim=-1).to(q.dtype)
        k = F.normalize(k.float(), dim=-1).to(k.dtype)

        # V0 Value Highway
        v0_out = v if self.layer_idx == 0 else v0
        if self.layer_idx > 0 and v0 is not None:
            mix = torch.sigmoid(self.v_residual_lamb).view(1, 1, H, 1).to(v.dtype)
            v = mix * v + (1.0 - mix) * v0

        # Beta gate: [B, 1, H]
        beta = 2.0 * torch.sigmoid(self.beta_proj(x_ctrl)).unsqueeze(1)

        # Alpha decay: log_alpha: [B, 1, H]
        log_alpha = -F.softplus(self.alpha_proj(x_ctrl)).unsqueeze(1)

        reset_m = torch.tensor([[is_boundary]], device=x.device, dtype=torch.bool).expand(B, 1)

        # Recurrent O(1) step update
        y, S_next = ssd_recurrent_step_v4(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            beta=beta,
            reset_mask=reset_m,
            initial_state=S_prev,
        )  # y is [B, 1, H, Dh]

        # Per-head RMSNorm
        y_float = y.float()
        y_normed = (y_float * torch.rsqrt(y_float.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype)
        y = (y_normed * self.y_norm.view(1, 1, H, Dh)).reshape(B, C)

        out = self.proj(y * F.silu(self.gate(x)))
        return out, (S_next, conv_next, token_next), v0_out


class SlabV4Block(nn.Module):
    """Transformer-style pre-norm block wrapping SlabV4Attention and SwiGLU MLP."""
    def __init__(self, cfg: SlabV4Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SlabV4Attention(cfg, layer_idx=layer_idx)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        reset_mask: torch.Tensor,
        doc_ids: torch.Tensor,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, v0_out = self.attn(self.ln1(x), reset_mask, doc_ids, v0=v0)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, v0_out

    def step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        is_boundary: bool = False,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        attn_out, new_state, v0_out = self.attn.step(self.ln1(x), state, is_boundary=is_boundary, v0=v0)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return (x, new_state), v0_out


class SlabV4Model(nn.Module):
    """
    Parentheses 1.0 (240M / slab_v4) Full Causal Language Model.
    Wide-and-shallow 16-layer architecture with single entry-point document reset wiring.
    """
    def __init__(self, cfg: Optional[SlabV4Config] = None):
        super().__init__()
        self.cfg = cfg or SlabV4Config()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([SlabV4Block(self.cfg, layer_idx=i) for i in range(self.cfg.n_layer)])
        self.ln_f = RMSNorm(self.cfg.n_embd)
        self.head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tying input and output embeddings
        if self.cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        # Re-apply decay projection biases after global init
        for block in self.blocks:
            block.attn._init_decay_bias()

        # Residual depth scaling (1 / sqrt(2 * n_layer))
        residual_std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            return sum(p.numel() for p in self.parameters()) - self.tok_emb.weight.numel()
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        use_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward language modeling computation with single entry-point document reset wiring.

        Args:
            input_ids:         [B, T] token IDs
            targets:           Optional [B, T] next-token targets
            reset_mask:        Optional [B, T] boolean reset mask (auto-computed if None)
            doc_ids:           Optional [B, T] int64 document IDs (auto-computed if None)
            use_checkpointing: Whether to enable activation checkpointing across layers

        Returns:
            logits: [B, T, vocab_size]
            loss:   Scalar cross-entropy loss (if targets provided) or None
        """
        B, T = input_ids.shape

        # Single entry-point document reset wiring: computed once at model entry
        if reset_mask is None or doc_ids is None:
            computed_mask, computed_docs = compute_doc_ids(input_ids, eot_token_id=self.cfg.eot_token_id)
            if reset_mask is None:
                reset_mask = computed_mask
            if doc_ids is None:
                doc_ids = computed_docs

        x = self.drop(self.tok_emb(input_ids))

        v0 = None
        for block in self.blocks:
            if use_checkpointing and self.training:
                x, v0 = checkpoint(block, x, reset_mask, doc_ids, v0, use_reentrant=False)
            else:
                x, v0 = block(x, reset_mask, doc_ids, v0=v0)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.ignore_index,
            )

        return logits, loss

    def empty_states(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> List[None]:
        """Returns empty recurrent states for streaming/step generation."""
        return [None for _ in self.blocks]

    def step(
        self,
        token_id: torch.Tensor,
        states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = None,
        is_boundary: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Streaming O(1) single-token recurrent generation step.

        Args:
            token_id:    [B] or [B, 1] integer token tensor
            states:      List of per-layer recurrent states
            is_boundary: True if an EOT boundary occurred before this token

        Returns:
            logits:     [B, vocab_size] next-token prediction logits
            new_states: List of updated layer recurrent states
        """
        if token_id.dim() == 2:
            assert token_id.size(1) == 1, f"step() takes 1 token at a time, got shape {token_id.shape}"
            token_id = token_id[:, 0]

        if states is None:
            states = self.empty_states(token_id.size(0), device=token_id.device)

        x = self.drop(self.tok_emb(token_id))

        new_states = []
        v0 = None
        for block, state in zip(self.blocks, states):
            (x, new_s), v0 = block.step(x, state=state, is_boundary=is_boundary, v0=v0)
            new_states.append(new_s)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_states
