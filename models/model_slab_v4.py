#!/usr/bin/env python3
# Engineered by uncoalesced
"""
models/model_slab_v4.py
Parentheses 1.0 (240M / slab_v4) Recurrent State Space Duality (SSD) Model.

Key Architectural Invariants:
  1. Wide-and-shallow aspect ratio: 16 layers, d_model=1024, 16 heads (rank 1024/layer), ~238M-253M params.
  2. Single Entry-Point Document Reset Wiring: compute_doc_ids(input_ids, eot_token_id=0) computes
     document reset mask m_t and cumulative document index d_t once at the model entry point.
  3. Tri-Channel Boundary Reset Isolation:
       - Channel 1 (token_shift_v4): mu-mixing with reset isolation at m_t == True.
       - Channel 2 (causal_conv1d_v4): depthwise causal conv1d (K=4) with document-ID tap masking.
       - Channel 3 (ssd_chunk_scan_v4): NaN-safe, seam-exact 4-regime chunked SSD associative scan (C=64).
  4. Layer-0 V0 Value Residual Highway: Layer 0 value activations propagate directly across all
     layers via a gated highway to ensure gradient flow and associative stability.
  5. Exact Step Parity: O(1) step inference with associative state reset on EOT boundaries.
  6. Block-Level Activation Checkpointing: Drops activation memory across 16 layers from ~3.4 GB
     to ~0.25 GB, permitting B_micro=2 at T=4096 inside 8 GB VRAM.
  7. Differential Parameter-Group Learning Rates: Configures 3 distinct parameter groups
     (matrix_projections, embeddings_and_norms, recurrent_dynamics) with 0.1x SFT ratio.
  8. Container-Aligned 4,096 DataLoader Integration: Zero cross-bin bleed, 100% C=64 divisibility.
"""

from __future__ import annotations
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

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
    slab_v4: bool = True

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
    def __init__(self, cfg: Union[SlabV4Config, Any]):
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
    def __init__(self, cfg: Union[SlabV4Config, Any], layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.chunk_size = getattr(cfg, "chunk_size", 64)

        # Channel 1: Token-Shift (mu-mixing parameter)
        self.mu = nn.Parameter(torch.full((self.n_embd,), 0.5))

        # Projections: Q, K, V in one linear pass
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=cfg.bias)

        # Channel 2: Depthwise Causal Conv1d (K=4)
        self.conv_kernel_size = 4
        self.conv_weight = nn.Parameter(
            torch.randn(3 * self.n_embd, 1, self.conv_kernel_size) * (1.0 / math.sqrt(self.conv_kernel_size))
        )
        self.conv_bias = nn.Parameter(torch.zeros(3 * self.n_embd))

        # Gated Decay Rate: log_alpha = -softplus(alpha_proj(x_ctrl))
        self.alpha_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
        self._init_decay_bias()

        # Beta-Gated Selective Write: beta = 2.0 * sigmoid(beta_proj(x_ctrl))
        self.beta_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
        with torch.no_grad():
            self.beta_proj.weight.zero_()
            self.beta_proj.bias.zero_()

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
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim

        # Channel 1: Document-boundary isolated Token-Shift
        x_ctrl, _ = token_shift_v4(x, self.mu, reset_mask)

        # QKV linear projection
        u = self.qkv(x)

        # Channel 2: Document-boundary isolated Depthwise Causal Conv1d
        u_conv, _, _ = causal_conv1d_v4(u, self.conv_weight, self.conv_bias, doc_ids)
        u_conv = F.silu(u_conv)

        qkv = u_conv.view(B, T, 3, H, Dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # L2 normalize Q and K in float32 for associative stability, scaled by 1/sqrt(head_dim)
        q = F.normalize(q.float(), dim=-1).to(q.dtype) * (1.0 / math.sqrt(Dh))
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
        )

        # Per-head RMSNorm
        y_float = y.float()
        y_normed = (y_float * torch.rsqrt(y_float.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype)
        w_norm = self.y_norm.view(1, 1, H, Dh).to(y.dtype)
        y = (y_normed * w_norm).reshape(B, T, C)

        # Output gate + linear projection (gated by x_ctrl)
        out = self.proj(y * F.silu(self.gate(x_ctrl)))
        return out, v0_out

    def step(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        is_boundary: bool = False,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
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
        token_next = x.unsqueeze(1)

        # 2. QKV projection + Causal Conv1d step
        u = self.qkv(x)
        window = torch.cat([conv_state, u.unsqueeze(1)], dim=1)
        w_conv = self.conv_weight.squeeze(1).unsqueeze(0)
        u_conv = (window.permute(0, 2, 1) * w_conv).sum(dim=-1)
        if self.conv_bias is not None:
            u_conv = u_conv + self.conv_bias.unsqueeze(0)
        u_conv = F.silu(u_conv)
        conv_next = window[:, 1:, :]

        qkv = u_conv.view(B, 1, 3, H, Dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = F.normalize(q.float(), dim=-1).to(q.dtype) * (1.0 / math.sqrt(Dh))
        k = F.normalize(k.float(), dim=-1).to(k.dtype)

        v0_out = v if self.layer_idx == 0 else v0
        if self.layer_idx > 0 and v0 is not None:
            mix = torch.sigmoid(self.v_residual_lamb).view(1, 1, H, 1).to(v.dtype)
            v = mix * v + (1.0 - mix) * v0

        beta = 2.0 * torch.sigmoid(self.beta_proj(x_ctrl)).unsqueeze(1)
        log_alpha = -F.softplus(self.alpha_proj(x_ctrl)).unsqueeze(1)

        reset_m = torch.tensor([[is_boundary]], device=x.device, dtype=torch.bool).expand(B, 1)

        y, S_next = ssd_recurrent_step_v4(
            q=q,
            k=k,
            v=v,
            log_alpha=log_alpha,
            beta=beta,
            reset_mask=reset_m,
            initial_state=S_prev,
        )

        y_float = y.float()
        y_normed = (y_float * torch.rsqrt(y_float.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype)
        y = (y_normed * self.y_norm.view(1, 1, H, Dh)).reshape(B, C)

        out = self.proj(y * F.silu(self.gate(x_ctrl)))
        return out, (S_next, conv_next, token_next), v0_out


class SlabV4Block(nn.Module):
    """Transformer-style pre-norm block wrapping SlabV4Attention and SwiGLU MLP."""
    def __init__(self, cfg: Union[SlabV4Config, Any], layer_idx: int = 0):
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
    Wide-and-shallow 16-layer architecture with single entry-point document reset wiring,
    block-level activation checkpointing, and segregated differential AdamW configuration.
    """
    def __init__(self, cfg: Optional[Union[SlabV4Config, Any]] = None):
        super().__init__()
        self.cfg = cfg or SlabV4Config()
        self.grad_checkpointing: bool = False
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

    def gradient_checkpointing_enable(self) -> None:
        """Enable block-level activation checkpointing to drop memory from ~3.4GB to ~0.25GB."""
        self.grad_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.grad_checkpointing = False

    def num_params(self, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            return sum(p.numel() for p in self.parameters()) - self.tok_emb.weight.numel()
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(
        self,
        lr: float = 3.0e-4,
        weight_decay: float = 0.1,
        recurrent_lr_ratio: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        use_fused: bool = True,
        use_8bit: bool = False,
    ) -> torch.optim.Optimizer:
        """
        Configure Segregated Differential AdamW optimizer.
        Separates all model parameters into 3 distinct groups:
          1. Matrix Projections (qkv, proj, gate, mlp): base_lr, weight_decay = 0.1
          2. Embeddings & RMSNorm (tok_emb, head, ln1, ln2, ln_f, y_norm, biases): base_lr, weight_decay = 0.0
          3. Recurrent Dynamics (alpha_proj, beta_proj, mu, v_residual_lamb, conv1d):
             recurrent_lr = base_lr * recurrent_lr_ratio, weight_decay = 0.0
        """
        matrix_params: List[torch.nn.Parameter] = []
        emb_norm_params: List[torch.nn.Parameter] = []
        recurrent_params: List[torch.nn.Parameter] = []

        recurrent_keywords = ("alpha", "beta", "mu", "conv", "v_residual")
        norm_keywords = ("ln", "norm", "bias")

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            name_lower = name.lower()
            if any(k in name_lower for k in recurrent_keywords):
                recurrent_params.append(p)
            elif "tok_emb" in name_lower or p.ndim < 2 or any(k in name_lower for k in norm_keywords):
                emb_norm_params.append(p)
            else:
                matrix_params.append(p)

        total_param_count = sum(len(p) for p in [matrix_params, emb_norm_params, recurrent_params])
        assert total_param_count == len(list(self.parameters())), (
            f"Parameter mismatch in optimizer grouping: {total_param_count} vs {len(list(self.parameters()))}"
        )

        groups = [
            {
                "params": matrix_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "matrix_projections",
            },
            {
                "params": emb_norm_params,
                "lr": lr,
                "weight_decay": 0.0,
                "name": "embeddings_and_norms",
            },
            {
                "params": recurrent_params,
                "lr": lr * recurrent_lr_ratio,
                "weight_decay": 0.0,
                "name": "recurrent_dynamics",
            },
        ]

        if use_8bit:
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
            except ImportError:
                print("[warn] bitsandbytes not installed, falling back to torch.optim.AdamW")

        fused = (
            use_fused
            and torch.cuda.is_available()
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)

    def chunked_loss(self, x: torch.Tensor, targets: torch.Tensor, rows: int = 4096) -> torch.Tensor:
        """Mean cross-entropy without materializing (B*T, vocab) logits."""
        ign = getattr(self.cfg, "ignore_index", -100)
        h, t = x.reshape(-1, x.size(-1)), targets.reshape(-1)
        if ign == -100 and (t == -1).any():
            t = t.clone().masked_fill_(t == -1, -100)
        part = lambda hh, tt: F.cross_entropy(self.head(hh).float(), tt, ignore_index=ign, reduction="sum")
        total = sum(checkpoint(part, h[i:i + rows], t[i:i + rows], use_reentrant=False)
                    for i in range(0, h.size(0), rows))
        valid_targets = ((t != ign) & (t >= 0)).sum().clamp_min(1)
        return total / valid_targets

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        reset_mask: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        use_checkpointing: bool = False,
        return_logits: bool = True,
        ce_rows: int = 4096,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward language modeling computation with single entry-point document reset wiring.
        """
        B, T = input_ids.shape
        eot_id = getattr(self.cfg, "eot_token_id", 0)

        # Single entry-point document reset wiring: computed once at model entry
        if reset_mask is not None and reset_mask.dtype in (torch.uint8, torch.int8):
            # Compatibility with train.py boundary_reset_mask (where 0 indicates document boundary)
            reset_mask = (reset_mask == 0)

        if reset_mask is None:
            reset_mask, computed_docs = compute_doc_ids(input_ids, eot_token_id=eot_id)
            if doc_ids is None:
                doc_ids = computed_docs
        elif doc_ids is None:
            doc_ids = torch.cumsum(reset_mask.to(torch.int64), dim=1)

        x = self.drop(self.tok_emb(input_ids))

        v0 = None
        ckpt = (use_checkpointing or getattr(self, "grad_checkpointing", False)) and self.training and torch.is_grad_enabled()

        for block in self.blocks:
            if ckpt:
                x, v0 = checkpoint(block, x, reset_mask, doc_ids, v0, use_reentrant=False)
            else:
                x, v0 = block(x, reset_mask, doc_ids, v0=v0)

        x = self.ln_f(x)
        if targets is not None and not return_logits:
            return None, self.chunked_loss(x, targets, rows=ce_rows)

        logits = self.head(x) if return_logits else None

        loss = None
        if targets is not None:
            ign = getattr(self.cfg, "ignore_index", -100)
            t_flat = targets.view(-1)
            if ign == -100 and (t_flat == -1).any():
                t_flat = t_flat.clone().masked_fill_(t_flat == -1, -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                t_flat,
                ignore_index=ign,
            )

        return logits, loss

    def empty_states(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> List[None]:
        return [None for _ in self.blocks]

    def step(
        self,
        token_id: torch.Tensor,
        states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = None,
        is_boundary: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
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


class SlabV4SFTDataset(Dataset):
    """
    Container-Aligned 4,096-Token Dataset for First-Fit-Decreasing (FFD) SFT Bins.
    Enforces exact 4,096 token boundary alignment:
      x:         row[:4096]
      y:         concat(row[1:4096], [EOT_ID])
      loss_mask: concat(mask[1:4096], [0])
      target y:  y.masked_fill(~loss_mask, ignore_index)
    """
    def __init__(
        self,
        bin_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        seq_len: int = 4096,
        eot_token_id: int = 0,
        ignore_index: int = -100,
        mmap: bool = True,
    ):
        self.seq_len = seq_len
        self.eot_token_id = eot_token_id
        self.ignore_index = ignore_index

        bin_p = Path(bin_path)
        if not bin_p.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_p}")

        if mmap:
            self.data = np.memmap(bin_p, dtype=np.uint16, mode="r").reshape(-1, seq_len)
        else:
            self.data = np.fromfile(bin_p, dtype=np.uint16).reshape(-1, seq_len)

        self.num_bins = len(self.data)

        if mask_path is not None:
            mask_p = Path(mask_path)
            if not mask_p.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_p}")
            if mmap:
                self.mask = np.memmap(mask_p, dtype=np.uint8, mode="r").reshape(-1, seq_len)
            else:
                self.mask = np.fromfile(mask_p, dtype=np.uint8).reshape(-1, seq_len)
            assert len(self.mask) == self.num_bins, (
                f"Data/Mask length mismatch: {self.num_bins} vs {len(self.mask)}"
            )
        else:
            self.mask = None

    def __len__(self) -> int:
        return self.num_bins

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_ids = torch.from_numpy(self.data[idx].astype(np.int64))

        x = row_ids[:self.seq_len]
        y = torch.cat([row_ids[1:self.seq_len], torch.tensor([self.eot_token_id], dtype=torch.int64)])

        if self.mask is not None:
            row_mask = torch.from_numpy(self.mask[idx].astype(bool))
            loss_mask = torch.cat([row_mask[1:self.seq_len], torch.tensor([False])])
        else:
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
            loss_mask[-1] = False

        targets = y.masked_fill(~loss_mask, self.ignore_index)
        return x, targets, loss_mask


def get_container_batch(
    data: Union[np.memmap, np.ndarray],
    mask: Optional[Union[np.memmap, np.ndarray]],
    block_size: int,
    batch_size: int,
    device: str,
    eot_token_id: int = 0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample 4,096-aligned FFD container bins with exact within-bin causal target shift.
    Guarantees:
      - x: [B, block_size] (length 4096, divisible by C=64)
      - y: [B, block_size] (shift within container, terminal position padded with EOT and masked)
      - zero cross-bin leakage
      - reset_mask: [B, block_size] (0 at document starts, 1 elsewhere, matching train.py convention)
    """
    num_bins = len(data) // block_size
    bin_ix = torch.randint(num_bins, (batch_size,))

    x_list, y_list, m_list = [], [], []
    for b in bin_ix:
        start = int(b) * block_size
        end = start + block_size
        row_x = torch.from_numpy(data[start:end].astype(np.int64))
        row_y = torch.cat([
            torch.from_numpy(data[start + 1:end].astype(np.int64)),
            torch.tensor([eot_token_id], dtype=torch.int64),
        ])
        if mask is not None:
            row_m = torch.cat([
                torch.from_numpy(mask[start + 1:end].astype(bool)),
                torch.tensor([False]),
            ])
        else:
            row_m = torch.ones(block_size, dtype=torch.bool)
            row_m[-1] = False

        row_y = row_y.masked_fill(~row_m, ignore_index)
        x_list.append(row_x)
        y_list.append(row_y)
        m_list.append(row_m)

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # Compute boundary reset mask directly from x (m_t = 1 at t if x_{t-1} == EOT)
    reset_m = torch.zeros((batch_size, block_size), dtype=torch.bool)
    if block_size > 1:
        reset_m[:, 1:] = (x[:, :-1] == eot_token_id)
    reset_mask_uint8 = (~reset_m).to(torch.uint8)

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        reset_mask_uint8 = reset_mask_uint8.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        reset_mask_uint8 = reset_mask_uint8.to(device)

    return x, y, reset_mask_uint8
