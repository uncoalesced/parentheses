"""
Vocabulary surgery and warm-start routines for Parentheses.

Engineered by uncoalesced

Expands model embedding and LM head dimensions when transitioning between tokenizers
(e.g., from Plan One byte-level vocabulary to Plan Two Akshara subword vocabulary).
Implements norm-preserving centroid pooling with symmetry-breaking Gaussian noise
to eliminate variance collapse and representation shock.

Reference: docs/FORMAL_MATHEMATICAL_SPECIFICATION.md Section 8.
"""

from typing import Callable, List, Tuple
import torch
import torch.nn as nn


def warm_start_embeddings(
    old_embedding: nn.Embedding,
    old_lm_head: nn.Linear | None,
    new_vocab_size: int,
    decompose_fn: Callable[[int], List[int]],
    tie_embeddings: bool = False,
    noise_scale: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[nn.Embedding, nn.Linear | None]:
    """
    Expands and warm-starts embedding and linear output projection layers.
    
    Args:
        old_embedding: Pretrained nn.Embedding (V_old, D).
        old_lm_head: Pretrained nn.Linear (D, V_old) or None if tied.
        new_vocab_size: Target vocabulary dimension V_new.
        decompose_fn: Function mapping new_token_id -> list of old_token_ids.
        tie_embeddings: If True, ties new_lm_head.weight directly to new_embedding.weight.
        noise_scale: Relative variance scaling for symmetry breaking.
        eps: Numerical stability constant.
    """
    old_e_weight = old_embedding.weight.data
    v_old, dim = old_e_weight.shape
    device = old_e_weight.device
    dtype = old_e_weight.dtype

    # 1. Compute empirical baseline metrics from base embeddings
    old_e_norms = torch.norm(old_e_weight, p=2, dim=-1, keepdim=True)  # (V_old, 1)
    target_e_norm = old_e_norms.mean().item()
    e_std = old_e_weight.std().item()

    # 2. Allocate new input embedding weights
    new_e_weight = torch.empty((new_vocab_size, dim), device=device, dtype=dtype)

    # 3. Handle output projection (LM Head)
    has_head = old_lm_head is not None and not tie_embeddings
    if has_head:
        old_u_weight = old_lm_head.weight.data  # (V_old, D)
        old_u_norms = torch.norm(old_u_weight, p=2, dim=-1, keepdim=True)
        target_u_norm = old_u_norms.mean().item()
        u_std = old_u_weight.std().item()
        new_u_weight = torch.empty((new_vocab_size, dim), device=device, dtype=dtype)

        if old_lm_head.bias is not None:
            old_bias = old_lm_head.bias.data
            target_bias = old_bias.mean().item()
            new_bias = torch.full((new_vocab_size,), target_bias, device=device, dtype=dtype)
        else:
            new_bias = None
    else:
        new_bias = None

    # 4. Vectorized/composite token assignment
    for new_idx in range(new_vocab_size):
        constituent_ids = decompose_fn(new_idx)
        
        # Edge case fallback: unmapped tokens default to mean token representation
        if not constituent_ids:
            constituent_ids = list(range(min(v_old, 256)))

        sub_indices = torch.tensor(constituent_ids, dtype=torch.long, device=device)

        # Input embedding warm-start
        e_slice = old_e_weight[sub_indices]  # (k, D)
        e_mean = e_slice.mean(dim=0)          # (D,)
        e_norm = torch.norm(e_mean, p=2) + eps
        scaled_e = (e_mean / e_norm) * target_e_norm
        
        # Add symmetry breaking noise if composite token (k > 1)
        if len(constituent_ids) > 1:
            noise_e = torch.randn_like(scaled_e) * (e_std * noise_scale)
            scaled_e = scaled_e + noise_e
            
        new_e_weight[new_idx] = scaled_e

        # Output projection warm-start (only if untied)
        if has_head:
            u_slice = old_u_weight[sub_indices]
            u_mean = u_slice.mean(dim=0)
            u_norm = torch.norm(u_mean, p=2) + eps
            scaled_u = (u_mean / u_norm) * target_u_norm
            
            if len(constituent_ids) > 1:
                noise_u = torch.randn_like(scaled_u) * (u_std * noise_scale)
                scaled_u = scaled_u + noise_u
                
            new_u_weight[new_idx] = scaled_u
            
            if new_bias is not None and len(constituent_ids) == 1:
                new_bias[new_idx] = old_bias[sub_indices[0]]

    # 5. Build replacement modules
    new_embedding = nn.Embedding(new_vocab_size, dim).to(device=device, dtype=dtype)
    new_embedding.weight.data.copy_(new_e_weight)

    if tie_embeddings:
        new_lm_head = nn.Linear(dim, new_vocab_size, bias=False).to(device=device, dtype=dtype)
        new_lm_head.weight = new_embedding.weight
    elif has_head:
        new_lm_head = nn.Linear(dim, new_vocab_size, bias=(new_bias is not None)).to(device=device, dtype=dtype)
        new_lm_head.weight.data.copy_(new_u_weight)
        if new_bias is not None:
            new_lm_head.bias.data.copy_(new_bias)
    else:
        new_lm_head = None

    return new_embedding, new_lm_head


def freeze_backbone_for_alignment(model: nn.Module):
    """
    Stage 1: Freezes all recurrent/attention backbone layers and MLPs.
    Leaves only the input embedding and LM head active for representation warmup.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze input embeddings (supports both tok_emb and embedding attribute names)
    emb_module = getattr(model, "tok_emb", getattr(model, "embedding", None))
    if emb_module is not None:
        for param in emb_module.parameters():
            param.requires_grad = True

    # Unfreeze output LM head (supports both head and lm_head attribute names)
    head_module = getattr(model, "head", getattr(model, "lm_head", None))
    if head_module is not None:
        for param in head_module.parameters():
            param.requires_grad = True


def get_differential_optimizer(
    model: nn.Module,
    base_lr: float = 3e-4,
    backbone_lr_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Stage 2: Differential learning rate optimizer.
    Backbone operates at an order of magnitude lower learning rate (e.g. 0.1 * base_lr)
    to prevent catastrophic forgetting while newly seeded subwords settle.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tok_emb" in name or "embedding" in name or "head" in name or "lm_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * backbone_lr_ratio, "weight_decay": weight_decay},
            {"params": head_params, "lr": base_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    return optimizer


def self_test():
    """Validates warm_start_embeddings, freeze_backbone_for_alignment, and differential optimizer."""
    from model.config import ModelConfig
    from model.backbone import Parentheses

    cfg = ModelConfig(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        attn_type="selective_linear",
        tie_embeddings=True,
    )
    model = Parentheses(cfg)

    # Mock decompose_fn: 0-255 map to identity byte, 256-511 map to 3-byte composite sequence
    def mock_decompose(new_id: int) -> List[int]:
        if new_id < 256:
            return [new_id]
        # Simulate a multi-byte Akshara (e.g. 3 bytes)
        return [(new_id * 7) % 256, (new_id * 13) % 256, (new_id * 23) % 256]

    new_v = 512
    new_emb, new_head = warm_start_embeddings(
        old_embedding=model.tok_emb,
        old_lm_head=model.head,
        new_vocab_size=new_v,
        decompose_fn=mock_decompose,
        tie_embeddings=cfg.tie_embeddings,
    )

    assert new_emb.weight.shape == (512, 32)
    assert new_head.weight.shape == (512, 32)
    assert new_head.weight is new_emb.weight, "Tied embeddings must share the same underlying tensor"

    # Check norm preservation: composite tokens should have norm close to base mean norm
    old_norm = torch.norm(model.tok_emb.weight, p=2, dim=-1).mean().item()
    comp_norm = torch.norm(new_emb.weight[300], p=2).item()
    diff = abs(old_norm - comp_norm)
    assert diff < old_norm * 0.1, f"Norm drifted excessively: old={old_norm:.4f}, new={comp_norm:.4f}"

    # Verify Stage 1 Freeze
    model.tok_emb = new_emb
    model.head = new_head
    model.cfg.vocab_size = new_v
    freeze_backbone_for_alignment(model)

    assert model.tok_emb.weight.requires_grad is True
    for block in model.blocks:
        for p in block.parameters():
            assert p.requires_grad is False, "Backbone layers must be frozen during alignment"

    # Verify Stage 2 Differential Optimizer
    for p in model.parameters():
        p.requires_grad = True
    opt = get_differential_optimizer(model, base_lr=1e-3, backbone_lr_ratio=0.1)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-4  # backbone
    assert opt.param_groups[1]["lr"] == 1e-3  # embeddings

    print("[self-test] vocab_surgery: warm-start, norm preservation, freeze, and differential optimizer passed ok.")


if __name__ == "__main__":
    self_test()
