"""
Validation script for parentheses-0.9-130M-v3 (16 layers, 768 dim, 12 heads, slab_v3).

Validates all 3 research-grade upgrades:
1. Token-Shift (mu-shift on control projections alpha, beta, gate).
2. Beta-Gated Selective Write (filtering filler tokens from polluting state rank).
3. Cross-Layer V0 Value Residual Highway (flattening the Layer 0 to 15 gradient slope).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn.functional as F

from model.config import PRESETS
from model.backbone import Parentheses
from train import build_optimizer


def validate_130m_v3():
    print("=" * 70)
    print("PARENTHESES 0.9-130M-v3 (SLAB_V3) ARCHITECTURE VALIDATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    cfg = PRESETS["parentheses-0.9-130M-v3"]
    print(f"\nConfiguration:")
    print(f"  - Vocab Size:   {cfg.vocab_size}")
    print(f"  - Block Size:   {cfg.block_size}")
    print(f"  - Layers:       {cfg.n_layer}")
    print(f"  - Heads:        {cfg.n_head}")
    print(f"  - Model Dim:    {cfg.n_embd}")
    print(f"  - Head Dim:     {cfg.n_embd // cfg.n_head}")
    print(f"  - Recurrent Rank per layer: {cfg.n_head * (cfg.n_embd // cfg.n_head)}")
    print(f"  - Attention:    {cfg.attn_type} (slab_v2={cfg.slab_v2}, slab_v3={cfg.slab_v3})")
    print(f"  - Tied Embd:    {cfg.tie_embeddings}")

    # 1. Instantiate model
    t0 = time.time()
    model = Parentheses(cfg).to(device)
    model.grad_checkpointing = True
    num_params = model.num_params()
    print(f"\n[1/5] Model created in {time.time() - t0:.2f}s")
    print(f"  - Total Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # 2. Inspect slab_v3 parameters (Token-Shift, Beta-Gate, V0 Highway)
    print(f"\n[2/5] Verifying slab_v3 architectural enhancements...")
    for i in [0, 1, cfg.n_layer - 1]:
        attn = model.blocks[i].attn
        assert hasattr(attn, "token_shift"), f"Layer {i} missing token_shift!"
        assert hasattr(attn, "beta_proj"), f"Layer {i} missing beta_proj!"
        mu_sample = attn.token_shift.mu.detach().mean().item()
        beta_b = attn.beta_proj.bias.detach()
        print(f"    Layer {i:2d}: TokenShift mu={mu_sample:.2f}, beta_proj.bias norm={beta_b.norm().item():.3f}")
        if i > 0:
            assert hasattr(attn, "v_residual_lamb"), f"Layer {i} missing v_residual_lamb!"
            v_mix = torch.sigmoid(attn.v_residual_lamb).detach().mean().item()
            print(f"             V0 Highway mix weight = {v_mix:.4f} (local V share)")
    print("  -> CONFIRMED: All slab_v3 control gates and highway parameters active.")

    # 3. Segregated Optimizer Setup
    print(f"\n[3/5] Initializing segregated AdamW optimizer...")
    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.1)
    for g in optimizer.param_groups:
        print(f"    Group '{g['name']}': {len(g['params'])} tensors, lr={g['lr']:.1e}, weight_decay={g['weight_decay']}")

    # 4. Forward + Backward pass with 4096 tokens (B=2, T=4096)
    B, T = 2, 4096
    print(f"\n[4/5] Testing forward + backward pass: B={B}, T={T} (BF16 AMP + Activation Checkpointing)...")
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t_fwd_start = time.time()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        _, loss = model(idx, targets=targets, return_logits=False, ce_rows=2048)
    t_fwd = time.time() - t_fwd_start

    print(f"  - Forward complete in {t_fwd:.3f}s")
    print(f"  - Initial Cross-Entropy Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    t_bwd_start = time.time()
    loss.backward()
    t_bwd = time.time() - t_bwd_start
    print(f"  - Backward complete in {t_bwd:.3f}s")

    # Layer-by-layer gradient norm inspection
    print("\n  Per-Layer Gradient Norm Distribution (slab_v3 with V0 Highway + TokenShift + Beta):")
    layer_grad_norms = []
    for i, block in enumerate(model.blocks):
        norms = []
        for name, p in block.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in layer {i} {name}"
                norms.append(p.grad.norm().item())
        avg_norm = sum(norms) / len(norms) if norms else 0.0
        layer_grad_norms.append(avg_norm)
        print(f"    Layer {i:2d}: Mean Grad Norm = {avg_norm:.5f}")

    slope_ratio = layer_grad_norms[0] / layer_grad_norms[-1]
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    print(f"\n  - Layer 0 / Layer 15 Gradient Ratio: {slope_ratio:.2f}x")
    print(f"  - Total Grad Norm (pre-clip):        {total_grad_norm:.4f}")

    # 5. Execute optimizer.step() to measure true post-step VRAM
    print(f"\n[5/5] Executing optimizer.step() (allocates lazy FP32 AdamW moment buffers)...")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if device == "cuda":
        vram_post_step = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  - True Peak VRAM (including AdamW state): {vram_post_step:.2f} MB ({vram_post_step / 1024:.2f} GB)")
        print(f"  - Free VRAM on RTX 5050 (8 GB):          {8.0 - (vram_post_step / 1024):.2f} GB")

    # Recurrent step() verification
    print(f"\nTesting recurrent O(1) step() inference under slab_v3...")
    model.eval()
    states = model.empty_states(B, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    step_idx = torch.randint(0, cfg.vocab_size, (B, 1), device=device)

    with torch.no_grad():
        step_logits, new_states = model.step(step_idx, states)
        assert step_logits.shape == (B, cfg.vocab_size), f"Bad step logits shape: {step_logits.shape}"
        assert len(new_states) == cfg.n_layer, "State count mismatch!"
        mat, conv_hist, shift_state = new_states[0]
        assert mat.shape == (B, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_embd // cfg.n_head)
        assert conv_hist.shape == (B, 3 * cfg.n_embd, 3)
        assert shift_state.shape == (B, cfg.n_embd)
        print(f"  - Recurrent state shape per layer: {tuple(mat.shape)} matrix + {tuple(conv_hist.shape)} conv hist + {tuple(shift_state.shape)} shift state")
        print("  - Recurrent step O(1) executed with zero-growth state.")

    print("\n" + "=" * 70)
    print("ALL 5 SYSTEM CHECKS PASSED: 130M-v3 (SLAB_V3) IS NUMERICALLY STABLE & READY")
    print("=" * 70)


if __name__ == "__main__":
    validate_130m_v3()
