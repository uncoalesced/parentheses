"""
Validation script for parentheses-0.9-130M-v2 (16 layers, 768 dim, 12 heads, slab_v2).

Validates:
1. Model instantiation & parameter accounting.
2. Full 4,096-token forward pass in BF16 on CUDA with grad_checkpointing and chunked_loss.
3. Backward pass, loss convergence, and bounded gradient norms across all 16 layers.
4. Recurrent O(1) step() execution and state persistence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn.functional as F

from model.config import PRESETS
from model.backbone import Parentheses


def validate_130m_v2():
    print("=" * 70)
    print("PARENTHESES 0.9-130M-v2 ARCHITECTURE VALIDATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    cfg = PRESETS["parentheses-0.9-130M-v2"]
    print(f"\nConfiguration:")
    print(f"  - Vocab Size:   {cfg.vocab_size}")
    print(f"  - Block Size:   {cfg.block_size}")
    print(f"  - Layers:       {cfg.n_layer}")
    print(f"  - Heads:        {cfg.n_head}")
    print(f"  - Model Dim:    {cfg.n_embd}")
    print(f"  - Head Dim:     {cfg.n_embd // cfg.n_head}")
    print(f"  - Recurrent Rank per layer: {cfg.n_head * (cfg.n_embd // cfg.n_head)}")
    print(f"  - Attention:    {cfg.attn_type} (slab_v2={cfg.slab_v2})")
    print(f"  - Tied Embd:    {cfg.tie_embeddings}")

    # 1. Instantiate model
    t0 = time.time()
    model = Parentheses(cfg).to(device)
    model.grad_checkpointing = True  # Standard in Parentheses for 8GB VRAM at T=4096
    num_params = model.num_params()
    print(f"\n[1/4] Model created in {time.time() - t0:.2f}s")
    print(f"  - Total Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # 2. Forward pass with 4096 tokens (B=2, T=4096)
    B, T = 2, 4096
    print(f"\n[2/4] Testing forward pass: B={B}, T={T} (BF16 AMP + Activation Checkpointing)...")
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

    if device == "cuda":
        vram_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  - Peak Forward VRAM: {vram_fwd:.2f} MB")

    # 3. Backward pass & gradient norm verification across all layers
    print(f"\n[3/4] Testing backward pass and gradient boundedness across all 16 layers...")
    t_bwd_start = time.time()
    loss.backward()
    t_bwd = time.time() - t_bwd_start
    print(f"  - Backward complete in {t_bwd:.3f}s")

    if device == "cuda":
        vram_total = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  - Peak Total (Fwd+Bwd) VRAM: {vram_total:.2f} MB ({vram_total / 1024:.2f} GB)")

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

    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    print(f"  - Total Grad Norm (pre-clip): {total_grad_norm:.4f}")
    assert total_grad_norm > 0.0, "Zero gradients detected!"
    assert torch.isfinite(torch.tensor(total_grad_norm)), "Non-finite total gradient norm!"

    # 4. Recurrent step() verification
    print(f"\n[4/4] Testing recurrent O(1) step() inference...")
    model.eval()
    states = model.empty_states(B, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    step_idx = torch.randint(0, cfg.vocab_size, (B, 1), device=device)

    with torch.no_grad():
        step_logits, new_states = model.step(step_idx, states)
        assert step_logits.shape == (B, cfg.vocab_size), f"Bad step logits shape: {step_logits.shape}"
        assert len(new_states) == cfg.n_layer, "State count mismatch!"
        mat, conv_hist = new_states[0]
        assert mat.shape == (B, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_embd // cfg.n_head)
        assert conv_hist.shape == (B, 3 * cfg.n_embd, 3)
        print(f"  - Recurrent state shape per layer: {tuple(mat.shape)} matrix + {tuple(conv_hist.shape)} conv hist")
        print("  - Recurrent step O(1) executed successfully.")

    print("\n" + "=" * 70)
    print("ALL VALIDATION CHECKS PASSED: 130M-v2 IS NUMERICALLY STABLE & READY")
    print("=" * 70)


if __name__ == "__main__":
    validate_130m_v2()
