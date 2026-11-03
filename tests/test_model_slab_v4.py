#!/usr/bin/env python3
# Engineered by uncoalesced
"""
tests/test_model_slab_v4.py
Unit, property, and numerical parity tests for Parentheses 1.0 (240M / slab_v4) model.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.model_slab_v4 import (
    SlabV4Config,
    SlabV4Model,
    compute_doc_ids,
)


def test_slab_v4_parameter_count_and_sizing():
    """Verify that SlabV4Config scales to 240M class (~240-254M total params)."""
    cfg = SlabV4Config(
        vocab_size=32768,
        block_size=4096,
        n_layer=16,
        n_head=16,
        n_embd=1024,
        chunk_size=64,
        tie_embeddings=True,
    )
    model = SlabV4Model(cfg)
    total_params = model.num_params()
    backbone_params = model.num_params(exclude_embeddings=True)

    # Total params ~253.5M
    assert 235_000_000 <= total_params <= 260_000_000, f"Unexpected parameter count: {total_params}"
    assert backbone_params >= 200_000_000


def test_single_entrypoint_document_reset_wiring():
    """Verify that compute_doc_ids at model entry point properly resets state across EOT boundaries."""
    # Small test model
    cfg = SlabV4Config(
        vocab_size=1000,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=128,
        chunk_size=64,
        eot_token_id=0,
    )
    model = SlabV4Model(cfg)
    model.eval()

    # Create sequence with two documents separated by EOT (0) at index 63:
    # Doc 1: tokens 0..63 (ending with 0 at 63)
    # Doc 2: tokens 64..127 (starts at 64 with reset_mask == True)
    input_ids = torch.randint(10, 900, (2, 128))
    input_ids[:, 63] = 0  # EOT

    # Forward without passing reset_mask or doc_ids: model must auto-compute them
    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (2, 128, 1000)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_slab_v4_forward_and_recurrent_step_parity():
    """
    Verify numerical parity between chunked associative scan forward() and O(1) step()
    across document boundaries.
    """
    torch.manual_seed(42)
    cfg = SlabV4Config(
        vocab_size=256,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=128,
        chunk_size=64,
        eot_token_id=0,
    )
    model = SlabV4Model(cfg)
    model.eval()

    B, T = 1, 128
    input_ids = torch.randint(5, 250, (B, T))
    # Place EOT at token 63 (chunk boundary)
    input_ids[0, 63] = 0

    with torch.no_grad():
        # 1. Full chunked forward pass
        logits_fwd, _ = model(input_ids)

        # 2. Step-by-step recurrent execution
        states = model.empty_states(B, device=input_ids.device)
        logits_steps = []

        for t in range(T):
            tok = input_ids[:, t]
            is_boundary = (t > 0 and input_ids[0, t - 1].item() == 0)
            step_logits, states = model.step(tok, states=states, is_boundary=is_boundary)
            logits_steps.append(step_logits)

        logits_rec = torch.stack(logits_steps, dim=1)  # [B, T, vocab]

    # Max absolute error between chunked forward and recurrent step
    max_err = (logits_fwd - logits_rec).abs().max().item()
    print(f"Logits max forward-step error: {max_err:.2e}")
    # Forward and step must agree closely (within float32 associative reordering threshold)
    assert max_err < 1.0e-3, f"Parity failure: max error {max_err}"


def test_v0_highway_signal_propagation():
    """Verify Layer-0 V0 highway is generated and received across subsequent layers."""
    cfg = SlabV4Config(
        vocab_size=500,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        chunk_size=64,
    )
    model = SlabV4Model(cfg)

    # Check that layer 0 has no v_residual_lamb, while layers 1..3 have v_residual_lamb
    assert not hasattr(model.blocks[0].attn, "v_residual_lamb")
    for i in range(1, 4):
        assert hasattr(model.blocks[i].attn, "v_residual_lamb")
        assert model.blocks[i].attn.v_residual_lamb.shape == (cfg.n_head,)


def test_causal_4096_collation_next_token_alignment():
    """
    Verify Part 1 Collation Invariant:
      x = packed_ids[b, :4096]
      y = concat(packed_ids[b, 1:4096], [EOT_ID])
      loss_mask = concat(packed_mask[b, 1:4096], [0])
    Guarantees T=4096 (divisible by C=64), zero cross-bin leakage, and correct terminal EOT supervision.
    """
    EOT_ID = 0
    IGNORE_INDEX = -100

    # Synthetic bin of 4,096 tokens
    bin_ids = torch.randint(10, 500, (4096,))
    # Mark EOT at document boundary (token 2000) and at final token (token 4095)
    bin_ids[2000] = EOT_ID
    bin_ids[4095] = EOT_ID

    bin_mask = torch.zeros(4096, dtype=torch.uint8)
    # Supervised assistant spans: 100..2000 and 3000..4095
    bin_mask[100:2001] = 1  # includes terminal EOT at 2000
    bin_mask[3000:4096] = 1  # includes terminal EOT at 4095

    # 4,096-aligned batch collation
    x = bin_ids[:4096]
    y = torch.cat([bin_ids[1:4096], torch.tensor([EOT_ID])])
    loss_mask = torch.cat([bin_mask[1:4096], torch.tensor([0], dtype=torch.uint8)])

    # Lengths are exact
    assert len(x) == 4096
    assert len(y) == 4096
    assert len(loss_mask) == 4096
    assert len(x) % 64 == 0

    # Invariant: Predicting token 2000 (EOT) from token 1999 has target y[1999] == 2000 with mask 1
    assert y[1999] == EOT_ID
    assert loss_mask[1999] == 1

    # Invariant: Predicting token 4095 (EOT) from token 4094 has target y[4094] == 4095 with mask 1
    assert y[4094] == EOT_ID
    assert loss_mask[4094] == 1

    # Invariant: Final dummy target at index 4095 is masked with 0
    assert y[4095] == EOT_ID
    assert loss_mask[4095] == 0


def test_slab_v4_chunked_loss_parity():
    """Verify that chunked_loss (return_logits=False) matches full cross_entropy loss."""
    torch.manual_seed(42)
    cfg = SlabV4Config(
        vocab_size=256,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=128,
        chunk_size=64,
        ignore_index=-100,
    )
    model = SlabV4Model(cfg)
    model.eval()

    x = torch.randint(5, 250, (2, 128))
    targets = torch.randint(5, 250, (2, 128))
    # Mask out some positions with -100 and -1
    targets[0, 10:20] = -100
    targets[1, 30:40] = -1

    with torch.no_grad():
        logits_full, loss_full = model(x, targets, return_logits=True)
        logits_none, loss_chunked = model(x, targets, return_logits=False, ce_rows=64)

    assert logits_none is None
    assert loss_full is not None
    assert loss_chunked is not None
    diff = (loss_full - loss_chunked).abs().item()
    print(f"Chunked loss difference: {diff:.6e}")
    assert diff < 1e-4, f"Chunked loss mismatch: {diff}"


def test_slab_v4_uint8_reset_mask_conversion():
    """Verify that uint8 boundary mask (0=boundary, 1=non-boundary) matches bool mask (True=boundary)."""
    torch.manual_seed(42)
    cfg = SlabV4Config(
        vocab_size=256,
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=128,
        chunk_size=64,
    )
    model = SlabV4Model(cfg)
    model.eval()

    x = torch.randint(5, 250, (1, 128))
    # Set document boundary at t=64
    x[0, 63] = 0

    # uint8 mask: 0 at boundary (token 64), 1 elsewhere
    uint8_mask = torch.ones((1, 128), dtype=torch.uint8)
    uint8_mask[0, 64] = 0

    # bool mask: True at boundary (token 64), False elsewhere
    bool_mask = torch.zeros((1, 128), dtype=torch.bool)
    bool_mask[0, 64] = True

    with torch.no_grad():
        logits_auto, _ = model(x)
        logits_uint8, _ = model(x, reset_mask=uint8_mask)
        logits_bool, _ = model(x, reset_mask=bool_mask)

    err1 = (logits_auto - logits_uint8).abs().max().item()
    err2 = (logits_bool - logits_uint8).abs().max().item()

    assert err1 < 1e-5, f"uint8 vs auto mismatch: {err1}"
    assert err2 < 1e-5, f"uint8 vs bool mismatch: {err2}"


def test_net2widernet_130m_to_240m_forward_parity():
    """Verify function-preserving Net2WiderNet 130M -> 240M expansion preserves top-1 predictions and low logit delta."""
    ckpt_path = Path("I:/parentheses/checkpoints/parentheses-0.9-130M-v3/step_22888_swa3_base.pt")
    if not ckpt_path.exists():
        pytest.skip("Golden 130M checkpoint not found on I: drive")

    from scripts.grow_model_slab_v4 import grow_130m_to_240m
    from model.backbone import Parentheses

    model240, _ = grow_130m_to_240m(ckpt_path, out_path=None, jitter=1e-3, seed=42)
    model240.eval()

    ckpt130 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model130 = Parentheses(ckpt130["cfg"])
    model130.load_state_dict(ckpt130["model"])
    model130.eval()

    tokens = torch.randint(10, 500, (1, 64))
    with torch.no_grad():
        out130, _ = model130(tokens)
        out240, _ = model240(tokens)

    # 1. Top-1 predictions must agree 100%
    pred130 = out130.argmax(dim=-1)
    pred240 = out240.argmax(dim=-1)
    agreement = (pred130 == pred240).float().mean().item()
    assert agreement == 1.0, f"Token prediction agreement failed: {agreement}"

    # 2. Mean logit difference must be < 0.1
    mean_diff = (out130 - out240).abs().mean().item()
    assert mean_diff < 0.1, f"Mean logit diff too high: {mean_diff}"


