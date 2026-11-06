#!/usr/bin/env python3
# Engineered by uncoalesced
"""
tests/test_slab_v4_delta_parity.py
Verifies the opt-in gated delta-rule (Gated DeltaNet) kernels of slab_v4:
  - ssd_chunk_scan_delta_v4 (chunked, C=64) == ssd_recurrent_step_delta_v4 (literal step rule),
    forward and backward, across the 4 boundary regimes of build_four_regime_tokens (fp64, CPU)
  - _erase=False reproduces ssd_chunk_scan_v4 bit for bit (fp32)
  - with mutually orthogonal keys and S_0 = 0 the erase term vanishes, so delta == additive
  - SlabV4Model(delta_rule=True) forward() == step(); delta_rule=False keeps state_dict keys;
    an old pickled SlabV4Config without the field still loads and compares equal
Runs on CPU only (the GPU may be training).
"""

import pickle
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.slab_v4_boundary import (
    compute_doc_ids,
    ssd_chunk_scan_v4,
    ssd_chunk_scan_delta_v4,
    ssd_recurrent_step_delta_v4,
)
from models.model_slab_v4 import SlabV4Config, SlabV4Model
from tests.test_slab_v4_boundary_parity import build_four_regime_tokens, EOT_ID

DEVICE = "cpu"


def _make_inputs(B, T, H, Dh, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    scale = Dh ** -0.5
    q = torch.randn(B, T, H, Dh, generator=g, dtype=dtype) * scale
    k = F.normalize(torch.randn(B, T, H, Dh, generator=g, dtype=dtype), dim=-1)
    v = torch.randn(B, T, H, Dh, generator=g, dtype=dtype)
    log_a = -torch.rand(B, T, H, generator=g, dtype=dtype) * 0.5 - 0.01
    beta = torch.rand(B, T, H, generator=g, dtype=dtype) * 1.85 + 0.1   # (0.1, 1.95): b > 1 regime
    init_s = torch.randn(B, H, Dh, Dh, generator=g, dtype=dtype) * scale
    return [t.requires_grad_(True) for t in (q, k, v, log_a, beta, init_s)]


def _clone(ts):
    return [t.detach().clone().requires_grad_(True) for t in ts]


def test_delta_four_regime_forward_and_backward_parity_fp64():
    torch.manual_seed(42)
    B, T, C, H, Dh = 4, 256, 64, 4, 32

    input_ids = build_four_regime_tokens(T=T, C=C, device=DEVICE)
    reset_mask, doc_ids = compute_doc_ids(input_ids, eot_token_id=EOT_ID)

    a = _make_inputs(B, T, H, Dh, torch.float64, seed=42)
    b = _clone(a)
    q1, k1, v1, la1, be1, s1 = a
    q2, k2, v2, la2, be2, s2 = b

    y_chunk, s_chunk = ssd_chunk_scan_delta_v4(
        q1, k1, v1, la1, be1, reset_mask, doc_ids, chunk_size=C, initial_state=s1
    )
    y_step, s_step = ssd_recurrent_step_delta_v4(
        q2, k2, v2, la2, be2, reset_mask, initial_state=s2
    )
    assert y_chunk.dtype == torch.float64 and s_chunk.dtype == torch.float64

    # The erase term must be live: the delta rule has to differ from the additive rule here
    with torch.no_grad():
        y_add, _ = ssd_chunk_scan_v4(q1, k1, v1, la1, be1, reset_mask, doc_ids, chunk_size=C, initial_state=s1)
    gap = (y_chunk.detach() - y_add).abs().max().item()
    print(f"delta vs additive max |y| gap: {gap:.3e}")
    assert gap > 1e-2, f"delta rule indistinguishable from additive (gap {gap})"

    for r in range(B):
        max_y_err = (y_chunk[r] - y_step[r]).abs().max().item()
        max_s_err = (s_chunk[r] - s_step[r]).abs().max().item()
        print(f"Regime {r+1}: max |y| err {max_y_err:.3e}, max |S| err {max_s_err:.3e}")
        assert max_y_err < 1e-12, f"Regime {r+1} forward y mismatch: {max_y_err}"
        assert max_s_err < 1e-12, f"Regime {r+1} terminal state mismatch: {max_s_err}"

    g = torch.Generator().manual_seed(3)
    grad_out = torch.randn(y_chunk.shape, generator=g, dtype=torch.float64)
    ((y_chunk * grad_out).sum() + 0.1 * s_chunk.sum()).backward()
    ((y_step * grad_out).sum() + 0.1 * s_step.sum()).backward()

    for name, t_c, t_s in zip(("q", "k", "v", "log_alpha", "beta", "init_state"), a, b):
        g_c, g_s = t_c.grad, t_s.grad
        assert torch.isfinite(g_c).all(), f"Non-finite chunked {name}.grad"
        max_grad_err = (g_c - g_s).abs().max().item()
        print(f"grad {name}: max err {max_grad_err:.3e} (max |grad| {g_s.abs().max().item():.3e})")
        assert max_grad_err < 1e-11, f"Backward {name}.grad mismatch: {max_grad_err}"


def test_delta_erase_disabled_equals_additive_kernel_bitwise():
    B, T, C, H, Dh = 4, 256, 64, 4, 32
    input_ids = build_four_regime_tokens(T=T, C=C, device=DEVICE)
    reset_mask, doc_ids = compute_doc_ids(input_ids, eot_token_id=EOT_ID)

    a = _make_inputs(B, T, H, Dh, torch.float32, seed=11)
    b = _clone(a)
    y_ref, s_ref = ssd_chunk_scan_v4(*a[:5], reset_mask, doc_ids, chunk_size=C, initial_state=a[5])
    y_new, s_new = ssd_chunk_scan_delta_v4(
        *b[:5], reset_mask, doc_ids, chunk_size=C, initial_state=b[5], _erase=False
    )
    print(f"erase off: max |y| diff {(y_ref - y_new).abs().max().item():.3e}, "
          f"max |S| diff {(s_ref - s_new).abs().max().item():.3e}")
    assert torch.equal(y_ref, y_new), "y differs from ssd_chunk_scan_v4"
    assert torch.equal(s_ref, s_new), "final state differs from ssd_chunk_scan_v4"

    g = torch.Generator().manual_seed(5)
    grad_out = torch.randn(y_ref.shape, generator=g)
    ((y_ref * grad_out).sum() + 0.1 * s_ref.sum()).backward()
    ((y_new * grad_out).sum() + 0.1 * s_new.sum()).backward()
    for name, t_r, t_n in zip(("q", "k", "v", "log_alpha", "beta", "init_state"), a, b):
        assert torch.equal(t_r.grad, t_n.grad), f"{name}.grad differs with erase off"


def test_delta_orthogonal_keys_reduce_to_additive_fp64():
    # One chunk, S_0 = 0, keys one-hot and distinct per position: k_i . k_j = 0 for i != j,
    # so A = 0 and the erase read k_t S_{t-1} is 0; the delta rule equals the additive rule.
    B, T, C, H, Dh = 2, 64, 64, 3, 64
    input_ids = torch.randint(10, 500, (B, T))
    input_ids[1, 20] = EOT_ID
    reset_mask, doc_ids = compute_doc_ids(input_ids, eot_token_id=EOT_ID)

    q, _, v, log_a, beta, _ = [t.detach() for t in _make_inputs(B, T, H, Dh, torch.float64, seed=9)]
    perm = torch.randperm(Dh, generator=torch.Generator().manual_seed(1))
    k = torch.eye(Dh, dtype=torch.float64)[perm].view(1, T, 1, Dh).expand(B, T, H, Dh).contiguous()

    y_add, s_add = ssd_chunk_scan_v4(q, k, v, log_a, beta, reset_mask, doc_ids, chunk_size=C)
    y_del, s_del = ssd_chunk_scan_delta_v4(q, k, v, log_a, beta, reset_mask, doc_ids, chunk_size=C)
    y_stp, _ = ssd_recurrent_step_delta_v4(q, k, v, log_a, beta, reset_mask)
    # ssd_chunk_scan_v4 always computes in fp32, so compare at fp32 precision against it
    err_add = (y_del - y_add.to(torch.float64)).abs().max().item()
    err_stp = (y_del - y_stp).abs().max().item()
    print(f"orthogonal keys: delta vs additive {err_add:.3e}, delta chunk vs step {err_stp:.3e}")
    assert err_add < 1e-5 and (s_del - s_add.to(torch.float64)).abs().max().item() < 1e-5
    assert err_stp < 1e-12


def _tiny_cfg(**kw):
    return SlabV4Config(vocab_size=256, block_size=128, n_layer=2, n_head=4, n_embd=128,
                        chunk_size=64, eot_token_id=0, **kw)


def test_delta_model_forward_equals_step_fp64():
    torch.manual_seed(42)
    model = SlabV4Model(_tiny_cfg(delta_rule=True)).double().eval()
    with torch.no_grad():   # non-zero beta logits so b spans (0, 2) instead of sitting at 1.0
        for blk in model.blocks:
            blk.attn.beta_proj.weight.normal_(0.0, 0.2)
    B, T = 1, 128
    input_ids = torch.randint(5, 250, (B, T))
    for pos in (40, 63, 64, 100):
        input_ids[0, pos] = 0

    with torch.no_grad():
        logits_fwd, _ = model(input_ids)
        states = model.empty_states(B, device=input_ids.device)
        steps = []
        for t in range(T):
            is_boundary = (t > 0 and input_ids[0, t - 1].item() == 0)
            lg, states = model.step(input_ids[:, t], states=states, is_boundary=is_boundary)
            steps.append(lg)
        logits_rec = torch.stack(steps, dim=1)

    max_err = (logits_fwd - logits_rec).abs().max().item()
    print(f"delta model forward-step max err (fp64): {max_err:.3e}")
    assert max_err < 1e-9, f"delta_rule forward/step mismatch: {max_err}"


def test_delta_flag_state_dict_and_old_cfg_pickle():
    sd_off = SlabV4Model(_tiny_cfg()).state_dict()
    sd_on = SlabV4Model(_tiny_cfg(delta_rule=True)).state_dict()
    assert list(sd_off.keys()) == list(sd_on.keys())
    assert all(sd_off[k].shape == sd_on[k].shape for k in sd_off)

    # Old checkpoints pickle a SlabV4Config whose __dict__ has no delta_rule field
    old = _tiny_cfg()
    del old.__dict__["delta_rule"]
    loaded = pickle.loads(pickle.dumps(old))
    assert "delta_rule" not in loaded.__dict__
    assert loaded == _tiny_cfg() and loaded.delta_rule is False
    model = SlabV4Model(loaded)
    model.load_state_dict(sd_off)
    assert all(not blk.attn.delta_rule for blk in model.blocks)


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_delta_four_regime_forward_and_backward_parity_fp64()
    test_delta_erase_disabled_equals_additive_kernel_bitwise()
    test_delta_orthogonal_keys_reduce_to_additive_fp64()
    test_delta_model_forward_equals_step_fp64()
    test_delta_flag_state_dict_and_old_cfg_pickle()
    print("[PASS] delta-rule parity: 4 regimes fwd+bwd fp64, erase-off bitwise, orthogonal keys, model fwd==step, cfg compat.")
