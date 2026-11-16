"""The parallel and recurrent forms must compute the same function, and the
rewrite must reproduce the 1.0 release.

Structural tests run on a tiny random model in float64, where a real bug shows
up as a large error and rounding does not.

The release tests need two environment variables and are skipped without them:

    PARENTHESES_BASE_CKPT   path to the 1.0 checkpoint (.pt or .safetensors)
    PARENTHESES_REF_REPO    path to a checkout of the 1.0 code (reference model)
"""

import os
import sys

import pytest
import torch

from parentheses import Parentheses, SlabConfig
from parentheses.checkpoint import load_model, load_state

EOT = 0
TINY = SlabConfig(vocab_size=97, n_layer=3, n_head=4, n_embd=32, chunk_size=8, eot_token_id=EOT)


def tiny_model(seed: int = 0) -> Parentheses:
    """Random model with every parameter moved off its initial value, so the
    conv history, the value highway and the gates all matter."""
    torch.manual_seed(seed)
    model = Parentheses(TINY).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("conv_weight", "conv_bias", "mu", "v_residual_lamb", "y_norm")):
                p.add_(torch.randn_like(p) * 0.3)
            elif "beta_proj" in name or name.endswith("alpha_proj.weight"):
                p.add_(torch.randn_like(p) * 0.5)
            elif p.dim() >= 2:
                p.mul_(8.0)      # 0.02-scale init barely moves the output; make layers count
    return model


def ids_with_documents(seed: int, B: int, T: int, eot_at) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, TINY.vocab_size, (B, T), generator=g)
    for b, positions in enumerate(eot_at):
        for t in positions:
            ids[b, t] = EOT
    return ids


def step_all(model: Parentheses, row: torch.Tensor, states=None, prev_eot: bool = False):
    """Drive step() over one sequence (T,). Returns logits (T, vocab) and final state."""
    out = []
    for t in range(row.numel()):
        logits, states = model.step(row[t:t + 1], states, boundary=prev_eot)
        prev_eot = int(row[t]) == EOT
        out.append(logits[0])
    return torch.stack(out), states


# eot at 7 / 15 puts the reset exactly on a block edge (chunk 8); 9 puts it one
# token past an edge so conv taps and the token shift straddle it; 23,24 is two
# boundaries in a row; row 3 has none.
EOT_PLAN = [(7, 30), (9, 15), (23, 24, 40), ()]


@torch.no_grad()
@pytest.mark.parametrize("T", [1, 3, 8, 21, 64, 77])
def test_forward_equals_step(T):
    model = tiny_model()
    plan = [tuple(p for p in row if p < T) for row in EOT_PLAN]
    ids = ids_with_documents(1, len(plan), T, plan)
    parallel = model(ids)
    for b in range(ids.size(0)):
        recurrent, _ = step_all(model, ids[b])
        assert (parallel[b] - recurrent).abs().max().item() < 1e-9, f"row {b}"


@torch.no_grad()
@pytest.mark.parametrize("split", [1, 2, 3, 7, 8, 9, 10, 16, 24, 25, 41, 76])
def test_prefill_then_step(split):
    """Read a prefix with the parallel form, continue with step()."""
    model = tiny_model()
    T = 77
    ids = ids_with_documents(2, len(EOT_PLAN), T, EOT_PLAN)
    for b in range(ids.size(0)):
        row = ids[b]
        want, want_states = step_all(model, row)
        head_logits, states = model.forward_with_state(row[None, :split])
        tail_logits, got_states = step_all(model, row[split:], states,
                                           prev_eot=int(row[split - 1]) == EOT)
        got = torch.cat([head_logits[0], tail_logits])
        assert (got - want).abs().max().item() < 1e-9, f"row {b}"
        for a, w in zip(got_states, want_states):
            assert (a.memory - w.memory).abs().max().item() < 1e-9


@torch.no_grad()
@pytest.mark.parametrize("cuts", [(5,), (8, 16), (10, 11, 40), (24, 25), (1, 2, 3, 70)])
def test_prefill_in_pieces(cuts):
    """Reading a sequence in several parallel pieces equals reading it at once."""
    model = tiny_model()
    T = 77
    ids = ids_with_documents(3, len(EOT_PLAN), T, EOT_PLAN)
    for b in range(ids.size(0)):
        row = ids[b:b + 1]
        whole, whole_states = model.forward_with_state(row)
        pieces, states, start, prev_eot = [], None, 0, False
        for end in (*cuts, T):
            logits, states = model.forward_with_state(row[:, start:end], states, prev_eot)
            pieces.append(logits)
            prev_eot, start = int(row[0, end - 1]) == EOT, end
        assert (torch.cat(pieces, dim=1) - whole).abs().max().item() < 1e-9, f"row {b}"
        for a, w in zip(states, whole_states):
            assert (a.memory - w.memory).abs().max().item() < 1e-9
            assert (a.conv - w.conv).abs().max().item() < 1e-9
            assert (a.prev - w.prev).abs().max().item() < 1e-9


@torch.no_grad()
def test_documents_are_isolated():
    """Two documents packed into one row score exactly as they do alone."""
    model = tiny_model()
    g = torch.Generator().manual_seed(4)
    a = torch.randint(2, TINY.vocab_size, (1, 19), generator=g)
    b = torch.randint(2, TINY.vocab_size, (1, 30), generator=g)
    a[0, -1] = EOT
    packed = model(torch.cat([a, b], dim=1))
    assert (packed[:, :19] - model(a)).abs().max().item() < 1e-9
    assert (packed[:, 19:] - model(b)).abs().max().item() < 1e-9


@torch.no_grad()
def test_state_size_is_constant():
    model = tiny_model()
    ids = ids_with_documents(5, 1, 200, [()])
    _, short = model.forward_with_state(ids[:, :10])
    _, long = model.forward_with_state(ids)
    assert model.state_bytes(short) == model.state_bytes(long)


@torch.no_grad()
def test_stream_matches_manual_greedy():
    model = tiny_model()
    prompt = ids_with_documents(6, 1, 13, [()])
    got = list(model.stream(prompt, max_new_tokens=20, stop_token_ids=()))
    logits, states = step_all(model, prompt[0])
    nxt, want, prev_eot = int(logits[-1].argmax()), [], False
    for _ in range(20):
        want.append(nxt)
        step_logits, states = model.step(torch.tensor([nxt]), states, boundary=prev_eot)
        prev_eot = nxt == EOT
        nxt = int(step_logits[0].argmax())
    assert got == want


def test_gradients_are_finite():
    model = tiny_model().train()
    ids = ids_with_documents(7, len(EOT_PLAN), 40, [tuple(p for p in row if p < 40) for row in EOT_PLAN])
    model(ids).logsumexp(-1).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------------------------------------------------------------------------
# Against the 1.0 release
# ---------------------------------------------------------------------------

CKPT = os.environ.get("PARENTHESES_BASE_CKPT")
REF_REPO = os.environ.get("PARENTHESES_REF_REPO")
needs_release = pytest.mark.skipif(not (CKPT and REF_REPO),
                                   reason="set PARENTHESES_BASE_CKPT and PARENTHESES_REF_REPO")
PARAMS_1_0 = 253_544_176
TOLERANCE = float(os.environ.get("PARENTHESES_PARITY_TOL", "1e-5"))


@pytest.fixture(scope="module")
def release():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ours = load_model(CKPT, device=device)
    sys.path.insert(0, REF_REPO)
    try:
        from models.model_slab_v4 import SlabV4Config, SlabV4Model
    finally:
        sys.path.remove(REF_REPO)
    ref = SlabV4Model(SlabV4Config())
    state, _ = load_state(CKPT)
    ref.load_state_dict(state, strict=False)
    ref.head.weight = ref.tok_emb.weight
    return ours, ref.to(device).eval(), device


def release_prompts(device, n=64, T=128):
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(2, 32768, (n, T), generator=g)
    for b in range(0, n, 4):                       # a quarter of the rows hold two documents
        ids[b, int(torch.randint(8, T - 8, (1,), generator=g))] = EOT
    return ids.to(device)


@needs_release
def test_release_param_count(release):
    ours, _, _ = release
    assert ours.num_params() == PARAMS_1_0


@needs_release
@torch.no_grad()
def test_release_forward_matches_reference(release):
    ours, ref, device = release
    ids = release_prompts(device)
    worst, scale = 0.0, 0.0
    for i in range(0, ids.size(0), 8):
        batch = ids[i:i + 8]
        want = ref(batch)[0].float()
        got = ours(batch).float()
        worst = max(worst, (got - want).abs().max().item())
        scale = max(scale, want.abs().max().item())
        assert torch.equal(got.argmax(-1), want.argmax(-1))
    print(f"\nforward parity: max abs logit diff {worst:.3e} (largest logit {scale:.1f}, "
          f"{ids.size(0)} prompts x {ids.size(1)} tokens, {device})")
    assert worst < TOLERANCE


@needs_release
@torch.no_grad()
def test_release_step_matches_reference(release):
    ours, ref, device = release
    ids = release_prompts(device, n=4, T=96)
    worst = 0.0
    for b in range(ids.size(0)):
        s_ours = s_ref = None
        prev_eot = False
        for t in range(ids.size(1)):
            tok = ids[b, t:t + 1]
            got, s_ours = ours.step(tok, s_ours, boundary=prev_eot)
            want, s_ref = ref.step(tok, s_ref, is_boundary=prev_eot)
            prev_eot = int(tok) == EOT
            worst = max(worst, (got.float() - want.float()).abs().max().item())
    print(f"\nstep parity: max abs logit diff {worst:.3e} (4 prompts x 96 tokens, {device})")
    assert worst < TOLERANCE


@needs_release
@torch.no_grad()
def test_release_prefill_matches_own_step(release):
    """On the real weights: parallel prefill, then step, equals step throughout."""
    ours, _, device = release
    ids = release_prompts(device, n=2, T=200)
    worst = 0.0
    for b in range(ids.size(0)):
        row = ids[b]
        want, _ = step_all(ours, row)
        head, states = ours.forward_with_state(row[None, :137])
        tail, _ = step_all(ours, row[137:], states, prev_eot=int(row[136]) == EOT)
        worst = max(worst, (torch.cat([head[0], tail]).float() - want.float()).abs().max().item())
    print(f"\nprefill-then-step vs step: max abs logit diff {worst:.3e} (2 prompts x 200 tokens)")
    assert worst < 1e-3
