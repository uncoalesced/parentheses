"""
Selective linear-attention block -- the alternative to CausalSelfAttention
(model/transformer.py), selected by ModelConfig.attn_type="selective_linear"
and wired into Block/Parentheses as of Phase 2.

Phase 1 of the architecture-pivot experiment (see documentation.md 2026-09-05):
scalar-per-head selective decay, matrix state per head. Two forward paths that
must agree exactly (up to float rounding):

    forward(x)   -- dual/parallel form, whole sequence at once. What training uses.
                    Y_h = ((Q_h K_h^T / sqrt(d_head)) . M_h) V_h, where M_h is a
                    T x T causal decay mask built from cumulative log-decay:
                    log M[i,j] = Lambda[i] - Lambda[j] for i>=j, else -inf.
    step(x, state) -- recurrent form, one token at a time. What FreeThink and a
                    future C++ export use. O(1) per step, no growing cache:
                    S_t = alpha_t * S_{t-1} + (k_t outer v_t) / sqrt(d_head)
                    y_t = q_t @ S_t

Decay is scalar-per-head (not per-channel/GLA-style) on purpose: the log-cumsum
in forward() never divides by anything, so there's no b_s^-1 underflow risk to
worry about at block_size 256-512. Per-channel selective decay is a possible
upgrade if Phase 4's long-FreeThink-run test shows tag-conditioning (<kn>/<fr>/
<zh>) doesn't survive scalar-per-head decay well enough -- not attempted here.

    python -m model.selective_linear_attention --self-test
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SelectiveLinearAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # H outputs only (one decay scalar per head) -- negligible param cost
        # (n_embd * n_head + n_head, e.g. 72*4+4=292 at the -300k preset;
        # ModelConfig.approx_params does not count these, see its docstring).
        # bias=True on purpose, NOT cfg.bias -- the one deliberate exception to
        # this project's "bias follows cfg.bias" convention. cfg.bias is False on
        # every real preset, and with no bias the gate starts at
        # softplus(0)=0.693 -> alpha~0.5, i.e. state half-gone every byte and
        # essentially wiped every ~5 bytes. A fresh model would have to climb out
        # of total amnesia before it could learn anything. See init_decay_bias().
        self.alpha_proj = nn.Linear(cfg.n_embd, cfg.n_head, bias=True)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # ponytail: no dropout on attention weights here (every real preset
        # trains with cfg.dropout=0.0 anyway) -- add if a preset ever sets
        # dropout > 0 and this block is actually in use.
        self._causal_bias_cache: torch.Tensor | None = None
        self.init_decay_bias()

    def init_decay_bias(self):
        """Give each head a different starting memory horizon.

        Mamba/S4-family models initialize a *spread* of timescales rather than
        one constant so heads can specialize into short- and long-range memory
        from step 0 instead of having to discover the spread by gradient. Same
        idea here: alpha targets are log-spaced in (1 - alpha) across [0.1,
        0.001], i.e. per-byte retention 0.9 (~7-byte half-life) through 0.999
        (~700-byte, longer than block_size). At n_head=4 that is alpha ~=
        {0.900, 0.978, 0.995, 0.999}.

        forward()/step() use log_alpha = -softplus(alpha_proj(x)), so the bias
        that yields a target alpha is softplus^-1(-log alpha) = log(exp(r) - 1)
        with r = -log(alpha).

        Public and idempotent because Parentheses.__init__ runs
        self.apply(_init_weights), which zeros every nn.Linear bias in the model
        -- this has to be re-applied after that sweep or the spread is lost.
        """
        one_minus_alpha = torch.logspace(-1, -3, self.n_head)   # 0.1 ... 0.001
        rate = -torch.log1p(-one_minus_alpha)                   # r = -log(alpha) > 0
        with torch.no_grad():
            self.alpha_proj.bias.copy_(torch.log(torch.expm1(rate)).to(self.alpha_proj.bias.dtype))

    def _causal_bias(self, T: int, device, dtype) -> torch.Tensor:
        """Cached additive causal mask: 0.0 where j<=i, -inf where j>i.

        Replaces rebuilding `triu(ones(T,T))` on every forward of every layer
        (5 allocations per step at the -300k preset) and lets the future be
        masked by an *add* rather than `masked_fill`, which profiled at 23% of
        the block -- the single most expensive op in it, dearer than either
        matmul. Adding -inf to a clamped-finite value gives exactly -inf, so
        exp() yields exactly 0.0, bitwise identical to the masked_fill it
        replaces.

        Deliberately a plain attribute, not register_buffer(): a buffer would
        enter state_dict and break loading existing selective-v1 checkpoints.
        Cached at the largest T seen and sliced, so varying inference lengths
        reuse one allocation.
        """
        c = self._causal_bias_cache
        if (c is None or c.shape[-1] < T
                or c.device != device or c.dtype != dtype):
            c = torch.zeros(T, T, device=device, dtype=dtype).masked_fill_(
                torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1),
                float("-inf"))
            self._causal_bias_cache = c
            return c
        return c[:T, :T]

    def empty_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Zero initial state: (B, H, head_dim, head_dim), one matrix per head."""
        return torch.zeros(batch_size, self.n_head, self.head_dim, self.head_dim,
                            device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dual/parallel form. x: (B, T, C) -> (B, T, C). Training-time only."""
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim
        qkv = self.qkv(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, Dh)

        log_alpha = -F.softplus(self.alpha_proj(x))          # (B, T, H), <= 0
        Lambda = torch.cumsum(log_alpha, dim=1).transpose(1, 2)  # (B, H, T)
        log_M = Lambda.unsqueeze(-1) - Lambda.unsqueeze(-2)   # (B,H,T,T): [i,j] = Lam_i - Lam_j
        # Clamp BEFORE the causal mask, never after: the mask writes -inf, and a
        # floor applied afterwards would turn those into exp(-30) ~ 9e-14 and
        # quietly leak the future into every row. exp(-30) is already zero to
        # within fp32's resolution of a decayed contribution, so this changes no
        # correctness-relevant output -- it only stops a mixed-precision run from
        # producing -inf/nan gradients out of an extreme decay excursion.
        log_M = torch.clamp(log_M, min=-30.0)
        # Mask by adding -inf rather than masked_fill: same clamp-then-mask
        # order, same exact zeros out of exp(), one cheaper pass over (B,H,T,T).
        log_M = log_M + self._causal_bias(T, log_M.device, log_M.dtype)
        M = torch.exp(log_M)  # i>=j: in (0, 1] (i==j -> 1, undecayed); i<j -> 0

        # scale folded onto q (B,H,T,Dh) instead of the (B,H,T,T) scores it used
        # to multiply -- same product, one fewer full pass over the big tensor.
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))  # (B,H,T,T)
        y = torch.matmul(scores * M, v)  # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def step(self, x: torch.Tensor, state: torch.Tensor | None):
        """Recurrent form. x: (B, C) single-token input at this layer.

        state: (B, H, head_dim, head_dim) from the previous step, or None for t=0
        (zero state). Returns (y, new_state); y is (B, C).
        """
        B, C = x.shape
        H, Dh = self.n_head, self.head_dim
        if state is None:
            state = self.empty_state(B, device=x.device, dtype=x.dtype)

        qkv = self.qkv(x).view(B, 3, H, Dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, H, Dh)

        log_alpha = -F.softplus(self.alpha_proj(x))  # (B, H)
        alpha = torch.exp(log_alpha)

        kv_outer = torch.einsum("bhd,bhe->bhde", k, v) * self.scale  # (B,H,Dh,Dh)
        new_state = alpha.unsqueeze(-1).unsqueeze(-1) * state + kv_outer
        y = torch.einsum("bhd,bhde->bhe", q, new_state).reshape(B, C)
        return self.proj(y), new_state


def self_test():
    """forward() (dual/parallel) and step() (recurrent) must agree exactly.

    Run in float64 to isolate an algorithm bug from ordinary fp32 accumulation
    noise -- cumsum-over-256-steps vs. 256 sequential multiplies round
    differently even when both are correct.
    """
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, block_size=64, n_layer=2, n_head=4, n_embd=32)
    attn = SelectiveLinearAttention(cfg).double().eval()

    B, T = 3, 40  # T well past a "single chunk", no block_size dependency here
    x = torch.randn(B, T, cfg.n_embd, dtype=torch.float64)

    with torch.no_grad():
        y_dual = attn(x)

        state = None
        steps = []
        for t in range(T):
            y_t, state = attn.step(x[:, t, :], state)
            steps.append(y_t)
        y_rec = torch.stack(steps, dim=1)

    assert y_dual.shape == (B, T, cfg.n_embd) == y_rec.shape
    assert torch.isfinite(y_dual).all() and torch.isfinite(y_rec).all()
    delta = (y_dual - y_rec).abs().max().item()
    assert delta < 1e-9, f"dual/recurrent mismatch: max abs diff {delta:.3e}"

    # state shape sanity: matches the 18x18-per-head footprint the design
    # settled on for the -300k preset (here 32/4=8 per head instead of 72/4=18,
    # tiny-smoke-style dims -- the shape contract is what's being checked).
    assert state.shape == (B, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.n_embd // cfg.n_head)

    print(f"[self-test] selective_linear_attention: dual/recurrent parity ok "
          f"(max abs diff {delta:.3e})")


if __name__ == "__main__":
    self_test()
