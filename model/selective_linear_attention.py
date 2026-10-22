"""
Selective linear-attention block -- the alternative to CausalSelfAttention
(model/backbone.py), selected by ModelConfig.attn_type="selective_linear"
and wired into Block/Parentheses as of Phase 2.

Engineered by uncoalesced

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


def _up(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> fp32 for a reduction; fp32/fp64 untouched (fp64 parity tests)."""
    return t.float() if t.element_size() < 4 else t


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
        # slab_v2 (GLA / Mamba-2 / GatedDeltaNet practice). Created only when on, so
        # a slab_v2=False model has exactly the old state_dict.
        #  - depthwise causal conv k=4 on q/k/v (+silu): cheap local mixing, and it is
        #    reset-masked so packed documents still never see each other;
        #  - L2-normalized q/k: bounded scores, no per-head scale drift;
        #  - per-head RMSNorm on y, then y * silu(W_g x): y sums ~1/(1-alpha) terms
        #    and is otherwise unbounded at long horizons.
        self.slab_v2 = getattr(cfg, "slab_v2", False)
        if self.slab_v2:
            self.conv_k = 4
            self.conv_w = nn.Parameter(torch.zeros(3 * cfg.n_embd, self.conv_k))
            self.conv_b = nn.Parameter(torch.zeros(3 * cfg.n_embd))
            self.y_norm = nn.Parameter(torch.ones(cfg.n_embd))
            self.gate = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
            self.init_conv()
        self.init_decay_bias()

    def init_conv(self):
        """Identity taps (current token weight 1, history 0): at init the conv is a
        pass-through and the block starts as plain silu(qkv). Parentheses'
        _init_weights only touches nn.Linear/nn.Embedding, so this survives it."""
        if self.slab_v2:
            with torch.no_grad():
                self.conv_w.zero_()
                self.conv_w[:, -1] = 1.0
                self.conv_b.zero_()

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
        # slab_v2: 1-alpha down to 10^-4.5 (~30k-token horizon) -- 0.999 is only
        # ~700 tokens, too short once block_size is 2048-4096.
        lo = -4.5 if self.slab_v2 else -3
        one_minus_alpha = torch.logspace(-1, lo, self.n_head)   # 0.1 ... 0.001 (v1)
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

    def empty_state(self, batch_size: int, device=None, dtype=None):
        """Zero initial state: (B, H, head_dim, head_dim), one matrix per head.
        slab_v2: (that matrix, conv history (B, 3C, k-1))."""
        S = torch.zeros(batch_size, self.n_head, self.head_dim, self.head_dim,
                        device=device, dtype=dtype)
        if not self.slab_v2:
            return S
        return S, torch.zeros(batch_size, 3 * self.n_head * self.head_dim, self.conv_k - 1,
                              device=device, dtype=dtype)

    def _conv(self, u: torch.Tensor, reset_mask: torch.Tensor | None) -> torch.Tensor:
        """Causal depthwise conv over time, u: (B, T, 3C). A tap reaching back
        across a document start is zeroed, same isolation as the state reset."""
        # One depthwise conv1d kernel in the activation dtype (bf16 under autocast): an
        # elementwise version promoted to fp32 against the fp32 taps and saved several
        # (B, T, 3C) fp32 copies per layer -- enough to OOM at T=2048 (2026-10-07).
        w = self.conv_w.to(u.dtype)
        out = F.conv1d(F.pad(u.transpose(1, 2), (self.conv_k - 1, 0)), w.unsqueeze(1),
                       self.conv_b.to(u.dtype), groups=u.size(-1)).transpose(1, 2)
        if reset_mask is not None:
            # Document isolation: subtract the few taps that reach back across a
            # document start (<= k-1 positions per boundary), sparsely, in one index_put.
            bi, ti, s = self._crossing_taps(reset_mask)
            if bi.numel():
                out = out.index_put((bi, ti), -(u[bi, ti - s] * w.t()[self.conv_k - 1 - s]), accumulate=True)
        return F.silu(out)

    def _crossing_taps(self, reset_mask: torch.Tensor):
        """(batch, position, shift) of every conv tap that crosses a document start.

        nonzero() is a host sync, and every layer sees the same reset_mask object, so
        the result is cached on that tensor: one sync per forward instead of one per
        shift per layer (48 at 16 layers, which cost 26% of a slab_v2 step).
        """
        cached = getattr(reset_mask, "_slab_taps", None)
        if cached is not None and cached[0] == reset_mask._version:
            return cached[1]
        T, K = reset_mask.size(1), self.conv_k
        doc = torch.cumsum((reset_mask == 0).long(), dim=1)  # (B, T)
        cross = torch.zeros(K - 1, *doc.shape, dtype=torch.bool, device=doc.device)
        for s in range(1, min(K, T)):
            cross[s - 1, :, s:] = doc[:, s:] != doc[:, :-s]      # tap t-s is in an earlier doc
        si, bi, ti = cross.nonzero(as_tuple=True)
        taps = (bi, ti, si + 1)
        reset_mask._slab_taps = (reset_mask._version, taps)
        return taps

    def _project(self, x: torch.Tensor, reset_mask: torch.Tensor | None = None):
        """q, k, v (B, H, T, Dh) and log_alpha (B, T, H) for the whole sequence."""
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim
        u = self.qkv(x)
        if self.slab_v2:
            u = self._conv(u, reset_mask)
        qkv = u.view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.slab_v2:   # norms in >= fp32, result back in the activation dtype
            q, k = F.normalize(_up(q), dim=-1).to(q.dtype), F.normalize(_up(k), dim=-1).to(k.dtype)
        return q, k, v, -F.softplus(self.alpha_proj(x))

    def _output(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """y: (B, H, T, Dh) or (B, H, Dh) step form; x: the block input (B, T, C) / (B, C)."""
        if self.slab_v2:
            w = self.y_norm.view(self.n_head, self.head_dim).to(y.dtype)
            w = w.unsqueeze(1) if y.dim() == 4 else w
            yf = _up(y)
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype) * w
        y = y.transpose(1, 2).reshape(x.shape) if y.dim() == 4 else y.reshape(x.shape)
        if self.slab_v2:
            y = y * F.silu(self.gate(x))
        return self.proj(y)

    def compile_projections(self, backend: str = "inductor", mode: str | None = None):
        """Opt-in compilation of _project and _output paths."""
        kwargs = {"backend": backend}
        if mode is not None:
            kwargs["mode"] = mode
        self._project = torch.compile(self._project, **kwargs)
        self._output = torch.compile(self._output, **kwargs)

    def _forward_dense(self, x: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dense quadratic dual/parallel form O(T^2). Fallback for arbitrary/short sequence lengths."""
        B, T, C = x.shape
        q, k, v, log_alpha = self._project(x, reset_mask)  # q/k/v (B, H, T, Dh); log_alpha (B, T, H) <= 0
        Lambda = torch.cumsum(log_alpha, dim=1).transpose(1, 2)  # (B, H, T)
        log_M = Lambda.unsqueeze(-1) - Lambda.unsqueeze(-2)   # (B,H,T,T): [i,j] = Lam_i - Lam_j
        log_M = torch.clamp(log_M, min=-30.0)
        log_M = log_M + self._causal_bias(T, log_M.device, log_M.dtype)

        if reset_mask is not None:
            doc_id = torch.cumsum((reset_mask == 0).long(), dim=1)  # (B, T)
            same_doc = (doc_id.unsqueeze(-1) == doc_id.unsqueeze(-2)).unsqueeze(1)  # (B, 1, T, T)
            log_M = log_M.masked_fill(~same_doc, float("-inf"))

        M = torch.exp(log_M)  # i>=j: in (0, 1] (i==j -> 1, undecayed); i<j -> 0

        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))  # (B,H,T,T)
        y = torch.matmul(scores * M, v)  # (B, H, T, Dh)
        return self._output(y, x)

    def forward(self, x: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Dual/parallel form with chunked linear attention (C=64) per
        FORMAL_MATHEMATICAL_SPECIFICATION.md Section 10.

        Closes the 34% training throughput gap against causal self-attention
        while strictly preserving machine-precision FP64 parity (< 1e-9) and
        zero document boundary leakage.

        reset_mask: optional (B, T) or (B, 1, T) tensor in {0, 1} where 0 indicates
        the start of a packed document boundary, resetting state to prevent
        cross-document memory leakage.
        """
        if reset_mask is not None and reset_mask.dim() == 3:
            reset_mask = reset_mask.squeeze(1)

        B, T, C = x.shape
        chunk_size = 64

        # Fallback to dense if sequence length is not a positive multiple of chunk_size
        if T % chunk_size != 0 or T < chunk_size:
            return self._forward_dense(x, reset_mask)

        H, Dh = self.n_head, self.head_dim
        n_chunks = T // chunk_size

        q, k, v, log_alpha = self._project(x, reset_mask)  # each (B, H, T, Dh)
        log_alpha = log_alpha.transpose(1, 2)  # (B, H, T)

        q_c = q.view(B, H, n_chunks, chunk_size, Dh)
        k_c = k.view(B, H, n_chunks, chunk_size, Dh)
        v_c = v.view(B, H, n_chunks, chunk_size, Dh)
        log_a_c = log_alpha.view(B, H, n_chunks, chunk_size)

        Lambda_c = torch.cumsum(log_a_c, dim=-1)
        log_M_intra = (Lambda_c.unsqueeze(-1) - Lambda_c.unsqueeze(-2)).clamp(min=-30.0)
        causal_mask = self._causal_bias(chunk_size, x.device, x.dtype)
        log_M_intra = log_M_intra + causal_mask

        if reset_mask is not None:
            doc_id = torch.cumsum((reset_mask == 0).long(), dim=-1)  # (B, T)
            doc_id_c = doc_id.view(B, n_chunks, chunk_size)
            same_doc_intra = (doc_id_c.unsqueeze(-1) == doc_id_c.unsqueeze(-2)).unsqueeze(1)
            log_M_intra = log_M_intra.masked_fill(~same_doc_intra, float("-inf"))

        M_intra = torch.exp(log_M_intra)
        scores_intra = torch.matmul(q_c * self.scale, k_c.transpose(-2, -1))
        o_intra = torch.matmul(scores_intra * M_intra, v_c)  # (B, H, n_chunks, C, Dh)

        # Inter-chunk recurrence (spec Section 10.6 Proofs 1-3), in closed form.
        # The loop it replaces ran S_out[c] = g[c] * surv[c] * S_out[c-1] + dS[c] chunk by
        # chunk: ~10 small kernels per chunk per layer, forward and backward, so at
        # T=2048 (32 chunks) the GPU sat idle waiting on Python launches (profiled
        # 2026-10-07: CPU time == step time). Decay is a scalar per head, so the state
        # entering chunk c unrolls to
        #     S_in[c] = sum_{j<c} exp(LA[c-1] - LA[j]) * same_doc(c-1, j) * dS[j]
        # with A[c] = chunk c's total log decay and LA its cumsum: one masked (n x n)
        # weight matrix and one einsum, launch count independent of T.
        # Resets: dS[j] holds only chunk j's final document (as before), state survives
        # chunk i iff its last doc id equals chunk i-1's, so it survives j+1..c-1 iff
        # E[c-1] == E[j] (doc ids are non-decreasing), and token i of chunk c reads it
        # iff its doc id == E[c-1] -- exactly the old loop's can_see_prev / survives.
        A = Lambda_c[..., -1]                                          # (B, H, n)
        decay_to_end = torch.exp((A.unsqueeze(-1) - Lambda_c).clamp(min=-30.0))
        valid = torch.ones(n_chunks, n_chunks, dtype=torch.bool, device=x.device).tril(-1)
        can_see = None
        if reset_mask is not None:
            E = doc_id_c[:, :, -1]                                       # (B, n) last doc id per chunk
            E_prev = torch.cat([E[:, :1], E[:, :-1]], dim=1)             # E[c-1] (c=0: no j<0 anyway)
            decay_to_end = decay_to_end * (doc_id_c == E.unsqueeze(-1)).unsqueeze(1).to(decay_to_end.dtype)
            valid = valid & (E_prev.unsqueeze(-1) == E.unsqueeze(-2)).unsqueeze(1)          # (B, 1, n, n)
            can_see = (doc_id_c == E_prev.unsqueeze(-1)).unsqueeze(1).unsqueeze(-1)         # (B, 1, n, C, 1)
        k_end = k_c * decay_to_end.unsqueeze(-1) * self.scale
        dS = torch.matmul(k_end.transpose(-2, -1), v_c)                 # (B, H, n, Dh, Dh)
        LA = torch.cumsum(A, dim=-1)
        expo = (LA - A).unsqueeze(-1) - LA.unsqueeze(-2)                # [c, j] = LA[c-1] - LA[j]
        # where() before exp: masked (j >= c) entries are -inf -> exactly 0 with zero grad,
        # never exp(+large) * 0 = nan in backward
        W = torch.exp(torch.where(valid, expo, torch.full_like(expo, float("-inf"))))
        # CRITICAL FIX: Upcast decay weights and chunk state increments to FP32 (or FP64 for parity tests).
        # In BF16, 7-8 mantissa bits truncate small exponential weights (e^-10 to e^-30)
        # across 64 chunks, zeroing out early historical context.
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        W_acc = W.to(acc_dtype)
        dS_acc = dS.to(acc_dtype)
        S_in_acc = torch.einsum("bhcj,bhjde->bhcde", W_acc, dS_acc)
        S_in = S_in_acc.to(q_c.dtype)
        o_inter = torch.matmul(q_c, S_in) * torch.exp(Lambda_c).unsqueeze(-1)
        if can_see is not None:
            o_inter = o_inter * can_see.to(o_inter.dtype)
        o_all = (o_intra + o_inter).reshape(B, H, T, Dh)
        return self._output(o_all, x)

    def step(self, x: torch.Tensor, state: torch.Tensor | None, reset: torch.Tensor | None = None):
        """Recurrent form. x: (B, C) single-token input at this layer.

        state: (B, H, head_dim, head_dim) from the previous step, or None for t=0
        (zero state).
        reset: optional (B,) tensor in {0, 1} where 0 clears memory state (document boundary).
        Returns (y, new_state); y is (B, C).
        """
        B, C = x.shape
        H, Dh = self.n_head, self.head_dim
        if state is None:
            state = self.empty_state(B, device=x.device, dtype=x.dtype)
        elif reset is not None:
            r = reset.view(B, 1, 1).to(x.dtype)
            state = (state[0] * r.unsqueeze(-1), state[1] * r) if self.slab_v2 else state * r.unsqueeze(-1)

        u = self.qkv(x)
        if self.slab_v2:
            state, hist = state
            window = torch.cat([hist, u.unsqueeze(-1)], dim=-1)          # (B, 3C, k)
            u = F.silu((window * self.conv_w).sum(-1) + self.conv_b)
            hist = window[:, :, 1:]
        qkv = u.view(B, 3, H, Dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, H, Dh)
        if self.slab_v2:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        log_alpha = -F.softplus(self.alpha_proj(x))  # (B, H)
        alpha = torch.exp(log_alpha)

        kv_outer = torch.einsum("bhd,bhe->bhde", k, v) * self.scale  # (B,H,Dh,Dh)
        new_state = alpha.unsqueeze(-1).unsqueeze(-1) * state + kv_outer
        y = torch.einsum("bhd,bhde->bhe", q, new_state)
        out = self._output(y, x)
        return out, ((new_state, hist) if self.slab_v2 else new_state)


class CausalDepthwiseConv1d(nn.Module):
    """Causal Depthwise Conv1d (k=4) for local token induction.

    Operates in dual-mode:
      - Training: Parallel Conv1d with left zero-padding (k-1)
      - Inference: Step-by-step rolling state buffer updates in O(1)
    """
    def __init__(self, dim: int, kernel_size: int = 4, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=0,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, l, d = x.shape
        if conv_state is not None and l == 1:
            x_step = x.transpose(1, 2)
            x_window = torch.cat([conv_state, x_step], dim=-1)
            new_conv_state = x_window[:, :, 1:]
            out = self.conv(x_window)
            out = F.silu(out).transpose(1, 2)
            return out, new_conv_state

        x_trans = x.transpose(1, 2)
        if conv_state is not None:
            x_padded = torch.cat([conv_state, x_trans], dim=-1)
            new_conv_state = x_padded[:, :, -self.pad:]
        else:
            x_padded = F.pad(x_trans, (self.pad, 0), mode="constant", value=0.0)
            new_conv_state = x_padded[:, :, -self.pad:]

        conv_out = self.conv(x_padded)
        out = F.silu(conv_out).transpose(1, 2)
        return out, new_conv_state


class GatedDeltaNetFunction(torch.autograd.Function):
    """Gated DeltaNet with decay-gated Widrow-Hoff error correction.

    Recurrent update:
      S_t = a_t * S_{t-1} + beta_t * K_t^T (V_t - a_t * K_t S_{t-1})
    """
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,      # (B, H, L, d_k)
        k: torch.Tensor,      # (B, H, L, d_k)
        v: torch.Tensor,      # (B, H, L, d_v)
        alpha: torch.Tensor,  # (B, H, L) - retention factor in (0, 1)
        beta: torch.Tensor,   # (B, H, L) - write rate in (0, 1)
        reset: torch.Tensor,  # (B, 1, L) - document reset mask (0 or 1)
    ) -> torch.Tensor:
        b, h, l, d_k = q.shape
        d_v = v.shape[-1]
        device = q.device
        dtype = q.dtype
        s_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32

        out = torch.empty((b, h, l, d_v), dtype=dtype, device=device)
        saved_states = torch.empty((l + 1, b, h, d_k, d_v), dtype=s_dtype, device=device)
        saved_errors = torch.empty((l, b, h, d_v), dtype=dtype, device=device)

        curr_s = torch.zeros((b, h, d_k, d_v), dtype=s_dtype, device=device)
        saved_states[0] = curr_s
        a_eff = alpha * reset

        for t in range(l):
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            a_t = a_eff[:, :, t, None, None].to(s_dtype)
            b_t = beta[:, :, t, None, None].to(s_dtype)

            s_decayed = a_t * curr_s
            v_ret = torch.matmul(k_t.unsqueeze(-2).to(s_dtype), s_decayed).squeeze(-2)
            e_t = v_t - v_ret.to(dtype)
            saved_errors[t] = e_t

            k_t_s = k_t.to(s_dtype)
            e_t_s = e_t.to(s_dtype)
            delta_s = b_t * torch.matmul(k_t_s.unsqueeze(-1), e_t_s.unsqueeze(-2))
            curr_s = s_decayed + delta_s
            saved_states[t + 1] = curr_s

            o_t = torch.matmul(q_t.unsqueeze(-2).to(s_dtype), curr_s).squeeze(-2)
            out[:, :, t] = o_t.to(dtype)

        ctx.save_for_backward(q, k, v, alpha, beta, reset, saved_states, saved_errors)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, alpha, beta, reset, saved_states, saved_errors = ctx.saved_tensors
        b, h, l, d_k = q.shape
        d_v = v.shape[-1]
        device = q.device
        s_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dalpha = torch.empty_like(alpha)
        dbeta = torch.empty_like(beta)
        a_eff = alpha * reset

        curr_ds = torch.zeros((b, h, d_k, d_v), dtype=s_dtype, device=device)

        for t in range(l - 1, -1, -1):
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            e_t = saved_errors[t]
            s_prev = saved_states[t]
            s_curr = saved_states[t + 1]
            dout_t = dout[:, :, t]
            b_t = beta[:, :, t, None, None].to(s_dtype)

            curr_ds = curr_ds + torch.matmul(
                q_t.unsqueeze(-1).to(s_dtype),
                dout_t.unsqueeze(-2).to(s_dtype)
            )
            dq_t = torch.matmul(
                dout_t.unsqueeze(-2).to(s_dtype),
                s_curr.transpose(-1, -2)
            ).squeeze(-2)
            dq[:, :, t] = dq_t.to(q.dtype)

            k_ds = torch.matmul(k_t.unsqueeze(-2).to(s_dtype), curr_ds)
            dv[:, :, t] = (b_t * k_ds).squeeze(-2).to(v.dtype)
            dbeta_t = torch.matmul(k_ds, e_t.unsqueeze(-1).to(s_dtype)).squeeze(-1).squeeze(-1)
            dbeta[:, :, t] = dbeta_t.to(beta.dtype)

            a_t_val = a_eff[:, :, t, None, None].to(s_dtype)
            term1 = torch.matmul(e_t.unsqueeze(-2).to(s_dtype), curr_ds.transpose(-1, -2))
            term2 = a_t_val * torch.matmul(k_ds, s_prev.transpose(-1, -2))
            dk_t = b_t * (term1 - term2)
            dk[:, :, t] = dk_t.squeeze(-2).to(k.dtype)

            r_t_val = reset[:, :, t].to(s_dtype)
            k_sprev = torch.matmul(k_t.unsqueeze(-2).to(s_dtype), s_prev)
            tr_term = torch.sum(curr_ds * s_prev, dim=(-2, -1))
            quad_term = torch.matmul(k_ds, k_sprev.transpose(-1, -2)).squeeze(-1).squeeze(-1)
            dalpha_t = r_t_val * (tr_term - beta[:, :, t].to(s_dtype) * quad_term)
            dalpha[:, :, t] = dalpha_t.to(alpha.dtype)

            k_proj = torch.matmul(k_t.unsqueeze(-1).to(s_dtype), k_ds)
            curr_ds = a_t_val * (curr_ds - b_t * k_proj)

        return dq, dk, dv, dalpha, dbeta, None



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

    # Document boundary reset parity test
    reset_mask = torch.ones(B, T, dtype=torch.float64)
    reset_mask[:, 15] = 0.0  # boundary at step 15 in all batches

    with torch.no_grad():
        y_dual_reset = attn(x, reset_mask=reset_mask)
        state_r = None
        steps_r = []
        for t in range(T):
            r_t = reset_mask[:, t]
            y_t, state_r = attn.step(x[:, t, :], state_r, reset=r_t)
            steps_r.append(y_t)
        y_rec_reset = torch.stack(steps_r, dim=1)

    delta_reset = (y_dual_reset - y_rec_reset).abs().max().item()
    assert delta_reset < 1e-9, f"reset dual/recurrent mismatch: max abs diff {delta_reset:.3e}"

    # CausalDepthwiseConv1d dual-mode parity test
    conv = CausalDepthwiseConv1d(cfg.n_embd, kernel_size=4).double().eval()
    conv_out_parallel, _ = conv(x)
    c_state = None
    c_steps = []
    for t in range(T):
        c_step_out, c_state = conv(x[:, t:t+1, :], conv_state=c_state)
        c_steps.append(c_step_out)
    conv_out_step = torch.cat(c_steps, dim=1)
    conv_diff = (conv_out_parallel - conv_out_step).abs().max().item()
    assert conv_diff < 1e-9, f"conv1d parallel/step mismatch: {conv_diff:.3e}"

    # GatedDeltaNet autograd check
    H = cfg.n_head
    Dh = cfg.n_embd // H
    q_dn = torch.randn(B, H, 10, Dh, dtype=torch.float64, requires_grad=True)
    k_dn = torch.randn(B, H, 10, Dh, dtype=torch.float64, requires_grad=True)
    v_dn = torch.randn(B, H, 10, Dh, dtype=torch.float64, requires_grad=True)
    alpha_dn = torch.rand(B, H, 10, dtype=torch.float64, requires_grad=True)
    beta_dn = torch.rand(B, H, 10, dtype=torch.float64, requires_grad=True)
    reset_dn = torch.ones(B, 1, 10, dtype=torch.float64)
    out_dn = GatedDeltaNetFunction.apply(q_dn, k_dn, v_dn, alpha_dn, beta_dn, reset_dn)
    loss_dn = out_dn.sum()
    loss_dn.backward()
    assert q_dn.grad is not None and k_dn.grad is not None and torch.isfinite(q_dn.grad).all()

    # Chunked dual/recurrent parity test (C=64, T=128 across 2 chunks)
    T_chunk = 128
    x_chunk = torch.randn(B, T_chunk, cfg.n_embd, dtype=torch.float64)
    with torch.no_grad():
        y_dual_chunk = attn(x_chunk)
        state_c = None
        steps_c = []
        for t in range(T_chunk):
            y_t, state_c = attn.step(x_chunk[:, t, :], state_c)
            steps_c.append(y_t)
        y_rec_chunk = torch.stack(steps_c, dim=1)
    delta_chunk = (y_dual_chunk - y_rec_chunk).abs().max().item()
    assert delta_chunk < 1e-9, f"chunked dual/recurrent mismatch: max abs diff {delta_chunk:.3e}"

    # Packed sequence isolation test (BRIEF_NEMOTRON.md Gate 1 requirement)
    L_A, L_B = 64, 64
    x_A = torch.randn(B, L_A, cfg.n_embd, dtype=torch.float64)
    x_B = torch.randn(B, L_B, cfg.n_embd, dtype=torch.float64)
    x_packed = torch.cat([x_A, x_B], dim=1)
    r_packed = torch.ones(B, L_A + L_B, dtype=torch.float64)
    r_packed[:, L_A] = 0.0  # boundary at start of B

    with torch.no_grad():
        y_A = attn(x_A)
        y_B = attn(x_B)
        y_packed = attn(x_packed, reset_mask=r_packed)
        # Shape test: (B, 1, L) acceptance
        y_packed_3d = attn(x_packed, reset_mask=r_packed.unsqueeze(1))

    delta_A = (y_packed[:, :L_A] - y_A).abs().max().item()
    delta_B = (y_packed[:, L_A:] - y_B).abs().max().item()
    delta_3d = (y_packed - y_packed_3d).abs().max().item()
    assert delta_A < 1e-9, f"packed A leakage: {delta_A:.3e}"
    assert delta_B < 1e-9, f"packed B leakage: {delta_B:.3e}"
    assert delta_3d == 0.0, f"(B, 1, L) mask mismatch: {delta_3d:.3e}"

    print(f"[self-test] selective_linear_attention: dual/recurrent parity ok "
          f"(dense diff {delta:.3e}, chunk diff {delta_chunk:.3e}, reset diff {delta_reset:.3e}, "
          f"packed isolation ok, conv diff {conv_diff:.3e}, deltanet ok)")


def self_test_v2():
    """slab_v2 gates (fp64): forward == step, chunked == dense, reset parity,
    packed isolation (conv taps included), and slab_v2=False keeps the old
    state_dict keys. Random conv taps, so the history path is actually tested."""
    torch.manual_seed(1)
    from dataclasses import replace
    base = ModelConfig(vocab_size=64, block_size=128, n_layer=2, n_head=4, n_embd=32)
    assert set(SelectiveLinearAttention(base).state_dict()) == {
        "qkv.weight", "alpha_proj.weight", "alpha_proj.bias", "proj.weight"}
    attn = SelectiveLinearAttention(replace(base, slab_v2=True)).double().eval()
    with torch.no_grad():
        attn.conv_w.normal_(0, 0.5)
        attn.conv_b.normal_(0, 0.1)
        attn.y_norm.normal_(1, 0.2)
    B, T = 3, 128

    def run_steps(x, reset=None):
        st, out = None, []
        for t in range(x.size(1)):
            y, st = attn.step(x[:, t], st, reset=None if reset is None else reset[:, t])
            out.append(y)
        return torch.stack(out, 1)

    x = torch.randn(B, T, 32, dtype=torch.float64)
    reset = torch.ones(B, T, dtype=torch.float64)
    reset[:, 37] = 0.0
    reset[1, 64] = 0.0       # boundary exactly on a chunk edge
    reset[2, 66] = 0.0       # boundary 2 tokens past a chunk edge (conv taps cross it)
    with torch.no_grad():
        d_step = (attn(x) - run_steps(x)).abs().max().item()
        d_dense = (attn(x) - attn._forward_dense(x)).abs().max().item()
        d_reset = (attn(x, reset_mask=reset) - run_steps(x, reset)).abs().max().item()
        d_reset_dense = (attn(x, reset_mask=reset) - attn._forward_dense(x, reset)).abs().max().item()
        xa, xb = torch.randn(B, 64, 32, dtype=torch.float64), torch.randn(B, 64, 32, dtype=torch.float64)
        r = torch.ones(B, 128, dtype=torch.float64)
        r[:, 64] = 0.0
        yp = attn(torch.cat([xa, xb], 1), reset_mask=r)
        d_iso = max((yp[:, :64] - attn(xa)).abs().max().item(), (yp[:, 64:] - attn(xb)).abs().max().item())
        x40 = x[:, :40]
        d_short = (attn(x40) - run_steps(x40)).abs().max().item()     # dense fallback path
    for name, d in (("step", d_step), ("dense", d_dense), ("reset", d_reset),
                    ("reset_dense", d_reset_dense), ("isolation", d_iso), ("short", d_short)):
        assert d < 1e-9, f"slab_v2 {name} mismatch {d:.3e}"
    one_minus = 1 - torch.exp(-F.softplus(attn.alpha_proj.bias))
    assert abs(one_minus.min().item() - 10 ** -4.5) < 1e-6, one_minus

    # Many chunks (closed-form inter-chunk recurrence): T=320 = 5 chunks, a document
    # spanning chunks 2-4, a reset on a chunk edge and one mid-chunk, both settings.
    # With only 2 chunks (T=128) the closed form is trivially exact, so this is the
    # test that actually exercises the decay matrix.
    for v2 in (False, True):
        a = SelectiveLinearAttention(replace(base, slab_v2=v2)).double().eval()
        if v2:
            with torch.no_grad():
                a.conv_w.normal_(0, 0.5)
        xs = torch.randn(2, 320, 32, dtype=torch.float64)
        r5 = torch.ones(2, 320, dtype=torch.float64)
        r5[0, 64] = 0.0
        r5[0, 150] = 0.0          # doc 150..319 spans chunks 2, 3, 4
        r5[1, 200] = 0.0          # row 1: doc 0..199 spans chunks 0-3
        st, outs = None, []
        with torch.no_grad():
            for t in range(320):
                y, st = a.step(xs[:, t], st, reset=r5[:, t])
                outs.append(y)
            ref = torch.stack(outs, 1)
            for rm, name in ((r5, "reset"), (None, "plain")):
                got = a(xs, reset_mask=rm)
                dense = a._forward_dense(xs, rm)
                if rm is None:
                    st, outs = None, []
                    for t in range(320):
                        y, st = a.step(xs[:, t], st)
                        outs.append(y)
                    want = torch.stack(outs, 1)
                else:
                    want = ref
                d1, d2 = (got - want).abs().max().item(), (got - dense).abs().max().item()
                assert d1 < 1e-9 and d2 < 1e-9, f"5-chunk {name} slab_v2={v2}: step {d1:.2e} dense {d2:.2e}"
    # gradients flow and stay finite through the closed form (masked exp must not nan)
    xg = torch.randn(2, 320, 32, dtype=torch.float64, requires_grad=True)
    a(xg, reset_mask=r5).sum().backward()
    assert torch.isfinite(xg.grad).all() and all(torch.isfinite(p.grad).all() for p in a.parameters()
                                                   if p.grad is not None)
    print(f"[self-test] slab_v2: step {d_step:.1e}, dense {d_dense:.1e}, reset {d_reset:.1e}, "
          f"isolation {d_iso:.1e} ok")


if __name__ == "__main__":
    self_test()
    self_test_v2()

