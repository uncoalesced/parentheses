"""slab_v4: a recurrent language model built on selective linear attention.

There is no attention matrix over the sequence and no cache that grows with it.
Each layer keeps one small matrix per head, and every token does two things to
it: decays what is there, and writes one key/value pair.

    S_t = alpha_t * S_{t-1} + beta_t * k_t v_t^T
    y_t = q_t S_t

The same function is computed two ways:

    forward()  the parallel form. A scan over blocks of `chunk_size` tokens.
               Used for training, scoring and reading a prompt quickly.
    step()     the recurrent form. One token in, one state update, constant
               memory at any length. Used for generation.

`forward_with_state()` joins them: read a prompt with the parallel form, get
back the recurrent state, and continue with `step()`.

Parameter names match the released 1.0 weights, so they load unchanged.
"""

from __future__ import annotations

import math
from typing import Iterator, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SlabConfig

CONV_KERNEL = 4


class LayerState(NamedTuple):
    """Everything one layer remembers between tokens. Fixed size."""

    memory: torch.Tensor   # (B, H, Dh, Dh)  the associative state S, float32 or wider
    conv: torch.Tensor     # (B, K-1, 3D)    last projections, for the causal conv
    prev: torch.Tensor     # (B, D)          previous layer input, for the token shift


def _wide(t: torch.Tensor) -> torch.dtype:
    """Accumulate in float32 at least. Half precision loses long-range decay."""
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def find_documents(ids: torch.Tensor, eot_token_id: int, prev_eot: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mark where documents start. Returns (reset, doc).

    reset[b, t] is True when token t is the first of a new document, which is
    the token after <|endoftext|>. doc[b, t] is a running document number.
    `prev_eot` says the token before this sequence was <|endoftext|>.
    """
    reset = torch.zeros_like(ids, dtype=torch.bool)
    reset[:, 1:] = ids[:, :-1] == eot_token_id
    if prev_eot:
        reset[:, 0] = True
    return reset, torch.cumsum(reset.to(torch.int64), dim=1)


def chunk_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, log_alpha: torch.Tensor,
               reset: torch.Tensor, doc: torch.Tensor, chunk: int,
               memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel form of the recurrence. Returns (y, final memory).

    q, k, v: (B, T, H, Dh), with the write gate already folded into v.
    log_alpha: (B, T, H), log of the per-token decay.
    T must be a multiple of `chunk`.

    Inside a block, tokens see each other through a decay matrix. Between
    blocks, one state matrix is handed forward. Nothing crosses a document
    boundary.
    """
    B, T, H, Dh = q.shape
    N = T // chunk
    wide = _wide(q)
    neg_inf = float("-inf")

    def blocks(t: torch.Tensor) -> torch.Tensor:          # (B, T, H, x) -> (B, N, H, chunk, x)
        return t.to(wide).view(B, N, chunk, H, -1).transpose(2, 3)

    qc, kc, vc = blocks(q), blocks(k), blocks(v)
    la = log_alpha.to(wide).view(B, N, chunk, H).transpose(2, 3)      # (B, N, H, chunk)
    dc = doc.view(B, N, chunk)
    starts_new = reset.view(B, N, chunk)[:, :, 0]

    P = la.cumsum(-1)                                                # log decay from block start

    # Within a block: token i reads token j <= i of the same document.
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    same_doc = (dc.unsqueeze(-1) == dc.unsqueeze(-2)).unsqueeze(2)   # (B, N, 1, chunk, chunk)
    visible = causal & same_doc
    decay = torch.exp((P.unsqueeze(-1) - P.unsqueeze(-2)).masked_fill(~visible, neg_inf))
    y = torch.matmul(torch.matmul(qc, kc.transpose(-1, -2)) * decay, vc)

    # Between blocks. A block's incoming state belongs to the document that was
    # open just before its first token; only tokens of that document read it.
    doc_before = dc[:, :, 0] - starts_new.to(dc.dtype)               # (B, N)
    doc_at_end = dc[:, :, -1]
    in_last_doc = (dc == doc_at_end.unsqueeze(-1)).unsqueeze(2)      # (B, N, 1, chunk)
    in_first_doc = (dc == doc_before.unsqueeze(-1)).unsqueeze(2)

    to_end = torch.exp((P[..., -1:] - P).masked_fill(~in_last_doc, neg_inf))
    from_start = torch.exp(P.masked_fill(~in_first_doc, neg_inf))
    survives = torch.exp(P[..., -1]) * (doc_at_end == doc_before).unsqueeze(-1).to(wide)   # (B, N, H)
    written = torch.matmul((kc * to_end.unsqueeze(-1)).transpose(-1, -2), vc)   # (B, N, H, Dh, Dh)

    S = (torch.zeros(B, H, Dh, Dh, dtype=wide, device=q.device)
         if memory is None else memory.to(wide))
    carried = []
    for c in range(N):
        carried.append(torch.matmul(qc[:, c] * from_start[:, c].unsqueeze(-1), S))
        S = survives[:, c].view(B, H, 1, 1) * S + written[:, c]
    y = y + torch.stack(carried, dim=1)
    return y.transpose(2, 3).reshape(B, T, H, Dh).to(q.dtype), S


def sample_next(scores: torch.Tensor, temperature: float = 0.0, top_k: Optional[int] = None,
                min_p: Optional[float] = None) -> int:
    """Pick the next token from logits (vocab,). `temperature=0` is greedy. `min_p`
    drops tokens less likely than min_p times the most likely one."""
    scores = scores.float()
    if temperature <= 1e-4:
        return int(scores.argmax())
    scores = scores / temperature
    if top_k:
        kth = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    if min_p:
        probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
        probs = probs / probs.sum().clamp_min(1e-9)
    return int(torch.multinomial(probs, 1))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, cfg: SlabConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SelectiveLinearAttention(nn.Module):
    """One recurrent mixing layer.

    The input is blended with the previous token (token shift) to drive the
    gates. Queries, keys and values come from a linear projection followed by a
    short causal convolution. Layers after the first mix their values with the
    first layer's values.
    """

    def __init__(self, cfg: SlabConfig, layer_idx: int):
        super().__init__()
        D, H = cfg.n_embd, cfg.n_head
        self.layer_idx = layer_idx
        self.n_embd, self.n_head, self.head_dim = D, H, cfg.head_dim
        self.chunk_size = cfg.chunk_size
        self.beta_scale = cfg.beta_scale

        self.mu = nn.Parameter(torch.full((D,), 0.5))
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.conv_weight = nn.Parameter(torch.zeros(3 * D, 1, CONV_KERNEL))
        self.conv_bias = nn.Parameter(torch.zeros(3 * D))
        self.alpha_proj = nn.Linear(D, H, bias=True)
        self.beta_proj = nn.Linear(D, H, bias=True)
        if layer_idx > 0:
            self.v_residual_lamb = nn.Parameter(torch.zeros(H))
        self.y_norm = nn.Parameter(torch.ones(H, cfg.head_dim))
        self.gate = nn.Linear(D, D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    # -- shared pieces ------------------------------------------------------

    def _split(self, u: torch.Tensor, v0: Optional[torch.Tensor]):
        """Convolved projection (B, T, 3D) -> q, k, v of shape (B, T, H, Dh), and v0."""
        B, T, _ = u.shape
        H, Dh = self.n_head, self.head_dim
        q, k, v = F.silu(u).view(B, T, 3, H, Dh).unbind(dim=2)
        wide = _wide(u)
        q = F.normalize(q.to(wide), dim=-1).to(u.dtype) * (1.0 / math.sqrt(Dh))
        k = F.normalize(k.to(wide), dim=-1).to(u.dtype)
        v0_out = v if self.layer_idx == 0 else v0
        if self.layer_idx > 0 and v0 is not None:
            mix = torch.sigmoid(self.v_residual_lamb).view(1, 1, H, 1).to(v.dtype)
            v = mix * v + (1.0 - mix) * v0
        return q, k, v, v0_out

    def _gates(self, x_ctrl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta_scale * torch.sigmoid(self.beta_proj(x_ctrl))
        log_alpha = -F.softplus(self.alpha_proj(x_ctrl))
        return beta, log_alpha

    def _emit(self, y: torch.Tensor, x_ctrl: torch.Tensor) -> torch.Tensor:
        """y: (..., H, Dh). Normalise each head, gate, project back to (..., D)."""
        yw = y.to(_wide(y))
        y = (yw * torch.rsqrt(yw.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype)
        y = (y * self.y_norm.to(y.dtype)).flatten(-2)
        return self.proj(y * F.silu(self.gate(x_ctrl)))

    # -- parallel form ------------------------------------------------------

    def forward(self, x: torch.Tensor, reset: torch.Tensor, doc: torch.Tensor,
                v0: Optional[torch.Tensor] = None, state: Optional[LayerState] = None,
                n_real: Optional[int] = None):
        """x: (B, T, D) with T a multiple of chunk_size. Positions from `n_real`
        on are padding: they write nothing and decay nothing.
        Returns (output, v0, state after token n_real - 1)."""
        B, T, D = x.shape
        K = CONV_KERNEL
        n_real = T if n_real is None else n_real

        prev = x.new_zeros(B, 1, D) if state is None else state.prev.unsqueeze(1).to(x.dtype)
        x_prev = torch.cat([prev, x[:, :-1]], dim=1) * (~reset).unsqueeze(-1).to(x.dtype)
        x_ctrl = self.mu * x + (1.0 - self.mu) * x_prev

        u = self.qkv(x)
        history = u.new_zeros(B, K - 1, 3 * D) if state is None else state.conv.to(u.dtype)
        u_padded = torch.cat([history, u], dim=1)                          # (B, T + K - 1, 3D)
        doc_before = doc[:, :1] - reset[:, :1].to(doc.dtype)
        doc_padded = torch.cat([doc_before.expand(B, K - 1), doc], dim=1)
        taps = u_padded.unfold(1, K, 1)                                    # (B, T, 3D, K)
        tap_doc = doc_padded.unfold(1, K, 1)                               # (B, T, K)
        tap_ok = (tap_doc == tap_doc[:, :, -1:]).unsqueeze(2).to(u.dtype)
        u_conv = (taps * tap_ok * self.conv_weight.squeeze(1)).sum(-1) + self.conv_bias

        q, k, v, v0_out = self._split(u_conv, v0)
        beta, log_alpha = self._gates(x_ctrl)
        if n_real < T:
            live = (torch.arange(T, device=x.device) < n_real).view(1, T, 1).to(beta.dtype)
            beta, log_alpha = beta * live, log_alpha * live
        y, memory = chunk_scan(q, k, v * beta.unsqueeze(-1), log_alpha, reset, doc,
                               self.chunk_size, None if state is None else state.memory)
        out = self._emit(y, x_ctrl)

        conv = u_padded[:, n_real:n_real + K - 1]
        conv = conv * (doc_padded[:, n_real:n_real + K - 1] == doc[:, n_real - 1:n_real]
                       ).unsqueeze(-1).to(conv.dtype)
        return out, v0_out, LayerState(memory, conv, x[:, n_real - 1])

    # -- recurrent form -----------------------------------------------------

    def step(self, x: torch.Tensor, state: Optional[LayerState], boundary: bool,
             v0: Optional[torch.Tensor] = None):
        """x: (B, D), one token. Returns (output, new state, v0)."""
        B, D = x.shape
        H, Dh = self.n_head, self.head_dim
        wide = _wide(x)
        if state is None:
            memory = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=wide)
            conv = x.new_zeros(B, CONV_KERNEL - 1, 3 * D)
            prev = x.new_zeros(B, D)
        else:
            memory, conv, prev = state
        if boundary:
            conv, prev = torch.zeros_like(conv), torch.zeros_like(prev)

        x_ctrl = self.mu * x + (1.0 - self.mu) * prev.to(x.dtype)
        window = torch.cat([conv.to(x.dtype), self.qkv(x).unsqueeze(1)], dim=1)      # (B, K, 3D)
        u_conv = (window.transpose(1, 2) * self.conv_weight.squeeze(1)).sum(-1) + self.conv_bias
        q, k, v, v0_out = self._split(u_conv.unsqueeze(1), v0)

        beta, log_alpha = self._gates(x_ctrl)
        alpha = torch.exp(log_alpha.to(wide)).view(B, H, 1, 1)
        if boundary:
            alpha = torch.zeros_like(alpha)
        q1, k1, v1 = q[:, 0].to(wide), k[:, 0].to(wide), v[:, 0].to(wide)
        write = beta.to(wide).view(B, H, 1, 1) * torch.matmul(k1.unsqueeze(-1), v1.unsqueeze(-2))
        memory = alpha * memory.to(wide) + write
        y = torch.matmul(q1.unsqueeze(-2), memory).squeeze(-2).to(x.dtype)           # (B, H, Dh)
        return self._emit(y, x_ctrl), LayerState(memory, window[:, 1:], x), v0_out


class Block(nn.Module):
    def __init__(self, cfg: SlabConfig, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelectiveLinearAttention(cfg, layer_idx)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, reset, doc, v0=None, state=None, n_real=None):
        mixed, v0, state = self.attn(self.ln1(x), reset, doc, v0, state, n_real)
        x = x + mixed
        return x + self.mlp(self.ln2(x)), v0, state

    def step(self, x, state, boundary, v0=None):
        mixed, state, v0 = self.attn.step(self.ln1(x), state, boundary, v0)
        x = x + mixed
        return x + self.mlp(self.ln2(x)), state, v0


class Parentheses(nn.Module):
    """The language model: embedding, a stack of blocks, a tied output head."""

    def __init__(self, cfg: SlabConfig = SlabConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.reset_parameters()
        self.head.weight = self.tok_emb.weight

    def reset_parameters(self) -> None:
        """Fresh initialisation. A loaded checkpoint overwrites all of it."""
        cfg = self.cfg
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        # Heads start with different memory spans, from about 10 tokens to tens of thousands.
        forget = torch.logspace(-1, -4.5, cfg.n_head)
        decay_bias = torch.log(torch.expm1(-torch.log1p(-forget)))
        with torch.no_grad():
            for block in self.blocks:
                a = block.attn
                a.conv_weight.zero_()
                a.conv_weight[:, 0, -1] = 1.0          # the conv starts as a pass-through
                a.alpha_proj.bias.copy_(decay_bias)
                a.beta_proj.weight.zero_()
                nn.init.normal_(a.proj.weight, std=residual_std)
                nn.init.normal_(block.mlp.w2.weight, std=residual_std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- parallel form ------------------------------------------------------

    def forward_with_state(self, ids: torch.Tensor,
                           states: Optional[Sequence[Optional[LayerState]]] = None,
                           prev_eot: bool = False) -> Tuple[torch.Tensor, List[LayerState]]:
        """Logits for every position, plus the recurrent state after the last token.

        ids: (B, T), any T >= 1. Pass `states` from an earlier call (or from
        `step()`) to continue a sequence, and `prev_eot=True` if the token just
        before `ids` was <|endoftext|>.
        """
        B, T = ids.shape
        cfg = self.cfg
        reset, doc = find_documents(ids, cfg.eot_token_id, prev_eot)
        pad = (-T) % cfg.chunk_size
        if pad:
            # The scan works on whole blocks. Padding goes on the right, belongs
            # to the last document, and is switched off inside each layer.
            ids = F.pad(ids, (0, pad), value=cfg.pad_token_id)
            reset = F.pad(reset, (0, pad), value=False)
            doc = torch.cat([doc, doc[:, -1:].expand(B, pad)], dim=1)
        if states is None:
            states = [None] * len(self.blocks)

        x = self.tok_emb(ids)
        v0, new_states = None, []
        for block, state in zip(self.blocks, states):
            x, v0, state = block(x, reset, doc, v0, state, T)
            new_states.append(state)
        return self.head(self.ln_f(x[:, :T])), new_states

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits (B, T, vocab) for a batch of independent sequences."""
        return self.forward_with_state(ids)[0]

    # -- recurrent form -----------------------------------------------------

    def step(self, token: torch.Tensor,
             states: Optional[Sequence[Optional[LayerState]]] = None,
             boundary: bool = False) -> Tuple[torch.Tensor, List[LayerState]]:
        """One token per row. Returns next-token logits (B, vocab) and the new state.

        Set `boundary=True` for the token that follows <|endoftext|>.
        """
        if token.dim() == 2:
            token = token[:, 0]
        if states is None:
            states = [None] * len(self.blocks)
        x = self.tok_emb(token)
        v0, new_states = None, []
        for block, state in zip(self.blocks, states):
            x, state, v0 = block.step(x, state, boundary, v0)
            new_states.append(state)
        return self.head(self.ln_f(x)), new_states

    @staticmethod
    def state_bytes(states: Sequence[LayerState]) -> int:
        return sum(t.numel() * t.element_size() for layer in states for t in layer)

    @torch.no_grad()
    def stream(self, ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 0.0,
               top_k: Optional[int] = None, stop_token_ids: Optional[Sequence[int]] = None
               ) -> Iterator[int]:
        """Generate from one sequence (1, T). Yields token ids.

        The prompt is read once with the parallel form; after that each new
        token is a single recurrent step. `temperature=0` is greedy.
        """
        if ids.dim() != 2 or ids.size(0) != 1:
            raise ValueError("stream() takes one sequence of shape (1, T)")
        eot = self.cfg.eot_token_id
        stops = {eot} if stop_token_ids is None else set(stop_token_ids)
        logits, states = self.forward_with_state(ids)
        scores = logits[0, -1]
        prev_eot = int(ids[0, -1]) == eot
        for _ in range(max_new_tokens):
            next_id = sample_next(scores, temperature, top_k)
            if next_id in stops:
                return
            yield next_id
            token = torch.tensor([next_id], dtype=torch.long, device=ids.device)
            step_logits, states = self.step(token, states, boundary=prev_eot)
            scores, prev_eot = step_logits[0], next_id == eot
