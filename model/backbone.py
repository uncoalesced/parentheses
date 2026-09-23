"""
A small, dependency-light byte-level recurrent model (RWKV-style Selective Linear Attention) for Parentheses 0.9.

Engineered by uncoalesced

Deliberately close to RWKV/Mamba-2-style recurrent architecture (RMSNorm, RoPE, SwiGLU
MLP) with Selective Linear Attention for O(1) state updates and streaming generation.
Kept in a single file so it's easy to read end to end.
"""

import itertools
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, PRESETS
from .selective_linear_attention import SelectiveLinearAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, base: int = 10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: int = 0) -> torch.Tensor:
    # x: (B, n_head, T, head_dim). `pos` is the absolute position of x[..., 0, :],
    # nonzero only when decoding into a KV-cache that already holds `pos` tokens.
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[pos: pos + x.size(-2)].unsqueeze(0).unsqueeze(0)
    sin = sin[pos: pos + x.size(-2)].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        cos, sin = precompute_rope(self.head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, cache: list | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_head, T, head_dim)
        # `cache` is this layer's KV slot: [] before the first step, [k, v] after.
        # Its length doubles as the absolute position the new tokens start at, so
        # no separate position counter has to be threaded down from generate().
        pos = cache[0].size(2) if cache else 0
        q = apply_rope(q, self.rope_cos, self.rope_sin, pos)
        k = apply_rope(k, self.rope_cos, self.rope_sin, pos)
        if cache is not None:
            if cache:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            cache[:] = (k, v)
        # T == k_len: ordinary causal pass (training, or a cache prefill).
        # T == 1 with a longer cache: the new token attends to all of it, and
        # is_causal would wrongly mask it down to the first cached key (SDPA
        # aligns the causal mask top-left, not bottom-right).
        assert T == 1 or T == k.size(2), "chunked prefill into a non-empty cache needs an explicit mask"
        # cache is None (training, or embedding via hidden()) always means an
        # ordinary causal pass -- T == k.size(2) is trivially true there, but
        # under torch.onnx.export's tracer with a dynamic seq-len axis, T is a
        # traced value rather than a Python int, so the comparison would trace
        # to a tensor and scaled_dot_product_attention requires is_causal to be
        # a plain bool. Only the cache branch (incremental generation, never
        # traced) needs the dynamic comparison at all.
        is_causal = True if cache is None else T == k.size(2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)  # LLaMA-style expansion, rounded via multiple below
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        if cfg.attn_type == "causal":
            self.attn = CausalSelfAttention(cfg)
        elif cfg.attn_type == "selective_linear":
            self.attn = SelectiveLinearAttention(cfg)
        else:
            raise ValueError(
                f"unknown attn_type {cfg.attn_type!r} -- expected 'causal' or 'selective_linear'")
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    @property
    def is_selective(self) -> bool:
        return isinstance(self.attn, SelectiveLinearAttention)

    def forward(self, x: torch.Tensor, cache: list | None = None,
                reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.is_selective:
            # No KV cache exists for this path -- its incremental form is
            # step()/Parentheses.step(), carrying a fixed-size matrix state.
            # Silently ignoring a cache would let stream()/generate() run and
            # return plausible-looking garbage with no state carried at all.
            assert cache is None, "selective_linear attention has no KV cache -- use Parentheses.step()"
            x = x + self.attn(self.ln1(x), reset_mask=reset_mask)
        else:
            # Same "loud on the wrong path" rule as the cache assert above:
            # CausalSelfAttention has no recurrent state to reset, so a
            # reset_mask reaching here would be silently dropped and a
            # document-packed run would train with cross-document leakage
            # while looking like it was isolating boundaries.
            assert reset_mask is None, (
                "reset_mask is selective_linear-only -- causal attention carries no "
                "recurrent state across a document boundary; use a --preset with "
                "attn_type='selective_linear'")
            x = x + self.attn(self.ln1(x), cache)
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x: torch.Tensor, state):
        """One token through the block, recurrent form. x: (B, C) -> ((B, C), new_state)."""
        if not self.is_selective:
            raise RuntimeError(
                "Block.step() requires attn_type='selective_linear'; this block uses causal "
                "attention, whose incremental path is forward(x, cache) with empty_cache().")
        attn_out, new_state = self.attn.step(self.ln1(x), state)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_state


class Parentheses(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # _init_weights zeros every nn.Linear bias, which would undo the decay
        # gate's deliberately spread init (zero bias => alpha~0.5, state wiped
        # every ~5 bytes). Re-apply it after the sweep. No-op on causal presets.
        for m in self.modules():
            if isinstance(m, SelectiveLinearAttention):
                m.init_decay_bias()
        if cfg.tie_embeddings:
            # At sub-1M-param scale the input+output embedding tables can be
            # most of the whole budget (see ModelConfig.approx_params /
            # docs/training-time-estimate.md) -- tying them to one shared
            # matrix is not optional here, it's the difference between a
            # model that fits the target size and one that's 2x over.
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def empty_cache(self) -> list[list]:
        """One empty KV slot per layer. Pass to forward()/stream() to enable caching."""
        return [[] for _ in self.blocks]

    def empty_states(self, batch_size: int, device=None, dtype=None) -> list:
        """One zero recurrent state per layer, for step(). selective_linear only.

        The selective-linear counterpart to empty_cache(): fixed size
        (n_layer x B x n_head x head_dim x head_dim) no matter how long the run
        goes, which is the whole point of the pivot for Free Think streaming.
        """
        self._require_selective("empty_states()")
        return [b.attn.empty_state(batch_size, device, dtype) for b in self.blocks]

    def _require_selective(self, what: str):
        if not all(b.is_selective for b in self.blocks):
            raise RuntimeError(
                f"{what} requires attn_type='selective_linear'; this model uses causal "
                f"attention -- use empty_cache() / forward(idx, cache=...) instead.")

    def step(self, idx: torch.Tensor, states: list | None = None):
        """One token per sequence through the recurrent path. selective_linear only.

        idx: (B,) or (B, 1) token ids. states: the list from empty_states(), or
        None for a fresh (zero) state. Returns (logits (B, vocab_size), states).
        """
        self._require_selective("step()")
        if idx.dim() == 2:
            assert idx.size(1) == 1, f"step() takes one token per sequence, got {tuple(idx.shape)}"
            idx = idx[:, 0]
        if states is None:
            states = [None] * len(self.blocks)
        x = self.drop(self.tok_emb(idx))  # (B, C)
        new_states = []
        for block, state in zip(self.blocks, states):
            x, state = block.step(x, state)
            new_states.append(state)
        return self.head(self.ln_f(x)), new_states

    def hidden(self, idx: torch.Tensor, cache: list | None = None,
               reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Final-layer hidden states (B, T, n_embd), post-ln_f, pre-LM-head.

        Split out of forward() so model/embedding_head.py can mean-pool these
        without re-implementing the block walk or paying for a vocab-sized
        projection it would only throw away. forward() calls it, so there is
        still exactly one definition of the stack.
        """
        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, None if cache is None else cache[i], reset_mask)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: list | None = None, reset_mask: torch.Tensor | None = None):
        x = self.hidden(idx, cache, reset_mask)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def stream(self, idx: torch.Tensor, max_new_tokens: int | None = None,
               temperature: float = 1.0, top_k: int | None = None,
               min_p: float | None = None, sink_tokens: int = 0):
        """Yield sampled token ids (B, 1) one at a time, KV-cached.

        max_new_tokens=None streams until the consumer stops iterating -- that's
        the open-ended decode loop Free Think Mode (features/free_think.py) runs on.

        min_p: dynamic confidence-based truncation floor (typically 0.05-0.10).
        Truncates candidate tokens whose probability is below min_p * p_max.

        Context handling: the window is the last block_size tokens, same as the
        uncached loop's idx[:, -block_size:], and output matches that loop exactly
        until the window first fills. After that the cache is dropped and the next
        step re-prefills from half a window, so decoding stays cached for another
        block_size//2 steps instead of paying a full-prefix pass per token.

        sink_tokens: keep this many tokens from the *start* of `idx` pinned into
        every post-reset window (StreamingLLM-style attention sink). Without
        this, a run that goes on well past block_size has zero memory of its
        own opening once the window first slides past it -- each post-reset
        segment is locally coherent but has completely forgotten where the run
        started. Clamped to at most a quarter of block_size, so the reset
        window is never more than half sink. 0 = old behavior (no sink).
        On the selective_linear path sink_tokens is accepted and ignored: the
        recurrent state is a fixed-size matrix, so there is no window to slide
        past and nothing to pin. Callers pass it unconditionally.
        """
        if self.cfg.attn_type == "selective_linear":
            states = self.empty_states(idx.size(0), device=idx.device,
                                       dtype=self.tok_emb.weight.dtype)
            for i in range(idx.size(1) - 1):  # prefill all but the last byte
                _, states = self.step(idx[:, i:i + 1], states)
            curr = idx[:, -1:]
            steps = itertools.count() if max_new_tokens is None else range(max_new_tokens)
            for _ in steps:
                logits, states = self.step(curr, states)  # already (B, vocab_size)
                logits = logits / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if min_p is not None and min_p > 0.0:
                    p_max = probs.max(dim=-1, keepdim=True).values
                    p_threshold = min_p * p_max
                    probs = torch.where(probs >= p_threshold, probs, torch.zeros_like(probs))
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                next_id = torch.multinomial(probs, num_samples=1)
                yield next_id
                curr = next_id
            return

        cache = self.empty_cache()
        keep = self.cfg.block_size // 2
        sink_tokens = min(sink_tokens, keep // 2)
        sink = idx[:, :sink_tokens] if sink_tokens else None
        steps = itertools.count() if max_new_tokens is None else range(max_new_tokens)
        for _ in steps:
            step_in = idx[:, -1:] if cache[0] else idx[:, -self.cfg.block_size:]
            logits, _ = self(step_in, cache=cache)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if min_p is not None and min_p > 0.0:
                p_max = probs.max(dim=-1, keepdim=True).values
                p_threshold = min_p * p_max
                probs = torch.where(probs >= p_threshold, probs, torch.zeros_like(probs))
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            next_id = torch.multinomial(probs, num_samples=1)
            # only the last block_size tokens are ever read back, so don't let an
            # open-ended session grow idx forever
            idx = torch.cat([idx, next_id], dim=1)[:, -self.cfg.block_size:]
            if cache[0][0].size(2) >= self.cfg.block_size:
                # ponytail: window full -> drop the cache and restart from half a
                # window (plus the pinned sink, if any), buying block_size//2
                # cached steps before the next prefill. Ceiling: context dips to
                # block_size//2 right after each reset (resetting to a *full*
                # window would re-fill it instantly and prefill every step, i.e.
                # no cache at all). Upgrade path is evicting just the oldest KV
                # entries instead, which needs an absolute position counter
                # threaded through forward() -- cache length stops tracking it
                # once entries are dropped -- and a RoPE table that grows past
                # block_size.
                cache = self.empty_cache()
                if sink is not None:
                    idx = torch.cat([sink, idx[:, -(keep - sink_tokens):]], dim=1)
                else:
                    idx = idx[:, -keep:]
            yield next_id

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, min_p: float | None = None):
        for next_id in self.stream(idx, max_new_tokens, temperature, top_k, min_p=min_p):
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def self_test():
    """Cached decoding must match the naive re-run-the-whole-prefix loop.

    Run with `python -m model.backbone` (relative imports need the package).
    """
    torch.manual_seed(0)
    cfg = PRESETS["tiny-smoke"]
    model = Parentheses(cfg).eval()

    # 1. prefill + incremental steps == one full forward over the same tokens
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    cache = model.empty_cache()
    ref, _ = model(idx)
    got, _ = model(idx[:, :3], cache=cache)
    for t in range(3, 5):
        got, _ = model(idx[:, t:t + 1], cache=cache)
    delta = (ref[:, -1] - got[:, -1]).abs().max()
    assert delta < 1e-4, delta

    # 2. greedy generation matches the uncached loop while the window fills
    #    (10 prompt + 50 new < block_size 64)
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    cached = model.generate(prompt, 50, top_k=1)
    naive = prompt
    for _ in range(50):
        logits, _ = model(naive[:, -cfg.block_size:])
        naive = torch.cat([naive, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(cached, naive), (cached != naive).nonzero()[:5]

    # 3. past block_size, decoding must stay cached rather than falling back to
    #    a full-prefix pass per step (the whole point of the reset-to-half-window)
    widths = []
    handle = model.register_forward_pre_hook(lambda m, a: widths.append(a[0].size(1)))
    out = model.generate(prompt, 4 * cfg.block_size, top_k=1)
    handle.remove()
    prefills = sum(1 for w in widths if w > 1)
    assert prefills < 20, f"{prefills} prefills in {4 * cfg.block_size} tokens -- cache is degenerating"
    assert out.shape == (1, 10 + 4 * cfg.block_size) and out.min() >= 0 and out.max() < cfg.vocab_size

    # 4. sink_tokens must survive every window reset -- the whole point of the
    #    attention-sink Free Think uses to stay anchored on long runs.
    sink_n = 3
    sink_vals = prompt[:, :sink_n].clone()
    prefill_inputs = []
    handle = model.register_forward_pre_hook(
        lambda m, a: prefill_inputs.append(a[0]) if a[0].size(1) > 1 else None
    )
    list(model.stream(prompt, 4 * cfg.block_size, top_k=1, sink_tokens=sink_n))
    handle.remove()
    assert len(prefill_inputs) > 1, "need at least one reset to test sink retention"
    for p in prefill_inputs[1:]:  # [0] is the initial prompt prefill, not a reset
        assert torch.equal(p[:, :sink_n], sink_vals), "sink tokens lost after a window reset"
    print("[self-test] kv-cache ok")


def self_test_selective():
    """Whole-model forward() must equal token-by-token step() accumulation.

    model/selective_linear_attention.py already proves one block's dual and
    recurrent forms agree; this is the *wiring* regression test -- that
    Block.step and Parentheses.step thread state, the pre-norms, the residual
    stream, ln_f and the head in the same order forward() does. A bug in the
    wiring (e.g. normalizing the residual instead of the branch, or dropping a
    layer's state) shows up here and nowhere else.

    float64 for the same reason the attention file uses it: cumsum over T steps
    and T sequential multiplies round differently even when both are correct, so
    fp32 noise would mask a real algorithmic mismatch.
    """
    torch.manual_seed(0)
    cfg = replace(PRESETS["tiny-smoke"], attn_type="selective_linear")
    model = Parentheses(cfg).double().eval()

    B, T = 2, 24
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        ref, _ = model(idx)
        states = model.empty_states(B, dtype=torch.float64)
        steps = []
        for t in range(T):
            logits, states = model.step(idx[:, t], states)
            steps.append(logits)
        got = torch.stack(steps, dim=1)

    assert ref.shape == got.shape == (B, T, cfg.vocab_size), (ref.shape, got.shape)
    assert torch.isfinite(ref).all() and torch.isfinite(got).all()
    delta = (ref - got).abs().max().item()
    assert delta < 1e-9, f"forward/step mismatch: max abs diff {delta:.3e}"

    head_dim = cfg.n_embd // cfg.n_head
    assert len(states) == cfg.n_layer
    assert all(s.shape == (B, cfg.n_head, head_dim, head_dim) for s in states)

    # A causal model must refuse the recurrent path loudly rather than dying on
    # a cryptic AttributeError three frames down.
    causal = Parentheses(PRESETS["tiny-smoke"]).eval()
    for name, call in (("step", lambda: causal.step(idx[:, 0])),
                       ("empty_states", lambda: causal.empty_states(B)),
                       ("Block.step", lambda: causal.blocks[0].step(torch.zeros(B, causal.cfg.n_embd), None))):
        try:
            call()
        except RuntimeError as e:
            assert "selective_linear" in str(e), (name, e)
        else:
            raise AssertionError(f"causal model accepted {name}()")

    print(f"[self-test] selective-linear wiring: forward/step parity ok "
          f"(max abs diff {delta:.3e})")


def self_test_selective_stream():
    """stream()/generate() on the recurrent path must match hand-driven step().

    self_test_selective() proves forward() == step(); this proves stream()
    actually *uses* step() correctly -- prefills every prompt byte but the last,
    threads state across yields, and feeds each sampled token back in. A branch
    that dropped the prefill, or reset the state per token, would still emit
    plausible bytes and pass every other test in this file.
    """
    torch.manual_seed(0)
    cfg = replace(PRESETS["tiny-smoke"], attn_type="selective_linear")
    model = Parentheses(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (2, 7))
    N = 12
    got = model.generate(prompt, N, top_k=1)  # greedy -> deterministic

    # independent re-derivation: drive empty_states()/step() by hand, not stream()
    with torch.no_grad():
        states = model.empty_states(prompt.size(0))
        for i in range(prompt.size(1) - 1):
            _, states = model.step(prompt[:, i:i + 1], states)
        ref = prompt
        curr = prompt[:, -1:]
        for _ in range(N):
            logits, states = model.step(curr, states)
            curr = logits.argmax(-1, keepdim=True)
            ref = torch.cat([ref, curr], dim=1)

    assert got.shape == (2, 7 + N), got.shape
    assert torch.equal(got, ref), (got != ref).nonzero()[:5]

    # max_new_tokens=None: the consumer stops iterating, not the generator --
    # the open-ended decode Free Think Mode runs on.
    partial = list(itertools.islice(model.stream(prompt, None, top_k=1), 5))
    assert len(partial) == 5 and all(t.shape == (2, 1) for t in partial), [t.shape for t in partial]
    assert torch.equal(torch.cat(partial, dim=1), ref[:, 7:12])

    # sink_tokens is a no-op here but must not raise -- free_think.py always passes one
    assert len(list(model.stream(prompt, 3, top_k=1, sink_tokens=4))) == 3

    print("[self-test] selective-linear stream: generate/step parity ok")


if __name__ == "__main__":
    self_test()
    self_test_selective()
    self_test_selective_stream()
