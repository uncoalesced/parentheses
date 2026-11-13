# Engineered by uncoalesced
"""Hugging Face model class for parentheses-1.0-240M (slab_v4).

Self-contained inference port of `models/model_slab_v4.py` and
`models/slab_v4_boundary.py` from https://github.com/uncoalesced/parentheses.
It needs only `torch` and `transformers`, so it loads with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("uncoalesced-org/parentheses-1.0-240M")
    model = AutoModelForCausalLM.from_pretrained(
        "uncoalesced-org/parentheses-1.0-240M", trust_remote_code=True
    )

parentheses is a recurrent selective linear attention model, not a transformer.
There are two ways to run it, and they compute the same function:

  * `forward(input_ids)` is the parallel form: a chunked scan over the whole
    sequence. `generate()` uses it and re-reads the sequence at every new token.
  * `step(token, states)` is the recurrent form: one token in, one fixed-size
    state update, constant memory at any length. `stream()` uses it and is the
    fast path for generation.

Prompt format the release was tuned on (no system prompt):

    <|user|><lang> question <|assistant|>

where `<lang>` is one of `<kan> <tam> <tel> <mal> <eng> <tcy> <kff> <kxv> <nit>`.
`build_prompt()` assembles it.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_parentheses import ParenthesesConfig

LayerState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Document-boundary primitives (from models/slab_v4_boundary.py)
# ---------------------------------------------------------------------------

def compute_doc_ids(input_ids: torch.Tensor, eot_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reset mask (True on the token after <|endoftext|>) and running document index."""
    B, T = input_ids.shape
    reset_mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
    if T > 1:
        reset_mask[:, 1:] = input_ids[:, :-1] == eot_token_id
    doc_ids = torch.cumsum(reset_mask.to(torch.int64), dim=1)
    return reset_mask, doc_ids


def token_shift(x: torch.Tensor, mu: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
    """Mix each position with the previous one, never across a document boundary."""
    B, T, D = x.shape
    x_prev = torch.cat([torch.zeros((B, 1, D), dtype=x.dtype, device=x.device), x[:, :-1, :]], dim=1)
    keep_prev = (~reset_mask).unsqueeze(-1).to(x.dtype)
    mu_view = mu.view(1, 1, D)
    return mu_view * x + (1.0 - mu_view) * (x_prev * keep_prev)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution (kernel 4); taps from another document are masked out."""
    B, T, D = x.shape
    K = weight.shape[-1]
    pad = K - 1
    x_padded = torch.cat([torch.zeros((B, pad, D), dtype=x.dtype, device=x.device), x], dim=1)
    d_padded = torch.cat([torch.zeros((B, pad), dtype=doc_ids.dtype, device=doc_ids.device), doc_ids], dim=1)
    x_windows = x_padded.unfold(dimension=1, size=K, step=1)          # [B, T, D, K]
    d_windows = d_padded.unfold(dimension=1, size=K, step=1)          # [B, T, K]
    d_curr = d_windows[:, :, -1:].expand_as(d_windows)
    tap_mask = (d_windows == d_curr).unsqueeze(2).to(x.dtype)         # [B, T, 1, K]
    w = weight.squeeze(1).view(1, 1, D, K)
    return (x_windows * tap_mask * w).sum(dim=-1) + bias.view(1, 1, D)


def ssd_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    reset_mask: torch.Tensor,
    doc_ids: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Parallel form of the recurrence S_t = alpha_t S_{t-1} + beta_t k_t v_t^T, y_t = q_t S_t.

    Positions inside a chunk interact through a masked decay matrix; chunks are
    linked by one float32 state hand-off. T must be a multiple of `chunk_size`.
    """
    B, T, H, Dh = q.shape
    C = chunk_size
    N = T // C
    orig_dtype = q.dtype
    q_c = q.float().view(B, N, C, H, Dh).permute(0, 1, 3, 2, 4)                       # [B, N, H, C, Dh]
    k_c = k.float().view(B, N, C, H, Dh).permute(0, 1, 3, 2, 4)
    v_c = (v * beta.unsqueeze(-1)).float().view(B, N, C, H, Dh).permute(0, 1, 3, 2, 4)
    log_a_c = log_alpha.float().view(B, N, C, H).permute(0, 1, 3, 2)                  # [B, N, H, C]
    d_c = doc_ids.view(B, N, C)
    m_c = reset_mask.view(B, N, C)

    P = torch.cumsum(log_a_c, dim=-1)                                                 # [B, N, H, C]

    P_diff = P.unsqueeze(-1) - P.unsqueeze(-2)                                        # [B, N, H, C, C]
    tril = torch.tril(torch.ones((C, C), dtype=torch.bool, device=q.device))
    same_doc = (d_c.unsqueeze(-1) == d_c.unsqueeze(-2)).unsqueeze(2)
    valid = tril.view(1, 1, 1, C, C) & same_doc
    L = torch.exp(P_diff.masked_fill(~valid, -1e30)) * valid.to(torch.float32)
    y_intra = torch.matmul(torch.matmul(q_c, k_c.transpose(-1, -2)) * L, v_c)         # [B, N, H, C, Dh]

    d_0 = d_c[:, :, 0]
    d_prev = d_0 - m_c[:, :, 0].to(d_c.dtype)
    d_last = d_c[:, :, -1]

    same_as_last = (d_c == d_last.unsqueeze(-1)).unsqueeze(2)
    w_out = torch.exp((P[:, :, :, -1:] - P).masked_fill(~same_as_last, -1e30)) * same_as_last.to(torch.float32)
    A = torch.exp(P[:, :, :, -1]) * (d_last == d_prev).unsqueeze(-1).to(torch.float32)  # [B, N, H]
    same_as_prev = (d_c == d_prev.unsqueeze(-1)).unsqueeze(2)
    w_in = torch.exp(P.masked_fill(~same_as_prev, -1e30)) * same_as_prev.to(torch.float32)

    U = torch.matmul((k_c * w_out.unsqueeze(-1)).transpose(-1, -2), v_c)              # [B, N, H, Dh, Dh]
    S = torch.zeros((B, H, Dh, Dh), dtype=torch.float32, device=q.device)
    y_inter = []
    for c in range(N):
        y_inter.append(torch.matmul(q_c[:, c] * w_in[:, c].unsqueeze(-1), S))
        S = A[:, c].view(B, H, 1, 1) * S + U[:, c]
    y = y_intra + torch.stack(y_inter, dim=1)
    return y.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, Dh).to(orig_dtype)


# ---------------------------------------------------------------------------
# Layers (from models/model_slab_v4.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, cfg: ParenthesesConfig):
        super().__init__()
        hidden = ((int(8 * cfg.n_embd / 3) + 63) // 64) * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SlabV4Attention(nn.Module):
    """Selective linear attention: token shift, causal conv, gated decay and write, V0 highway."""

    def __init__(self, cfg: ParenthesesConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.chunk_size = cfg.chunk_size
        self.beta_scale = 2.0 if getattr(cfg, "beta_mode", "2sig") == "2sig" else 1.0
        self.conv_kernel_size = 4

        self.mu = nn.Parameter(torch.full((self.n_embd,), 0.5))
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=cfg.bias)
        self.conv_weight = nn.Parameter(torch.zeros(3 * self.n_embd, 1, self.conv_kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(3 * self.n_embd))
        self.alpha_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
        self.beta_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
        if layer_idx > 0:
            self.v_residual_lamb = nn.Parameter(torch.zeros(self.n_head))
        self.y_norm = nn.Parameter(torch.ones(self.n_head, self.head_dim))
        self.gate = nn.Linear(self.n_embd, self.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=cfg.bias)

    def _qkv(self, u_conv: torch.Tensor, B: int, T: int, v0: Optional[torch.Tensor]):
        H, Dh = self.n_head, self.head_dim
        qkv = F.silu(u_conv).view(B, T, 3, H, Dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = F.normalize(q.float(), dim=-1).to(q.dtype) * (1.0 / math.sqrt(Dh))
        k = F.normalize(k.float(), dim=-1).to(k.dtype)
        v0_out = v if self.layer_idx == 0 else v0
        if self.layer_idx > 0 and v0 is not None:
            mix = torch.sigmoid(self.v_residual_lamb).view(1, 1, H, 1).to(v.dtype)
            v = mix * v + (1.0 - mix) * v0
        return q, k, v, v0_out

    def _out(self, y: torch.Tensor, x_ctrl: torch.Tensor, shape) -> torch.Tensor:
        y_float = y.float()
        y_normed = (y_float * torch.rsqrt(y_float.pow(2).mean(-1, keepdim=True) + 1e-5)).to(y.dtype)
        y = (y_normed * self.y_norm.view(1, 1, self.n_head, self.head_dim).to(y.dtype)).reshape(shape)
        return self.proj(y * F.silu(self.gate(x_ctrl)))

    def forward(self, x, reset_mask, doc_ids, v0=None):
        B, T, C = x.shape
        x_ctrl = token_shift(x, self.mu, reset_mask)
        u_conv = causal_conv1d(self.qkv(x), self.conv_weight, self.conv_bias, doc_ids)
        q, k, v, v0_out = self._qkv(u_conv, B, T, v0)
        beta = self.beta_scale * torch.sigmoid(self.beta_proj(x_ctrl))
        log_alpha = -F.softplus(self.alpha_proj(x_ctrl))
        y = ssd_chunk_scan(q, k, v, log_alpha, beta, reset_mask, doc_ids, self.chunk_size)
        return self._out(y, x_ctrl, (B, T, C)), v0_out

    def step(self, x, state: Optional[LayerState], is_boundary: bool, v0=None):
        B, C = x.shape
        H, Dh = self.n_head, self.head_dim
        pad = self.conv_kernel_size - 1
        if state is None:
            S = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=torch.float32)
            conv_state = torch.zeros(B, pad, 3 * C, device=x.device, dtype=x.dtype)
            token_state = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        else:
            S, conv_state, token_state = state
        if is_boundary:
            token_state = torch.zeros_like(token_state)
            conv_state = torch.zeros_like(conv_state)

        x_ctrl = self.mu.view(1, C) * x + (1.0 - self.mu.view(1, C)) * token_state.squeeze(1)
        window = torch.cat([conv_state, self.qkv(x).unsqueeze(1)], dim=1)
        u_conv = (window.permute(0, 2, 1) * self.conv_weight.squeeze(1).unsqueeze(0)).sum(dim=-1)
        u_conv = u_conv + self.conv_bias.unsqueeze(0)
        q, k, v, v0_out = self._qkv(u_conv, B, 1, v0)

        beta = (self.beta_scale * torch.sigmoid(self.beta_proj(x_ctrl))).float().view(B, H, 1, 1)
        alpha = torch.exp(-F.softplus(self.alpha_proj(x_ctrl)).float()).view(B, H, 1, 1)
        if is_boundary:
            alpha = torch.zeros_like(alpha)
        q_f, k_f, v_f = q[:, 0].float(), k[:, 0].float(), v[:, 0].float()
        S = alpha * S.float() + beta * torch.matmul(k_f.unsqueeze(-1), v_f.unsqueeze(-2))
        y = torch.matmul(q_f.unsqueeze(-2), S).squeeze(-2).to(x.dtype).unsqueeze(1)   # [B, 1, H, Dh]
        out = self._out(y, x_ctrl, (B, C))
        return out, (S, window[:, 1:, :], x.unsqueeze(1)), v0_out


class SlabV4Block(nn.Module):
    def __init__(self, cfg: ParenthesesConfig, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SlabV4Attention(cfg, layer_idx)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, reset_mask, doc_ids, v0=None):
        attn_out, v0 = self.attn(self.ln1(x), reset_mask, doc_ids, v0=v0)
        x = x + attn_out
        return x + self.mlp(self.ln2(x)), v0

    def step(self, x, state, is_boundary, v0=None):
        attn_out, state, v0 = self.attn.step(self.ln1(x), state, is_boundary, v0=v0)
        x = x + attn_out
        return x + self.mlp(self.ln2(x)), state, v0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ParenthesesPreTrainedModel(PreTrainedModel):
    config_class = ParenthesesConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)


class ParenthesesForCausalLM(ParenthesesPreTrainedModel, GenerationMixin):
    """parentheses slab_v4 causal language model.

    Module names (`tok_emb`, `blocks`, `ln_f`, `head`) match the training code,
    so the released `model.safetensors` loads without any key remapping.
    """

    _tied_weights_keys = {"tok_emb.weight": "head.weight"}

    def __init__(self, config: ParenthesesConfig):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([SlabV4Block(config, i) for i in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.head

    # -- parallel form ------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Logits for every position. `attention_mask` is accepted and ignored:
        the model is strictly causal, so right padding cannot affect earlier positions."""
        B, T = input_ids.shape
        C = self.config.chunk_size
        pad = (-T) % C
        ids = input_ids
        if pad:
            # The chunked scan needs a multiple of chunk_size. Padding on the right
            # is safe because nothing looks ahead; padded positions are dropped below.
            filler = torch.full((B, pad), self.config.pad_token_id or 1, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, filler], dim=1)

        reset_mask, doc_ids = compute_doc_ids(ids, self.config.eot_token_id)
        x = self.tok_emb(ids)
        v0 = None
        for block in self.blocks:
            x, v0 = block(x, reset_mask, doc_ids, v0=v0)
        logits = self.head(self.ln_f(x))[:, :T]

        loss = None
        if labels is not None:
            # Standard causal LM convention: position t predicts labels[t + 1].
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # No key-value cache exists. generate() re-reads the sequence each step;
        # use stream() for constant-memory recurrent decoding.
        return {"input_ids": input_ids}

    # -- recurrent form -----------------------------------------------------

    def empty_states(self, batch_size: int = 1) -> List[Optional[LayerState]]:
        """Fresh recurrent state. Layers allocate their tensors on the first step."""
        return [None for _ in self.blocks]

    def step(
        self,
        token_id: torch.LongTensor,
        states: Optional[List[Optional[LayerState]]] = None,
        is_boundary: bool = False,
    ) -> Tuple[torch.Tensor, List[LayerState]]:
        """Feed one token per batch row. Returns next-token logits [B, vocab] and the new state.

        Set `is_boundary=True` on the token that follows `<|endoftext|>` to clear the state.
        """
        if token_id.dim() == 2:
            token_id = token_id[:, 0]
        if states is None:
            states = self.empty_states(token_id.size(0))
        x = self.tok_emb(token_id)
        new_states, v0 = [], None
        for block, state in zip(self.blocks, states):
            x, state, v0 = block.step(x, state, is_boundary, v0=v0)
            new_states.append(state)
        return self.head(self.ln_f(x)), new_states

    @staticmethod
    def state_bytes(states: List[LayerState]) -> int:
        """Total size in bytes of a recurrent state returned by `step()`."""
        return sum(t.numel() * t.element_size() for layer in states if layer is not None for t in layer)

    @torch.no_grad()
    def stream(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_k: Optional[int] = 40,
        min_p: Optional[float] = 0.1,
        no_repeat_ngram_size: int = 8,
        stop_token_ids: Optional[Tuple[int, ...]] = None,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time with constant memory (batch size 1).

        Defaults are the settings the release is tested with: low temperature and
        an 8-gram loop block, no repetition penalty (a penalty stops the model
        from reusing a variable name in code). `temperature=0` is greedy.
        Stops at `<|endoftext|>` or a new `<|user|>` turn unless `stop_token_ids` is given.
        """
        assert input_ids.dim() == 2 and input_ids.size(0) == 1, "stream() takes one sequence: shape [1, T]"
        eot = self.config.eot_token_id
        stops = set(stop_token_ids) if stop_token_ids is not None else {eot, 13}  # 13 = <|user|>
        states = self.empty_states(1)
        prev_eot = False
        logits = None
        for i in range(input_ids.size(1)):
            tok = input_ids[:, i]
            logits, states = self.step(tok, states, is_boundary=prev_eot)
            prev_eot = int(tok) == eot

        history: List[int] = []
        for _ in range(max_new_tokens):
            scores = logits[0].float().clone()
            n = no_repeat_ngram_size
            if n >= 2 and len(history) >= n:
                prefix = history[-(n - 1):]
                for j in range(len(history) - n + 1):
                    if history[j:j + n - 1] == prefix:
                        scores[history[j + n - 1]] = -float("inf")
            if temperature <= 1e-4:
                next_id = int(torch.argmax(scores))
            else:
                scores = scores / temperature
                if top_k:
                    kth = torch.topk(scores, min(top_k, scores.size(-1))).values[-1]
                    scores[scores < kth] = -float("inf")
                probs = F.softmax(scores, dim=-1)
                if min_p:
                    probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
                    probs = probs / probs.sum().clamp_min(1e-9)
                next_id = int(torch.multinomial(probs, num_samples=1))
            if next_id in stops:
                return
            yield next_id
            history.append(next_id)
            if len(history) > 64:
                del history[:-64]
            tok = torch.tensor([next_id], dtype=torch.long, device=input_ids.device)
            logits, states = self.step(tok, states, is_boundary=prev_eot)
            prev_eot = next_id == eot

    @staticmethod
    def build_prompt(tokenizer, text: str, lang: str = "eng") -> torch.LongTensor:
        """Token ids for one user turn in the format the release was tuned on, shape [1, T]."""
        ids = (
            [tokenizer.convert_tokens_to_ids("<|user|>"), tokenizer.convert_tokens_to_ids(f"<{lang}>")]
            + tokenizer.encode(text, add_special_tokens=False)
            + [tokenizer.convert_tokens_to_ids("<|assistant|>")]
        )
        return torch.tensor([ids], dtype=torch.long)

    @torch.no_grad()
    def chat(self, tokenizer, text: str, lang: str = "eng", **stream_kwargs) -> str:
        """Answer one user turn with recurrent decoding and return the decoded text."""
        ids = self.build_prompt(tokenizer, text, lang).to(self.tok_emb.weight.device)
        return tokenizer.decode(list(self.stream(ids, **stream_kwargs)), skip_special_tokens=True)
