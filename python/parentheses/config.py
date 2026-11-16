"""Model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlabConfig:
    """Shape of a slab_v4 model. Defaults are the released 1.0 model."""

    vocab_size: int = 32768
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 1024
    chunk_size: int = 64       # block length of the parallel scan
    block_size: int = 4096     # longest sequence the model was trained on
    eot_token_id: int = 0      # <|endoftext|>: the token after it starts a new document
    pad_token_id: int = 1
    beta_scale: float = 2.0    # write gate is beta_scale * sigmoid(...)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden(self) -> int:
        return ((int(8 * self.n_embd / 3) + 63) // 64) * 64

    @classmethod
    def from_state_dict(cls, state: dict, **overrides) -> "SlabConfig":
        """Read the shape off a checkpoint's tensors."""
        vocab_size, n_embd = state["tok_emb.weight"].shape
        layers = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
        n_head = state["blocks.0.attn.alpha_proj.weight"].shape[0]
        return cls(vocab_size=vocab_size, n_layer=max(layers) + 1, n_head=n_head,
                   n_embd=n_embd, **overrides)


SLAB_V4_240M = SlabConfig()
