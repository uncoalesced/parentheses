# Engineered by uncoalesced
"""Hugging Face configuration for parentheses-1.0-240M (slab_v4)."""

from transformers import PretrainedConfig


class ParenthesesConfig(PretrainedConfig):
    """Configuration for the parentheses slab_v4 recurrent language model.

    Field names match `SlabV4Config` in the parentheses repository
    (`models/model_slab_v4.py`) so one `config.json` serves both loaders.

    Fields are declared as annotated class attributes with defaults. On
    transformers 5 they become dataclass fields of the config; on transformers 4
    they act as class-level defaults that values from `config.json` override.
    """

    model_type = "parentheses_slab_v4"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "max_position_embeddings": "block_size",
    }

    vocab_size: int = 32768
    block_size: int = 4096
    n_layer: int = 16
    n_head: int = 16
    n_embd: int = 1024
    chunk_size: int = 64
    eot_token_id: int = 0
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    beta_mode: str = "2sig"
    tie_word_embeddings: bool = True
    eos_token_id: int | None = 0
    pad_token_id: int | None = 1
    bos_token_id: int | None = None
    use_cache: bool = False

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
