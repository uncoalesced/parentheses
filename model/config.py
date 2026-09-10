"""Model configuration for Parentheses 0.9."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256        # byte-level by default -- see note below on why
    block_size: int = 256        # max sequence length (context)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 48
    dropout: float = 0.0
    bias: bool = False           # False = slightly faster, matches modern practice (LLaMA-style)
    tie_embeddings: bool = True  # share input/output embedding matrix -- see note below
    attn_type: str = "causal"    # "causal" (CausalSelfAttention) | "selective_linear"
    #                              (model/selective_linear_attention.py). Added last, with a
    #                              default, so a ModelConfig unpickled from a pre-pivot
    #                              checkpoint (no attn_type in its __dict__) still reads
    #                              "causal" off the class attribute and compares equal to
    #                              PRESETS["parentheses-0.9-300k"] -- train.py --resume-from
    #                              and features/free_think.py both depend on that equality.

    @property
    def approx_params(self) -> int:
        """Rough parameter count for sanity-checking a config before training.

        Known gap: undercounts attn_type="selective_linear" by the decay gate's
        alpha_proj (n_embd * n_head + n_head per layer, ~1,460 total at the
        -300k dims, ~0.1% of the model). Left as an approximation on purpose --
        this is a pre-flight sanity number, not an accounting of the real
        parameter count, which Parentheses.num_params() reports exactly.
        """
        embd = self.vocab_size * self.n_embd * (1 if self.tie_embeddings else 2)
        per_layer = 12 * self.n_embd ** 2          # attention + MLP (4x expansion) rule of thumb
        return embd + self.n_layer * per_layer


# Why vocab_size=256 and tie_embeddings=True by default:
# at a few-hundred-thousand-parameter budget, a normal subword tokenizer
# (e.g. 32000-vocab BPE, what most LLMs use) would make the embedding table
# ALONE bigger than the entire model. Untied, vocab=32000 x n_embd=256 x 2
# = 16.4M params just for embeddings -- 16x over a 1M-param budget, before a
# single transformer layer. Byte-level tokenization (vocab_size=256, every
# UTF-8 byte is a token, no tokenizer training needed at all) keeps the
# embedding table small enough that most of the param budget still goes to
# the transformer layers doing the actual work. Cost: sequences get longer
# per unit of text (~4x more tokens than word-level BPE), so context
# (block_size) has to stretch further to see the same amount of text. As
# the model grows toward the "1M+" stretch tier, a small trained BPE vocab
# (a few hundred to ~1000 merges) becomes worth revisiting -- see
# data/tokenizer/train_tokenizer.py.

# Named presets referenced by train.py's --preset flag and docs/training-time-estimate.md.
# (No separate configs/*.yaml -- presets live here as the single source of truth.)
PRESETS = {
    "tiny-smoke": ModelConfig(vocab_size=64, block_size=64, n_layer=2, n_head=2, n_embd=32),

    # Current target range: a few hundred thousand params, byte-level vocab.
    "parentheses-0.9-100k": ModelConfig(vocab_size=256, block_size=256, n_layer=4, n_head=4, n_embd=48),
    "parentheses-0.9-300k": ModelConfig(vocab_size=256, block_size=256, n_layer=5, n_head=4, n_embd=72),
    "parentheses-0.9-600k": ModelConfig(vocab_size=320, block_size=384, n_layer=5, n_head=4, n_embd=96),

    # Architecture-pivot arm (documentation.md 2026-09-05): byte-for-byte the
    # same dims as -300k, only the attention swapped, so a run against the same
    # data/processed_all_sources_v1 split is a controlled comparison against
    # all-sources-v1's val loss 1.4784. -300k itself is deliberately unchanged.
    "parentheses-0.9-300k-selective": ModelConfig(vocab_size=256, block_size=256, n_layer=5, n_head=4, n_embd=72,
                                                  attn_type="selective_linear"),

    # Stretch tier ("over time... a million or more"): small trained BPE
    # vocab instead of raw bytes, since there's finally enough budget for it.
    "parentheses-0.9-1m": ModelConfig(vocab_size=512, block_size=512, n_layer=5, n_head=8, n_embd=128),

    # Long-term / Plan Two territory -- kept for reference, not a near-term
    # target on the RTX 5050 laptop. See docs/training-time-estimate.md.
    "parentheses-0.9-50m": ModelConfig(vocab_size=32000, block_size=1024, n_layer=8, n_head=8, n_embd=512, tie_embeddings=False),
    "parentheses-0.9-150m": ModelConfig(vocab_size=32000, block_size=1024, n_layer=12, n_head=12, n_embd=768, tie_embeddings=False),
    "parentheses-0.9-350m": ModelConfig(vocab_size=32000, block_size=1024, n_layer=24, n_head=16, n_embd=1024, tie_embeddings=False),
}


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        n = cfg.approx_params
        label = f"{n/1e6:7.2f}M" if n >= 1e6 else f"{n/1e3:7.1f}K"
        print(f"{name:<26} ~{label} params  (vocab={cfg.vocab_size}, layer={cfg.n_layer}, embd={cfg.n_embd}, tied={cfg.tie_embeddings})")
