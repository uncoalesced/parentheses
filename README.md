██████╗  █████╗ ██████╗ ███████╗ ███╗   ██╗ ████████╗ ██╗  ██╗ ███████╗ ███████╗ ███████╗ ███████╗
██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║  ██║ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
██████╔╝ ███████║ █████╔╝ █████╗ ██╔██╗ ██║   ██║ ███████║ █████╗ ███████╗ █████╗ ███████╗
██╔═══╝ ██╔══██║ ██╔══██╗ ██╔══╝ ██║╚██╗██║   ██║ ██╔══██║ ██╔══╝ ╚════██║ ██╔══╝ ╚════██║
██║ ██║  ██║ ██║  ██║ ███████╗ ██║ ╚████║   ██║ ██║  ██║ ███████╗ ███████║ ███████╗ ███████║
╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝  ╚═══╝   ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚══════╝ ╚══════╝


# Parentheses

A non-Transformer language model implementing selective linear attention (similar to RWKV and Mamba-2), built and trained from scratch on a single consumer laptop GPU (RTX 5050, 8GB VRAM).

Logo: `parentheses_mark.svg` (transparent variant: `parentheses_mark_transparent.svg`), located one directory above this repository.

### Status

Active development, pre-benchmark. There are no standardized benchmark results yet because the core architecture is still changing (detailed in the Architecture pivot section below) and surrounding components (kernel optimizations, tokenization, evaluation harnesses) remain in progress. The documentation reflects the repository as of the latest commit.

## Overview

Parentheses tests a selective linear attention architecture along with two inference features, Free Think Mode and Modular Free Think (RAG), at small parameter scales on consumer hardware. The current codebase focuses on small models that run locally. Larger 1B to 2B parameter configurations requiring cloud GPU compute are planned for later stages and are not implemented here.

Training currently uses English text. Expanding to Dravidian languages and adding a translation layer (detailed in `docs/translation-corpus-sourcing.md`) requires larger model capacities and is reserved for future work.

## Architecture pivot

Two attention implementations exist side by side, controlled by `attn_type` in `model/config.py`:

- `causal`: A standard GPT-style decoder-only transformer using `CausalSelfAttention` (fused QKV, `F.scaled_dot_product_attention`), RMSNorm, RoPE, SwiGLU MLP, and tied input/output embeddings. This implementation serves as a comparison baseline and receives no new development.
- `selective_linear`: Implemented as `SelectiveLinearAttention` in `model/selective_linear_attention.py`, this is the primary architecture under active development. It uses selective linear attention from the RWKV and Mamba-2 families instead of softmax attention. Because it currently relies on several unfused operations (cumsum, mask build, clamp, exp, and two matmuls) compared to the single fused kernel in causal attention, training steps are roughly 34% slower at identical model sizes. Closing this gap is in progress (see `OPTIMIZE_SELECTIVE_LINEAR_AGENT.md` for profiling details).

Both architectures share the same `generate()` and `stream()` decoding paths, so Free Think Mode, Modular Free Think, conversation memory, and `chat.py` work across checkpoints for either type without modification.

### Tokenization

All near-term presets use byte-level tokenization (`vocab_size=256`, where each UTF-8 byte represents one token). For models under 1 million parameters, a standard 32,000-token subword vocabulary would require roughly 16.4 million parameters for the embedding table alone, which is 16 times the size of the entire model. The trade-off is higher sequence length: byte-level encoding uses approximately 4 times as many tokens per unit of text as word-level BPE, requiring a proportionally larger `block_size` to cover equivalent text spans. The `parentheses-0.9-1m` preset is the only small configuration that includes a trained tokenizer, which uses a compact 512-token BPE vocabulary. Larger, script-aware tokenizers for the Dravidian language expansion are deferred to future scaling work (see `TOOLING.md`).

### KV-cached decoding

`Parentheses.stream()` implements token-by-token generation with a KV cache. On the `parentheses-0.9-300k` preset, caching increases CPU decoding throughput by roughly 2.5x (490 versus 192 tokens/second). On the RTX 5050 GPU, however, it yields a consistent 13% throughput drop (128 versus 147 tokens/second uncached). At this scale, per-layer kernel launch times and Python interpreter overhead outweigh the compute savings of caching attention states. The cache is retained by default because it provides proper asymptotic scaling and ensures usable CPU inference speeds.

## Size presets

| Preset | Layers | Heads | Dim | Vocab | Parameters | Notes |
|---|---|---|---|---|---|---|
| tiny-smoke | 2 | 2 | 32 | 64 | ~35K | CPU smoke test only |
| parentheses-0.5-100k | 4 | 4 | 48 | 256 (byte-level) | ~123K | Pipeline validation |
| parentheses-0.6-300k | 5 | 4 | 72 | 256 (byte-level) | ~330K | Recommended target |
| parentheses-0.7-600k | 5 | 4 | 96 | 320 | ~585K | Intermediate scale |
| parentheses-0.8-1m (stretch) | 5 | 8 | 128 | 512 (trained BPE) | ~1.13M | BPE test preset |
| parentheses-0.9-50m/150m/350m | - | - | - | 32000 | Reference configs | Future scaling targets |

The `parentheses-0.9-300k` configuration serves as the primary development target. On a 2-core CPU baseline, a compute-optimal run completes in roughly 5.5 minutes, and a full pass through the current training corpus takes about 2.25 hours. For quicker testing, run `-100k` to confirm the pipeline before moving up to `-1m`.

For compute-optimal training with the default batch size of 64, set `--max-steps` based on the preset:
- 100k: ~150 steps
- 300k: ~400 steps
- 600k: ~475 steps
- 1m: ~690 steps

The script `train.py` defaults to 100,000 steps, which exceeds compute-optimal bounds for these small models. Set `--max-steps` manually during execution (refer to `docs/training-time-estimate.md` for token budget calculations).

## Repository layout

model/            selective_linear and causal baseline attention, RMSNorm, RoPE, SwiGLU, tied embeddings, size presets
train.py          Training loop supporting AMP, gradient accumulation, and optional 8-bit optimization
data/             Corpus pipeline (Wikipedia, books, OPUS-parallel, Dravidian sourcing) and tokenizer (see data/README.md)
features/         Free Think Mode, Modular Free Think (RAG with BM25), and conversation memory
scripts/          Utility scripts including chat.py, benchmark_step.py, benchmark_retrieval.py, and check_docs.py
docs/             Documentation including training-time-estimate.md and translation-corpus-sourcing.md
TOOLING.md        Target-state MLOps specifications and tooling roadmap


## Quickstart

```bash
pip install -r requirements.txt

# 1. pull + clean data
python3 data/prepare_wikipedia.py   # defaults to Simple English Wikipedia
python3 data/prepare_books.py
# (skip tokenizer training for 100k/300k/600k -- byte-level needs none)

# 2. tokenize into train.py's expected uint16 .bin format
python3 data/tokenize_corpus.py

# 3. sanity-check model size for a preset
python3 model/config.py

# 4. train (--max-steps: train.py defaults to 100_000 regardless of preset --
#    that's way past compute-optimal for these tiny presets, see above)
python3 train.py --preset parentheses-0.9-300k --attn-type selective_linear \
    --data data/processed/train.bin --max-steps 400

# 5. interactive chat against a trained checkpoint
python3 scripts/chat.py --checkpoint checkpoints/<run-name>/<checkpoint-file>

# 6. think out loud about a statement (Free Think Mode)
python3 -m features.free_think --checkpoint checkpoints/<run-name>/<checkpoint-file> \
    --prompt "The ocean is very deep." --max-tokens 400

# 7. same, but grounded in your own files (Modular Free Think / RAG)
python3 -m features.modular_free_think --checkpoint checkpoints/<run-name>/<checkpoint-file> \
    --ingest notes/*.txt --prompt "The ocean is very deep." --max-tokens 400

# 8. a conversation that free-thinks when it should and remembers the thread
python3 -m features.conversation_memory --checkpoint checkpoints/<run-name>/<checkpoint-file>
```

Replace `checkpoints/<run-name>/<checkpoint-file>` with your actual checkpoint path under `checkpoints/` (such as a step checkpoint under `checkpoints/selective-v1/`).

Run the standalone self-tests to verify parity and baseline functionality without external dependencies:

```bash
python3 -m model.transformer                 # causal baseline: KV-cached decoding == uncached decoding
python3 -m model.selective_linear_attention  # selective_linear: dual/recurrent parity
python3 -m features.free_think --self-test
python3 -m features.modular_free_think --self-test
python3 -m features.conversation_memory --self-test
python3 data/prepare_parallel.py --self-test
python3 scripts/check_docs.py                # docs don't name code that no longer exists
```

## Free Think Mode
Located in features/free_think.py. Given an input statement (questions are rejected unless --force is set), the model streams continuous text reflection until stopped. Results can be saved using --export output.json or --export output.txt. This is an inference-only feature running directly on Parentheses.stream(). Setting --max-tokens 0 allows indefinite streaming. An attention sink (sink_tokens, similar to StreamingLLM) keeps the first N prompt tokens in context across window resets to maintain topic anchor points. Because models at this sub-1M parameter scale have limited capacity, outputs will gradually drift over long generations while remaining locally related to the initial prompt.

## Modular Free Think (RAG)
Located in features/modular_free_think.py. This extends Free Think Mode with retrieval, pulling context from local text files to ground the generated tokens. Lexical BM25 (rank_bm25) is the default and recommended retrieval backend.

An optional vector retrieval backend using TurboVec and a small trained pooling head (model/embedding_head.py) matches BM25 throughput (215.0 tokens/second modular versus 214.4 for BM25 and 221.4 plain on an RTX 5050 with 3,334 indexed chunks). However, it produces substantially lower retrieval quality for monolingual English paragraphs. Although the head shows strong sentence-pair recall across multilingual benchmarks (~40x random chance across 22 languages), that capability does not carry over to monolingual chunk retrieval at this parameter scale. A 330K-parameter byte-level model with a 64-dimensional head does not function well as a general sentence embedder; fine-tuning the top backbone block caused the embeddings to cluster by string length rather than semantic content. Consequently, no default embedding model is bundled, and selecting backend="vector" requires passing an explicit embedder instance.

## Conversation memory
Implemented in features/conversation_memory.py. The module adds a second trigger, warrants_free_think(), which detects extended, first-person statements that is_question() would overlook. ConversationNotes writes each turn to a Markdown file in conversations/. When triggered, previous turns are re-indexed through the Modular Free Think pipeline, retrieving relevant conversation history into context despite the small 256-byte model context window.

## Benchmarks
Standardized quality benchmarks are not yet available. Throughput, retrieval latency, and KV-cache metrics reported elsewhere in this document reflect system execution speeds rather than language modeling performance. Formal quality evaluations (such as perplexity benchmarks against baseline architectures, standard task suites, or morphological evaluation for the upcoming Dravidian dataset described in TOOLING.md) are pending two milestones:

Finalizing the transition from causal attention to selective linear attention and closing the 34% training throughput gap.

Training checkpoints with sufficient semantic coherence. At current sub-1M parameter sizes, outputs are grammatically structured and valid UTF-8, but lack semantic depth.

Comparative benchmarks will be added once trained weights reach viable quality thresholds.

## Roadmap
Current focus: Evaluate selective linear attention at small parameter budgets, eliminate the training-throughput gap relative to the causal transformer baseline, and maintain functional parity for Free Think Mode and Modular Free Think across both attention implementations.

Long-term goals: Scale to 1B to 2B parameters on multi-GPU cloud hardware, introduce Dravidian and broader Indic language corpora, build a dedicated morphological tokenizer, and set up the distributed training and deployment infrastructure outlined in TOOLING.md (including FSDP, Kubeflow, DVC, and model serving). Transitioning to larger scale depends on performance outcomes from the current small-scale experiments.

## License
This project is licensed under the MIT License.