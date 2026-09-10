# Parentheses

A from-scratch, non-Transformer language model — currently RWKV/Mamba-2-style
selective linear attention — built and trained end-to-end on a single
consumer GPU (RTX 5050 laptop, 8GB VRAM).

Logo: `parentheses_mark.svg` (transparent variant:
`parentheses_mark_transparent.svg`), one level up from this repo.

**Status: active development, pre-benchmark.** There are no standardized
benchmark numbers yet — the architecture itself is still being locked down
(see "Architecture pivot" below) and a lot of the surrounding code (kernel
optimization, tokenizer, evaluation harness) is still being written. What's
below is accurate as of the latest commit, not aspirational.

## What this is

Parentheses is Plan One of a two-stage roadmap: prove a non-Transformer
architecture and two signature inference-time features (Free Think Mode,
Modular Free Think / RAG) at small scale, on hardware anyone can own, before
committing to Plan Two — a 1B-2B parameter model that needs rented/cloud
GPU compute regardless of how Plan One goes. Plan Two is a separate,
later, not-yet-started stage; nothing in this repo currently targets it.

Trained fully in English at this stage, with a Dravidian-language data pivot
and an eventual translation layer planned as later, separate work (see
`docs/translation-corpus-sourcing.md`) — not trainable at this parameter
budget yet, gated on Plan Two.

## Architecture pivot

Two attention implementations exist side by side, dispatched by
`attn_type` in `model/config.py`:

- **`causal`** — a standard GPT-style decoder-only transformer:
  `CausalSelfAttention` (fused QKV, `F.scaled_dot_product_attention`),
  RMSNorm, RoPE, SwiGLU MLP, tied input/output embeddings. Kept in the
  codebase as a comparison baseline, no longer a target for new work.
- **`selective_linear`** — `SelectiveLinearAttention`
  (`model/selective_linear_attention.py`), the only architecture getting
  further investment as of this stage. RWKV/Mamba-2-family selective linear
  attention rather than softmax attention. Currently several unfused ops
  (cumsum, mask build, clamp, exp, two matmuls) against causal attention's
  one fused kernel — measured ~34% slower per training step at the same
  model size; closing that gap is active, ongoing work (profiling first,
  see `OPTIMIZE_SELECTIVE_LINEAR_AGENT.md`).

Both share the same `generate()`/`stream()` decoding path, so every feature
below (Free Think Mode, Modular Free Think, conversation memory, `chat.py`)
works against either checkpoint unmodified.

### Tokenization

Byte-level (`vocab_size=256`, every UTF-8 byte is one token) for every
near-term preset, deliberately — not an oversight. At a sub-1M-parameter
budget, a normal 32,000-token subword vocabulary would make the embedding
table alone ~16.4M params, 16x the entire model. Cost: ~4x more tokens per
unit of text than word-level BPE, so `block_size` has to be proportionally
larger to see the same amount of text. The `parentheses-0.9-1m` stretch
preset is the only one with budget to spend on a trained tokenizer, and
even there it's a small 512-token BPE vocab, not a large one. A larger,
script-aware tokenizer designed for the Dravidian-language pivot is a Plan
Two concern — see `TOOLING.md`.

### KV-cached decoding

`Parentheses.stream()` is a KV-cached generator, one token at a time.
Measured on `parentheses-0.9-300k`: a 2.5x win on CPU (490 vs 192 tok/s),
and a consistent ~13% *loss* on the RTX 5050 (128 vs 147 tok/s uncached) —
at this model size, per-layer kernel-launch and Python overhead dominate on
GPU, not the attention math the cache removes. Kept on unconditionally
anyway: it's the right structure asymptotically and it's what makes CPU
decoding usable.

## Size presets

| preset | layers | heads | dim | vocab | real params |
|---|---|---|---|---|---|
| tiny-smoke | 2 | 2 | 32 | 64 | ~35K (CPU smoke test only) |
| parentheses-0.9-100k | 4 | 4 | 48 | 256 (byte-level) | ~123K |
| **parentheses-0.9-300k** | 5 | 4 | 72 | 256 (byte-level) | ~330K |
| parentheses-0.9-600k | 5 | 4 | 96 | 320 | ~585K |
| parentheses-0.9-1m (stretch) | 5 | 8 | 128 | 512 (trained BPE) | ~1.13M |
| parentheses-0.9-50m/150m/350m | — | — | — | 32000 | Plan-Two-adjacent, not a near-term target — kept for reference |

`parentheses-0.9-300k` is the current recommended main target: a full
compute-optimal run in ~5.5 minutes and a full pass over the current corpus
in ~2.25 hours on a weak 2-core CPU baseline (a real laptop should do at
least as well). Start with `-100k` to validate the pipeline faster, then
push toward `-1m` once that's working end to end.

`--max-steps` for one compute-optimal pass (batch size 64 default, see
`docs/training-time-estimate.md` for the token-budget reasoning): 100k →
~150, 300k → ~400, 600k → ~475, 1m → ~690. `train.py` defaults to 100,000
regardless of preset, which is far past compute-optimal for these presets —
pick a real number on purpose rather than relying on the default.

## Layout

```
model/            selective_linear + causal-baseline attention, RMSNorm, RoPE, SwiGLU, tied embeddings, size presets
train.py          Training loop: AMP, grad accumulation, optional 8-bit optimizer
data/             Corpus pipeline (Wikipedia, books, OPUS-parallel, Dravidian sourcing) + tokenizer — see data/README.md
features/         Free Think Mode + Modular Free Think (RAG, BM25) + conversation memory
scripts/          chat.py, benchmark_step.py, benchmark_retrieval.py, check_docs.py, and others
docs/             training-time-estimate.md, translation-corpus-sourcing.md + -download.md
TOOLING.md        Target-state MLOps/tooling spec and Plan One / Plan Two build-order gating
```

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

Replace `checkpoints/<run-name>/<checkpoint-file>` with an actual checkpoint
path under `checkpoints/` (e.g. a step file under `checkpoints/selective-v1/`)
— left generic here rather than naming one, since which checkpoint is
"current" changes as training continues.

Self-tests (assert-based, no framework, no data or network needed):

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

`features/free_think.py`. Given a *statement* (it refuses questions unless
`--force`), the model streams open-ended "thinking" about it until stopped;
`--export foo.json`/`foo.txt` dumps the session. Decoding-time only, no
architecture change — runs on `Parentheses.stream()`. `--max-tokens 0`
streams forever; a `sink_tokens` attention sink (StreamingLLM-style) pins
the first N prompt tokens into every post-window-reset context so a long
run doesn't forget its own opening. Partial fix, not true long-range
memory — real thread-following across thousands of tokens needs far more
capacity than a few-hundred-K to 1M-param model has. At this scale, expect
locally coherent drift anchored to the opening topic.

## Modular Free Think (RAG)

`features/modular_free_think.py`. Free Think Mode's decode loop plus a
retrieval hook: point it at your own text files and the "thinking" gets
grounded in them. Retrieval defaults to lexical BM25 (`rank_bm25`) and that
remains the recommended backend.

**Vector retrieval exists and is measured, and is deliberately not the
default.** A TurboVec index over a small trained pooling head
(`model/embedding_head.py`) matches BM25 on throughput (215.0 vs 214.4
tok/s modular vs 221.4 tok/s plain, RTX 5050, 3,334-chunk store) but loses
badly on quality for the actual Modular Free Think use case — monolingual
English paragraph retrieval. Cross-lingual sentence-pair recall is real
(~40x chance across 22 languages), but that doesn't transfer to
monolingual chunks at this backbone size; a 330K-param byte-level backbone
with a 64-d head is not a sentence-embedding model, and unfreezing the
backbone's top block made the monolingual side-by-side comparison worse by
teaching the embedding to sort on text length instead of meaning. No
embedding model ships by default — `backend="vector"` requires the caller
to pass an `embedder` and raises without one.

## Conversation memory

`features/conversation_memory.py`. A second trigger
(`warrants_free_think()`) fires on long, first-person, non-question turns
that `is_question()` alone would miss. `ConversationNotes` appends every
turn to a Markdown file under `conversations/`; each triggered turn
re-ingests that file through the Modular Free Think retrieval path, so
earlier turns come back as retrieved chunks rather than context the
256-byte window can't hold.

## Benchmarks

**None yet.** The numbers in this README (decode throughput, retrieval
latency, KV-cache speedup) are engineering/infrastructure measurements —
they say how fast the current code runs, not how good the model is.
Standardized model-quality evaluation (perplexity/loss comparisons against
reference architectures, task benchmarks, or the planned morphological/
akshara-level evaluation harness for the Dravidian pivot — see
`TOOLING.md`) doesn't exist yet, for two concrete reasons: the
architecture itself is mid-pivot (causal → selective_linear, with an open
~34% training-speed gap still being closed), and there isn't yet a
checkpoint at a quality worth benchmarking — current output across
architectures is script-correct and grammatically plausible but not
semantically coherent, consistent with the small parameter budgets in the
size-preset table above. This section gets filled in once both of those
are further along.

## Roadmap

- **Plan One (here, now):** prove `selective_linear` at small scale, close
  its training-speed gap against the causal baseline, keep Free Think Mode
  and Modular Free Think working across both architectures.
- **Plan Two (later, separate, not started):** 1B-2B parameters
  ("Parentheses 1.0"), rented/cloud multi-GPU compute, the
  Dravidian-then-broader-Indic data pivot at scale, a script-aware
  morphological tokenizer, and the full tooling stack in `TOOLING.md`
  (FSDP, Kubeflow, DVC, serving, monitoring). Whether/how to move from
  Plan One to Plan Two is a decision made after Plan One's results are in,
  not a default.

## Development

See `ARCHITECTURE.md` for the technical deep-dive, `AGENT_WORKFLOW.md` and
`CODING_STANDARDS.md` for contribution/agent-task conventions, and
`documentation.md` for the append-only session-by-session project journal.

## License

Not yet chosen. This needs an explicit decision before this repo goes
public — don't assume a default.
