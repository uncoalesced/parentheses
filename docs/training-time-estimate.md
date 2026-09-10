---
title: Parentheses 0.9 — Training Time Estimate
date: 2026-08-19 (revised same day — target scale changed)
status: revised estimate for the ~100K-1M parameter tier; original 50M-350M analysis kept below for future reference
---

## Revision note

Joel rescoped Plan One from "a few hundred million params" down to **a few
hundred thousand, with a stretch goal of ~1M+ over time**. That's a big
enough change that the method from the first pass (pure FLOPs vs. peak
GPU TFLOPS) stops being useful — see "Why the method changed" below — so
this is a fresh estimate, not just new numbers in the old formula. The
original 50M-350M analysis is kept at the bottom in case the project scales
back up later.

## Headline numbers

**Training compute is no longer the bottleneck at this scale — corpus size
is.** On a weak 2-core cloud CPU (much less capable than the RTX 5050
laptop or its CPU), a full training run at the Chinchilla-optimal token
budget takes **1 to 45 minutes**, depending on preset. What actually takes
real time now is reading through your data source, and "the whole of
Wikipedia" is 100-1000x more data than a model this size can usefully
absorb in one pass.

| Preset | Real params | Chinchilla-optimal tokens (20×N) | Measured time (this budget) |
|---|---|---|---|
| parentheses-0.9-100k | 123,312 | ~2.47M | ~68 sec |
| parentheses-0.9-300k | 330,264 | ~6.6M | ~5.5 min |
| parentheses-0.9-600k | 584,736 | ~11.7M | ~13.7 min |
| parentheses-0.9-1m | 1,131,904 | ~22.6M | ~44.3 min |

These are measured, not estimated from a formula — see "Why the method
changed" for why that matters here. Numbers came from
`scripts/benchmark_step.py` run on this session's 2-core Xeon cloud sandbox
(see caveats). Your RTX 5050 laptop, and even its own CPU, should match or
beat these; nothing here suggests you need to wait around.

## Why the method changed

The first estimate (50M-350M tier) used `time ≈ (6 × params × tokens) /
(peak GPU FLOPs × utilization)` — appropriate when a training step is big
enough to actually saturate the GPU's matrix units. At a few hundred
thousand parameters, a training step is nowhere near that size: wall-clock
time is dominated by Python loop overhead, kernel dispatch, and the
optimizer step, not raw FLOPs. Plugging these param counts into the old
formula would predict sub-millisecond steps, which isn't a real number —
so this time I benchmarked actual forward+backward+optimizer steps
(`scripts/benchmark_step.py`) instead of computing a theoretical one.

One finding from that benchmark worth noting: on the 2-core CPU used here,
tokens/sec stayed roughly flat as batch size increased (16 → 32 → 128),
meaning even this weak CPU is compute-bound, not overhead-bound, on the
300k preset. That's a good sign — it means the measured numbers are a real
throughput floor, not an artifact of small-batch launch overhead that a
GPU would trivially fix.

## VRAM: no longer a real constraint

The previous estimate spent a full section on 8GB VRAM budgeting because
it mattered at 50M-350M params. At 100K-1M params, model + gradients +
Adam optimizer state (even untied, unoptimized) is a few MB, total. The
RTX 5050's 8GB is enormously overkill for this tier — VRAM stops being a
design constraint entirely. (It'll matter again if the "over time" growth
path goes well past the stretch goal.)

## The real constraint: corpus size vs. model capacity

Full English Wikipedia is about **16GB of cleaned article text** (~7B
subword tokens per a Feb 2025 Wikipedia dump measurement, which is roughly
16 billion tokens under the byte-level tokenization these presets use by
default — see the vocab-size note below). At the measured throughputs, a
**single pass over all of English Wikipedia** would take:

| Preset | Time for one full-Wikipedia pass |
|---|---|
| 100k | ~5.1 days |
| 300k | ~9.3 days |
| 600k | ~13.1 days |
| 1m | ~21.7 days |

That's not a compute problem, it's a mismatch: a 100K-1M parameter model
cannot meaningfully absorb billions of tokens of Wikipedia — there isn't
enough capacity to compress that much information, so most of a full pass
would just be wasted read-through. Chinchilla-optimal for this tier is
2.5-23M tokens, roughly 0.03-0.3% of full Wikipedia's size.

**Recommendation: use Simple English Wikipedia instead of the full dump**
for this tier ([pszemraj/simple_wikipedia](https://huggingface.co/datasets/pszemraj/simple_wikipedia)
on Hugging Face — 226K articles, ~65M subword tokens, 161MB text, roughly
161M tokens under byte-level tokenization). It's sized almost perfectly
for this parameter range: comfortably larger than the compute-optimal
budget for every preset here (so no need to worry about running out of
data or over-repeating it), but small enough that **one full pass takes
1-5 hours**, not days:

| Preset | Time for one full Simple-Wikipedia pass |
|---|---|
| 100k | ~1.2 hours |
| 300k | ~2.25 hours |
| 600k | ~3.15 hours |
| 1m | ~5.25 hours |

This leaves room to layer in a handful of short public-domain books (the
Bible, a few Gutenberg classics) without blowing the time budget, while
keeping data volume roughly matched to what the model can actually learn
from. `data/prepare_wikipedia.py` now defaults to Simple English Wikipedia
for this reason — full enwiki is still an option to switch to later if the
model scales up past the stretch tier.

## Vocab size: byte-level by default, and why that's now required

At 100K-1M total params, a standard subword tokenizer (32,000-vocab BPE,
what most LLMs use, including the original 50M-350M plan) would make the
**embedding table alone** bigger than the entire model: even tied,
32,000 × 256 (a modest embedding width) = 8.2M params — 8x over the 1M
stretch budget before a single transformer layer exists. So these presets
default to **byte-level tokenization** (`vocab_size=256`, every UTF-8 byte
is a token, no tokenizer training step needed at all) plus **tied
input/output embeddings** (`tie_embeddings=True`, now implemented in
`model/transformer.py`). Together those keep the embedding table small
enough that most of the param budget goes to the transformer layers doing
the actual work, not to a lookup table.

Trade-off: byte-level tokens mean ~4x more tokens per unit of English text
than word-level BPE, which is why the "tokens" and "time" figures above
are larger than a first intuition might suggest for this much text. As the
model grows past the 1M-param stretch tier, revisit a small trained BPE
vocab (a few hundred to ~1000 merges) — `data/tokenizer/train_tokenizer.py`
is already wired for that when it's worth it.

## Honest calibration on what this tier can actually learn

Worth saying plainly: a 100K-1M parameter model is not going to "know
Wikipedia" in any broad sense, even Simple Wikipedia. Models at this scale
(think: smaller than early-2010s n-gram-era language models in effective
capacity) are good for validating that the pipeline, architecture, and the
Free Think / Modular Free Think mechanics actually work end-to-end — not for
general-knowledge coverage. That's a fine and sensible goal for a first
build, and it lines up with the "if we're lucky, scale to a million or
more over time" framing — just flagging so the first trained checkpoint's
output quality doesn't read as a bug when it's really a capacity ceiling.
Meaningful broad-knowledge coverage the way the original "whole of
Wikipedia" framing implied would need several more orders of magnitude of
parameters (tens to low hundreds of millions, back in the original Plan
One range) — which is exactly why proving the small version first is a
reasonable way to de-risk before committing to that.

## Caveats

- Benchmarked on a 2-core cloud Xeon @ 2.8GHz — a weak, non-representative
  stand-in for the RTX 5050 laptop (or even that laptop's own CPU, which
  almost certainly has more cores). Treat these numbers as a credible
  upper bound on training time, not a lower bound — real hardware should
  be at or faster than this.
- "Time" here is training compute only, same caveat as the original
  estimate: data loading stalls, checkpointing, eval passes, and repeated
  experiments in practice add real wall-clock time on top.
- Wikipedia/Simple-Wikipedia token counts are order-of-magnitude estimates
  cross-checked across two sources (see Sources below), not an exact count
  of whatever specific dump/snapshot you end up using.

Sources:
- [Clean-Wikipedia-English-Articles dataset card (7B tokens, 16.1GB, Feb 2025 dump) — Hugging Face](https://huggingface.co/datasets/OVHaiLLM/Clean-Wikipedia-English-Articles)
- [English Wikipedia word count context — Dan Taylor Watt](https://dantaylorwatt.substack.com/p/how-much-text-are-large-language)
- [Simple English Wikipedia dataset card (65M tokens, 161MB) — pszemraj/simple_wikipedia on Hugging Face](https://huggingface.co/datasets/pszemraj/simple_wikipedia)

---

## Appendix: original 50M-350M estimate (superseded, kept for reference)

Kept in case the project scales back up toward this range later in the
"over time" growth path. This section used the peak-FLOPs method, which
is valid at this larger scale (unlike the 100K-1M tier above).

### Your primary question: how long will this take?

For a model in the tens-to-low-hundreds-of-millions-of-parameter
range, expect roughly half a day to a few weeks of pure training time on
the RTX 5050 laptop, depending on final size and how optimized the
training code is. Anything approaching 700M-1B parameters stretches into
3-8+ months of wall-clock training on this GPU alone, and is genuinely
tight on 8GB VRAM. This is training time only.

### Hardware: RTX 5050 Laptop (8GB)

- 2,560 CUDA cores, ~2.5 GHz boost clock, 8GB VRAM (GDDR7 per NVIDIA's
  laptop-line announcement; some outlets report GDDR6) ([Newegg](https://www.newegg.com/insider/news-nvidia-unveils-geforce-rtx-5050-laptop-gpu-2560-cuda-cores-8gb-gddr7-memory/), [Notebookcheck](https://notebookcheck.net/GeForce-RTX-5050-laptop-GPU-Geekbench-debut-confirms-key-specs.1037277.0.html))
- Desktop RTX 5050 (closest fully-published reference): 13.17 TFLOPS FP32,
  52.68 TFLOPS FP16 tensor (sparsity-rated, ~26.3 TFLOPS dense), 320 GB/s
  bandwidth, 130W TGP ([GPUpoet](https://gpupoet.com/gpu/learn/card/nvidia-geforce-rtx-5050))
- Laptop-adjusted estimate used: ~22.4 TFLOPS FP16 tensor (dense).

### Estimated training time by model size (original table)

| Model size | Chinchilla-optimal tokens | Naive PyTorch | Optimized (AMP+FlashAttention+compile) |
|---|---|---|---|
| 50M params | ~1.0B | 1.3 days | 13.3 hours |
| 150M params | ~3.0B | 11.6 days | 5.0 days |
| 350M params | ~7.0B | 63.3 days | 27.1 days |
| 700M params | ~14.0B | 253.3 days | 108.6 days |

Recomputable with `scripts/estimate_time.py`.

### VRAM feasibility on 8GB (original)

Rough rule of thumb for full pretraining with AdamW: ~16 bytes/param
before activations; ~8 bytes/param with an 8-bit optimizer. Standard
AdamW comfortably fits ~50-300M params; 8-bit optimizer + gradient
checkpointing can stretch to ~500-700M.

Sources: [nanoGPT speedrun worklog — Tyler Romero](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/), [Training GPT-2 on a budget (RTX 5080) — RecsysML](https://recsysml.substack.com/p/training-gpt-2-on-a-budget)
