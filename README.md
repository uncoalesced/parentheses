```
██████╗   █████╗  ██████╗  ███████╗ ███╗   ██╗ ████████╗ ██╗  ██╗ ███████╗ ███████╗ ███████╗ ███████╗
██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║  ██║ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
██████╔╝ ███████║ █████╔╝  █████╗   ██╔██╗ ██║    ██║    ███████║ █████╗   ███████╗ █████╗   ███████╗
██╔═══╝  ██╔══██║ ██╔══██╗ ██╔══╝   ██║╚██╗██║    ██║    ██╔══██║ ██╔══╝   ╚════██║ ██╔══╝   ╚════██║
██║      ██║  ██║ ██║  ██║ ███████╗ ██║ ╚████║    ██║    ██║  ██║ ███████╗ ███████║ ███████╗ ███████║
╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚══════╝ ╚══════╝
```

# parentheses-1.1

**A manifesto for scaling an attention-free, constant-memory recurrent model past 500M parameters on consumer hardware: smarter, faster, context-aware, uncensored, honest, and multimodal.**

Engineered by uncoalesced

> **Project status (November 2026): Planning and Manifesto Stage.**
> `parentheses-1.0-240M-tanzanite` (`253,544,176` parameters, `slab_v4`) is in release closeout. Nothing in `parentheses-1.1` has been implemented or trained yet. Active development moves into this directory once `1.0` closes.
>
> **Target timeline:** `parentheses-1.1` is targeted for **mid-December 2026**, ahead of the broader **Parentheses family release in January to February 2027**.

---

## Why parentheses-1.1 exists

`parentheses-1.0` proved that an attention-free State Space Duality (`slab_v4`) model built from scratch on a single 8 GB laptop GPU (NVIDIA GeForce RTX 5050 Laptop) can retain a **constant 4.85 MB recurrent state** at any sequence length, keep Dravidian aksharas whole, handle short Python and shell workflows, and abstain reliably on trap questions.

It also exposed five hard ceilings that define everything `parentheses-1.1` (`slab_v5`) is built to solve:

1. **Capacity and breadth:** At `253.54M` parameters, world knowledge, reading comprehension, and multi-domain depth are tightly constrained.
2. **Reasoning and math:** On a 30-question sample of grade-school math (`GSM8K`), `1.0` scored in the single digits, and its `<think>` mode emitted templated restatements rather than real reasoning.
3. **Inference and prompt-ingestion speed:** Generating ~20 to 34 tokens/s on an RTX 5050 Laptop GPU is far too slow for a `253M` model when quantized multi-billion-parameter transformers approach ~100 tokens/s on the same GPU.
4. **Context resolution vs. false abstention:** `1.0` handles specific prompts well, yet slightly vague queries or Turn-2 follow-ups either lock onto Turn 1 or bail straight out to `"I don't know"`.
5. **Single-modality isolation:** `1.0` is text-only in weights. Real local workflows require reading pasted images and documents directly, speaking answers on demand, and hearing spoken prompts.

`parentheses-1.1` does **not** throw away `1.0`. It inherits the exact `1.0` weights, recurrent algebra, precision guarantees, and Akshara-preserving tokenizer, and grows them past **500 million parameters (`>= 506M`, target ~540M to 610M)** while keeping the entire system strictly **zero-attention**.

---

## Core invariants (non-negotiable)

- **Continuous lineage from `1.0`:** `1.0` weights load 1:1 into `core_spine`. New capacity, experts, and modality adapters initialize cleanly on the write side so step-0 behavior matches `1.0` without wiping pretrained features.
- **Zero attention anywhere:** No softmax attention and no growing KV cache in the backbone, the vision encoder, or the audio codec. State memory remains bounded and `O(1)` in context length.
- **Intelligence and output quality come first:** Speed optimizations, quantization, pruning, or fractional loading must never degrade reasoning, accuracy, or multilingual fluency. If a shortcut costs intelligence, it is rejected.
- **Uncensored and open for research:** Zero safety-refusal rails, zero moralizing boilerplate. The model is an open instrument for studying reasoning, coding, chemistry, security, and logic without guardrail distortion.
- **Open weights, logged data:** Inference code under MIT, weights under Apache-2.0. The training pipeline and data are not published. Every source is logged privately with its licence and hash, and summarised by licence tier in [`docs/LICENSE_TABLE.md`](docs/LICENSE_TABLE.md). Non-commercial and do-not-train sources stay out.
- **Numbers from a standard harness:** Every published score follows [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md): lm-evaluation-harness, full test sets, greedy, one run, raw results committed.
- **Local-first hardware envelope:** Trainable and runnable on one 8 GB consumer laptop GPU (RTX 5050 Laptop) alongside modern consumer CPUs and NPUs.

---

## The five pillars of parentheses-1.1

### 1. Expanded intelligence first (253.54M -> 500M+ parameters)

Priority one is making the model broadly smarter across everything it touches: deeper technical reasoning, stronger English and Dravidian fluency (Kannada, Tamil, Telugu, Malayalam, plus experimental Tulu, Koya, Kuvi, and Kolami), wider factual grounding, and stronger code synthesis.

- **Modular capacity growth (`slab_v5`):** Keep the `16`-layer, `d_model=1024`, `16`-head additive recurrent spine (`253.5M` + vocabulary extension) intact, and grow total parameters past `506M+` via modular `experts` and modality shards.
- **No domain-cliff forgetting:** In `1.0`, pushing STEM data caused all nine languages' bits-per-byte (BPB) to regress within 100 steps until multilingual replay restored balance. Every training phase in `1.1` enforces replay floors and per-language BPB gates so gaining knowledge in one domain never erases another.
- **Wider and cleaner corpora:** Upgrade data quality, length coverage, multi-turn dialogue, and non-contaminated benchmarks, strictly verifying near-duplicate deduplication against every held-out evaluation split.

### 2. State-native reasoning and custom Chain of Thought

`parentheses-1.0` failed on multi-step grade-school arithmetic because it neither thought long enough nor thought in a structure compatible with recurrent memory.

Off-the-shelf Transformer Chain-of-Thought (CoT) datasets fail when distilled blindly onto a `250M to 500M` recurrent model:

- A Transformer has an `O(T)` KV cache and can wander through 1,500 tokens of meta-chatter ("wait, let me check again...") while still attending verbatim back to token 12.
- `parentheses` keeps one `64 x 64` matrix per head (`4.85 MB` total across 16 layers), holding roughly 64 near-orthogonal entries per head before older writes decay or interfere. Verbose filler literally flushes the problem's givens out of state before the last step runs.

Therefore, `parentheses-1.1` builds **custom, state-native Chain of Thought**:

- **Structured, dense, low-filler `<think>...</think>` traces:** Compact working memory steps that extract givens cleanly, keep arithmetic atomic, re-anchor intermediate state right before consumption, and self-verify before emitting the final answer.
- **No silent regex crutches:** Benchmark evaluation measures genuine model reasoning rather than hiding behind input-pattern calculator intercepts.
- **Multilingual reasoning parity:** Reasoning traces are constructed for both English and Dravidian languages from licensed, non-contaminated sources (without letting English-centric small teachers write broken Indic text).

### 3. Speed without compromising intelligence

In `1.0`, text generation on an RTX 5050 Laptop GPU sits around `20 to 34 tok/s` (FP32 Python loop), while multi-billion-parameter quantized Transformers approach `~100 tok/s` on the same GPU. For a model this compact, `30 tok/s` is an engineering bottleneck, not a hardware limit.

Profiling the `1.0` runtime reveals why:

1. **Sequential prompt ingestion:** `Engine` and `stream()` currently feed prompt tokens one-by-one through `model.step()`, launching thousands of tiny Python/CUDA ops per prompt instead of running a parallel chunked scan (`C=64`).
2. **Python eager kernel-launch latency:** Each generated token runs ~2,000+ uncompiled Python/PyTorch tensor ops across 16 layers (including per-step scalar tensor allocations inside the inner loop), leaving the GPU idle waiting on CPU dispatch.
3. **Unfused FP32 projections at chat time:** Running `1.0` inference weights in `float32` (`1.01 GB`) doubles memory traffic compared to `bfloat16` linear weights paired with `float32` recurrent state accumulation.
4. **Sequential sampling overhead:** Per-step Python list scans and `fp32` sorts over `32,768` logits add avoidable latency after every step.

`parentheses-1.1` attacks both **prompt ingestion** and **token output throughput** while keeping outputs mathematically faithful:

- **Parallel chunked prefill + state caching:** Process prompts with the `C=64` chunked SSD scan (`forward_with_state`) and reuse prefix-hashed `.pstate` checkpoints across conversation turns.
- **Compiled / CUDA-Graphed / fused `step()`:** Capture static-shape single-token decoding in CUDA Graphs (`torch.cuda.CUDAGraph`) or fused Triton/CUDA recurrent kernels to eliminate Python launch overhead.
- **Mixed precision done right:** Store linear weights in `bfloat16` (or lossless/near-lossless weight-only quantization verified against intelligence gates) while keeping the `4.85 MB` recurrent state accumulator `S_t` and decay math strictly in `float32` so long-context precision never drifts.
- **Fractional active compute:** Load only the required parameter slice per task (`1/5 to 1/3` active params for speech read-aloud and lightweight modes) without dragging unused modality towers into memory.

**Non-negotiable speed gate:** Every kernel, dtype, or runtime speedup must pass zero-regression checks against the reference `float32` accuracy suite before shipping.

### 4. Context understanding, uncensored openness, and 100% honest "I don't know"

`1.0` exhibits two coupled failure modes at the boundary of what it knows:

- **Vague prompts and Turn-2 lock-in:** When given a precise prompt (such as a clear Python specification), `1.0` performs well. When a prompt is slightly vague or arrives as a Turn-2 follow-up, it either repeats Turn 1's trajectory or panics directly into `"I don't know"` instead of inferring sensible intent or asking one focused clarifying question.
- **Risk of conflating `"I don't know"` with a safety refusal:** Standard open SFT datasets carry hidden RLHF refusal habits (`"As an AI, I cannot..."`). On a small model, training both abstention and refusal merges them into one generic dodge circuit, risking fake `"I don't know"` responses or refusals on valid technical, logical, chemistry, or mechanical questions (such as explaining how different lock mechanisms work as a test of mechanical reasoning).

`parentheses-1.1` enforces a strict **three-way behavioral separation**:

| Situation | Required 1.1 Behavior | Target Metric |
|---|---|---|
| **Answerable query** (including sensitive, dual-use, adversarial logic, or mechanical puzzles) | Direct, complete, technical answer with zero moralizing and zero false abstention | **0.0% refusals** |
| **Vague or underspecified query** (or multi-turn follow-up on a known topic) | Resolve context from conversation state, answer under stated assumptions, or ask a targeted clarifying question | **0% premature IDK bailouts** |
| **Genuinely unknowable, unverified, or false-premise query** | Clear, honest epistemic abstention stating specifically what is unknown or missing | **100% honest abstention** |

Training corpora are explicitly scrubbed of RLHF safety-refusal templates, paired with multi-turn context-resolution traces and calibrated epistemic abstention data verified independently in every supported language.

### 5. Attention-free multimodality and the new unified harness

`parentheses-1.1` expands from a text-only model into a unified, zero-attention multimodal system wired directly into a rebuilt, interoperable harness (`OpenAI`-compatible server, CLI, local UI, and external harness adapters):

1. **Image Input (Top Multimodal Priority):**
   - Paste screenshots, charts, data tables, scanned pages, code screenshots, diagrams, and Indic-script documents straight into the chat or harness.
   - Processed by an **attention-free convolutional / state-space vision encoder** (`modality_vision`) into dense `1024`-wide visual tokens fed between `<|image|>` and `<|/image|>` into the exact same recurrent loop.
   - Focus is strictly **analytical comprehension and reading** (extracting values, interpreting plots, reading dense text and Dravidian scripts accurately), alongside **deterministic analytical visual output** (typed scene graphs compiled to exact SVG charts, schematics, and ASCII diagrams; zero raster image generation).
2. **Text-to-Speech (On-Demand Read-Aloud):**
   - Integrated directly into the new harness via a dedicated **read-aloud key combination**: pressing the shortcut reads the assistant response aloud cleanly.
   - Runs in a **fractional parameter mode (`1/5 to 1/3` of total parameters)** using an attention-free neural audio codec (`modality_audio`, e.g., convolutional RVQ codec tokens interleaved between `<|audio|>` and `<|/audio|>`), avoiding loading the full vision and reasoning weights just to speak a reply.
3. **Speech Input (Secondary, Speaker-Agnostic):**
   - Understands spoken prompts from any speaker's voice through the same attention-free audio token interface.
   - Scoped intentionally leaner than image input and core reasoning so acoustic training neither crowds out text intelligence nor balloons hardware cost.

---

## Planned target architecture (`slab_v5`)

All `1.1` figures below are **design targets** to be measured and updated once Stage 2 and Stage 3 pilots execute.

| Shard / Module | Prefix | Target Params | Role |
|---|---|---:|---|
| Core recurrent spine + vocab extension | `spine.` | `~258M to 262M` | `1.0` `slab_v4` weights (`253.54M`) + audio/multimodal control tokens |
| Modular experts | `experts.` | `~180M+` | Zero-init write-side expansion for broad intelligence, STEM, CoT, and multilingual depth |
| Vision tower + projector | `vision.` | `~50M to 85M` | Zero-attention conv/SSM image, chart, table, and document reader |
| Audio speech bridge / head | `audio.` | `~45M to 85M` | Attention-free neural audio codec interface (TTS read-aloud + ASR) |
| **Total (`slab_v5`)** | | **`~540M to 610M+`** | **Strict `>= 506M` minimum, constant `O(1)` spine state** |

### Selective load modes (target)

| Mode | Active Shards | Purpose |
|---|---|---|
| `text-lite` | `spine` | Fast baseline `1.0`-compatible text inference |
| `text` | `spine` + `experts` | Full intelligence and custom CoT reasoning |
| `vision` | `spine` + `experts` + `vision` | Pasted images, charts, tables, and document reading |
| `tts` / `asr` | Fractional spine slice (`1/5 to 1/3` target total) + `audio` | Low-footprint read-aloud hotkey and voice input |
| `full` | All shards | Complete omni-modal session |

---

## Planned repository layout

The `1.1` codebase is a rewrite in Python and Rust. What is public:

- `python/parentheses/`: `slab_v5` model definition, shard loader, tokenizer files, and the lm-evaluation-harness adapter
- `rust/`: inference runtime (recurrent `step()`, chunked prefill, session state cache), tokenizer, OpenAI-compatible server, CLI, and Python bindings
- `eval/`: benchmark task lists, run configs, and raw results
- `tests/`: scan/step parity, step-0 `1.0` equivalence, and speed regression checks
- `docs/`: benchmark protocol and licence tiers

Training code, data builders and data ledgers are kept out of the repository.

---

## Baseline reference (`parentheses-1.0-240M-tanzanite`)

Every `1.1` milestone will be judged directly against the measured `1.0` baseline (float32 weights, RTX 5050 Laptop GPU):

| Metric | 1.0 Measured Baseline | 1.1 Direction |
|---|---|---|
| Parameters | `253,544,176` | `>= 506M` (`~540M to 610M` modular) |
| Recurrent spine state | `4.85 MB` (`fp32`, constant `O(1)`) | Preserved `O(1)` fixed state |
| Inference throughput (RTX 5050) | `20 to 27 tok/s` typical (`11 to 34` release range; `33.8` median in fp32 chat) | Chunked prefill + CUDA Graph / BF16 + fused kernels (zero quality loss) |
| Benchmarks (MMLU, GSM8K, TruthfulQA MC, Belebele, MBPP, HumanEval) | Being re-measured with lm-evaluation-harness; no number is quoted until that run is reviewed | Improve on the re-measured baseline, same protocol |
| False refusals (internal prompt suite, author-written) | `0/90` on ordinary prompts | Keep at zero, and add a suite not written by the author |
| Modalities | Text only | Image-In (#1 multimodal), Harness TTS hotkey (#2), Speech-In (#3) |

The 1.0 release tables were produced by a small in-house script and overstated the model. [`HISTORY.md`](HISTORY.md) explains what was wrong with them.

---

## Target release schedule

- **`parentheses-1.0` closeout:** Release finalization of the `253.54M` text baseline before active implementation begins in this directory.
- **`parentheses-1.1` release:** Targeted for **mid-December 2026** (`>= 506M` modular `slab_v5`, intelligence, state-native reasoning, runtime speedups, context/honesty calibration, and multimodal support).
- **Parentheses family release:** Targeted for **January to February 2027**, bringing the full model family together on the shared architecture and harness.

---

## Licenses

- Code and runtime: MIT License ([`LICENSE`](LICENSE))
- Model weights: Apache 2.0 ([`LICENSE-MODEL`](LICENSE-MODEL))
- Training data: not redistributed; see [`docs/LICENSE_TABLE.md`](docs/LICENSE_TABLE.md)

<p align="center"><sub>Engineered by uncoalesced</sub></p>
