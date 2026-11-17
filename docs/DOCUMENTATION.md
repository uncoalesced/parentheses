# parentheses-1.0-240M (slab_v4): technical reference

This document describes the 253,544,176-parameter recurrent language model
released as parentheses-1.0-240M. The model is a selective linear attention
network (`slab_v4`, in the Mamba-2 and RWKV family). It was grown from a
smaller checkpoint and trained on one 8 GB laptop GPU.

The model covers four anchor Dravidian languages (Kannada, `kan`; Tamil,
`tam`; Telugu, `tel`; Malayalam, `mal`), English (`eng`), and four experimental
low-resource languages (`tcy`, `kff`, `kxv`, `nit`). Its vocabulary has 32,768
byte-level BPE tokens. At inference, the recurrent state is a fixed 4.85 MB per
sequence, whatever the prompt length.

The code in this repository is the 1.1 package in `python/parentheses/`. It is
a clean rewrite of the 1.0 code, and it loads the 1.0 weights without changes.
The default configuration, `SLAB_V4_240M`, is the 1.0 shape. Work on the 1.1
model, which is planned to grow past 500M parameters and add image and speech
input, is tracked in [ROADMAP.md](ROADMAP.md) and [HISTORY.md](../HISTORY.md).
Where this document describes 1.0 material whose code is not in this
repository, it says so.

This reference has two parts:

- Part A covers running the model, the release package, and verification.
- Part B covers the architecture and the mathematics behind it.

For installation steps, see [SETUP.md](SETUP.md). For how benchmark numbers
are produced, see [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).

---

## Contents

- Part A: Using the codebase and release package
  - [1. Workspace layout and the code in this repository](#1-workspace-layout-and-the-code-in-this-repository)
  - [2. Release artifacts, precision, and checksums](#2-release-artifacts-precision-and-checksums)
  - [3. Running the model in Python and the terminal](#3-running-the-model-in-python-and-the-terminal)
  - [4. Static-shape step graph for NPUs](#4-static-shape-step-graph-for-npus)
  - [5. Growth, training, and SFT packing](#5-growth-training-and-sft-packing)
  - [6. Benchmarks and verification](#6-benchmarks-and-verification)
- Part B: Architecture, mathematics, and state-space engineering
  - [7. Notation](#7-notation)
  - [8. Model at a glance and parameter budget](#8-model-at-a-glance-and-parameter-budget)
  - [9. Tokenization and the 32k vocabulary](#9-tokenization-and-the-32k-vocabulary)
  - [10. Macro structure, norms, and initialization](#10-macro-structure-norms-and-initialization)
  - [11. The mixing layer, step by step](#11-the-mixing-layer-step-by-step)
  - [12. Associative memory algebra and a worked trace](#12-associative-memory-algebra-and-a-worked-trace)
  - [13. Chunked parallel scan](#13-chunked-parallel-scan)
  - [14. Document isolation](#14-document-isolation)
  - [15. Head decay timescales](#15-head-decay-timescales)
  - [16. SwiGLU, loss, and optimizer](#16-swiglu-loss-and-optimizer)
  - [17. Inference memory: constant state versus KV cache](#17-inference-memory-constant-state-versus-kv-cache)
  - [18. Decoding, sampling, and chat state](#18-decoding-sampling-and-chat-state)
  - [19. Known limits](#19-known-limits)
  - [20. Appendix: shapes, tests, source map, and licenses](#20-appendix-shapes-tests-source-map-and-licenses)

---

# Part A: Using the codebase and release package

## 1. Workspace layout and the code in this repository

| Path | Role |
|---|---|
| `python/parentheses/model.py` | `Parentheses` (the model), `Block`, `SelectiveLinearAttention`, `RMSNorm`, `SwiGLU`, `chunk_scan`, `find_documents`, `LayerState` |
| `python/parentheses/config.py` | `SlabConfig` and the 1.0 preset `SLAB_V4_240M` |
| `python/parentheses/checkpoint.py` | `load_model`, `load_state`, `save_safetensors`, `sha256_file` |
| `python/parentheses/session.py` | `ChatSession` (state kept between turns) and `RELEASE_SAMPLING` |
| `python/parentheses/graph_step.py` | `GraphStepper`, single-token decoding replayed from a CUDA graph |
| `python/parentheses/chat.py` | Terminal chat, `python -m parentheses.chat` |
| `python/parentheses/selftest.py` | Device check against the CPU, `python -m parentheses.selftest` |
| `python/parentheses/evaluate.py` | lm-evaluation-harness runner, `python -m parentheses.evaluate` |
| `python/parentheses/lm_eval_adapter.py` | `ParenthesesLM`, the harness model class |
| `python/parentheses/tokenizer_data/` | `tokenizer.json` (32,768 tokens) and `meta.json` |
| `tests/` | Parity and behavior tests (see [Section 20](#20-appendix-shapes-tests-source-map-and-licenses)) |
| `docs/` | Setup, benchmark protocol, licence table, roadmap |
| `tools/check_public_tree.py` | Leak checker that runs from the git hooks |
| `private/` | Gitignored. Training material, data notes and research files are kept here and never published |

Several parts of the 1.0 code are not in this repository. These include the
1.0 training loop, the `models/` and `features/` packages, the Web Studio
server, and the ONNX export script. The 1.0 history is in [HISTORY.md](../HISTORY.md).

Install the package from the repository root:

```bash
pip install -e .            # runtime: torch, safetensors, tokenizers
pip install -e ".[eval]"    # adds lm-evaluation-harness for python -m parentheses.evaluate
pip install -e ".[dev]"     # adds pytest for the test suite
```

On Windows, `pip install torch` often installs a CPU-only build. See
[SETUP.md](SETUP.md) for choosing a CUDA build and for Apple Silicon notes in
[APPLE_SILICON.md](APPLE_SILICON.md).

---

## 2. Release artifacts, precision, and checksums

The 1.0 weights are published on the Hugging Face Hub as
`uncoalesced-org/parentheses-1.0-240M`.

| File | Size (bytes) | Precision and role |
|---|---:|---|
| `model.safetensors` | 1,014,203,240 (about 967 MiB) | float32 weights. This is the canonical file, and what the commands in this document load |
| `model.pt` | 1,014,250,863 | Raw PyTorch state dictionary from the 1.0 training code. `load_state` reads it with `weights_only=True` |

The terminal chat and the selftest also accept the folder that holds the file,
so `--checkpoint weights` works as well as `--checkpoint weights/model.safetensors`.

Download the weights into a `weights` folder:

```bash
hf download uncoalesced-org/parentheses-1.0-240M model.safetensors --local-dir weights
```

The SHA-256 digest of the canonical file is
`bff69c7945533edaf379e69ffc50e47ed3b455c83cafb312defbd1b4c2208a6e`. To check
a download in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 weights\model.safetensors
```

The file should be about 1,014 MB. A partial download can still load and then
give wrong output, so re-download it if the size or digest differs.

`save_safetensors` writes the tied output head once, as the embedding. The
loader ties it back when the model is built, so a file without `head.weight`
loads.

The `.pt` file from 1.0 needs no code from the 1.0 repository. `load_state`
installs a small stand-in for the pickled config class and reads the weights
with `weights_only=True`, so no arbitrary pickle code runs.

For bfloat16 inference, pass `dtype` to the loader:

```python
import torch
from parentheses.checkpoint import load_model

model = load_model("weights/model.safetensors", device="cpu", dtype=torch.bfloat16)
```

The 1.0 release also shipped a bfloat16 file and an ONNX graph. Those files
are not produced or read by this repository.

Weights are licensed under Apache 2.0 (`LICENSE-MODEL`), and source code under
MIT (`LICENSE`). Training data licence tiers are listed in
[LICENSE_TABLE.md](LICENSE_TABLE.md).

---

## 3. Running the model in Python and the terminal

### Terminal chat

```bash
python -m parentheses.chat --checkpoint weights/model.safetensors
```

Type a message and press Enter. An empty line exits. The commands inside the
chat are:

- `/lang <code>` sends every message in that language. The codes are `kan`,
  `tam`, `tel`, `mal`, `eng`, `tcy`, `kff`, `kxv`, and `nit`.
- `/auto` picks the language from the script of each message.
- `/<code> text` sends one message in that language, for example `/tam vanakkam`.
- `/new` starts a fresh conversation. It only matters with `--keep-history`.

Other options are `--device` (`auto`, `cuda`, `mps` or `cpu`), `--format`
(`chat` or `base`), `--max-new-tokens`, `--temperature`, `--top-k`, `--min-p`,
`--no-repeat-ngram`, and `--seed`. The defaults are the release sampling
settings from [Section 18](#18-decoding-sampling-and-chat-state).

`--format chat` wraps each message in the chat template. `--format base` only
prefixes the language tag and continues your text. Auto-detection looks only at
the Kannada, Tamil, Telugu, and Malayalam Unicode blocks, so the experimental
languages are never chosen automatically. Force them with `--lang` or `/<code>`.

Each message starts a new conversation unless `--keep-history` is given. The
1.0 model was trained on single turns.

### Python: one reply

```python
import torch
from tokenizers import Tokenizer
from parentheses import TOKENIZER_PATH
from parentheses.checkpoint import load_model
from parentheses.session import RELEASE_SAMPLING, ChatSession

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("weights/model.safetensors", device=device)
tok = Tokenizer.from_file(str(TOKENIZER_PATH))

session = ChatSession(model, tokenizer=tok, lang="eng")
session.add_user("Why is water conservation important in summer?")
reply_ids = session.generate(max_new_tokens=128, **RELEASE_SAMPLING)
print(tok.decode(reply_ids))
```

`load_model` builds the model from the checkpoint's own tensor shapes, so it
needs no config file. It returns the model in eval mode.

### Python: raw token streaming

`Parentheses.stream` takes one sequence of token ids and yields the generated
ids. It has no `min_p` and no n-gram block.

```python
import torch
from tokenizers import Tokenizer
from parentheses import TOKENIZER_PATH
from parentheses.checkpoint import load_model

model = load_model("weights/model.safetensors", device="cpu")
tok = Tokenizer.from_file(str(TOKENIZER_PATH))

prompt = ([tok.token_to_id("<|user|>"), tok.token_to_id("<eng>")]
          + tok.encode("Explain why recurrent state memory stays fixed.", add_special_tokens=False).ids
          + [tok.token_to_id("<|assistant|>")])
ids = torch.tensor([prompt], dtype=torch.long)
print(tok.decode(list(model.stream(ids, max_new_tokens=128, temperature=0.2, top_k=40))))
```

`stream` reads the prompt with the parallel form, then generates with one
recurrent step per token. `temperature=0` is greedy decoding. It stops at
`<|endoftext|>`.

### Two details to know

1. `forward()` accepts any length. The parallel scan works on blocks of 64
   tokens, so `forward_with_state` pads the input on the right and switches the
   padding off inside each layer. The padding writes nothing and decays
   nothing. `step()` takes one token at a time, so a prompt of any length can
   be read either way.
2. There is no lazy state allocation to manage. Pass `None` for the first
   state, and each call returns the list of states to pass to the next call.
   `Parentheses.state_bytes(states)` reports the size of that list.

### Chat format

The 1.0 chat format, which the release was tuned on, has no system prompt:

```text
<|user|><lang> message <|assistant|> reply <|user|><lang> message <|assistant|> ...
```

`ChatSession.user_turn_ids` builds each user turn in this format. The stop
tokens are `<|endoftext|>`, `<|user|>`, `<|system|>`, `<|assistant|>`, and
`<|pad|>`. A reply ends at the first stop token, and the stop token is not
written into the state.

The 1.0 notes describe a `<think>` format as experimental. The tokens exist
(`<think>` is 15 and `</think>` is 16, see [Section 9](#9-tokenization-and-the-32k-vocabulary)),
but neither the terminal chat nor `ChatSession` adds them.

### Keeping state between turns

`ChatSession` keeps one recurrent state for the whole conversation. A new turn
only reads its own tokens; the history is never fed again. `session.ids_seen`
holds every token written into the state, in order. Reading those tokens in
one pass with `forward_with_state` gives the same state, and the tests check
this.

Never feed `<|endoftext|>` between turns. The token after it starts a new
document and resets the state, so it would wipe the conversation.

### Device check

```bash
python -m parentheses.selftest --checkpoint weights/model.safetensors
```

The selftest generates 16 tokens greedily on the CPU. It then replays the same
tokens on the selected device (CUDA or Apple `mps`), compares the logits, and
times both runs. It prints a report and one `PASS` or `FAIL` line. A pass means
the largest logit difference is below `1e-3` and the greedy tokens match. The
report is useful to paste into an issue if the output on an accelerator looks
wrong.

### Fixed-shape CUDA graph decoding

`GraphStepper` captures one recurrent step as a CUDA graph and replays it for
each token. It needs CUDA, and the model must already be on the GPU.

```python
from parentheses.graph_step import GraphStepper

g = GraphStepper(model)           # model already on CUDA
g.load_states(states)             # optional, e.g. the states after a parallel prefill
logits = g.step(token_id)         # a view of a static buffer; clone it to keep it
states = g.read_states()          # clones of the current states
```

The graph runs the same `Parentheses.step` code on the same tensors, so its
output matches eager mode. The tests check this when CUDA is available. The
step after `<|endoftext|>` runs eagerly, because it is rare and resets the state.

---

## 4. Static-shape step graph for NPUs

The 1.0 release also included a static-shape, single-token ONNX graph for
DirectML and AMD Ryzen AI. Stacking all 16 layers into contiguous buffers lets
an NPU run a step without dynamic allocation. The export script is not in this
repository, so the graph cannot be rebuilt from this code. The interface is
described here because the state layout is part of the architecture.

| Tensor name | Role | Shape (batch 1) | Dtype | Elements |
|---|---|---|---|---:|
| `token_id` | Input token | `[1, 1]` | int64 | 1 |
| `s_states` / `new_s_states` | Stacked associative matrices | `[16, 1, 16, 64, 64]` | float32 | 1,048,576 |
| `conv_states` / `new_conv_states` | Stacked causal conv buffers (3 taps) | `[16, 1, 3, 3072]` | float32 | 147,456 |
| `shift_states` / `new_shift_states` | Stacked previous layer inputs | `[16, 1, 1024]` | float32 | 16,384 |
| `logits` | Output distribution | `[1, 32768]` | float32 | 32,768 |

The recurrent buffers total 1,212,416 floats, or 4,849,664 bytes (4.85 MB), in
each direction per step.

The graph bakes in `is_boundary=False`, so it has no reset branch. To start an
unrelated prompt, or to cross an `<|endoftext|>` boundary, the host must zero
the three state arrays. The same applies to `GraphStepper`: call
`load_states(None)` to clear the state.

---

## 5. Growth, training, and SFT packing

This section describes how the 1.0 model was produced. The training code, the
growth script, the data packers, and the averaging script are not in this
repository. Nothing here can be rerun from this code.

### 5.1 Function-preserving growth

The 1.0 model was grown from an earlier 120M checkpoint rather than trained
from random weights. Growth widened the residual stream from 768 to 1,024 and
kept 16 layers and a head dimension of 64. The method was Net2WiderNet:

- Heads and layers are unchanged, so $12 \times 64 = 768$ becomes $16 \times 64 = 1024$.
- New residual-writing columns and rows start at exactly zero. The grown model
  therefore starts with the same output distribution as the source.
- The RMSNorm weights are scaled by $\sqrt{768 / 1024}$ to keep normalized
  activations balanced across the wider stream.
- Heads 0 to 11 inherit the decay, write-gate, and token-shift parameters of
  the source. Heads 12 to 15 copy heads 0 to 3, with small Gaussian jitter.

### 5.2 Training stages

Pretraining, annealing, supervised fine-tuning (SFT), preference tuning,
checkpoint averaging, and final alignment ran in sequence on one 8 GB laptop
GPU. The stages were:

- Width-expansion warm-up, then multilingual pretraining on the Dravidian and
  English corpora.
- A short phase on STEM, mathematics, and Python code. Loss on the training
  mix fell, but loss on the held-out languages rose. This was the main lesson
  of the run.
- Replay of Dravidian and English text, to bring the held-out losses back.
- A learning-rate anneal on commercially licensed corpora.
- SFT on 4,096-token packed containers across many sources, with a share of
  Dravidian replay.
- Preference tuning on a small set of on-policy pairs.
- Checkpoint averaging of the last few SFT checkpoints, and two short final
  passes on decontaminated Dravidian data and abstention examples.

The lessons from the run were these. Read loss curves together with the data
mix, because a mix can lower training loss while held-out loss rises. Keep
replay in the mix to hold the held-out languages in place. Use a loss floor
and an abort threshold during SFT, so a short patch cannot memorize its
examples.

### 5.3 Abstention data and packing

Two data pipelines were built for SFT, both from the 1.0 training code:

- A generator that wrote several thousand multilingual abstention traces. Each
  trace is a question with a false premise, such as a fabricated event or an
  unanswerable question, and a reasoning trace that declines to invent an
  answer.
- A first-fit-decreasing packer on a segment tree. It packs whole conversations
  into 4,096-token containers with three rules: no document is cut in the
  middle, every `<think>` block is closed, and every document ends with
  `<|endoftext|>`. The ending token triggers the boundary reset described in
  [Section 14](#14-document-isolation). Masked prompt spans and padding carry
  the ignore index `-100`.

Checkpoint averaging is done on the CPU in float32, so it does not use GPU
memory.

---

## 6. Benchmarks and verification

The benchmark tables from the 1.0 documentation have been removed from this
reference. They were produced by scripts outside the 1.1 code, and readers found problems
with how they were scored. See
[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) for how published numbers are produced
now, and [HISTORY.md](../HISTORY.md) for the record of earlier releases.

Verification in this repository is done with the test suite and the device
selftest:

```bash
python -m pytest                          # all tests in tests/
python -m parentheses.selftest --checkpoint weights/model.safetensors
```

Some tests need extra inputs and skip without them:

- `PARENTHESES_BASE_CKPT` points at a checkpoint. The adapter, session, chat,
  and release tests use it.
- `PARENTHESES_REF_REPO` points at the reference implementation. The release
  tests in `tests/test_slab_v4.py` need both variables.
- `PARENTHESES_PARITY_TOL` sets the tolerance for those release tests. It
  defaults to `1e-5`.
- The graph tests need CUDA.

Evaluation runs go through `python -m parentheses.evaluate`. See
[Section 3](#3-running-the-model-in-python-and-the-terminal) for the command. A
run with `--limit` is a smoke test. It is saved under a name that starts with
`SMOKE`, and its numbers are not results.

---

# Part B: Architecture, mathematics, and state-space engineering

The sections below describe `python/parentheses/model.py` and the other files
listed in [Section 1](#1-workspace-layout-and-the-code-in-this-repository).
Where a section describes 1.0 code that is not in this repository, it is
marked as such.

## 7. Notation

| Symbol | Value in 1.0 | Code name | Meaning |
|---|---|---|---|
| $B$ | runtime | `B` | Batch size |
| $T$ | runtime | `T` | Sequence length. Any length works; the scan pads to a multiple of $C$ |
| $C$ | 64 | `chunk_size` | Block length of the parallel scan |
| $d$ | 1,024 | `n_embd` | Residual stream width |
| $H$ | 16 | `n_head` | Number of heads |
| $d_h$ | 64 | `head_dim` | Per-head width, $d / H$ |
| $L$ | 16 | `n_layer` | Number of blocks |
| $d_{\text{ff}}$ | 2,752 | `mlp_hidden` | Inner width of the SwiGLU MLP |
| $V$ | 32,768 | `vocab_size` | Vocabulary size |
| $K$ | 4 | `CONV_KERNEL` | Causal conv width. Three past inputs are cached |
| $\mu$ | $\in \mathbb{R}^{1024}$ | `mu` | Learned per-channel mix for the token shift |
| $S_t$ | $\in \mathbb{R}^{16 \times 64 \times 64}$ | `memory` | Per-layer associative matrix, one $64 \times 64$ matrix per head, kept in float32 |
| $\alpha_t$ | $\in (0, 1]^{16}$ | `exp(log_alpha)` | Per-head decay |
| $\beta_t$ | $\in (0, 2)^{16}$ | `beta` | Per-head write gate, `beta_scale * sigmoid(...)` |
| $m_t$ | boolean | `reset` | True when token $t$ is the first token of a document |
| $d_t$ | integer | `doc` | Document index, the running count of resets |

Subscripts are dropped when an operation acts on each sample and layer on its
own.

---

## 8. Model at a glance and parameter budget

The 1.0 shape is the default `SlabConfig`:

| Hyperparameter | Value | Field |
|---|---:|---|
| Layers | 16 | `n_layer` |
| Model width | 1,024 | `n_embd` |
| Heads | 16 | `n_head` |
| Head dimension | 64 | `head_dim` (derived) |
| MLP hidden width | 2,752 | `mlp_hidden` (derived) |
| Vocabulary | 32,768 | `vocab_size` |
| Scan block length | 64 | `chunk_size` |
| Longest training length | 4,096 | `block_size` |
| Write-gate scale | 2.0 | `beta_scale` |
| `<|endoftext|>` id | 0 | `eot_token_id` |

`MLP hidden` is $\lceil 8d/3 \rceil$ rounded up to a multiple of 64, which
gives 2,752 for $d = 1024$.

Building the model from the default config and counting its parameters gives:

```python
from parentheses.config import SLAB_V4_240M
from parentheses.model import Parentheses

print(Parentheses(SLAB_V4_240M).num_params())   # 253544176
```

The breakdown by component is:

| Component | Shapes across 16 layers | Parameters | Share |
|---|---|---:|---:|
| Tied embedding and output head (`tok_emb.weight`, `head.weight` shares it) | $32{,}768 \times 1{,}024$, stored once | 33,554,432 | 13.23% |
| SwiGLU MLPs (`w1`, `w3`, `w2`) | $16 \times 3 \times (2{,}752 \times 1{,}024)$ | 135,266,304 | 53.35% |
| Input projection (`attn.qkv`) | $16 \times (3{,}072 \times 1{,}024)$ | 50,331,648 | 19.85% |
| Output gate (`attn.gate`) | $16 \times (1{,}024 \times 1{,}024)$ | 16,777,216 | 6.62% |
| Output projection (`attn.proj`) | $16 \times (1{,}024 \times 1{,}024)$ | 16,777,216 | 6.62% |
| Norms (`ln1`, `ln2`, `ln_f`) | $32 \times 1{,}024 + 1{,}024$ | 33,792 | 0.01% |
| Decay gate (`alpha_proj`, weight and bias) | $16 \times (16 \times 1{,}024 + 16)$ | 262,400 | 0.10% |
| Write gate (`beta_proj`, weight and bias) | $16 \times (16 \times 1{,}024 + 16)$ | 262,400 | 0.10% |
| Causal conv (`conv_weight`, `conv_bias`) | $16 \times (3{,}072 \times 4 + 3{,}072)$ | 245,760 | 0.10% |
| Token-shift `mu` and per-head `y_norm` | $16 \times 1{,}024 + 16 \times (16 \times 64)$ | 32,768 | 0.01% |
| Value residual mix (`v_residual_lamb`, layers 1 to 15) | $15 \times 16$ | 240 | 0.00% |
| **Total** | | **253,544,176** | **100%** |

The total matches the count from `num_params()` above. The recurrent gating,
conv, token-shift, value mix, and norm parameters together make 837,360
parameters (0.33%). Almost all of the weights are in the embedding and the
linear layers.

---

## 9. Tokenization and the 32k vocabulary

Indic scripts split badly under a plain letter-based regex. In Kannada, Tamil,
Telugu, and Malayalam, a consonant, a virama, and a dependent vowel sign fall
in different Unicode categories: letters (`\p{L}`) and combining marks
(`\p{M}`). A pattern such as `\p{L}+` detaches each vowel sign into its own
token, which costs sequence length and stops BPE from learning syllables.

The tokenizer in `python/parentheses/tokenizer_data/tokenizer.json` applies
Unicode NFC normalization, then a split with this pattern, then byte-level
pre-tokenization without a prefix space:

```text
'(?i:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}‌‍]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}‌‍]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Because `[\p{L}\p{M}‌‍]+` keeps vowel signs and viramas in the same
pre-token as their letter, the BPE merges are never taken across the mark and
its base. That is what the pattern is for.

The pre-tokenizer does not keep every akshara whole, however. BPE merges work
inside a pre-token, and they can still split an akshara before a vowel sign.
For example, `ಭಾಷೆಯಲ್ಲಿ` is encoded as two tokens, `ಭಾಷ` followed by `ೆಯಲ್ಲಿ`,
so the vowel sign `ೆ` starts the second token. The tokenizer keeps the marks
attached to their letters before merging, but it does not guarantee that a
syllable is one token. Do not describe it as akshara-preserving without that
qualification.

The tokenizer has 32,768 tokens. The first 256 ids are reserved for special
tokens:

| Token id | Literal | Purpose |
|---:|---|---|
| 0 | `<|endoftext|>` | End of a document. The token after it starts a new document and resets the state |
| 1 | `<|pad|>` | Padding. Masked with `-100` in SFT containers |
| 2 to 10 | `<kan>`, `<tam>`, `<tel>`, `<mal>`, `<tcy>`, `<kff>`, `<kxv>`, `<nit>`, `<eng>` | Language tags |
| 11 | `<esg>` | Reserved. Its language is excluded from the release training |
| 12 to 14 | `<|system|>`, `<|user|>`, `<|assistant|>` | Turn delimiters |
| 15 to 16 | `<think>`, `</think>` | Reasoning delimiters |
| 17 to 18 | `<tool_call>`, `</tool_call>` | Reserved tool markers. Stripped from chat output |
| 19 to 255 | reserved | Free for later use, for example with image or speech input |

The token ids in this table were read from `tokenizer.json`. The byte counts
per token that the 1.0 notes gave for each language are not reproduced here,
because the measurement script is not in this repository.

---

## 10. Macro structure, norms, and initialization

`Parentheses.forward_with_state` does these steps. The input ids are first
turned into document flags with `find_documents`. The flags are `reset`, which
is true at the first token after each `<|endoftext|>`, and `doc`, which is a
running document count. The token embeddings are then passed through the 16
blocks, followed by a final norm and the tied output head.

Each block applies pre-norm residual connections:

$$\begin{aligned}
a^{(l)}, v_0 &= \operatorname{SelectiveLinearAttention}_l\bigl(\operatorname{RMSNorm}(x^{(l)}),\; \text{reset},\; \text{doc},\; v_0\bigr) \\
\hat{x}^{(l)} &= x^{(l)} + a^{(l)} \\
x^{(l+1)} &= \hat{x}^{(l)} + \operatorname{SwiGLU}_l\bigl(\operatorname{RMSNorm}(\hat{x}^{(l)})\bigr)
\end{aligned}$$

The norm is `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight` with $\varepsilon = 10^{-5}$.
The norm is computed in the dtype of its input. Unlike the 1.0 description,
this code does not cast to float32 first. The difference is small for float32
inference, but the 1.0 notes stated a float32 upcast that this code does not
perform.

The head is tied: `head.weight` is the same tensor as `tok_emb.weight`. The
model has no dropout.

### Initialization in `Parentheses.reset_parameters`

A fresh model starts with these rules:

1. Linear and embedding weights are drawn from $\mathcal{N}(0, 0.02^2)$, and
   linear biases are zero.
2. The depthwise conv starts as a pass-through. Its last tap is 1 and the
   others are 0.
3. The decay bias is set so that each head begins with a different span. See
   [Section 15](#15-head-decay-timescales).
4. The write gate weight `beta_proj.weight` starts at zero.
5. The residual writers, `attn.proj.weight` and `mlp.w2.weight`, are drawn from
   $\mathcal{N}(0, \sigma_{\text{res}}^2)$ with $\sigma_{\text{res}} = 0.02 / \sqrt{2L} \approx 0.003536$.

The 1.0 notes describe a slightly different fresh start, with `alpha_proj.weight`
at zero and `beta_proj.weight` at the normal draw. This code differs in both
places. The difference only matters when training from scratch. A loaded
checkpoint overwrites all of these values.

---

## 11. The mixing layer, step by step

The layer is `SelectiveLinearAttention` in `model.py`. Its input is the
normalized residual stream $x \in \mathbb{R}^{B \times T \times 1024}$. The
steps run in this order.

### Step 1: Token shift for the gates

The gates read a blend of the current token and the previous one:

$$x^{\text{ctrl}}_t = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$$

Here $\mu$ starts at 0.5. At a document start, $x_{t-1}$ is replaced by zero.

The gates, the write strength, and the output gate read $x^{\text{ctrl}}$. The
query, key, and value projection reads the unshifted $x$. The gates can notice
delimiters and script changes one token early, while the content projection
stays unblurred for the conv that follows.

### Step 2: Projection

$$u_t = W_{qkv}\, x_t \in \mathbb{R}^{3072}$$

### Step 3: Causal depthwise conv and SiLU

Each of the 3,072 channels is convolved over the last four positions, and only
over positions in the same document:

$$\bar{u}_t = \operatorname{SiLU}\!\left( b_{\text{conv}} + \sum_{k=0}^{3} \mathbb{1}[d_t = d_{t-k}] \; w_{\text{conv}, k} \odot u_{t-k} \right)$$

The output is split into 16 heads of 64 values each, for $q$, $k$, and $v$.

### Step 4: Normalize keys and scale queries

Within each head, $q$ and $k$ are L2-normalized in float32. Then $q$ is scaled
by $1 / \sqrt{d_h}$:

$$q_t = \frac{1}{8}\,\frac{q_{\text{raw}, t}}{\max(\|q_{\text{raw}, t}\|_2,\, 10^{-12})}, \qquad k_t = \frac{k_{\text{raw}, t}}{\max(\|k_{\text{raw}, t}\|_2,\, 10^{-12})}$$

Unit-length keys keep the outer products $k_t^\top v_t$ bounded over long
sequences.

### Step 5: Value residual from layer 0

Layer 0 keeps its values as $v_0$. Each later layer mixes its own values with
$v_0$ through a learned per-head sigmoid:

$$v^{(l)} \leftarrow \sigma(\lambda^{(l)}) \odot v^{(l)} + \bigl(1 - \sigma(\lambda^{(l)})\bigr) \odot v_0$$

`v_residual_lamb` starts at zero, so the first mix is an even split. In the
trained 1.0 checkpoint the layer means of $\sigma(\lambda)$ fall between about
0.47 and 0.54, so the later layers keep drawing about half of their values
from the layer-0 signal.

### Step 6: Write gate

$$\beta_t = 2 \cdot \sigma\!\left(W_\beta x^{\text{ctrl}}_t + b_\beta\right) \in (0, 2)$$

The factor 2 is `beta_scale`. A salient token can write with a gate near 2,
and a filler token can write with a gate near 0. The gate is applied to the
value, as `v * beta`, before the scan.

### Step 7: Decay

$$\log \alpha_t = -\operatorname{softplus}\!\left(W_\alpha x^{\text{ctrl}}_t + b_\alpha\right) \le 0$$

So $\alpha_t = \exp(\log \alpha_t) \in (0, 1]$. A head can forget quickly or hold
almost everything, but its memory never grows exponentially.

### Step 8: Scan

`chunk_scan` runs the parallel form over whole blocks during `forward`. During
`step`, the recurrent update in [Section 12](#12-associative-memory-algebra-and-a-worked-trace)
runs instead. Both keep one $64 \times 64$ state per head in float32 or wider.

### Step 9: Per-head norm

Each head is normalized on its own, in the input dtype or wider:

$$y \leftarrow \frac{y}{\sqrt{\operatorname{mean}(y^2) + 10^{-5}}} \odot \gamma_{\text{norm}}$$

where $\gamma_{\text{norm}} \in \mathbb{R}^{16 \times 64}$ starts at one. Each
head is normalized before the gate, so fast-decaying heads and slow-decaying
heads contribute on the same scale.

### Step 10: Output gate and projection

$$\text{out}_t = W_{\text{proj}}\!\left(\operatorname{flatten}(y_t) \odot \operatorname{SiLU}(W_{\text{gate}}\, x^{\text{ctrl}}_t)\right)$$

---

## 12. Associative memory algebra and a worked trace

### 12.1 The recurrence

For one head at step $t$, with $m_t$ the reset flag, `step` computes:

$$\begin{aligned}
S_t &= \bigl(\alpha_t \cdot (1 - m_t)\bigr)\, S_{t-1} + \beta_t\, k_t^\top v_t \\
y_t &= q_t\, S_t
\end{aligned}$$

Unrolled inside one document:

$$y_t = \sum_{\substack{s \le t \\ d_s = d_t}} \left( \prod_{r = s+1}^{t} \alpha_r \right) \beta_s\, (q_t \cdot k_s)\, v_s$$

Standard attention normalizes $\exp(q_t \cdot k_s)$ over every past position, and
its cache grows with the sequence. Here each token is written once into a fixed
matrix. The write is scaled by $\beta_s$, and older writes fade by the product
of decays. The matrix is shared by all positions, so recall of any single
earlier token is limited, as [Section 19](#19-known-limits) explains.

### 12.2 A worked 4-token trace

Take one head with $d_h = 4$, and the token stream `[12, 45, 0, 88]`, where 0 is
`<|endoftext|>`. The reset flags are `[False, False, False, True]`, and the
document indices are `[0, 0, 0, 1]`. These flags come from `find_documents`
(the reset fires after the `<|endoftext|>` token).

Start from $S_{-1} = 0$:

| Step $t$ | Token (document) | Effective decay $\alpha_t (1 - m_t)$ | $\beta_t$ | State $S_t[:2,:2]$ after update | Output $y_t[:2]$ |
|---:|---|---:|---:|---|---|
| 0 | 12 (0) | 0.50 | 1.00 | $\begin{bmatrix} 2.00 & 0.00 \\ 0.00 & 0.00 \end{bmatrix}$ | 0.60, 0.00 |
| 1 | 45 (0) | 0.50 | 0.50 | $\begin{bmatrix} 1.00 & 0.00 \\ 0.00 & 2.00 \end{bmatrix}$ | 0.40, 0.60 |
| 2 | 0 (0) | 0.25 | 1.00 | $\begin{bmatrix} 0.85 & 0.60 \\ 0.80 & 1.30 \end{bmatrix}$ | 0.40, 0.65 |
| 3 | 88 (1) | 0.00 | 1.60 | $\begin{bmatrix} 0.00 & 0.00 \\ 0.00 & 8.00 \end{bmatrix}$ | 0.00, 0.00 |

Step 3 is the reset. The matrix from document 0 is wiped before the new write,
so the query at step 3 reads zero and does not see stale state from document 0.
The same numbers come out of the recurrence, and `chunk_scan` gives the same
outputs for steps 0 to 2 on this input.

### 12.3 Delta rule (not in this repository)

The 1.0 code also had an opt-in error-correcting write, called the gated delta
rule. It replaces the write with the residual of what the state already
predicts for the key:

$$S_t = a_t\, S_{t-1} + k_t^\top \bigl(\beta_t (v_t - a_t\, k_t S_{t-1})\bigr), \qquad a_t = \alpha_t (1 - m_t)$$

The released 1.0 checkpoint uses the plain additive write above, and so does
1.1. The delta rule is not implemented in `python/parentheses/model.py`.

---

## 13. Chunked parallel scan

`chunk_scan(q, k, v, log_alpha, reset, doc, chunk, memory)` computes the
recurrence for a whole sequence at once. It returns the outputs and the final
state. The sequence is split into $N = T / C$ blocks of $C = 64$ tokens.
`forward_with_state` pads $T$ to a multiple of 64 before calling it.

The scan works in float32 at least. `_wide` keeps float64 if the input is
float64. Half precision loses the long decays: a decay near 0.99997 is not
represented well in 7 mantissa bits, and the errors compound over thousands of
steps.

For each block, with $P_t = \sum_{\tau \le t} \log \alpha_\tau$ taken inside the
block:

1. Inside the block, token $i$ reads token $j \le i$ only if they share a
   document. The decay between them is $\exp(P_i - P_j)$:

   $$Y^{\text{intra}} = \bigl((Q K^\top) \odot L\bigr) V^{\beta}, \qquad L_{i,j} = \exp(P_i - P_j) \cdot \mathbb{1}[i \ge j,\; d_i = d_j]$$

   The disallowed entries are set to negative infinity before the `exp`, so
   they become exactly zero without producing infinities in the gradient.

2. Between blocks, the state is handed forward. The block writes its own
   contribution to the state and decays the state it received:

   $$\begin{aligned}
   U_c &= (K_c \odot w_{\text{out}})^\top V^{\beta}_c, \qquad w_{\text{out}}(j) = \exp(P_{C-1} - P_j)\,\mathbb{1}[d_{C-1} = d_j] \\
   A_c &= \exp(P_{C-1})\,\mathbb{1}[d_{C-1} = d_{\text{prev}}] \\
   Y^{\text{inter}}_c &= (Q_c \odot w_{\text{in}})\, S_{c-1}, \qquad w_{\text{in}}(i) = \exp(P_i)\,\mathbb{1}[d_i = d_{\text{prev}}] \\
   S_c &= A_c\, S_{c-1} + U_c
   \end{aligned}$$

   where $d_{\text{prev}}$ is the document open just before the block (see
   [Section 14](#14-document-isolation)). Output is $Y_c = Y^{\text{intra}}_c + Y^{\text{inter}}_c$.

The inter-block carry is a Python loop over the $N$ blocks. For a 4,096-token
sequence, $N = 64$. The per-block matrices are computed in parallel before the
loop, so the loop only does $N$ small matrix products.

---

## 14. Document isolation

Training packs complete documents one after another in a window, separated by
`<|endoftext|>`. Resetting the matrix state alone would leave two leaks: the
token shift would blend the last token of one document into the first token of
the next, and the conv would mix three trailing tokens across the boundary. The
code closes all three channels:

| Channel | Where | Rule |
|---|---|---|
| Token shift | `SelectiveLinearAttention.forward` and `step` | The previous token is replaced by zero where `reset` is true |
| Causal conv | `forward` (taps) and `step` (conv buffer) | A tap counts only when its document equals the current token's document. `step` zeroes the buffer at a boundary |
| Scan | `chunk_scan` | Inside a block, tokens read only the same document. Across blocks, `w_in` and `A_c` use `d_prev` |

`d_prev` is the document open just before the block. It is computed as
`doc[:, 0] - reset[:, 0]`, so a reset on the block's first token is handled
correctly. The scan covers four cases:

1. No reset in the block. The carried state flows through normally.
2. A reset on the block's first token. The incoming state is zeroed, and the
   block's own writes stay.
3. One or more resets inside the block. Only the tokens before the first reset
   read the incoming state. Only the tokens after the last reset are written
   out.
4. A reset on the last token. The incoming state is zeroed, and only the last
   token is written out.

`step` handles a boundary by zeroing the decay, the conv buffer, and the token
shift, which is the same as the parallel form. The tests `test_documents_are_isolated`
and `test_endoftext_inside_a_feed_and_at_a_turn_end` in `tests/` cover this
behavior.

---

## 15. Head decay timescales

### 15.1 Fresh initialization

`reset_parameters` sets each head's decay to a forget rate on a log scale.
Writing $f_h = 10^{-1 - 3.5 h / 15}$ for head $h$, the bias is chosen so that
$\alpha_h = 1 - f_h$. The half-life in tokens is $t_{1/2} = \ln 2 / (-\ln \alpha_h)$.

| Head | $\alpha_h$ at rest | Half-life (tokens) |
|---:|---:|---:|
| 0 | 0.90000000 | 6.6 |
| 1 | 0.94156586 | 11.5 |
| 2 | 0.96585451 | 20.0 |
| 3 | 0.98004738 | 34.4 |
| 4 | 0.98834086 | 59.1 |
| 5 | 0.99318708 | 101.4 |
| 6 | 0.99601893 | 173.8 |
| 7 | 0.99767369 | 297.6 |
| 8 | 0.99864064 | 509.6 |
| 9 | 0.99920567 | 872.3 |
| 10 | 0.99953584 | 1,493.0 |
| 11 | 0.99972877 | 2,555.2 |
| 12 | 0.99984151 | 4,373.1 |
| 13 | 0.99990739 | 7,484.1 |
| 14 | 0.99994588 | 12,808.0 |
| 15 | 0.99996838 | 21,918.9 |

These are the values before training. The input-dependent projection
$W_\alpha x^{\text{ctrl}}_t$ moves them at each token.

### 15.2 Trained values in the 1.0 checkpoint

Training changed these biases. The values below were read from layer 0 of the
1.0 checkpoint, at zero input. They are the 1.0 values, and they show the heads
keep a spread of spans after training.

| Head | Rest $\alpha_h$ | Half-life (tokens) | Role |
|---:|---:|---:|---|
| 0 | 0.893299 | 6.1 | Immediate syntax and bigrams |
| 1 | 0.955306 | 15.2 | Clause and local morphology |
| 2 | 0.981171 | 36.5 | Phrase window |
| 3 | 0.990871 | 75.6 | Multi-sentence state |
| 4 | 0.995243 | 145.4 | Short paragraph context |
| 5 | 0.997676 | 297.9 | Multi-paragraph state |
| 6 | 0.998867 | 611.2 | Page-scale topic |
| 7 | 0.999447 | 1,253.7 | Extended instruction span |
| 8 | 0.999722 | 2,493.6 | Near the block size |
| 9 | 0.999860 | 4,952.4 | Full training context |
| 10 | 0.999930 | 9,854.8 | Long session |
| 11 | 0.999969 | 22,192.6 | Document-level scaffold |
| 12 to 15 | 0.8933 to 0.9906 | 6.1 to 73.3 | Extra fast and mid-span heads, copied from heads 0 to 3 at growth |

The role names are descriptions of the spans, not measured functions.

---

## 16. SwiGLU, loss, and optimizer

### 16.1 SwiGLU

Each block ends with a bias-free SwiGLU MLP:

$$\operatorname{SwiGLU}(u) = W_2\!\left(\operatorname{SiLU}(W_1 u) \odot (W_3 u)\right)$$

with a hidden width of 2,752. It has 135,266,304 parameters, which is 53.35%
of the model. It is the largest store of static knowledge in the network. The
recurrent state holds the context of the current sequence, and the MLP holds
the rest.

### 16.2 Loss and training (not in this repository)

The 1.0 training code used these choices. None of them are in 1.1, so they
cannot be rerun from this repository.

- Cross-entropy was computed in row chunks, so the full logit tensor of a
  batch was never materialized. Checkpointing kept the chunks out of memory.
- A weight on `<|endoftext|>` could be raised, so that the model stopped more
  cleanly. The weighted loss is
  $\mathcal{L} = \sum_i w_i \mathrm{CE}_i / \sum_i w_i$, with $w_i = e_w$ for
  end-of-text targets and $w_i = 1$ otherwise.
- Target tokens marked `-1` were mapped to the ignore index `-100`.

### 16.3 Optimizer groups (not in this repository)

The 1.0 training code split the parameters into three AdamW groups, with
$\beta = (0.9, 0.95)$ and $\varepsilon = 10^{-8}$:

| Group | Parameters | Learning rate | Weight decay |
|---|---|---|---:|
| Matrix projections | Linear weights of `qkv`, `gate`, `proj`, `w1`, `w2`, `w3` | $\text{lr}$ | 0.1 |
| Embeddings and norms | `tok_emb`, the norms, `y_norm`, and any other one-dimensional parameter | $\text{lr}$ | 0.0 |
| Recurrent dynamics | Parameters with `alpha`, `beta`, `mu`, `conv`, or `v_residual` in the name | $0.1 \cdot \text{lr}$ | 0.0 |

The reduced rate for recurrent parameters protects the decay horizons and conv
filters from large steps. Weight decay is kept off the embeddings and norms,
because decay would shrink them.

---

## 17. Inference memory: constant state versus KV cache

Each layer keeps three tensors per sequence. They are listed here in float32:

| Buffer | Shape per layer | Bytes per layer |
|---|---|---:|
| Associative matrix `memory` | `(1, 16, 64, 64)` | 262,144 |
| Conv history | `(1, 3, 3072)` | 36,864 |
| Token-shift `prev` | `(1, 1024)` | 4,096 |
| **Per layer** | | **303,104** |

Over 16 layers the total is 4,849,664 bytes, or 4.85 MB. `Parentheses.state_bytes`
returns this number for the states from `step`, and the value was checked
against the code for this configuration. The total does not depend on the
number of tokens read.

For comparison, a Transformer of the same depth and width, in float32 with no
grouped-query attention, keeps a key and a value per token and layer:

$$\text{KV}_{\text{Transformer}}(T) = 2 \times 16 \times 1024 \times 4\ \text{bytes} \times T = 131{,}072\, T\ \text{bytes}$$

- The two match at $4{,}849{,}664 / 131{,}072 = 37.0$ tokens. Past about 37
  tokens, the recurrent state is smaller than the equivalent cache.
- At $T = 4{,}096$, the equivalent cache is 536.87 MB. The recurrent state
  stays at 4.85 MB.

Storing the state in half precision halves these figures. The state is kept
in float32 or wider for accuracy, so the figures above are the ones the code
uses.

---

## 18. Decoding, sampling, and chat state

### 18.1 The three entry points

- `Parentheses.forward_with_state(ids, states, prev_eot)` reads a sequence with
  the parallel form and returns logits for every position and the state after
  the last token. Use it for prompts and for scoring.
- `Parentheses.step(token, states, boundary)` takes one token per row and
  returns the next-token logits and the new state. Use it for generation.
  Set `boundary=True` for the token that follows `<|endoftext|>`.
- `ChatSession` wraps these for conversations. It keeps the state between
  turns and exposes `add_user`, `add_assistant`, `feed`, `feed_one`, and
  `generate`.

### 18.2 Sampling in `ChatSession.generate`

`generate` runs these steps for each new token:

1. Block any token that would complete an n-gram already seen in the last 64
   generated tokens. The size is set by `no_repeat_ngram_size`.
2. If `temperature` is at most $10^{-4}$, take the argmax. Otherwise divide the
   logits by the temperature and keep the top `top_k` logits.
3. Turn the logits into probabilities with softmax. If `min_p` is set, drop every
   token whose probability is below `min_p` times the largest probability, and
   renormalize.
4. Sample from the result with `torch.multinomial`.

The release settings are in `RELEASE_SAMPLING`: temperature 0.2, `top_k` 40,
`min_p` 0.1, and an 8-gram block. There is no repetition penalty. The terminal
chat uses the same values as defaults.

The terminal chat holds back a trailing replacement character until the next
token completes the UTF-8 sequence, so an Indic character is never printed in
pieces.

### 18.3 Features from 1.0 that are not in this repository

The 1.0 code had several runtime features that 1.1 does not include. Their code
is not in this repository.

- Free Think mode let the model keep generating in an open-ended "thinking"
  mode, with a question gate. It used the same recurrent step.
- Modular Free Think spliced retrieved text into the decode loop. It split a
  local document collection into chunks, indexed the chunks with BM25, and
  inserted the best matches every 64 generated tokens. The 1.0 notes gave the
  splice budget as half the block size for retrieved text and a quarter for the
  running thought.
- A conversation memory module that carried context across sessions. 1.1 keeps
  context inside `ChatSession` instead.
- A visual prefill read image patches into the recurrent state. The 1.0 notes
  say the visual projection was not trained jointly with the language model.
- A local Web Studio server wrapped the 0.9 model.

Image input, speech output, and speech input are planned for 1.1 (see
[ROADMAP.md](ROADMAP.md)). None of them exist in this code yet.

---

## 19. Known limits

1. Finite associative recall. Every head compresses its history into one
   $64 \times 64$ matrix. The matrix superposes older keys and forgets them on
   the decay schedule. It does not store tokens verbatim. Exact recall of an
   arbitrary string from far back in the context needs an external retrieval
   step. The parametric state is not enough for that.
2. No alignment rule for lengths. `forward` and `forward_with_state` pad any
   length to a multiple of 64 and switch the padding off, so a sequence of any
   length can be read in one call. The 1.0 notes required lengths that were a
   multiple of 64, and that rule no longer applies.
3. No server in this repository. The package provides the terminal chat and the
   Python API. A web or HTTP server is not included.
4. The NPU step graph is not in this repository. Its boundary behavior is in
   [Section 4](#4-static-shape-step-graph-for-npus). The host must zero the state
   at each new conversation.
5. Arithmetic without tools. The model is not a calculator. Put a deterministic
   evaluator or a code-execution tool in front of any workflow that depends on
   exact arithmetic. Benchmark results on arithmetic are in
   [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) and [HISTORY.md](../HISTORY.md).
6. Experimental languages. Kannada, Tamil, Telugu, and Malayalam are the
   anchor languages. English is also supported. Tulu (`tcy`), Koya (`kff`), Kuvi
   (`kxv`), and Kolami (`nit`) are experimental: their public validation text is
   small, and their training text under a commercial licence is scarce. The
   terminal chat never picks them automatically.
7. CUDA graph decoding needs CUDA. `GraphStepper` raises an error on other
   devices. Eager `step()` works everywhere.

---

## 20. Appendix: shapes, tests, source map, and licenses

### 20.1 Tensor shapes

`B` is the batch size, `T` the sequence length, `D = 1024`, `H = 16`, `Dh = 64`,
`V = 32768`.

| Tensor | `forward_with_state` | `step` |
|---|---|---|
| `ids` / `token` | `[B, T]` int64 | `[B]` int64 |
| Residual stream `x` | `[B, T, D]` | `[B, D]` |
| `u = qkv(x)` and the conv output | `[B, T, 3D]` | `[B, 3D]` |
| `q`, `k`, `v` after the split | `[B, T, H, Dh]` | `[B, H, Dh]` |
| `beta`, `log_alpha` | `[B, T, H]` | `[B, H]` |
| `memory` | `[B, H, Dh, Dh]`, wide dtype | `[B, H, Dh, Dh]`, wide dtype |
| Conv state | `[B, K-1, 3D]` | `[B, K-1, 3D]` |
| Token-shift state | `[B, D]` | `[B, D]` |
| `logits` | `[B, T, V]` | `[B, V]` |

### 20.2 Test inventory

| File | Lines | What it checks |
|---|---:|---|
| `tests/test_slab_v4.py` | 258 | Parallel form equals the recurrent step; prefill then step; prefill in pieces; document isolation; constant state size; `stream` matches a manual greedy loop; finite gradients; release parameter count, forward and step against a reference (needs the environment variables in [Section 6](#6-benchmarks-and-verification)) |
| `tests/test_session.py` | 114 | Turns and generation match one pass; state is carried, not reset; `<|endoftext|>` inside a feed and at a turn end; the stop token is not written; turn format; three release turns match a re-read |
| `tests/test_graph_step.py` | 70 | CUDA graph output equals eager mode, with documents, after a prefill, and on the release weights |
| `tests/test_chat.py` | 43 | Language detection by script; the n-gram block; generation never repeats an n-gram; `min_p` of 1 is greedy |
| `tests/test_lm_eval_adapter.py` | 96 | Log-likelihood matches manual stepping; batching does not change scores; rolling equals the sum of token log-probabilities; `generate_until` is greedy and stops; tokenizer round trip |
| `tests/conftest.py` | 4 | Test setup |

### 20.3 Source map

| File | Lines | Contents |
|---|---:|---|
| `python/parentheses/model.py` | 420 | `chunk_scan`, `find_documents`, `LayerState`, `RMSNorm`, `SwiGLU`, `SelectiveLinearAttention`, `Block`, `Parentheses` |
| `python/parentheses/config.py` | 44 | `SlabConfig` and `SLAB_V4_240M` |
| `python/parentheses/checkpoint.py` | 96 | `load_state`, `load_model`, `save_safetensors`, `sha256_file` |
| `python/parentheses/session.py` | 171 | `ChatSession`, `RELEASE_SAMPLING`, n-gram blocking |
| `python/parentheses/graph_step.py` | 105 | `GraphStepper` |
| `python/parentheses/chat.py` | 175 | Terminal chat, language detection, device choice, checkpoint lookup |
| `python/parentheses/selftest.py` | 113 | CPU-versus-device check |
| `python/parentheses/evaluate.py` | 105 | Harness runner and result files |
| `python/parentheses/lm_eval_adapter.py` | 146 | `ParenthesesLM` |
| `tools/check_public_tree.py` | 81 | Leak checker run by the git hooks |

### 20.4 Licenses

The code is MIT licensed, and the released weights are Apache 2.0. The training
data licence tiers, including the exclusion of the `esg` language from the
release training, are in [LICENSE_TABLE.md](LICENSE_TABLE.md).
