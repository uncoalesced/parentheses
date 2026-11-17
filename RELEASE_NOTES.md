# Release notes

## Unreleased: 1.1 in development

This repository is the start of 1.1. So far it has:

- A clean Python rewrite of the 1.0 model code in `python/parentheses/`. Its outputs match 1.0 exactly.
- Prefill-then-step generation. The prompt is read in one pass, then each new token is one recurrent step.
- `ChatSession`, which carries memory between chat turns without re-reading the conversation.
- A CUDA-graph single-token step. Its outputs are identical to the plain step.
- `python -m parentheses.chat` for chat and `python -m parentheses.selftest` for the self-test.
- An lm-evaluation-harness adapter and runner (`python -m parentheses.evaluate`).
- A new benchmark protocol in [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md).
- A repository history reset in November 2026 to remove private names and paths.

Setup for 1.1 is in [docs/SETUP.md](docs/SETUP.md).

---

## parentheses-1.0-240M-tanzanite

**Release:** `v1.0.0-tanzanite`, November 2026
**Model:** `parentheses-1.0-240M` (253,544,176 parameters)
**Architecture:** recurrent State Space Duality / selective linear attention (`slab_v4`). It is not a transformer: there is no softmax attention and no KV-cache.
**Licenses:** weights Apache 2.0, code MIT
**Languages:** Kannada (`kan`), Tamil (`tam`), Telugu (`tel`), Malayalam (`mal`) and English (`eng`) as primary languages; Tulu (`tcy`), Koya (`kff`), Kuvi (`kxv`) and Kolami (`nit`) as experimental.

We grew this model from our 120M proof-of-concept on one 8 GB laptop GPU (NVIDIA GeForce RTX 5050 Laptop). It reads one token at a time and keeps a fixed-size memory, so generation uses the same 4.85 MB of state at token 16,000 as at token 1.

---

### 1. What is in this release

#### Weights and where to get them

| File | Format | Size | Where |
|:---|:---|:---|:---|
| `model.safetensors` | float32 | 1,014,203,240 bytes | Hugging Face Hub: `uncoalesced-org/parentheses-1.0-240M` |
| `model-bf16.safetensors` | bfloat16 | 507,115,040 bytes | GitHub release `v1.0.0-tanzanite` |
| `model.pt` | PyTorch state dict | 1,014,250,863 bytes | GitHub release `v1.0.0-tanzanite` |
| `parentheses_240m_v4_step.onnx` | ONNX recurrent step graph | 1,148,704,972 bytes | GitHub release `v1.0.0-tanzanite` |

The GitHub release also has `onnx_metadata.json`, `parity_report.json` and `MANIFEST_sha256.tsv`. SHA-256 hashes for every file are in `MANIFEST_sha256.tsv`. The float32 weights hash is `bff69c7945533edaf379e69ffc50e47ed3b455c83cafb312defbd1b4c2208a6e`.

The float32 file is the reference copy. The bf16 file is an optional smaller download that has not been tested separately, so start with the float32 one.

#### Loading through transformers

The Hub repository includes `modeling_parentheses.py` and `configuration_parentheses.py`, so the model loads with `AutoModelForCausalLM` and `trust_remote_code=True`. Only `torch` and `transformers` are needed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "uncoalesced-org/parentheses-1.0-240M"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
print(model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng"))
```

`model.chat()` and `model.stream()` use the constant-memory recurrent path. `model.generate()` also works, but it has no cache and re-reads the whole sequence at each new token. The Hugging Face port was checked against the training code to within 1e-5 on the logits, using a test in the old 0.9 and 1.0 repository (`tests/test_hf_automodel.py`, not in this repository).

#### Running from the repository

```bash
python -m parentheses.chat --checkpoint weights --device auto
```

This is the 1.1 command. Setup steps are in [docs/SETUP.md](docs/SETUP.md). The 1.0 release was first run with the old `scripts/chat.py` and `release/parentheses-1.0-240M-tanzanite`, which are not in this repository.

The 1.1 chat command uses the sampling settings 1.0 was tuned with: temperature 0.2, top-k 40, min-p 0.1 and an 8-gram loop block. The 0.9 defaults (identity system prompt, repetition penalty 1.12) broke the 1.0 model. It drifted into identity answers and broke simple code, because the penalty stopped it from reusing variable names. That is why no repetition penalty is set.

---

### 2. Architecture (`slab_v4`)

| Setting | Value |
|:---|:---|
| Layers | 16 |
| Model width | 1024 |
| Heads | 16 heads of dimension 64 |
| SwiGLU hidden width | 2752 |
| Vocabulary | 32,768 (byte-level BPE) |
| Training context | 4,096 tokens |
| Chunk size for the parallel scan | 64 |
| Recurrent state per sequence | 4,849,664 bytes in float32: 16 layers x (16x64x64 state + 3x3072 conv buffer + 1024 shift buffer) |

The tokenizer is byte-level BPE. Most splits fall on akshara boundaries, but BPE can split an akshara before a vowel sign inside a word. An akshara is not guaranteed to stay whole.

Each layer updates its memory with a gated decay and a gated write:

$$S_t = \alpha_t S_{t-1} + \beta_t\,(k_t v_t^\top), \qquad y_t = q_t S_t$$

Main pieces:

- **Token shift.** Each position mixes in the previous token's activation before the gates are computed.
- **Causal depthwise convolution** (kernel 4) on the query, key and value projections.
- **Layer-0 value highway.** Every later layer blends its values with the values from layer 0 through a learned per-head gate.
- **Document isolation.** The token after `<|endoftext|>` resets the state, the token shift and the convolution taps, so documents packed into one training row never leak into each other.
- **Two equivalent forms.** Training uses a chunked parallel scan (chunk size 64) with float32 accumulation between chunks. Inference uses the one-token recurrent step. Both compute the same function.

---

### 3. Benchmarks

The benchmark tables published with this release overstated the model, so they have been removed from these notes; see [the benchmark tables](HISTORY.md#the-benchmark-tables-november-2026) in HISTORY.md.

---

### 4. Memory

Generation keeps the recurrent state at 4,849,664 bytes in float32 at any length. Nothing grows with the context, so there is no KV-cache to manage.

---

### 5. DirectML and AMD Ryzen AI NPU export

`parentheses_240m_v4_step.onnx` is a static-shape recurrent step graph: one token and the previous state in, logits and the next state out. Measured against the PyTorch reference over 32 sequential decoding steps (`parity_report.json`):

- CPU execution provider: max logit error 3.75e-5
- DirectML execution provider: max logit error 5.48e-5

We have not measured NPU power draw or NPU throughput yet. The graph targets AMD Ryzen AI NPUs through DirectML.

---

### 6. Known limits

- **Short Python functions work best.** Factorial, palindrome and Fibonacci functions come out correct with the default settings. Longer programs are less reliable.
- **Dravidian answers are shallow.** They stay in the right script and on topic, but are often brief and can repeat themselves.
- **English factual answers are weak.** This is a 253M model, so check facts before relying on them.
- **The `<think>` reasoning format is experimental.** On this checkpoint, forcing a `<think>` block gives a templated restatement of the question rather than useful reasoning. Use the plain chat format.
- **Math is weak.** The old `scripts/chat.py` answered simple arithmetic with an exact calculator instead of the model. The 1.1 chat command has no such step.

---

### 7. Other changes in this release

These changes were made in the 0.9 and 1.0 repository, where the files below lived. None of these paths exist in this repository.

- `model/sampling.py`: one shared sampler for the 0.9 and 1.0 models. `SlabV4Model.stream()` gave the 1.0 model the same streaming interface, so chat, FreeThink and the multimodal features ran on it through one loader.
- `features/multimodal.py`: the visual state prefill worked with the 1.0 model, whose per-layer state is allocated on the first step.
- `tests/eval_heldout_release.py`: a `release_240m` entry and a `--limit` flag for quick checks. Its held-out results are not reported here.
- `scripts/stage_release_240m.py`: built the Hub upload folder (float32 only) and the GitHub release folder, then checked every file against the manifest.

---

## Historical lineage: parentheses-0.9-120M release notes

**Release date:** October 17, 2026
**Model:** parentheses 0.9-120M (`119,702,272` unique parameters)
**Architecture:** recurrent selective linear attention / State Space Duality (SSD). It is not a transformer.
**Languages supported (9):** Kannada (`kan`), Tamil (`tam`), Telugu (`tel`), Malayalam (`mal`), Tulu (`tcy`), Koya (`kff`), Kuvi (`kxv`), Naikpod / Kolami (`nit`), and English (`eng`)

> The code named in this section (`features/`, `chat.py`, `app.py`, the `/think` and `/freethink` commands) belongs to the 0.9 repository and is not in this repository. Commands below were run against that code.

The 0.9 model is not a transformer. It has no quadratic softmax attention matrices and no growing KV-cache.

- **Parallel training form:** chunked associative linear attention (chunk size 64) with closed-form decay between chunks and `reset_mask` document-boundary isolation. Cost is linear in sequence length.
- **Recurrent inference form:** one matrix state update per token, $S_t = \alpha_t S_{t-1} + (k_t \otimes v_t)$, $y_t = q_t S_t$. With 32 layers, 8 heads, 64 x 64 state and 4 bytes per value, the fixed footprint is about 4.19 MB regardless of sequence length.

---

### 1. Highlights of the 0.9-120M release

1. **1.25 billion token pretraining and masked SFT v4 (`step_10036`):**
   - Grown from our 120M proof-of-concept model across 5 continuous pretraining sessions (1,250,033,664 tokens, 9,536 steps) on a single NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM) with no driver crashes, followed by assistant-span-masked SFT v4 (`step_10036`, 1,744 validation documents, 536,117 tokens evaluated). Best validation loss was 1.9777 (PPL 7.23).
2. **Internal thinking and FreeThink commands:** `/think` and `/freethink` (and the other mode commands) are stripped from user-visible text and handled internally, mapped to the internal `</freethink>` closing tag.
   - All mode-switching commands (`/think`, `/freethink`, `/free-think`, `/nothink`, `/no-think`, `/chat`, `/base`) and control tags (`<think>`, `</think>`, `<freethink>`, `</freethink>`) are removed from visible text.
3. **Structured reasoning and a constant-memory repetition gate:**
   - After the `<|assistant|><think>` boundary, a state re-anchor (`STRUCTURED_THINK_ANCHOR`) across all 9 languages keeps the topic and native-script grounding.
   - A constant-memory 64-token ring-buffer repetition penalty (`repetition_penalty=1.16`) and a 5-gram blocker (`no_repeat_ngram_size=5`, `min_p=0.06`) removed phrase loops (0.0% loop rate on `reason250`) without corrupting subwords.
4. **Recurrent SSM multimodal subsystem (`features/multimodal.py`):**
   - Constant-memory 2D image patch state prefill (`prefill_visual_state`), grounded visual attribute summaries (`describe_image`), and pure-Python document extraction (`extract_document_text`) for PDF, DOCX, HTML, XML, CSV, JSON, TXT and Markdown.
5. **Copyright and linguistic hygiene:**
   - Aheri Gondi (`esg`, © Wycliffe, research-only) is excluded from all training and SFT binaries (`esg_count: 0`).

---

### 2. Internal thinking and FreeThink commands (0.9 code only)

parentheses 0.9-120M has four operating modes (`think`, `chat`, `freethink`, `base`). All mode commands and tags are kept internal:

- **Command routing (`parse_internal_command`):** a prompt that starts with `/think`, `/freethink`, `/free-think`, `/nothink`, `/no-think`, `/chat` or `/base` routes the session to that mode. The command string is stripped before tokenization and display.
- **Internal `</freethink>` closure:** when a FreeThink session (`FreeThinkSession` or `ModularFreeThinkSession`) finishes streaming, it closes internally with:
  - `sess.internal_open_tag == "<freethink>"`
  - `sess.internal_close_tag == "</freethink>"`
  - `sess.closed_freethink == True`
  - `sess.internal_transcript == "<freethink>\n{clean_body}\n</freethink>"`
- **No visible tag leakage (`strip_internal_tags`):** raw `<think>`, `</think>`, `<freethink>`, `</freethink>`, `/think` and `/freethink` markers are not written to stdout, the terminal chat, the Web Studio stream or the HTML demo cards.

---

### 3. Recurrent SSM multimodal capabilities (0.9 code only)

The 0.9 code processed 2D images and multi-format documents in `features/multimodal.py` without a Vision Transformer, self-attention or a growing KV-cache:

- **Image patch state prefill (`SSMVisualProjector`, `prefill_visual_state`):** slices an image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`, `.tiff`) into non-overlapping 2D raster patches, projects them into `n_embd` with a deterministic projection derived from `model.tok_emb.weight` plus 2D sinusoidal positional embeddings, and steps each patch through `block.step(x_p, state)` in constant state memory of shape `(B, n_head, head_dim, head_dim)`.
- **Visual attribute descriptor (`describe_image`):** computes resolution, aspect ratio, colour mode, dominant RGB and HSL palette names, Rec.601 luminance, RMS and Michelson contrast, Sobel edge density, a 2 x 2 quadrant layout, and EXIF or PNG metadata, in all 9 languages.
- **Document ingestion (`extract_document_text`):** pure-Python UTF-8 and Dravidian script (`Knda`, `Taml`, `Telu`, `Mlym`, `Orya`) text extraction from PDF, DOCX, HTML and XML, CSV and TSV, JSON and JSONL, and plain text or code files.

Python API example (0.9 code):

```python
from features.multimodal import prefill_visual_state, describe_image, extract_document_text

# 1. Step 2D image patches sequentially into the constant-size SSM state S_t
states = prefill_visual_state(model, "photo.png", patch_size=16, grid_size=64)

# 2. Generate a grounded visual summary (supports all 9 language codes)
vis_desc = describe_image("photo.png", lang="eng")
print(vis_desc.summary)

# 3. Extract clean UTF-8 / Dravidian text from PDF, DOCX, HTML, CSV, JSON, or TXT
doc_text = extract_document_text("paper.pdf")
```

Command-line use (0.9 `chat.py`):

```bash
# Pre-load an image and PDF document at startup
python chat.py --lang eng --format think --image photo.png --doc paper.pdf

# Or attach images and documents inline during a live session:
#   /image photo.png Analyze the visual layout and contrast
#   /doc paper.pdf Summarize the methodology section
```

Web Studio (0.9 `app.py`):

- Run `python app.py` and use the attach button to upload images or documents into the active session.
- REST endpoint: `POST /api/ingest` with `{"filename": "paper.pdf", "data_b64": "...", "lang": "eng"}`.

---

### 4. Verified results (SFT v4 final, `step_10036`)

These are the 0.9 evaluation results as recorded at the time. They were measured on the 0.9 code and have not been rerun under the 1.1 protocol.

#### Core reasoning, script fidelity and language routing

| Benchmark / evaluation suite | Metric | 41M-v4 (step 8082) | 120M pretrain (9536) | 120M SFT v2 (9786) | 120M SFT v4 final (10036) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Masked SFT validation loss (PPL)** | CE loss (PPL) | - | 2.2179 (9.19) | 2.0722 (7.94) | **1.9777 (7.23)** |
| **Structured `<think>` validation loss** | CE loss (PPL) | - | 1.6851 (5.39) | 1.4466 (4.25) | **1.4018 (4.06)** |
| **8-language Dravidian tag accuracy** | `tag_acc` | - | 91.25% | 93.75% | **94.14%** |
| **Main-4 Dravidian mean BPB** | `bpb` (lower is better) | 0.5203 | **0.4896** | 0.4909 | **0.4920 (+0.0024)** |
| **`reason250` native script fidelity** | `script_ok` | - | - | 99.6% | **99.6%** |
| **`reason250` degenerate loop rate** | `loop_rate` | - | - | 0.0% | **0.0%** |
| **`reason250` logic exact match** | `exact_match` | - | - | 20.0% | **20.0%** |
| **Multi-choice reasoning (`mc_reason`)** | `acc_norm` | 0.3333 | 0.3704 | 0.4074 | **0.3868** |
| **FreeThink 9-language suite** | Script OK / leaks | - | - | 100% / 0 | **100% (72/72) / 0 leaks** |
| **Empirical SSM O(1) latency ratio** | L_last / L_first | 1.00x | 1.00x | 1.010x | **1.006x (flat)** |

#### Supervised fine-tuning loss and perplexity across stages

| Stage / checkpoint | Steps | Masked SFT loss (PPL) | Unmasked CE loss (PPL) | Supervised fraction | Evaluated docs / tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base pretrain** | 9,536 | 2.2179 (9.19) | 2.4039 (11.07) | 60.5% | 200 / 53,271 |
| **SFT v1** | 9,636 | 2.1384 (8.49) | 2.3683 (10.68) | 60.5% | 200 / 53,271 |
| **SFT v2** | 9,786 | 2.0722 (7.94) | 2.3354 (10.33) | 60.5% | 200 / 53,271 |
| **SFT v3** | 9,886 | 1.9882 (7.30) | 2.3359 (10.34) | 55.0% | 1,744 / 536,117 |
| **SFT v4 final** | **10,036** | **1.9777 (7.23)** | **2.3352 (10.33)** | 55.0% | **1,744 / 536,117** |
| **Net delta** | **+500 steps** | **-0.2402 (-1.96 PPL)** | **-0.0687 (-0.74 PPL)** | - | **Best SFT loss** |

#### Per-category masked SFT loss and perplexity (SFT v4 final vs. base)

| Category | Base (9536) loss (PPL) | SFT v2 (9786) loss (PPL) | SFT v4 final (10036) loss (PPL) | Net delta |
| :--- | :---: | :---: | :---: | :---: |
| **`structured_think`** | 1.6851 (5.39) | 1.4466 (4.25) | **1.4018 (4.06)** | **-0.2833** |
| **`drav_chat`** | 3.0476 (21.06) | 3.0211 (20.51) | **2.9289 (18.71)** | **-0.1187** |
| **`eng_chat`** | 2.5318 (12.58) | 2.4580 (11.68) | **2.4074 (11.11)** | **-0.1244** |
| **`lowres_chat`** | 2.2911 (9.89) | 2.1451 (8.54) | **1.9992 (7.38)** | **-0.2919** |
| **`freethink_cont`** | 3.2924 (26.91) | 3.2089 (24.75) | **2.6242 (13.79)** | **-0.6682** |
| **`tool_hermes`** | 0.9369 (2.55) | 0.8811 (2.41) | **0.4601 (1.58)** | **-0.4768** |

#### Held-out validation bits per byte (BPB) and tag routing accuracy (all 9 languages)

| Language | ISO | Script | 41M baseline | 120M pretrain (9536) | 120M SFT v4 final (10036) | Tag accuracy | Tokens/byte |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Kannada** | `kan` | Kannada (`Knda`) | 0.522 | **0.4878** | 0.4899 | **100.0%** (32/32) | 0.0998 |
| **Tamil** | `tam` | Tamil (`Taml`) | 0.492 | **0.4674** | 0.4698 | **93.75%** (30/32) | 0.0953 |
| **Telugu** | `tel` | Telugu (`Telu`) | 0.623 | **0.5916** | 0.5944 | **93.75%** (30/32) | 0.1170 |
| **Malayalam** | `mal` | Malayalam (`Mlym`) | 0.444 | **0.4116** | 0.4135 | **100.0%** (32/32) | 0.0935 |
| **Main-4 Dravidian mean** | - | 4 anchor scripts | 0.5203 | **0.4896** | **0.4920** | **96.88%** | 0.1014 |
| **Tulu** | `tcy` | Kannada (`Knda`) | 0.652 | **0.6399** | 0.6479 | **96.88%** (31/32) | 0.1215 |
| **Koya** | `kff` | Odia (`Orya`) | 0.525 | **0.5391** | 0.5498 | **90.63%** (29/32) | 0.1351 |
| **Kuvi** | `kxv` | Odia (`Orya`) | 0.453 | **0.4560** | 0.4606 | **93.75%** (30/32) | 0.1362 |
| **Kolami / Naikpod** | `nit` | Telugu (`Telu`) | 0.718 | **0.7235** | 0.7415 | **84.38%** (27/32) | 0.1393 |
| **8-language Dravidian mean** | - | 8 languages | - | 91.25% | **94.14%** (mean tag acc) | **94.14%** | - |
| **English** | `eng` | Latin | 1.653 | **1.3929** | 1.4533 | - | 0.3934 |

---

### 5. Licensing (0.9)

- **Software code and architecture:** [MIT License](LICENSE)
- **Model weights (0.9-120M and 0.9-130M-v3):** CC BY-NC 4.0, non-commercial use only. The weights are not under [LICENSE-MODEL](LICENSE-MODEL), which is the Apache 2.0 text for the 1.0 weights.
- **Why non-commercial:** some instruction data used in 0.9 carried non-commercial licences, so the weights were released under the same terms. See [the licence rule](HISTORY.md#the-licence-rule-2-november-2026) in HISTORY.md.
