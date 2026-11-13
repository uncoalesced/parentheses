# parentheses-1.0-240M-tanzanite: release notes

**Release:** `v1.0.0-tanzanite`, November 2026
**Model:** `parentheses-1.0-240M` (253,544,176 parameters)
**Architecture:** recurrent State Space Duality / selective linear attention (`slab_v4`). It is not a transformer: there is no softmax attention and no KV-cache.
**Licenses:** weights Apache 2.0, code MIT
**Languages:** Kannada (`kan`), Tamil (`tam`), Telugu (`tel`), Malayalam (`mal`) and English (`eng`) as primary languages; Tulu (`tcy`), Koya (`kff`), Kuvi (`kxv`) and Kolami (`nit`) as experimental.

We trained this model from scratch on one 8 GB laptop GPU (NVIDIA GeForce RTX 5050 Laptop). It reads one token at a time and keeps a fixed-size memory, so generation uses the same 4.85 MB of state at token 16,000 as at token 1.

---

## 1. What is in this release

### Weights and where to get them

| File | Format | Size | Where |
|:---|:---|:---|:---|
| `model.safetensors` | float32 | 1,014,203,240 bytes | Hugging Face Hub: `uncoalesced-org/parentheses-1.0-240M` |
| `model-bf16.safetensors` | bfloat16 | 507,115,040 bytes | GitHub release `v1.0.0-tanzanite` |
| `model.pt` | PyTorch state dict | 1,014,250,863 bytes | GitHub release `v1.0.0-tanzanite` |
| `parentheses_240m_v4_step.onnx` | ONNX recurrent step graph | 1,148,704,972 bytes | GitHub release `v1.0.0-tanzanite` |

All benchmark numbers below were measured on the float32 weights. The bf16 file is an optional smaller download and was not separately benchmarked. SHA-256 hashes for every file are in `MANIFEST_sha256.tsv`. The float32 weights hash is `bff69c7945533edaf379e69ffc50e47ed3b455c83cafb312defbd1b4c2208a6e`.

### Loading through transformers

The Hub repository includes `modeling_parentheses.py` and `configuration_parentheses.py`, so the model loads with `AutoModelForCausalLM` and `trust_remote_code=True`. Only `torch` and `transformers` are needed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "uncoalesced-org/parentheses-1.0-240M"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
print(model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng"))
```

`model.chat()` and `model.stream()` use the constant-memory recurrent path. `model.generate()` also works, but it has no cache and re-reads the whole sequence at each new token. The Hugging Face port matches the training code to within 1e-5 on the logits; `tests/test_hf_automodel.py` checks this.

### Running from the repository

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda
```

For this model, `chat.py` now defaults to no system prompt, temperature 0.2, no repetition penalty and an 8-gram loop block. That matches how the release was tuned. With the older 0.9 defaults (identity system prompt, repetition penalty 1.12), the model drifted into identity answers and broke simple code, because the penalty stopped it from reusing variable names.

---

## 2. Architecture (`slab_v4`)

| Setting | Value |
|:---|:---|
| Layers | 16 |
| Model width | 1024 |
| Heads | 16 heads of dimension 64 |
| SwiGLU hidden width | 2752 |
| Vocabulary | 32,768 (byte-level BPE with akshara-preserving splits) |
| Training context | 4,096 tokens |
| Chunk size for the parallel scan | 64 |
| Recurrent state per sequence | 4,849,664 bytes in float32: 16 layers x (16x64x64 state + 3x3072 conv buffer + 1024 shift buffer) |

Each layer updates its memory with a gated decay and a gated write:

$$S_t = \alpha_t S_{t-1} + \beta_t\,(k_t v_t^\top), \qquad y_t = q_t S_t$$

Main pieces:

- **Token shift.** Each position mixes in the previous token's activation before the gates are computed.
- **Causal depthwise convolution** (kernel 4) on the query, key and value projections.
- **Layer-0 value highway.** Every later layer blends its values with the values from layer 0 through a learned per-head gate.
- **Document isolation.** The token after `<|endoftext|>` resets the state, the token shift and the convolution taps, so documents packed into one training row never leak into each other.
- **Two equivalent forms.** Training uses a chunked parallel scan (chunk size 64) with float32 accumulation between chunks. Inference uses the one-token recurrent step. Both compute the same function.

---

## 3. External benchmarks (9 public datasets)

Four runs per benchmark: run 1 greedy (T=0.0), runs 2 to 4 sampled at T=0.7 with different seeds. Each benchmark uses a small sample of a few dozen items, so read the best and average columns together: on a sample this small, one lucky run moves the best score a lot.

| Benchmark | Domain | Best | Average | Peer baseline |
|:---|:---|:---:|:---:|:---|
| TerminalBench | Bash and CLI generation | 78.33% | 74.16% | Harbor CLI baseline, 66.00% |
| TruthfulQA | Truthfulness | 91.00% | 90.00% | OpenAI Instruct-Ada (350M), 84.50% |
| MBPP | Python function synthesis | 90.67% | 81.92% | Salesforce CodeGen-350M Mono, 82.20% |
| HaluEval QA | Hallucination detection | 80.00% | 78.34% | OpenAI InstructGPT, 68.20% |
| MMLU (STEM subset) | Academic knowledge | 50.00% | 30.00% | SmolLM-135M, 30.20% |
| HellaSwag | Commonsense continuation | 33.33% | 24.16% | SmolLM-360M, 42.80% |
| ARC-Challenge | Science QA | 26.67% | 16.67% | SmolLM-360M, 34.20% |
| Belebele (Dravidian) | Reading comprehension | 25.00% | 17.18% | BLOOM-560M, 31.25% |
| GSM8K | Grade-school math | 10.00% | 5.83% | MobileLLM-350M, 13.20% |

We are ahead of the peer baseline on TerminalBench, TruthfulQA, MBPP and HaluEval QA. We are behind on HellaSwag, ARC-Challenge, Belebele and GSM8K. On MMLU our average matches SmolLM-135M. Per-run numbers and citations are in `BENCHMARK_SPEC.md`.

---

## 4. Memory compared with 1B-scale models

At 4,096 tokens of context, transformer KV-caches for 1B-scale models take 536.9 MB (TinyLlama-1.1B), 1,258.3 MB (OPT-1.3B) and 1,440.0 MB (Pythia-1.4B). Our recurrent state stays at 4.85 MB at any length. That is 99.1% less than TinyLlama-1.1B at 4,096 tokens, and the gap grows with context.

On the benchmarks above, the 253M model scores higher than TinyLlama-1.1B, OPT-1.3B and Pythia-1.4B on MMLU, GSM8K, TruthfulQA and HaluEval QA, and lower on ARC-Challenge. The full table is in `BENCHMARK_SPEC.md`, section 3.

---

## 5. Internal validation (300 held-out prompts)

| Check | Result |
|:---|:---|
| Clean stop at `<|endoftext|>` | 95.67% (287/300) |
| Internal tag leaks | 0.00% (0/300) |
| Dravidian script purity | 100.00% (80/80) |
| Dravidian format conformance | 96.25% (77/80) |
| Abstention on held-out trap questions | 94.00% (94/100) |
| False refusals on ordinary questions | 0.00% (0/90) |
| Multilingual held-out bits-per-byte drift | +0.012 |
| Recurrent state over 16,384 tokens | stable, no NaN or drift |

---

## 6. DirectML and AMD Ryzen AI NPU export

`parentheses_240m_v4_step.onnx` is a static-shape recurrent step graph: one token and the previous state in, logits and the next state out. Measured against the PyTorch reference over 32 sequential decoding steps (`parity_report.json`):

- CPU execution provider: max logit error 3.75e-5
- DirectML execution provider: max logit error 5.48e-5

We have not measured NPU power draw or NPU throughput yet. The graph targets AMD Ryzen AI NPUs through DirectML.

---

## 7. Known limits

- **Short Python functions work best.** Factorial, palindrome and Fibonacci functions come out correct with the default settings. Longer programs are less reliable.
- **Dravidian answers are shallow.** They stay in the right script and on topic, but are often brief and can repeat themselves.
- **English factual answers are weak.** This is a 253M model, so check facts before relying on them.
- **The `<think>` reasoning format is experimental.** On this checkpoint, forcing a `<think>` block gives a templated restatement of the question rather than useful reasoning. Use the plain chat format.
- **Math is weak.** GSM8K average is 5.83%. `chat.py` answers simple arithmetic with an exact calculator instead of the model.
- **Speed.** On the RTX 5050 Laptop GPU we measured 11 to 34 tokens per second with the Python recurrent loop, typically 20 to 27.

---

## 8. Other changes in this release

- `model/sampling.py`: one shared sampler for the 0.9 and 1.0 models. `SlabV4Model.stream()` gives the 1.0 model the same streaming interface, so chat, FreeThink and the multimodal features run on it through one loader.
- `features/multimodal.py`: the visual state prefill works with the 1.0 model, whose per-layer state is allocated on the first step.
- `tests/eval_heldout_release.py`: a `release_240m` entry and a `--limit` flag for quick checks.
- `scripts/stage_release_240m.py`: builds the Hub upload folder (float32 only) and the GitHub release folder, then checks every file against the manifest.

---

# Historical lineage: parentheses-0.9-120M release notes

**Release Date:** October 17, 2026  
**Model:** parentheses 0.9-120M (`119,702,272` unique parameters)  
**Architecture:** Recurrent Selective Linear Attention / State Space Duality (SSD) : **NOT a Transformer**  
**Languages Supported (9):** Kannada (`kan`), Tamil (`tam`), Telugu (`tel`), Malayalam (`mal`), Tulu (`tcy`), Koya (`kff`), Kuvi (`kxv`), Naikpod / Kolami (`nit`), and English (`eng`)

> [!IMPORTANT]
> **ARCHITECTURAL MANDATE (Mamba-2 / RWKV / Selective Linear Attention SSM):**
> parentheses 0.9-120M is **NOT a Transformer / GPT model**. It does **NOT** use quadratic $\mathcal{O}(T^2)$ Softmax attention matrices or growing KV-caches.
> - **Parallel Training Form:** Chunked ($C=64$) associative linear attention with closed-form inter-chunk decay propagation and `reset_mask` document-boundary isolation in linear $\mathcal{O}(T)$ complexity.
> - **Recurrent Inference Form:** Constant $\mathcal{O}(1)$ recurrent matrix state updates per step:
>   $$S_t = \alpha_t S_{t-1} + (k_t \otimes v_t), \quad y_t = q_t S_t$$
>   with a fixed state footprint of $32 \text{ layers} \times 8 \text{ heads} \times 64 \times 64 \times 4\text{B} \approx 4.19\text{ MB}$ regardless of sequence length.

---

## 1. Highlights of the 0.9-120M Release

1. **1.25 Billion Token Pretraining + Masked SFT v4 (`step_10036`):**
   - Trained from scratch across 5 continuous pretraining sessions (1,250,033,664 tokens, 9,536 steps) on a single NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM) with zero driver crashes, followed by assistant-span-masked SFT v4 (`step_10036`, 1,744 validation documents, 536,117 tokens evaluated, achieving best validation loss of **1.9777** / PPL 7.23).
2. **100% Internal Thinking & FreeThink Command Encapsulation (`/think`, `/freethink` $\to$ Internal `</freethink>`):**
   - All mode-switching commands (`/think`, `/freethink`, `/free-think`, `/nothink`, `/no-think`, `/chat`, `/base`) and control tags (`<think>`, `</think>`, `<freethink>`, `</freethink>`) are automatically stripped from user-visible text and managed strictly internally.
3. **Careful, Structured Reasoning & $\mathcal{O}(1)$ Repetition Gate:**
   - Post-`<think>` SSM state re-anchoring (`STRUCTURED_THINK_ANCHOR`) across all 9 languages preserves topic and native-script grounding after the `<|assistant|><think>` boundary.
   - Constant-memory 64-token ring-buffer repetition penalty (`repetition_penalty=1.16`) and 5-gram blocker (`no_repeat_ngram_size=5`, `min_p=0.06`) eliminate phrase loops (`0.0%` loop rate on `reason250`) without subword corruption.
4. **Recurrent SSM Multimodal Subsystem (`features/multimodal.py`):**
   - Constant-memory $\mathcal{O}(1)$ 2D image patch state prefill (`prefill_visual_state`), grounded visual attribute summaries (`describe_image`), and pure-Python multi-format document extraction (`extract_document_text` for PDF, DOCX, HTML, XML, CSV, JSON, TXT, MD).
5. **Strict Copyright & Linguistic Hygiene:**
   - Aheri Gondi (`esg`, © Wycliffe, research-only) is 100% excluded (`esg_count: 0` across all training and SFT binaries).

---

## 2. Internal Thinking & FreeThink Commands (`/think`, `/freethink` $\to$ Internal `</freethink>`)

parentheses 0.9-120M provides four operating modes (`think`, `chat`, `freethink`, `base`) while keeping all mode commands and tags **100% internal**:

- **Automatic Command Hiding & Routing (`parse_internal_command`):**
  - Prefixing any prompt with `/think`, `/freethink`, `/free-think`, `/nothink`, `/no-think`, `/chat`, or `/base` automatically routes the session to the target mode and strips the command string before tokenization and display.
- **Internal `</freethink>` Closure:**
  - When a FreeThink session (`FreeThinkSession` or `ModularFreeThinkSession`) finishes streaming, it closes internally with:
    - `sess.internal_open_tag == "<freethink>"`
    - `sess.internal_close_tag == "</freethink>"`
    - `sess.closed_freethink == True`
    - `sess.internal_transcript == "<freethink>\n{clean_body}\n</freethink>"`
- **Zero Visible Tag Leakage (`strip_internal_tags`):**
  - Raw `<think>`, `</think>`, `<freethink>`, `</freethink>`, `/think`, and `/freethink` markers are never emitted into visible stdout, terminal chat, Web Studio SSE streams, or HTML showcase cards.

---

## 3. Recurrent SSM Multimodal Capabilities (Image Patch State Prefill + Multi-Format Document Ingestion)

parentheses 0.9-120M processes both 2D images and multi-format documents natively in `features/multimodal.py` without Vision Transformer ($\mathcal{O}(N^2)$) self-attention or growing KV-caches:

- **Image Patch State Prefill (`SSMVisualProjector`, `prefill_visual_state`):** Slices images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`, `.tiff`) into non-overlapping 2D raster patches, projects them into `n_embd` via a deterministic Xavier/orthogonal projection derived from `model.tok_emb.weight` plus 2D sinusoidal positional embeddings, and steps each patch sequentially through `block.step(x_p, state)` in constant $\mathcal{O}(1)$ state memory `(B, n_head, head_dim, head_dim)`.
- **Visual Attribute Descriptor (`describe_image`):** Computes resolution, aspect ratio, color mode, dominant RGB/HSL palette names, Rec.601 luminance, RMS & Michelson contrast, Sobel edge density, $2\times2$ spatial quadrant layout, and EXIF/PNG metadata across all 9 languages.
- **Multi-Format Document Ingestion (`extract_document_text`):** Pure-Python UTF-8 and Dravidian script (`Knda`, `Taml`, `Telu`, `Mlym`, `Orya`) text extraction from **PDF** (`.pdf`), **Word** (`.docx`), **HTML/XML** (`.html`, `.htm`, `.xml`), **CSV/TSV** (`.csv`, `.tsv`), **JSON/JSONL** (`.json`, `.jsonl`), and plain text/code (`.txt`, `.md`, `.rst`, `.py`).

### Python API Example
```python
from features.multimodal import prefill_visual_state, describe_image, extract_document_text

# 1. Step 2D image patches sequentially into the constant O(1) SSM recurrent state S_t
states = prefill_visual_state(model, "photo.png", patch_size=16, grid_size=64)

# 2. Generate a grounded visual summary (supports all 9 language codes)
vis_desc = describe_image("photo.png", lang="eng")
print(vis_desc.summary)

# 3. Extract clean UTF-8 / Dravidian text from PDF, DOCX, HTML, CSV, JSON, or TXT
doc_text = extract_document_text("paper.pdf")
```

### CLI Multimodal Usage (`chat.py`)
```bash
# Pre-load an image and PDF document at startup
python chat.py --lang eng --format think --image photo.png --doc paper.pdf

# Or attach images and documents inline during a live session:
#   /image photo.png Analyze the visual layout and contrast
#   /doc paper.pdf Summarize the methodology section
```

### Web Studio Multimodal Usage (`app.py`)
- **Interactive UI:** Run `python app.py` and click **`📎 Attach Image / Document`** to upload images or documents into the active session.
- **REST Endpoint:** Call `POST /api/ingest` with `{"filename": "paper.pdf", "data_b64": "...", "lang": "eng"}`.

---

## 4. Verified Benchmark Results (SFT v4 Final `step_10036`)

### Core Reasoning, Script Fidelity & Language Routing Metrics
| Benchmark / Evaluation Suite | Metric | 41M-v4 (step 8082) | 120M Pretrain (9536) | 120M SFT v2 (9786) | 120M SFT v4 Final (10036) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Masked SFT Validation Loss (PPL)** | CE Loss (PPL) | - | 2.2179 (9.19) | 2.0722 (7.94) | **1.9777 (7.23)** |
| **Structured `<think>` Validation Loss** | CE Loss (PPL) | - | 1.6851 (5.39) | 1.4466 (4.25) | **1.4018 (4.06)** |
| **8-Language Dravidian Tag Accuracy** | `tag_acc` | - | 91.25% | 93.75% | **94.14%** |
| **Main-4 Dravidian Mean BPB** | `bpb` (lower=better) | 0.5203 | **0.4896** | 0.4909 | **0.4920 (+0.0024)** |
| **`reason250` Native Script Fidelity** | `script_ok` | - | - | 99.6% | **99.6%** |
| **`reason250` Degenerate Loop Rate** | `loop_rate` | - | - | 0.0% | **0.0%** |
| **`reason250` Logic Exact Match** | `exact_match` | - | - | 20.0% | **20.0%** |
| **Multi-Choice Reasoning (`mc_reason`)** | `acc_norm` | 0.3333 | 0.3704 | 0.4074 | **0.3868** |
| **FreeThink 9-Language Suite** | Script OK / Leaks | - | - | 100% / 0 | **100% (72/72) / 0 leaks** |
| **Empirical SSM $\mathcal{O}(1)$ Latency Ratio** | $L_\text{last} / L_\text{first}$ | 1.00× | 1.00× | 1.010× | **1.006× (Flat $\mathcal{O}(1)$)** |

### Supervised Fine-Tuning Loss & Perplexity Progression Across Stages
| Stage / Checkpoint | Steps | Masked SFT Loss (PPL) | Unmasked CE Loss (PPL) | Supervised Fraction | Evaluated Docs / Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Pretrain** | 9,536 | 2.2179 (9.19) | 2.4039 (11.07) | 60.5% | 200 / 53,271 |
| **SFT v1** | 9,636 | 2.1384 (8.49) | 2.3683 (10.68) | 60.5% | 200 / 53,271 |
| **SFT v2** | 9,786 | 2.0722 (7.94) | 2.3354 (10.33) | 60.5% | 200 / 53,271 |
| **SFT v3** | 9,886 | 1.9882 (7.30) | 2.3359 (10.34) | 55.0% | 1,744 / 536,117 |
| **SFT v4 Final** | **10,036** | **1.9777 (7.23)** | **2.3352 (10.33)** | 55.0% | **1,744 / 536,117** |
| **Net Delta** | **+500 steps** | **-0.2402 (-1.96 PPL)** | **-0.0687 (-0.74 PPL)** | - | **Best SFT Loss** |

### Per-Category Masked SFT Loss & Perplexity (SFT v4 Final vs. Base)
| Category | Base (9536) Loss (PPL) | SFT v2 (9786) Loss (PPL) | SFT v4 Final (10036) Loss (PPL) | Net Delta |
| :--- | :---: | :---: | :---: | :---: |
| **`structured_think`** | 1.6851 (5.39) | 1.4466 (4.25) | **1.4018 (4.06)** | **-0.2833** |
| **`drav_chat`** | 3.0476 (21.06) | 3.0211 (20.51) | **2.9289 (18.71)** | **-0.1187** |
| **`eng_chat`** | 2.5318 (12.58) | 2.4580 (11.68) | **2.4074 (11.11)** | **-0.1244** |
| **`lowres_chat`** | 2.2911 (9.89) | 2.1451 (8.54) | **1.9992 (7.38)** | **-0.2919** |
| **`freethink_cont`** | 3.2924 (26.91) | 3.2089 (24.75) | **2.6242 (13.79)** | **-0.6682** |
| **`tool_hermes`** | 0.9369 (2.55) | 0.8811 (2.41) | **0.4601 (1.58)** | **-0.4768** |

### Held-Out Validation Bits Per Byte (BPB) & Tag Routing Accuracy (All 9 Languages)
| Language | ISO | Script | 41M Baseline | 120M Pretrain (9536) | 120M SFT v4 Final (10036) | Tag Accuracy | Tokens/Byte |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Kannada** | `kan` | Kannada (`Knda`) | 0.522 | **0.4878** | 0.4899 | **100.0%** (32/32) | 0.0998 |
| **Tamil** | `tam` | Tamil (`Taml`) | 0.492 | **0.4674** | 0.4698 | **93.75%** (30/32) | 0.0953 |
| **Telugu** | `tel` | Telugu (`Telu`) | 0.623 | **0.5916** | 0.5944 | **93.75%** (30/32) | 0.1170 |
| **Malayalam** | `mal` | Malayalam (`Mlym`) | 0.444 | **0.4116** | 0.4135 | **100.0%** (32/32) | 0.0935 |
| **Main-4 Dravidian Mean** | - | 4 Anchor Scripts | 0.5203 | **0.4896** | **0.4920** | **96.88%** | 0.1014 |
| **Tulu** | `tcy` | Kannada (`Knda`) | 0.652 | **0.6399** | 0.6479 | **96.88%** (31/32) | 0.1215 |
| **Koya** | `kff` | Odia (`Orya`) | 0.525 | **0.5391** | 0.5498 | **90.63%** (29/32) | 0.1351 |
| **Kuvi** | `kxv` | Odia (`Orya`) | 0.453 | **0.4560** | 0.4606 | **93.75%** (30/32) | 0.1362 |
| **Kolami / Naikpod** | `nit` | Telugu (`Telu`) | 0.718 | **0.7235** | 0.7415 | **84.38%** (27/32) | 0.1393 |
| **8-Language Dravidian Mean**| - | 8 Languages | - | 91.25% | **94.14%** (Mean Tag Acc) | **94.14%** | - |
| **English** | `eng` | Latin | 1.653 | **1.3929** | 1.4533 | - | 0.3934 |

---

## 4. Licensing

- **Software Code & Architecture:** [MIT License](LICENSE)
- **Model Weights & Checkpoints:** [Apache License 2.0](LICENSE-MODEL)
- **Usage:** Full permissive open-source release for research and production applications.
