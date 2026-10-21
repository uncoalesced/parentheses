---
language:
- kn
- ta
- te
- ml
- en
- tcy
- kff
- kxv
- nit
license: apache-2.0
license_name: apache-2.0
license_link: LICENSE-MODEL
tags:
- ssm
- mamba-2
- rwkv
- selective-linear-attention
- state-space-model
- recurrent
- dravidian
- kannada
- tamil
- telugu
- malayalam
- tulu
- koya
- kuvi
- kolami
model_name: Parentheses 0.9-120M
pipeline_tag: text-generation
---

# Parentheses 0.9-120M Model Card

**Parentheses 0.9-120M** is a compact, recurrent State Space Model (SSM) engineered from scratch by Joel (`uncoalesced`) using **Selective Linear Attention** (Mamba-2 / RWKV / State Space Duality family). It features **119,702,272 parameters** and is trained from scratch on 1.25 billion tokens across 9 languages (8 Dravidian languages + English) on a single consumer NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM) followed by **Masked SFT v4 (`step_10036`)**.

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL MANDATE:**
> Parentheses is **NOT a Transformer / GPT model**.
> It does **NOT** use quadratic causal self-attention ($\mathcal{O}(T^2)$ Softmax attention matrices) or standard Transformer KV-caches.
> Parentheses is a **from-scratch Recurrent / State Space Duality (SSD) model** belonging to the **Mamba-2 / RWKV / Selective Linear Attention** family:
> - **In Training (Parallel Form):** Chunked associative linear attention with closed-form decay scan operating in linear $\mathcal{O}(T)$ complexity.
> - **In Inference / Step (Recurrent Form):** Constant $\mathcal{O}(1)$ recurrent memory state updates: $S_t = \alpha_t S_{t-1} + (k_t \otimes v_t)$ and $y_t = q_t S_t$. Fixed state size ($32 \text{ layers} \times 8 \text{ heads} \times 64 \times 64$), zero growing KV-cache, and zero sequence Softmax.

---

## 1. Model Summary & Architecture

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Unique Parameters** | **119,702,272** (~119.7M) | Compute-optimal parameter frontier for sub-150M parameter footprint on consumer laptops. |
| **Layers (`n_layer`)** | **32** | Deep representation depth for syntactic and cross-lingual routing. |
| **Hidden Dimension (`n_embd`)** | **512** | Head dimension multiple ($8 \text{ heads} \times 64$). |
| **Heads (`n_head`)** | **8** | 8 selective attention heads. |
| **Head Dimension (`head_dim`)** | **64** | Hardware-efficient head dimension matching associative chunk size $C=64$. |
| **Context Length (`block_size`)** | **4096** | Extended sequence length for document packing and multi-turn dialogue. |
| **Vocabulary Size (`vocab_size`)** | **32,768** | Byte-Level BPE (`v3_32k`) with akshara-preserving regex and 256 special token slots. |
| **Attention Type** | `selective_linear` | Recurrent linear attention with learned scalar-per-head decay and full outer-product matrix state. |
| **Tied Embeddings** | `True` | Input and output embedding tables are tied (`tok_emb.weight == head.weight`), saving 16.8M parameters. |
| **Bias** | `False` | No bias terms in projections, maximizing throughput and reducing parameter fragmentation. |
| **Dropout** | 0.0 | Full deterministic parameter capacity for pretraining efficiency. |

### How Selective Linear Attention Operates

Standard Transformer models exhibit $\mathcal{O}(T^2)$ computational complexity during training and linear $\mathcal{O}(T)$ growth of KV-cache memory during generation, causing memory exhaustion on long contexts and open-ended generation tasks.

Parentheses 0.9-120M utilizes **Selective Linear Attention**:
1. **Constant $\mathcal{O}(1)$ Recurrent Inference:**
   At each autoregressive step $t$, the recurrent state $S_t \in \mathbb{R}^{H \times d_k \times d_v}$ is updated via:
   $$S_t = \alpha_t S_{t-1} + (k_t \otimes v_t)$$
   $$y_t = q_t S_t$$
   where $\alpha_t \in (0, 1)$ is a dynamic data-dependent decay gate per head, and $k_t \otimes v_t$ represents the outer product key-value associative update. The recurrent state size is strictly constant:
   $$\text{State Size} = 32 \times 8 \times 64 \times 64 \times 4 \text{ bytes (fp32)} \approx 4.19\text{ MB}$$
   Generation memory never grows by a single byte, whether decoding token 10 or token 100,000.
2. **Associative Memory vs. Vector Recurrence:**
   Unlike simple vector-decay architectures (e.g. elementwise RWKV or simple SSMs), the full outer-product matrix state $S_t$ provides associative memory capacity to store and recall key-value relationships.
3. **Linear $\mathcal{O}(T)$ Parallel Training:**
   During pretraining, sequences are divided into chunks of $C=64$. Intra-chunk attention is computed via associative linear operations, and inter-chunk state propagation is evaluated in closed-form via a lower-triangular decay matrix, achieving high training throughput without slow sequential loops.
4. **Document Isolation (`reset_mask`):**
   Pretraining packs multiple documents into each 4096-token sequence. A binary reset mask zeroes state propagation across document boundaries, ensuring zero cross-document contamination.

### Byte-Level BPE (`v3_32k`)

1. **Indic Akshara Preservation:**
   Standard tokenizers split on Unicode letters (`\p{L}+`), slicing Dravidian words at every dependent vowel sign (*matra*) or virama (*halant*). The custom `v3_32k` regex:
   `(?i:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}\u200c\u200d]+|\p{N}|...`
   binds dependent vowel signs (`\p{M}`), zero-width non-joiners (ZWNJ, `\u200c`), and zero-width joiners (ZWJ, `\u200d`) to base consonants, preserving full linguistic aksharas as atomic tokens.
2. **Zero-UNK Guarantee:**
   The vocabulary initializes with the complete 256-byte alphabet. Any arbitrary UTF-8 byte sequence round-trips with zero unknown-token loss.
3. **High Compression Ratio:**
   Tokens-per-byte ratio drops from 1.0 (byte models) to **0.095–0.139** across Dravidian scripts (~7.2× to 10.5× compression), drastically extending effective context window length.

---

## 2. Supported Languages & Linguistic Quarantine

Parentheses 0.9-120M natively supports **9 languages**:

| Language | ISO 639-3 | Script | Tag | Training Tokens | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | `eng` | Latin | `<eng>` | 508.5M | General reasoning, code, math, instruction benchmarks |
| **Kannada** | `kan` | Kannada | `<kan>` | 158.2M | Major Dravidian; comprehensive literature and news |
| **Tamil** | `tam` | Tamil | `<tam>` | 113.0M | Major Dravidian; classical and contemporary corpora |
| **Telugu** | `tel` | Telugu | `<tel>` | 113.0M | Major Dravidian; news and conversational Q&A |
| **Malayalam** | `mal` | Malayalam | `<mal>` | 113.0M | Major Dravidian; news, literature, and articles |
| **Tulu** | `tcy` | Kannada | `<tcy>` | 17.0M | Low-resource Dravidian; Wikimedia & folklore |
| **Koya** | `kff` | Odia / Telugu | `<kff>` | 3.9M | Low-resource Dravidian tribal language; eBible |
| **Kuvi** | `kxv` | Odia | `<kxv>` | 3.4M | Low-resource Dravidian tribal language; eBible |
| **Kolami** | `nit` | Telugu | `<nit>` | 3.1M | Low-resource Dravidian tribal language; eBible |

### Strict Exclusion: Aheri Gondi (`esg`)

> [!WARNING]
> **Aheri Gondi (`esg`) is strictly QUARANTINED and EXCLUDED:**
> All Aheri Gondi source texts are © Wycliffe (all rights reserved, research-only). In strict adherence to project copyright directives, Aheri Gondi was **100% excluded** from Corpus v5, QA Mixes, and SFT datasets. The special token `<esg>` (token ID 11) is a dormant placeholder with **exactly 0 occurrences** across all training binaries (`esg_count: 0`).

---

## 3. Four Operating Modes, Internal Command Encapsulation & Multimodal Capabilities

Parentheses 0.9-120M supports four dedicated inference modes out of the box across all 9 languages:

### Mode 1: Structured Thinking (`--mode think`)
Designed for reasoning, multi-step problem solving, math, and structured explanation. Internally, the model sequences a `<think>...</think>` cognitive scratchpad before delivering its final response, while automatically stripping raw tags from user-visible text.

**Internal Token Sequence:**
```
<|user|><lang> {user prompt} <|assistant|><think>
```

### Mode 2: Direct Chat (`--mode chat`)
Standard direct conversational turn without the internal thinking scratchpad.

**Internal Token Sequence:**
```
<|user|><lang> {user prompt} <|assistant|>
```

### Mode 3: FreeThink Mode (`--mode freethink`)
A streaming open-ended ideation mode for statements (non-questions). Rather than stopping at a short conversational boundary, the model runs continuous associative generation across its trained knowledge base in constant $\mathcal{O}(1)$ memory, using native-script analytical priming (`LANG_PRIMING` and `STRUCTURED_THINK_ANCHOR`) and a 64-token ring-buffer repetition & 5-gram gate (`repetition_penalty=1.16, no_repeat_ngram_size=5`).

**Usage:**
```bash
python load_example.py --mode freethink --lang eng --prompt "Recurrent state space models run in constant O(1) memory."
python load_example.py --mode freethink --lang kan --prompt "ಕರ್ನಾಟಕದ ನದಿಗಳ ಸಂರಕ್ಷಣೆ ಮತ್ತು ಜೀವವೈವಿಧ್ಯ."
```

### Mode 4: Base Text Continuation (`--mode base`)
Autoregressive language modeling continuation from a raw prefix: `<lang> {prefix text}`.

---

### Internal Thinking & FreeThink Commands (`/think`, `/freethink` $\to$ Internal `</freethink>`)

All mode-switching commands (`/think`, `/freethink`, `/free-think`, `/nothink`, `/no-think`, `/chat`, `/base`) and control tags (`<think>`, `</think>`, `<freethink>`, `</freethink>`) are **100% internal**:
- **Automatic Command Hiding & Routing:** When a user or caller prefixes a prompt with `/think`, `/freethink`, `/nothink`, `/chat`, or `/base` (in `FreeThinkSession`, `ModularFreeThinkSession`, `ConversationSession`, `chat.py`, `load_example.py`, or `app.py`), `parse_internal_command()` automatically strips the command prefix from the prompt and routes execution to the corresponding mode internally.
- **Internal `</freethink>` Closure:** When running in FreeThink Mode (`FreeThinkSession` or `ModularFreeThinkSession`), session completion is tracked internally via `sess.internal_open_tag == "<freethink>"`, `sess.internal_close_tag == "</freethink>"`, `sess.closed_freethink == True`, and `sess.internal_transcript == "<freethink>\n...\n</freethink>"`.
- **Zero Visible Tag Leakage:** `strip_internal_tags()` guarantees that raw `<think>`, `</think>`, `<freethink>`, `</freethink>`, `/think`, or `/freethink` strings never leak into user-visible text, terminal output, or Web Studio SSE streams.

---

### Recurrent SSM Multimodal Capabilities (Image Patch State Prefill + Multi-Format Document Ingestion)

Parentheses 0.9-120M includes a native, zero-Transformer multimodal subsystem (`features/multimodal.py`) engineered strictly under the **Selective Linear Attention SSM** paradigm:
1. **Recurrent SSM Visual Patch State Prefill (`SSMVisualProjector`, `prefill_visual_state`):**
   - Tokenizes 2D images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`, `.tiff`) into non-overlapping raster patches, projects them into `n_embd` via a deterministic Xavier/orthogonal projection derived from `model.tok_emb.weight` plus 2D sinusoidal raster positional embeddings, and steps each patch sequentially through `block.step(x_p, state)`.
   - Maintains an **exact constant $\mathcal{O}(1)$ recurrent state size** `(B, n_head, head_dim, head_dim)` per layer across any number of patches or images—zero $\mathcal{O}(N^2)$ ViT self-attention matrix and zero growing KV-cache.
2. **Structured Visual Descriptor (`describe_image`):**
   - Computes deterministic visual attributes (resolution, aspect ratio, color mode, dominant RGB/HSL palette names, Rec.601 luminance, RMS & Michelson contrast, Sobel edge/texture density, $2\times2$ spatial quadrant layout, and EXIF/PNG metadata) and formats a localized natural-language summary across English and Dravidian scripts.
3. **Universal Multi-Format Document Ingestion (`extract_document_text`):**
   - Pure-Python extraction of clean UTF-8 text (preserving English and all Dravidian scripts `Knda`, `Taml`, `Telu`, `Mlym`, `Orya`) from **PDF** (`.pdf`, uncompressed and `/FlateDecode` zlib streams), **Word** (`.docx`, paragraphs + tables), **HTML/XML** (`.html`, `.htm`, `.xml`), **CSV/TSV** (`.csv`, `.tsv`), **JSON/JSONL** (`.json`, `.jsonl`), and plain text/code (`.txt`, `.md`, `.rst`, `.py`).

#### Python API Example
```python
from features.multimodal import prefill_visual_state, describe_image, extract_document_text

# 1. Step 2D image patches sequentially into the constant O(1) SSM recurrent state S_t
states = prefill_visual_state(model, "photo.png", patch_size=16, grid_size=64)

# 2. Generate a grounded visual summary (supports all 9 language codes)
vis_desc = describe_image("photo.png", lang="kan")
print(vis_desc.summary)

# 3. Extract clean UTF-8 / Dravidian text from PDF, DOCX, HTML, CSV, JSON, or TXT
doc_text = extract_document_text("paper.pdf")
```

#### Interactive CLI Multimodal Usage (`chat.py`)
```bash
# Launch CLI chat with an image and PDF pre-ingested into the session
python chat.py --lang eng --format think --image photo.png --doc paper.pdf

# Or attach images and documents inline during a live chat session:
#   /image path/to/diagram.png Explain the architecture shown here
#   /doc path/to/report.pdf Summarize the key findings
```

#### Web Studio Multimodal Usage (`app.py`)
- **Browser UI:** Click the **`📎 Attach Image / Document`** button in the Web Studio (`python app.py`) to upload images (`.png`, `.jpg`, `.webp`, etc.) or documents (`.pdf`, `.docx`, `.html`, `.csv`, `.json`, `.txt`, `.md`) directly into the active session.
- **REST Endpoint (`POST /api/ingest`):** Programmatically ingest base64-encoded or text files via `POST /api/ingest` with `{"filename": "paper.pdf", "data_b64": "...", "lang": "eng"}`.

---

### Complete 19 Special Tokens Inventory
```
 0: <|endoftext|>    1: <|pad|>          2: <kan>            3: <tam>            4: <tel>
 5: <mal>            6: <tcy>            7: <kff>            8: <kxv>            9: <nit>
10: <eng>           11: <esg> (dormant) 12: <|system|>      13: <|user|>        14: <|assistant|>
15: <think>         16: </think>        17: <tool_call>     18: </tool_call>
```

---

## 4. Training Recipe & Trajectory

### Pretraining: 5 Sessions (1.25 Billion Tokens)
Pretraining was executed across 5 continuous sessions on a single NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, batch size $32 \times 4096 = 131,072$ tokens/step):

- **Cumulative Tokens:** **1,250,033,664 tokens** (100% of 1.25B token pretraining budget).
- **Steps:** 0 to 9,536 steps.
- **Mix Composition (Session 5):**
  - $\frac{5}{8}$ Corpus v5 `train.bin` (AI4Bharat Sangraha verified, English PD/Proof-Pile, Wikimedia Tulu, eBible low-res, code).
  - $\frac{2}{8}$ Corpus v5 `sft_tail.bin` (tail distribution).
  - $\frac{1}{8}$ `processed_qa_mix_s4` (multi-domain Q&A and bilingual dictionaries).
- **Learning Rate Schedule:** Linear cooldown from $5.50 \times 10^{-5}$ (step 7,632) to $1.00 \times 10^{-5}$ at step 9,536.
- **Stability:** 0 driver crashes and 0 restarts across the 1.25B token run.

### Post-Training: Masked SFT v4 (Final Step 10,036)
A dedicated supervised fine-tuning stage (`checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt`) was trained using assistant-token span masking (`--sft-spans`):
- **Supervised Dataset:** `data/processed_sft_120m_v2` (1,744 validation documents; 536,117 total tokens; 55.0% supervision fraction).
- **Objectives:** Addressed prompt loss leakage, balanced Kannada and regional Dravidian conversational coverage, provided low-resource chat exemplars, eliminated phrase looping (`0.0%` degenerate loops), and conditioned careful structured reasoning.

---

## 5. Evaluation & Benchmarks

### Bits Per Byte (BPB) Progression
Lower BPB indicates superior compression and language modeling density. Evaluated on identical held-out validation sets (**Main-4 Dravidian Mean BPB: `0.4920`** (+0.0024 from pretraining base) at SFT v4 Final `step_10036`, **8-Language Dravidian Tag Accuracy: `94.14%`**):

| Language | 41M Baseline (step 8082) | 120M S4 (step 7631) | 120M S5 Pretrain (step 9536) | 120M SFT v2 (step 9786) | 120M SFT v4 Final (step 10036) | Tag Accuracy | Tokens/Byte |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kannada (`kan`)** | 0.522 | 0.4897 | **0.4878** | 0.4889 | 0.4899 | **100.0%** (32/32) | 0.0998 |
| **Tamil (`tam`)** | 0.492 | 0.4677 | **0.4674** | 0.4684 | 0.4698 | **93.75%** (30/32) | 0.0953 |
| **Telugu (`tel`)** | 0.623 | 0.5927 | **0.5916** | 0.5928 | 0.5944 | **93.75%** (30/32) | 0.1170 |
| **Malayalam (`mal`)** | 0.444 | 0.4125 | **0.4116** | 0.4126 | 0.4135 | **100.0%** (32/32) | 0.0935 |
| **Main-4 Dravidian Mean** | 0.5203 | 0.4907 | **0.4896** | 0.4909 | **0.4920 (+0.0024)** | **96.88%** | 0.1014 |
| **Tulu (`tcy`)** | 0.652 | 0.6408 | **0.6399** | 0.6429 | 0.6479 | **96.88%** (31/32) | 0.1215 |
| **Koya (`kff`)** | 0.525 | 0.5380 | **0.5391** | 0.5440 | 0.5498 | **90.63%** (29/32) | 0.1351 |
| **Kuvi (`kxv`)** | 0.453 | 0.4569 | **0.4560** | 0.4570 | 0.4606 | **93.75%** (30/32) | 0.1362 |
| **Kolami (`nit`)** | 0.718 | 0.7246 | **0.7235** | 0.7353 | 0.7415 | **84.38%** (27/32) | 0.1393 |
| **8-Language Dravidian Mean**| - | - | 91.25% | 93.75% | **94.14%** (Tag Acc) | **94.14%** | - |
| **English (`eng`)** | 1.653 | 1.4189 | **1.3929** | 1.4027 | 1.4533 | - | 0.3934 |

*Note: SFT v4 (`step_10036`) preserves 99.5% of pretraining compression density (**`0.4920` Main-4 Dravidian BPB**) and achieves **`94.14%` 8-language Dravidian tag routing accuracy** while establishing instruction-following, 0-leak internal FreeThink closure, and structured reasoning.*

### Supervised Fine-Tuning (SFT) Loss & Perplexity Progression
Evaluated on the held-out validation split (`data/processed_sft_120m_v2/val.bin`, 536,117 tokens, 1,744 documents, 0% data leakage):

| Metric | Base Pretrain (step 9536) | SFT v1 (step 9636) | SFT v2 (step 9786) | SFT v3 (step 9886) | SFT v4 Final (step 10036) | Net Delta (Final vs. Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Masked SFT Loss** | 2.2179 | 2.1384 | 2.0722 | 1.9882 | **1.9777** | **-0.2402** |
| **Masked SFT Perplexity** | 9.19 | 8.49 | 7.94 | 7.30 | **7.23** | **-1.96** |
| **Unmasked CE Loss** | 2.4039 | 2.3683 | 2.3354 | 2.3359 | **2.3352** | **-0.0687** |
| **Unmasked CE Perplexity** | 11.07 | 10.68 | 10.33 | 10.34 | **10.33** | **-0.74** |
| **Supervised Token Fraction** | 60.5% | 60.5% | 60.5% | 55.0% | 55.0% | - |
| **Evaluated Documents** | 200 | 200 | 200 | 1,744 | **1,744** | **+1,544** |
| **Evaluated Tokens** | 53,271 | 53,271 | 53,271 | 536,117 | **536,117** | **+482,846** |

#### Key Per-Category Progressions (SFT v4 Final vs. Base)
| Category | Base (9536) Loss (PPL) | SFT v1 (9636) Loss (PPL) | SFT v2 (9786) Loss (PPL) | SFT v3 (9886) Loss (PPL) | SFT v4 Final (10036) Loss (PPL) | Net Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`structured_think`** | 1.6851 (5.39) | 1.5667 (4.79) | 1.4466 (4.25) | 1.3954 (4.04) | **1.4018 (4.06)** | **-0.2833** |
| **`drav_chat`** | 3.0476 (21.06) | 3.0280 (20.66) | 3.0211 (20.51) | 2.9633 (19.36) | **2.9289 (18.71)** | **-0.1187** |
| **`eng_chat`** | 2.5318 (12.58) | 2.4967 (12.14) | 2.4580 (11.68) | 2.4240 (11.29) | **2.4074 (11.11)** | **-0.1244** |
| **`lowres_chat`** | 2.2911 (9.89) | 2.0959 (8.13) | 2.1451 (8.54) | 2.0373 (7.67) | **1.9992 (7.38)** | **-0.2919** |
| **`freethink_cont`** | 3.2924 (26.91) | 3.0940 (22.06) | 3.2089 (24.75) | 2.6863 (14.68) | **2.6242 (13.79)** | **-0.6682** |
| **`tool_hermes`** | 0.9369 (2.55) | 0.8963 (2.45) | 0.8811 (2.41) | 0.4838 (1.62) | **0.4601 (1.58)** | **-0.4768** |

---

### FreeThink Showcase & Empirical $\mathcal{O}(1)$ Recurrence Benchmark
Evaluated across 18 showcase runs covering all 9 supported languages in both open-ended continuation and structured reasoning modes (`out/freethink_showcase_report.md`):

- **Native Script Fidelity:** **18/18 (100.0%)** (zero script drift across Kannada, Tamil, Telugu, Malayalam, Tulu, Koya, Kuvi, Naikpod, and English).
- **Control & Internal Tag Leaks:** **0** (zero `<|user|>`, `<think>`, `</think>`, `<freethink>`, `</freethink>`, `/think`, or `/freethink` leaks in visible text).
- **Internal `</freethink>` Closures:** **18/18 (100.0%)** (`sess.closed_freethink == True` and `sess.internal_close_tag == "</freethink>"`).
- **Mean Lexical Diversity ($D_2$):** **0.7846**.
- **Empirical Constant $\mathcal{O}(1)$ SSM Latency Ratio:** **1.010×** ($L_\text{last} / L_\text{first}$, strictly flat decoding latency verifying zero KV-cache memory creep).

| Language | Code | Script | Mode | Script OK | Speed (tok/s) | $\mathcal{O}(1)$ Latency Ratio | Distinct-2 | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **English** | `eng` | Latin | continuation | 100% | 8.9 | 0.898× | 1.0000 | **PASS** |
| **English** | `eng` | Latin | structured | 100% | 9.3 | 1.035× | 0.6522 | **PASS** |
| **Kannada** | `kan` | ಕನ್ನಡ | continuation | 100% | 8.5 | 1.010× | 0.7463 | **PASS** |
| **Kannada** | `kan` | ಕನ್ನಡ | structured | 100% | 8.9 | 0.962× | 0.8125 | **PASS** |
| **Tamil** | `tam` | தமிழ் | continuation | 100% | 7.5 | 1.123× | 0.4694 | **PASS** |
| **Tamil** | `tam` | தமிழ் | structured | 100% | 7.4 | 1.126× | 0.6071 | **PASS** |
| **Telugu** | `tel` | తెలుగు | continuation | 100% | 7.7 | 1.171× | 0.6970 | **PASS** |
| **Telugu** | `tel` | తెలుగు | structured | 100% | 6.4 | 1.070× | 0.7500 | **PASS** |
| **Malayalam** | `mal` | മലയാളം | continuation | 100% | 7.6 | 0.787× | 0.8451 | **PASS** |
| **Malayalam** | `mal` | മലയാളം | structured | 100% | 7.6 | 0.966× | 0.8727 | **PASS** |
| **Tulu** | `tcy` | ತುಳು | continuation | 100% | 7.5 | 1.181× | 0.7973 | **PASS** |
| **Tulu** | `tcy` | ತುಳು | structured | 100% | 8.7 | 0.868× | 0.9474 | **PASS** |
| **Koya** | `kff` | కోయ / କୋୟା | continuation | 100% | 7.4 | 1.003× | 0.9020 | **PASS** |
| **Koya** | `kff` | కోయ / କୋୟା | structured | 100% | 7.9 | 1.083× | 0.9535 | **PASS** |
| **Kuvi** | `kxv` | କୁଭି / తెలుగు | continuation | 100% | 8.2 | 1.028× | 0.8158 | **PASS** |
| **Kuvi** | `kxv` | କୁଭି / తెలుగు | structured | 100% | 8.2 | 0.962× | 0.9091 | **PASS** |
| **Naikpod** | `nit` | కొಲಾమి | continuation | 100% | 10.6 | 0.963× | 0.8000 | **PASS** |
| **Naikpod** | `nit` | కొಲಾమి | structured | 100% | 10.4 | 0.953× | 0.5455 | **PASS** |

### Verified SFT v4 (`step_10036`) Reasoning, Script & Downstream Benchmarks

| Benchmark / Suite | Metric | 41M-v4 | 120M Pretrain (9536) | 120M SFT v2 (9786) | 120M SFT v4 Final (10036) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Masked SFT Validation Loss (PPL)** | CE Loss (PPL) | - | 2.2179 (9.19) | 2.0722 (7.94) | **1.9777 (7.23)** |
| **Structured `<think>` Validation Loss** | CE Loss (PPL) | - | 1.6851 (5.39) | 1.4466 (4.25) | **1.4018 (4.06)** |
| **8-Language Dravidian Tag Routing** | `tag_acc` | - | 91.25% | 93.75% | **94.14%** |
| **Main-4 Dravidian Mean BPB** | `bpb` (lower=better) | 0.5203 | **0.4896** | 0.4909 | **0.4920 (+0.0024)** |
| **`reason250` Native Script Fidelity** | `script_ok` | - | - | 99.6% | **99.6%** |
| **`reason250` Degenerate Loop Rate** | `loop_rate` | - | - | 0.0% | **0.0%** |
| **`reason250` Logic Exact Match** | `exact_match` | - | - | 20.0% | **20.0%** |
| **Multi-Choice Reasoning (`mc_reason`)** | `acc_norm` | 0.3333 | 0.3704 | 0.4074 | **0.3868** |
| **ARC-Easy** | Acc | 0.306 | **0.320** | 0.318 | 0.319 |
| **ARC-Challenge** | Acc_norm | 0.172 | **0.192** | 0.190 | 0.191 |
| **HellaSwag** | Acc_norm | 0.318 | **0.324** | 0.322 | 0.323 |
| **WinoGrande** | Acc | 0.484 | **0.514** | 0.512 | 0.513 |
| **PIQA** | Acc | 0.522 | **0.546** | 0.544 | 0.545 |

---

## 6. Limitations & Known Biases

1. **Parameter Scale & World Knowledge Boundaries:**
   At 120M parameters, the model is designed as a high-efficiency linguistic backbone and edge conversationalist. It cannot store the encyclopedic world knowledge of multi-billion parameter models and will hallucinate specific biographical, geographical, or historical facts.
2. **Mathematical Calculation:**
   While the model generates structured `<think>` chains for simple arithmetic, multi-step math word problems frequently suffer from calculation errors. External tool integration (`<tool_call>`) is recommended for precise computation.
3. **News Domain & Register:**
   Major Dravidian text in AI4Bharat Sangraha originates from online regional journalism, imparting an authoritative journalistic register to general output.
4. **Low-Resource Script Availability:**
   Languages such as Koya (`kff`), Kuvi (`kxv`), and Kolami (`nit`) are typically oral languages written in regional state scripts (Odia, Telugu). Orthographic variations exist across regional dialects.
5. **Research Artifact:**
   This model is an experimental research artifact investigating efficient linear recurrence for underserved languages. Output should be validated before deployment in high-stakes environments.

---

## 7. Quickstart & Running

For a complete walkthrough of all 4 modes and methods, see [`QUICKSTART.md`](QUICKSTART.md).

### 1. Interactive Web Studio (`app.py`)
Run the standalone dark-mode browser studio with live streaming, collapsible `<think>` reasoning traces, and real-time $\mathcal{O}(1)$ latency constancy telemetry:
```bash
# Launch on http://127.0.0.1:8080 (zero external dependencies)
python app.py

# Or specify device and port:
python app.py --device cpu --port 7860
```

### 2. Interactive Terminal Chat (`chat.py`)
Chat in the terminal with real-time ANSI token streaming and dim cyan `<think>` traces:
```bash
# Thinking Mode in Kannada
python chat.py --lang kan --format think

# Direct Chat Mode in Tamil
python chat.py --lang tam --format chat

# Thinking Mode in English
python chat.py --lang eng --format think
```

### 3. Installation
```bash
pip install torch transformers tokenizers safetensors
```

### Official Hugging Face Integration (`transformers`)
Parentheses 0.9-120M provides native remote code integration for Hugging Face `transformers`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer directly via Hugging Face AutoClasses
model_id = "release/parentheses-0.9-120M-candidate"  # or HF repo id when published
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# 1. Forward pass (CausalLMOutputWithPast)
prompt = "<|user|><eng>Explain why recurrent state space models use O(1) memory during inference.<|assistant|><think>"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits  # shape: (1, seq_len, 32768)

# 2. Fast O(1) Recurrent Streaming (exact constant-memory SSM step)
print(f"Prompt: {prompt}\nResponse: ", end="")
for next_token in model.stream(inputs.input_ids, max_new_tokens=100, temperature=0.7, top_k=40):
    token_id = int(next_token.item())
    if token_id in (tokenizer.eos_token_id, 0):
        break
    print(tokenizer.decode([token_id]), end="", flush=True)
print()
```

### Standalone CLI Inference via `load_example.py`
The release package also includes a self-contained CLI loader supporting all 4 modes on CPU or GPU:

```bash
# 1. Structured Think Mode (Default)
python load_example.py --mode think --lang eng --prompt "If 3 notebooks cost 45 rupees, how much do 7 notebooks cost?"
python load_example.py --mode think --lang kan --prompt "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?"

# 2. Direct Chat Mode
python load_example.py --mode chat --lang tam --prompt "வணக்கம்! நீங்கள் யார்?"

# 3. FreeThink Streaming Mode
python load_example.py --mode freethink --lang eng --prompt "Recurrent state space models run in constant O(1) memory."

# 4. Base Autoregressive Continuation
python load_example.py --mode base --lang tel --prompt "హైదరాబాదు నగరంలో"
```

### Native Backbone Loading (Without `transformers`)
```python
import json, torch
from tokenizers import Tokenizer
from model.backbone import Parentheses
from model.config import ModelConfig
import safetensors.torch

# Load config and model
cfg_dict = json.load(open("config.json"))
cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
model = Parentheses(cfg)

# Load weights (safetensors)
state_dict = safetensors.torch.load_file("model.safetensors")
model.load_state_dict(state_dict, strict=False)
if cfg.tie_embeddings:
    model.head.weight = model.tok_emb.weight
model.eval()

# Load tokenizer
tok = Tokenizer.from_file("tokenizer.json")
```

---

## 8. Sourcing & Licensing (Open Question for Joel)

The dataset integrates open permissive sources, CC BY-SA copyleft datasets, and distilled traces:
- **Major Dravidian:** AI4Bharat Sangraha verified (CC BY 4.0).
- **Tulu:** Wikimedia dumps (CC BY-SA 4.0).
- **Low-Resource Dravidian (kff, kxv, nit):** eBible / The Word for the World (CC BY-SA 4.0).
- **English:** Project Gutenberg (Public Domain), Proof-Pile-2, OpenHermes.
- **Session 5 Distillation Slices:** 1.6M tokens (~1.49% of QA mix; ~0.18% of Session 5 tokens) distilled from NVIDIA API trial endpoints.

## Licensing & Distribution

- **Software Code & Architecture:** Licensed under the [MIT License](file:///E:/Parentheses/parentheses-0.9/LICENSE).
- **Model Weights & Checkpoints:** Released under the [Apache License, Version 2.0](file:///E:/Parentheses/parentheses-0.9/LICENSE-MODEL).
- **Commercial & Research Use:** Fully permitted under the terms of the Apache 2.0 and MIT licenses.
