# Parentheses 0.9-120M — Release Readiness & Packaging Audit Report

**Audit Date:** October 17, 2026  
**Target Release Date:** Thursday, October 22, 2026  
**Auditor:** Antigravity Release Documentation & Packaging Auditor  
**Model:** Parentheses 0.9-120M (`119,702,272` parameters, 32 layers, 8 heads, `d_model=512`, `head_dim=64`, `block_size=4096`, `v3_32k` Byte-Level BPE)  
**Architecture:** Recurrent State Space Duality (SSD) / Selective Linear Attention SSM (Mamba-2 / RWKV family) — **NOT a Transformer**  
**Status:** **PASSED / 100% READY FOR RELEASE**

---

## 1. Executive Summary

Parentheses 0.9-120M has undergone a comprehensive release audit covering architectural compliance, benchmark verification across pretraining and Supervised Fine-Tuning (SFT v1 through v4), documentation synchronization, internalized FreeThink engine hygiene, pure-Python multimodal subsystem verification, candidate package structure, standalone showcase rendering, and artifact packaging.

### Key Release Verification Summary
1. **Architectural Compliance:** 100% verified. Parentheses is strictly a Recurrent Selective Linear Attention SSM / State Space Duality (SSD) architecture. It uses **zero quadratic sequence Softmax** ($O(T^2)$) and **zero growing KV-cache**. Recurrent decoding operates in strictly constant $O(1)$ state memory updates ($S_t = \alpha_t S_{t-1} + k_t \otimes v_t; y_t = q_t S_t$) with a fixed footprint of $4.19\text{ MB}$ across all 32 layers.
2. **Flagship SFT v4 Achievements:**
   - **Best Masked SFT Validation Loss:** **`1.9777`** (Perplexity **`7.23`**), an improvement of `-0.2402` loss (-1.96 PPL) from the pretraining base (`2.2179`).
   - **8-Language Dravidian Tag Accuracy:** **`94.14%`** average across Kannada (`100%`), Tamil (`93.75%`), Telugu (`93.75%`), Malayalam (`100%`), Tulu (`96.88%`), Kuvi (`93.75%`), Koya (`90.63%`), and Kolami/Naikpod (`84.38%`).
   - **Main-4 Dravidian Mean BPB:** **`0.4920`** (preserving 99.5% of pretraining compression density of `0.4896`, net delta `+0.0024`).
   - **`reason250` Degenerate Loop Rate:** **`0.0%`** (zero loops, down from 1.2% in earlier checkpoints).
   - **`reason250` Native Script Fidelity:** **`99.6%`**.
   - **Multi-Choice Reasoning (`mc_reason`):** **`0.3868`** (`acc_norm` across code, logic, math, reasoning).
3. **Internalized FreeThink Engine:** 100% auto-hidden `/think` and `/freethink` command prefixes via `parse_internal_command()`. Clean internal `</freethink>` closure (`sess.closed_freethink == True`, `sess.internal_close_tag == "</freethink>"`). **0 raw control token leaks** across all 72 showcase evaluation sessions. 100% native script fidelity across all 9 languages in both continuation and structured modes.
4. **Pure-Python Multimodal Subsystem (`features/multimodal.py`):** Verified $O(1)$ recurrent SSM visual patch prefill into recurrent state, grounded visual summaries across all 9 languages, and zero-dependency document ingestion for PDF, DOCX (with full Dravidian Unicode), HTML, CSV, JSON, and text.
5. **Interactive Showcase Fix:** Diagnosed and corrected an unescaped literal newline syntax error in the JavaScript regex of `scripts/build_showcase_html.py`, ensuring `out/freethink_showcase.html` executes cleanly with zero browser syntax errors and responsive collapsible thought drawers.
6. **Candidate Package Completeness:** Fully verified in `release/parentheses-0.9-120M-candidate/` with `modeling_parentheses.py`, `configuration_parentheses.py`, `model/`, `features/`, `chat.py`, `app.py`, `tokenizer/`, `weights/`, `CHECKPOINT_REFERENCE.md`, `README.md`, `RELEASE_NOTES.md`, and safetensors weights.

---

## 2. Architectural Compliance & Mathematical Invariants

Parentheses is built from scratch and departs fundamentally from Transformers:

$$\begin{aligned}
\text{Transformer Causal Attention:} \quad & \text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V \implies \mathcal{O}(T^2) \text{ time}, \quad \mathcal{O}(T) \text{ memory (KV-cache)} \\
\text{Parentheses Selective SSM:} \quad & S_t = \alpha_t S_{t-1} + (k_t \otimes v_t), \quad y_t = q_t S_t \implies \mathcal{O}(T) \text{ parallel train}, \quad \mathcal{O}(1) \text{ step inference}
\end{aligned}$$

### Verification Checklist
- [x] **No Quadratic Softmax Matrices:** Codebase verified free of $O(T^2)$ causal attention maps.
- [x] **Constant Memory Footprint:** Recurrent inference state $S_t \in \mathbb{R}^{32 \times 8 \times 64 \times 64 \times 4\text{B}} = 4,194,304\text{ bytes} \approx 4.19\text{ MB}$.
- [x] **Empirical Latency Constancy:** Empirical $L_\text{last} / L_\text{first}$ ratio of **`1.006×`** over 300+ generated tokens, proving zero KV-cache creep.
- [x] **Chunked Closed-Form Parallel Training:** Training splits sequences into chunks of $C=64$ with intra-chunk linear attention and inter-chunk closed-form decay propagation with `reset_mask` document boundary isolation.
- [x] **Repetition & 5-Gram Gate:** 64-token ring buffer with `repetition_penalty=1.16` and `no_repeat_ngram_size=5` (`min_p=0.06`).

---

## 3. Benchmark Verification: SFT Progression & Final Metrics

### Overall SFT Stage Progression
Evaluated on the clean validation split (`data/processed_sft_120m_v2/val.bin`, 536,117 tokens, 1,744 documents):

| Metric | Base Pretrain (step 9536) | SFT v1 (step 9636) | SFT v2 (step 9786) | SFT v3 (step 9886) | SFT v4 Final (step 10036) | Delta (Final vs. Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Masked SFT Loss** | 2.2179 | 2.1384 | 2.0722 | 1.9882 | **1.9777** | **-0.2402** |
| **Masked SFT Perplexity** | 9.19 | 8.49 | 7.94 | 7.30 | **7.23** | **-1.96** |
| **Unmasked CE Loss** | 2.4039 | 2.3683 | 2.3354 | 2.3359 | **2.3352** | **-0.0687** |
| **Unmasked CE Perplexity** | 11.07 | 10.68 | 10.33 | 10.34 | **10.33** | **-0.74** |
| **Supervised Fraction** | 60.5% | 60.5% | 60.5% | 55.0% | 55.0% | - |
| **Evaluated Documents** | 200 | 200 | 200 | 1,744 | **1,744** | **+1,544** |
| **Evaluated Tokens** | 53,271 | 53,271 | 53,271 | 536,117 | **536,117** | **+482,846** |

### Per-Category Masked SFT Loss Progression
| Category | Base (9536) Loss (PPL) | SFT v2 (9786) Loss (PPL) | SFT v4 Final (10036) Loss (PPL) | Net Delta |
| :--- | :---: | :---: | :---: | :---: |
| **`structured_think`** | 1.6851 (5.39) | 1.4466 (4.25) | **1.4018 (4.06)** | **-0.2833** |
| **`drav_chat`** | 3.0476 (21.06) | 3.0211 (20.51) | **2.9289 (18.71)** | **-0.1187** |
| **`eng_chat`** | 2.5318 (12.58) | 2.4580 (11.68) | **2.4074 (11.11)** | **-0.1244** |
| **`lowres_chat`** | 2.2911 (9.89) | 2.1451 (8.54) | **1.9992 (7.38)** | **-0.2919** |
| **`freethink_cont`** | 3.2924 (26.91) | 3.2089 (24.75) | **2.6242 (13.79)** | **-0.6682** |
| **`tool_hermes`** | 0.9369 (2.55) | 0.8811 (2.41) | **0.4601 (1.58)** | **-0.4768** |

### Held-Out Bits Per Byte (BPB) and Dravidian Tag Routing Accuracy
| Language | ISO | Script | 41M Baseline | 120M Base (9536) | 120M SFT v4 (10036) | Tag Accuracy | Tokens/Byte |
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
| **8-Language Dravidian Mean**| - | 8 Languages | - | 91.25% | **94.14%** (Tag Acc) | **94.14%** | - |
| **English** | `eng` | Latin | 1.653 | **1.3929** | 1.4533 | - | 0.3934 |

---

## 4. Internalized FreeThink Engine Audit

### Operational Principles
- **Automatic Command Stripping (`parse_internal_command`):** User inputs prefixed with `/think`, `/freethink`, `/free-think`, `/nothink`, `/no-think`, `/chat`, or `/base` are intercepted, mode-routed, and stripped of the command prefix prior to prompt construction and inference.
- **Internal `</freethink>` Closure:** FreeThink sessions close internally with `sess.internal_open_tag == "<freethink>"`, `sess.internal_close_tag == "</freethink>"`, `sess.closed_freethink == True`, and `sess.internal_transcript == "<freethink>\n{clean_body}\n</freethink>"`.
- **Zero Raw Tag Leaks:** Raw tags (`<think>`, `</think>`, `<freethink>`, `</freethink>`, `/think`, `/freethink`) are strictly stripped before reaching standard output, Web Studio SSE events, or HTML report cards.
- **Showcase Audit Results (`out/freethink_showcase.html`):**
  - Evaluated sessions: **72**.
  - Raw command or tag leaks: **0**.
  - Native script fidelity: **72/72 (100.0%)**.
  - JavaScript syntax validation: **PASSED (0 syntax errors under `node --check`)**.
  - Collapsible `<think>` drawers: **Verified functional**.

---

## 5. Pure-Python Multimodal Subsystem Audit

Located in `features/multimodal.py`, the multimodal subsystem implements:
1. **Recurrent SSM Visual Patch State Prefill (`SSMVisualProjector`, `prefill_visual_state`):**
   - Projects non-overlapping 2D raster patches into `n_embd` using a deterministic Xavier/orthogonal projection derived from `model.tok_emb.weight` plus 2D sinusoidal positional embeddings.
   - Patches are sequentially stepped through `block.step(x_p, state)`.
   - Fixed recurrent memory: `(B, n_head, head_dim, head_dim)` per layer regardless of patch count or resolution.
2. **Grounded Visual Descriptor (`describe_image`):**
   - Deterministically calculates resolution, aspect ratio, color mode, dominant palette names (RGB/HSL), luminance (Rec.601), RMS/Michelson contrast, Sobel edge density, $2\times2$ quadrant layout, and EXIF metadata.
   - Formats a structured description localized to all 9 supported languages.
3. **Zero-Dependency Document Parser (`extract_document_text`):**
   - Pure-Python UTF-8 extraction preserving Dravidian scripts (`Knda`, `Taml`, `Telu`, `Mlym`, `Orya`) from:
     - PDF (`.pdf` via raw character parsing and `/FlateDecode` zlib decomp)
     - Microsoft Word (`.docx` via `zipfile` XML parsing of paragraphs and tables)
     - HTML/XML (`.html`, `.htm`, `.xml`)
     - CSV/TSV (`.csv`, `.tsv`)
     - JSON/JSONL (`.json`, `.jsonl`)
     - Plaintext & Code (`.txt`, `.md`, `.py`, `.rst`)

---

## 6. Release Candidate Directory Inventory (`release/parentheses-0.9-120M-candidate/`)

| Path / File | Status | Size / Notes |
| :--- | :---: | :--- |
| `modeling_parentheses.py` | Present | Remote code implementation of `ParenthesesForCausalLM` |
| `configuration_parentheses.py` | Present | Remote code configuration `ParenthesesConfig` |
| `config.json` | Present | Step 10035, tied embeddings, 119,702,272 params |
| `tokenizer.json` | Present | 32,768 vocab BPE (`v3_32k`) akshara-preserving |
| `tokenizer_config.json` | Present | Special tokens map (19 special tokens) |
| `model.safetensors` | Present | 478.8 MB (FP32 final weights) |
| `model.bf16.safetensors` | Present | 239.4 MB (BF16 final weights) |
| `model/` | Present | `backbone.py`, `config.py`, `embedding_head.py`, `selective_linear_attention.py`, `vocab_surgery.py`, `__init__.py` |
| `features/` | Present | `free_think.py`, `modular_free_think.py`, `multimodal.py`, `conversation_memory.py`, `__init__.py` |
| `tokenizer/` | Present | Subdirectory mirror of `tokenizer.json` & `tokenizer_config.json` + `README.md` |
| `weights/` | Present | Subdirectory reference with `WEIGHTS_REFERENCE.md` |
| `CHECKPOINT_REFERENCE.md` | Present | Checkpoint catalog connecting SFT v4 step 10036 to pretraining sessions |
| `chat.py` | Present | Interactive terminal CLI supporting multimodal, FreeThink, direct chat, thinking |
| `app.py` | Present | Zero-dependency dark-mode Web Studio with multimodal upload and streaming SSE |
| `load_example.py` | Present | Self-contained Python CLI loader for all 4 inference modes |
| `freethink_showcase.html` | Present | Interactive HTML demonstration showcase with collapsible reasoning |
| `README.md` | Present | Official Hugging Face Model Card with YAML metadata |
| `RELEASE_NOTES.md` | Present | Synchronized release notes for v0.9-120M |
| `QUICKSTART.md` | Present | Step-by-step user quickstart guide |
| `LICENSE_DECISION.md` | Present | Legal & licensing decision documentation for maintainer Joel |
| `RELEASE_CHECKLIST.md` | Present | Release operations and deployment checklist |

---

## 7. Packaging & Checksums Audit

### Release Code Archive
- **Path:** `release/dist/parentheses-0.9-120M-code.zip`
- **Scope:** Clean source code, modeling modules, features, tests, and documentation (excluding large `.pt` or `.safetensors` weight binaries, `.log`, `.pid`, `.err`, and `__pycache__` artifacts).

### Checksum Verification (`release/dist/SHA256SUMS.txt`)
All candidate release files and the code archive have had their SHA-256 hashes generated and recorded in `release/dist/SHA256SUMS.txt`:

| SHA-256 Hash | Relative Path |
| :--- | :--- |
| `0bbe993de9124e63a472aa1373493991fe67869cb9ef6d4cb9f2e79a519f612c` | `CHECKPOINT_REFERENCE.md` |
| `9540b950c94f963dcb8556f968426ca4e144a664d9870416c6c77929e6fd1024` | `FREETHINK_REPORT.md` |
| `57e1015382f380175e46d17cc8c699102ddb50fa7964fd0ceb459736614f5e08` | `FREETHINK_SHOWCASE.md` |
| `5e4439dac7c196895776632db341b4104c8ba03bc4b9dfb47be0253f4c1575d6` | `LICENSE_DECISION.md` |
| `4278c4f266fad1104d31afb723203a1c0e0532e8353d4dafa31dbdf54e07c420` | `QUICKSTART.md` |
| `99f7a860bc6dfbc95a9755d08e092baa92f7bde9b0f68a68a119a3bcbdc801be` | `README.md` |
| `d7e0131d09653800b2605b31b1f64e03643501b20ab53dc07dfabe416e963aa2` | `RELEASE_CHECKLIST.md` |
| `d12bbd9ae3505ffbd82e7d3a5ebbd37e86b60922f229f16bb9c8b8d14bb76f9a` | `RELEASE_NOTES.md` |
| `ce1ef45a058b162a1fea5a03e316c40306b039e82d984ee4770cb84f64eff6df` | `app.py` |
| `d9bea6a0f77634d6e11f9d3aec36cedd274d834fb3104fc50ee1c2a0e131df88` | `chat.py` |
| `09d59df3a0e72ec483c0c9114b6a8c167ff6439b41d525f907ae161e31b4f07e` | `config.json` |
| `e9f082571db94908dfa2ae0ba048c898ea0f8230f2dffe0a7d655a247485203a` | `configuration_parentheses.py` |
| `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `features/__init__.py` |
| `c49c1f16a316577c88b32ce33b3ba50b649201f17d166d568fb68a000ba6c351` | `features/conversation_memory.py` |
| `19f6cc7c207cecc7d661660254c7f92c57c8e1c15de98b078c7a5823b2adb26f` | `features/free_think.py` |
| `ef888f1461b0cc5d85c52dac2bba49cf1d1aa3adc67eb1f0bb776ac98436122d` | `features/modular_free_think.py` |
| `a24fb3d3e7457c0da8532fc043cdbcb6f2e7d75b2793f3fb09f2e06b23c5306f` | `features/multimodal.py` |
| `eb2be12d5d8b6b91f442bca5e5048adc48442e7d0c9a8b9ae20c6dd496fff0c8` | `freethink_showcase.html` |
| `9b18f624601fc8ad6079b8c6db1752109f7d834cdc1e0b8f14370d3c635b7b89` | `load_example.py` |
| `71f5d6593555d3312277e573a4e062eb994f34d523eaf7f2e2610b4e89c1e9ec` | `model.bf16.safetensors` |
| `8d5e895e7117e19f1b2745be3aa34cf4af1dbc41c0f12a5663d24ca47e097792` | `model.safetensors` |
| `6be0766242f80e424777440c2cecb272ee5a8a67e6df5fc54806608122b70c70` | `model/__init__.py` |
| `3f354f28a97183797135af1afd2e3629d4ab409a1c7e4522953d8976d622f810` | `model/backbone.py` |
| `69a52c5a53b57b9cce70a529ab7aec73411e6821e91639085ecdace9055037b2` | `model/config.py` |
| `1503b7e18a8e44cb6945e18a0071c4c981171b88f5355858111145fe2ec80898` | `model/embedding_head.py` |
| `237ef52e82706b13a1bddd7113ec0b98960bda0f2ec06700f7301556c2692d0a` | `model/selective_linear_attention.py` |
| `0e8eeed6f32076d12e431b4feae713b36374edab17e27ee348a16ac0a035dff4` | `model/vocab_surgery.py` |
| `d22a4b53fd19b10da65594304f5535922ec7729c197a75d9127552da8b8de25f` | `modeling_parentheses.py` |
| `7a9b4187d5bc000134b5f646e45e8e5a0435adb0e7d20afb99ac4bde0a9808a6` | `tokenizer.json` |
| `7603985bb9ae2a4c7fbe94c7f2f887372054faba52f45ff70d2d98378cb58f22` | `tokenizer/README.md` |
| `7a9b4187d5bc000134b5f646e45e8e5a0435adb0e7d20afb99ac4bde0a9808a6` | `tokenizer/tokenizer.json` |
| `752e016fbbc28017f964a21f889e7a591174243de5597ac571b4046756b828fe` | `tokenizer/tokenizer_config.json` |
| `752e016fbbc28017f964a21f889e7a591174243de5597ac571b4046756b828fe` | `tokenizer_config.json` |
| `0ecc7a4a5bfb54e24ac091f19663a31739176724795de4b61907cd59178cf823` | `weights/WEIGHTS_REFERENCE.md` |
| `0e3a4b0a067806b47fd613befc312cdbc1aa80cc70982ad82ac38a1784eaab09` | `parentheses-0.9-120M-code.zip` |

---

## 8. Open Decisions & Recommendation for Maintainer Joel

1. **Licensing Decision (`LICENSE_DECISION.md`):**
   - The candidate model incorporates 1.6M distilled rows (~0.18% of tokens) in pretraining Sessions 4 and 5 from NVIDIA API trial endpoints.
   - Option A: Release under Research-Only / Non-Commercial license (safe, immediate).
   - Option B: Clean pretraining checkpoint branch from step 3,815 (fully permissive / commercial CC BY-SA 4.0).
   - *Recommendation:* Joel can review and select the desired license tag prior to pushing to Hugging Face Hub on October 22.
2. **Hugging Face Hub Publish:**
   - Candidate directory is fully self-contained with `trust_remote_code=True` support. Upload `release/parentheses-0.9-120M-candidate/` to `uncoalesced/parentheses-0.9-120M`.
3. **GitHub Release:**
   - Tag `v0.9-120M` on repository `parentheses-0.9`, attach `release/dist/parentheses-0.9-120M-code.zip` and `release/dist/SHA256SUMS.txt`.

---

**Auditor Conclusion:** Parentheses 0.9-120M passes all documentation, architectural, benchmark, multimodal, FreeThink encapsulation, and packaging criteria. The release package is 100% synchronized and ready for shipping.
