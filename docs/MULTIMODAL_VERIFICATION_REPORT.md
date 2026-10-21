# Parentheses 0.9-120M Multimodal (Image + Document) Subsystem Verification Report

**Date:** 2026-10-17  
**Architecture:** Mamba-2 / RWKV / Selective Linear Attention (State Space Duality — $O(T)$ Parallel Training, $O(1)$ Recurrent State Inference)  
**Status:** **100% PASS (15/15 Unit & Integration Tests + Real 120M Checkpoint + Web Studio & CLI Self-Tests Verified on CPU)**

---

## 1. Executive Summary & Architectural Invariant

Per the Parentheses architectural mandate (`AGENTS.md`), Parentheses is a **Recurrent Selective Linear Attention State Space Model (SSM)**, not a quadratic Transformer. The Multimodal Subsystem (`features/multimodal.py`) and its integration into Modular Free Think / RAG (`features/modular_free_think.py`), CLI (`scripts/chat.py`, `release/parentheses-0.9-120M-candidate/chat.py`), and Web Studio (`app.py`, `release/parentheses-0.9-120M-candidate/app.py`) are designed strictly around the constant-$O(1)$ recurrent state update law:

$$S_t = \alpha_t S_{t-1} + \frac{k_t \otimes v_t}{\sqrt{d_{\text{head}}}}, \qquad y_t = q_t S_t$$

- **Zero $O(N^2)$ ViT Self-Attention Matrices:** Images are tokenized into 2D raster patches and stepped sequentially through `block.step(x_patch, state)`.
- **Constant $O(1)$ Recurrent Memory Proof:** Whether stepping 16 patches ($64 \times 64$, $P=16$), 64 patches ($64 \times 64$, $P=8$), 256 patches ($128 \times 128$, $P=8$), or 10 consecutive images, the recurrent state tensor list `states` retains the exact same shape `(B, n_head, head_dim, head_dim)` per layer (`[(1, 12, 64, 64)] * 16` = **3,145,728 bytes** on the real 120M SFT v2 model) and exact same byte footprint (`state_memory_bytes(states)`).
- **Zero External Checkpoint Overhead:** Visual patch projection (`SSMVisualProjector`) deterministically derives a Xavier-scaled projection matrix from `model.tok_emb.weight` combined with 2D sinusoidal raster positional embeddings.
- **Zero Extra Third-Party Dependencies:** Built entirely with `.venv` standard libraries (`torch`, `numpy`, `Pillow`, `zipfile`, `zlib`, `xml.etree.ElementTree`, `html.parser`, `csv`, `json`).
- **Hidden Control & Multimodal Commands:** Inline `/image <path>` and `/doc <path>` directives in user prompts are parsed and ingested into the active session/RAG store and automatically stripped before tokenization, and internal `<think>`/`</think>` and `<freethink>`/`</freethink>` tags remain strictly hidden from visible output.

---

## 2. Delivered Components

| File | Description |
| :--- | :--- |
| [`features/multimodal.py`](file:///E:/Parentheses/parentheses-0.9/features/multimodal.py) | Core multimodal subsystem: `SSMVisualProjector`, `prefill_visual_state()`, `ImageProcessor`, `describe_image()`, `VisualDescription`, `DocumentProcessor`, `extract_document_text()`, `extract_pdf_text()`, `extract_docx_text()`, `extract_html_text()`, `extract_csv_text()`, `extract_json_text()`, `create_pdf_bytes()`, `create_docx_bytes()`. |
| [`release/parentheses-0.9-120M-candidate/features/multimodal.py`](file:///E:/Parentheses/parentheses-0.9/release/parentheses-0.9-120M-candidate/features/multimodal.py) | Synced release-candidate copy of the multimodal subsystem. |
| [`features/modular_free_think.py`](file:///E:/Parentheses/parentheses-0.9/features/modular_free_think.py) | Updated `IngestedStore.ingest()` to automatically extract text/descriptions from `.pdf`, `.docx`, `.html`, `.xml`, `.csv`, `.tsv`, `.json`, `.jsonl`, and images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`); added `IngestedStore.ingest_image()`, `IngestedStore.ingest_text()`, `ModularFreeThinkSession.ingest_image()`, `session.image_states`, and `--image` CLI flag. Synced to `release/parentheses-0.9-120M-candidate/features/modular_free_think.py`. |
| [`scripts/chat.py`](file:///E:/Parentheses/parentheses-0.9/scripts/chat.py) & [`release/parentheses-0.9-120M-candidate/chat.py`](file:///E:/Parentheses/parentheses-0.9/release/parentheses-0.9-120M-candidate/chat.py) | Added `--image` and `--doc` (`--ingest`) CLI arguments plus inline `/image <path>` and `/doc <path>` commands that ingest images and documents into the active `IngestedStore` and `SSMVisualProjector` while hiding `/image` and `/doc` directives from visible output. |
| [`app.py`](file:///E:/Parentheses/parentheses-0.9/app.py) & [`release/parentheses-0.9-120M-candidate/app.py`](file:///E:/Parentheses/parentheses-0.9/release/parentheses-0.9-120M-candidate/app.py) | Added `GenerationEngine.ingest_multimodal()`, `POST /api/ingest` endpoint (accepting base64 or path uploads for PDF, DOCX, HTML, CSV, JSON, PNG, JPEG, WebP, BMP), and the Web Studio `📎 Attach Image / Document` upload button with live ingested chunk badges. |
| [`tests/test_multimodal_suite.py`](file:///E:/Parentheses/parentheses-0.9/tests/test_multimodal_suite.py) | Comprehensive 15-test `unittest` verification suite covering image patch tokenization, $O(1)$ SSM state prefill, `slab_v2` state prefill, visual feature/quadrant/metadata analysis, Dravidian visual summaries, PDF, DOCX, HTML, CSV/TSV/JSON/JSONL, mixed BM25 + Vector (`turbovec`) retrieval, end-to-end BPE generation, real 120M checkpoint (`step_9786_final.pt`) evaluation, and Web Studio `/api/ingest`. |

---

## 3. Test Suite Verification Matrix (`tests/test_multimodal_suite.py`)

Executed on CPU (`$env:CUDA_VISIBLE_DEVICES='-1'`, `$env:PYTHONIOENCODING='utf-8'`):

| Test # | Test Name | Modality & Invariant Verified | Result |
| :---: | :--- | :--- | :---: |
| **01** | `test_01_image_loading_and_patch_tokenization` | PNG, JPEG, BMP, WebP, raw bytes, `data:image/png;base64,...`, `torch.Tensor` `(C,H,W)`/`(B,C,H,W)`, `np.ndarray`, and `(1, 16, 768)` raster patch extraction. | **PASS** |
| **02** | `test_02_ssm_o1_recurrent_visual_state_prefill` | Proves `prefill_visual_state()` keeps `state_shapes` `[(1, 4, 16, 16)] * 4` and `state_memory_bytes` strictly constant across 16, 64, and 256 patches and across 10 sequential images, with deterministic state outputs. | **PASS** |
| **03** | `test_03_ssm_slab_v2_visual_state_prefill` | Verifies `prefill_visual_state()` preserves constant $O(1)$ state + causal conv history shapes when `slab_v2=True`. | **PASS** |
| **04** | `test_04_visual_feature_and_quadrant_analysis` | Bright vs. dark luminance, uniform vs. high-contrast checkerboard, Sobel + 1px Nyquist edge density, $2 \times 2$ quadrant layout, and PNG `tEXt` metadata extraction. | **PASS** |
| **05** | `test_05_dravidian_localized_image_descriptions` | Localized visual summaries across `eng`, `kan` (Kannada), `tam` (Tamil), `tel` (Telugu), and `mal` (Malayalam). | **PASS** |
| **06** | `test_06_pdf_generation_and_extraction` | Pure-Python PDF generation & parsing: `/FlateDecode` (`zlib`) streams, uncompressed streams, `TJ` kerning arrays, octal/parenthesis escapes, and UTF-16BE (`<FEFF...>`) Kannada Unicode text. | **PASS** |
| **07** | `test_07_docx_generation_and_extraction_dravidian` | Pure-Python OpenXML `.docx` extraction of English, Kannada, Tamil, Telugu, Malayalam, and Odia paragraphs (`<w:p>`, `<w:t>`) and tables (`<w:tbl>`, `<w:tr>`, `<w:tc>`). | **PASS** |
| **08** | `test_08_html_and_xml_parsing` | Strips `<script>` and `<style>` blocks while preserving `<title>`, `<h1>`, `<p>` (including Kannada text), and `<table>` rows. | **PASS** |
| **09** | `test_09_csv_tsv_and_json_jsonl_structured_ingestion` | Converts CSV, TSV, JSON arrays, and JSONL streams (with Kannada & Tamil Unicode fields) into self-contained paragraph blocks. | **PASS** |
| **10** | `test_10_ingested_store_mixed_multimodal_bm25_and_vector` | Indexes `.pdf`, `.docx` (Kannada), `.html`, `.csv`, and `.png` (with metadata) into `IngestedStore` and verifies accurate chunk retrieval on both **BM25** and **TurboVec (`vector`)** backends. | **PASS** |
| **11** | `test_11_modular_freethink_end_to_end_bpe_multimodal` | End-to-end BPE `ModularFreeThinkSession` generation grounded in ingested PDF/DOCX/PNG store + `session.ingest_image()`, verifying internal `</freethink>` closure (`session.closed_freethink == True`) and zero role/control tag leaks. | **PASS** |
| **12** | `test_12_security_and_corrupted_input_boundaries` | Verifies clean `ValueError` / `FileNotFoundError` / `RuntimeError` exceptions on corrupted PDFs, bad ZIP/DOCX archives, corrupted images, invalid patch grid ratios, and causal-attention models. | **PASS** |
| **13** | `test_13_real_120m_ssm_visual_prefill_and_freethink` | Real `step_9786_final.pt` (120M SFT v2, 16L, 12H, $d=768$) SSM visual prefill (`[(1, 12, 64, 64)] * 16` = 3,145,728 bytes constant across $64 \times 64$ and $128 \times 128$ images) + grounded `ModularFreeThinkSession` response. | **PASS** |
| **14** | `test_14_real_120m_pdf_and_dravidian_docx_grounded_qa` | Real 120M checkpoint (`step_9786_final.pt`) grounded QA over ingested `.pdf` + Kannada `.docx` in both English and Kannada (`kan`), plus `scripts/chat.py --image --doc` verification with hidden `/image` & `/doc` directives. | **PASS** |
| **15** | `test_15_web_studio_api_ingest_endpoint` | Spins up Web Studio HTTP server (`app.py`) on an ephemeral port and verifies `POST /api/ingest` for base64 `.pdf` and `.png` uploads, confirming chunk indexing and `SSMVisualProjector` recurrent state memory reporting (`3,145,728` bytes). | **PASS** |

---

## 4. Reproduction Commands

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONIOENCODING='utf-8'

# 1. Run the 15-test multimodal verification suite (includes real 120M checkpoint & Web Studio tests)
E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe tests\test_multimodal_suite.py

# 2. Run the multimodal module self-test
E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe -m features.multimodal --self-test

# 3. Run the Modular Free Think (RAG + Multimodal) self-test
E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe -m features.modular_free_think --self-test

# 4. Run the CLI chat self-test (repo & release candidate)
E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe scripts\chat.py --self-test
E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe release\parentheses-0.9-120M-candidate\chat.py --self-test

# 5. Run the Web Studio self-test (verifies GET /, GET /api/health, POST /api/ingest, POST /api/stream)
E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe release\parentheses-0.9-120M-candidate\app.py --self-test
```
