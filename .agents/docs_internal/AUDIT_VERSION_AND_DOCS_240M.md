# Master Version & Documentation Audit Report: Parentheses 1.0-240M (Project Tanzanite)

**Auditor:** Version & Documentation Auditor Subagent  
**Date:** November 12, 2026  
**Target Repository:** `E:\Parentheses\parentheses-0.9\`  
**Active Release Target:** `parentheses-1.0-240M-tanzanite` (`slab_v4`, 253,544,176 parameters)  
**Historical Predecessors:** `parentheses-0.9-130M-v3` (`slab_v3`, 148.38M), `parentheses-0.9-120M` (119.7M), `parentheses-0.9-41M` (40.9M)

---

## 1. Executive Summary & Audit Scope

This audit scanned all markdown and documentation assets in `E:\Parentheses\parentheses-0.9\` to prepare the repository for the certified golden release of **Parentheses 1.0-240M-tanzanite**.

### Critical Architectural Mandate (Re-Affirmed)
Parentheses is **NOT a Transformer / GPT model**. It does **NOT** use quadratic causal self-attention ($\mathcal{O}(T^2)$ Softmax matrices) or growing KV-caches. Parentheses is a **from-scratch Recurrent / State Space Duality (SSD) model** belonging to the Mamba-2 / RWKV / Selective Linear Attention family:
- **Training (Parallel Form):** Chunked associative linear attention / semi-separable state space duality operating in linear $\mathcal{O}(T)$ complexity.
- **Inference (Recurrent Form):** Constant $\mathcal{O}(1)$ recurrent memory state updates ($S_t = \alpha_t S_{t-1} + \beta_t(k_t \otimes v_t)$, $y_t = q_t S_t$) with zero growing KV-cache and zero sequence Softmax.

---

## 2. File Categorization: Update Directly vs. Preserve for Lineage

Across the repository, 68 first-party markdown files were categorized into three operational classes:

### Category A: Canonical & Public Docs (Update Directly to 1.0-240M-tanzanite)
These are primary documentation and public-facing files that end users, developers, and packagers read. They must describe the current golden release (`parentheses-1.0-240M-tanzanite`):

| File Path | Role | Current Subject | Action Required |
|:---|:---|:---|:---|
| `docs/DOCUMENTATION.md` | Primary reference manual (Parts A & B) | 0.9-130M-v3 (`slab_v3`) | **Update completely** to 1.0-240M (`slab_v4`), updating Part A APIs/NPU and Part B mathematical derivations. |
| `README.md` | Public repository landing page | 0.9-130M preview | **Update** badges, overview, quickstart commands, parameter counts, architecture summary, and at-a-glance tables. |
| `MODEL_CARD.md` (root) | Canonical Hugging Face model card | 0.9-120M (outdated) | **Replace** with certified 1.0-240M card containing YAML metadata, certified gates G1–G9, and NPU specifications. |
| `RELEASE_NOTES.md` | Official release notes | 0.9-120M (Oct 17, 2026) | **Prepend** official 1.0-240M-tanzanite release notes while preserving earlier release notes below as historical archive. |
| `docs/ARCHITECTURE.md` | Conceptual architecture guide | 0.9-130M-v3 (`slab_v3`) | **Update** primary specs to 1.0-240M (`slab_v4`), highlighting wide-and-shallow 16-layer scaling and tri-channel reset. |
| `docs/SETUP.md` | Step-by-step user installation guide | 0.9-130M-v3 checkpoint paths | **Update** checkpoint directory names, download commands, and Python API snippets to 1.0-240M. |
| `docs/ROADMAP.md` | Project development roadmap | 1.0-240M listed as "In progress" | **Update** status table: mark 1.0-240M as **Certified Golden Release**. |
| `TRAINING_DATA.md` | Training data overview | Covers up to S4/S5 | **Add** Session 6–10 pretraining, precision SFT (abstention), and preference tuning provenance. |
| `.agents/ARCHITECTURE.md` | Agent engineering reference | Mixed 130M / 240M target | **Update** current target section to reflect 1.0-240M (`slab_v4`) as the golden baseline. |

### Category B: Historical Lineage & Evaluation Artifacts (Strictly Preserved)
These files represent immutable scientific and engineering records of earlier milestones, training runs, and evaluations. They **MUST NOT** be overwritten:

| File Path | Historical Milestone Recorded | Preservation Rule |
|:---|:---|:---|
| `docs/HISTORY.md` | Full project lineage (64M $\to$ 41M $\to$ 120M $\to$ 130M-v3 $\to$ 240M) | **Preserve historical chapters**. Only append 1.0-240M capstone section if desired. |
| `docs/EVAL_JOEL_SUITE_ANALYSIS.md` | 0.9-120M evaluation on Joel's Suite | **Preserve unmodified**. |
| `docs/EVAL_S9_INDEPENDENT_REPORT.md` | Session 9 Independent Multi-Agent Evaluation | **Preserve unmodified**. |
| `docs/EVAL_S9_STRICT_REPORT.md` | Session 9 Strict Gate Verification Report | **Preserve unmodified**. |
| `docs/FREETHINK_VERIFICATION_REPORT.md` | 0.9-120M FreeThink & Tag-Hiding Audit | **Preserve unmodified**. |
| `docs/MULTIMODAL_VERIFICATION_REPORT.md` | 0.9-120M Multimodal Prefill & Document Parsing | **Preserve unmodified**. |
| `docs/RELEASE_READINESS_AUDIT.md` | 0.9-120M Release Readiness Review | **Preserve unmodified**. |
| `docs/S10_BUILD_LOG.md` | Session 10 Build Log & Decisions | **Preserve unmodified**. |
| `docs/S10_PLAN_IMPROVED_VS_NOT.md` | Session 10 SFT Comparison Study | **Preserve unmodified**. |
| `docs/HANDOFF_SESSION_3_240M.md` | Session 3 Training Handoff | **Preserve unmodified**. |
| `docs/SESSION_3_TO_8_MASTER_BLUEPRINT.md` | Multi-Session Master Blueprint | **Preserve unmodified**. |
| `GATES.md` | Historical Overnight 0.9-NM Gates | **Preserve unmodified**. |
| `DATA_LEDGER.md` | 2026-10-13 Disk Audit | **Preserve unmodified**. |

### Category C: Internal Agent Task & Review Logs (Preserved)
All files in `.agents/done/`, `.agents/reviews/`, `.agents/open/`, and `out/` (e.g. `out/swarm/`, `out/cmp_130m_v3/`) remain historical session transcripts and operational logs.

---

## 3. Detailed Audit of Version References (0.9, 130M, 120M, 41M)

### 3.1 `README.md`
- **Line 12:** `### parentheses-0.9-130M (Project Tanzinite)`  
  $\to$ Update to: `### parentheses-1.0-240M-tanzanite (Project Tanzanite)`
- **Lines 25–26:** Badges pointing to `v0.9--130M--v3--golden` and `uncoalesced/parentheses-v0.9-130M-v3`  
  $\to$ Update to: `v1.0--240M--tanzanite--golden` and `uncoalesced/parentheses-1.0-240M-tanzanite`.
- **Line 46:** Describes current model as "148 million parameters" preview.  
  $\to$ Update to: "253.54 million parameters" certified golden release.
- **Lines 71–89:** Quick setup downloads from `uncoalesced/parentheses-v0.9-130M-v3` into `checkpoints/parentheses-0.9-130M-v3/`.  
  $\to$ Update to: `uncoalesced/parentheses-1.0-240M-tanzanite` and `release/parentheses-1.0-240M-tanzanite/` (or `checkpoints/parentheses-1.0-240M-tanzanite/`).
- **Lines 102–104:** Discusses 130M evaluation scores (5/24 on fresh reasoning).  
  $\to$ Update to: Session 10 certified gates (G1 Clean-Stop: 95.67%, G2 Tool-Call Leak: 0.0%, G3 Dravidian Format: 96.25%, G4 Dravidian Script: 100%, G6 Epistemic Honesty: 94.0%, G7 Benign Refusal: 0.0%, G8 BPB Drift: +0.012, G9 16k Recurrent Stability: 100%).
- **Lines 130–138:** Recurrent state size (about 1.6 MB in bf16 / 3.64 MB in fp32 for 12 heads).  
  $\to$ Update to: 16 heads $\times$ 64 $\times$ 64 per layer, total recurrent state across 16 layers $\approx 4.85\text{ MB}$ in fp32 ($2.42\text{ MB}$ in bf16).
- **Lines 142–151:** Roadmap table lists 1.0-240M in progress.  
  $\to$ Update table to reflect 1.0-240M as Certified Golden Release.
- **Lines 165–177:** "Model at a glance" table:  
  $\to$ Update Model: `parentheses-1.0-240M-tanzanite`, Params: `253.54M`, Layers/Width/Heads: `16 / 1024 / 16 heads of 64`, State: `4.85 MB`, License: `Apache 2.0`.

### 3.2 `MODEL_CARD.md` (Root)
- Currently contains outdated text from **Parentheses 0.9-120M** (119.7M params, 32 layers, width 512, 8 heads).
- Must be completely synchronized with `release/parentheses-1.0-240M-tanzanite/MODEL_CARD.md` and `docs/MODEL_CARD_240M_DRAFT.md`.
- Architecture: `slab_v4`, 253,544,176 parameters, 16 layers, $d_{\text{model}}=1024$, 16 heads, rank 1024, context 4096 (16k verified stable), BPE `v3_32k` (32,768 vocab).
- Incorporates certified gates G1 through G9 and AMD Ryzen AI / DirectML NPU ONNX export specifications.

### 3.3 `RELEASE_NOTES.md`
- Currently contains release notes for **Parentheses 0.9-120M** (October 17, 2026).
- Action: Add Section 1 for **Parentheses 1.0-240M-tanzanite** (November 2026 release) highlighting:
  1. Wide-and-shallow 16-layer recurrent topology ($d_{\text{model}}=1024$, 16 heads, 253.54M parameters).
  2. Single entry-point document reset wiring (`compute_doc_ids`).
  3. Tri-Channel Boundary Reset Isolation (Token-Shift, Causal Conv1d, Chunked SSD).
  4. Layer-0 $V_0$ Value Residual Highway & Token-Shift ($\mu$-mixing).
  5. Calibrated Epistemic Abstention (94.0% honesty, 0.0% false refusals on benign prompts).
  6. DirectML / AMD Ryzen AI NPU static-shape recurrent step ONNX export (< 1.15 GB, $< 1.0 \times 10^{-4}$ max logit error).
  7. Retain the 0.9-120M release notes beneath a clear `--- Historical Lineage: Parentheses 0.9-120M ---` divider.

### 3.4 `docs/SETUP.md`
- **Lines 1, 3:** "Setting up parentheses-0.9" $\to$ "Setting up parentheses-1.0-240M".
- **Lines 152, 174, 182, 199, 205:** Checkpoint paths pointing to `parentheses-0.9-130M-v3` $\to$ `parentheses-1.0-240M-tanzanite` (or `release/parentheses-1.0-240M-tanzanite`).
- **Lines 424, 452–456:** Python API snippet using `PRESETS["parentheses-0.9-130M-v3"]` $\to$ update to `PRESETS["parentheses-1.0-240M"]` and add the standalone loader from `models.model_slab_v4.SlabV4Model`.

### 3.5 `docs/ARCHITECTURE.md`
- **Lines 1, 3:** "How parentheses-0.9 works" / `parentheses-0.9-130M-v3` $\to$ "How parentheses-1.0-240M works" (`slab_v4`).
- **Lines 26–43:** "At a glance" table: Update to 253,544,176 parameters, 16 layers, 1024 width, 16 heads of 64, 4.85 MB state.
- **Section 4 ("Why wide and shallow"):** Retain the historical motivation for halving depth from 32 to 16 layers, and add the 1.0-240M scaling step: maintaining 16 layers while expanding width from 768 to 1024 and heads from 12 to 16, preserving the optimal 16.29 gradient ratio while boosting recurrent rank to 1024 per layer (+33%).
- **Section 9:** Detail the Tri-Channel Document Boundary Reset Isolation implemented in `slab_v4`.

---

## 4. Deep-Dive Specification for `docs/DOCUMENTATION.md`

`docs/DOCUMENTATION.md` is the repo's technical cornerstone (1,150 lines). It currently documents `parentheses-0.9-130M-v3` (`slab_v3`). Below are the exact diffs and structural updates required to document `parentheses-1.0-240M-tanzanite` (`slab_v4`).

### 4.1 Part A: Using the Code (Exact Required Updates)

#### 1. Title & Scope (Lines 5–13)
```diff
-# Parentheses 0.9-130M-v3: how the model works
+# Parentheses 1.0-240M-tanzanite: how the model works
 
 This document has two parts. Part A: using the code is a practical reference for running, loading and exporting the model from this repository. Part B: the math covers sections 1 to 19, which explain how the recurrence works and trace each formula to the code. For a gentler explanation, read [ARCHITECTURE.md](ARCHITECTURE.md); for installation and weights, read [SETUP.md](SETUP.md).
 
 Parentheses is a recurrent language model. It reads one token at a time and keeps a fixed-size memory: one small matrix per attention head per layer. Each token updates that memory with a decay and a write, and reads it back with a single matrix-vector product. There is no softmax over past positions, no key-value cache and no positional encoding. The same layer can also be computed in parallel across a whole sequence, and that is how it trains.
 
-Part B covers the mathematics and runtime behaviour of the `parentheses-0.9-130M-v3` preset in `model/config.py`. It describes what the code computes. Every formula can be traced to a line in `model/selective_linear_attention.py`, `model/backbone.py` or the decoding features. Training (data, schedule, hardware) is out of scope.
+Part B covers the mathematics and runtime behaviour of the `parentheses-1.0-240M` preset (`slab_v4`) in `model/config.py` and `models/model_slab_v4.py`. It describes what the code computes. Every formula can be traced to a line in `models/model_slab_v4.py`, `models/slab_v4_boundary.py`, `model/selective_linear_attention.py`, or the decoding features. Training (data, schedule, hardware) is out of scope.
```

#### 2. Section A.1 Repository Map (Lines 56–79)
Add explicit documentation for:
- `models/`: The primary `slab_v4` implementation: `model_slab_v4.py` (`SlabV4Model`, `SlabV4Config`, `SlabV4Block`, `SlabV4Attention`) and `slab_v4_boundary.py` (`compute_doc_ids`, `token_shift_v4`, `causal_conv1d_v4`, `ssd_chunk_scan_v4`, `ssd_recurrent_step_v4`).
- `release/parentheses-1.0-240M-tanzanite/`: The certified standalone release package containing `model.safetensors`, `model.bf16.safetensors`, `model.pt`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `load_example.py`, `MODEL_CARD.md`, `MANIFEST_sha256.tsv`, and `npu/parentheses_240m_v4_step.onnx`.
- `scripts/export_npu_240m_v4.py`: Standalone exporter producing static-shape recurrent ONNX graphs for AMD Ryzen AI / DirectML.

#### 3. Section A.2 Running the Chat (Lines 83–89)
```diff
-python scripts/chat.py --checkpoint checkpoints/parentheses-0.9-130M-v3 --device cuda
+python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda
```

#### 4. Section A.4 Python API (Lines 122–165)
Provide both loading methods:
1. **Direct Standalone Loader (`models/model_slab_v4.py`):**
```python
import json
import torch
import safetensors.torch
from tokenizers import Tokenizer
from models.model_slab_v4 import SlabV4Model, SlabV4Config

# Load config and initialize model
with open("release/parentheses-1.0-240M-tanzanite/config.json", "r", encoding="utf-8") as f:
    cfg_data = json.load(f)
valid_keys = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd", "chunk_size", "eot_token_id", "dropout", "bias"}
cfg = SlabV4Config(**{k: v for k, v in cfg_data.items() if k in valid_keys})
model = SlabV4Model(cfg).eval().to("cuda")

# Load weights (safetensors or PyTorch .pt)
safetensors.torch.load_model(model, "release/parentheses-1.0-240M-tanzanite/model.safetensors")
tokenizer = Tokenizer.from_file("release/parentheses-1.0-240M-tanzanite/tokenizer.json")
```

2. **Unified Preset Loader (`model/backbone.py`):**
```python
from model.backbone import Parentheses
from model.config import PRESETS

model = Parentheses(PRESETS["parentheses-1.0-240M"])
```

3. **Recurrent Step Inference:**
Document the state tuple `(s_states, conv_states, shift_states)`:
- `s_states`: `(16, B, 16, 64, 64)` float32
- `conv_states`: `(16, B, 3, 3072)` float32
- `shift_states`: `(16, B, 1024)` float32

#### 5. Section A.5 Training and Evaluation Self-Tests (Lines 190–202)
Add self-tests for `models/`:
```bash
python -m models.slab_v4_boundary
python -m models.model_slab_v4
```

#### 6. Section A.6 Exports (Lines 204–214)
```diff
-`scripts/export_npu_130m_v3.py` exports the single-step recurrent graph of the 130M-v3 model to ONNX. The graph has fixed shapes. A step takes one token and a fixed-size state, so no cache grows with the text. The script reads a `.pt` training checkpoint:
-```bash
-python scripts/export_npu_130m_v3.py --checkpoint <path-to-checkpoint>.pt --verify
-```
-In the project's checks, the export is 694 MB, runs on ONNX Runtime with DirectML, and matches PyTorch to within about 2e-5.
+`scripts/export_npu_240m_v4.py` exports the single-step recurrent graph of the 240M-tanzanite model to static ONNX. The graph has strictly fixed shapes, zero sequence allocations, and constant O(1) step execution.
+```bash
+python scripts/export_npu_240m_v4.py --checkpoint release/parentheses-1.0-240M-tanzanite/model.pt --verify
+```
+The verified export is located at `release/parentheses-1.0-240M-tanzanite/npu/parentheses_240m_v4_step.onnx` (< 1.15 GB). Verified on ONNX Runtime with DirectML and CPU execution providers, matching native PyTorch across 32 sequential autoregressive steps to a max logit error < 1.0e-4 (CPU: 3.75e-5, DML: 5.48e-5).
```

---

### 4.2 Part B: The Math (Exact Required Updates)

#### 1. Section 1: Notation (Lines 218–242)
| Symbol | Meaning | Value in 130M-v3 | **Value in 1.0-240M (`slab_v4`)** |
|:---|:---|:---:|:---:|
| $V$ | vocabulary size | 32,768 | **32,768** |
| $d$ | model width (`n_embd`) | 768 | **1024** |
| $L$ | number of blocks (`n_layer`) | 16 | **16** |
| $H$ | heads per block (`n_head`) | 12 | **16** |
| $d_h$ | head dimension, $d / H$ | 64 | **64** |
| $T$ | sequence length in one forward pass | up to 4,096 | **up to 4,096 (16k verified)** |
| $C$ | chunk length in parallel form | 64 | **64** |
| $S_t$ | matrix state of one head | $64 \times 64$ | **$64 \times 64$** |
| Conv channels | depthwise causal conv width | $3d = 2304$ | **$3d = 3072$** |
| Shift state | previous normalized input | 768 | **1024** |

#### 2. Section 2: Model at a Glance & Parameter Accounting (Lines 246–304)
- **Parameters:** Exactly **253,544,176** (~253.54M unique parameters with tied head; 287,098,608 total tensor elements untied).
- **Recurrent State:** Per block: 16 matrices of $64 \times 64$, 3-step conv history of 3072 channels, one shifted input of 1024 channels.
  - Across all 16 blocks: $16 \times 16 \times 64 \times 64 = 1,048,576$ state matrix floats.
  - Conv history: $16 \times 3 \times 3072 = 147,456$ floats.
  - Shift state: $16 \times 1024 = 16,384$ floats.
  - **Total State Footprint:** $1,212,416$ floats = **4.85 MB in fp32** (2.42 MB in bf16).

**Exact Parameter Budget Table (Section 2.1):**
```markdown
| Component | Shape per layer | Per layer | × 16 layers |
| :--- | :--- | ---: | ---: |
| `qkv` projection | $1024 \times 3072$ | 3,145,728 | 50,331,648 |
| output projection $W_o$ | $1024 \times 1024$ | 1,048,576 | 16,777,216 |
| output gate $W_g$ | $1024 \times 1024$ | 1,048,576 | 16,777,216 |
| decay gate $W_\alpha$, $b_\alpha$ | $1024 \times 16 + 16$ | 16,400 | 262,400 |
| write gate $W_\beta$, $b_\beta$ | $1024 \times 16 + 16$ | 16,400 | 262,400 |
| causal conv taps and bias | $3072 \times 4 + 3072$ | 15,360 | 245,760 |
| per-head output norm scale | $1024$ | 1,024 | 16,384 |
| token-shift mix $\mu$ | $1024$ | 1,024 | 16,384 |
| value-residual mix $\lambda$ | $16$, in layers 1 to 15 only | 16 | 240 |
| two RMSNorm scales | $2 \times 1024$ | 2,048 | 32,768 |
| SwiGLU $W_1, W_3, W_2$ | $3 \times 1024 \times 2752$ | 8,454,144 | 135,266,304 |
| token embedding $E$ (tied head) | $32768 \times 1024$ | | 33,554,432 |
| final RMSNorm | $1024$ | | 1,024 |
| **Total** | | | **253,544,176** |
```

#### 3. Section 5: The Selective Linear Attention Layer (Lines 394–520)
- $u_t = W_{qkv} x_t \in \mathbb{R}^{3d} = \mathbb{R}^{3072}$.
- Depthwise causal conv runs on $3072$ channels with $K=4$.
- $\hat{u}_t$ splits into 16 heads of $(q_t, k_t, v_t)$, each in $\mathbb{R}^{64}$.
- Gated write strength: $\beta_t = 2\sigma(W_\beta \tilde{x}_t + b_\beta) \in (0, 2)$, $W_\beta \in \mathbb{R}^{16 \times 1024}$.
- Value highway: Layer 0 passes $v^{(0)}$ to all layers $\ell \ge 1$, mixed via $\sigma(\lambda^{(\ell)}) \in \mathbb{R}^{16}$.
- Output gate $W_g \in \mathbb{R}^{1024 \times 1024}$, $W_o \in \mathbb{R}^{1024 \times 1024}$.

#### 4. Section 8: Chunked Form Used for Training (Lines 677–739)
- Number of chunks in sequence $T=4096$: $n = 4096 / 64 = 64$ chunks.
- Chunked tensors operate over $H=16$ heads.
- Details the seam-exact 4-regime decomposition implemented in `models/slab_v4_boundary.py` (`ssd_chunk_scan_v4`).

#### 5. Section 9: Document Isolation — Tri-Channel Boundary Reset (Lines 741–777)
**Crucial Architectural Difference in `slab_v4`:**
In `slab_v3`, boundary resets relied on setting decay mask entries to $-\infty$ in log-space. When an EOT occurred mid-chunk at $t^*$, subsequent cumulative prefix sums $P_t = \sum_{\tau=0}^t \log \alpha_\tau$ became $-\infty$, leading to IEEE-754 $(-\infty) - (-\infty) = \text{NaN}$ in intra-chunk difference operations $P_i - P_j$.

**The `slab_v4` Tri-Channel Solution:**
1. **Single Entry Point (`compute_doc_ids`):** Computes boolean reset mask $m_t = \mathbb{I}[x_{t-1} == \text{EOT}]$ and integer document ID $d_t = \sum_{\tau=0}^t m_\tau$ once at the model entrance.
2. **Channel 1 (Token-Shift):** At $m_t == \text{True}$, forces $\mu_{\text{effective}} = 1$ so $\tilde{x}_t = x_t$ (zero bleed from $x_{t-1}$).
3. **Channel 2 (Causal Conv1d):** Depthwise causal conv taps that cross boundaries are masked using document IDs ($d_t == d_{t-k}$).
4. **Channel 3 (Chunked SSD Scan):** Log decays $\log \alpha_t \le 0$ remain strictly finite. Boundary isolation is enforced exclusively through integer document ID comparisons:
   - **Intra-Chunk ($64 \times 64$):** $L_{i,j} = \exp(P_i - P_j) \cdot \mathbb{I}[i \ge j] \cdot \mathbb{I}[d_i == d_j]$
   - **Chunk-to-State ($B_k$):** $w_{\text{out}}(j) = \exp(P_{C-1} - P_j) \cdot \mathbb{I}[d_{C-1} == d_j]$
   - **Inter-Chunk Decay ($A_k$):** $A_k = \exp(P_{C-1}) \cdot \mathbb{I}[d_{C-1} == d_0 - m_0]$
   - **State-to-Chunk Read ($C_k$):** $w_{\text{in}}(i) = \exp(P_i) \cdot \mathbb{I}[d_i == d_0 - m_0]$
This guarantees zero cross-document bleed and zero NaN generation across any arbitrary boundary placement.

#### 6. Section 10: Memory Timescales (Lines 778–826)
Update table from 12 heads to 16 heads ($1 - \alpha \in [10^{-1}, 10^{-4.5}]$):
```markdown
| Head $h$ | Initial $\alpha_h$ | Initial $1 - \alpha_h$ | Half-life (tokens) |
| :---: | :---: | :---: | :---: |
| 0 | 0.900000 | $1.00 \times 10^{-1}$ | 6.6 |
| 1 | 0.941566 | $5.84 \times 10^{-2}$ | 11.5 |
| 2 | 0.965855 | $3.41 \times 10^{-2}$ | 20.0 |
| 3 | 0.980047 | $2.00 \times 10^{-2}$ | 34.4 |
| 4 | 0.988341 | $1.17 \times 10^{-2}$ | 59.1 |
| 5 | 0.993187 | $6.81 \times 10^{-3}$ | 101.4 |
| 6 | 0.996019 | $3.98 \times 10^{-3}$ | 173.8 |
| 7 | 0.997674 | $2.33 \times 10^{-3}$ | 297.6 |
| 8 | 0.998641 | $1.36 \times 10^{-3}$ | 509.6 |
| 9 | 0.999206 | $7.94 \times 10^{-4}$ | 872.3 |
| 10 | 0.999536 | $4.64 \times 10^{-4}$ | 1,493.0 |
| 11 | 0.999729 | $2.71 \times 10^{-4}$ | 2,555.2 |
| 12 | 0.999842 | $1.58 \times 10^{-4}$ | 4,373.1 |
| 13 | 0.999907 | $9.26 \times 10^{-5}$ | 7,484.1 |
| 14 | 0.999946 | $5.41 \times 10^{-5}$ | 12,808.0 |
| 15 | 0.999968 | $3.16 \times 10^{-5}$ | 21,918.9 |
```

#### 7. Section 11: The SwiGLU Channel Mixer (Lines 827–871)
- Hidden width: $\lfloor 8d/3 \rfloor = \lfloor 8 \times 1024 / 3 \rfloor = 2730 \to 2752$ (aligned to 64).
- $W_1, W_3 \in \mathbb{R}^{2752 \times 1024}$, $W_2 \in \mathbb{R}^{1024 \times 2752}$.
- Total SwiGLU parameters: $16 \times (3 \times 1024 \times 2752) = 135,266,304$ (53.35% of model).

#### 8. Section 13: Inference in Constant Memory (Lines 873–920)
**Constant Recurrent Memory Footprint vs. Transformer KV-Cache:**
- Parentheses 1.0-240M Recurrent State Size: Strictly **4.85 MB in fp32** ($2.42\text{ MB}$ in bf16), constant for any sequence length $T \in [1, \infty)$.
- Transformer (16 layers, $d=1024$, 16 heads): Stores $2 \times 16 \times 1024 = 32,768$ floats ($128\text{ KiB}$) per token.
  - At $T=1,024$: KV cache = $128\text{ MB}$ (26× larger than Parentheses state)
  - At $T=4,096$: KV cache = $512\text{ MB}$ (105× larger)
  - At $T=16,384$: KV cache = $2,048\text{ MB} = 2.0\text{ GB}$ (422× larger)
  - At $T=32,768$: KV cache = $4.0\text{ GB}$ (844× larger)
Parentheses' recurrent state memory footprint is overtaken by the Transformer KV cache after only **37 tokens**.

#### 9. Section 19: Appendix (Lines 1080–1150)
- **19.1 Tensor Shapes:**
  - Residual stream: $(B, T, 1024)$
  - `qkv` output: $(B, T, 3072)$
  - $q, k, v$: $(B, 16, T, 64)$
  - $\log \alpha, \beta$: $(B, T, 16)$
  - Chunked $q, k, v$: $(B, 16, n, 64, 64)$ where $n = T/64$
- **19.2 Recurrent State Shapes:**
  - `S`: $(B, 16, 64, 64)$
  - Conv history: $(B, 3072, 3)$
  - Shift state: $(B, 1024)$
- **19.4 Source Map:**
  - Add `models/model_slab_v4.py`: `SlabV4Model`, `SlabV4Config`, `SlabV4Attention`, `SlabV4Block`.
  - Add `models/slab_v4_boundary.py`: `compute_doc_ids`, `token_shift_v4`, `causal_conv1d_v4`, `ssd_chunk_scan_v4`, `ssd_recurrent_step_v4`.
  - Add `scripts/export_npu_240m_v4.py`: Export static-shape ONNX step graph.
  - Add `release/parentheses-1.0-240M-tanzanite/`: Shipped model weights, standalone `load_example.py`, and NPU artifacts.

---

## 5. Verification Check Results

The auditor ran direct self-tests and verified:
1. `models.slab_v4_boundary`: PASSED (exit code 0, all parity tests and doc reset assertions pass).
2. `models.model_slab_v4`: PASSED (exit code 0, step parity and architecture wiring pass).
3. `release/parentheses-1.0-240M-tanzanite/npu/parity_report.json`:
   - `cpu`: Max error $3.75 \times 10^{-5}$ (PASSED)
   - `dml` (DirectML): Max error $5.48 \times 10^{-5}$ (PASSED)
   - Graph size: 1,148.7 MB (< 1.15 GB)
4. Parameter verification of `models.model_slab_v4.SlabV4Model`:
   - Unique parameter count: exactly **253,544,176**.
   - Tied weights: `head.weight === tok_emb.weight` (33,554,432 elements).

---
*Report prepared by Version & Documentation Auditor Subagent for Coordinator.*
