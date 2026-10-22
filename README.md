```
██████╗   █████╗  ██████╗  ███████╗ ███╗   ██╗ ████████╗ ██╗  ██╗ ███████╗ ███████╗ ███████╗ ███████╗
██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║  ██║ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
██████╔╝ ███████║ █████╔╝  █████╗   ██╔██╗ ██║    ██║    ███████║ █████╗   ███████╗ █████╗   ███████╗
██╔═══╝  ██╔══██║ ██╔══██╗ ██╔══╝   ██║╚██╗██║    ██║    ██╔══██║ ██╔══╝   ╚════██║ ██╔══╝   ╚════██║
██║      ██║  ██║ ██║  ██║ ███████╗ ██║ ╚████║    ██║    ██║  ██║ ███████╗ ███████║ ███████╗ ███████║
╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚══════╝ ╚══════╝
```

# Parentheses

### *Alias: Project Tanzinite*
**Engineered by uncoalesced**

<div align="center">
  <img src="assets/parentheses_lockup_transparent.svg" alt="Parentheses / Project Tanzinite" width="380">
</div>

<p align="center">
  <b>A from-scratch recurrent language model engineered with selective linear attention (State Space Duality / RWKV / Mamba-2 family), optimized for Dravidian languages and English on consumer hardware.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Work_in_Progress-yellow.svg" alt="Status">
  <img src="https://img.shields.io/badge/Release-Coming_Soon-blue.svg" alt="Coming Soon">
  <img src="https://img.shields.io/badge/Architecture-Recurrent_SSM_%2F_SSD-9cf.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/Hardware-Consumer_Laptop_GPU-green.svg" alt="Hardware">
</p>

---

> [!NOTE]
> ### 🚧 Project Status: Active Work in Progress — Coming Soon
> **Project Tanzinite (Parentheses)** is currently under active development. The current experimental models (0.9-41M and 0.9-120M) serve as proof-of-concept foundations as we scale data pipelines, multimodal grounding, and conversational reasoning toward **Parentheses 1.0 (240M)**. Public weights, checkpoints, and complete benchmark packages will be published soon.

---

> [!IMPORTANT]
> ### Architectural Paradigm: Recurrent State Space Duality (SSD)
> **Parentheses is NOT a Transformer / GPT model.** It does not use quadratic causal self-attention ($\mathcal{O}(T^2)$ Softmax attention matrices) and requires zero growing KV-caches.
> 
> Instead, Parentheses implements **from-scratch Selective Linear Attention** within the **Mamba-2 / RWKV / State Space Duality** family:
> - **During Training (Parallel Form):** Chunked associative linear attention computes representations across sequence lengths in linear $\mathcal{O}(T)$ time via closed-form inter-chunk decay.
> - **During Inference (Recurrent Form):** Steps forward with **constant $\mathcal{O}(1)$ memory consumption**. At each token step $t$, it updates a fixed-size associative memory matrix:
>   $$S_t = \alpha_t S_{t-1} + (k_t \otimes v_t), \quad y_t = q_t S_t$$
>   The state size never increases with sequence length, eliminating out-of-memory bottlenecks during prolonged generation.

---

## How It Works Currently

### 1. Consumer-Trained Recurrence
Parentheses is built and trained completely from scratch on a single consumer laptop GPU (NVIDIA GeForce RTX 5050 Laptop, 8 GB VRAM). It proves that high-efficiency recurrent architectures can be engineered, scaled, and aligned without industrial data center clusters.

### 2. Multi-Lingual Dravidian Specialization
Standard open LLMs suffer from severe tokenization bloat and poor script comprehension on non-Latin scripts. Parentheses utilizes custom Akshara-preserving byte-level BPE (`v3_32k`) optimized for:
- **Major Dravidian Languages:** Kannada (`kan`), Tamil (`tam`), Telugu (`tel`), Malayalam (`mal`).
- **Low-Resource & Regional Dialects:** Tulu (`tcy`), Koya (`kff`), Kuvi (`kxv`), Kolami (`nit`).
- **Global Pivot:** English (`eng`).

### 3. Pure Neural Reasoning & Calibrated Honesty
- **Explicit `<think>` Scratchpads:** Generates step-by-step thinking traces before delivering final answers.
- **Uncensored & Unrefusing:** Calibrated for honest answers, factual boundaries, and zero corporate refusal boilerplate ("As an AI language model...").
- **100% Neural Execution:** Cleaned of artificial regex cheats or synthetic prefill anchors—all reasoning stems purely from the model's recurrent weights.

### 4. Recurrent Multimodal & Document Grounding
Implemented in `features/multimodal.py`, the engine processes visual inputs and complex documents within the same $\mathcal{O}(1)$ recurrent paradigm:
- **Constant-Memory Image Patch Prefill:** 2D image patches are projected and stepped sequentially into the recurrent state matrix without ViT self-attention or sequence explosion.
- **Universal Multi-Format Ingestion:** Extracts clean UTF-8 text from PDF, DOCX, HTML, XML, CSV, JSON, and TXT directly for recurrent consumption.

---

## Model Progression & Scale

| Preset / Stage | Parameters | Context ($T$) | Recurrent State Size | Status | Focus |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`parentheses-0.9-41M`** | 40.9M | 4,096 | ~2.3 MB | Trained (1.06B tokens) | Architecture & tokenizer validation |
| **`parentheses-0.9-120M`** | 119.7M | 4,096 | ~4.2 MB | Baseline (32L/512D) | Baseline deep-narrow configuration |
| **`parentheses-0.9-130M-v2`** | 148.2M | 4,096 | ~1.57 MB | Validated Baseline | Wide (16L/768D/12H, rank 768, slab_v2) |
| **`parentheses-0.9-130M-v3`** | **148.4M** | **4,096** | **~1.60 MB** | **Validated & Ready** | **slab_v3: Token-Shift, Beta Write Gate, V0 Residual Highway** |
| **`parentheses-1.0-240M`** | ~240M | 8,192 | ~8.4 MB | **Coming Soon** | Full conversational scale, audio/speech grounding |

---

## Quickstart (Local Preview)

To test the current development checkpoints locally:

```bash
# Clone and install dependencies
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
pip install -r requirements.txt

# Launch interactive terminal chat with thinking trace
python scripts/chat.py --lang kan --format think
python scripts/chat.py --lang eng --format think

# Run with an image or document prefilled into recurrent state
python scripts/chat.py --lang eng --image document.png --doc report.pdf

# Launch local Web Studio
python release/parentheses-0.9-120M-candidate/app.py
```

---

## What's Happening Right Now (In Progress)

- **Conversational Intelligence Scaling:** Generating and tokenizing high-fidelity conversational dialogue and pragmatic turn-taking datasets (including Karya conversational data and synthetic distillation) towards a 1B+ token conversational backbone.
- **DPO & Honesty Alignment:** Tuning abstention calibrations so the model cleanly states what it does not know rather than fabricating facts.
- **Preparing Public Release:** Packaging standalone Hugging Face candidates with `AutoModelForCausalLM` remote-code support, GGUF/NPU export exploration, and open weights.

---

## License
- **Code & Architecture:** [MIT License](LICENSE)
- **Model Checkpoints & Weights:** [Apache License 2.0](LICENSE-MODEL)
