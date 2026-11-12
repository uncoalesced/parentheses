# Parentheses 1.0-240M (`slab_v4`): Master Session 3-8 Architectural Blueprint

## Overview
This document outlines the master execution blueprint for **Parentheses 1.0-240M (slab_v4)** from Session 3 through Session 8, covering dual-memmap micro-batch interleaving, progressive curriculum phases, dual validation gates, SFT abstention scaling, and Session 8 on-policy DPO v2 with SWA-3.

---

## 1. Session 3: Dual-Memmap Micro-Batch Interleaving & Curriculum

### 1.1 Objectives & Rationale
Session 3 resumes training from Step 4,000 (checkpoint `step_4000.pt`) up to Step 6,000. To prevent catastrophic forgetting of structured reasoning (`<think>` blocks) and avoid low-resource Dravidian embedding shock, Session 3 introduces strict dual-memmap micro-batch interleaving.

### 1.2 Data Sources & Ratio
- **`processed_v5`**: High-density reasoning corpus & conversational/SFT grounding data.
- **`processed_240m_apache`**: Massive multilingual Apache/Indic/Dravidian pretraining pool.
- **Interleaving Ratios (Two-Stage Curriculum)**:
  - **Phase 3A (Steps 4,001 – 4,250)**: 10:6 ratio (`processed_v5` : `processed_240m_apache`), maintaining stable transition.
  - **Phase 3B (Steps 4,251 – 6,000)**: Transitioning to 6:10 ratio to maximize pretraining throughput on the Apache/Dravidian corpus while preserving core reasoning capabilities.

### 1.3 Dual Validation Gate
Monitoring requires tracking two independent validation losses simultaneously:
- **`val_loss_apache`**: Evaluates general multilingual pretraining perplexity.
- **`val_loss_v5`**: Evaluates reasoning integrity and token preservation.

---

## 2. Post-Training Roadmap (Sessions 4–8)

### 2.1 Sessions 4–6: Pretraining Expansion & Annealing
- Continued scaling across high-throughput data shards on expansion storage (`the expansion drive`).
- Cosine learning-rate annealing and gradient norm stability clipping.

### 2.2 SFT v5 Phase
- Supervised Fine-Tuning incorporating **8,000 abstention-aware reasoning traces** to ensure robust epistemic calibration and refusal handling.

### 2.3 Session 8: On-Policy DPO v2 & SWA-3
- Script: `scripts/train_dpo_v2_apache.py`
- Modes: `--mode generate`, `--mode precompute-ref`, `--mode train` (with `0.1 * NLL` anchor), `--mode swa3`, and `--mode all`.
- Ensures alignment stability and preference optimization without degrading core recurrent state dynamics (`beta_write`, `v0_mix`).

---
*Certified by Flash 3.8 Observer Sub-agent.*
