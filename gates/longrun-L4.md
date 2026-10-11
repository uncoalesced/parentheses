# Gates: Long-Run Preparation — Leaf L4 (End-to-End Smoke Test & Resume Verification)

Owner: worker LONGRUN  
Date: 2026-10-08  
Hardware: 1x NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)  
Dataset: `data/processed_v3/train.bin` (760,006,005 tokens)  
Tokenizer: `data/tokenizer/v3_32k` (sha256 `7a9b4187d5bc000134b5f646e45e8e5a0435adb0e7d20afb99ac4bde0a9808a6`)  

---

### Gate L4.1 — 50-Step Training Smoke Test Execution

- **CHECK:** Execute a 50-step training dry run on GPU using the exact overnight launch configuration:
  - Preset: `parentheses-0.9-41M` (40.9M params, `attn_type=selective_linear`, `slab_v2=False`)
  - Initial Checkpoint: `checkpoints/parentheses-0.9-41M/step_8082_final.pt`
  - Step Range: 8,082 to 8,132 (`--max-steps 8133`)
  - Batch parameters: `--batch-size 2 --grad-accum 16 --ce-rows 4096` (131,072 tokens/step)
  - Optimizer parameters: `--lr 3e-4 --warmup-steps 160 --decay-start 14483 --min-lr 1e-5 --seed 1234`
  - Checkpoint parameters: `--ckpt-every 25 --log-every 10`
- **EXPECT:**
  - Zero exit code (no crashes, no OOM, no tensor mismatch).
  - Throughput within 3% of baseline reference (0.2343–0.2376 steps/s).
  - Smooth loss trajectory with no numerical instability or NaN/Inf values.
  - LR smoothly warms up without gradient shock.
- **EVIDENCE:**
  ```
  [info] parentheses-0.9-41M: 40.9M params on cuda
  [info] document-boundary reset on: 900,161 documents over 760,006,005 tokens (844 tokens/doc mean)
  [info] resumed from checkpoints/parentheses-0.9-41M/step_8082_final.pt at step 8082, continuing to step 8132
  [info] tokenizer data/tokenizer/v3_32k sha256 7a9b4187d5bc, slab_v2=False
  step   8090 | loss 3.3906 | lr 1.50e-05 | 34.7s elapsed
  step   8100 | loss 3.3094 | lr 3.38e-05 | 77.5s elapsed
  [info] saved checkpoint: checkpoints/scratch_smoke_l4\step_8100.pt
  step   8110 | loss 3.2934 | lr 5.25e-05 | 120.3s elapsed
  step   8120 | loss 3.2699 | lr 7.12e-05 | 162.4s elapsed
  [info] saved checkpoint: checkpoints/scratch_smoke_l4\step_8125.pt
  step   8130 | loss 3.2739 | lr 9.00e-05 | 205.1s elapsed
  [info] saved final checkpoint: checkpoints/scratch_smoke_l4\step_8132_final.pt
  ```
  - **Exit Code:** `0` (Success).
  - **Sustained Throughput:** Between step 8090 and 8130 (40 steps), elapsed time is $205.1\text{ s} - 34.7\text{ s} = 170.4\text{ s} \implies \mathbf{0.2347\text{ steps/s}}$ ($\mathbf{30,762\text{ tokens/s}}$).
    - Comparison vs 0.9-41M baseline mean ($0.2357\text{ steps/s}$): **0.42% delta**, well within the $\le 3\%$ requirement.
  - **Loss Stability:** Decreased smoothly from 3.3906 to 3.2739 as warmup progressed without gradient explosion or spikes.
- **STATUS:** PASS

---

### Gate L4.2 — Checkpoint Serialization & Architecture Integrity

- **CHECK:** Verify that intermediate and final checkpoints exist, contain valid state dicts, and record model config & tokenizer hashes.
- **EXPECT:**
  - Checkpoints saved: `step_8100.pt`, `step_8125.pt`, `step_8132_final.pt`.
  - File size: ~491.5 MB each.
  - `tokenizer_sha256` matching `7a9b4187d5bc...`.
  - Architecture config confirms `slab_v2=False` and `attn_type='selective_linear'`.
- **EVIDENCE:**
  - File sizes on disk:
    - `step_8100.pt`: 491,509,323 bytes.
    - `step_8125.pt`: 491,509,323 bytes.
    - `step_8132_final.pt`: 491,509,323 bytes.
  - Checkpoint metadata inspection via `torch.load`:
    - `step`: `8132`
    - `tokenizer`: `{'path': 'data/tokenizer/v3_32k', 'sha256': '7a9b4187d5bc000134b5f646e45e8e5a0435adb0e7d20afb99ac4bde0a9808a6'}`
    - `cfg`: `ModelConfig(vocab_size=32768, block_size=4096, n_layer=16, n_head=6, n_embd=384, dropout=0.0, bias=False, tie_embeddings=True, attn_type='selective_linear', slab_v2=False)`
- **STATUS:** PASS

---

### Gate L4.3 — Checkpoint Resume Verification

- **CHECK:** Perform an independent resume run starting from intermediate checkpoint `step_8100.pt` for 5 steps (steps 8101 to 8105).
- **EXPECT:**
  - Resume succeeds cleanly without tensor shape mismatch, device mismatch, or optimizer state corruption.
  - Final checkpoint `step_8105_final.pt` written.
  - Exit code `0`.
- **EVIDENCE:**
  ```
  [info] parentheses-0.9-41M: 40.9M params on cuda
  [info] document-boundary reset on: 900,161 documents over 760,006,005 tokens (844 tokens/doc mean)
  [info] resumed from checkpoints/scratch_smoke_l4/step_8100.pt at step 8100, continuing to step 8105
  [info] tokenizer data/tokenizer/v3_32k sha256 7a9b4187d5bc, slab_v2=False
  step   8101 | loss 3.4294 | lr 1.87e-06 | 4.7s elapsed
  step   8102 | loss 3.2703 | lr 3.75e-06 | 8.9s elapsed
  step   8103 | loss 3.2817 | lr 5.62e-06 | 13.1s elapsed
  step   8104 | loss 3.3214 | lr 7.50e-06 | 17.3s elapsed
  step   8105 | loss 3.3706 | lr 9.37e-06 | 21.5s elapsed
  [info] saved final checkpoint: checkpoints/scratch_smoke_l4/resume_test\step_8105_final.pt
  ```
  - **Exit Code:** `0`.
- **STATUS:** PASS

---

### Gate L4.4 — Workspace Hygiene & Lock Release

- **CHECK:** Scratch test directory `checkpoints/scratch_smoke_l4` deleted, GPU lock `out/locks/GPU.lock` released.
- **EXPECT:**
  - `checkpoints/scratch_smoke_l4` does not exist.
  - `out/locks/GPU.lock` does not exist.
- **EVIDENCE:**
  - `Test-Path E:\Parentheses\parentheses-0.9\checkpoints\scratch_smoke_l4` -> `False`.
  - `Test-Path E:\Parentheses\parentheses-0.9\out\locks\GPU.lock` -> `False`.
- **STATUS:** PASS
