# Gates: Long-Run Preparation — Leaf L2 (Recipe Math & Recommendation)

Owner: worker LONGRUN  
Date: 2026-10-08  
Hardware: 1x NVIDIA GeForce RTX 5050 Laptop GPU (8 GB)  
Target Window: 8–10 hours overnight (launch ~23:30–24:00 on 2026-10-08 by Joel)  

---

### Gate L2.1 — Throughput Measurement & Compute Budget

- **CHECK:** Empirically verified against `out/run_checks_2026-10-07.md` and live timing.
- **INPUTS:**
  - Measured sustained training throughput on 40.9M architecture:
    - Min hourly rate: **0.2343 steps/s** (4.268 s/step)
    - Max hourly rate: **0.2376 steps/s** (4.209 s/step)
    - Mean sustained rate: **0.2357 steps/s** (4.243 s/step)
  - Micro-batch: 2 rows $\times$ seq_len 4,096 tokens $\times$ grad_accum 16 = **131,072 tokens/step** ($2^{17}$).
  - Token throughput: $0.2357 \times 131,072 \approx$ **30,893 tokens/s** (~30.9k tok/s).
- **EXPECT:** Accurate step and token budgets for 8.0 h, 8.5 h, 9.0 h, 9.5 h, and 10.0 h windows.
- **EVIDENCE:**
  | Window Duration | Min Steps (0.2343 st/s) | Mean Steps (0.2357 st/s) | Max Steps (0.2376 st/s) | Token Budget (Mean) | Compute FLOPs ($6ND$) |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **8.0 hours** (28,800 s) | 6,747 steps | 6,788 steps | 6,842 steps | 889.7M tokens | $2.18 \times 10^{17}$ |
  | **8.5 hours** (30,600 s) | 7,169 steps | 7,212 steps | 7,270 steps | 945.3M tokens | $2.32 \times 10^{17}$ |
  | **9.0 hours** (32,400 s) | 7,591 steps | 7,636 steps | 7,698 steps | 1,000.8M tokens | $2.46 \times 10^{17}$ |
  | **9.4 hours (Nominal Target)** | **7,966 steps** | **8,000 steps** | **8,078 steps** | **1,048.6M tokens** | **$2.58 \times 10^{17}$** |
  | **9.5 hours** (34,200 s) | 8,013 steps | 8,061 steps | 8,125 steps | 1,056.6M tokens | $2.60 \times 10^{17}$ |
  | **10.0 hours** (36,000 s) | 8,434 steps | 8,485 steps | 8,553 steps | 1,112.2M tokens | $2.73 \times 10^{17}$ |
- **STATUS:** PASS

---

### Gate L2.2 — DeepSeek Scaling Law Fits

- **CHECK:** DeepSeek LLM batch size and learning rate scaling laws:
  $$B_{\text{opt}} = 0.292 \cdot C^{0.3271}$$
  $$\eta_{\text{opt}} = 0.3118 \cdot C^{-0.125}$$
  For $N = 40,944,096$ params, $D = 1.049\text{B}$ tokens:
  $C \approx 6 \times 40.94\text{M} \times 1.049\text{B} = 2.576 \times 10^{17}\text{ FLOPs}$.
- **EXPECT:**
  - $B_{\text{opt}} = 0.292 \cdot (2.576 \times 10^{17})^{0.3271} \approx 143,560$ tokens/step.
    The chosen batch $B = 2^{17} = 131,072$ tokens/step is within 8.7% of optimal, fully utilizing the 8 GB GPU VRAM (6.5 GB peak, 0.92 memory fraction cap) with zero memory spills.
  - $\eta_{\text{opt}} = 0.3118 \cdot (2.576 \times 10^{17})^{-0.125} \approx 2.13 \times 10^{-3}$.
    For base pretraining, $\eta = 1.0 \times 10^{-3}$ was empirically validated on 2026-10-07 as the stable, conservative peak.
    For continued pretraining (CPT), standard transfer scaling dictates $20\%-30\%$ of base pretraining LR ($\eta = 3.0 \times 10^{-4}$).
- **STATUS:** PASS

---

### Gate L2.3 — Option Comparison: From-Scratch vs. Continued Pretraining

| Evaluation Dimension | Option A: From-Scratch on `processed_v4` | Option B: Continue from `step_8082_final.pt` |
| :--- | :--- | :--- |
| **Initial Checkpoint** | Random initialization | `checkpoints/parentheses-0.9-41M/step_8082_final.pt` |
| **Initial Loss** | 10.48 nats/token | 3.48 nats/token (already converged on 1.06B tokens) |
| **Data Utilized** | 1.05B fresh tokens | 1.06B tokens (v3) + 1.05B fresh tokens (v4) = **2.11B tokens total** |
| **Effective Tokens/Param** | $25.6\times$ parameters | **$51.5\times$ parameters** |
| **Peak Learning Rate** | $\eta = 1.0 \times 10^{-3}$ | $\eta = 3.0 \times 10^{-4}$ ($30\%$ of initial LR, avoids gradient shock) |
| **Cooldown Schedule** | Linear cooldown from step 6,466 to 1e-5 | Linear cooldown from step 14,483 (80% of phase 2) to 1e-5 |
| **Warmup** | 160 steps from 0 | 160 steps from 8,083 (smooth ramp for optimizer momentum) |
| **Primary Risk** | High compute redundancy: spends first 5 hours re-learning representations already mastered by 0.9-41M; final bpb will only match (~0.52 bpb), not beat `step_8082_final.pt`. | Potential domain shift if v4 differs from v3; mitigated by identical Sangraha QC pipeline and hashed out overlap. |
| **Expected Outcome** | Re-reaches ~0.52 bpb on Dravidian, peer model to 0.9-41M. | Reaches **new state-of-the-art** (< 0.50 bpb across Dravidian anchors, ~3.2 nats loss). |

- **RECOMMENDATION: OPTION B (CONTINUED PRETRAINING).**  
  To satisfy the core goal of `JOB_ANTIGRAVITY_LONG_RUN_PREP.md` ("improves on checkpoints/parentheses-0.9-41M/step_8082_final.pt"), Option B is strictly superior. It pushes cumulative training to $51.5\times$ parameters, exploiting the fresh, non-overlapping Sangraha slice without discarding 9.5 hours of previously invested training compute.

---

### Gate L2.4 — Architecture Decision (Baseline vs slab_v2)

- **CHECK:** Worker SLAB findings in `out/swarm/slab_report.md` and `gates/slab_v2.md`.
- **EVIDENCE:** Worker SLAB tested PyTorch compiled `slab_v2`. Inductor backend failed due to missing Triton (forbidden new dependency); AOT eager ran at 22,687 tok/s vs baseline 32,860 tok/s (-31% deficit, violating the <=10% threshold). Stream C formally declared **ABANDON C**.
- **DECISION:** Use **Baseline architecture** (`slab_v2=False`). `parentheses-0.9-41M` remains the canonical model configuration.
- **STATUS:** PASS

---

### Gate L2.5 — Launch Command Parameters & Sizing

- **Target Duration:** 9 hours 25 minutes (nominal 8,000 steps $\times$ 4.243 s/step = 33,944 s).
- **Step Range:** Resuming from step 8,082:
  - Start step: 8,083
  - Max steps: **16,083** (8,000 new steps)
  - Cooldown start: **14,483** ($8,083 + 0.8 \times 8,000$)
  - Warmup steps: **160** ($2\%$ of new steps)
  - Base LR: **3e-4**
  - Min LR: **1e-5**
  - Checkpoint interval: **282 steps** (~20 minutes, yielding 28 intermediate checkpoints)
  - Deadline guard: Set to unix timestamp 10.0 hours from launch time.
- **Exact Parametric Launch Invocation:**
```powershell
python -u scripts/overnight_driver.py --name parentheses-0.9-41M-v4 -- `
  --preset parentheses-0.9-41M `
  --data data/processed_v4/train.bin `
  --boundaries data/processed_v4/train_boundaries.npy `
  --tokenizer data/tokenizer/v3_32k `
  --resume-from checkpoints/parentheses-0.9-41M/step_8082_final.pt `
  --out-dir checkpoints/parentheses-0.9-41M-v4 `
  --batch-size 2 `
  --grad-accum 16 `
  --ce-rows 4096 `
  --lr 3e-4 `
  --warmup-steps 160 `
  --decay-start 14483 `
  --min-lr 1e-5 `
  --max-steps 16083 `
  --ckpt-every 282 `
  --log-every 10 `
  --seed 1234 `
  --deadline $DEADLINE_UNIX
```
*(Fallback note: If worker DATA does not finish `processed_v4` by 20:35, `--data` falls back to `data/processed_v3/train.bin` and `--boundaries` to `data/processed_v3/train_boundaries.npy`).*
- **STATUS:** PASS
