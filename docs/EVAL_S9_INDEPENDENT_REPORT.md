# Parentheses 1.0-240M: Independent Post-Session 9 Evaluation Report

**Date:** 2026-11-08  
**Architecture:** Parentheses 1.0-240M (`SlabV4Model`, 253.54M parameters, recurrent SSD / selective linear attention)  
**Evaluated Checkpoints:**
- **S6 Baseline:** `checkpoints/parentheses-1.0-240M-s6/step_12000.pt` (Pre-training conclusion)
- **S8 DPO-SWA3:** `checkpoints/parentheses-1.0-240M-s8-golden/parentheses-1.0-240M-dpo-swa3-golden.pt` (On-policy DPO v2 + SWA-3 soup)
- **Golden Blend:** `checkpoints/parentheses-1.0-240M-s8-golden/parentheses-1.0-240M-s6-s8-blend-best.pt` ($0.60 \times \text{S6} + 0.40 \times \text{S8}$)

---

## 1. Executive Summary

This report provides an independent empirical evaluation of the claim that blending Session 6 base pre-training weights with Session 8 DPO weights produces an all-time high 450/900 capability model.

### Key Conclusions:
1. **The 450/900 Heuristic Score is Replicated but Invalid:** The heuristic scoring script reproduces 450/900 for the blend (vs 413 for S6 and 381 for S8). However, strict blind hand-grading confirms this is an artifact of lenient rubric rules awarding points to prompt echoes and repetitive loops. Strict correctness out of 60 probes is: S6 = 3, S8 = 5, **Blend = 2**; paired bootstrap difference is within noise (+16 points, 95% CI [$-52$, $+83$], $p = 0.14$).
2. **Hermes 1,000-Prompt Benchmark Reverses the Finding:** Under uncapped greedy decoding with a 10s watchdog, **S8 is clearly the most capable and coherent checkpoint**:
   - Clean Completion (`stop_token`): S8 = **68.4%**, S6 = **60.5%**, Blend = **53.0%**.
   - Watchdog Timeout Rate: S8 = **31.6%**, S6 = **39.5%**, Blend = **47.0%**.
   - Degenerate Loops ($\ge 4$ repetitions): S8 = **36.0%**, S6 = **43.3%**, Blend = **48.3%**.
   - Usable Responses (No loops, no tool call hallucinations): S8 = **61.3%**, Blend = **46.9%**, S6 = **30.3%**.
3. **Architectural Insight:** Mixing pre-SFT base representations with post-DPO representations creates destructive interference during autoregressive decoding, causing severe repetitive looping. S8 DPO-SWA3 is the true golden release of this lineage.

---

## 2. Benchmark Metrics (Hermes 1,000 Suite)

| Metric | S6 Baseline | S8 DPO-SWA3 | Golden Blend |
|---|---|---|---|
| **Prompts Evaluated** | 1,000 | 1,000 | 1,000 |
| **Natural Stop (`stop_token`)** | 605 (60.5%) | **684 (68.4%)** | 530 (53.0%) |
| **Watchdog Timeout (10s)** | 395 (39.5%) | **316 (31.6%)** | 470 (47.0%) |
| **Degenerate Loops ($\ge 4$ reps)** | 433 (43.3%) | **360 (36.0%)** | 483 (48.3%) |
| **Tool Call Hallucinations** | 270 (27.0%) | 18 (1.8%) | **9 (0.9%)** |
| **`<think>` Blocks Opened / Closed** | 281 / 45 (16.0%) | **323 / 157 (48.6%)** | 88 / 21 (23.9%) |
| **Usable Responses** | 303 (30.3%) | **613 (61.3%)** | 469 (46.9%) |
| **Mean Tokens / Response** | 273.5 | **224.6** | 253.5 |
| **Average Decode Speed** | **51.60 tok/s** | 51.02 tok/s | 45.28 tok/s |

---

## 3. Domain Performance Breakdown (Completion %)

| Domain | S6 Baseline | S8 DPO-SWA3 | Golden Blend |
|---|---|---|---|
| **D01: Quantitative Reasoning** | 42% | **71%** | 30% |
| **D02: Code Tracing & Repair** | 70% | **99%** | 97% |
| **D03: Scientific Inference** | 58% | **73%** | 42% |
| **D04: Statistics & Causal Reasoning** | **77%** | 70% | 38% |
| **D05: Engineering Systems Diagnosis**| 38% | **50%** | 32% |
| **D06: Evidence Synthesis** | 57% | 58% | **59%** |
| **D07: Heritage & Space Science** | **75%** | 29% | 44% |
| **D08: Epistemic Calibration** | 39% | **84%** | 42% |
| **D09: Dravidian Reading Comprehension**| 83% | 80% | **87%** |
| **D10: Dravidian Controlled Generation**| 66% | **70%** | 59% |

---

## 4. Dravidian Script Adherence

All 200 Dravidian prompts maintained native script integrity:
- Kannada (`kn`): 100.0% valid across all models.
- Tamil (`ta`): 100.0% valid across all models.
- Telugu (`te`): 100.0% valid across all models.
- Malayalam (`ml`): 100.0% in S6 and Blend; 94.0% in S8.

---

## 5. Certification Gates (Blend)

- **C1 BPB Compression:** **PASS** (negative deltas across all 11 languages).
- **C2 60-Probe Adversarial:** **FAIL** (Greedy 2.00/60, Sampled 5/60).
- **C4 Identity (15 probes):** **FAIL** (3/15 passed, 20.0%).
- **C5 Benign Refusal (50 probes):** **PASS** (0/50 refusals, 0.0% false refusal rate).
- **C6 Honesty (10 probes):** **FAIL** (0/10 passed, 0.0%).
- **C8 16k Long Context:** **PASS** (16,384 tokens processed, zero NaN/Inf, 53.95 tok/s).
