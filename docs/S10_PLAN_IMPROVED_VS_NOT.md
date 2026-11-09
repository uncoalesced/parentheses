# Parentheses 1.0-240M: what improved, what didn't, and the road to 85–90%

**Date:** 2026-11-08 · **Author:** Claude Code (Opus) · **Status:** analysis and plan only; nothing
executed.

---

## 0. Summary

- **Real improvements across S6 → S9:**
  - language modelling (small BPB gains);
  - native-script fidelity in all four major Dravidian languages;
  - stable 16k context;
  - zero false refusals;
  - fewer tool-call outputs on code prompts after SFT.
- **No improvement:**
  - answering correctly (about 2% on held-out Hermes-1000 for every checkpoint);
  - stopping cleanly (32–47% of answers loop until cut off);
  - honesty ("I don't know": 0 in every session);
  - identity (fails);
  - answer format (`Final:` almost never used).
- **Why:**
  1. The S7.5 SFT data taught tool calls and long rambling think traces, and never taught "short
     answer, then stop" or "I don't know".
  2. S8 DPO never actually changed the model.
  3. Checkpoints were chosen on the test probes with a lenient grader, which hid all of this.
- **Target, as the maintainer chose ("both, staged"):**
  - Stage 1: behaviour gates at 85–90% in Session 10 (maybe S11).
  - Stage 2: strict correctness climbs session by session toward 85–90% on a private held-out suite.
- **Honest expectation:** Stage 1 is reachable at 240M. Stage 2's 85–90% correctness on reasoning
  tasks very likely needs a bigger model or heavy distillation. The plan says when to escalate.

---

## 1. Evidence base (all numbers come from files, not handoffs)

| Source | Path |
|---|---|
| BPB S6 / S7.5 / S8 / blend | `out/eval_s6_12k/bpb_12k.json`, `out/eval_s7_5/bpb_240m.json`, `out/eval_s8/bpb_240m.json`, `out/eval_s9_indep/cert_blend/c1_bpb_240m.json` |
| 60-probe pass@1 / pass@5 | `out/eval_s6_12k/scores_12k.txt`, `out/eval_s7_5/scores.txt`, `out/eval_s8/scores.txt`, `out/eval_s9_indep/cert_blend/c2_scores.json` |
| 900 heuristic | `out/eval_loop_900/rubric_900_scores.json`, `out/eval_s9_indep/rescore_60p_heuristic.json` |
| Strict blind 60-probe | `out/eval_s9_indep/blind60_results.json` (my hand grades, 180 rows) |
| Hermes-1000 strict | `out/eval_s9_indep/final_report/hermes1000_report.json`, `REPORT.md` |
| Gates | `out/eval_s7_5/gate_verdict_claude.json`, `out/eval_s8/s8_verdict.json`, `out/eval_s9/s9_certification_report.json`, `out/eval_s9_indep/cert_blend/s9_certification_report.json` |
| Training config and data | `scripts/launch_s7_5_240m.ps1`, `scripts/run_session8_auto.ps1`, `scripts/train_dpo_v2_apache.py`, `data/processed_sft_240m_v4_s7_5/manifest.json`, `data/dpo_240m_apache/manifest.json`, `out/s8_train.log` |

---

## 2. Session by session: what each step did

### 2.1 Bits per byte (lower is better)

| Lang | S6 (step 12000) | S7.5 (13000) | S8 golden | Blend |
|---|---|---|---|---|
| kan | 0.479 | 0.483 | 0.483 | **0.477** |
| tam | 0.473 | 0.477 | 0.477 | **0.471** |
| tel | 0.570 | 0.571 | 0.571 | **0.565** |
| mal | 0.392 | 0.396 | 0.396 | **0.390** |
| tcy | 0.654 | 0.667 | 0.667 | **0.653** |
| kff | 0.517 | 0.514 | 0.514 | **0.507** |
| kxv | 0.442 | 0.435 | 0.435 | **0.432** |
| nit | 0.695 | 0.717 | 0.717 | **0.692** |
| eng | 1.018 | 1.027 | 1.027 | **1.002** |
| hin | 0.708 | **0.683** | **0.683** | 0.688 |
| san | 1.221 | **1.041** | **1.041** | 1.122 |

**Reading this table:**
- SFT (S7.5) cost a little BPB on most languages but helped Hindi and Sanskrit.
- S8 is identical to S7.5 to three decimals, so DPO changed nothing measurable.
- The blend is best on 9 of 11 languages, because averaging with S6 restores pretraining
  statistics. This is language-modelling quality, not answering ability.

### 2.2 Answering ability

| Metric | S6 | S7.5 | S8 | Blend |
|---|---|---|---|---|
| 60-probe greedy pass@1 (lenient scorer) | 5/60 | 6/60 | 6/60 | 2/60 |
| 60-probe pass@5, sampled | 8/60 | 9/60 | 10/60 | 5/60 |
| 900 heuristic | 434 (413 in blend precision) | 393 | 381 | 450 |
| **Strict blind 60-probe, correct** | 3 | not graded | 5 | **2** |
| Strict 60-probe, fabricates | 13 | — | 17 | **19** |
| **Hermes-1000 strict score (/1000)** | 18.4 | not run | 17.0 | 20.7 |
| Hermes fully correct answers | 0 | — | 1 | 0 |
| Hermes objective items correct (/150) | 0 | — | 1 | 0 |

**Reading:** every checkpoint is at the floor. Differences between them are noise: the paired 95%
CIs cross 0, and the sign tests give p ≈ 1. The 900-point heuristic rewarded prompt echoes and
loops, and so pointed the wrong way.

### 2.3 Behaviour

| Behaviour | S6 | S7.5 | S8 | Blend |
|---|---|---|---|---|
| Hermes clean stop, no loop (/1000) | 519 | — | **619** | 476 |
| Hermes watchdog stops (/1000) | 395 | — | **316** | 470 |
| 4-gram loops on 60 probes | 12 | 15 | 16 | — |
| Tool-call answer on code probes | 7/10 | **0/10** | **0/10** | — |
| Bare "I don't know" (60 probes) | 0 | 0 | 0 | — |
| Honesty gate C6 (10) | — | — | 0/10 | 0/10 |
| Identity gate C4 (15, bar 14) | — | spot-check only | 6/15 | 3/15 |
| Closed math `<think>` | 2/10 | 1/10 | 1/10 | — |
| `Final:` / `Output:` marker on objective items (/150) | 0 | — | 9 | 0 |
| Correct native script, Dravidian | 10/10 | 10/10 | 10/10 | ok |
| False refusals, benign (C5) | — | — | 0/50 | 0/50 |
| 16k long context (C8) | — | — | pass | pass |

### 2.4 Verdict table

| Area | Verdict |
|---|---|
| Language modelling (BPB) | **Improved slightly** (blend best). Not a release criterion. |
| Native-script Dravidian output | **Solid since S6.** Keep. |
| Long context 16k | **Solid.** Keep. |
| No false refusals | **Solid.** Keep. |
| Tool-call contamination on code | **Improved** in S7.5/S8, but still appears on Hermes prompts (S6 heavy, S8 light). |
| Clean termination / no loops | **Not fixed.** S8 is best at 62% clean, and still 32% loop. |
| Honesty / "I don't know" | **Never worked.** 0 in every session. This is the maintainer's core goal. |
| Identity | **Fails** and got worse in the blend. |
| Answer format | **Never learned.** |
| Correctness (math, code, stats, reading, Dravidian QA) | **No progress.** About 2% for all. |

---

## 3. Root causes (with evidence)

1. **The S7.5 SFT mix taught the wrong habits** (`data/processed_sft_240m_v4_s7_5/manifest.json`,
   `scripts/build_s7_5_sft_240m.py`).
   - About 5.9% of doc tokens come from tool-call datasets (`tool_hermes_reasoning` 1.86M,
     `tool_interstellarninja` 1.40M, `tool_llamafactory_1500` 0.60M). That is where the
     `<tool_call>[...]` answers to math and code prompts come from.
   - 30% is STEM think traces. Long, unconstrained reasoning teaches the model to keep going, so it
     loops.
   - There are no examples of a short answer followed by stop, and no `Final:` line format.
   - `data/sft_v4_abstention_4000.jsonl` (4,000 honesty rows) was **not** included, so the model
     never saw "I don't know".
2. **S8 DPO did not train** (`out/s8_train.log`, `out/session8_dpo_metrics.jsonl`,
   `scripts/run_session8_auto.ps1`).
   - DPO loss stayed at about 0.69, which is the value for an untrained pair; the reward margin
     stayed at about 0.01.
   - LR was 5e-7. The launcher's own comment notes this rounds to zero in bf16.
   - It ran only 150 steps on 805 pairs.
   - The starting checkpoint is unconfirmed: `candidate_base` in `gate_verdict_claude.json` is null,
     and the trainer default is `parentheses-1.0-240M-s7-sft/latest.pt` (S7, not S7.5).
   - BPB identical to S7.5 confirms there was no update.
3. **Selection and reporting failures.**
   - The blend weight was picked on the 60 test probes.
   - The 900-point grader is lenient: it accepts "95" for 48 + 37, gives up to 12/15 to a looping
     keyword answer, and gives 3/3 for `def` + `return`.
   - The handoffs over-claimed: "100% marker accuracy" (nothing was graded), "math restored" (the
     scored run says x = 19), and S7.5 "G5 PARTIAL" (the gate file says FAIL).
4. **Evaluation instruments are flawed.**
   - Hermes v1 has 14 broken items and many references that are rubric text rather than answers.
   - `--prepare-review` drops watchdog answers, which hides failures.
   - The runner's watchdog counted prefill (fixed in 4430a36).
5. **Capacity.** At 240M, near-zero reasoning correctness is partly expected. Behaviour problems
   (loops, format, honesty, tool calls) are not capacity limits; they are data and objective
   problems, and they are fixable.

---

## 4. Target definition (the maintainer: "both, staged")

### Stage 1: behaviour gates (S10, maybe S11). Pass bar is ≥ 85–90%.

Measured on held-out test prompts, greedy decoding, no repetition penalty.

| Gate | Metric | Pass |
|---|---|---|
| G1 Clean stop | stop token reached, no 6-gram × 4 loop | ≥ 90% |
| G2 Tool-call leak | `<tool_call>` in a non-tool task | ≤ 2% |
| G3 Format | `Final:` / `Output:` present on objective items | ≥ 85% |
| G4 Honesty | correct abstention on a new 50-probe false-premise/unknown set (C6-50) | ≥ 85% |
| G5 Identity | C4 identity probes | ≥ 13/15 (87%) |
| G6 Language | answer in the requested native script | ≥ 95% |
| G7 No false refusal | C5 benign set | ≤ 2% |
| G8 LM health | BPB drift vs S6, every language | ≤ +0.06 |
| G9 Long context | C8 16k | pass |

### Stage 2: correctness (strict blind judging, private test split)

| Session | Target (strict score, test v2 and Hermes v1) |
|---|---|
| S10 | ≥ 10% (from 2%) |
| S11–S12 | ≥ 25% |
| Long-term | 85–90% |

**Escalation rule:** if correctness stays below 25% after two Stage-2 sessions, write the maintainer a
scale-up or distillation proposal (a bigger model, or 240M distilled from a permissive teacher)
instead of more of the same.

---

## 5. Session 10 plan (proposed; not executed)

### 5.1 Base checkpoint
**S6 `step_12000.pt`.** It has the best clean BPB, and it does not carry S7.5's tool-call and loop
habits. A fresh SFT re-teaches the chat format.

Alternative considered: start from S8. It loops least, but carries the S7.5 data habits and its true
base is unconfirmed.

### 5.2 SFT v5 data (about 60–65M tokens, same budget as S7.5)

| Bucket | Share | Content / rule |
|---|---|---|
| Replay, 11 languages | 25% | Protects BPB (G8) |
| Short worked math / quant | 20% | ≤ 256-token reasoning, mandatory last line `Final: <answer>` (G3) |
| Code trace / execution | 15% | Answer with `Output: <value>`; permissive sources only |
| Dravidian QA and controlled generation | 15% | Native-script answers, arithmetic word problems, dialogues, notices; new template families disjoint from Hermes D09/D10 (G6) |
| Abstention / false premise / unknown | 15% | `sft_v4_abstention_4000.jsonl` plus new sets: short "I don't know / that premise is false, here is why"; never fabricate; never refuse benign (G4, G7) |
| Identity | 5% | `data/curated/parentheses_identity_sft_v1.jsonl`, upsampled and paraphrased (G5) |
| Grounded reading | 5% | Answer only from the supplied card or evidence (D06/D07 style) |

**Hard data rules:**
- 0 tool-call sources.
- Every assistant turn ends with `<|endoftext|>`.
- Think traces ≤ 256 tokens, always closed.
- Licence per `LICENSE_POLICY.md`.
- Qwen (Apache-2.0) teaches logic and code only, never Dravidian text.
- No NVIDIA trial-key outputs and no Llama data.
- Honesty data follows the maintainer's rules (never fabricate, say "I don't know", never refuse).

**Contamination:** `scripts/check_eval_contamination.py` plus a digit-masked template check against
Hermes v1, suite v2, the 60 probes and the C4/C6/C5 sets. Required: 0 flagged.

### 5.3 Training changes
- **`train.py --eos-weight`** (S10 uses 3.0): up-weights the loss on the end-of-text target so the
  model learns to stop. Comes with a self-test.
- **SFT run:** S7.5 settings (1,000 steps, LR 1e-4 then decay to 5e-6, grad-accum 16, seq 4096,
  differential AdamW, recurrent LR ratio 0.1). About 6.6 h on the RTX 5050. Milestones at
  500/750/1000. Detached under `overnight_driver.py`, `GPU.lock`, swarm post.

### 5.4 DPO v3 (fixed trainer)
- **Required explicit `--sft-ckpt`:** no default, and refuse to run without it.
- **Updates must land:** fp32 master weights, or bf16 with LR ≥ 5e-6, plus an automatic failure if
  the parameter delta after 10 steps is 0.
- **Pairs (about 1,500) from the S10 model's own samples:**
  - chosen = short, terminated, correct, or an honest "I don't know";
  - rejected = looping, fabricated, tool-call, wrong-language, or missing `Final:`.
  - Prompts disjoint from all evals.
- **Gate:** held-out pair margin > 0.5 and accuracy > 80%, otherwise discard the DPO checkpoint.
- SWA only inside one training phase. No more cross-phase blends.

### 5.5 Evaluation protocol (fixes the selection problem)
- **Private suite v2.** Rebuild Hermes-style with fixes: the 14 broken items, real reference
  answers, D10 per-item keys, and new seeds. Keep it on `I:` only, never pushed.
  - Split: **dev 300** for choosing checkpoints, **test 700** for reporting only.
  - Hermes v1 stays as a second frozen test.
- **Gate runner** `scripts/run_session10_gates.py`: reuses the S9 certification functions and the
  fixed Hermes runner, and writes one verdict JSON with G1–G9.
- **Correctness:** the existing strict blind pipeline (`build_judge_sheet.py`, `aggregate_final.py`,
  `JUDGE_BRIEF.md`). Negative canaries gate the batches; I hand-grade about 100 rows for agreement.
- **Independent review** of the verdict by a fresh agent. No self-certified "done".

### 5.6 Execution order and time

| # | Step | Where | Est. time |
|---|---|---|---|
| 2 | Build private suite v2 and C6-50 honesty set; contamination checks | CPU | 2–3 h |
| 3 | Generate SFT v5 data (teacher jobs), pack bins, write manifest | CPU/API | 4–8 h |
| 4 | `train.py --eos-weight` patch plus all self-tests | CPU | 1 h |
| 5 | SFT run | GPU | ~6.6 h |
| 6 | Dev-split selection across milestones | GPU | ~1.5 h |
| 7 | DPO v3 pairs and training, margin gate | GPU | ~1.5 h |
| 8 | G1–G9 gates on test, strict judging, report, handoff, gates file, journal | GPU + judges | ~4–5 h |

### 5.7 Loop rule (session after session)
- **All Stage-1 gates pass:** Stage 2 starts (a correctness-focused data mix, more Qwen
  logic/code distillation, Dravidian QA from native-reviewed data).
- **Any gate fails:** the next session targets only the failing gates, with data aimed at those
  failures.
- **Correctness below 25% after two Stage-2 sessions:** the maintainer gets a scale-up / distillation proposal.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Fresh SFT from S6 loses S7.5's Hindi/Sanskrit BPB gains | 25% replay; G8 gate |
| EOS weighting makes answers too short or empty | Format gate G3 plus correctness tracking; tune 2.0–3.0 on dev |
| Abstention over-generalises into refusals | C5 gate G7; benign-QA pairs in DPO |
| Teacher-data licence slips | Manifest records every source and licence; review before training |
| Training to the test | Private dev/test split, contamination checks, Hermes v1 never trained on |
| Agents over-claiming again | Every gate from files; independent reviewer; canary-gated blind judging |

---

## 7. Open items for the maintainer
- Approve S6 as the S10 base (vs S8).
- Approve the teacher budget and sources for the SFT v5 buckets.
- Decide whether `docs/EVAL_S9_INDEPENDENT_REPORT.md` (Antigravity's untracked doc, which reads clean
  stops as "solves") should be deleted or corrected. It should not be committed as is.
