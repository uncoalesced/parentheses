# Parentheses 1.0-240M: independent post-S9 evaluation

**Date:** 2026-11-08 · **Evaluator:** Claude Code (Opus), with Haiku judges · **Checkpoints:**
S6 (`step_12000.pt`), S8 golden (`dpo-swa3-golden.pt`), post-S9 blend (`s6-s8-blend-best.pt`, w_S8 = 0.40).

## Bottom line

1. **The "450/900 all-time high" does not hold up.**
   - It comes from a lenient keyword grader.
   - The blend was picked on the same 60 probes it was scored on.
   - The margin over S6 is statistical noise.
   - Under strict grading the blend is *last* of the three on the 60 probes.
2. **On the held-out Hermes 1000 suite, all three checkpoints are at the floor: about 2% of available
   points, and 0–1 fully correct answers out of 1000.**
   - None is significantly better than another.
   - The blend is the most fluent-looking but loops the most.
3. **The certification gates on the blend are worse than or equal to S8 golden on everything that
   tests answers** (60-probe pass rate, identity, honesty). It is better only on BPB, which measures
   language modelling, not answering.
4. **What the model is today:** a fluent Dravidian/English text continuer. At 240M parameters it
   cannot yet follow a task to a correct answer. It mostly echoes the prompt, loops, emits tool-call
   JSON, or invents facts confidently. It almost never says "I don't know."

**Recommendation:** do not ship the blend as the 1.0-240M release on the strength of 450/900.
Neither S6, S8 nor the blend is release-ready as an assistant.

---

## 1. Hermes 1000 (held-out, strict, blind)

**Protocol** (pre-registered before seeing outputs):
- Greedy decoding, no repetition penalty, 10 s decode-only watchdog.
- Objective items (150) are auto-scored by the suite's own scorer.
- Rubric items are judged blind by Haiku against `JUDGE_BRIEF.md` (strict anchors), with planted
  canaries.
- Deterministic vetoes: the Dravidian answer number must match, and the answer must be ≥ 50% in the
  native script.
- PRIMARY score = points earned, with 0 for any response that hit the watchdog or looped.

| | Blend | S6 | S8 |
|---|---|---|---|
| Score (out of 1000 items) | **20.7 (2.1%)** | 18.4 (1.8%) | 17.0 (1.7%) |
| Fully correct answers | 0 | 0 | 1 |
| Usable responses (stopped cleanly, no loop) | 476 | 519 | **619** |
| Watchdog stops (unfinished / looping) | **470** | 395 | 316 |
| Objective items correct (strict scorer) | 0 / 150 | 0 / 150 | 1 / 150 |
| Objective answer appears anywhere (lenient upper bound) | 14 / 150 | 15 / 150 | 16 / 150 |
| Content score on a judged sample of unusable answers | 0.014 | 0.019 | 0.030 |

**Paired comparison** (986 items, broken suite items excluded):
- blend − S6: +0.002 per item, 95% CI [−0.003, +0.008], 39 better / 38 worse. **Not significant.**
- blend − S8: +0.004, CI [−0.003, +0.011], 43 / 42. **Not significant.**
- S8 − S6: −0.001, CI [−0.009, +0.006]. **Not significant.**

**By domain** (points out of 100 items each, PRIMARY):

| Domain | Blend | S6 | S8 |
|---|---|---|---|
| D01 Quantitative reasoning | 0.5 | 0.0 | 1.8 |
| D02 Code tracing | 0.0 | 0.0 | 0.0 |
| D03 Scientific inference | 0.0 | 0.0 | 0.3 |
| D04 Statistics / causal | 0.0 | 0.0 | 0.0 |
| D05 Engineering diagnosis | 0.2 | 0.0 | 0.0 |
| D06 Long-context synthesis | 0.8 | 2.0 | 0.0 |
| D07 Heritage / space cards | 3.0 | 4.2 | 0.8 |
| D08 Epistemic calibration | 3.2 | 2.2 | 3.2 |
| D09 Dravidian reading (arithmetic) | 1.0 | 1.5 | 2.5 |
| D10 Dravidian controlled generation | 12.0 | 8.4 | 8.4 |

The only domain with visible signal is D10. That credit is mostly for writing in the right script
and copying given dates or rooms, not for meeting the format.

**Typical outputs** (verbatim, checkpoint hidden):
- *Mixing problem* "7 L at 10% + 3 L at 20%" (answer 13%) gives a page of LaTeX ending `\boxed{3x}`.
- *Rectangle 19 × 10* gives "A = 19 × 10 × 10 = 1900 cm²", then "P = 3800 cm²".
- *Tamil "17 L, use 5, add 3"* gives "தோட்டத் தொட்டியில் 17 லிட்டர் தண்ணீர் உள்ளது" (restates the first
  fact).
- *False-premise "2024 Westmoor report"* produces an invented description of the report.
- *Precision/recall*: `<tool_call>[{"name": "preprocess_tensor", ...}]`.
- *Best kind of answer seen*: "Report the conflict (23 vs 26) and avoid choosing a value…". Rare.

**Judge quality:**
- I hand-graded 100 rows blind (40 from the pilot, 60 from the full sheet). Exact agreement was
  88/100, MAE 0.1–0.2 points, κ(any credit) 0.74–0.75. Disagreements went both directions.
- All 55 negative canaries (wrong answer, loop, wrong language) passed in every accepted batch.
- Positive canaries (the reference pasted as the answer) often scored low. That is a flaw in the suite,
  not in the judges: many "reference answers" are rubric descriptions, not answers (see §5).
- 6 malformed judge rows were scored 0.

## 2. The 450/900 claim

- **Same grader, same files:** S6 434, S8 381, blend 450. These reproduce exactly.
- **Same harness, same precision (controls re-run):** S6 413, blend 450, S8 381. The 434 baseline was
  S6 in fp32; the sweep casts to S8's bf16.
- **Selection bias.** w = 0.40 was the best of 6 blends scored on these 60 probes. The sweep curve
  zig-zags (400 / 450 / 419 / 401 / 391 / 433).
- **Noise.** Paired bootstrap blend − S6 = +16, 95% CI [−52, +83], sign test p = 0.14.
- **The grader is lenient:**
  - any 20+ words without a loop earns Fluency 5;
  - any Dravidian keyword earns ≥ 4/4/4, even inside a loop (one looping answer scored 12/15);
  - code earns 3/3 for `def` + `return`;
  - arithmetic takes the first number found;
  - it accepts "95" for 48 + 37.
- **Strict blind re-grade of the same 180 outputs** (correct / usable / fabricates, out of 60):
  - S6: 3 correct, 9 usable, 13 fabricate.
  - S8: 5 correct, 17 usable, 17 fabricate.
  - **Blend: 2 correct, 21 usable, 19 fabricate.**
  - Major Dravidian and English conceptual were 0/30 for every checkpoint.
- **Showcase claims not supported by the scored run:**
  - In the run that scored 450, the blend answered 3x + 15 = 42 with "3x = 37, x = 19". The x = 9
    transcript came from a separate `converse.py` chat.
  - "100% marker accuracy" on 100 Hermes cases: the score file shows nothing was graded
    (`objective_checkable: 0`).

## 3. Certification gates on the blend vs S8 golden

| Gate | Blend | S8 golden |
|---|---|---|
| C1 BPB (all languages vs S6) | pass, slightly lower BPB everywhere (eng −0.016, san −0.098) | pass |
| C2 60-probe, greedy pass@1 | **2/60** | 6/60 |
| C2 sampled pass@5 | **5/60** | 10/60 |
| C4 identity | **20%** (fail) | 40% (fail) |
| C5 benign refusal | 0% (pass) | 0% (pass) |
| C6 honesty | 0% (fail) | 0% (fail) |
| C8 16k long context | pass, 54 tok/s | pass, 53 tok/s |

C3 (sheet only) and C7 (ONNX export) were not re-run on the blend.

## 4. What this means

- **Strength:** language modelling. The blend has the best BPB on every language. Script fidelity
  in Kannada, Tamil, Telugu and Malayalam is good. 16k context is stable.
- **Weakness:** instruction following and correctness.
  - On held-out tasks it almost never reaches a correct, complete answer.
  - About half of all responses loop until the watchdog.
  - Its default reaction to a question is to restate it, invent facts, or emit tool-call JSON
    learned from SFT data.
- **Honesty goal:** 0% on C6. False-premise prompts get fabricated answers. This is a training-data
  and objective problem, not just a 240M capacity limit: saying "I don't know" is cheap.
- **Blend vs parents:** averaging weights trades S8's better termination (fewer loops) for S6's
  fluency. It does not add capability.

## 5. Problems with the instruments themselves

- **Hermes suite:**
  - 14 broken items: D04 "90 successes out of 10" (rates over 100%), and D06 references with leaked
    template variables (`cap+i*5`).
  - 10 cosmetic items ("one-3th").
  - D07 and D10 references are rubric text, not answers.
  - D10 references share one generic template.
  - Native-speaker review of the 200 Dravidian prompts is still pending.
- **`score_specialized_eval_1000.py --prepare-review`** silently drops watchdog-stopped answers,
  which hides about half the failures.
- **`grade_rubric_900.py`** is too lenient to serve as a release gate (see §2).
- **The runner's watchdog originally counted prompt prefill.** It is fixed in commit 4430a36, and
  all three runs here use the fix.
- **Blend file:** 1,381 rows for 1,000 IDs. Antigravity restarted the run 3 times. Re-runs were
  byte-identical, and deduplication keeps the latest complete row.

## 6. Inputs for the next-session discussion

1. **Fix looping first.** Over a third to a half of responses never terminate under greedy decoding.
   Options:
   - SFT on short, terminated answers, with EOS weighting;
   - unlikelihood or a repetition penalty in training;
   - a default repetition penalty at inference, measured separately.
2. **Teach "answer, then stop" and "I don't know".**
   - Curated SFT that pairs false-premise / unanswerable prompts with short abstentions.
   - Remove tool-call-format examples from general SFT.
   - Make the C6 honesty gate a hard gate.
3. **Arithmetic and format.** Short worked arithmetic with a mandatory `Final:` line. The model
   never emits the format the evaluators require.
4. **Evaluation.**
   - Retire the 900-point heuristic as a gate.
   - Fix the Hermes generator (v1.1).
   - Build a private, never-pushed held-out suite.
   - Keep blind judging with canaries.
   - Select checkpoints on a dev split and report on a separate test split.
5. **Scale or distillation.** At 240M, reasoning tasks score near 0 even when the format is right.
   Decide whether 1.0 targets "fluent multilingual LM" (which the evidence supports) or "assistant"
   (which it does not support yet).

## Evidence
All files are under `out/eval_s9_indep/` in the repo:
- `final_report/hermes1000_report.json`: every number in §1.
- `final_report/judge/`: blind batches, judge grades, `PRIVATE_map.json`.
- `final_report/human_totals_batch005.jsonl` and `judge_pilot/`: hand grades.
- `blind60_grades.tsv` and `blind60_results.json`: strict 60-probe re-grade.
- `controls_60p.log`, `cert_blend/`, `suite_qa.json`, `template_overlap.json`, `contam_*.txt`.
- `JUDGE_BRIEF.md`.
- Scripts: `aggregate_final.py`, `filter_runs.py`, `build_judge_sheet.py`.
