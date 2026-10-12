# Gates: Parentheses 0.9-120M Session 2 (Claude Code, 2026-10-12)

Window: 2026-10-13 00:00 → 14:00, one unattended run:
- training, steps 1908..3815, ~7.8 h at 14.7 s/step;
- then `out/session2_post_chain.ps1`: evals, the 250 queries, a decoding sweep, a short SFT trial, and the report.

Joel 2026-10-12:
- no GPU use before midnight;
- "tuning" = an SFT trial on a copy + a decoding sweep + session-3 hyperparameters.

Python = `.venv\Scripts\python.exe`. `ABANDON: <id> <reason>` = honest exit.

## P. Prep (before midnight, no GPU)
- [x] P1 val watcher fixed (min-step, exit race, alert) and runs on step_1907
  EVIDENCE: 19:5x `--once --min-step 1907` -> "step 1907 val losses: kan 3.6749, tam 3.3217, tel 3.7512, mal 3.0818, tcy 3.4937, kff 2.6288, kxv 2.4791, nit 3.5167, eng 3.6628" (scratch log); session-1 checkpoints skipped; stale Gemini val log moved to out/session2_val_log.gemini_test_2026-10-11.jsonl
- [x] P2 preflight has disk / V5_DATA_READY / newest-checkpoint gates and passes
  CHECK: .venv\Scripts\python.exe scripts\session2_preflight.py
  EXPECT: "ALL CRITICAL GATES PASSED"
  EVIDENCE: 20:06 "RESULT: ALL CRITICAL GATES PASSED -> READY FOR SESSION 2 LAUNCH" (E: >= 22 GB, V5_DATA_READY, newest = step_1907_final all PASS)
- [x] P3 distiller rewritten (hosted teacher, verified rows only) and self-test passes
  CHECK: .venv\Scripts\python.exe scripts\distill_reasoning.py --self-test
  EXPECT: "[self-test] distill_reasoning ok"
  EVIDENCE: 20:06 "[self-test] distill_reasoning ok"; real seeds 7,473 GSM8K train + 474 MBPP; MBPP reference solutions pass the gate 20/20; not live-tested (no API key yet)
- [x] P4 `run_120m_eval_250.py` takes `--min-p` and `--limit-per-lang` (sweep subset); its self-check passes
  CHECK: .venv\Scripts\python.exe scripts\run_120m_eval_250.py --self-test
  EXPECT: "[self-test] run_120m_eval_250 ok"
  EVIDENCE: 20:06 "[self-test] run_120m_eval_250 ok"; CPU smoke on step_1907 (chat, min-p 0.05, 1/lang, 10 rows) wrote rows with decode {temperature 0.7, top_k 50, min_p 0.05}
- [x] P5 `scripts/build_sft_trial.py` merges sft_v1 + verified distill rows (mask/boundaries aligned); self-test passes
  CHECK: .venv\Scripts\python.exe scripts\build_sft_trial.py --self-test
  EXPECT: "[self-test] build_sft_trial ok"
  EVIDENCE: 20:06 "[self-test] build_sft_trial ok"; real build: 10,103,998 tokens, 36,760 docs, byte-identical to processed_sft_v1 (0 distill rows yet)
- [x] P6 `scripts/session2_report.py` builds the report from whatever outputs exist; self-test passes
  CHECK: .venv\Scripts\python.exe scripts\session2_report.py --self-test
  EXPECT: "[self-test] session2_report ok"
  EVIDENCE: 20:06 "[self-test] session2_report ok"; on real out/ prints "not run" for missing sections and the step_1601 queries row 0.927/0.172/0.020
- [x] P7 post-chain dry run: every stage's command line printed, time budget logic exercised, no GPU used
  CHECK: powershell -ExecutionPolicy Bypass -File out\session2_post_chain.ps1 -DryRun
  EXPECT: "dry run ok"
  EVIDENCE: 20:06 "dry run ok"; full budget plans all 16 stages (~4 h 47 min); with -End now+150 min it SKIPs sft_train, q_sft_chat, q_1907, lm_final, lm_1907. CPU smokes of the real commands passed: query eval, bpb eval (list, step 1907, 9 langs), lm-eval arc_easy limit 2, SFT build
- [x] P8 launcher: training stops by 10:30 (post window >= 3 h 20 min to 13:50), chain armed; both .ps1 parse with 0 errors
  EVIDENCE: launcher param -Until 2026-10-13 10:30 -End 2026-10-13 13:50; starts session2_post_chain.ps1 after training launch; parse errors 0 / 0 (launch_120M_session2.ps1, session2_post_chain.ps1); chain aborts on missing driver PID
- [x] P9 journal entry appended (documentation.md) with the plan + timings
  EVIDENCE: documentation.md "## 2026-10-12 — Session 2 window plan: 00:00 → 14:00 unattended (Claude Code)" count 1; check_docs 10 stale: 9 not from this work (build_corpus_v3/v4, tokenize_lowres, check_judging) + 1 Gemini handoff line naming the deleted distill_coder_7b.py

## R. Run (filled 2026-10-13)
- [ ] R1 training reached step 3815 (or a clean deadline save); final checkpoint loads; status DONE
  EVIDENCE: pending
- [ ] R2 val watcher logged ≥ 6 rows, no ALERT_DIVERGENCE (or alert explained)
  EVIDENCE: pending
- [ ] R3 bits/byte step_1907 vs step_3815, 9 langs (`out/bpb_eval_120M_s2.json`)
  EVIDENCE: pending
- [ ] R4 250 queries, base format: step_1907 and step_3815 (script_ok / eos / loop), same seed
  EVIDENCE: pending
- [ ] R5 decoding sweep on step_3815 (6 configs × 60-query subset), best config named by the stated rule
  EVIDENCE: pending
- [ ] R6 SFT trial (150 steps on a copy, `checkpoints/parentheses-0.9-120M-sft-trial`) + 250 queries in chat format vs base
  EVIDENCE: pending
- [ ] R7 lm-eval (5 tasks, limit 500) step_1907 vs step_3815, if the budget allows
  EVIDENCE: pending
- [ ] R8 `out/session2_report.md` written; session-3 hyperparameter recommendation with the numbers behind it
  EVIDENCE: pending
- [ ] R9 everything done by 14:00; GPU free; journal + gates evidence filled from re-measured numbers
  EVIDENCE: pending
