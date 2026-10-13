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
- [x] P3 distiller rewritten (hosted teacher, verified rows only) and self-test passes (script moved to F:\Lingual_Data\scripts 2026-10-13)
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
- [x] R1 training reached step 3815 (or a clean deadline save); final checkpoint loads; status DONE
  EVIDENCE: launch 00:00 (preflight "ALL CRITICAL GATES PASSED", driver PID 15308); chain 07:56:34 "driver 15308 exited: stage: G main run parentheses-0.9-120M DONE | step: 3810 | loss: 3.0329 | restarts: 0"; `step_3815_final.pt` 1,436,790,527 B 07:55; s2_load.log "load <All keys matched successfully> / step 3815 cfg==preset True ... optimizer True"; ~15 s/step (0.067 steps/s)
- [x] R2 val watcher logged ≥ 6 rows, no ALERT_DIVERGENCE (or alert explained)
  EVIDENCE: out/session2_val_log.jsonl 8 rows (1907, 2100, 2400, 2700, 3000, 3300, 3600, 3815), all "alert": false; no out/locks/ALERT_DIVERGENCE. Step 3815 vs 1907: kan 3.6598/3.6749, tam 3.3084/3.3217, tel 3.7515/3.7512, mal 3.0646/3.0818, tcy 3.4642/3.4937, kff 2.6041/2.6288, kxv 2.4713/2.4791, nit 3.5301/3.5167, eng 3.6434/3.6628 (mean -0.013; slow drift, near the 4x1024-token noise floor)
- [x] R3 bits/byte step_1907 vs step_3815, 9 langs (`out/bpb_eval_120M_s2.json`)
  EVIDENCE: chain "END bpb exit 0 in 5 min" (~1.0 MB/lang, same val text). 1907 -> 3815: kan 0.5017->0.4966 (-1.0%), tam 0.4755->0.4737 (-0.4%), tel 0.6038->0.6011 (-0.45%), mal 0.4244->0.4207 (-0.9%), tcy 0.6471->0.6455 (-0.2%), kff 0.5324->0.5331 (+0.1%), kxv 0.4530->0.4530 (0), nit 0.7165->0.7220 (+0.8%), eng 1.5245->1.4791 (-3.0%). Generated-sample repetition rose (tam 0.09->0.16, tel 0.12->0.17, mal 0.10->0.14); distinct-2 fell in every Dravidian lang
- [x] R4 250 queries, base format: step_1907 and step_3815 (script_ok / eos / loop), same seed
  EVIDENCE: q_final "exit 0 in 29 min", q_1907 "exit 0 in 31 min"; both decode {t 0.7, top_k 50, min_p None}, seed = script default 1234 (chain passes no --seed). n=250 each. step_1907: script_ok .931 / eos .176 / loop .008 / rep .181 / d2 .553; step_3815: .963 / .124 / .012 / .217 / .527. More on-script, but fewer EOS stops and more repetition (matches R3 sample diversity drop)
- [x] R5 decoding sweep on step_3815 (6 configs × 60-query subset), best config named by the stated rule
  EVIDENCE: 6 configs x 58 queries (up to 8/lang: eng/kan/tam/tel/mal 8, tcy 6, hin 4, kff 3, kxv 3, nit 2; script_ok scored on 54; the gate's "60" was approximate), all "exit 0 in 7 min". script_ok/eos/loop/rep/d2: t0.5_k40 .889/.086/.069/.502/.284; t0.6_k40_minp0.1 .870/.121/.138/.624/.248; t0.7_k20 .926/.172/0/.227/.473; t0.7_k50 .944/.121/0/.214/.507; t0.7_k50_minp0.05 .944/.138/.121/.399/.393; t0.9_k50 .963/.190/0/.035/.700. Rule (max script_ok-loop, then eos) -> **t0.9_k50** (0.963 vs 0.944 = 1 query of 58; repetition 0.035 vs 0.214 is the real gap). Lower temperature / min-p cause loops
- [x] R6 SFT trial (150 steps on a copy, `checkpoints/parentheses-0.9-120M-sft-trial`) + 250 queries in chat format vs base
  EVIDENCE: sft_build = sft_v1 only (0 distill rows, no API key; 10,103,998 tokens, 36,760 docs). sft_train "exit 0 in 37 min", status DONE, restarts 0, `step_3965_final.pt` (3816..3965 = 150 steps), loss 3.5252 @3820 -> 3.0512 @3960. 250 queries (same decode t0.7 k50): SFT chat script_ok .947 / eos .768 / loop .028 / rep .123 / d2 .728 vs base step_3815 .963 / .124 / .012 / .217 / .527. Format learned (stops 77% vs 12%); content still weak (row 0, Python lambda question -> "Students.cpp -11.5 -1.05 -0.000 ..."); these metrics do not score correctness
- [x] R7 lm-eval (5 tasks, limit 500) step_1907 vs step_3815, if the budget allows
  EVIDENCE: lm_final / lm_1907 "exit 0 in 7 min" each; json "limit": 500. acc_norm (acc for winogrande) 1907 -> 3815: arc_challenge .192 -> .200, arc_easy .330 -> .334, hellaswag .340 -> .342, piqa .502 -> .534, winogrande .468 -> .486. stderr ~.017-.022, so every difference is within ~1.5 SE; both near chance (arc .25, piqa/winogrande .50)
- [x] R8 `out/session2_report.md` written; session-3 hyperparameter recommendation with the numbers behind it
  EVIDENCE: chain 10:51:44 "report exit 0 -> out\session2_report.md" (all sections filled, none "not run"); Claude Code appended "Extra: full 250 queries at the sweep pick" (t0.9 k50 on 3815: loop .012->0, rep .217->.045, script_ok .963 both) and "Session-3 recommendation" (8 items: LR 1e-4 constant; keep decay 7632 + rule "<0.5% main-4 bpb in s3 -> decay at 5724"; checkpoint cadence blocker 13.36 GiB free vs ~18.7 GiB needed -> CkptEvery 300 or Joel deletes intermediates; decode t0.9 k50; diversity watch; low-res flat; SFT needs distill rows; judge on bpb). Re-running session2_report.py would drop the appended sections
- [x] R9 everything done by 14:00; GPU free; journal + gates evidence filled from re-measured numbers
  EVIDENCE: chain "chain done, GPU free" 10:51:44; spare evals done 11:44:05, GPU.lock removed; 11:46 nvidia-smi "0 MiB, 0 %", no compute apps; PIDs 32656 (timer), 14700 (keep-awake), 15308 (driver), 17912 (chain), 36568 (val watcher, "driver DONE; exiting.") all exited. Journal "## 2026-10-13 — Session 2 results: 120M steps 1908 → 3815 + evals (Claude Code)" appended. Every number in R1-R8 read from output files with .venv python / session2_report.query_stats, not from memory; checkpoint sizes re-measured (1.338 GiB each)

REVIEW: `.agents/reviews/2026-10-13-session2.md` (fresh-context agent; Gemini CLI unauthenticated) — R1–R9 CONFIRMED; 4 slips fixed: tel -0.45%, sweep per-lang counts, report item 3 disk numbers (12.49 GiB not 13.83; CkptEvery 300 = 8 files 10.7 GiB not 9.4)
