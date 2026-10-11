# Gates: Parentheses 0.9-120M code path (Claude Code, 2026-10-10 overnight)

Plan: `C:\Users\Joel Anthony\.claude\plans\lets-begin-the-next-glimmering-otter.md`.
Independent review: `.agents/reviews/2026-10-10-120M-growth.md` (APPROVE WITH WARNINGS, no blockers). Its 4 majors were fixed; see G9.

- [x] **G1 preset** `parentheses-0.9-120M` added last in `model/config.py` (d512, L32, 8×64 heads, T=4096, v32768, selective_linear).
  - EVIDENCE: `num_params()` = 119,702,272 (reviewer reproduced it).
- [x] **G2 growth is function-preserving**.
  - EVIDENCE: `scripts/grow_model.py --self-test` passed (fp64, eps=0: logits equal on the chunked + dense paths; identity blocks / new MLP units get gradient). Reviewer reproduced it.
  - Reviewer also checked the real 41M→120M shapes with random weights: max logit diff 2.1e-6.
- [x] **G3 step-0 loss on the real 41M-v4 checkpoint**, ratio ≤ 1.02.
  - EVIDENCE (`out\grow_120M.log`, 00:49):
    - kan val: src 3.6672 → grown 3.6674, **ratio 1.00004**.
    - eng val: src 3.3108 → grown 3.3107, **ratio 0.99997**.
  - Saved `checkpoints/parentheses-0.9-120M/step_grown.pt` (step -1, no optimizer).
- [x] **G4 Free Think on BPE**: `load_bpe_tokenizer`, `_stream_bpe`, `--lang`. Stops on turn-end tokens or `looping()`, and flushes held-back bytes at the end.
  - EVIDENCE: `free_think --self-test` ok; `modular_free_think --self-test` ok.
- [x] **G5 launcher** `out\launch_120M_session.ps1 -Session N`.
  - Defaults: bs 8 × accum 4 (131,072 tok/step), `--grad-checkpointing`, 8.5 h deadline. 9,537 steps total, 1,908 per session; cooldown from 7,632 (session 5 only).
  - Session 5: `--mix sft_tail.bin 2` = 25% of rows ≈ 62M tokens ≈ 1 pass of the real 62.6M v5 tail. Changed from 50% after the v5 audit: the tail is already ≤3× repeated internally. No SFT span mask (prompts are trained too).
  - EVIDENCE: parses with 0 errors. Not launched. Reviewer verified the session chaining arithmetic.
- [x] **G6 lm-eval** 0.4.14.dev0 installed (it only added packages; torch is unchanged) + `scripts/lm_eval_adapter.py`.
  - EVIDENCE: self-test passed. Real smoke on 41M-v4 at limit 50: arc_easy acc 0.22 / acc_norm 0.36, hellaswag 0.32 / 0.32 (`out\lm_eval_smoke_41M_v4.json`).
- [x] **G7 v5 audit script** `out/audit_v5_data.py`.
  - EVIDENCE: on the clean synthetic fixture, every gate is true.
  - Planted problems all fail as they should:
    - val tcy#158 in train → exact, prefix and near-dup checks all fail;
    - val kan#5 quoted inside an SFT tail answer → `G3_tail` false;
    - an `<esg>` token → G2 false.
  - Real run: ~09:00 after `V5_DATA_READY`.
- [x] **G8 self-tests after touching model/**.
  - EVIDENCE: selective_linear_attention ok; backbone ok, including the new `activation checkpointing == plain (grad diff 0.0e+00)`; `train.py --self-test` ok.
  - `check_docs.py`: 8 stale references, all pre-existing (F:-hosted build scripts).
- [x] **G9 120M fits and is fast enough** (new). Without activation checkpointing the 120M OOMed at batch 1 (`out\bench_120M.log`).
  - Fix: real per-block checkpointing in `Parentheses.hidden`, wired to `train.py --grad-checkpointing` (previously a no-op print) and `benchmark_step.py --grad-checkpointing`.
  - EVIDENCE (`out\bench_120M_gc.log`): bs 8, 5.6 GB, 3.52 s/step, **9,309 tok/s** → 1.25B tokens ≈ 37 h ≈ 5 × 7.5 h.
  - Reviewer major fixes:
    - `overnight_driver.py` OOM halving now halves `--mix` rows and never leaves zero main-data rows;
    - audit: near-dup 32-token span scan on train **and** sft_tail; licence allowlist + deny list ("all rights reserved", "in copyright", "research-only", NC); structural gates (tag-first, `<|endoftext|>`-last, ids < vocab, tokenizer sha).
