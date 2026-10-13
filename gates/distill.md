# Gates: Task A distillation on F: (2026-10-13, Claude Code)
Plan: `C:\Users\Joel Anthony\.claude\plans\read-and-execute-handoff-2026-10-13-post-iterative-micali.md`
Judge: Antigravity (model Gemini 3.8 Flash), via files; judges only, never writes rows.
PY = E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe

## D1 Move finished
- [x] D1a repo copy removed, build_sft_trial + launcher + gates point to F:
  CHECK: git -C E:\Parentheses\parentheses-0.9 status --short scripts/distill_reasoning.py
  EXPECT: D  scripts/distill_reasoning.py
  EVIDENCE: `git rm` done (history 4db7066); build_sft_trial.py:101 default F:\...\out\sft_tail_distill.jsonl; launcher lines 26-37 use F: script/logs; session2.md P3 note added
- [x] D1b self-tests
  CHECK: PY scripts\build_sft_trial.py --self-test ; PY -I F:\Lingual_Data\scripts\distill_reasoning.py --self-test
  EXPECT: "[self-test] build_sft_trial ok" and "[self-test] distill_reasoning ok"
  EVIDENCE: both printed 2026-10-13; `_tmp` count 0 after
- [x] D1c check_docs no new stale refs from the move
  EVIDENCE: 11 stale (was 10); the 1 extra is out/swarm/COORDINATION_120M.md:439, Antigravity's shared log naming the F: path (left as is). Old handoff line fixed.
- [x] D1d F:\Lingual_Data\distillation\README.md written (layout, commands, English-only, seed licenses, _tmp exception, per-teacher output terms, Antigravity judging)
  EVIDENCE: written 19:10. NVIDIA API Trial ToS v.2025-09-19 read in full (§1.2/§1.4 internal testing only, §4.2 no distribution) -> Joel: all rows research-only, llama-3.2-11b dropped; per-teacher licence table with sources

## D2 Refusal filter
- [x] D2a REFUSAL_RE + stats["refused"] + --log-rejects, self-test case (disclaimer-wrapped correct code is refused and logged)
  EVIDENCE: self-test prints "[distill] refused c1" then "[self-test] distill_reasoning ok"

## D3 Seeds
- [x] D3a load_seeds code=474 math=7473, cached under F:\Lingual_Data\distillation\hf_cache
  EVIDENCE: "code 474 math 7473"; hf_cache 8,247,931 B

## D4 Live
- [x] D4a --list-models works
  EVIDENCE: 80 ids -> F:\Lingual_Data\distillation\logs\nim_models.txt
- [x] D4b smoke: deepseek-ai/deepseek-v4.1-flash code --limit 3, rows hand-read
  EVIDENCE: mbpp-601 kept (real reasoning: tests two chain definitions against the assert); mbpp-602 rejected correctly (test expects string "None"); 3rd call 504 x5 -> crash, fixed by per-seed api_error handling (self-test case added)

## D5 Bake-off (every eligible text model, --limit 20 code + 20 math)
- [ ] D5a eligible list written in README (safeguard/embed/vision/reward/translate excluded)
  EVIDENCE: pending
- [ ] D5b per-teacher table: pass rate, refused, too_long, reasoning present, latency
  EVIDENCE (partial, 18:20): out\bakeoff\summary.md. 54 probed: 38 dead (404 EOL), 8 complete, 7 slow-but-alive re-running with --probe-timeout 300 (run3). 0 refusals anywhere. Harness fixes: probe timeout, resume skips logged rejects, summary/pack dedupe retried rejects. Hand-read: glm-5.3-flash, super-120b traces real and English; llama-3.2-11b answer-only (no think).

## D6 Antigravity eval
- [x] D6a blind packet + key + RUBRIC.md in F:\Lingual_Data\distillation\eval\
  EVIDENCE: bakeoff1_000..010 = 521 rows, 521 unique ids, no teacher names in inbox; keys in eval\keys; distill_eval.py --self-test ok
- [x] D6b Antigravity verdicts returned, audited per row (all ids, row-specific comments, no generator script)
  EVIDENCE: bakeoff1: 11/11 batches clean (521 verdicts, 520 unique comments, 0 keep on non-pass rows; spot-checked arithmetic correct). Found a real verifier bug (bold Answer) -> fixed, rescore moved 3 rows. Note: Antigravity wrote eval\bakeoff1_table.json itself (unblinded after judging).
- [ ] D6c bakeoff2 (slow teachers) packed, judged, audited
  EVIDENCE: pending (pass 2 still running 19:25)

## D7 Joel decisions
- [x] D7a teacher set + output-terms OK from Joel
  EVIDENCE: 2026-10-13: all live teachers incl. rate-limited; research-only (trial terms); drop llama-3.2-11b; ising dropped on quality (keep_pass 0.43); muse-glimmer HOLD (licence unknown)

## D8 Full run + SFT rebuild + review
- [ ] D8a full runs code then math (resume-safe), logs on F:
  EVIDENCE (running): distill_full.ps1 19:18, 5 teachers x {code,math}, disjoint --shard k/5; ~180 rows in first 3 min; pack loop hourly -> eval\inbox\full_NNN (first: full_000..003, 161 rows). Merge so far: 349 rows (bakeoff1) in out\sft_tail_distill.jsonl, 0 dup
- [ ] D8b documentation.md entry, vault handoff, independent review .agents/reviews/2026-10-13-distill.md
  EVIDENCE: pending
