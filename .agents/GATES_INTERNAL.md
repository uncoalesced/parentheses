# GATES — Overnight 0.9-NM (brief: .agents/open/OVERNIGHT_0.9_NM_BRIEF.md)

Rule: a box is checked only with EVIDENCE replaced by real proof (output lines, paths, numbers).
`ABANDON: <id> <reason>` = honest exit for an impossible gate. Python = `.venv\Scripts\python.exe`.

## A. Math review + slab_v2
- [x] A1 SLA self-test (old + slab_v2 fp64 parity: step, dense, reset, isolation, short)
  CHECK: .venv\Scripts\python.exe -m model.selective_linear_attention
  EXPECT: "[self-test] slab_v2:" line, no AssertionError
  EVIDENCE: 02:24 "[self-test] slab_v2: step 1.1e-15, dense 9.8e-16, reset 8.9e-16, isolation 0.0e+00 ok"; old path "chunk diff 4.635e-15 ... deltanet ok". RE-RUN 10:15 after closed-form recurrence + conv1d rewrite: "slab_v2: step 9.2e-16, dense 1.1e-15, reset 8.9e-16, isolation 2.2e-16 ok" + new 5-chunk test (T=320, docs spanning 3-4 chunks, both settings, step & dense < 1e-9, finite grads) passes
- [x] A2 backbone self-test, both slab_v2 settings (forward==step, stream==step)
  CHECK: .venv\Scripts\python.exe -m model.backbone
  EXPECT: "(slab_v2=True): generate/step parity ok"
  EVIDENCE: 02:24 "wiring (slab_v2=True): forward/step parity ok (max abs diff 2.220e-16)", "stream (slab_v2=True): generate/step parity ok"
- [x] A3 train.py --self-test (incl. warmup lr_mult asserts)
  CHECK: .venv\Scripts\python.exe train.py --self-test
  EXPECT: "[self-test] train ok"
  EVIDENCE: 02:24 "[self-test] train ok"
- [x] A4 old checkpoints still load and compare equal (v2 + v3 cfg == preset-compatible, state_dict loads)
  CHECK: python one-liner: load v2 ckpt into Parentheses(ckpt cfg), strict load
  EXPECT: loads with no missing/unexpected keys
  EVIDENCE: 02:24 v2 step_760999_final + v3 step_790000: "<All keys matched successfully>", cfg==PRESETS['parentheses-0.9-300k-selective'] True, slab_v2 False

## B/E. Corpus v3
- [x] B1 stage-1 text for kan/tam/tel/mal/eng/tcy/kff/kxv/nit, no esg
  CHECK: dir F:\Lingual_Data\data\processed_v3\text
  EXPECT: 9 jsonl, no esg.jsonl
  EVIDENCE: 02:24 "eng kan kff kxv mal nit tam tcy tel" .jsonl (9, no esg); stats: kan 274,756 docs/1.50 GB 471 s, tam 220,623, tel 276,047, mal 289,156, eng 172,965 pseudo-docs/1.50 GB, tcy 3,839/27.5 MB, kff 248, kxv 243, nit 248
- [x] E1 encoded train.bin + boundaries at target shares (eng 40, main four 12.5 + leftover, low-res <=10% capped 8 epochs)
  CHECK: manifest.json languages[*].share / epochs
  EXPECT: shares within 1 pt of target, every low-res epochs <= 8
  EVIDENCE: 10:55 v3_32k mix, 760,006,005 tokens, 900,161 docs: eng 0.40; kan/tam/tel/mal 0.1383 each (12.5 + 1.33 leftover); tcy 0.0348 kff 0.0046 kxv 0.0040 nit 0.0036 (low-res 4.7%, all at the 8-epoch cap: 8.0/8.004/8.003/8.001 — whole-doc granularity overshoots by <0.01 epoch); main four 0.64-0.74 epochs (no repeats); eng 0.69
- [x] E2 val round trip exact on every language file; boundaries strictly increasing, first 0
  CHECK: manifest.json val[*].round_trip_exact + encode asserts
  EXPECT: all true
  EVIDENCE: val exact {kan..nit, eng: True} x9; train boundaries b0 0, strictly increasing True, last < len True; max id 32767; tags at doc starts kan 189,914 tam 164,232 tel 176,065 mal 213,987 tcy 30,713 kff 1,985 kxv 1,945 nit 1,985 eng 119,335; <esg> (id 11) count in train = 0
- [x] E3 manifest has sources, licenses, shares, epochs, checksums, tokenizer hash, esg-excluded note
  CHECK: keys in manifest.json
  EXPECT: tokenizer_sha256, checksums_sha256, esg, languages.*.license present
  EVIDENCE: tokenizer_sha256 7a9b4187d5bc...; 20 checksums; "esg": "EXCLUDED -- Aheri Gondi is (c) Wycliffe, all rights reserved; <esg> is a reserved, unused token"; per-language license + share + epochs; sources block
- [x] E4 copy to E:\...\data\processed_v3 (checksums match)
  EVIDENCE: "files 20 checksum mismatches []" (sha256 of every E: copy == manifest)

## C. Tokenizer
- [x] C1 32k / 48k / 64k (indic regex) + 48k gpt-regex ablation exist
  CHECK: dir data\tokenizer\v3_*\tokenizer.json
  EXPECT: v3_32k v3_48k v3_64k v3_48k_gptregex
  EVIDENCE: 02:23 "v3_64k: 65536 ids, sha 9d9b78b3923c", "v3_32k: 32768 ids, sha 7a9b4187d5bc", "v3_48k: 49152 ids, sha 9c7cd56a1188"; 02:17 "v3_48k_gptregex: 49152 ids, sha 98f1af2a38d2"
- [x] C2 train_tokenizer.py --self-test (regex keeps aksharas whole, digits split, round trip, truncation)
  CHECK: .venv\Scripts\python.exe data\tokenizer\train_tokenizer.py --self-test
  EXPECT: "[self-test] train_tokenizer ok"
  EVIDENCE: 02:24 "[self-test] train_tokenizer ok" (asserts: indic regex keeps ಕನ್ನಡ as 1 piece, gpt regex splits it, "2026" -> 4 pieces, round trip incl. emoji/ZWNJ, truncate 600->560)
- [x] C3 study table shown in a terminal (mcp__terminal), round trip exact on every val language
  EVIDENCE: terminal tab c1 (~02:24): "round trip: exact on every language and tokenizer"; mean Indic bytes/tok 32k 9.94, 48k 10.76, gptregex48k 3.67, 64k 11.34; English 2.96/3.11/3.49/3.21. Pool encode also asserted exact round trip on every val doc of all 9 langs (out/v3pool_v3_64k.log)
- [x] C4 vocab picked by embedding <= ~35% of params rule, reason logged
  EVIDENCE: 32k chosen. At the chosen d384 L16: 32k emb 12.6M = 31% of 40.9M; 48k emb 18.9M = 40% (fails); 64k grid: every config with >=15x had emb 62-64% (d384 L8), so 64k fails at any size that trains to 15x in 6 h. Logged in documentation.md §11.

## D. Benchmark
- [x] D1 grid shown in a terminal; config + N chosen (T >= 2048, tokens >= ~15x params)
  EVIDENCE: tab c4 (64k, d384-640 rows; stopped at d640 L12 since all larger configs already < 6.2x), tab c5 (32k, d384/512 x L12/14/16 x T2048/4096 x v2, complete). Pick: "384  16  4096   0    40.9M   31%   2  6.5G      256   31,967     690M     16.9" = largest >= 15x with T >= 2048 (T=4096 same speed as 2048). d512 L12 32k = 10.8x (fails).
- [x] D2 preset parentheses-0.9-<N>M in model/config.py
  CHECK: python -c "from model import PRESETS, Parentheses; ..."
  EVIDENCE: "parentheses-0.9-41M 40944096", "parentheses-0.9-43M-slab2 43401696" (A/B arm)

## v3 (old run) eval
- [x] V1 out/eval_dravidian_v3.json exists + report_stats run
  EVIDENCE: v3 final step_820999_final.pt 02:52; old session's watcher wrote eval + Heafod Training Stats 02:54 (history.jsonl has v3-lowres@820999): drav 0.5972, kan .602 tam .526 tel .661 mal .569, tcy .776 kff .563 kxv .534 nit .725 esg .613, all_sources 1.7237; tags .94/.97/.94/1.0, low-res .97/.88/.88/.84/.91

## F. A/B
- [x] F1 baseline vs slab_v2 same data/seed/time, val bits/byte compared, winner kept (or ABANDON with reason)
  EVIDENCE: out/ab_bpb.json (11:26). Equal wall time: baseline 430 steps / 911 s, slab_v2 284 steps / ~900 s, seed 1234, 65,536 tok/step. bits/byte kan/tam/tel/mal/eng: baseline .965/.944/1.059/.854/3.382 (mean 1.4408), slab_v2 .989/.953/1.066/.876/3.300 (mean 1.4368). Diff 0.004 = 0.3% -> tie -> baseline (brief: ties to baseline); baseline wins 4/5 langs + tcy/kff/kxv; slab_v2 nit tag acc 0.00. Token-weighted mean would favour slab_v2 by 1.3% — logged openly in documentation.md §16.

## G. Main run
- [x] G1 launched detached via scripts/overnight_driver.py with warmup, cooldown 80%, min-lr 1e-5, ~0.5M tok/step
  EVIDENCE: 11:26:58 "preset=parentheses-0.9-41M steps=5379 warmup=108 decay_start=4303 ckpt_every=282 ce_rows=4096 tokens=705036288"; driver pid 24944; 131,072 tok/step (reasoned deviation from 0.5M, documentation.md §12); 11:29:58 status "step 40/5379, loss 9.861, steps_per_s 0.250, eta 17:25", GPU 97% 6.9 GB
- [x] G2 loss falling; final/latest checkpoint loads; tokens trained reported
  EVIDENCE: driver "DONE" 20:58:49, "saved final checkpoint: checkpoints/parentheses-0.9-41M\step_8082_final.pt"; out/h_final_check.txt: "load <All keys matched successfully>", "step 8082 cfg==preset True tokenizer {'path': 'data/tokenizer/v3_32k', 'sha256': '7a9b4187d5bc...'} params 40944096 tokens_trained 1059454976"; hourly mean train loss 7.058 -> 5.131 -> 4.340 -> 4.006 -> 3.835 -> 3.716 -> 3.643 -> 3.576 -> 3.513 -> 3.391 (every hour lower); 0 crash restarts

## M. Run monitoring (Joel 17:1x: "Run checks every hour starting at 6pm, also do one at 5:30pm ... logged properly and in detail. No summarising")
- [x] M1 all six scheduled checks present in out/run_checks_2026-10-07.md: 17:30, 18:00, 19:00, 20:00, 21:00, end-of-training
  CHECK: python -c "import re; s=open('out/run_checks_2026-10-07.md',encoding='utf-8').read(); print([t for t in ('hourly schedule 17:30','hourly schedule 18:00','hourly schedule 19:00','hourly schedule 20:00','hourly schedule 21:00','end of training') if 'scheduled: '+t in s])"
  EXPECT: all 6 labels listed
  EVIDENCE: 21:02 "['hourly schedule 17:30', 'hourly schedule 18:00', 'hourly schedule 19:00', 'hourly schedule 20:00', 'hourly schedule 21:00', 'end of training']" (+ setup baseline 17:26:52 + 17:30 correction note)
- [x] M2 every check reviewed by me within its hour: a status-log line here per check with rate vs previous (%), interval mean loss and delta, ETA margin vs 21:15, GPU temp/power, anomalies (or "none")
  EVIDENCE: status-log "M2 review" lines for 17:30 (artifact found + corrected), 18:00, 19:00, 20:00 below; 21:00 + end-of-training reviewed in documentation.md §24 (21:00: 0.2343 st/s -0.18%, hour 20-21 mean 3.3905 -0.1236, LR 1.120e-05 MATCH; end check = 0-step interval, "SLOWER by 100%" is a zero-interval artifact)
- [x] M3 the full checks file appended VERBATIM (not summarised) to documentation.md and to the Heafod session log
  CHECK: python: body of out/run_checks_2026-10-07.md (after its header) is a substring of .agents/documentation.md and of Heafod "Session Log 2026-10-07.md"
  EXPECT: True True
  EVIDENCE: 21:04 "M3 True True"

## H. Eval + report
- [x] H1 bpb eval: new final (out/bpb_eval_0.9.json, via out/h_chain.ps1) + v2 + v3 (out/bpb_eval_old.json, done 11:34), 8 langs + English, tag accuracy
  CHECK: both JSON files exist; new one has bpb for kan tam tel mal tcy kff kxv nit eng
  EVIDENCE: 21:00:17 eval exit 0; "H1 new langs ['eng','kan','kff','kxv','mal','nit','tam','tcy','tel']"; old = v2 step_760999_final + v3 step_820999_final. bits/byte new kan .5218 tam .4922 tel .6232 mal .4443 tcy .6524 kff .5248 kxv .4530 nit .7178 eng 1.6527 (vs v2 -39.2/-37.2/-39.4/-46.0/-58.0/-92.0/-93.3/-60.2/-27.4%); tags kan .94 tam 1.00 tel .94 mal .97 tcy 1.00 kff .88 kxv 1.00 nit .97; swap gaps +0.106..+0.201
- [x] H2 generation samples (3/lang, 300 tokens) in Heafod
  EVIDENCE: Heafod "Samples 0.9-41M.md" (21:00:17): "H2 sections 9 samples 27" (9 languages x 3, 300 tokens, from tag alone, T 0.8 top-k 40)
- [x] H3 report_stats.py: new run in BPE_RUNS, bits/byte chart + table regenerated
  EVIDENCE: 21:00:21 "[info] 13 evals, 3 bpb evals, 8 charts -> ...Training Stats"; Training Stats.md contains parentheses-0.9-41M@8082 True, "Bits per byte" True, 0.9-41M BPE run row True, loss_curves_bpe.png True, bits_per_byte.png True

## Journal (Joel's explicit ask: "journal everything in documentation.md and also in heafod")
- [x] J1 documentation.md appended (never edited above), covers every stage, timings, numbers, fallbacks, deviations
  EVIDENCE: 10 append-only writes 74,901 -> ~143.6k chars (each asserted the previous tail before appending); entry "2026-10-07 — overnight 0.9-NM" §1-30: A slab_v2, B/E corpus, C tokenizer + study table, D grids + pick, F A/B, G launch + 2 extensions, hourly reviews, H results table, 22-item failure/fallback list (§27), next schedule (§28), verbatim checks (§29), heading note (§30)
- [x] J2 Heafod `Session Log 2026-10-07.md` covers what ran, timings, numbers, failures, fallbacks, schedule
  EVIDENCE: written 21:03:33, 36,380 chars: result table, timeline 01:50-21:00, hourly table, decisions, NPU, 22 failures/fallbacks, next steps, verbatim run-check log ("checks verbatim: True")
- [x] J3 Heafod HANDOFF.md LATEST box updated
  EVIDENCE: 21:04 LATEST (2026-10-07, evening) box: 0.9-41M checkpoint path, recipe, bits/byte vs v2, tags, links to Session Log 2026-10-07 / Samples 0.9-41M / Training Stats, esg/v3 status, next steps; old box kept as "Previous (2026-10-07, morning)"
- [x] J4 final chat summary: results table, failures, every fallback, next schedule with times; numbers re-measured
  EVIDENCE: final chat message ~21:06; every number re-measured 21:00-21:05 from out/bpb_eval_0.9.json, out/bpb_eval_old.json, out/h_final_check.txt, out/run_checks_2026-10-07.md, GATES count "checked: 28 unchecked: 1" before this box -> 29/29

## Status log (append only)
- 02:00 B extract launched (5 detached procs); C tokenizers launched; A coded
- 02:09 B done (all 9 langs, ~1.5 GB main/eng); C relaunched on jsonl sample after English-block crash
- 02:17 C 48k gpt-regex done; 64k running
- 02:23 C all tokenizers done; study shown (tab c1)
- 02:25-09:51 BLOCKED: Write to Heafod Session Log (outside session folders) waited on a permission prompt while Joel slept; v3 finished 02:52 meanwhile, watcher ran its eval
- 09:51 Joel approved; 09:56 Joel: no time cuts; then: deadline 19:15
- 09:55 64k pools encoding (val round trip exact all 9); grid attempt 1 showed spill (15 GB "VRAM") + 2.7k tok/s
- 10:00-10:15 fixes: chunked CE (no full logits), memory-fraction cap, closed-form inter-chunk recurrence (3.1x: 10.7k -> 33.6k tok/s), slab_v2 conv via conv1d + sparse reset correction, fp32-only-for-reductions; all parity gates re-pass + new 5-chunk stress test
- 10:16 full grid running (tab c4)
- 10:21 64k grid stopped at d640 L12 (rest ineligible); 48k pool stopped; 32k pool started
- 10:23 32k grid (tab c5) complete -> parentheses-0.9-41M; slab_v2 nonzero-sync fix 17.8k -> 21.0k tok/s
- 10:39-10:58 32k pool done; mix (HDD random-read stall -> in-RAM rewrite, 70 s); copied to E:, checksums ok
- 10:55 A/B chain launched (out/ab_chain.ps1); 10:56-11:05 driver tests: success, crash x3 -> FAILED, kill mid-run -> resume from step_600 -> DONE
- 11:10 A/B arm A done (430 steps, 911 s, loss 6.11); arm B running
- 11:26 A/B eval done -> tie -> baseline; 11:26:58 main run launched (5379 steps, 705M tokens)
- 11:30-11:34 old-model bpb eval on CPU (slowed GPU run to ~0.21 st/s; no more CPU work during training)
- 11:34-11:44 clean rate 0.233 st/s -> end ~17:52 (< 18:05 guard)
- 11:4x GATES.md mojibake from a PowerShell Get-Content/Set-Content round trip; repaired byte-exactly in Python
- 14:36 status step 2680 loss 3.7852 eta 17:46; Joel: limit now 20:30
- 14:45:35 step_2820.pt saved -> driver stopped (no steps lost) -> 14:46:07 resumed, max_steps 6739, decay_start 5391, 883,294,208 tokens (21.6x), guard 19:40; "resumed ... at step 2820, continuing to step 6738", step 2830 loss 3.9149
- 14:50 Joel: limit 22:15 hard, zero slack; option 1 (extend to ~21:00) agreed
- 15:06 step_3102 saved but my wait command was held until 15:58; 16:05:27 step_3948 saved -> stopped -> 16:05:49 resumed: max_steps 8083, decay_start 6466, 1,059,454,976 tokens (25.9x), deadline 21:15, watcher hard-kill 21:25; step 3960 loss 3.7319
- 16:09:54 stage-H chain armed (out/h_chain.ps1, pid 50808): on driver exit -> load check, bpb eval, samples -> Heafod, report_stats
- 16:30 clean rate 0.2333 st/s at step 4280 -> step 8083 at 21:01
- 17:11 Joel status check: step 4870, loss 3.7443, 0.237 st/s, eta 20:56
- ~17:15 Joel: "Run checks every hour starting at 6pm, also do one at 5:30pm ... logged properly and in detail. No summarising" -> out/run_check.py (full detailed section per check -> out/run_checks_2026-10-07.md) + detached out/run_checks_scheduler.ps1 (pid 11248: 17:30, 18:00, 19:00, 20:00, 21:00, end-of-training)
- 17:26:52 setup baseline check: step 5090, loss 3.6095, interval 16:30->17:26 0.2374 st/s (FASTER 1.75% vs 0.2333), interval mean loss 3.6760; ETA 20:56:55 (+18.1 min margin); GPU 97% 86 C 85 W 7.5 GB; AC 100%; stderr clean; 18 checkpoints
- M2 review 17:30:00 check: step 5140, last loss 3.5418, interval mean 3.6330 (-0.0430, -1.17% vs 3.6760); printed rate 0.2650 (+11.65%) / ETA 20:35 = QUANTIZATION ARTIFACT (3.1-min interval timed on wall clock vs 10-step log lines) -> re-measured on log timestamps 5090@17:26:06 -> 5140@17:29:39 = 0.2342 st/s (-1.35% vs 0.2374); correction appended to the checks file; run_check.py now times intervals on log timestamps; hour 17:00-18:00 0.2364 st/s; real ETA ~20:56 (margin ~+18 min vs 21:15); GPU 97% 87 C (+1) 84.07 W 7.48 GB P2; AC 100%; RAM free 0.54 GB (counter excludes standby cache); stderr clean; LR MATCH 1e-3; 0 crash restarts; anomalies: none in the run itself
- ~17:45 Joel: "my Ryzen 7 250 AI CPU has an NPU which is idle, if possible find a use for it ... just try using it" -> found NPU Compute Accelerator (PCI DEV_1502, "NPU Phoenix", XDNA1), XRT 2.21.0 + xrt-smi/pyxrt shipped in C:\Windows\System32\AMD, driver 32.0.20102.3930, fw 1.5.5.391; ORT: venv 1.24.4 [Dml, CPU], system 1.26.0 [Azure, CPU] -> no VitisAI EP (needs Ryzen AI SDK = new dependency)
- 17:51:26 NPU used: xrt-smi validate -r all on NPU, Performance mode: latency 85.0 us PASSED, throughput 26,323 op/s PASSED (out/npu_validate_2026-10-07.json); model inference on NPU deferred (SDK install = new dep + CPU contention vs run + 22:15 close); proposed next session: INT8 retrieval embedder on NPU via export_embedder_onnx.py + VitisAI EP, and NPU-side eval while GPU trains
- M2 review 18:00:00 check: step 5560 (68.80%), 728,891,392 tokens (17.80x); rate on log timestamps 5140@17:29:39 -> 5560@17:59:31 = 0.2344 st/s (4.267 s/step, 30,720 tok/s), +0.08% vs 0.2342 = flat; interval mean loss 3.6284 (n=42, min 3.4623, max 3.7492), -0.0047 (-0.13%) vs 3.6330; hour 17:00-18:00: 840 steps, 0.2354 st/s, mean 3.6428 (-0.0728 vs 16-17); ETA 20:58:52 (+16.1 min vs 21:15); LR MATCH 1e-3, constant phase (cooldown at 6466); GPU 95% 84 C (-3) 85.57 W 7.41 GB P2; AC 100%; RAM free 1.48 GB; E: 74.1 GB free; stderr clean; 0 crash restarts; new ckpt step_5358 17:45:24; anomalies: none (empty 18:00-19:00 table row = no steps logged yet at 18:00:00, expected)
- M2 review 19:00:03 check: step 6410 (79.31%), 840,302,592 tokens (20.52x); rate on log timestamps 5560@17:59:31 -> 6410@18:59:33 = 0.2360 st/s (4.237 s/step, 30,937 tok/s), FASTER 0.71% vs 0.2344; interval (= hour 18:00-19:00) n=85 mean 3.5757, min 3.3764 (lowest of run so far), max 3.7420, -0.0527 (-1.45%) vs 3.6284; hour delta -0.0671 vs 17-18 (slightly smaller drop than -0.0728, still constant LR); ETA 20:57:36 (+17.4 min vs 21:15); LR MATCH 1e-3, still constant (cooldown at step 6466, ~56 steps / ~4 min after this check); GPU 95% 86 C (+2) 83.24 W 7.41 GB P2 2722 MHz; AC 100%; RAM free 0.9 GB; E: 72.7 GB free (22 ckpts x 0.49 GB); stderr clean; 0 crash restarts; new ckpts step_5640 18:05:19, step_5922 18:25:09, step_6204 18:45:06; anomalies: none
- M2 review 20:00:01 check: step 7250 (89.71%), 950,403,072 tokens (23.21x); rate on log timestamps 6410@18:59:33 -> 7250@19:59:11 = 0.2347 st/s (4.260 s/step, 30,769 tok/s), SLOWER 0.54% vs 0.2360 (within the run's 0.2344-0.2376 hourly band); interval (= hour 19:00-20:00) n=84 mean 3.5130, min 3.3074 (new run low), max 3.8220, -0.0627 (-1.75%) vs 3.5757; COOLDOWN ACTIVE since step 6466 (~19:03): last LR 5.200e-04 vs expected 5.197e-04 MATCH; hourly loss drop -0.0627 vs -0.0671 in the last constant-LR hour (cooldown not yet visibly accelerating the hourly mean; LR only halved so far); ETA 20:58:15 (+16.7 min vs 21:15); GPU 95% 87 C (+1) 80.19 W 7.41 GB P2 2760 MHz; AC 100%; RAM free 0.57 GB; E: 71.3 GB free; stderr clean; 0 crash restarts; new ckpts step_6486 19:05:07, step_6768 19:25:07, step_7050 19:45:09; anomalies: none
- 12:07 step_564.pt: "<All keys matched successfully>", cfg==preset True, tokenizer sha 7a9b4187..., optimizer present; status step 560 loss 5.6305, 0.235 st/s, eta 17:48

