# Session 10 (Tanzanite) build log: data, trainer patch, launcher

**Date:** 2026-11-09 (Mon) · **Author:** Claude Code (Sonnet 5.5) for Joel · **Status:** CPU-side build done, nothing trained yet.
Plan: `C:\Users\Joel Anthony\.claude\plans\based-on-e-parentheses-parentheses-0-9-d-soft-valiant.md`
(also copied to the vault as `Briefs\BRIEF_S10_DATA_AND_SCHEDULE.md`). Reference: `docs/S10_PLAN_IMPROVED_VS_NOT.md`,
`docs/EVAL_S9_STRICT_REPORT.md`. Release target: Friday 2026-11-13 night, "Tanzanite" (1.0-240M).

## 1. Decisions (Joel, 2026-11-09)

| # | Decision | Consequence |
|---|---|---|
| 1 | Base = S9 blend `checkpoints/parentheses-1.0-240M-s8-golden/parentheses-1.0-240M-s6-s8-blend-best.pt` | Saved step 13080, bf16, no optimizer state, cfg equals the `parentheses-1.0-240M` preset, 274/274 keys load. Training resumes at step 13081 with a 30-step LR warmup. Fallback: S6 `step_12000.pt` |
| 2 | Native abstention phrases: "anything" | Claude wrote fixed short phrases for ta/te/kn/ml. **Not reviewed by a native speaker.** tcy/kff/kxv/nit get none |
| 3 | NIM teachers: licence is the issue | NIM outputs stay out of training (policy fails trial-key outputs). NIM may be used for internal testing / judging only |
| 4 | Friday night: release "Tanzanite", a final version Joel is happy with | Friday daytime = packaging only (ONNX export check, model card, HF upload), at most one 2 h patch before noon |

## 2. Findings that change the S10 doc (do not edit the original; this is the correction)

1. **Abstention data was not absent from S7.5.** The S10 doc says `sft_v4_abstention_4000.jsonl` was left out. The S7.5
   manifest lists `explanatory_abstention_v4` at 1,485,429 tokens (2.3%), and the S6 anneal used the same file
   (3,744 unique traces repeated to 11,031 rows). It was present, long-form `<think>` text, and low in variety.
2. **Tool-call data entered earlier than S7.5.** The S6 anneal manifest (`processed_s6_anneal_240m`) includes curated
   SmolTalk with the `apigen-80k` function-calling subset (3.4M tokens). That likely feeds the tool-call habit seen on S6.
3. **Sources already trained on** (Trained-On ledger + manifests): Sangraha kan/tam/tel/mal, MetaMathQA, Orca-Math,
   NuminaMath, SmolTalk (curated v2), Aya tail, Belebele. Nothing in S10 re-sources them; they are only reshaped.
4. **E: is nearly full** (about 17 GB free). Checkpoints go to `I:\parentheses\checkpoints\...` (about 99 GB free).

## 3. Sources and licences (checked against `LICENSE_POLICY.md`)

New downloads, read live from the HF card by `scripts/s10_fetch_sources.py` (it refuses a dataset whose tag is not on
the pass list) into `F:\Lingual_Data\raw\hf\s10\<name>\` with a `SOURCE.txt` each:

| Dataset | Licence tag | Size | Use |
|---|---|---|---|
| rajpurkar/squad_v2 | cc-by-sa-4.0 | 17.7 MB | grounded reading + "passage doesn't say" (train split only; validation untouched) |
| google-research-datasets/nq_open | cc-by-sa-3.0 | 4.7 MB | question pool for the R-Tuning probe (not used yet) |
| lime-nlp/Synthetic_Unanswerable_Math | mit | 9.6 MB | unanswerable side only: "not enough information" |

Rejected after checking: `mandarjoshi/trivia_qa` (licence unknown), `ai4bharat/samanantar` and the IndicNLG sets (CC-BY-NC),
FaithEval / `orca-unanswerable*` / `sarvamai/gsm8k-indic` (no licence tag), raw `HuggingFaceTB/smoltalk` (Llama-generated rows;
only the already-curated `smoltalk_v2` subsets `smol-constraints`, `smol-rewrite`, `smol-summarize` are used), and every
tool-call set. The Trained-On ledger is **not** updated yet: it records only data a finished run consumed. Append the three
new sources after the S10 run consumes them.

## 4. Files created or changed

| File | What |
|---|---|
| `scripts/s10_fetch_sources.py` | licence-gated downloader (F: only, TEMP/HF cache on F:). `--self-test` |
| `scripts/s10_gen_math.py` | short worked math, ends `Final: x`; 11 English families + native-script arithmetic (tam/tel/kan/mal). Answers computed in Python; self-test re-derives 1,063 answers independently |
| `scripts/s10_gen_code.py` | 16 program templates executed for the real stdout, answer `Output: ...`; self-test re-runs 120 programs in a fresh `python -I` |
| `scripts/s10_gen_honesty.py` | false premise (fake English and native-script entities, wrong-location corrections, future events), benign facts, SQuAD2 grounded rows, unanswerable math, short form of abstention_4000, identity variants. Eval prompts of identity_15 / honesty_10 / benign_refusal_50 are removed |
| `scripts/build_s10_sft_240m.py` | the corpus builder (reuses S7.5 encoder, replay and FFD packer); digit-masked contamination gate; `--smoke` for a 4% build |
| `models/model_slab_v4.py` | `eos_weight` attribute (default 1.0, absent = unchanged) in `chunked_loss` and the full-logits loss |
| `train.py` | `--eos-weight` flag, `_eos_weight_self_test()` called from `--self-test` |
| `scripts/launch_s10_240m.ps1` | `-Mode pilot|full`, `-Base blend|s6`; checkpoints to `I:`; GPU.lock + free-space preflight. **Not started** |
| `data/s10_gen/*.jsonl` | generated rows (math 134,000; code 80,000; false_premise 40,000; grounded 16,000; unans_math 14,000; abstention_short 3,400; identity 304; benign_facts 270) |
| `data/processed_sft_240m_s10/` | the corpus (see section 5). `data/processed_sft_240m_s10_smoke/` = 4% build |

## 5. The corpus (`data/processed_sft_240m_s10`, built 2026-11-08T19:37Z, i.e. Mon 2026-11-09 IST)

16,008 bins × 4,096 = **65,546,453 doc tokens** (44,265,209 supervised), 400,614 documents, 99.966% packing, 0 truncated.
SHA256: `train.bin` `eccd58f841d2…db54`, `train.mask.bin` `768e7420fe60…7885`, `manifest.json` `6e221f49d9b8…f240c`
(full values: run `sha256sum` on the files; the first 12 characters identify the build).
Checks run on the output: 0 `<tool_call>` tokens (id 17), `<think>` 483 = `</think>` 483, every document ends in `<|endoftext|>`,
decoded samples show prompt masked and response supervised.

| Bucket | Target | Actual tokens | Actual share |
|---|---|---|---|
| replay_11lang | 25% | 28,667,816 | 43.7% |
| short_math (+ short chat) | 20% | 12,452,514 | 19.0% |
| code_output | 15% | 9,002,146 | 13.7% |
| dravidian_qa | 15% | 8,738,709 | 13.3% |
| abstention | 15% | 2,872,035 | 4.4% |
| grounded_reading | 5% | 3,276,893 | 5.0% |
| identity | 5% | 536,340 | 0.8% |

**Deviations from the plan, and why**
- Replay is 43.7%, not 25%: the builder lets replay absorb every shortfall so the corpus still reaches 1,000 steps. It also
  protects BPB (gate G8). The cost is that behaviour data is about 56% of tokens, not 75%.
- Abstention is 4.4%: there are only so many distinct false-premise shapes before the model just memorises them. The R-Tuning
  rows (section 7) add a little. A 253M model gets few NQ-open answers right, so they add well under 1M tokens.
- Identity is 0.8%: only 304 unique rows exist (47 canonical answers); upsampled 20×. More unique paraphrases would help. Past
  identity results were 3/15 to 6/15, so expect gate G5 to be the hardest.
- Dravidian QA is mostly the Aya tail (about 5.0M of the 8.74M tokens after the 700-token filter; the manifest's per-source
  figure of 8.85M counts it before that filter), then Kannada/Telugu curated (2.4M) and Belebele (1.35M). Aya rows go up to
  700 tokens, so this bucket is not as short as the others. Watch G1 (clean stop) on Dravidian prompts.
- Quarantine (final build): 393 probe-contaminated, 3,140 too long, 408 duplicates, 5,524 IndicQA-marked, 539 over 4,096 tokens.
  The digit-masked gate checked 1,075 eval prompts. It recorded no `eval_contaminated` rows, so none were dropped; the generators
  were already written with wording different from Hermes. Re-run the gate against suite v2 once that exists.
- **Gate evidence (2026-11-09):** the gate fired on 400/400 copies of real eval prompts with their numbers changed (so it does catch
  templates). On the generated files it flagged 0 rows: math 0/134,000, code 0/80,000, false_premise 0/40,000, unans_math
  0/14,000, grounded 0/16,000. Highest 8-gram share against any eval prompt in a 4,000-row sample: math 0.00, code 0.06,
  false_premise 0.00, unans_math 0.00, grounded 0.00.

## 6. How to reproduce

```bash
.venv\Scripts\python.exe scripts\s10_fetch_sources.py
.venv\Scripts\python.exe scripts\s10_gen_math.py --n-en 110000 --n-drav 24000 --out data\s10_gen\math.jsonl
.venv\Scripts\python.exe scripts\s10_gen_code.py --n 80000 --out data\s10_gen\code.jsonl
.venv\Scripts\python.exe scripts\s10_gen_honesty.py --out-dir data\s10_gen --n-squad 16000 --n-sum 14000 --n-fp 40000
.venv\Scripts\python.exe scripts\build_s10_sft_240m.py            # about 3 minutes
```

Self-tests that passed today: `s10_fetch_sources`, `s10_gen_math`, `s10_gen_code`, `s10_gen_honesty`, `build_s10_sft_240m`,
`train.py --self-test` (includes `--eos-weight`: weighted loss equals a hand computation, chunked equals full-logits, 1.0
changes nothing), `pytest tests/test_model_slab_v4.py` (7 passed, 1 skipped), `model.selective_linear_attention`,
`model.backbone`, `check_eval_contamination --self-test`. The launcher parses under the PowerShell parser.

## 7. Not done yet (in order)

1. **R-Tuning probe** (`scripts/s10_probe_rtuning.py`, GPU, about 1 h): ask the base model NQ-open questions; wrong answers become
   "I don't know" rows, right ones keep the gold answer; write `data/s10_gen/rtuning.jsonl`, then rebuild (the builder picks it up).
2. **Pilot** (`launch_s10_240m.ps1 -Mode pilot`, about 1 h): go/no-go is clean stop above 60% and `Final:` use above 30% on dev; if not,
   switch to `-Base s6`.
3. **Private eval suite v2** (dev 300 / test 700) and the C6-50 honesty set, kept on `I:`, never pushed.
4. Native-speaker check of the ta/te/kn/ml phrases in `s10_gen_math.py` (`DRAV`) and `s10_gen_honesty.py` (`IDK_NAT`, `NAT_Q`).
5. Full SFT (Tuesday), DPO v3 (Wednesday), gates on the test split (Thursday), packaging (Friday).
6. Append squad_v2 / nq_open / Synthetic_Unanswerable_Math to the Trained-On ledger after the run.

## 8. Housekeeping notes

- An empty folder `E:\tmp_s10` was created by mistake and could not be removed (the safety check blocks `rmdir`); delete it by hand.
- GateGuard asked for the same facts before each new file and first edit; they were given each time (callers, overlap check,
  schema, instruction).
- Nothing was written to `C:\`. The HF cache, temp and downloads are on `F:`.
- Nothing is committed to git.
