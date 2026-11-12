# DATA LEDGER (disk accounting)

Disk-usage accounting for the Parentheses 0.9 line. Training-data provenance lives in `F:\Lingual_Data\Trained-On\LEDGER\`; this file tracks what occupies disk and what may be reclaimed.

## 2026-10-13 cleanup accounting (Sonnet subagent for Claude Code)

Read-only audit. Nothing was deleted, moved or modified. Sizes are measured byte counts (du -sb or file sizes), shown in GiB (2^30). The machine-readable DELETE/ASK list is `out\cleanup_candidates.tsv` (category, bytes, absolute path, reason; no overlaps).

No single disk ledger existed (the closest are `F:\Lingual_Data\Trained-On\LEDGER\` for what was trained, and its `delete_list_2026-10-10.md` for the last purge), so this file was created.

### Method and verification facts
- Live corpus for sessions 3-5: `data/processed_v5/{train.bin,train_boundaries.npy,sft_tail.bin,sft_tail_boundaries.npy}` (from `out/launch_120M_session.ps1`, tokenizer `data/tokenizer/v3_32k`). The E: copy of `processed_v5/train.bin` is byte-identical to the F: original (cmp).
- `scripts/build_corpus_v5.py` replays F:\ pools `processed_v3/pools/v3_32k` and `processed_v4/pools/v4_32k` and reads `raw-sourced/common-pile-stackexchange-cs-hw-robotics`; val = `data/processed_v4/val`. `build_sft_tail.py` reads raw-sourced aya, indicqa, oasst2, bespoke_stratos, hermes.
- E: `processed_v3/train.bin`, `train_boundaries.npy` and `processed_v4/train.bin`, `train_boundaries.npy` were cmp'd against the F: originals: byte-identical, so the E: copies are pure duplicates.
- No script or launcher reads any 120M intermediate checkpoint (grep of scripts/, out/*.ps1, out/*.py, train.py); they appear only in historical docs. The launcher resumes from the newest file by LastWriteTime, which is step_3815_final.pt.
- `F:\Lingual_Data\raw\kn.txt`, `raw\kan.parquet`, `raw\tam.parquet` have exactly the same byte size as `raw\indiccorp\kn.txt`, `interim\normalized\kan.parquet`, `interim\normalized\tam.parquet`. Probable duplicates (36.57 GB) but NOT hash-verified (F: is a USB HDD; no big random reads).
- esg (Aheri Gondi, research-only) lives at: `E:\Parentheses\parentheses-0.9\data\processed_dravidian_lowres_v1\esg` (5.9 MB, inside a KEEP dir), `F:\Lingual_Data\data\processed_dravidian_lowres_v1\esg`, and raw `F:\Lingual_Data\raw\tribal\esg` (USFM). It is not in `processed_v3/v4/v5` (a gate verified 0 esg tokens in v3).

### Free space now
- E: 13.34 GiB free (14320316416 bytes). Session 3 needs about 10.7 GiB.
- F: 113.03 GiB free (121365594112 bytes).

### Table

| path | size (GiB) | category | reason |
|---|---:|---|---|
| `E checkpoints/parentheses-0.9-120M intermediates` | 29.88 | DELETE | 23 files: session-1 step_150..1800 (9) + step_grown + session-2 step_1950..3750 (13). No script reads them (only historical docs); resume uses newest = step_3815_final; *_final.pt files are kept/ASK |
| `E:\Parentheses\parentheses-0.9-archive\out` | 6.06 | DELETE | archived out/: old NPU exports (npu_eval 6.50 GB, includes probe600 cooled export) + 107 old logs; Joel's cleanup brief says archive dead items are deletable; MANIFEST.tsv lists each path |
| `E:\Parentheses\parentheses-0.9-archive\venv` | 4.62 | DELETE | legacy venv (live one is repo .venv); only a docstring usage line named it |
| `E:\Parentheses\parentheses-0.9-archive\checkpoints\parentheses-0.9-120M-run1-lr3e-4` | 2.68 | DELETE | dead run: lr 3e-4 degraded the grown model (launcher comment); only its log was referenced and the log is in out/ |
| `E:\Parentheses\parentheses-0.9\data\processed_v4\train.bin` | 2.05 | DELETE | E: duplicate of F:\Lingual_Data\data\processed_v4\train.bin (cmp byte-identical); 41M-v4 run finished; v5 builder reads F: pools, not this. Kept: val/ (eval + watcher read it) and manifest.json |
| `E:\Parentheses\parentheses-0.9-archive\checkpoints\parentheses-0.9-41M-sft-v1` | 1.83 | DELETE | superseded 41M chat SFT v1 (documentation.md 2026-10-09: v2 replaced it); no script input |
| `F:\Lingual_Data\data\processed_v3\pools\v3_64k` | 1.76 | DELETE | 64k-vocab pool; 32k tokenizer is the live one; only model_stats.py mentions it in a string |
| `E:\Parentheses\parentheses-0.9\data\processed_v3\train.bin` | 1.42 | DELETE | E: duplicate of F:\Lingual_Data\data\processed_v3\train.bin (cmp byte-identical); no eval reads it (41M release trained on it, finished). Kept: val/ and manifest.json (check_model_card reads manifest) |
| `E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-120M-sft-trial` | 1.34 | DELETE | SFT trial copy (150 steps on sft_v1); output of session2_post_chain.ps1, nothing reads it as an input; regenerable |
| `E:\Parentheses\parentheses-0.9-archive\checkpoints\parentheses-0.9-120M-probe600` | 1.34 | DELETE | dead probe run |
| `F:\Lingual_Data\data\processed_v3\pools\v3_48k` | 0.73 | DELETE | partial 48k-vocab pool; tokenizer v3_48k is not used by the 32k line (41M/120M use v3_32k); nothing reads it |
| `E:\Parentheses\NPU\pip_cache` | 0.40 | DELETE | pip download cache (re-downloadable) |
| `E:\Parentheses\NPU\downloads` | 0.14 | DELETE | Miniforge3 installer .exe, already installed |
| `E:\Parentheses\parentheses-0.9-archive\data\processed_english_v1` | 0.13 | DELETE | old English-only processed set; superseded by all_sources_v1 and v3/v4 English pools; no code refs |
| `E:\Parentheses\NPU\conda_pkgs` | 0.12 | DELETE | conda package cache (env conda_envs\ryzen-ai is already built) |
| `E data/raw/manual/Unconfirmed *.crdownload (8 partial downloads)` | 0.09 | DELETE | 8 aborted Chrome partial downloads |
| `E:\Parentheses\parentheses-0.9\data\processed_sft_trial` | 0.03 | DELETE | SFT trial data copy; regenerated by scripts/build_sft_trial.py; only the trial checkpoint (also DELETE) consumed it |
| `E:\Parentheses\NPU\cache` | 0.01 | DELETE | pip cache folder |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\ChromeSetup.exe` | 0.01 | DELETE | browser installer, not data |
| `E:\Parentheses\parentheses-0.9\data\processed_v4\train_boundaries.npy` | 0.01 | DELETE | E: duplicate of the F: copy (cmp byte-identical) |
| `E:\Parentheses\parentheses-0.9\data\processed_v3\train_boundaries.npy` | 0.01 | DELETE | E: duplicate of the F: copy (cmp byte-identical) |
| `F:\Lingual_Data\tmp_ebible.html` | 0.00 | DELETE | temp scrape page in the data root |
| `E/archive __pycache__ dirs (6)` | 0.00 | DELETE | 6 dirs: regenerated python bytecode cache |
| `F:\Lingual_Data\tmp_tcy_index.html` | 0.00 | DELETE | temp scrape page in the data root |
| `F:\Lingual_Data\raw\hf` | 74.18 | ASK | 79.7 GB, 99.9% Sangraha-verified (CC-BY-4.0, re-downloadable); v3-v5 read only the first capped shards, so the rest was never trained on. Smaller sibling dirs: banjara, kannada_news, kurukh, opus, tulu_kannada |
| `F:\Lingual_Data\interim\normalized` | 53.85 | ASK | 57.8 GB normalized parquets (tam/tel/mal/mar/urd/kan + indiccorp); the intermediate step between raw and normalized/; mar and urd are unused by Parentheses; deletion needs the delete_list proposal procedure |
| `F:\Lingual_Data\raw\indiccorp` | 18.73 | ASK | 20.1 GB IndiCorp kn + gom raw; only copy of the Konkani text; Kannada part already folded into Trained-On v2 per storage_lifecycle_v2.md |
| `F:\Lingual_Data\raw\kn.txt` | 18.24 | ASK | 19.6 GB; same size as raw\indiccorp\kn.txt (probable duplicate copy; not hash-verified because F: is a USB HDD). IndiCorp Kannada, trained via Trained-On v2 |
| `F:\Lingual_Data\raw\tam.parquet` | 11.24 | ASK | 12.1 GB; same size as interim\normalized\tam.parquet (probable duplicate; not hash-verified) |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\project-gutenberg` | 10.88 | KEEP (Joel 2026-10-13: kept pending Antigravity eval job JOB_ANTIGRAVITY_GUTENBERG_EVAL) | 28,603 PG ebooks (PD, re-downloadable per SOURCE.txt) staged 2026-09-02; NOT in any processed corpus (all_sources_v1 used project-gutenberg-sample, v5 reads only stackexchange from here). Largest single E: item; never trained on |
| `F:\Lingual_Data\normalized` | 8.87 | ASK | 9.53 GB per-language normalized text (kan/tel/tam/mal/eng/gom/tcy); human-written sources, likely only copy of the cleaned form |
| `F:\Lingual_Data\interim\extracted_text` | 7.69 | ASK | 8.26 GB extracted ta/te/ml/kn jsonl; pipeline intermediate, no current script named |
| `F:\Lingual_Data\data\processed_v3\text` | 7.06 | ASK | 7.59 GB stage-1 text jsonl used to build v3 pools; read by train_npu_byte_filter.py and clean_corpus_npu.py; rederivable only from raw/ |
| `E:\Parentheses\parentheses-0.9\data\processed_all_sources_v1\train.bin` | 5.71 | ASK | byte-level replay corpus of the old runs; eval reads only val.bin. May be the only processed copy of wikipedia + 22 parallel corpora (their raw was purged from F:\Used_Data on 2026-10-10 per delete_list_2026-10-10.md); English part lives on as v3/v4 BPE pools on F: |
| `F:\Lingual_Data\data\processed_v4\text` | 5.61 | ASK | 6.02 GB stage-1 text jsonl for v4; same status as v3\text |
| `F:\Lingual_Data\raw\kan.parquet` | 4.58 | ASK | 4.9 GB; same size as interim\normalized\kan.parquet (probable duplicate; not hash-verified) |
| `E:\Parentheses\parentheses-0.9-archive\data\processed_multilingual_22` | 3.07 | ASK | 22-corpus multilingual byte set; nightly.ps1 can still be pointed at it; not on F:, likely the only processed copy of those parallel corpora |
| `E:\Parentheses\parentheses-0.9\data\processed_dravidian_v1\train.bin` | 2.92 | ASK | byte-run (v1-v3) Dravidian train set; eval reads only val.bin. F: copy exists but is NOT byte-identical (3,077,637,812 vs 3,140,034,496 bytes), so check which one the ledger hashes before deleting |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\common-pile-arxiv-filtered` | 2.47 | ASK | 42,706 CC arXiv papers (HF common-pile/arxiv_papers_filtered, re-downloadable); not read by build_corpus_v5.py or any processed set; never trained on |
| `E:\Parentheses\parentheses-0.9-archive\checkpoints\parentheses-0.9-41M-sft-v2` | 2.29 | ASK | the only chat-tuned 41M checkpoint (best SFT of the 41M line); no script input, but retraining costs a full SFT run |
| `F:\Lingual_Data\raw\papers` | 2.20 | ASK | 2.36 GB research PDFs/sources (screening ledger on F:); licenses mixed |
| `E data/raw/manual/*.pdf (loose textbook PDFs)` | 1.60 | ASK | 26 PDFs: OpenStax/other textbook PDF staged for review; not promoted into any corpus (prepare_manual_pdfs extracts to _pdf_extracted only); CC BY re-downloadable |
| `F:\Lingual_Data\raw\kannada_literature` | 1.46 | ASK | 1.57 GB Kannada literature; memory note: mostly in copyright, may be the only copy; do not delete without Joel |
| `F:\Lingual_Data\raw\sangraha` | 1.36 | ASK | 1.47 GB Kannada Sangraha shards read by the v3 build (Trained-On ledger); re-downloadable |
| `E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-120M\step_1601_final.pt` | 1.34 | ASK | session-1 interrupted 'final'; default --checkpoint of scripts/run_120m_eval_250.py and a column in session2_report.py (those only need out/*.jsonl, not the .pt) |
| `E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-120M\step_1907_final.pt` | 1.34 | ASK | session-1 end checkpoint; session2_preflight.py / session2_post_chain.ps1 name it as the session-2 base; deletable once session 2 is closed and the doc/preflight references are retired |
| `F:\Lingual_Data\raw\wikimedia` | 0.85 | ASK | 0.91 GB wikimedia dumps; re-downloadable |
| `E:\Parentheses\parentheses-0.9\data\raw\curated\project-gutenberg-sample` | 0.76 | ASK | 811 MB PG sample trained into all_sources_v1; processed copy is all_sources_v1/train.bin (ASK) |
| `E:\Parentheses\parentheses-0.9\data\processed_multilingual` | 0.56 | ASK | legacy multilingual gz train/val; only scripts/sample_translation.py default arg names it; ledger coverage unclear |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\proof-pile-2-open-web-math` | 0.55 | ASK | trained in all_sources_v1 (576M bytes); re-downloadable from HF; processed copy is all_sources_v1/train.bin (itself ASK) |
| `E:\Parentheses\parentheses-0.9\data\processed` | 0.50 | ASK | legacy byte-level processed train/val (0.53 GB); no current script reads it, AGENTS cleanup guard kept it; docs call it uncovered by the ledger |
| `E:\Parentheses\NPU\miniforge` | 0.46 | ASK | second Miniforge install next to miniforge3; looks like a duplicate of the base conda, verify nothing activates it |
| `F:\Lingual_Data\.feynman` | 0.40 | ASK | 0.43 GB unidentified tool state; unclear |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\glaive-function-calling-v2` | 0.25 | ASK | no script reads it (hermes copy is used instead); re-downloadable |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\oasst1` | 0.22 | ASK | OASST1 (Apache-2.0, re-downloadable); build_sft_tail uses OASST2 only; no script reads oasst1 |
| `E:\Parentheses\parentheses-0.9\data\processed_sft` | 0.14 | ASK | old SFT set (byte-era) with masks; no current script reads it; superseded by processed_sft_v1 |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\train.jsonl` | 0.11 | ASK | unidentified loose QA dataset files (BoliviaQA-looking, Spanish) with no SOURCE.txt; no script reads them |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\hh-rlhf` | 0.09 | ASK | carries DO_NOT_TRAIN.txt; excluded from all training; re-downloadable |
| `E:\Parentheses\parentheses-0.9\data\raw\sft\hermes-function-calling-v1` | 0.05 | ASK | older copy of hermes used only by data/prepare_sft.py (byte-era SFT); raw-sourced/hermes is the live copy |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\_pdf_extracted` | 0.04 | ASK | pdftotext review-queue output of the PDFs above; not promoted to training |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\toolace` | 0.03 | ASK | no script reads it; re-downloadable |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\validation` | 0.02 | ASK | loose directory next to the QA jsonl files, no SOURCE.txt; no script reads it |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\validation.jsonl` | 0.00 | ASK | unidentified loose QA dataset files (BoliviaQA-looking, Spanish) with no SOURCE.txt; no script reads them |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\test.jsonl` | 0.00 | ASK | unidentified loose QA dataset files (BoliviaQA-looking, Spanish) with no SOURCE.txt; no script reads them |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\boliviaqa-v1.0.json` | 0.00 | ASK | unidentified loose QA dataset files (BoliviaQA-looking, Spanish) with no SOURCE.txt; no script reads them |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\boliviaqa-v1.0.jsonl` | 0.00 | ASK | unidentified loose QA dataset files (BoliviaQA-looking, Spanish) with no SOURCE.txt; no script reads them |
| `E:\Parentheses\parentheses-0.9\.venv` | 5.02 | KEEP | live torch venv |
| `E:\Parentheses\NPU\ryzenai_sdk` | 4.54 | KEEP | NPU SDK install (4.9 GB) |
| `E:\Parentheses\NPU\conda_envs` | 3.51 | KEEP | ryzen-ai conda env (3.8 GB) |
| `F:\Lingual_Data\data\processed_dravidian_v1` | 3.02 | KEEP | F: copy of the byte-run Dravidian set (E: train.bin is ASK) |
| `F:\Lingual_Data\data\processed_v5` | 2.15 | KEEP | F: original of v5 (E: copy cmp identical) |
| `E:\Parentheses\parentheses-0.9\data\processed_v5` | 2.13 | KEEP | 120M training corpus: launch_120M_session.ps1 reads train.bin/boundaries/sft_tail (F: copy cmp identical) |
| `F:\Lingual_Data\data\processed_v4\train.bin` | 2.05 | KEEP | F: original of v4 corpus |
| `F:\Lingual_Data\data\processed_v3\pools\v3_32k` | 1.98 | KEEP | replayed by build_corpus_v5.py (--v3-pool-dir) |
| `F:\Lingual_Data\data\processed_v3\train.bin` | 1.42 | KEEP | F: original of v3 corpus (E: duplicate is DELETE) |
| `E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-120M\step_3815_final.pt` | 1.34 | KEEP | live model; session 3 resumes from it |
| `F:\Lingual_Data\data\processed_v4\pools\v4_32k` | 1.15 | KEEP | replayed by build_corpus_v5.py (--v4-pool-dir) |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\common-pile-stackexchange-cs-hw-robotics` | 1.06 | KEEP | build_corpus_v5.py input (code_dir) |
| `E:\Parentheses\NPU\miniforge3` | 0.58 | KEEP | conda base for the ryzen-ai env |
| `E:\Parentheses\parentheses-0.9\out` | 0.52 | KEEP | logs, query jsonl, bpb/eval json, session reports, npu_eval/120M_lr1e-4_step_1907 (0.55 GB, referenced by generate_120m_charts) |
| `F:\Lingual_Data\.venv` | 0.50 | KEEP | F: tools venv (0.53 GB) |
| `E:\Parentheses\parentheses-0.9\release` | 0.48 | KEEP | release scripts read it (restored 2026-10-12) |
| `E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-41M-v4\step_16082_final.pt` | 0.46 | KEEP | 41M-v4 foundation model and SFT parent (delete_list_2026-10-10 keeps it) |
| `E:\Parentheses\parentheses-0.9\checkpoints\parentheses-0.9-41M\step_8082_final.pt` | 0.46 | KEEP | 41M release candidate |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\oasst2` | 0.32 | KEEP | build_sft_tail.py input for the session-5 SFT tail |
| `E:\Parentheses\cordon` | 0.26 | KEEP | separate cordon project (0.27 GB) |
| `E:\Parentheses\parentheses-0.9\data\processed_dravidian_v1\val.bin` | 0.15 | KEEP | byte-eval val read by eval_dravidian.py |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\aya_dataset` | 0.13 | KEEP | build_sft_tail.py input for the session-5 SFT tail |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\bespoke_stratos` | 0.12 | KEEP | build_sft_tail.py input for the session-5 SFT tail |
| `E:\Parentheses\parentheses-0.9\data\processed_dravidian_lowres_v1` | 0.08 | KEEP | low-resource val read by eval_dravidian.py. CONTAINS esg (Aheri Gondi, research-only) subdir 5.9 MB; also at F:\Lingual_Data\raw\tribal\esg (USFM) and F:\Lingual_Data\data\processed_dravidian_lowres_v1\esg |
| `F:\Lingual_Data\data\processed_dravidian_lowres_v1` | 0.08 | KEEP | F: copy; contains esg (research-only) subdir |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\gutenberg-literature-swarm` | 0.07 | KEEP | 70 MB, small, hand-curated literature |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\hermes-function-calling-v1` | 0.06 | KEEP | build_sft_tail.py input for the session-5 SFT tail |
| `E:\Parentheses\parentheses-0.9\data\raw\curated\hermes-function-calling-v1` | 0.06 | KEEP | read by scripts/sample_freeform.py |
| `E:\Parentheses\parentheses-0.9\data\processed_all_sources_v1\val.bin` | 0.06 | KEEP | English/old-domain val read by eval_dravidian.py |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\openstax-textbooks` | 0.05 | KEEP | 58 MB, small; SOURCE.txt + licensed textbooks |
| `F:\Lingual_Data\Trained-On` | 0.05 | KEEP | historical ledger (55 MB) |
| `E:\Parentheses\parentheses-0.9\data\processed_sft_v1` | 0.03 | KEEP | SFT base for session 5 / build_sft_trial |
| `E:\Parentheses\parentheses-0.9\data\processed_v3\val` | 0.02 | KEEP | val set (eval / check_model_card) |
| `E:\Parentheses\parentheses-0.9\data\processed_v4\val` | 0.02 | KEEP | val set read by eval_dravidian, background_val_watcher, build_corpus_v5, npu_ckpt_eval, leak_probe |
| `F:\Lingual_Data\data\processed_v3\val` | 0.02 | KEEP | val |
| `F:\Lingual_Data\data\processed_v4\val` | 0.02 | KEEP | val |
| `F:\Lingual_Data\raw\tribal` | 0.02 | KEEP | 18.6 MB tribal-language sources incl. esg (research-only) at raw\tribal\esg; only copy |
| `E:\Parentheses\parentheses-0.9\data\tokenizer` | 0.01 | KEEP | v3_32k tokenizer (live) plus 48k/64k variants (12 MB total) |
| `F:\Lingual_Data\distillation` | 0.01 | KEEP | active work; do not touch |
| `E:\Parentheses\parentheses-0.9\data\raw\monolingual\tcy` | 0.01 | KEEP | Tulu monolingual with SOURCE.txt (11.9 MB, human-written) |
| `F:\Lingual_Data\data\processed_v4\train_boundaries.npy` | 0.01 | KEEP | F: original |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\indicqa` | 0.01 | KEEP | build_sft_tail.py input for the session-5 SFT tail |
| `F:\Lingual_Data\data\processed_v3\train_boundaries.npy` | 0.01 | KEEP | F: original |
| `F:\Lingual_Data\LEDGER` | 0.00 | KEEP | pipeline ledgers (2 MB) |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\LLM instructions` | 0.00 | KEEP | LLM instruction originals (1.4 MB, never delete) |
| `E:\Parentheses\parentheses-0.9\data\raw\llm-instructions` | 0.00 | KEEP | LLM instructions (1.3 MB) |
| `E:\Parentheses\parentheses-0.9\checkpoints\npu_byte_filter` | 0.00 | KEEP | NPU byte filter model + ONNX (1 MB) |
| `F:\Lingual_Data\Agents-Brain` | 0.00 | KEEP | coordination journal |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\steam-manuals` | 0.00 | KEEP | small |
| `F:\Lingual_Data\scripts` | 0.00 | KEEP | active work; do not touch |
| `E:\Parentheses\parentheses-0.9\data\processed_dravidian_v1\val_boundaries.npy` | 0.00 | KEEP | byte-eval val boundaries |
| `F:\Lingual_Data\languages` | 0.00 | KEEP | registry.yaml |
| `E:\Parentheses\parentheses-0.9-archive\MANIFEST.tsv` | 0.00 | KEEP | archive manifest (25 KB) |
| `E:\Parentheses\parentheses-0.9\data\raw\manual\personality-spec` | 0.00 | KEEP | personality spec (12 KB, never delete) |
| `E:\Parentheses\parentheses-0.9\data\processed_v4\manifest.json` | 0.00 | KEEP | corpus manifest |
| `E:\Parentheses\parentheses-0.9\data\processed_v3\manifest.json` | 0.00 | KEEP | read by check_model_card.py / model card |
| `E:\Parentheses\parentheses-0.9-archive\README.md` | 0.00 | KEEP | archive readme |
| `E:\Parentheses (all)` everything not itemised (code, docs, small dirs) | 0.06 | KEEP | not itemised: scope total 109.15 GiB minus all itemised rows |
| `F:\Lingual_Data (all)` everything not itemised (code, docs, small dirs) | 0.87 | KEEP | not itemised: scope total 232.18 GiB minus all itemised rows |

### Totals per category per drive (GiB)

| drive | DELETE | ASK | KEEP (itemised + remainder) | scope total |
|---|---:|---:|---:|---:|
| E:\Parentheses | 52.15 | 35.38 | 21.61 | 109.15 |
| F:\Lingual_Data | 2.50 | 216.34 | 13.34 | 232.18 |

Scope total for E: = repo + archive + NPU + cordon + small dirs under `E:\Parentheses` only (the drive holds other unrelated data, 240 GB used overall). F: = `F:\Lingual_Data` only.

### What DELETE alone buys on E:
- DELETE on E: 52.15 GiB, so E: free would go from 13.34 to about 65.49 GiB.
- Of that, the 120M intermediate checkpoints (session 1 step_150..1800 + step_grown, session 2 step_1950..3750) are 29.88 GiB.
- ASK items are not counted as freed. The biggest E: ASK: raw-sourced/project-gutenberg (10.88 GiB, never trained on) and the archive/data/dravidian/all_sources train.bin files.
- Raw-source rule applied: nothing human-written and possibly sole-copy is DELETE. Raw items that are re-downloadable open datasets and never trained on are still ASK, with that fact stated in the reason.


## 2026-10-13 follow-up (Claude Code)
- Joel ran out\cleanup_delete.ps1 -Really: 58/58 DELETE items gone (out\cleanup_deleted.log); E: free 66 GB after.
- raw-sourced/project-gutenberg moved ASK -> KEEP pending Antigravity's evaluation job.
