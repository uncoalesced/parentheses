# AGENTS.md — Parentheses 0.9

Canonical instructions for **every** coding agent working in this repo (Claude Code, Codex, Antigravity,
Cursor, Gemini, …). `CLAUDE.md` just imports this file. `.agents/AGENTS.md` points here. Keep one
source of truth: edit this file, not copies.

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL MANDATE FOR ALL AGENTS (Mamba-2 / RWKV / Selective Linear Attention):**
> Parentheses is **NOT a Transformer / GPT model**. Do **NOT** treat this codebase as a standard Transformer!
> - It does **NOT** use quadratic causal self-attention ($O(T^2)$ Softmax attention matrices).
> - It does **NOT** use standard Transformer KV-caches during generation.
> - Parentheses is a **from-scratch Recurrent / State Space Duality (SSD) model** belonging to the **Mamba-2 / RWKV / Selective Linear Attention** family:
>   - **Training (Parallel Form):** Chunked associative linear attention / semi-separable state space duality with linear $O(T)$ complexity.
>   - **Inference / Decoding (Recurrent Form):** Constant $O(1)$ recurrent memory state updates: $S_t = \alpha_t S_{t-1} + (k_t \otimes v_t)$ and $y_t = q_t S_t$. Fixed state size, zero growing KV-cache, zero sequence Softmax.
> Any agent proposing optimizations, exports (ONNX/Vitis AI/DirectML), quantization, evals, NPU offloads, or multimodal integrations **MUST** strictly operate under this linear-recurrent/SSM paradigm, not Transformers.

Parentheses 0.9: a from-scratch recurrent LM (selective linear attention, RWKV/Mamba-2 family) trained
on one RTX 5050 Laptop (8 GB) for Dravidian languages + English. Owner: Joel ("uncoalesced").

**Read before any non-trivial change:**
1. this file;
2. the **last entry** of `.agents/documentation.md` (session journal; the newest state);
3. `.agents/CLAUDE.md` (map of where things are) and `.agents/ARCHITECTURE.md` /
   `.agents/CODING_STANDARDS.md` for depth;
4. your brief in `.agents/open/`, if you were given one. Open work lives there; finished briefs
   move to `.agents/done/`.

**README.md's "Status" section is stale (2026-09-18).** When it disagrees with the newest
documentation.md entry, the journal wins.

## Current state (2026-10-07)
- Release candidate: `checkpoints/parentheses-0.9-41M/step_8082_final.pt`.
  - Preset `parentheses-0.9-41M`: 40.9M params, d384, 16 layers, 6 heads x 64, T=4096.
  - BPE tokenizer `data/tokenizer/v3_32k`.
  - Trained on 1.06B tokens of `data/processed_v3`.
- Beats the old byte models on every language (bits/byte, documentation.md 2026-10-07 §26).
- Open work: `.agents/open/HANDOFF_2026-10-08_POST_41M.md` (release, NPU, slab_v2 retry,
  distillation), `JOB_ANTIGRAVITY_LONG_RUN_PREP.md`, `JOB_DATA_LEDGER_AND_CLEANUP.md`.
- Ledger of the last session: `GATES.md` (repo root).

## Commands
- Torch Python: `.venv\Scripts\python.exe` (torch dev cu128, tokenizers, onnxruntime-directml).
  System `python` (3.11) has matplotlib (for `scripts/report_stats.py`) and onnxruntime 1.26.
- Run from the repo root: imports assume that cwd.
- Set `PYTHONIOENCODING=utf-8` for anything that prints Indic text (cp1252 crashes otherwise).

**Self-tests.** There is no pytest suite: each module carries assert-based self-tests on synthetic
data (no network, no real data). Run the relevant ones before and after touching code, and ALL of
these after touching `model/` or `train.py`:
```
.venv\Scripts\python.exe -m model.selective_linear_attention   # fp64: chunked==step==dense, resets, 5-chunk, slab_v2
.venv\Scripts\python.exe -m model.backbone                     # kv-cache, chunked CE==full CE, forward==step, stream
.venv\Scripts\python.exe train.py --self-test                  # batching, reset_mask, lr schedule, resume, AGC
.venv\Scripts\python.exe scripts\eval_dravidian.py --self-test
.venv\Scripts\python.exe data\tokenizer\train_tokenizer.py --self-test
python scripts\report_stats.py --self-test
python scripts\check_docs.py      # fails if a doc backtick-names code that no longer exists; run after any rename
```

Other self-tests that exist (flag-based):
- `-m features.npu_byte_filter --self-test`
- `-m features.free_think --self-test`
- `data/prepare_parallel.py --self-test`
- `data/tokenize_corpus.py --self-test` (`--verify` audits a real output dir; it is not a unit test)
- `scripts/mqar_eval.py --self-test`

**Common runs:**
- Benchmark grid (bf16, OOM back-off):
  `scripts\benchmark_step.py --grid --vocab 32768 --d-model 384 --layers 16 --seq-lens 4096`
- Eval, bits per UTF-8 byte on identical val text for byte and BPE checkpoints alike:
  `scripts\eval_dravidian.py --bpb --checkpoints <ckpt...> --out out\bpb_eval_<name>.json`
- Stats → Heafod: `python scripts\report_stats.py`. Add new runs to `RUNS` (byte) or
  `BPE_RUNS` (BPE).
- Long training: always detached, under the watchdog:
  `.venv\Scripts\python.exe scripts\overnight_driver.py --name <run> -- <train.py args>`
  - It resumes from the newest checkpoint on crash (×3), halves the batch / doubles grad-accum on
    OOM, and writes `out\overnight_status.txt`.
  - Detailed checks: `python out\run_check.py "<label>"`. It times intervals on log timestamps.
  - Useful train.py flags: `--seed`, `--warmup-steps`, `--decay-start`/`--min-lr`, `--deadline`,
    `--tokenizer`, `--ce-rows`, `--boundaries`, `--mix PATH ROWS [BOUNDARIES]`.
- Corpus v3 (scripts live on F:, not in the repo):
  `F:\Lingual_Data\scripts\build_corpus_v3.py extract|pool|mix`.

## Architecture (the parts that span files)
- **`model/config.py`**: `ModelConfig` + `PRESETS`, the single source of truth (no YAML).
  - New fields are always added **last with a default** (`attn_type`, `slab_v2`), so old pickled
    checkpoint cfgs still load and compare equal. `train.py --resume-from` refuses a cfg mismatch,
    and `features/free_think.py` depends on the same equality.
  - Byte presets use vocab 256.
- **`model/selective_linear_attention.py`** (primary attention): three paths that must agree to
  ~1e-15 in fp64.
  - `forward()`: chunked, C=64. The inter-chunk recurrence is a **closed form** (masked n×n
    decay matrix + one einsum, no Python loop over chunks), valid because decay is a scalar per
    head. This replaced a loop that made training launch-bound (3.1× speedup, 2026-10-07).
  - `_forward_dense()`: O(T²) fallback for T not a multiple of 64.
  - `step()`: the recurrent form used by `stream()` / Free Think.
  - `reset_mask` (0 at a document start) isolates packed documents.
  - `slab_v2=True` adds:
    - a reset-masked conv k=4 on qkv;
    - L2-normed q/k;
    - a per-head RMSNorm + silu output gate;
    - a longer decay init.
    With it off, ops and state_dict are identical to the old ones. It tied the baseline in the
    2026-10-07 A/B and is off by default.
  - `CausalSelfAttention` (in backbone.py) is the legacy baseline; no new development.
- **`model/backbone.py`**: `Parentheses`.
  - `forward(..., return_logits=False)` uses `chunked_loss` (checkpointed cross-entropy slices),
    so 32k–64k-vocab logits are never materialized. train.py relies on this.
  - `stream()`/`generate()` route selective models through `step()`.
  - Embeddings are tied.
- **`train.py`**: a plain loop, no Trainer abstraction.
  - Batches are random windows from a uint16 memmap. `--boundaries` → `boundary_reset_mask`.
  - LR = `lr_mult`: linear warmup → constant → linear cooldown.
  - `--seed` re-seeds the sampler after model init, so architectures with different param counts
    see identical batches.
  - `--deadline` gives a clean final save.
  - Checkpoints store `{model, optimizer, cfg, step, tokenizer:{path,sha256}}`.
  - Caps CUDA at 92% of memory: Windows otherwise **spills silently into system RAM at ~10× slower**.
- **Data:**
  - Byte corpora: `data/processed_dravidian_v1/`, `processed_dravidian_lowres_v1/<lang>/`,
    `processed_all_sources_v1/`. uint16 byte ids; docs `<kan> text\n`; `*_boundaries.npy` = doc
    start offsets.
  - BPE corpus: `data/processed_v3/`. Docs = `<lang>` special token + text + `<|endoftext|>`.
    Shuffled `train.bin` + `train_boundaries.npy`, plus per-language `val/<lang>.bin`; the val
    text is the old val splits re-encoded, so byte and BPE models are scored on the same text.
    `manifest.json` has licenses, shares and checksums.
  - Tokenizers `data/tokenizer/v3_32k|48k|64k` (+ `v3_48k_gptregex` ablation):
    - byte-level BPE whose regex keeps Indic combining marks (`\p{M}`) inside words;
    - 256 reserved special ids: `<|endoftext|>`=0, `<|pad|>`=1, language tags 2–11 (`<esg>`
      reserved, unused), chat/think/tool tokens, then reserved slots.
  - The data pipeline for raw sourcing: `data/*.py`, `data/README.md`.
- **Features**: `features/free_think.py` (open-ended decode) and `features/modular_free_think.py`
  (BM25 RAG spliced into the decode loop) work with any checkpoint through `stream()`.
  `features/conversation_memory.py` handles chat memory.

## Rules that bite (each one learned the hard way)
- **`.agents/documentation.md` is append-only.**
  - Never edit, correct or delete a past entry; add a new entry instead ("Correction to entry N").
  - Newest at the bottom, format `## <date> — <topic>` then `### <n>. <step>`.
  - Be concise: what was done, the number, what it means.
  - Re-read the file right before appending if another session may be live.
  - Don't journal on behalf of another session.
- **Never edit UTF-8 files with PowerShell 5.1 `Get-Content`/`Set-Content`.** It reads UTF-8 as
  ANSI and writes mojibake (this happened to GATES.md). Use your editor tools or Python with
  `encoding="utf-8"`.
- **Aheri Gondi (`esg`) is © Wycliffe, all rights reserved, research-only.** It must never enter a
  releasable model's training data. `<esg>` stays a reserved, unused token. The v3 byte model
  trained on it and is not releasable.
- **`F:` is a USB HDD.** Read it sequentially; never random-access it through a memmap (that turned
  a 70-second job into hours).
- **No CPU-heavy work while the GPU trains.** The trainer is partly CPU-bound (data loading, kernel
  launches); a 4-thread CPU job cut training speed ~16%.
- **Never live-tail a training log.** Monitor via `out\overnight_status.txt`, `run_check.py`, or
  after-the-fact reads.
- Don't launch a long training run unless asked. `train.py` defaults `--max-steps` to 100,000; pick
  a real number.
- **Writes outside the repo (Heafod vault, F:) can block on a permission prompt.** One stalled an
  unattended run for 7.5 h. In unattended work, do them while Joel is around or at the end.
- **No new dependencies without Joel's approval.** The AMD Ryzen AI SDK (NPU) is approved.
- **NPU (AMD Phoenix/XDNA1):**
  - Use it whenever a job is NPU-suited (fixed-shape quantized inference: embedder, small-model
    eval during training).
  - Otherwise leave it idle (the driver power-gates it); no keep-alive loops.
- **Verify delegated-agent claims against the actual files** (license calls, merges, sorting) before
  acting on them. This project has caught real discrepancies that way.
- Don't name a script after a stdlib module.

## Don't touch
- `.venv/`: vendored packages.
- `E:\Parentheses\parentheses-0.9-archive\`: files moved out of the repo on 2026-10-12. They include dead checkpoints, old data, old `out/` logs and NPU exports, finished briefs and the legacy `venv/`. Its `MANIFEST.tsv` maps each old path to its new one. Look there before assuming a historical file is gone.
- `checkpoints/`: weights (≈0.49 GB per 41M checkpoint).
- `data/raw/`, `data/tests/`, `data/processed*`: bulk binary/corpus data. Don't read, enumerate or
  regenerate them casually.
- `data/raw/manual/`: 20k+ staged files. Read only its `README.md`; scanning it file-by-file is a
  known hang.
- Working in `data/`: read `.agents/AGENT_WORKFLOW.md` and `data/.agents/AGENTS.md` first. They
  cover:
  - the license bar;
  - `*_AGENT.md` task briefs;
  - the shared shard writer `data/_shard_io.py`;
  - the task board `data/.agents/status_and_coordination.md`.
  Don't hand-roll a shard writer or re-scope a task that already has a brief.

## License gating on data (`data/raw/manual/`)
- **`raw-sourced/`**: third-party open data. One subfolder per source with a `SOURCE.txt` (origin
  URL, license **quoted verbatim**, pull date, size; copy the shape of
  `data/raw/monolingual/tcy/SOURCE.txt`).
  - Pass: Apache, MIT, CC0, CC-BY (and CC BY-SA, which share-alike affects the release license).
  - Fail: CC-BY-NC, "no license stated", unclear coverage. If unclear, note it and leave it.
- **`personality-spec/`**: Joel's own writing; no license question.
- **`LLM instructions/`**: has **no license**. Don't fold it into either category or treat it as
  trainable; that's Joel's call.
- Nothing under `data/raw/manual/` is swept by any pipeline default (`data/tokenize_corpus.py`
  `--raw-dirs` defaults to wikipedia + books). Training on it is always a deliberate decision.
- PDFs: `data/prepare_manual_pdfs.py` → `_pdf_extracted/` review queue + `manifest.jsonl`, never
  auto-promoted.

## Reviews and gates (Joel's standing requirement)
- Every non-trivial task keeps a gates file: one checkbox per outcome, with `CHECK:`/`EXPECT:`
  where a command can decide it, and evidence pasted.
- An independent reviewer (fresh-context agent or a different model) re-runs the checks and
  writes `.agents/reviews/<date>-<task>.md`.
- No self-certified "done".
