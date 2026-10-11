# Gates: audit of Gemini's 2026-10-09 evening work (Claude Code, independent, 21:15–21:35)

Scope: everything in Gemini's log after the 20:20 update. Read-only except this file. **Result: 3 PASS, 3 FAIL. v5 must not be built as-is.**

- [x] G1. New raw-sourced folders have SOURCE.txt with a verbatim license on the pass list. **PASS (with caveat)**
  CHECK: list `data/raw/manual/raw-sourced/{indicqa,bespoke_stratos}`, read SOURCE.txt; HF card via hub API
  EXPECT: SOURCE.txt present; license Apache/MIT/CC0/CC-BY
  EVIDENCE: indicqa SOURCE.txt `license: cc-by-4.0` (HF card tag `license:cc-by-4.0`); bespoke_stratos SOURCE.txt `license: apache-2.0` (HF card `license:apache-2.0`, R1-distilled).
  CAVEAT: IndicQA is an eval benchmark (`task_ids:closed-domain-qa`). Training on it means IndicQA can never be reported as a clean score.
- [x] G2. Kannada_Literature license status. **FAIL**
  CHECK: `F:\Lingual_Data\Kannada_Literature` listing (63 files) + grep `KANNADA_LITERATURE_HARVEST_PLAN.md` for licen|copyright|public domain|rights
  EXPECT: no in-copyright author in the v5 mix
  EVIDENCE: the plan has zero license lines; the sources are archive.org scans. Indian copyright = life + 60 years from the year after death:
  - Kuvempu (d. 1994 → protected to 2054), 25 files ≈ 20 MB;
  - D.R. Bendre (d. 1981 → 2041), 18 files;
  - Masti (d. 1986 → 2046), 7 files;
  - G.S. Shivarudrappa (d. 2013 → 2073), 8 files, incl. *Sahitya Vaarshika* 1980–82 annuals;
  - Kailasam (d. 1946): **public domain** (2 files);
  - Janapada ×3: folk material, but edited compilations; compiler rights unclear.
  Roughly 95% of the bytes fail the license bar. `build_corpus_v5.py:48,106,254-265` feeds this folder into the Kannada slice first.
- [x] G3. `scripts/build_sft_tail.py` targets, tool mapping, text handling. **FAIL**
  CHECK: read `build_sft_tail.py:72-75,126-146,175-215,288-304`
  EXPECT: unmappable tool calls dropped; formatting preserved; capped upsample; no duplicate sources
  EVIDENCE:
  1. Lines 198-205: any function name without a keyword is renamed `web_search`, keeping its original arguments/result (e.g. `calculate_bmi` → `web_search(weight, height)`). That's fabricated tool semantics.
  2. Lines 72-75: `clean_text` = `re.sub(r"\s+", " ")` on every field, which destroys newlines, Python indentation in the Stratos code traces, and list/math layout.
  3. Lines 127-146: OASST2 + OASST1 both (OASST2 ⊇ OASST1 → duplicates), every assistant reply rather than rank 0, `synthetic` ignored.
  4. Lines 288-304: `tile_docs` has no repeat cap. Tool pool ≈1.5M tokens vs a 16.5M target ≈ 11×.
  5. No dedup of the tail against v3/v4 val (the brief required it).
  The budget itself is fine: dry-run sizes are hard-coded (276-280), and a full run uses `--budget 110M`.
- [x] G4. `scripts/build_corpus_v5.py` dedup + leak gate. **PARTIAL FAIL**
  CHECK: read `build_corpus_v5.py:21-27,175-265,378-398`
  EXPECT: pre-upsample dedup over all pools; full-file leak gate
  EVIDENCE:
  - Pre-upsample dedup is present for low-res (203-227, asserts the Tulu drop) and main pools (263, 275, 304, 331). PASS.
  - The leak gate is a random sample: `sample_size = min(5000 if args.dry_run else 50000, total_docs)` (387). The header (26-27) claims "every single document". FAIL.
  - Plus the G2 literature use.
- [x] G5. Repo edits in scope. **PASS**
  CHECK: files modified in the last 4 h (excluding checkpoints/.venv/out/processed); ATTRIBUTION.md search
  EXPECT: only scripts/, gates/, raw-sourced/ touched
  EVIDENCE: Gemini touched `scripts/build_corpus_v5.py`, `scripts/build_sft_tail.py`, `data/raw/manual/raw-sourced/{indicqa,bespoke_stratos}/`, and `data/raw/manual/raw-sourced/ATTRIBUTION.md` (entries for aya_dataset 217, indicqa 230, bespoke_stratos 245). No train.py/checkpoints/release edits. The in22_conv download was removed by Gemini itself.
- [x] G6. Factual claims corrected. **PASS (discrepancies listed)**
  CHECK: `out/queries_v1_base.jsonl` line count; my leak diagnosis (`out/val_leak_v4.log`)
  EXPECT: discrepancies listed
  EVIDENCE:
  1. Gemini: "497/532 queries … baseline". Actual: 612 queries run (532 Gemini + 80 low-res), done 21:05.
  2. Gemini first named the leak "doc 22", then "doc 3022". 3022 is the one confirmed bit-identical to val tcy doc 158, and it matches my finding (one 879-token Tulu doc ×8).
  3. Gemini: "cosine cooldown". It's linear (corrected earlier).

**G2 decision (Joel, 2026-10-09 ~21:40): Option A. Exclude the copyrighted literature from v5.** Only Kailasam (PD) may stay; Janapada only after a per-book source check. G3/G4 fixes were sent to Gemini; Claude re-checks them before the 120M launch.
