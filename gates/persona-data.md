# Gates: Persona Data (SWARM 2026-10-08)

Owner: worker PERSONA
Task: Download OpenAssistant OASST1 and OASST2 (Apache-2.0) and Anthropic hh-rlhf (MIT) as raw files into `data/raw/manual/raw-sourced/`, verify licenses verbatim, generate `SOURCE.txt` with SHA256 checksums, place prominent `DO_NOT_TRAIN.txt` in `hh-rlhf`, update `ATTRIBUTION.md`, update stale line in `data/raw/manual/README.md`, update status in `.agents/open/personality-data-brief.md`.

---

### Gate 1: Directory Staging & Siblings Location
- **CHECK:**
```powershell
python -c "import os; root = 'data/raw/manual/raw-sourced'; print(sorted(os.listdir(root)))"
```
- **EXPECT:**
`oasst1`, `oasst2`, and `hh-rlhf` directories exist as immediate subdirectories of `data/raw/manual/raw-sourced/`, alongside existing raw-sourced siblings.
- **EVIDENCE:**
```
['ATTRIBUTION.md', 'common-pile-arxiv-filtered', 'common-pile-stackexchange-cs-hw-robotics', 'gutenberg-literature-swarm', 'hermes-function-calling-v1', 'hh-rlhf', 'oasst1', 'oasst2', 'openstax-textbooks', 'project-gutenberg', 'proof-pile-2-open-web-math', 'steam-manuals']
```

---

### Gate 2: License Verification (Apache-2.0 & MIT)
- **CHECK:**
```powershell
python -c "from huggingface_hub import HfApi; api = HfApi(); print('oasst1:', api.dataset_info('OpenAssistant/oasst1').card_data.license); print('oasst2:', api.dataset_info('OpenAssistant/oasst2').card_data.license); print('hh-rlhf:', api.dataset_info('Anthropic/hh-rlhf').card_data.license)"
```
- **EXPECT:**
`oasst1: apache-2.0`, `oasst2: apache-2.0`, `hh-rlhf: mit`. Verbatim text of licenses captured in `LICENSE` and `SOURCE.txt` in each directory.
- **EVIDENCE:**
```
oasst1: apache-2.0
oasst2: apache-2.0
hh-rlhf: mit
```
Verbatim Apache-2.0 captured in `data/raw/manual/raw-sourced/oasst1/LICENSE` and `data/raw/manual/raw-sourced/oasst2/LICENSE`.
Verbatim Anthropic MIT license captured from `https://github.com/anthropics/hh-rlhf/raw/master/LICENSE` in `data/raw/manual/raw-sourced/hh-rlhf/LICENSE`.

---

### Gate 3: Integrity and SHA-256 Hashes (Re-hash Verification)
- **CHECK:**
```powershell
.venv\Scripts\python.exe scripts/verify_persona_data.py
```
- **EXPECT:**
All files across `oasst1` (10 files), `oasst2` (10 files), and `hh-rlhf` (12 files) exist, sizes match, and re-computed SHA-256 checksums match `SHA256SUMS.txt` with 0 errors.
- **EVIDENCE:**
```
======================================================================
PERSONA DATA VERIFICATION AUDIT
======================================================================

--- Checking oasst1 at E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\oasst1 ---
  OK: 2023-04-12_oasst_all.messages.jsonl.gz             (53,622,827 bytes)
  OK: 2023-04-12_oasst_all.trees.jsonl.gz                (53,625,064 bytes)
  OK: 2023-04-12_oasst_prompts.messages.jsonl.gz         (12,783,960 bytes)
  OK: 2023-04-12_oasst_ready.messages.jsonl.gz           (34,196,309 bytes)
  OK: 2023-04-12_oasst_ready.trees.jsonl.gz              (34,145,252 bytes)
  OK: 2023-04-12_oasst_spam.messages.jsonl.gz            (1,513,962 bytes)
  OK: LICENSE                                            (11,357 bytes)
  OK: README.md                                          (10,168 bytes)
  OK: SOURCE.txt                                         (13,861 bytes)
  OK: SHA256SUMS.txt                                     (1,032 bytes)
  OK: data/train-00000-of-00001-b42a775f407cee45.parquet (39,516,251 bytes)
  OK: data/validation-00000-of-00001-134b8fd0c89408b6.parquet (2,080,179 bytes)
  Re-hashing against SHA256SUMS.txt ...
    MATCH: 2023-04-12_oasst_all.messages.jsonl.gz (2ff4aa8999c911ff...)
    MATCH: 2023-04-12_oasst_all.trees.jsonl.gz (15c9d7d0a8c9f95b...)
    MATCH: 2023-04-12_oasst_prompts.messages.jsonl.gz (621ccd86a6ef320c...)
    MATCH: 2023-04-12_oasst_ready.messages.jsonl.gz (286a6e9a5a413b32...)
    MATCH: 2023-04-12_oasst_ready.trees.jsonl.gz (2a9a8fd343e9b28e...)
    MATCH: 2023-04-12_oasst_spam.messages.jsonl.gz (3ed9e51a883844f9...)
    MATCH: LICENSE (c71d239df91726fc...)
    MATCH: README.md (68483ac2fcc2f3f7...)
    MATCH: data/train-00000-of-00001-b42a775f407cee45.parquet (bbfadf5ed1278ba2...)
    MATCH: data/validation-00000-of-00001-134b8fd0c89408b6.parquet (24002597bb13a7ed...)
  -> All 10 hashes in oasst1 verified perfectly.

--- Checking oasst2 at E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\oasst2 ---
  OK: 2023-11-05_oasst2_all.messages.jsonl.gz            (75,076,341 bytes)
  OK: 2023-11-05_oasst2_all.trees.jsonl.gz               (75,314,383 bytes)
  OK: 2023-11-05_oasst2_prompts.messages.jsonl.gz        (15,560,794 bytes)
  OK: 2023-11-05_oasst2_ready.messages.jsonl.gz          (54,301,900 bytes)
  OK: 2023-11-05_oasst2_ready.trees.jsonl.gz             (54,370,156 bytes)
  OK: 2023-11-05_oasst2_spam.messages.jsonl.gz           (7,088,737 bytes)
  OK: LICENSE                                            (10,316 bytes)
  OK: README.md                                          (10,606 bytes)
  OK: SOURCE.txt                                         (13,884 bytes)
  OK: SHA256SUMS.txt                                     (1,038 bytes)
  OK: data/train-00000-of-00001-88ba0162028a73fc.parquet (63,494,018 bytes)
  OK: data/validation-00000-of-00001-1deeef95c3248fe0.parquet (3,180,111 bytes)
  Re-hashing against SHA256SUMS.txt ...
    MATCH: 2023-11-05_oasst2_all.messages.jsonl.gz (820146830e786341...)
    MATCH: 2023-11-05_oasst2_all.trees.jsonl.gz (2cf3d24faf92ca95...)
    MATCH: 2023-11-05_oasst2_prompts.messages.jsonl.gz (087bb0de13a9491e...)
    MATCH: 2023-11-05_oasst2_ready.messages.jsonl.gz (a9f240c4c77aa137...)
    MATCH: 2023-11-05_oasst2_ready.trees.jsonl.gz (7a886a16ccfc1173...)
    MATCH: 2023-11-05_oasst2_spam.messages.jsonl.gz (a621805c353828c6...)
    MATCH: LICENSE (94b9224462ffdd31...)
    MATCH: README.md (52bd472cf3cae04a...)
    MATCH: data/train-00000-of-00001-88ba0162028a73fc.parquet (9638227b41b0a4fb...)
    MATCH: data/validation-00000-of-00001-1deeef95c3248fe0.parquet (fd46c63babc07569...)
  -> All 10 hashes in oasst2 verified perfectly.

--- Checking hh-rlhf at E:\Parentheses\parentheses-0.9\data\raw\manual\raw-sourced\hh-rlhf ---
  OK: DO_NOT_TRAIN.txt                                   (1,266 bytes)
  OK: LICENSE                                            (1,087 bytes)
  OK: README.md                                          (5,771 bytes)
  OK: SOURCE.txt                                         (5,520 bytes)
  OK: SHA256SUMS.txt                                     (1,136 bytes)
  OK: harmless-base/test.jsonl.gz                        (742,801 bytes)
  OK: harmless-base/train.jsonl.gz                       (13,197,306 bytes)
  OK: helpful-base/test.jsonl.gz                         (875,191 bytes)
  OK: helpful-base/train.jsonl.gz                        (16,200,131 bytes)
  OK: helpful-online/test.jsonl.gz                       (1,053,578 bytes)
  OK: helpful-online/train.jsonl.gz                      (20,127,228 bytes)
  OK: helpful-rejection-sampled/test.jsonl.gz            (1,361,231 bytes)
  OK: helpful-rejection-sampled/train.jsonl.gz           (25,697,148 bytes)
  OK: red-team-attempts/red_team_attempts.jsonl.gz       (15,483,307 bytes)
  Re-hashing against SHA256SUMS.txt ...
    MATCH: DO_NOT_TRAIN.txt (0a3331bcfc3efde7...)
    MATCH: LICENSE (f853562a952135b2...)
    MATCH: README.md (f75f40db0268656b...)
    MATCH: harmless-base/test.jsonl.gz (ebfaaed21162a4de...)
    MATCH: harmless-base/train.jsonl.gz (bf3e1ecd3db61bf2...)
    MATCH: helpful-base/test.jsonl.gz (8be3fc1a13b27901...)
    MATCH: helpful-base/train.jsonl.gz (518a5bf288456fc9...)
    MATCH: helpful-online/test.jsonl.gz (39a475cc7e0df972...)
    MATCH: helpful-online/train.jsonl.gz (0994d18b82083f5c...)
    MATCH: helpful-rejection-sampled/test.jsonl.gz (abc35dbc56015db1...)
    MATCH: helpful-rejection-sampled/train.jsonl.gz (0cea2d2ab2187624...)
    MATCH: red-team-attempts/red_team_attempts.jsonl.gz (4c7b0069991460f0...)
  -> All 12 hashes in hh-rlhf verified perfectly.
  OK: Prominent DO_NOT_TRAIN.txt verified with correct refusal/hedge rationale.
```

---

### Gate 4: Anthropic hh-rlhf Exclusion (DO_NOT_TRAIN.txt)
- **CHECK:**
```powershell
python -c "print(open('data/raw/manual/raw-sourced/hh-rlhf/DO_NOT_TRAIN.txt', encoding='utf-8').read())"
```
- **EXPECT:**
Prominent warning file documenting exclusion from all training due to harmlessness refusal/hedge bias, referencing `.agents/open/project-gaps-and-fixes.md` item 1 and `SWARM_2026-10-08.md`.
- **EVIDENCE:**
```
================================================================================
DO NOT TRAIN ON THIS DIRECTORY -- PENDING JOEL'S DECISION ON REFUSAL/HEDGE FILTER
================================================================================

This dataset (Anthropic hh-rlhf) is EXCLUDED FROM ALL TRAINING in Parentheses.

Reason (see .agents/open/project-gaps-and-fixes.md item 1 and SWARM_2026-10-08.md):
Parentheses is an explicitly uncensored model.
Anthropic hh-rlhf contains human preference pairs ("chosen" vs "rejected") for
helpfulness and harmlessness. A substantial fraction of "chosen" responses are
refusals or safety-hedged answers. Training directly on the "chosen" responses
would quietly bake refusal/censorship behavior into the model.

Before this dataset can enter any training stage (pretraining or SFT), a filter
must be designed and implemented in data/prepare_sft.py to identify and drop
(or invert) examples where "chosen" is a refusal or safety hedge.

This filter design is explicitly gated on Joel's decision.
UNTIL JOEL DECIDES AND CLEARS THE FILTER, NO FILE IN THIS DIRECTORY MAY BE
TOKENIZED, PACKED, OR FED INTO ANY TRAINING RUN.
================================================================================
```

---

### Gate 5: Attribution Rollup & Documentation Indexes
- **CHECK:**
```powershell
.venv\Scripts\python.exe scripts/build_attribution.py
```
- **EXPECT:**
`[done] 11 source(s) -> data/raw/manual/raw-sourced/ATTRIBUTION.md` with 0 missing sources, 0 missing licenses, and 0 problems.
- **EVIDENCE:**
```
[done] 11 source(s) -> data/raw/manual/raw-sourced/ATTRIBUTION.md
  common-pile-arxiv-filtered       Openly licensed author uploads, verified per-record in metadata.licens
  common-pile-stackexchange-cs-hw-robotics Stack Exchange CC-BY-SA, verified per-record in metadata.license:
  gutenberg-literature-swarm       Public domain (US, all titles) per Project Gutenberg's own
  hermes-function-calling-v1       Apache 2.0. Quoted verbatim from dataset card YAML frontmatter at
  hh-rlhf                          MIT License. Quoted verbatim from dataset card YAML frontmatter at
  oasst1                           Apache 2.0. Quoted verbatim from dataset card YAML frontmatter at
  oasst2                           Apache 2.0. Quoted verbatim from dataset card YAML frontmatter at
  openstax-textbooks               CC BY 4.0 (Creative Commons Attribution 4.0 International License).
  project-gutenberg                Public Domain. Quoted verbatim from release boilerplate in header and 
  proof-pile-2-open-web-math       ODC-By 1.0, quoted verbatim in PROOFPILE2_REPORT.md
  steam-manuals                    GNU General Public License v2.0 or later (GPL-2.0-or-later)
```

- **CHECK (stale line in README.md & STATUS line in brief):**
```powershell
python -c "r = open('data/raw/manual/README.md', encoding='utf-8').read(); b = open('.agents/open/personality-data-brief.md', encoding='utf-8').read(); print('README has stale line:', 'Does not exist yet' in r); print('Brief status line:', b.splitlines()[0])"
```
- **EXPECT:**
`README has stale line: False`, `Brief status line: > STATUS: COMPLETE (...)`.
- **EVIDENCE:**
```
README has stale line: False
Brief status line: > STATUS: COMPLETE (2026-10-08, worker PERSONA in SWARM_2026-10-08). Downloaded OpenAssistant OASST1 and OASST2 (verified Apache-2.0 verbatim) and Anthropic hh-rlhf (verified MIT verbatim) into data/raw/manual/raw-sourced/{oasst1,oasst2,hh-rlhf}. Each source directory includes SOURCE.txt (URL, verbatim license, pull date, file count, byte count, per-file SHA256 checksums), a LICENSE file, and SHA256SUMS.txt. Placed a prominent DO_NOT_TRAIN.txt in hh-rlhf documenting exclusion from all training until Joel decides the refusal/hedge filter design (project-gaps-and-fixes item 1). Rebuilt ATTRIBUTION.md (now indexing 11 sources) via scripts/build_attribution.py. Updated data/raw/manual/README.md. Gates verified in gates/persona-data.md. Report: out/swarm/persona_report.md.
```

---

### Gate 6: Environment, GPU, and Locking Safeguards
- **CHECK:**
Verify no GPU touched, no new packages installed in repo `.venv`, and no concurrent I/O collisions on F:.
- **EXPECT:**
GPU unutilized; `pip list` in repo `.venv` unaltered; all writes staged on E: drive directly where sibling sources live (F: left unthridded for worker DATA).
- **EVIDENCE:**
- Downloads ran purely via `huggingface_hub` (already present v1.30.0) and Python standard library on CPU.
- GPU lock was never requested; zero CUDA calls executed.
- F: drive was untouched, leaving USB HDD bandwidth entirely free for concurrent worker DATA.
