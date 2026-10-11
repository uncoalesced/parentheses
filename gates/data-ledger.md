# Gate Verification: Data Ledger, Pruning Proposal & Model Stats Panel

**Stream:** `ledger` (Phase 1 Ledger + Phase 3 Deletion Proposal + Stats Panel)  
**Worker:** `LEDGER`  
**Date:** 2026-10-08  
**Status:** **ALL GATES PASSED**

---

### Gate 1: Candidate File Coverage (Completeness & Byte Accuracy)
- **CHECK:** `python scripts/verify_ledger.py`
- **EXPECT:** 100% of candidate files across `processed_dravidian_v1`, `processed_dravidian_lowres_v1`, `processed_all_sources_v1`, `processed_v3`, text JSONL, pools, and Sangraha parquet shards exist on disk with exact byte match.
- **EVIDENCE:**
```
Total ledger records: 222
Gate 1 (Completeness & Existence): PASSED (222/222 files exist with exact byte match)
```

---

### Gate 2: SHA-256 Checksum Re-Hash Verification
- **CHECK:** `python scripts/verify_ledger.py` (checks 10 randomly sampled non-deferred records)
- **EXPECT:** Recomputed SHA-256 on disk byte-for-byte matches ledger records. (F:-only files > 50 MB recorded as deferred with mtime; honest accounting).
- **EVIDENCE:**
```
Gate 2: Re-hashing 10 random records out of 195 non-deferred:
  PASS: [boundaries] eng_starts.npy            -> 7b62b86a35599c27...
  PASS: [train_bin ] train.bin                 -> ef56e340daced626...
  PASS: [manifest  ] manifest.json             -> d0fa80c6df46c0ad...
  PASS: [val       ] tam.bin                   -> 1d9b4dae35a4793b...
  PASS: [boundaries] nit_boundaries.npy        -> d8c6d70aba0790de...
  PASS: [boundaries] kan_boundaries.npy        -> fec423c4f8126be0...
  PASS: [boundaries] train_boundaries.npy      -> b5b3e9404f3c8c6e...
  PASS: [val       ] val.bin                   -> 729cc4cca0f06c4f...
  PASS: [boundaries] nit_boundaries.npy        -> 73e1894013329f5b...
  PASS: [val       ] val.bin                   -> 54adfb3c01e09578...
Gate 2 (Re-hash 10 random records): PASSED
```
*Note on Drive F Deferred Hashes:* 27 large F:-only files (> 50 MB) have size and mtime recorded, with `sha256: "deferred"` to prevent thrashing the USB HDD while streaming. They can be filled offline using:
`python -c "import json, hashlib, pathlib; p=pathlib.Path(r'F:\Lingual_Data\Trained-On\LEDGER\training_data_ledger_2026-10-07.jsonl'); recs=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]; [r.update({'sha256': hashlib.sha256(open(r['path'],'rb').read()).hexdigest()}) for r in recs if r.get('sha256')=='deferred']; p.write_text('\n'.join(json.dumps(r) for r in recs)+'\n', encoding='utf-8')"`

---

### Gate 3: Per-Run Token Accounting Match
- **CHECK:** `python scripts/verify_ledger.py`
- **EXPECT:** Calculated token volume per training run matches training run logs within ±1 step.
- **EVIDENCE:**
```
Gate 3: Per-run token totals:
  parentheses-0.9-41M: 1,059,454,976 tokens (8,083 steps x 131,072 tok/step, exact log match)
  dravidian-v1-selective: 1,638,400,000 tokens (100,000 steps x 16,384 tok/step, exact log match)
  dravidian-v2-replay: 1,638,400,000 tokens (100,000 steps x 16,384 tok/step, exact log match)
  dravidian-v3-lowres: 983,040,000 tokens (60,000 steps x 16,384 tok/step, exact log match)
Gate 3 (Token totals match logs): PASSED
```

---

### Gate 4: Parameter Count Formula Equivalence
- **CHECK:** `python -c "from model.backbone import Parentheses, ModelConfig; cfg = ModelConfig(vocab_size=32768, block_size=4096, n_layer=16, n_head=6, n_embd=384, attn_type='selective_linear', tie_embeddings=True); m = Parentheses(cfg); print(m.num_params())"`
- **EXPECT:** Exactly `40,944,096` parameters.
- **EVIDENCE:**
```
40944096
```

---

### Gate 5: Weights-Only Size Approximation
- **CHECK:** `python -c "from pathlib import Path; p = Path('release/parentheses-0.9-41M/model.safetensors'); sz = p.stat().st_size; exp = 40944096 * 4; print(f'actual={sz}, expected_approx={exp}, diff_bytes={sz - exp}')"`
- **EXPECT:** Actual export size matches `params * 4 bytes` (within safetensors header overhead < 20 KB).
- **EVIDENCE:**
```
actual=163790224, expected_approx=163776384, diff_bytes=13840
```

---

### Gate 6: `model_stats.py` Self-Test Execution
- **CHECK:** `python scripts/model_stats.py --self-test`
- **EXPECT:** Toy model parameter checks, disk statistics checks, tensor reduction checks, and formula verification pass with exit code 0.
- **EVIDENCE:**
```
[self-test] Running model_stats self-test...
  [self-test] Toy Parentheses model created: 41252 params
  [self-test] Disk stats ok
  [self-test] Tensor stats ok
  [self-test] 41M exact parameter formula check: 40,944,096 == 40,944,096
[self-test] model_stats ok
```

---

### Gate 7: `report_stats.py` Integration & Pipeline Pass
- **CHECK:** `python scripts/report_stats.py --self-test`
- **EXPECT:** Integration hook passes without error and regression tests exit 0.
- **EVIDENCE:**
```
[self-test] Running model_stats self-test...
  [self-test] Toy Parentheses model created: 41252 params
  [self-test] Disk stats ok
  [self-test] Tensor stats ok
  [self-test] 41M exact parameter formula check: 40,944,096 == 40,944,096
[self-test] model_stats ok
[self-test] report_stats ok
```

---

### Gate 8: Zero-Deletion Guardrail Verification
- **CHECK:** `python -c "from pathlib import Path; p = Path('checkpoints/parentheses-0.9-41M'); pts = list(p.glob('*.pt')); print(f'Total checkpoints preserved: {len(pts)}')"`
- **EXPECT:** Exactly 29 checkpoints intact. Zero deleted.
- **EVIDENCE:**
```
Total checkpoints preserved: 29
```
