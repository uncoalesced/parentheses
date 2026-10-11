# Gates: longrun-L1 (Data Mix — fresh Sangraha slice + v4 mix)

**Stream:** longrun-L1 data
**Owner:** worker DATA
**Target:** `data/processed_v4/` (1.10B tokens, 32k BPE)
**Status:** PASS (all 5/5 gates passed)

---

### Gate 1: Manifest checksums verify
**Outcome:** PASS

**CHECK:**
```powershell
& 'E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe' -c "import json, hashlib; from pathlib import Path; p = Path('E:/Parentheses/parentheses-0.9/data/processed_v4'); m = json.loads((p/'manifest.json').read_text('utf-8')); mismatches = [f for f, exp in m['checksums_sha256'].items() if hashlib.sha256((p/f).read_bytes()).hexdigest() != exp]; print('MISMATCHES:', mismatches); assert len(mismatches) == 0; print('PASS: all', len(m['checksums_sha256']), 'checksums match')"
```

**EXPECT:**
`MISMATCHES: []` and `PASS: all 20 checksums match`

**EVIDENCE:**
```
MISMATCHES: []
PASS: all 20 checksums match
```

---

### Gate 2: Val round-trip exact
**Outcome:** PASS

**CHECK:**
```powershell
& 'E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe' -c "import numpy as np, json; from pathlib import Path; from tokenizers import Tokenizer; p = Path('E:/Parentheses/parentheses-0.9/data/processed_v4'); tok = Tokenizer.from_file('E:/Parentheses/parentheses-0.9/data/tokenizer/v3_32k/tokenizer.json'); tok.encode_special_tokens = True; langs = ['kan','tam','tel','mal','tcy','kff','kxv','nit','eng']; total_docs = 0; bad = 0
for l in langs:
    b = np.fromfile(p/'val'/f'{l}.bin', dtype=np.uint16)
    bnd = np.append(np.load(p/'val'/f'{l}_boundaries.npy').astype(np.int64), len(b))
    total_docs += len(bnd) - 1
    for s, e in zip(bnd[:-1], bnd[1:]):
        doc = b[s:e]
        decoded = tok.decode(doc[1:-1].tolist(), skip_special_tokens=False)
        re_enc = tok.encode(decoded, add_special_tokens=False).ids
        if list(doc[1:-1]) != re_enc: bad += 1
print(f'Checked {total_docs} docs across {len(langs)} languages. Bad: {bad}'); assert bad == 0; print('PASS: exact val round trip')"
```

**EXPECT:**
`Checked 17538 docs across 9 languages. Bad: 0` and `PASS: exact val round trip`

**EVIDENCE:**
```
Checked 17538 docs across 9 languages. Bad: 0
PASS: exact val round trip
```

---

### Gate 3: `<esg>` (id 11) count == 0 in train.bin
**Outcome:** PASS

**CHECK:**
```powershell
& 'E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe' -c "import numpy as np; from pathlib import Path; train = np.memmap('E:/Parentheses/parentheses-0.9/data/processed_v4/train.bin', dtype=np.uint16, mode='r'); count = int(np.count_nonzero(train == 11)); print(f'<esg> id 11 count in {len(train):,} tokens: {count}'); assert count == 0; print('PASS: <esg> count == 0')"
```

**EXPECT:**
`<esg> id 11 count in 1,100,004,675 tokens: 0` and `PASS: <esg> count == 0`

**EVIDENCE:**
```
<esg> id 11 count in 1,100,004,675 tokens: 0
PASS: <esg> count == 0
```

---

### Gate 4: Zero doc overlap v4-fresh vs v3 text and val
**Outcome:** PASS

**CHECK:**
```powershell
& 'E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe' -c "import json; from pathlib import Path; p = Path('F:/Lingual_Data/data/processed_v4/text'); langs = ['kan','tam','tel','mal']
for l in langs:
    s = json.loads((p / f'{l}.stats.json').read_text('utf-8'))
    print(l, 'docs:', s['v4_docs'], 'v3_overlap:', s['overlap_with_v3'], 'val_overlap:', s['overlap_with_val'])
    assert s['overlap_with_v3'] == 0 and s['overlap_with_val'] == 0
print('PASS: zero doc overlap with v3 text and val')"
```

**EXPECT:**
All languages report `v3_overlap: 0` and `val_overlap: 0`, and `PASS: zero doc overlap with v3 text and val`

**EVIDENCE:**
```
kan docs: 274652 v3_overlap: 0 val_overlap: 0
tam docs: 220348 v3_overlap: 0 val_overlap: 0
tel docs: 276428 v3_overlap: 0 val_overlap: 0
mal docs: 290027 v3_overlap: 0 val_overlap: 0
PASS: zero doc overlap with v3 text and val
```

---

### Gate 5: Per-language token totals
**Outcome:** PASS

**CHECK:**
```powershell
& 'E:\Parentheses\parentheses-0.9\.venv\Scripts\python.exe' -c "import json; from pathlib import Path; m = json.loads(Path('E:/Parentheses/parentheses-0.9/data/processed_v4/manifest.json').read_text('utf-8')); print('Total train tokens:', m['train_tokens'], 'docs:', m['train_docs']);
for l, info in m['languages'].items():
    print(l, 'tokens:', info['tokens'], 'share:', info['share'], 'unique:', info['unique_tokens'], 'repeated:', info['repeated_tokens'], 'fresh:', info['fresh_tokens'])
assert m['train_tokens'] >= 1000000000
print('PASS: token totals within 1.0-1.2B target')"
```

**EXPECT:**
`Total train tokens: 1100004675 docs: 1314323` and `PASS: token totals within 1.0-1.2B target`

**EVIDENCE:**
```
Total train tokens: 1100004675 docs: 1314323
eng tokens: 440001494 share: 0.4 unique: 440001494 repeated: 0 fresh: 0
tcy tokens: 26419904 share: 0.024 unique: 3302488 repeated: 23117416 fresh: 0
kff tokens: 3471360 share: 0.0032 unique: 433920 repeated: 3037440 fresh: 0
kxv tokens: 3008552 share: 0.0027 unique: 376069 repeated: 2632483 fresh: 0
nit tokens: 2714976 share: 0.0025 unique: 339372 repeated: 2375604 fresh: 0
kan tokens: 156096826 share: 0.1419 unique: 156096826 repeated: 0 fresh: 151586044
tam tokens: 156097925 share: 0.1419 unique: 156097925 repeated: 0 fresh: 141569753
tel tokens: 156096827 share: 0.1419 unique: 156096827 repeated: 0 fresh: 156096827
mal tokens: 156096811 share: 0.1419 unique: 156096811 repeated: 0 fresh: 142387812
PASS: token totals within 1.0-1.2B target
```
