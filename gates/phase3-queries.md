# Gates: phase3-queries (Phase 3 Eval Query Set — Gemini)

**Stream:** Phase 3 Evaluation Suite
**Owner:** Gemini (Antigravity)
**Target:** `eval/queries_v1_gemini.jsonl` (532 queries, UTF-8 JSONL)
**Brief:** `.agents/open/GEMINI_PHASE3_QUERIES.md`
**Status:** PASS (all 6/6 gates passed)

---

### Gate 1: Validator Check (`eval_queries.py --validate`)
**Outcome:** PASS

**CHECK:**
```powershell
$env:PYTHONIOENCODING='utf-8'; $env:OMP_NUM_THREADS='1'
.venv\Scripts\python.exe scripts\eval_queries.py --validate eval\queries_v1_gemini.jsonl
```

**EXPECT:**
`[ok] eval\queries_v1_gemini.jsonl valid`

**EVIDENCE:**
```
[ok] eval\queries_v1_gemini.jsonl valid
```

---

### Gate 2: Quota & Language Distribution Check
**Outcome:** PASS

**CHECK:**
```powershell
.venv\Scripts\python.exe -c "import json,collections;c=collections.Counter((q['bucket'],q['lang']) for q in map(json.loads,open('eval/queries_v1_gemini.jsonl',encoding='utf-8')));[print(k,v) for k,v in sorted(c.items())];print(sum(c.values()))"
```

**EXPECT:**
532 total queries matching the quotas in §3:
```
('code', 'eng') 26
('english', 'eng') 170
('freethink', 'eng') 16
('freethink', 'kan') 4
('freethink', 'mal') 2
('freethink', 'tam') 2
('freethink', 'tel') 2
('instruction', 'kan') 25
('instruction', 'mal') 20
('instruction', 'tam') 20
('instruction', 'tel') 20
('lang_gen', 'kan') 25
('lang_gen', 'mal') 20
('lang_gen', 'tam') 20
('lang_gen', 'tel') 20
('probe_hin', 'hin') 10
('reasoning', 'eng') 60
('reasoning', 'kan') 6
('reasoning', 'mal') 4
('reasoning', 'tam') 4
('reasoning', 'tel') 4
('tool', 'eng') 40
('tool', 'kan') 12
532
```
*Note on total:* The sub-items in the §3 specification table sum to 492 (25+25+60+60+130+78+52+26+26+10 = 492). Because `scripts/eval_queries.py` strictly requires $\ge 500$ queries (`if len(seen) < 500: errs.append(...)`), and the deliverable specifically mandates 532 lines, English General Knowledge was expanded from 60 to 100 items (+40 items, bringing English to 170), preserving all Dravidian language quotas while hitting the exact 532 target.

**EVIDENCE:**
```
('code', 'eng') 26
('english', 'eng') 170
('freethink', 'eng') 16
('freethink', 'kan') 4
('freethink', 'mal') 2
('freethink', 'tam') 2
('freethink', 'tel') 2
('instruction', 'kan') 25
('instruction', 'mal') 20
('instruction', 'tam') 20
('instruction', 'tel') 20
('lang_gen', 'kan') 25
('lang_gen', 'mal') 20
('lang_gen', 'tam') 20
('lang_gen', 'tel') 20
('probe_hin', 'hin') 10
('reasoning', 'eng') 60
('reasoning', 'kan') 6
('reasoning', 'mal') 4
('reasoning', 'tam') 4
('reasoning', 'tel') 4
('tool', 'eng') 40
('tool', 'kan') 12
532
```

---

### Gate 3: Mandatory Kannada Items (§4)
**Outcome:** PASS

**CHECK:**
```powershell
.venv\Scripts\python.exe -c "import json; qs = {q['id']: q for q in map(json.loads, open('eval/queries_v1_gemini.jsonl', encoding='utf-8'))}; assert 'Answer in Kannada: What did you eat today?' in qs['kan-ins-001']['prompt']; assert 'Bengaluru traffic' in qs['kan-ins-002']['prompt']; assert 'Cubbon Park' in qs['kan-ins-003']['prompt']; assert qs['kan-ins-004']['reference'] == 'Tulu'; assert qs['kan-ins-005']['reference'] == 'देव बरें करूं ತುಮ್ಕಾಂ सर्वांक'; print('PASS: all 5 mandatory §4 prompts verified')"
```

**EXPECT:**
`PASS: all 5 mandatory §4 prompts verified`

**EVIDENCE:**
```
PASS: all 5 mandatory §4 prompts verified
```

---

### Gate 4: Reasoning References Verification
**Outcome:** PASS

**CHECK:**
```powershell
.venv\Scripts\python.exe -c "import json; rsn = [q for q in map(json.loads, open('eval/queries_v1_gemini.jsonl', encoding='utf-8')) if q['bucket'] == 'reasoning']; assert len(rsn) == 78; bad = [q['id'] for q in rsn if not q.get('reference')]; assert len(bad) == 0; print('PASS: all 78 reasoning queries have non-empty gold references')"
```

**EXPECT:**
`PASS: all 78 reasoning queries have non-empty gold references`

**EVIDENCE:**
```
PASS: all 78 reasoning queries have non-empty gold references
```

---

### Gate 5: Tool Probes Annotation
**Outcome:** PASS

**CHECK:**
```powershell
.venv\Scripts\python.exe -c "import json; tools = [q for q in map(json.loads, open('eval/queries_v1_gemini.jsonl', encoding='utf-8')) if q['bucket'] == 'tool']; assert len(tools) == 52; valid = {'web_search', 'fetch_url', 'read_file', 'python', 'none'}; bad = [q['id'] for q in tools if not any(t in q.get('notes', '') for t in valid)]; assert len(bad) == 0; print('PASS: all 52 tool queries specify valid tool in notes')"
```

**EXPECT:**
`PASS: all 52 tool queries specify valid tool in notes`

**EVIDENCE:**
```
PASS: all 52 tool queries specify valid tool in notes
```

---

### Gate 6: FreeThink Statements & Tokens
**Outcome:** PASS

**CHECK:**
```powershell
.venv\Scripts\python.exe -c "import json; fth = [q for q in map(json.loads, open('eval/queries_v1_gemini.jsonl', encoding='utf-8')) if q['bucket'] == 'freethink']; assert len(fth) == 26; bad_tok = [q['id'] for q in fth if not (1024 <= q.get('max_new_tokens', 0) <= 2048)]; bad_q = [q['id'] for q in fth if '?' in q['prompt']]; assert len(bad_tok) == 0; assert len(bad_q) == 0; print('PASS: all 26 freethink items are statements with max_new_tokens in 1024..2048')"
```

**EXPECT:**
`PASS: all 26 freethink items are statements with max_new_tokens in 1024..2048`

**EVIDENCE:**
```
PASS: all 26 freethink items are statements with max_new_tokens in 1024..2048
```
