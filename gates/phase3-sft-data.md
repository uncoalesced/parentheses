# Gates: phase3-sft-data (`data/processed_sft_v1/`, BPE chat SFT for 41M-v4)

**Owner:** Claude Code, 2026-10-09 ~17:45. **Builder:** `data/prepare_sft_bpe.py` (self-test passes). **Status:** PASS 7/7, **pending independent review**.

Sources:
- `aya_dataset` (Apache-2.0, rev f9ea0458, SOURCE.txt written 2026-10-09): kan/tam/tel/mal/eng rows.
- `oasst2` (Apache-2.0): English, rank-0 path, non-synthetic.
- `voice-and-persona.md` (Joel's writing), ×2.

Train: 36,760 docs / 10,103,998 tokens, 79% supervised. Val: 666 docs / 189,296 tokens.

Shares (train tokens):

| lang | share |
|---|---|
| tam | 37.1% |
| eng | 33.7% |
| tel | 20.2% |
| mal | 7.9% (×3) |
| kan | 1.2% (×3) |

**Dravidian = 66%.**

**Known gap:** Kannada is 1.2% because Aya has only ~330 Kannada rows. This conflicts with the Kannada priority mandate, so more human-written Kannada chat data is needed for 120M (flagged to Gemini).

- [x] 1. Manifest checksums verify. EVIDENCE: `checksum mismatches []`
- [x] 2. Mask length == data length (train, val). EVIDENCE: `mask==len True` ×2
- [x] 3. `<esg>` (id 11) == 0. EVIDENCE: `esg 0` ×2
- [x] 4. Boundaries sorted; each doc starts with `<|user|>` (chat) or `<eng>` (persona). EVIDENCE: `sorted True first tokens ['<eng>', '<|user|>']`
- [x] 5. Mask is 0 on every `<|user|>`, `<|assistant|>` and lang tag. EVIDENCE: `mask on user/assistant/tag tokens: 0` ×2
- [x] 6. 30 random train docs decode to the template. EVIDENCE: `template match 30 /30`
- [x] 7. Dravidian share ≥ 60%. EVIDENCE: kan+tam+tel+mal = 66.4%

The CHECK command for 1–6 is the one-liner in this session's transcript (it reads `manifest.json`, the `.bin` files, `.mask.bin` and `_boundaries.npy`). A reviewer should re-run it.

Not done: dedup against `data/processed_v4/val/`. Aya/OASST are chat sources, while v4 val is Sangraha web text, so overlap is very unlikely. Not proven, though.
