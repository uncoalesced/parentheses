# parentheses-1.1: entry point

parentheses is a from-scratch recurrent language model (selective linear attention, the Mamba-2 and RWKV family) for Kannada, Tamil, Telugu, Malayalam and English, trained on one 8 GB laptop GPU. 1.0 is released at 253,544,176 parameters (`slab_v4`). 1.1 grows it past 500M and adds image input, text-to-speech and speech input, with no attention anywhere.

This file is a map. `README.md` has the goals, `HISTORY.md` has how the project got here, `AGENTS.md` has the rules. Read `AGENTS.md` before changing anything.

## Where things are

| Need | Look here |
|---|---|
| Operating rules, public/private line, metrics rule | `AGENTS.md` |
| Goals and target architecture | `README.md` |
| Lineage of every version | `HISTORY.md` |
| How benchmark numbers are produced | `docs/BENCHMARK_SPEC.md` |
| Licence tiers of training data | `docs/LICENSE_TABLE.md` |
| Model definition, loader, lm-eval adapter | `python/parentheses/` |
| Tokenizer files (32k byte-level BPE; the pre-tokenizer keeps vowel signs with their letter, BPE can still split an akshara) | `python/parentheses/tokenizer_data/` |
| Inference runtime, server, CLI | `rust/` |
| Benchmark configs and raw results | `eval/` |
| Parity and equivalence tests | `tests/` |
| Leak checker run by the git hooks | `tools/check_public_tree.py` |
| Everything not published (training, data, ledgers, notes) | `private/`, gitignored |

## Hazards

- `private/` and `.claude/` must never be tracked. The checker fails the commit if they are.
- No personal names or local paths in tracked files.
- No metric in a public file without the maintainer's review. See `AGENTS.md`.
- Benchmark test sets are never training data.
- The 1.0 checkpoint must keep loading bit-compatibly; run the parity tests after touching model code.
