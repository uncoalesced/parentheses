# Operating rules for parentheses-1.1

These rules apply to every contributor and every coding agent working in this repository.

## The public and private line

This repository publishes the model weights' companion code: the model definition, the inference runtime, the tokenizer, the evaluation configs and the documentation. Training code, data builders, data ledgers, checkpoints and internal notes live under `private/`, which is gitignored and never committed.

- Never `git add -f` anything under `private/` or `.claude/`.
- Never write a personal name, a home directory, a drive path or a private email address into a tracked file. Refer to "the maintainer".
- `tools/check_public_tree.py` runs before every commit and every push. If it fails, fix the file. Do not skip the hook.
- Checkpoints and datasets are referred to by SHA-256 and by public repository id, never by local path.

## Metrics

No benchmark number goes into the README, the model card, release notes or any other public file unless all of these hold:

1. It was produced under `docs/BENCHMARK_SPEC.md` by lm-evaluation-harness.
2. The raw results JSON is committed under `eval/results/`.
3. The maintainer has read that file and at least 20 scored outputs for the task.

Agents may run benchmarks and build tables. Agents may not write a metric into a public document, and may not write the paragraph that interprets one.

## Data

- Every source used for training gets a row in the private ledger before it is tokenized: repository id, revision, licence text as found, tier, hash, rows used and rejected.
- Tiers are defined in `docs/LICENSE_TABLE.md`. `noncommercial` and `training-blocked` sources are never used. `unlicensed` and `model-generated` sources need a recorded decision by the maintainer for each source.
- Benchmark test sets are evaluation-only, and are added to the near-duplicate blocklist that training shards are checked against.
- Dataset facts reported by an agent (row counts, licences, sizes) are re-checked against the source before they enter the ledger.

## Code

- Python for the model definition, training and evaluation adapter. Rust for the inference runtime, tokenizer and data hot paths.
- The 1.0 weights must keep loading. Any change to the model code has to pass the parity tests in `tests/` against the 1.0 reference logits.
- No attention anywhere: not in the backbone, the vision encoder or the audio codec.
- Speed work must not change outputs beyond the parity tolerance.
- Don't name a script after a standard-library module.

## Working habits

- Internal session notes are append-only. If an old entry turns out wrong, add a new one that says so.
- Don't tail live training logs. Read status files and finished logs.
- When a task is delegated to another agent, check its report against the actual files before relying on it.
- Force-pushes, branch deletions and anything else that rewrites the public repository need the maintainer's explicit go-ahead each time.
