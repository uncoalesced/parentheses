# MANDATORY LICENSE & DATA GOVERNANCE POLICY — 100% APACHE 2.0

**Effective Immediately (2026-10-30):** All models, weights, tokenizers, training shards, SFT/DPO packs, and code under **Parentheses (`uncoalesced`)** are strictly governed under the **Apache License, Version 2.0**.

## 1. Whitelisted Permissive Data Licenses ONLY
Every pretraining shard, annealing pool, SFT dataset, and DPO preference pair MUST originate exclusively from commercially permissive licenses compatible with Apache 2.0:
- `Apache-2.0`
- `MIT`
- `BSD-2-Clause` / `BSD-3-Clause`
- `CC0-1.0` (Public Domain)
- `CC-BY-4.0` (Attribution only, commercial use permitted)
- `ODC-By` / `ODC-PDDL`
- First-party synthetic & epistemic abstention data authored by **uncoalesced** (`Apache-2.0`)

## 2. Strictly Prohibited Licenses (ZERO TOLERANCE)
Agents, scripts, and data pipelines must **NEVER** ingest, tokenize, pack, or train on any dataset or subset carrying non-commercial, share-alike copyleft, or restrictive terms, including:
- `CC BY-NC` (`2.0`, `3.0`, `4.0` — e.g., **Camel-AI**, **SciQ**, or non-commercial Karya/Indic subsets)
- `CC BY-NC-SA` / `CC BY-NC-ND` / `CC BY-ND`
- `GPL` / `AGPL` (for code/weights)
- Custom research-only or non-commercial EULAs (e.g., Llama/Gemma/ proprietary API outputs with non-compete restrictions)
- Any dataset with an `"unknown"`, `"unverified"`, or missing license field in its manifest

## 3. Mandatory Pipeline & Agent Enforcement Gate
1. Any dataset builder or packer script (`scripts/*.py`) MUST validate source license tags before writing `.bin` or `.parquet` shards and embed `"license_compliance": "Apache-2.0-Permissive-Only"` in `manifest.json`.
2. Any training launcher (`train.py` / `launch_*.ps1`) or observer subagent MUST abort immediately if a target data directory lacks a verified Apache-2.0-compliant `manifest.json`.

---
*Engineered by uncoalesced*
