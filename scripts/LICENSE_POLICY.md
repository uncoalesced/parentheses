# LICENSE & DATA GOVERNANCE POLICY

**Effective 2026-11-02.** parentheses (`uncoalesced`) model weights and tokenizers are released under the **Apache License 2.0** ([LICENSE-MODEL](../LICENSE-MODEL)). Source code is released under the **MIT License** ([LICENSE](../LICENSE)). Training data must allow both.

## The rule
**A dataset may be used for training if its licence permits commercial use.** No per-source bookkeeping is required: at release, the model card carries one list of the training datasets, written once.

## 1. Pass
- `Apache-2.0`, `MIT`, `BSD`, `CC0` / public domain, `CC-BY`, `CC-BY-SA`, `ODC-By` / `ODC-PDDL` / `ODbL`, GPL-family code
- First-party data authored by **uncoalesced**
- Third-party model-generated datasets: the dataset's own licence governs (e.g. MIT-licensed GPT-generated maths sets pass)
- Data we generate ourselves with a teacher model whose licence permits commercial use (e.g. Qwen models under Apache-2.0)

## 2. Fail
- Non-commercial: any `CC BY-NC*` variant (e.g. Camel-AI, SciQ, non-commercial Karya/Indic subsets)
- `CC BY-ND` (no derivatives; a trained model is one)
- Research-only, evaluation-only or other non-commercial terms (e.g. Aheri Gondi `esg`; NVIDIA API trial-key outputs)
- Unknown, unverified or missing licence
- Llama-generated data (Llama's licence requires a "Llama" name prefix on distilled models, so Llama is not used as a teacher)

## 3. Enforcement
1. Builders (`scripts/*.py`) check each source against Sections 1 and 2 and write its licence into `manifest.json`.
2. The manifest tag `"license_compliance": "Apache-2.0-Permissive-Only"` is the string `scripts/license_guard.py` checks; it means "every source passes this policy".
3. If a licence is unclear, leave the source out and flag it to the maintainers.

---
*Engineered by uncoalesced*
