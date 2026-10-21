#!/usr/bin/env python3
"""Parentheses 0.9-120M: GitHub Release Packaging & Distribution Tool.

Prepares the distribution assets for a GitHub Release in `release/dist/`:
  1. Computes cryptographic `SHA256SUMS.txt` for all release files.
  2. Bundles the standalone code archive `parentheses-0.9-120M-code.zip`.
  3. Generates rich GitHub `RELEASE_NOTES_v0.9.md`.

Usage:
    # 1. Package release assets:
    python scripts/package_github_release.py

    # 2. Dry-run validation:
    python scripts/package_github_release.py --dry-run

    # 3. Headless self-test:
    python scripts/package_github_release.py --self-test
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def collect_candidate_files(source_dir: Path) -> list[tuple[Path, str, int, str]]:
    """Walks source directory and computes SHA256 checksums."""
    entries = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for file in files:
            if file.endswith((".pyc", ".tmp", ".log")) or file.startswith("."):
                continue
            full_path = Path(root) / file
            rel_path = full_path.relative_to(source_dir).as_posix()
            size = full_path.stat().st_size
            sha = compute_sha256(full_path)
            entries.append((full_path, rel_path, size, sha))
    entries.sort(key=lambda x: x[1])
    return entries


def create_code_zip(source_dir: Path, out_zip: Path) -> tuple[int, str]:
    """Bundles code, scripts, configs, and tokenizer into a clean portable zip."""
    code_files = [
        "app.py",
        "chat.py",
        "config.json",
        "configuration_parentheses.py",
        "LICENSE_DECISION.md",
        "load_example.py",
        "modeling_parentheses.py",
        "QUICKSTART.md",
        "README.md",
        "RELEASE_CHECKLIST.md",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for cf in code_files:
            p = source_dir / cf
            if p.exists():
                zf.write(p, arcname=f"parentheses-0.9-120M/{cf}")

        # Add model package
        model_dir = source_dir / "model"
        if model_dir.exists():
            for mp in model_dir.glob("*.py"):
                zf.write(mp, arcname=f"parentheses-0.9-120M/model/{mp.name}")

    size = out_zip.stat().st_size
    sha = compute_sha256(out_zip)
    return size, sha


def generate_release_notes(files: list[tuple[Path, str, int, str]], zip_info: tuple[int, str] | None = None) -> str:
    """Generates GitHub Release Notes markdown."""
    file_rows = []
    for _, rel_path, size, sha in files:
        size_str = f"{size / (1024 * 1024):.2f} MB" if size >= 100_000 else f"{size:,} B"
        file_rows.append(f"| `{rel_path}` | {size_str} | `{sha}` |")

    if zip_info:
        zip_size, zip_sha = zip_info
        zip_size_str = f"{zip_size / (1024 * 1024):.2f} MB"
        file_rows.insert(0, f"| `parentheses-0.9-120M-code.zip` | {zip_size_str} | `{zip_sha}` |")

    table_content = "\n".join(file_rows)

    return f"""# Parentheses 0.9-120M: Multilingual Recurrent State Space Model

We are excited to announce the release candidate of **Parentheses 0.9-120M**, a compact, from-scratch recurrent **Selective Linear Attention State Space Model (SSM)** engineered by Joel (`uncoalesced`).

Belonging to the **Mamba-2 / RWKV / State Space Duality (SSD)** family, Parentheses 0.9-120M delivers constant $\mathcal{{O}}(1)$ inference memory while achieving high language modeling compression across **9 languages** (8 Dravidian languages + English).

---

## ⚡ Architectural Mandate: Not a Transformer

> **Parentheses is NOT a Transformer / GPT model.**
> It does **NOT** use quadratic causal self-attention ($\mathcal{{O}}(T^2)$ Softmax attention matrices) or standard Transformer KV-caches.
> - **In Inference / Generation:** It runs with **constant $\mathcal{{O}}(1)$ memory** via linear-recurrent state updates:
>   $$S_t = \\alpha_t S_{{t-1}} + (k_t \\otimes v_t)$$
>   $$y_t = q_t S_t$$
> - **Zero Memory Growth:** Fixed state size ($32 \\text{{ layers}} \\times 8 \\text{{ heads}} \\times 64 \\times 64 \\approx 4.19\\text{{ MB}}$). Memory consumption never grows by a single byte, whether decoding token 10 or token 100,000.
> - **In Training (Parallel Form):** Chunked associative linear attention ($C=64$) with closed-form decay matrix in $\mathcal{{O}}(T)$ linear time.

---

## 📊 Model Specifications

- **Total Unique Parameters:** **119,702,272** (~119.7M)
- **Architecture:** 32 Layers, 8 Attention Heads, 512 Hidden Dimension, 64 Head Dimension.
- **Attention Type:** `selective_linear` with learned per-head dynamic decay gating.
- **Context Length:** 4,096 tokens with document isolation (`reset_mask`).
- **Vocabulary Size:** 32,768 (`v3_32k` Byte-Level BPE with Indic akshara preservation & 256-byte zero-UNK fallback).
- **Embeddings:** Tied (`tok_emb.weight == head.weight`), saving 16.8M parameters.
- **Training Budget:** 1.25 Billion tokens across 5 pretraining sessions + assistant-span masked SFT (step 9,636) on a single consumer NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM).

---

## 🌐 Supported Languages & Linguistic Quarantine

| Language | Code | Native Script | Role |
| :--- | :--- | :--- | :--- |
| **Kannada** | `kan` | ಕನ್ನಡ | South Dravidian Anchor |
| **Tamil** | `tam` | தமிழ் | South Dravidian Anchor |
| **Telugu** | `tel` | తెలుగు | South-Central Dravidian Anchor |
| **Malayalam** | `mal` | മലയാളം | South Dravidian Anchor |
| **Tulu** | `tcy` | ತುಳು (ಕನ್ನಡ ಲಿಪಿ) | South Dravidian Low-Resource |
| **Koya** | `kff` | కొoya (తెలుగు లిపి) | South-Central Dravidian Low-Resource |
| **Kuvi** | `kxv` | କୁଭି (ଓଡ଼ିଆ / తెలుగు లిపి) | South-Central Dravidian Low-Resource |
| **Kolami / Naikpod** | `nit` | కొలామి (తెలుగు లిపి) | Central Dravidian Low-Resource |
| **English** | `eng` | Latin | Global Anchor |

> **Aheri Gondi (`esg`) Strict Exclusion:** Due to third-party copyright restrictions (© Wycliffe, all rights reserved), Aheri Gondi data was strictly excluded from pretraining and SFT mixes (`esg_count = 0`). Token ID 11 (`<esg>`) is dormant and quarantined.

---

## 🧠 4 Inference Modes

1. **Thinking Mode (`--mode think`):** Structured reasoning inside `<think>...</think>` traces followed by final answer.
2. **Chat Mode (`--mode chat`):** Direct multi-turn conversational response.
3. **FreeThink Mode (`--mode freethink`):** Continuous, open-ended associative thought stream on statements in constant memory.
4. **Base Mode (`--mode base`):** Autoregressive raw continuation from a language tag.

---

## 📈 Bits Per Byte (BPB) Compression Density

Evaluated on held-out test sets across pretraining and SFT:

| Language | 41M Baseline | 120M S5 Pretrain (step 9536) | 120M SFT Final (step 9636) | Compression Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Kannada (`kan`)** | 0.522 | **0.4878** | 0.4889 | 10.02× |
| **Tamil (`tam`)** | 0.492 | **0.4674** | 0.4684 | 10.49× |
| **Telugu (`tel`)** | 0.623 | **0.5916** | 0.5928 | 8.55× |
| **Malayalam (`mal`)** | 0.444 | **0.4116** | 0.4126 | 10.70× |
| **Tulu (`tcy`)** | 0.652 | **0.6399** | 0.6429 | 8.23× |
| **Koya (`kff`)** | 0.525 | **0.5391** | 0.5440 | 7.40× |
| **Kuvi (`kxv`)** | 0.453 | **0.4560** | 0.4570 | 7.34× |
| **Kolami (`nit`)** | 0.718 | **0.7235** | 0.7353 | 7.18× |
| **English (`eng`)** | 1.653 | **1.3929** | 1.4027 | 2.54× |

---

## 📉 Supervised Fine-Tuning (SFT) Loss & Perplexity Progression

Evaluated on clean SFT validation split (`data/processed_sft_120m_v2/val.bin`, 53,271 tokens, 200 documents, 0% data leakage):

| Metric | Base Pretrain (step 9536) | SFT v1 (step 9636) | SFT v2 (step 9650) | SFT v2 (step 9700) | SFT v2 Final (step 9786) | Net Delta (Final vs. Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Masked SFT Loss** | 2.2179 | 2.1384 | 2.1001 | 2.0847 | *[Pending 9786]* | *[Pending]* |
| **Masked SFT Perplexity** | 9.19 | 8.49 | 8.17 | 8.04 | *[Pending 9786]* | *[Pending]* |
| **Unmasked CE Loss** | 2.4039 | 2.3683 | 2.3464 | 2.3413 | *[Pending 9786]* | *[Pending]* |
| **Unmasked CE Perplexity** | 11.07 | 10.68 | 10.45 | 10.40 | *[Pending 9786]* | *[Pending]* |
| **Supervised Token Fraction** | 60.5% | 60.5% | 60.5% | 60.5% | 60.5% | - |

### Key Per-Category Progressions
| Category | Base (9536) Loss (PPL) | SFT v1 (9636) Loss (PPL) | SFT v2 (9650) Loss (PPL) | SFT v2 (9700) Loss (PPL) | SFT v2 Final (9786) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`structured_think`** | 1.6851 (5.39) | 1.5667 (4.79) | 1.4883 (4.43) | 1.4650 (4.33) | *[Pending 9786]* |
| **`drav_chat`** | 3.0476 (21.06) | 3.0280 (20.66) | 3.0297 (20.69) | 3.0234 (20.56) | *[Pending 9786]* |
| **`eng_chat`** | 2.5318 (12.58) | 2.4967 (12.14) | 2.4751 (11.88) | 2.4661 (11.78) | *[Pending 9786]* |
| **`lowres_chat`** | 2.2911 (9.89) | 2.0959 (8.13) | 2.1865 (8.90) | 2.1738 (8.79) | *[Pending 9786]* |
| **`freethink_cont`** | 3.2924 (26.91) | 3.0940 (22.06) | 3.2212 (25.06) | 3.2148 (24.90) | *[Pending 9786]* |
| **`tool_hermes`** | 0.9369 (2.55) | 0.8963 (2.45) | 0.9456 (2.57) | 0.9175 (2.50) | *[Pending 9786]* |

---

## 🌊 FreeThink Mode Showcase & Empirical $\mathcal{{O}}(1)$ Recurrence Benchmark

Evaluated across 18 showcase runs covering all 9 supported languages in both open-ended continuation and structured `<think>` modes (`out/freethink_showcase_report.md`):

- **Native Script Fidelity:** **18/18 (100.0%)** (zero script drift across all 9 languages).
- **Control Token Leaks:** **0** (zero unclosed tags or rogue role tokens emitted).
- **Structured Tag Closures:** **18/18** (100% of `<think>` blocks cleanly terminate at `</think>`).
- **Mean Lexical Diversity ($D_2$):** **0.7846**.
- **Empirical Constant $\mathcal{{O}}(1)$ SSM Latency Ratio:** **1.010×** ($L_\\text{{last}} / L_\\text{{first}}$, strictly flat decoding latency verifying zero KV-cache memory creep).

| Language | Code | Script | Mode | Script OK | Speed (tok/s) | $\mathcal{{O}}(1)$ Latency Ratio | Distinct-2 | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **English** | `eng` | Latin | continuation | 100% | 8.9 | 0.898× | 1.0000 | **PASS** |
| **English** | `eng` | Latin | structured | 100% | 9.3 | 1.035× | 0.6522 | **PASS** |
| **Kannada** | `kan` | ಕನ್ನಡ | continuation | 100% | 8.5 | 1.010× | 0.7463 | **PASS** |
| **Kannada** | `kan` | ಕನ್ನಡ | structured | 100% | 8.9 | 0.962× | 0.8125 | **PASS** |
| **Tamil** | `tam` | தமிழ் | continuation | 100% | 7.5 | 1.123× | 0.4694 | **PASS** |
| **Tamil** | `tam` | தமிழ் | structured | 100% | 7.4 | 1.126× | 0.6071 | **PASS** |
| **Telugu** | `tel` | తెలుగు | continuation | 100% | 7.7 | 1.171× | 0.6970 | **PASS** |
| **Telugu** | `tel` | తెలుగు | structured | 100% | 6.4 | 1.070× | 0.7500 | **PASS** |
| **Malayalam** | `mal` | മലയാളം | continuation | 100% | 7.6 | 0.787× | 0.8451 | **PASS** |
| **Malayalam** | `mal` | മലയാളം | structured | 100% | 7.6 | 0.966× | 0.8727 | **PASS** |
| **Tulu** | `tcy` | ತುಳು | continuation | 100% | 7.5 | 1.181× | 0.7973 | **PASS** |
| **Tulu** | `tcy` | ತುಳು | structured | 100% | 8.7 | 0.868× | 0.9474 | **PASS** |
| **Koya** | `kff` | కొoya | continuation | 100% | 7.4 | 1.003× | 0.9020 | **PASS** |
| **Koya** | `kff` | కొoya | structured | 100% | 7.9 | 1.083× | 0.9535 | **PASS** |
| **Kuvi** | `kxv` | କୁଭି / తెలుగు | continuation | 100% | 8.2 | 1.028× | 0.8158 | **PASS** |
| **Kuvi** | `kxv` | କୁଭି / తెలుగు | structured | 100% | 8.2 | 0.962× | 0.9091 | **PASS** |
| **Naikpod** | `nit` | కొలామి | continuation | 100% | 10.6 | 0.963× | 0.8000 | **PASS** |
| **Naikpod** | `nit` | కొలామి | structured | 100% | 10.4 | 0.953× | 0.5455 | **PASS** |

---

## 🚀 Quickstart

### Method 1: Hugging Face `transformers`
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("uncoalesced/parentheses-0.9-120M", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("uncoalesced/parentheses-0.9-120M")

prompt = "<|user|><kan>ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?<|assistant|><think>"
inputs = tok(prompt, return_tensors="pt")
for token in model.stream(inputs.input_ids, max_new_tokens=200):
    print(tok.decode([int(token.item())]), end="", flush=True)
```

### Method 2: Standalone Local Web Studio (`app.py`)
```bash
python app.py --device cpu --port 8080
```

### Method 3: Terminal Chat with ANSI Streaming (`chat.py`)
```bash
python chat.py --lang kan --format think
```

---

## 📦 Distribution Assets & Cryptographic Checksums

| File | Size | SHA-256 Checksum |
| :--- | :--- | :--- |
{table_content}

---

## 📜 Sourcing & Licensing

Candidate weights and data sourcing details are documented in [`LICENSE_DECISION.md`](LICENSE_DECISION.md).
- Major Dravidian text: AI4Bharat Sangraha verified (CC BY 4.0).
- Wikimedia Tulu & eBible low-res: CC BY-SA 4.0.
- English: Project Gutenberg (Public Domain) & Proof-Pile-2.
- Session 5 distillation slices (1.6M tokens from NVIDIA API trial): Under Joel's review for final licensing designation.
"""


def main():
    parser = argparse.ArgumentParser(description="Package Parentheses 0.9-120M GitHub Release Assets")
    parser.add_argument("--source-dir", default="release/parentheses-0.9-120M-candidate",
                        help="Candidate release source directory")
    parser.add_argument("--out-dir", default="release/dist",
                        help="Output distribution directory for GitHub release assets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate assets without writing distribution bundle")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test on candidate release directory")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    source_path = Path(args.source_dir).resolve()
    out_path = Path(args.out_dir).resolve()

    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("PARENTHESES 0.9-120M GITHUB RELEASE PACKAGING TOOL")
    print(f"Source Directory : {source_path}")
    print(f"Distribution Dir : {out_path}")
    print(f"Mode             : {'DRY-RUN' if args.dry_run else 'BUILD'}")
    print("=" * 80)

    # 1. Collect and checksum all source files
    print("\nScanning candidate files and calculating SHA-256...")
    files = collect_candidate_files(source_path)

    # 2. Check essential files
    required = ["model.safetensors", "model.bf16.safetensors", "config.json", "README.md", "QUICKSTART.md"]
    found = {f[1] for f in files}
    missing = [r for r in required if r not in found]
    if missing:
        print(f"\n[ERROR] Missing required candidate files: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanned {len(files)} files successfully.")

    if args.dry_run:
        print("\n[DRY-RUN] Manifest:")
        for _, rel_path, size, sha in files:
            print(f"  {size:10,d} B  {sha}  {rel_path}")
        print("\n[DRY-RUN COMPLETE] All distribution requirements satisfied.")
        return

    # 3. Create distribution directory
    out_path.mkdir(parents=True, exist_ok=True)

    # 4. Write SHA256SUMS.txt
    sha_file = out_path / "SHA256SUMS.txt"
    print(f"\nWriting {sha_file.relative_to(Path.cwd()) if sha_file.is_relative_to(Path.cwd()) else sha_file}...")
    with open(sha_file, "w", encoding="utf-8") as f:
        for _, rel_path, _, sha in files:
            f.write(f"{sha}  {rel_path}\n")

    # 5. Build code zip
    zip_file = out_path / "parentheses-0.9-120M-code.zip"
    print(f"Building portable code archive: {zip_file.name}...")
    zip_size, zip_sha = create_code_zip(source_path, zip_file)
    print(f"  -> {zip_size / 1024:.1f} KB, SHA256: {zip_sha[:16]}...{zip_sha[-8:]}")

    # Append zip checksum to SHA256SUMS.txt
    with open(sha_file, "a", encoding="utf-8") as f:
        f.write(f"{zip_sha}  {zip_file.name}\n")

    # 6. Generate RELEASE_NOTES_v0.9.md
    notes_file = out_path / "RELEASE_NOTES_v0.9.md"
    print(f"Generating GitHub Release Notes: {notes_file.name}...")
    notes_text = generate_release_notes(files, zip_info=(zip_size, zip_sha))
    notes_file.write_text(notes_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("DISTRIBUTION PACKAGE READY IN release/dist/:")
    for item in out_path.iterdir():
        sz = item.stat().st_size
        sz_str = f"{sz / 1024:.1f} KB" if sz < 1024 * 1024 else f"{sz / (1024*1024):.2f} MB"
        print(f"  - {item.name:32s}  {sz_str:>10s}")
    print("=" * 80)


def _self_test():
    """Headless self-test verifying release notes generation and checksum output."""
    print("[self-test] Running GitHub packaging self-test...")
    repo_root = Path(__file__).resolve().parent.parent
    candidate_dir = repo_root / "release" / "parentheses-0.9-120M-candidate"
    files = collect_candidate_files(candidate_dir)
    assert len(files) >= 10, f"Expected >= 10 candidate files, got {len(files)}"
    notes = generate_release_notes(files, zip_info=(12345, "mock_sha256_hash"))
    assert "Parentheses 0.9-120M" in notes
    assert "Architectural Mandate: Not a Transformer" in notes
    assert "SHA-256 Checksum" in notes
    print("[self-test] package_github_release ok")


if __name__ == "__main__":
    main()
