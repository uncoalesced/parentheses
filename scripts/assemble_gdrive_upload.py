#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/assemble_gdrive_upload.py

Builds a self-contained archive directory at:
  E:\\Parentheses\\parentheses-0.9\\GOOGLE_DRIVE_UPLOAD_130M
Ready to upload directly to Google Drive (MyDrive/parenthese-v0.9-120M/130M archives/).
"""

from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "GOOGLE_DRIVE_UPLOAD_130M"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"[*] Assembling Google Drive Upload package at: {TARGET_DIR}")
    
    # Create directory structure
    weights_dir = TARGET_DIR / "model_weights"
    code_dir = TARGET_DIR / "core_code"
    scripts_dir = code_dir / "scripts"
    tok_dir = TARGET_DIR / "tokenizer"
    manifests_dir = TARGET_DIR / "data_manifests"
    telemetry_dir = TARGET_DIR / "telemetry_and_logs"

    for d in [weights_dir, code_dir, scripts_dir, tok_dir, manifests_dir, telemetry_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. MODEL WEIGHTS
    raw_pt = REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-swa3" / "step_0100_final.pt"
    st_bf16 = REPO_ROOT / "hf_staging_130m" / "model.safetensors"
    cfg_json = REPO_ROOT / "hf_staging_130m" / "config.json"

    checksums = []

    if raw_pt.exists():
        dest_pt = weights_dir / "step_0100_final_with_optimizer.pt"
        print(f"[*] Copying full raw checkpoint ({raw_pt.stat().st_size / 1e6:.1f} MB)...")
        shutil.copy2(raw_pt, dest_pt)
        print(f"[*] Computing SHA256 for raw checkpoint...")
        pt_sha = sha256_file(dest_pt)
        checksums.append(f"{pt_sha}  model_weights/{dest_pt.name}")

    if st_bf16.exists():
        dest_st = weights_dir / "model.bf16.safetensors"
        print(f"[*] Copying stripped bare BF16 safetensors ({st_bf16.stat().st_size / 1e6:.1f} MB)...")
        shutil.copy2(st_bf16, dest_st)
        st_sha = sha256_file(dest_st)
        checksums.append(f"{st_sha}  model_weights/{dest_st.name}")

    if cfg_json.exists():
        shutil.copy2(cfg_json, weights_dir / "config.json")

    # 2. CORE ARCHITECTURE CODE
    print("[*] Copying core model architecture code...")
    shutil.copytree(REPO_ROOT / "model", code_dir / "model", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (REPO_ROOT / "models").exists():
        shutil.copytree(REPO_ROOT / "models", code_dir / "models", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    
    if (REPO_ROOT / "train.py").exists():
        shutil.copy2(REPO_ROOT / "train.py", code_dir / "train.py")
    if (REPO_ROOT / "app.py").exists():
        shutil.copy2(REPO_ROOT / "app.py", code_dir / "app.py")

    # Critical scripts
    critical_scripts = [
        "grow_model_slab_v4.py",
        "train_dpo_130m_v3.py",
        "export_npu_130m_v3.py",
        "pack_sft_4096_v4.py",
        "generate_sft_abstention_v4.py",
        "eval_130m_audit.py",
        "upload_to_hf.py",
        "export_release_120m.py",
        "chat.py",
        "stage_hf_130m.py",
    ]
    for sname in critical_scripts:
        spath = REPO_ROOT / "scripts" / sname
        if spath.exists():
            shutil.copy2(spath, scripts_dir / sname)

    # 3. TOKENIZER
    src_tok = REPO_ROOT / "hf_staging_130m" / "tokenizer.json"
    src_tok_cfg = REPO_ROOT / "hf_staging_130m" / "tokenizer_config.json"
    if src_tok.exists():
        shutil.copy2(src_tok, tok_dir / "tokenizer.json")
        tok_sha = sha256_file(tok_dir / "tokenizer.json")
        checksums.append(f"{tok_sha}  tokenizer/tokenizer.json")
    if src_tok_cfg.exists():
        shutil.copy2(src_tok_cfg, tok_dir / "tokenizer_config.json")

    # 4. DATA MANIFESTS
    manifest_sources = [
        REPO_ROOT / "data" / "processed_sft_240m_v4" / "manifest.json",
        Path("I:/parentheses/data/processed_130m_v1/manifest.json"),
        Path("I:/parentheses/data/processed_v5/manifest.json"),
    ]
    for mf in manifest_sources:
        if mf.exists():
            shutil.copy2(mf, manifests_dir / f"{mf.parent.name}_{mf.name}")

    # 5. TELEMETRY & EVAL LOGS
    out_dir = REPO_ROOT / "out"
    if out_dir.exists():
        for jf in out_dir.glob("*.jsonl"):
            shutil.copy2(jf, telemetry_dir / jf.name)

    # 6. CHECKSUMS & README
    with open(TARGET_DIR / "CHECKSUMS.sha256", "w", encoding="utf-8") as f:
        f.write("\n".join(checksums) + "\n")

    readme_content = f"""# Parentheses v0.9-130M-v3 Archive Package

This self-contained folder is ready to upload directly to Google Drive:
`MyDrive/parenthese-v0.9-120M/130M archives/`

## Package Structure
- `model_weights/`:
  - `step_0100_final_with_optimizer.pt` (~1.7 GB): Complete golden PyTorch checkpoint including model weights, AdamW momentum & variance buffers, scheduler, and step 2099 counter.
  - `model.bf16.safetensors` (~331 MB): Stripped bare BF16 weights (0 optimizer states) ready for lightweight inference or showcase loading.
  - `config.json`: Architecture parameters (16 layers, 12 heads, d_model=768, BPE 32k).
- `core_code/`:
  - `model/` & `models/`: Sovereign State Space Duality (SSD) / selective linear attention backbone, Token-Shift, beta-gates, and Net2WiderNet growth modules.
  - `train.py`: Main parallel training engine with document boundary resets.
  - `scripts/`: Critical pipeline scripts including `grow_model_slab_v4.py`, `train_dpo_130m_v3.py`, `export_npu_130m_v3.py`, and `pack_sft_4096_v4.py`.
- `tokenizer/`: BPE 32k (`v3_32k`) `tokenizer.json` and `tokenizer_config.json`.
- `data_manifests/`: Exact curriculum distribution maps and 4,096 FFD packing manifests.
- `telemetry_and_logs/`: Full `.jsonl` evaluation logs, FreeThink traces, and DPO v2 benchmark results.
- `CHECKSUMS.sha256`: Cryptographic SHA-256 hashes for integrity verification.
"""
    with open(TARGET_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"[+] Archive assembly complete!")
    total_size = sum(f.stat().st_size for f in TARGET_DIR.rglob("*") if f.is_file())
    print(f"[+] Total Archive Size: {total_size / 1e9:.2f} GB ({total_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
