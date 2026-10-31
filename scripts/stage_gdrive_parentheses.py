#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/stage_gdrive_parentheses.py

Creates and populates ./gdrive_staging_parentheses with:
- /checkpoints_raw/  : raw checkpoint reference / metadata / checksums
- /data_manifests/   : dataset mixture configs, packing manifests, tokenization configs
- /deferred_shards/  : (empty / marker: SSD headroom is 80.56 GB > 50 GB)
- /telemetry_130m/   : training logs, JSONL eval dumps, and telemetry
"""

from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_ROOT = REPO_ROOT / "gdrive_staging_parentheses"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"[*] Initializing Google Drive staging at: {STAGING_ROOT}")
    ckpts_dir = STAGING_ROOT / "checkpoints_raw"
    manifests_dir = STAGING_ROOT / "data_manifests"
    deferred_dir = STAGING_ROOT / "deferred_shards"
    telemetry_dir = STAGING_ROOT / "telemetry_130m"

    for d in [ckpts_dir, manifests_dir, deferred_dir, telemetry_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint raw info & archive
    raw_ckpt = REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-swa3" / "step_0100_final.pt"
    if raw_ckpt.exists():
        print(f"[*] Calculating SHA256 for {raw_ckpt.name} ({raw_ckpt.stat().st_size / 1e6:.2f} MB)...")
        c_hash = sha256_file(raw_ckpt)
        with open(ckpts_dir / "checksums.sha256", "w", encoding="utf-8") as f:
            f.write(f"{c_hash}  {raw_ckpt.name}\n")
        
        info = {
            "source_path": str(raw_ckpt),
            "size_bytes": raw_ckpt.stat().st_size,
            "sha256": c_hash,
            "architecture": "Parentheses 0.9-130M-v3 (slab_v3)",
            "step": 2099,
            "optimizer_state_included": True,
            "notes": "Full golden release checkpoint with AdamW buffers, TokenShift mu-mix, beta write gates, V0 highway."
        }
        with open(ckpts_dir / "checkpoint_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"[+] Staged raw checkpoint metadata and checksums.")

    # 2. Data manifests
    manifest_sources = [
        REPO_ROOT / "data" / "processed_sft_240m_v4" / "manifest.json",
        Path("I:/parentheses/data/processed_130m_v1/manifest.json"),
        Path("I:/parentheses/data/processed_v5/manifest.json"),
        REPO_ROOT / "data" / "tokenizer" / "tokenizer_config.json",
    ]
    for mf in manifest_sources:
        if mf.exists():
            dest = manifests_dir / f"{mf.parent.name}_{mf.name}"
            shutil.copy(mf, dest)
            print(f"[+] Staged data manifest: {dest.name}")

    # 3. Deferred shards note
    with open(deferred_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# Deferred Shards Staging\n\nNVMe E: has 80.56 GB free (> 50 GB threshold).\n"
                "All historical and 240M pretraining shards are safely hosted on I:\\parentheses\\data\\\n"
                "(processed_130m_v1: 2.84 GB, processed_v5: 2.13 GB).\n"
                "No local shards need to be quarantined.\n")

    # 4. Telemetry & Logs
    out_dir = REPO_ROOT / "out"
    jsonl_count = 0
    if out_dir.exists():
        for jf in out_dir.glob("*.jsonl"):
            shutil.copy(jf, telemetry_dir / jf.name)
            jsonl_count += 1
    print(f"[+] Staged {jsonl_count} telemetry and evaluation .jsonl logs.")

    print(f"[+] Google Drive staging complete at {STAGING_ROOT}")


if __name__ == "__main__":
    main()
