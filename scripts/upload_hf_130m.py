#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/upload_hf_130m.py

Uploads the staged bare BF16 weights and model card in hf_staging_130m
to Hugging Face repo: uncoalesced/Parentheses-v0.9-130M-v3.
"""

from __future__ import annotations
import os
import sys
import winreg
from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = REPO_ROOT / "hf_staging_130m"
REPO_ID = "uncoalesced/Parentheses-v0.9-130M-v3"


def get_hf_token() -> str:
    # 1. Environment
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    # 2. Windows registry user environment
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        val, _ = winreg.QueryValueEx(key, "HF_TOKEN")
        if val:
            return val.strip()
    except Exception:
        pass
    # 3. huggingface_hub default cache
    from huggingface_hub import get_token
    token = get_token()
    if token:
        return token.strip()
    raise ValueError("HF_TOKEN not found in environment, registry, or cache.")


def main():
    token = get_hf_token()
    api = HfApi(token=token)

    user_info = api.whoami()
    print(f"[*] Authenticated as: {user_info.get('name', 'unknown')} ({user_info.get('type', 'user')})")

    print(f"[*] Creating repository if it doesn't exist: {REPO_ID} (type=model, private=False)...")
    repo_url = api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
    print(f"[+] Repo ready: {repo_url}")

    print(f"[*] Uploading folder {STAGING_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Release Parentheses v0.9-130M-v3 showcase weights (BF16)",
    )
    print(f"[+] SUCCESS! Release is live at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
