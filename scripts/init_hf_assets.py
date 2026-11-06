# Engineered by uncoalesced
"""
Initialization and provisioning tool for Hugging Face assets under uncoalesced-org.
Provisions datasets and storage buckets on Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

DATASETS = [
    "uncoalesced-org/dravidian-pretrain-sample",
    "uncoalesced-org/indic-reasoning-eval",
    "uncoalesced-org/parentheses-tokenizer-vocab",
]

BUCKETS = [
    "uncoalesced-org/checkpoints",
    "uncoalesced-org/raw-scrapes-and-corpora",
    "uncoalesced-org/export-runtimes",
]


def resolve_token(arg_token: str | None = None) -> str | None:
    if arg_token:
        return arg_token.strip()
    cache_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token_path.exists():
        try:
            tok = cache_token_path.read_text(encoding="utf-8").strip()
            if tok:
                return tok
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


def create_datasets(api: HfApi) -> bool:
    print("[*] Initializing Hugging Face datasets...")
    all_ok = True
    for repo_id in DATASETS:
        try:
            url = api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            print(f"[+] Dataset verified/created: {repo_id} -> {url}")
        except Exception as e:
            print(f"[-] Failed to verify/create dataset {repo_id}: {e}")
            all_ok = False
    return all_ok


def create_buckets(api: HfApi) -> bool:
    print("[*] Initializing Hugging Face mutable storage buckets...")
    all_ok = True
    for bucket in BUCKETS:
        try:
            res = api.create_bucket(
                bucket_id=bucket,
                private=True,
                exist_ok=True,
            )
            print(f"[+] Bucket verified/created: {bucket} -> {res.url}")
        except Exception as e:
            print(f"[-] Failed to verify/create bucket {bucket}: {e}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Initialize HF datasets and buckets for uncoalesced-org")
    parser.add_argument("--token", help="Hugging Face User Access Token with write role")
    parser.add_argument("--datasets-only", action="store_true", help="Only provision datasets")
    parser.add_argument("--buckets-only", action="store_true", help="Only provision buckets")
    args = parser.parse_args()

    token = resolve_token(args.token)
    if not token:
        print("[!] HF token not found in arguments, cache (~/.cache/huggingface/token), or HF_TOKEN.")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        auth_info = user_info.get("auth", {})
        access_tok = auth_info.get("accessToken", {})
        role = access_tok.get("role", "unknown")
        name = access_tok.get("displayName", user_info.get("name", "unknown"))
        print(f"[*] Authenticated as: {user_info.get('name')} (token: {name}, role: {role})")
    except Exception as e:
        print(f"[!] Authentication check failed: {e}")
        sys.exit(1)

    ok = True
    if not args.buckets_only:
        if not create_datasets(api):
            ok = False

    if not args.datasets_only:
        if not create_buckets(api):
            ok = False

    if ok:
        print("[+] All Hugging Face assets provisioned and verified successfully.")
    else:
        print("[!] Provisioning encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
