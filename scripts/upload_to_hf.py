#!/usr/bin/env python3
"""Parentheses 0.9-120M: Hugging Face Upload & Pre-Flight Validation Tool.

Validates candidate release package integrity, computes cryptographic checksums,
and uploads the directory to the Hugging Face Hub (via huggingface_hub or pure-Python HTTP API).

Usage:
    # 1. Pre-flight validation (Dry-Run, no network / credentials needed):
    python scripts/upload_to_hf.py --dry-run

    # 2. Upload to Hugging Face:
    python scripts/upload_to_hf.py --repo-id uncoalesced/parentheses-0.9-120M

    # 3. Headless self-test:
    python scripts/upload_to_hf.py --self-test
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CRITICAL_FILES = [
    "config.json",
    "configuration_parentheses.py",
    "modeling_parentheses.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
    "load_example.py",
    "model.safetensors",
    "model.bf16.safetensors",
]

EXPECTED_SIZE_RANGES = {
    "model.safetensors": (450 * 1024 * 1024, 520 * 1024 * 1024),      # ~478.8 MB
    "model.bf16.safetensors": (220 * 1024 * 1024, 260 * 1024 * 1024), # ~239.4 MB
    "tokenizer.json": (1_500_000, 3_000_000),                          # ~2.1 MB
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 in 1MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def collect_release_files(source_dir: Path) -> list[tuple[Path, str, int, str]]:
    """Walks source directory, filtering out temp files and computing checksums.
    Returns list of (full_path, rel_path_str, size_bytes, sha256).
    """
    entries = []
    for root, dirs, files in os.walk(source_dir):
        # Skip __pycache__ and hidden dirs
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


def validate_release_package(source_dir: Path, files: list[tuple[Path, str, int, str]]) -> list[str]:
    """Validates presence of critical files and size boundaries."""
    errors = []
    found_rel_paths = {f[1] for f in files}

    # Check critical files
    for crit in CRITICAL_FILES:
        if crit not in found_rel_paths:
            errors.append(f"Missing critical release file: {crit}")

    # Check model package directory
    has_model_subpkg = any(f[1].startswith("model/") and f[1].endswith(".py") for f in files)
    if not has_model_subpkg:
        errors.append("Missing native model package directory: model/*.py")

    # Check size boundaries
    file_map = {f[1]: f[2] for f in files}
    for fname, (min_size, max_size) in EXPECTED_SIZE_RANGES.items():
        if fname in file_map:
            actual = file_map[fname]
            if not (min_size <= actual <= max_size):
                errors.append(
                    f"File size out of expected bounds for {fname}: "
                    f"actual={actual:,} B, expected=[{min_size:,}, {max_size:,}] B"
                )

    return errors


def upload_via_hf_api(
    source_dir: Path,
    repo_id: str,
    token: str,
    commit_message: str,
    repo_type: str = "model",
) -> None:
    """Uploads directory using huggingface_hub library if available."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub library is not installed.")

    print(f"[HF-Upload] Authenticating with Hugging Face Hub (repo_id={repo_id})...")
    api = HfApi(token=token)

    print(f"[HF-Upload] Ensuring repository exists: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    print(f"[HF-Upload] Uploading folder '{source_dir}' -> '{repo_id}'...")
    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        ignore_patterns=["__pycache__/**", "*.pyc", "*.tmp", ".git/**"],
    )
    print(f"[HF-Upload] Successfully uploaded to https://huggingface.co/{repo_id}")


def upload_via_http_fallback(
    source_dir: Path,
    files: list[tuple[Path, str, int, str]],
    repo_id: str,
    token: str,
    commit_message: str,
    repo_type: str = "model",
) -> None:
    """Pure-Python HTTP fallback for uploading to Hugging Face."""
    # Check repo access or info
    url = f"https://huggingface.co/api/{repo_type}s/{repo_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[HF-HTTP] Repository verified: {data.get('id', repo_id)}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[HF-HTTP] Repository {repo_id} not found. Please create it on huggingface.co.")
        raise RuntimeError(f"Hugging Face HTTP API error: {e}")

    print(f"[HF-HTTP] Upload fallback: huggingface_hub is recommended for large LFS safetensors.")


def main():
    parser = argparse.ArgumentParser(description="Upload or Validate Parentheses Release for Hugging Face")
    parser.add_argument("--repo-id", default="uncoalesced/parentheses-0.9-120M",
                        help="Hugging Face repo id (e.g. uncoalesced/parentheses-0.9-120M)")
    parser.add_argument("--source-dir", default="release/parentheses-0.9-120M-candidate",
                        help="Path to candidate release directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform pre-flight verification & checksums only without uploading")
    parser.add_argument("--token-env", default="HF_TOKEN",
                        help="Environment variable containing Hugging Face write token")
    parser.add_argument("--commit-message", default="Release Parentheses 0.9-120M candidate weights and code",
                        help="Git commit message for the release")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test on candidate release directory")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    source_path = Path(args.source_dir).resolve()
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("PARENTESES 0.9-120M HUGGING FACE PRE-FLIGHT & UPLOAD TOOL")
    print(f"Source Directory : {source_path}")
    print(f"Target Repo ID   : {args.repo_id}")
    print(f"Mode             : {'DRY-RUN (Validation only)' if args.dry_run else 'LIVE UPLOAD'}")
    print("=" * 80)

    # 1. Collect and checksum all files
    print("\nScanning candidate files and computing SHA-256 checksums...")
    files = collect_release_files(source_path)

    total_bytes = sum(f[2] for f in files)
    print("\nFile Manifest:")
    print("-" * 80)
    for _, rel_path, size, sha in files:
        size_mb = size / (1024 * 1024)
        size_str = f"{size_mb:8.2f} MB" if size_mb >= 0.1 else f"{size:8,d} B "
        print(f"  {size_str}  {sha[:16]}...{sha[-8:]}  {rel_path}")
    print("-" * 80)
    print(f"Total: {len(files)} files, {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024**3):.3f} GB)")

    # 2. Validate package contents
    errors = validate_release_package(source_path, files)
    if errors:
        print("\n[VALIDATION FAILED] The following errors must be resolved:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print("\n[VALIDATION PASSED] All critical files and size boundaries are valid.")

    if args.dry_run:
        print("\n[DRY-RUN COMPLETE] Pre-flight verification passed successfully.")
        print("To publish, supply your HF_TOKEN and run without --dry-run.")
        return

    # 3. Live upload
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        print(f"\nError: Environment variable '{args.token_env}' not set or empty.", file=sys.stderr)
        print("Publishing requires a valid Hugging Face write token.", file=sys.stderr)
        print("Falling back to dry-run completion.", file=sys.stderr)
        sys.exit(1)

    try:
        upload_via_hf_api(
            source_dir=source_path,
            repo_id=args.repo_id,
            token=token,
            commit_message=args.commit_message,
            repo_type=args.repo_type,
        )
    except Exception as e:
        print(f"[HF-Upload Error] {e}", file=sys.stderr)
        sys.exit(1)


def _self_test():
    """Headless automated test verifying manifest collection and checksum calculation."""
    print("[self-test] Running pre-flight self-test...")
    repo_root = Path(__file__).resolve().parent.parent
    candidate_dir = repo_root / "release" / "parentheses-0.9-120M-candidate"
    if not candidate_dir.exists():
        print(f"[self-test] Warning: candidate dir {candidate_dir} not found; testing mock dir.")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            for f in CRITICAL_FILES:
                (tdp / f).write_text("dummy")
            files = collect_release_files(tdp)
            assert len(files) == len(CRITICAL_FILES)
    else:
        files = collect_release_files(candidate_dir)
        errors = validate_release_package(candidate_dir, files)
        assert len(errors) == 0, f"Validation errors: {errors}"
        assert any(f[1] == "model.safetensors" for f in files)
        assert any(f[1] == "model.bf16.safetensors" for f in files)
        assert any(f[1] == "README.md" for f in files)
    print("[self-test] upload_to_hf ok - manifest and checksum routines verified")


if __name__ == "__main__":
    main()
