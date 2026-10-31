#!/usr/bin/env python3
# Engineered by uncoalesced
"""
License Guard for Parentheses 1.0-240M Apache-2.0 / Permissive Manifest Verification.
"""

from __future__ import annotations
import json
from pathlib import Path


def verify_apache_manifest(dir_path: str | Path) -> bool:
    """Verifies that the target directory contains a valid Apache-2.0-Permissive-Only manifest."""
    p = Path(dir_path)
    if p.is_file() and p.name == "manifest.json":
        manifest_path = p
    else:
        manifest_path = p / "manifest.json"

    if not manifest_path.exists():
        print(f"[LicenseGuard] Warning/Notice: Missing manifest at {manifest_path}")
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        compliance = m.get("license_compliance", "")
        print(f"[LicenseGuard] Verifying {manifest_path}: license_compliance = {compliance!r}")
        return compliance == "Apache-2.0-Permissive-Only"
    except Exception as e:
        print(f"[LicenseGuard] Error reading manifest {manifest_path}: {e}")
        return False


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    ok = verify_apache_manifest(target)
    sys.exit(0 if ok else 1)
