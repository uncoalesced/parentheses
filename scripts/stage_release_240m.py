#!/usr/bin/env python3
# Engineered by uncoalesced
"""Stage the parentheses-1.0-240M release for upload. Uploads nothing.

Starting from the exported bundle in release/parentheses-1.0-240M-tanzanite/,
this script:

  1. Syncs tracked sources into the bundle: the AutoModel files from hf/, the
     model card, the docs figures, and the auto_map entry in config.json.
  2. Rewrites the bundle MANIFEST_sha256.tsv from freshly computed hashes.
  3. Builds release/hf-staging/parentheses-1.0-240M/: exactly what goes to the
     Hugging Face Hub (float32 safetensors only, no pickle, no ONNX).
  4. Builds release/github-release-v1.0.0-tanzanite/: the GitHub release assets
     and the release body.
  5. Verifies every staged file against the bundle manifest.

Large files are hard-linked, so staging costs no extra disk space.

Usage:
    python scripts/stage_release_240m.py
    python scripts/stage_release_240m.py --mirror I:/parentheses/checkpoints/parentheses-1.0-240M-release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "release" / "parentheses-1.0-240M-tanzanite"
HF_STAGE = ROOT / "release" / "hf-staging" / "parentheses-1.0-240M"
GH_STAGE = ROOT / "release" / "github-release-v1.0.0-tanzanite"

# The float32 weights are the certified artifact. Staging must never change them.
CERTIFIED_SAFETENSORS_SHA256 = "bff69c7945533edaf379e69ffc50e47ed3b455c83cafb312defbd1b4c2208a6e"
EXPECTED_PARAMS = 253_544_176

AUTO_MAP = {
    "AutoConfig": "configuration_parentheses.ParenthesesConfig",
    "AutoModelForCausalLM": "modeling_parentheses.ParenthesesForCausalLM",
}

# bundle path -> path inside the Hub repo
HF_FILES = {
    "model.safetensors": "model.safetensors",
    "config.json": "config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "modeling_parentheses.py": "modeling_parentheses.py",
    "configuration_parentheses.py": "configuration_parentheses.py",
    "MODEL_CARD.md": "README.md",
    "BENCHMARK_SPEC.md": "BENCHMARK_SPEC.md",
}

# bundle path -> GitHub release asset name (asset names are flat)
GH_FILES = {
    "model-bf16.safetensors": "model-bf16.safetensors",
    "model.pt": "model.pt",
    "npu/parentheses_240m_v4_step.onnx": "parentheses_240m_v4_step.onnx",
    "npu/onnx_metadata.json": "onnx_metadata.json",
    "npu/parity_report.json": "parity_report.json",
    "BENCHMARK_SPEC.md": "BENCHMARK_SPEC.md",
    "MANIFEST_sha256.tsv": "MANIFEST_sha256.tsv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path) -> None:
    """Hard-link src at dst, falling back to a copy across volumes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def replace_text(path: Path, text: str) -> None:
    """Write text to a fresh file. Never write in place: path may be hard-linked into staging."""
    if path.exists():
        path.unlink()
    path.write_text(text, encoding="utf-8")


def copy_if_changed(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return False
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return True


def sync_bundle() -> list[str]:
    changed = []
    for name in ("modeling_parentheses.py", "configuration_parentheses.py"):
        if copy_if_changed(ROOT / "hf" / name, BUNDLE / name):
            changed.append(name)
    if copy_if_changed(ROOT / "MODEL_CARD.md", BUNDLE / "MODEL_CARD.md"):
        changed.append("MODEL_CARD.md")
    for svg in sorted((ROOT / "docs" / "figures").glob("*.svg")):
        if copy_if_changed(svg, BUNDLE / "figures" / svg.name):
            changed.append(f"figures/{svg.name}")

    cfg_path = BUNDLE / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["params"] == EXPECTED_PARAMS, f"config.json params {cfg['params']} != {EXPECTED_PARAMS}"
    new_cfg = dict(cfg)
    new_cfg["architectures"] = ["ParenthesesForCausalLM"]
    new_cfg["auto_map"] = AUTO_MAP
    # Explicit so older transformers versions (4.x), whose config base class sets
    # these from kwargs with its own defaults, still tie embeddings and stop at EOT.
    new_cfg["tie_word_embeddings"] = True
    new_cfg["eos_token_id"] = cfg.get("eot_token_id", 0)
    new_cfg["pad_token_id"] = 1
    new_cfg["torch_dtype"] = "float32"
    if new_cfg != cfg:
        replace_text(cfg_path, json.dumps(new_cfg, indent=2) + "\n")
        changed.append("config.json")
    return changed


def hash_folder(folder: Path, skip: str = "") -> dict[str, tuple[str, int]]:
    rows = {}
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.name != skip and "__pycache__" not in f.parts:
            rows[f.relative_to(folder).as_posix()] = (sha256(f), f.stat().st_size)
    return rows


def write_manifest(folder: Path) -> dict[str, str]:
    rows = hash_folder(folder, skip="MANIFEST_sha256.tsv")
    lines = ["filename\tsha256\tbytes"] + [f"{k}\t{h}\t{n}" for k, (h, n) in rows.items()]
    replace_text(folder / "MANIFEST_sha256.tsv", "\n".join(lines) + "\n")
    return {k: h for k, (h, _) in rows.items()}


def card_figures() -> list[str]:
    """Figure paths the model card links to, relative to the repo root."""
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    return sorted(set(re.findall(r'src="(docs/figures/[^"]+)"', card)))


def release_body() -> str:
    """The 1.0 section of RELEASE_NOTES.md (everything before the 0.9 history)."""
    text = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    cut = text.lower().find("\n# historical lineage")
    return (text[:cut] if cut != -1 else text).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mirror", help="also copy the small bundle files (not weights) to this archive folder")
    args = ap.parse_args()

    if not (BUNDLE / "model.safetensors").exists():
        print(f"[stage] bundle not found at {BUNDLE}; run scripts/export_release_240m.py first")
        return 1

    changed = sync_bundle()
    print(f"[stage] bundle synced, {len(changed)} file(s) updated: {', '.join(changed) or 'none'}")

    manifest = write_manifest(BUNDLE)
    assert manifest["model.safetensors"] == CERTIFIED_SAFETENSORS_SHA256, (
        "model.safetensors no longer matches the certified sha256: " + manifest["model.safetensors"]
    )
    print(f"[stage] bundle manifest rewritten ({len(manifest)} files); certified weights hash confirmed")

    # Hugging Face Hub staging
    if HF_STAGE.exists():
        shutil.rmtree(HF_STAGE)
    for src, dst in HF_FILES.items():
        place(BUNDLE / src, HF_STAGE / dst)
    for fig in card_figures():
        place(ROOT / fig, HF_STAGE / fig)
    if (ROOT / "LICENSE-MODEL").exists():
        place(ROOT / "LICENSE-MODEL", HF_STAGE / "LICENSE")
    hf_hashes = {k: h for k, (h, _) in hash_folder(HF_STAGE).items()}
    forbidden = [k for k in hf_hashes if k.endswith((".pt", ".onnx", ".bin", ".pkl")) or "bf16" in k]
    assert not forbidden, f"Hub staging must hold float32 safetensors only, found {forbidden}"

    # GitHub release staging
    if GH_STAGE.exists():
        shutil.rmtree(GH_STAGE)
    for src, dst in GH_FILES.items():
        place(BUNDLE / src, GH_STAGE / dst)
    (GH_STAGE / "RELEASE_BODY.md").write_text(release_body(), encoding="utf-8")
    gh_hashes = {k: h for k, (h, _) in hash_folder(GH_STAGE).items()}

    # Verify staged copies against the bundle manifest
    problems = [f"hub:{dst}" for src, dst in HF_FILES.items() if hf_hashes[dst] != manifest[src]]
    problems += [
        f"github:{dst}" for src, dst in GH_FILES.items()
        if src != "MANIFEST_sha256.tsv" and gh_hashes[dst] != manifest[src]
    ]
    if problems:
        print(f"[stage] HASH MISMATCH: {problems}")
        return 1

    def total(folder: Path) -> float:
        return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6

    print(f"[stage] Hub staging:    {HF_STAGE}  ({len(hf_hashes)} files, {total(HF_STAGE):,.1f} MB)")
    for k in hf_hashes:
        print(f"          {k}")
    print(f"[stage] GitHub staging: {GH_STAGE}  ({len(gh_hashes)} files, {total(GH_STAGE):,.1f} MB)")
    for k in gh_hashes:
        print(f"          {k}")
    print("[stage] all staged files match the bundle manifest")

    if args.mirror:
        mirror = Path(args.mirror)
        n = 0
        for rel in manifest:
            src = BUNDLE / rel
            if src.stat().st_size < 50_000_000 and copy_if_changed(src, mirror / rel):
                n += 1
        copy_if_changed(BUNDLE / "MANIFEST_sha256.tsv", mirror / "MANIFEST_sha256.tsv")
        print(f"[stage] mirrored {n} small file(s) and the manifest to {mirror}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
