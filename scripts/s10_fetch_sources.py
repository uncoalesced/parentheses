#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/s10_fetch_sources.py
Session 10 (Tanzanite) source fetcher: downloads the three new public datasets to F: and writes a
SOURCE.txt next to each with the licence quoted from the live HF card.

    squad_v2 (CC-BY-SA-4.0), nq_open (CC-BY-SA-3.0), Synthetic_Unanswerable_Math (MIT)

Rules honoured (AGENTS.md / LICENSE_POLICY.md):
  - nothing is written to C:. HF cache, temp and output all live on F:.
  - the licence is read from the card at download time and must be on the pass list, else the
    dataset is NOT downloaded (unknown / missing / NC / ND -> skip, tell Joel).
  - F: is a USB HDD: files are fetched sequentially, never memmapped.

Usage:
    .venv\\Scripts\\python.exe scripts\\s10_fetch_sources.py            # download all three
    .venv\\Scripts\\python.exe scripts\\s10_fetch_sources.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEST = Path(r"F:\Lingual_Data\raw\hf\s10")
HF_HOME = Path(r"F:\Lingual_Data\distillation\hf_cache")
TMP = DEST / "_tmp"

# set before huggingface_hub is imported so nothing lands on C:
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TEMP", str(TMP))
os.environ.setdefault("TMP", str(TMP))
os.environ.setdefault("TMPDIR", str(TMP))

PASS = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "cc-by-3.0",
        "cc-by-sa-4.0", "cc-by-sa-3.0", "odc-by", "odc-pddl", "odbl"}
FAIL_MARKERS = ("-nc", "-nd", "unknown", "other", "research")

DATASETS = [
    {"id": "rajpurkar/squad_v2", "name": "squad_v2", "role": "unanswerable + grounded reading"},
    {"id": "google-research-datasets/nq_open", "name": "nq_open", "role": "R-Tuning question pool"},
    {"id": "lime-nlp/Synthetic_Unanswerable_Math", "name": "Synthetic_Unanswerable_Math",
     "role": "unanswerable math (not enough information)"},
]


def licence_ok(tag: str | None) -> bool:
    """True only for an explicit pass-list licence tag. Missing / unknown / NC / ND fail."""
    if not tag:
        return False
    t = tag.strip().lower()
    if any(m in t for m in FAIL_MARKERS):
        return False
    return t in PASS


def card_licence(info) -> str | None:
    card = getattr(info, "card_data", None)
    lic = getattr(card, "license", None) if card is not None else None
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if lic:
        return str(lic)
    for t in getattr(info, "tags", []) or []:
        if t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def self_test() -> None:
    assert licence_ok("cc-by-sa-4.0") and licence_ok("MIT") and licence_ok("apache-2.0")
    assert not licence_ok("cc-by-nc-4.0")
    assert not licence_ok("unknown")
    assert not licence_ok(None)
    assert not licence_ok("cc-by-nd-4.0")
    assert not licence_ok("other")
    print("[self-test] s10_fetch_sources ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    assert not str(DEST).upper().startswith("C:"), "C: write ban"
    TMP.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    rc = 0
    for d in DATASETS:
        info = api.dataset_info(d["id"])
        lic = card_licence(info)
        out = DEST / d["name"]
        if not licence_ok(lic):
            print(f"[SKIP] {d['id']}: licence tag {lic!r} is not on the pass list")
            rc = 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        path = snapshot_download(repo_id=d["id"], repo_type="dataset", local_dir=str(out),
                                 allow_patterns=["*.parquet", "*.json", "*.jsonl", "*.md", "*.csv", "*.txt"])
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        (out / "SOURCE.txt").write_text(
            f"origin: https://huggingface.co/datasets/{d['id']}\n"
            f"license (verbatim card tag): {lic}\n"
            f"pulled: {time.strftime('%Y-%m-%d')}\n"
            f"revision: {info.sha}\n"
            f"size_bytes: {size}\n"
            f"role in S10: {d['role']}\n"
            f"policy: LICENSE_POLICY.md pass (commercial use permitted)\n",
            encoding="utf-8")
        print(f"[OK] {d['id']} lic={lic} {size/1e6:.1f} MB in {time.time()-t0:.0f}s -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
