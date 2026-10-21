#!/usr/bin/env python3
"""Download the open SFT sources for the conversational + reasoning recovery run (SFT-real v1).

Sources (all permissive):
  - HuggingFaceTB/smol-smoltalk           Apache-2.0  multi-turn chat built for 135M-360M models
  - microsoft/orca-math-word-problems-200k MIT        step-by-step grade-school math
  - meta-math/MetaMathQA                  MIT         GSM8K/MATH augmentations with CoT, "The answer is: X"

Writes to data/raw/sft_real/<repo_slug>/<file>. Resumable: existing files of the right size are skipped.
Usage: .venv\\Scripts\\python.exe scripts\\fetch_sft_real_sources.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "raw" / "sft_real"

SOURCES = {
    "HuggingFaceTB/smol-smoltalk": None,  # all parquet under data/
    "microsoft/orca-math-word-problems-200k": None,
    "meta-math/MetaMathQA": ["MetaMathQA-395K.json", "README.md"],
}


def list_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=30) as r:
        j = json.load(r)
    return [s["rfilename"] for s in j.get("siblings", [])
            if s["rfilename"].endswith((".parquet", ".json", "README.md"))]


def fetch(repo: str, fname: str) -> None:
    dst = OUT / repo.replace("/", "__") / fname
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            size = int(r.headers.get("Content-Length") or 0)
    except Exception:
        size = 0
    if dst.exists() and size and dst.stat().st_size == size:
        print(f"[skip] {repo}/{fname} ({size/1e6:.1f} MB)", flush=True)
        return
    t0 = time.time()
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.replace(dst)
    print(f"[ok] {repo}/{fname} {dst.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    for repo, files in SOURCES.items():
        files = files or list_files(repo)
        for fn in files:
            if "test-" in fn and repo.endswith("smol-smoltalk"):
                pass  # keep test split too (used as held-out chat val)
            for attempt in range(3):
                try:
                    fetch(repo, fn)
                    break
                except Exception as e:  # network hiccup: retry
                    print(f"[retry {attempt+1}] {repo}/{fn}: {e}", flush=True)
                    time.sleep(5)
            else:
                print(f"[FAIL] {repo}/{fn}", flush=True)
                return 1
    print("[done] all sources fetched", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
