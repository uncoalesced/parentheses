#!/usr/bin/env python3
"""Run Nemotron 3.5 Super VL distillation specs against NVIDIA NIM.

Reads spec JSONL files from scripts/gen_distill_specs.py and writes one raw
output JSONL per spec file. Resume-safe (skips ids already present), filters
refusals/empty/non-ASCII output, retries 429/5xx with backoff.

Rows are research-only under the NVIDIA API trial terms (Joel, 2026-10-13).

Usage:
  .venv\\Scripts\\python.exe scripts\\distill_nemotron_v2.py --specs out\\distill_v2\\specs\\articles.jsonl
  .venv\\Scripts\\python.exe scripts\\distill_nemotron_v2.py --specs out\\distill_v2\\specs --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
BASE_URL = "https://integrate.api.nvidia.com/v1"
LICENSE_TAG = "Research-Only (Joel) - NVIDIA API trial terms"

REFUSAL_RE = re.compile(
    r"(?:i\s+can(?:'|no)t\s+(?:help|assist|provide)"
    r"|i\s+cannot\s+(?:help|assist|provide|comply)"
    r"|i'?m\s+(?:sorry|unable)[, ]+(?:but|to)"
    r"|as\s+an\s+ai\b|as\s+a\s+language\s+model"
    r"|i\s+must\s+(?:decline|refuse))",
    re.IGNORECASE,
)
TRANS = str.maketrans({
    "\u00d7": "x", "\u00f7": "/", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...",
    "\u00b0": " degrees", "\u221a": "sqrt", "\u2265": ">=", "\u2264": "<=",
    "\u2260": "!=", "\u03c0": "pi", "\u00b2": "^2", "\u00b3": "^3", "\u20b9": "Rs ",
    "\u00e9": "e", "\u00e8": "e", "\u00ea": "e", "\u00e1": "a", "\u00e0": "a",
    "\u00f3": "o", "\u00f6": "o", "\u00fc": "u", "\u00df": "ss", "\u00e7": "c",
    "\u00ee": "i", "\u00f1": "n", "\u00a0": " ",
})


def clean_output(text: str) -> tuple[str | None, str | None]:
    """Returns (cleaned, reject_reason). Cleans common unicode, rejects leftovers."""
    if not text or not text.strip():
        return None, "empty"
    text = text.translate(TRANS).replace("\r\n", "\n").strip()
    if REFUSAL_RE.search(text):
        return None, "refusal"
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        return None, f"nonascii:{''.join(bad[:8])}"
    return text, None


def post(payload: dict, key: str, timeout: int = 360, retries: int = 6) -> dict:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            last = f"HTTP {e.code}: {body[:300]}"
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(last) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"connection: {e}"
            if attempt == retries - 1:
                raise RuntimeError(last) from e
        time.sleep(min(60, 4 * (2 ** attempt)))
    raise RuntimeError(last or "max retries")


def load_specs(path: Path) -> list[dict]:
    specs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            specs.append(json.loads(line))
    return specs


def done_ids(path: Path) -> set[str]:
    ids = set()
    if not path.is_file():
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("id"):
                        ids.add(d["id"])
                except json.JSONDecodeError:
                    continue
    return ids


def worker(spec: dict, key: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": spec["prompt"]},
        ],
        "max_tokens": int(spec.get("max_tokens", 1800)),
        "temperature": float(spec.get("temperature", 0.2)),
    }
    t0 = time.time()
    res = post(payload, key)
    dt = time.time() - t0
    choice = res["choices"][0]
    msg = choice.get("message") or {}
    content, reason = clean_output(msg.get("content") or "")
    usage = res.get("usage") or {}
    row = {
        "id": spec["id"],
        "kind": spec["kind"],
        "prompt": spec["prompt"],
        "meta": spec.get("meta", {}),
        "content": content,
        "reasoning": (msg.get("reasoning_content") or "").strip()[:12000],
        "reject": reason,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "latency_s": round(dt, 1),
        "teacher": MODEL,
        "base_url": BASE_URL,
        "license": LICENSE_TAG,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return row


def run_file(spec_path: Path, out_dir: Path, key: str, workers: int, limit: int = 0) -> dict:
    specs = load_specs(spec_path)
    if limit:
        specs = specs[:limit]
    out_path = out_dir / f"{spec_path.stem}.jsonl"
    rej_path = out_dir / f"{spec_path.stem}.rejects.jsonl"
    have = done_ids(out_path) | done_ids(rej_path)
    todo = [s for s in specs if s["id"] not in have]
    print(f"[distill] {spec_path.stem}: {len(specs)} specs, {len(todo)} todo, "
          f"{len(have)} done", flush=True)
    stats = {"ok": 0, "rejected": 0, "errors": 0, "started": time.time(), "todo": len(todo)}
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8", newline="\n") as f_out, \
         open(rej_path, "a", encoding="utf-8", newline="\n") as f_rej:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(worker, s, key): s for s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                spec = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {"id": spec["id"], "kind": spec["kind"], "reject": f"error:{e}",
                           "teacher": MODEL, "license": LICENSE_TAG,
                           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
                with lock:
                    if row.get("reject"):
                        f_rej.write(json.dumps(row, ensure_ascii=True) + "\n")
                        f_rej.flush()
                        stats["rejected" if not str(row["reject"]).startswith("error") else "errors"] += 1
                    else:
                        f_out.write(json.dumps(row, ensure_ascii=True) + "\n")
                        f_out.flush()
                        stats["ok"] += 1
                if i % 10 == 0 or i == len(todo):
                    dt = time.time() - t0
                    rate = i / dt if dt else 0
                    eta = (len(todo) - i) / rate if rate else 0
                    print(f"[distill] {spec_path.stem}: {i}/{len(todo)} ok={stats['ok']} "
                          f"rej={stats['rejected']} err={stats['errors']} "
                          f"{rate:.2f}/s eta {eta/60:.1f} min", flush=True)
    stats["elapsed_s"] = round(time.time() - stats["started"], 1)
    print(f"[distill] {spec_path.stem} done: {stats}", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="spec .jsonl or directory")
    ap.add_argument("--out-dir", default=str(Path("out") / "distill_v2" / "raw"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("[distill] NVIDIA_API_KEY not set", file=sys.stderr)
        return 2

    src = Path(args.specs)
    if src.is_dir():
        files = sorted(src.glob("*.jsonl"))
    else:
        files = [src]
    if not files:
        print(f"[distill] no spec files at {src}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        run_file(f, out_dir, key, args.workers, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
