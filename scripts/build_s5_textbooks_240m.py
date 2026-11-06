#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Session 5 (240M) textbook corpus builder: Cosmopedia-v2 + FineWeb-Edu on F: -> uint16 memmap on E:.

Every document is
    <eng> text <|endoftext|>
from the `text` field only. Cosmopedia's `prompt` is the instruction given to the generator, not part
of the document, so it is never included. FineWeb-Edu has no prompt.

Filters, per document: too short, looks truncated (no terminal punctuation), contains one of the
60 probe prompts, exact duplicate (hash of word-normalized text, across both sources).

Reads F: sequentially (USB HDD rule): Cosmopedia jsonl shards line by line, FineWeb parquet one
row group at a time. Each source stops at its token cap. Then a shuffled train.bin / val.bin.
    .venv\\Scripts\\python.exe scripts\\build_s5_textbooks_240m.py --out data\\processed_s5_textbooks_240m
    .venv\\Scripts\\python.exe scripts\\build_s5_textbooks_240m.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_s4_stem_240m import WORD, assemble, probe_keys  # noqa: E402

RAW = Path(r"F:\Lingual_Data\raw\hf_permissive_stem_code")
# name -> (shard glob, license, token cap)
SOURCES = {
    "cosmopedia_v2": ("curated/cosmopedia/shards/cosmo_*.jsonl", "Apache-2.0 rows / ODC-By card", 75_000_000),
    "fineweb_edu": ("smollm-corpus-textbooks-edu/fineweb-edu-dedup/train-*.parquet", "ODC-By", 75_000_000),
}
MIN_CHARS = 300
TERMINAL = set('.!?"\'\u201d\u2019)]}\u00bb\u2026`*')
TRAILING_CITATION = re.compile(r"\s*(\[\d+\]|\(\d{4}\))\s*$")


def looks_truncated(text: str) -> bool:
    """True if the document doesn't end like a finished sentence."""
    t = TRAILING_CITATION.sub("", text.rstrip())
    return not t or t[-1] not in TERMINAL


def keep(text: str, keys: list[str], seen: set, stats: dict) -> bool:
    if len(text) < MIN_CHARS:
        stats["too_short"] += 1
        return False
    if looks_truncated(text):
        stats["truncated"] += 1
        return False
    norm = " " + " ".join(WORD.findall(text.lower())) + " "
    if any(k in norm for k in keys):
        stats["contaminated"] += 1
        return False
    h = hashlib.sha1(norm.encode("utf-8")).digest()
    if h in seen:
        stats["dup"] += 1
        return False
    seen.add(h)
    return True


def iter_texts(name: str):
    """Yield raw `text` strings, reading each shard sequentially."""
    glob = SOURCES[name][0]
    for shard in sorted(RAW.glob(glob)):
        if name == "cosmopedia_v2":
            with open(shard, encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line).get("text", "")
        else:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(shard)
            for g in range(pf.metadata.num_row_groups):
                for t in pf.read_row_group(g, columns=["text"]).column("text").to_pylist():
                    yield t or ""


def encode_source(name: str, tok, ids: dict, out_dir: Path, cap: int, keys: list[str], seen: set) -> dict:
    bin_path, bnd_path = out_dir / f"_src_{name}.bin", out_dir / f"_src_{name}_starts.npy"
    stats = dict(rows=0, kept=0, too_short=0, truncated=0, contaminated=0, dup=0, tokens=0)
    starts, buf, total = [], [], 0
    batch: list[str] = []

    def flush():
        nonlocal total
        for enc in tok.encode_batch(batch, add_special_tokens=False):
            if total >= cap:
                break
            doc = [ids["eng"], *enc.ids, ids["eot"]]
            starts.append(total)
            buf.append(np.asarray(doc, dtype=np.uint16))
            total += len(doc)
            stats["kept"] += 1
        batch.clear()

    for text in iter_texts(name):
        if total >= cap:
            break
        stats["rows"] += 1
        text = text.strip()
        if keep(text, keys, seen, stats):
            batch.append(text)
            if len(batch) >= 1024:
                flush()
                print(f"[{name}] {total / 1e6:.1f}M tokens, {stats['rows']} rows", flush=True)
    if batch and total < cap:
        flush()
    arr = np.concatenate(buf) if buf else np.zeros(0, dtype=np.uint16)
    arr.tofile(bin_path)
    np.save(bnd_path, np.asarray(starts, dtype=np.int64))
    stats["tokens"] = int(total)
    stats["truncated_rate"] = round(stats["truncated"] / max(1, stats["rows"]), 4)
    print(f"[{name}] {stats}", flush=True)
    return stats


def verify(out_dir: Path, ids: dict) -> dict:
    """Every doc is <eng> ... <|endoftext|>, non-empty, with exactly one EOT and one tag."""
    res = {}
    for split in ("train", "val"):
        a = np.memmap(out_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        b = np.load(out_dir / f"{split}_boundaries.npy")
        e = np.append(b[1:], len(a))
        bad = 0
        for s, t in zip(b, e):
            d = np.asarray(a[s:t])
            ok = (len(d) > 2 and d[0] == ids["eng"] and d[-1] == ids["eot"]
                  and int((d == ids["eot"]).sum()) == 1 and int((d == ids["eng"]).sum()) == 1)
            bad += not ok
        res[split] = dict(docs=int(len(b)), tokens=int(len(a)), malformed=int(bad))
        assert bad == 0, f"{split}: {bad} malformed documents"
    return res


def special_ids(tok) -> dict:
    ids = {k: tok.token_to_id(v) for k, v in dict(eot="<|endoftext|>", eng="<eng>").items()}
    missing = [k for k, v in ids.items() if v is None]
    assert not missing, f"tokenizer lacks {missing}"
    return ids


def self_test() -> None:
    long = "word " * 80
    assert not looks_truncated(long + "end.")
    assert not looks_truncated(long + 'he said "yes."')
    assert not looks_truncated(long + "a claim. [12]")
    assert looks_truncated(long + "and then the")
    assert looks_truncated("")
    keys = [" what is 17 multiplied by 6 "]
    st = dict(too_short=0, truncated=0, contaminated=0, dup=0)
    seen: set = set()
    assert not keep("short.", keys, seen, st) and st["too_short"] == 1
    assert not keep(long + "and so", keys, seen, st) and st["truncated"] == 1
    assert not keep(long + "Question: What is 17 multiplied by 6? Done.", keys, seen, st) and st["contaminated"] == 1
    assert keep(long + "Fine.", keys, seen, st)
    assert not keep((long + "Fine.").upper(), keys, seen, st) and st["dup"] == 1
    print("build_s5_textbooks_240m self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed_s5_textbooks_240m")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=5001)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = special_ids(tok)
    args.out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.sources.split(",") if n]
    keys, seen, t0, per = probe_keys(), set(), time.time(), {}
    for n in names:
        per[n] = encode_source(n, tok, ids, args.out, SOURCES[n][2], keys, seen)
    splits = assemble(args.out, names, args.val_frac, args.seed)
    checks = verify(args.out, ids)
    total = sum(p["tokens"] for p in per.values())
    manifest = {
        "version": "processed_s5_textbooks_240m",
        "license_compliance": "Apache-2.0-Permissive-Only",
        "built_by": "scripts/build_s5_textbooks_240m.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "<eng> text <|endoftext|>",
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "sources": {n: dict(glob=str(RAW / SOURCES[n][0]), license=SOURCES[n][1], cap=SOURCES[n][2], **per[n])
                    for n in names},
        "shares": {n: round(per[n]["tokens"] / total, 4) for n in names} if total else {},
        "splits": splits,
        "verify": checks,
        "filters": {"min_chars": MIN_CHARS, "truncated": "last char not in terminal set", "probe_prompts": len(keys),
                    "dedup": "sha1 of word-normalized text, across sources"},
        "excluded": ["Cosmopedia `prompt` field (generator instruction, not document text)"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for n in names:
        (args.out / f"_src_{n}.bin").unlink()
        (args.out / f"_src_{n}_starts.npy").unlink()
    print(json.dumps({k: manifest[k] for k in ("shares", "splits", "verify")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
