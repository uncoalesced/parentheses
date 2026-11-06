#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Session 4 (240M) STEM/math corpus builder: curated jsonl on F: -> uint16 memmap on E:.

Fixes the curated files' `text` field, which drops the question (only `<think>reasoning</think>`
was kept for Stratos / orca / MetaMath / Numina). Every document is rebuilt as
    <|user|> <eng> prompt <|assistant|> body <|endoftext|>
where body is
    Stratos : <think> thought </think> solution     (from <|begin_of_thought|> / <|begin_of_solution|>)
    Numina  : <think> solution </think> \\boxed{...}  (last boxed answer; no think wrapper if none)
    MetaMath: <think> reasoning </think> response      ("The answer is: ...")
    orca    : reasoning                                 (already a short worked answer)

Only the full curated files are read (never the *_partN / *_single copies, which duplicate rows).
smoltalk is excluded by default: its rows lost the subset tag, and some subsets are Llama-generated.

Reads F: sequentially (USB HDD rule); writes per-source shards, then a shuffled train.bin / val.bin.
    .venv\\Scripts\\python.exe scripts\\build_s4_stem_240m.py --out data\\processed_s4_stem_240m
    .venv\\Scripts\\python.exe scripts\\build_s4_stem_240m.py --self-test
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
CURATED = Path(r"F:\Lingual_Data\raw\hf_permissive_stem_code\curated")

# name -> (file, license, token cap)
SOURCES = {
    "bespoke_stratos": ("bespoke_stratos/curated_stratos.jsonl", "Apache-2.0", 70_000_000),
    "metamathqa": ("metamathqa/curated_metamathqa.jsonl", "MIT", 60_000_000),
    "orca_math": ("orca_math/curated_orca_math.jsonl", "MIT", 60_000_000),
    "numinamath_cot": ("numinamath_cot/curated_numinamath.jsonl", "Apache-2.0", 120_000_000),
}
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
WORD = re.compile(r"\w+", re.UNICODE)


def body_for(name: str, rec: dict) -> str | None:
    reasoning, response = rec.get("reasoning", "").strip(), rec.get("response", "").strip()
    if name == "bespoke_stratos":
        m = re.search(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>.*?<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>",
                      reasoning, re.S)
        if not m:
            return None
        return f"<think>\n{m.group(1).strip()}\n</think>\n{m.group(2).strip()}"
    if name == "numinamath_cot":
        boxes = BOXED.findall(reasoning)
        return f"<think>\n{reasoning}\n</think>\n\\boxed{{{boxes[-1]}}}" if boxes else reasoning
    if name == "metamathqa":
        return f"<think>\n{reasoning}\n</think>\n{response}" if response else None
    if name == "orca_math":
        return reasoning or None
    raise KeyError(name)


def probe_keys() -> list[str]:
    """Word-normalized 60-probe prompts; a training prompt containing one (>= 5 words) is dropped."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.eval_60p_adversarial_suite import PROBES  # noqa: E402
    keys = []
    for p in PROBES:
        w = WORD.findall(p["prompt"].lower())
        if len(w) >= 5:
            keys.append(" " + " ".join(w) + " ")
    return keys


def encode_source(name: str, tok, ids: dict, out_dir: Path, cap: int, keys: list[str], seen: set) -> dict:
    rel, _lic, _ = SOURCES[name]
    src = CURATED / rel
    bin_path, bnd_path = out_dir / f"_src_{name}.bin", out_dir / f"_src_{name}_starts.npy"
    stats = dict(rows=0, kept=0, no_body=0, no_prompt=0, dup_prompt=0, contaminated=0, think_unbalanced=0, tokens=0)
    starts, buf, total = [], [], 0
    batch_docs: list[tuple[str, str]] = []

    def flush():
        nonlocal total
        if not batch_docs:
            return
        p_enc = tok.encode_batch([p for p, _ in batch_docs], add_special_tokens=False)
        b_enc = tok.encode_batch([b for _, b in batch_docs], add_special_tokens=False)
        for pe, be in zip(p_enc, b_enc):
            doc = [ids["user"], ids["eng"], *pe.ids, ids["assistant"], *be.ids, ids["eot"]]
            if total + len(doc) > cap:
                break
            starts.append(total)
            buf.append(np.asarray(doc, dtype=np.uint16))
            total += len(doc)
            stats["kept"] += 1
        batch_docs.clear()

    with open(src, encoding="utf-8") as fh:  # sequential read of F:
        for line in fh:
            if total >= cap:
                break
            stats["rows"] += 1
            rec = json.loads(line)
            prompt = rec.get("prompt", "").strip()
            if not prompt:
                stats["no_prompt"] += 1
                continue
            body = body_for(name, rec)
            if not body:
                stats["no_body"] += 1
                continue
            if body.count("<think>") != body.count("</think>"):
                stats["think_unbalanced"] += 1
                continue
            norm = " " + " ".join(WORD.findall(prompt.lower())) + " "
            if any(k in norm for k in keys):
                stats["contaminated"] += 1
                continue
            h = hashlib.sha1(norm.encode("utf-8")).digest()
            if h in seen:
                stats["dup_prompt"] += 1
                continue
            seen.add(h)
            batch_docs.append((prompt, body))
            if len(batch_docs) >= 2048:
                flush()
    flush()
    arr = np.concatenate(buf) if buf else np.zeros(0, dtype=np.uint16)
    arr.tofile(bin_path)
    np.save(bnd_path, np.asarray(starts, dtype=np.int64))
    stats["tokens"] = int(total)
    print(f"[{name}] {stats}", flush=True)
    return stats


def assemble(out_dir: Path, names: list[str], val_frac: float, seed: int) -> dict:
    """Shuffle all documents across sources into train.bin / val.bin (+ boundaries)."""
    rng = np.random.default_rng(seed)
    srcs = []
    for n in names:
        a = np.memmap(out_dir / f"_src_{n}.bin", dtype=np.uint16, mode="r")
        s = np.load(out_dir / f"_src_{n}_starts.npy")
        e = np.append(s[1:], len(a))
        srcs.append((a, s, e))
    docs = np.concatenate([np.stack([np.full(len(s), i), np.arange(len(s))], 1) for i, (_, s, _) in enumerate(srcs)])
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_frac))
    out = {}
    for split, sel in (("val", docs[:n_val]), ("train", docs[n_val:])):
        lens = np.array([srcs[i][2][j] - srcs[i][1][j] for i, j in sel], dtype=np.int64)
        bounds = np.concatenate([[0], np.cumsum(lens)[:-1]])
        mm = np.memmap(out_dir / f"{split}.bin", dtype=np.uint16, mode="w+", shape=(int(lens.sum()),))
        pos = 0
        for (i, j), L in zip(sel, lens):
            a, s, _ = srcs[i]
            mm[pos:pos + L] = a[s[j]:s[j] + L]
            pos += L
        mm.flush()
        del mm
        np.save(out_dir / f"{split}_boundaries.npy", bounds)
        out[split] = dict(docs=int(len(sel)), tokens=int(lens.sum()))
    del srcs
    return out


def verify(out_dir: Path, ids: dict) -> dict:
    """Every doc starts <|user|><eng>, has exactly one <|assistant|>, ends with EOT, think tags balance."""
    res = {}
    for split in ("train", "val"):
        a = np.memmap(out_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        b = np.load(out_dir / f"{split}_boundaries.npy")
        e = np.append(b[1:], len(a))
        bad = 0
        for s, t in zip(b, e):
            d = np.asarray(a[s:t])
            ok = (d[0] == ids["user"] and d[1] == ids["eng"] and d[-1] == ids["eot"]
                  and int((d == ids["assistant"]).sum()) == 1 and int((d == ids["eot"]).sum()) == 1
                  and int((d == ids["think_open"]).sum()) == int((d == ids["think_close"]).sum()))
            bad += not ok
        res[split] = dict(docs=int(len(b)), malformed=int(bad))
        assert bad == 0, f"{split}: {bad} malformed documents"
    return res


def special_ids(tok) -> dict:
    ids = {k: tok.token_to_id(v) for k, v in dict(user="<|user|>", assistant="<|assistant|>", eot="<|endoftext|>",
                                                     eng="<eng>", think_open="<think>", think_close="</think>").items()}
    missing = [k for k, v in ids.items() if v is None]
    assert not missing, f"tokenizer lacks {missing}"
    return ids


def self_test() -> None:
    s = {"reasoning": "<|begin_of_thought|>\n\nhmm 2+2\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n\\boxed{4}\n\n<|end_of_solution|>"}
    assert body_for("bespoke_stratos", s) == "<think>\nhmm 2+2\n</think>\n\\boxed{4}"
    assert body_for("bespoke_stratos", {"reasoning": "no markers"}) is None
    n = body_for("numinamath_cot", {"reasoning": "so x = \\frac{1}{2}, giving \\boxed{\\frac{1}{2}}."})
    assert n.endswith("</think>\n\\boxed{\\frac{1}{2}}"), n
    assert body_for("numinamath_cot", {"reasoning": "no box"}) == "no box"
    assert body_for("metamathqa", {"reasoning": "r", "response": "The answer is: 5"}) == "<think>\nr\n</think>\nThe answer is: 5"
    assert body_for("metamathqa", {"reasoning": "r", "response": ""}) is None
    assert body_for("orca_math", {"reasoning": "4 people."}) == "4 people."
    print("build_s4_stem_240m self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed_s4_stem_240m")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=4001)
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
        "version": "processed_s4_stem_240m",
        # Each source's license is its HF card's (Apache-2.0 / MIT); see sources[*].license.
        "license_compliance": "Apache-2.0-Permissive-Only",
        "built_by": "scripts/build_s4_stem_240m.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "<|user|> <eng> prompt <|assistant|> body <|endoftext|>",
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "sources": {n: dict(file=str(CURATED / SOURCES[n][0]), license=SOURCES[n][1], cap=SOURCES[n][2], **per[n]) for n in names},
        "shares": {n: round(per[n]["tokens"] / total, 4) for n in names} if total else {},
        "splits": splits,
        "verify": checks,
        "excluded": ["smoltalk (subset provenance lost; Llama-generated subsets)", "*_partN / *_single duplicate files",
                     "python-edu (download incomplete; blob ids only)"],
        "notes": "orca-math and MetaMathQA are MIT but GPT-generated; check LICENSE_POLICY.md before a release.",
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
