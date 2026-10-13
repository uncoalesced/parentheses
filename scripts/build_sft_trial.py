#!/usr/bin/env python3
"""SFT trial set for session 2 (Claude Code, 2026-10-12): data/processed_sft_v1 + verified
distill rows (F:\\Lingual_Data\\scripts\\distill_reasoning.py output), same format train.py --sft-spans expects.

    .venv\\Scripts\\python.exe scripts\\build_sft_trial.py --out data/processed_sft_trial
    .venv\\Scripts\\python.exe scripts\\build_sft_trial.py --self-test

Each distill row is `<|user|><eng> p<|assistant|>a<|endoftext|>`. Mask = 1 on every token after
<|assistant|> through <|endoftext|> (prepare_sft_bpe.py convention: mask[j] scores token j).
Rows are appended after the base docs; boundaries get one entry per appended doc.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def encode_rows(tok, rows: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    # tok must keep encode_special_tokens False so <|user|> etc. map to their single ids
    user, asst, eot = (tok.token_to_id(t) for t in ("<|user|>", "<|assistant|>", "<|endoftext|>"))
    out = []
    for text in rows:
        ids = tok.encode(text.rstrip("\n"), add_special_tokens=False).ids
        assert ids[0] == user and ids[-1] == eot and ids.count(asst) == 1, text[:80]
        mask = np.zeros(len(ids), np.uint8)
        mask[ids.index(asst) + 1:] = 1
        out.append((np.asarray(ids, np.uint16), mask))
    return out


def build(base: Path, distill: Path | None, tok, out: Path) -> dict:
    data = np.fromfile(base / "train.bin", np.uint16)
    mask = np.fromfile(base / "train.mask.bin", np.uint8)
    bnd = np.load(base / "train_boundaries.npy")
    assert len(data) == len(mask) and bnd[0] == 0 and (np.diff(bnd) > 0).all()
    rows = []
    if distill and distill.is_file():
        lines = [l for l in distill.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            try:
                json.loads(lines[-1])
            except ValueError:   # distiller still appending: drop the half-written last line
                lines.pop()
        rows = [json.loads(l)["text"] for l in lines]
    enc = encode_rows(tok, rows)
    starts = len(data) + np.cumsum([0] + [len(i) for i, _ in enc[:-1]]) if enc else np.array([], np.int64)
    data = np.concatenate([data] + [i for i, _ in enc])
    mask = np.concatenate([mask] + [m for _, m in enc])
    bnd = np.concatenate([bnd, starts]).astype(bnd.dtype)
    out.mkdir(parents=True, exist_ok=True)
    data.tofile(out / "train.bin")
    mask.tofile(out / "train.mask.bin")
    np.save(out / "train_boundaries.npy", bnd)
    man = {"base": str(base), "distill": str(distill) if rows else None, "distill_docs": len(rows),
           "distill_tokens": int(sum(len(i) for i, _ in enc)), "tokens": int(len(data)),
           "supervised": int(mask.sum()), "docs": int(len(bnd)),
           "sha256": hashlib.sha256(data.tobytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def self_test():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "data/tokenizer/v3_32k/tokenizer.json"))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = d / "base"
        base.mkdir()
        np.arange(10, dtype=np.uint16).tofile(base / "train.bin")
        np.ones(10, np.uint8).tofile(base / "train.mask.bin")
        np.save(base / "train_boundaries.npy", np.array([0, 4], np.int64))
        rows = ["<|user|><eng> 1+1?<|assistant|><think>\nadd\n</think>\n2<|endoftext|>\n",
                "<|user|><eng> hi<|assistant|>hello<|endoftext|>\n"]
        dj = d / "d.jsonl"
        dj.write_text("".join(json.dumps({"text": r}) + "\n" for r in rows), encoding="utf-8")
        man = build(base, dj, tok, d / "out")
        data = np.fromfile(d / "out/train.bin", np.uint16)
        mask = np.fromfile(d / "out/train.mask.bin", np.uint8)
        bnd = np.load(d / "out/train_boundaries.npy")
        assert len(data) == len(mask) == man["tokens"] and man["distill_docs"] == 2
        assert list(bnd[:2]) == [0, 4] and bnd[2] == 10 and len(bnd) == 4 and (np.diff(bnd) > 0).all()
        first = data[bnd[2]:bnd[3]].tolist()
        assert tok.decode(first, skip_special_tokens=False) == rows[0].rstrip("\n")
        a = first.index(tok.token_to_id("<|assistant|>"))
        m = mask[bnd[2]:bnd[3]]
        assert m[:a + 1].sum() == 0 and m[a + 1:].all()   # prompt unscored, answer + eot scored
        assert build(base, d / "missing.jsonl", tok, d / "o2")["tokens"] == 10   # no distill rows
    print("[self-test] build_sft_trial ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=str(ROOT / "data/processed_sft_v1"))
    p.add_argument("--distill", default=r"F:\Lingual_Data\distillation\out\sft_tail_distill.jsonl")
    p.add_argument("--out", default=str(ROOT / "data/processed_sft_trial"))
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "data/tokenizer/v3_32k/tokenizer.json"))
    print(json.dumps(build(Path(a.base), Path(a.distill), tok, Path(a.out))))


if __name__ == "__main__":
    sys.exit(main())
