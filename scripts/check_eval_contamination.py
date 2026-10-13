#!/usr/bin/env python3
"""Flag eval queries that overlap training/distillation prompts (Claude Code, 2026-10-13).

Checks every prompt of the given query JSONLs against the distillation seeds (MBPP train/validation/prompt,
GSM8K train; read from F:\\Lingual_Data\\distillation\\hf_cache) and the SFT v1 prompts, by word 8-gram overlap.
A query is flagged when >= --max-share of its 8-grams appear in one source (default 0.5).
    .venv\\Scripts\\python.exe scripts\\check_eval_contamination.py eval\\queries_120m_reason250.jsonl eval\\queries_120m_lingual250.jsonl
    .venv\\Scripts\\python.exe scripts\\check_eval_contamination.py --self-test
shortcut: does not scan the 1B-token v5 pretrain corpus; val-text continuations can still be near-dups of
pretrain docs (~1% for Dravidian val, see memory "val near-dup overlap").
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"\w+", re.UNICODE)


def grams(text: str, n: int = 8) -> set:
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def flag(queries: list[dict], sources: dict[str, set], max_share: float) -> list[tuple]:
    out = []
    for q in queries:
        g = grams(q["prompt"])
        if not g:
            continue
        for name, pool in sources.items():
            share = len(g & pool) / len(g)
            if share >= max_share:
                out.append((q["id"], name, round(share, 2)))
    return out


def load_sources() -> dict[str, set]:
    os.environ.setdefault("HF_HOME", r"F:\Lingual_Data\distillation\hf_cache")
    sys.path.insert(0, r"F:\Lingual_Data\scripts")
    import distill_reasoning as dr
    src = {"gsm8k_train": set(), "mbpp_train_val_prompt": set(), "sft_v1": set()}
    for s in dr.load_seeds("math"):
        src["gsm8k_train"] |= grams(s["prompt"])
    for s in dr.load_seeds("code"):
        src["mbpp_train_val_prompt"] |= grams(s["prompt"])
    from tokenizers import Tokenizer
    import numpy as np
    sft = ROOT / "data" / "processed_sft_v1" / "train.bin"
    if sft.is_file():   # decode in 1M-token slices; prompts and answers both count
        tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))
        ids = np.fromfile(sft, np.uint16)
        for i in range(0, len(ids), 1_000_000):
            src["sft_v1"] |= grams(tok.decode(ids[i:i + 1_000_000].tolist()))
    return src


def self_test():
    src = {"a": grams("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May")}
    qs = [{"id": "dup", "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many"},
          {"id": "new", "prompt": "A train leaves the station at noon travelling sixty miles per hour toward the coast"}]
    f = flag(qs, src, 0.5)
    assert [x[0] for x in f] == ["dup"], f
    print("[self-test] check_eval_contamination ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*")
    p.add_argument("--max-share", type=float, default=0.5)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    src = load_sources()
    print({k: len(v) for k, v in src.items()})
    bad = 0
    for f in a.files:
        qs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        hits = flag(qs, src, a.max_share)
        bad += len(hits)
        print(f"{f}: {len(qs)} queries, {len(hits)} flagged")
        for h in hits:
            print("  ", *h)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
