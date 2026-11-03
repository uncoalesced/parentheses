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


def norm_words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def prompt_overlap(queries: list[dict], prompts: list[str], max_share: float = 0.5) -> list[tuple]:
    """Flag queries that repeat a training prompt (DPO/SFT pairs are short, so the 8-gram check misses them).

    A query is flagged on an exact word-normalized match, or when >= max_share of its word n-grams
    (n = min(4, query length)) occur in a single training prompt. Returns (id, matched prompt, share).
    """
    exact = {" ".join(norm_words(p)): p for p in prompts}
    out = []
    for q in queries:
        w = norm_words(q["prompt"])
        if not w:
            continue
        key = " ".join(w)
        if key in exact:
            out.append((q["id"], exact[key], 1.0))
            continue
        # a whole training prompt (3+ words) inside the query, or the query inside one
        inside = next((p for k, p in exact.items() if len(k.split()) >= 3 and
                       (f" {k} " in f" {key} " or (len(w) >= 3 and f" {key} " in f" {k} "))), None)
        if inside is not None:
            out.append((q["id"], inside, 1.0))
            continue
        n = min(4, len(w))
        g = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
        best, best_p = 0.0, None
        for p in prompts:
            pw = norm_words(p)
            pg = {" ".join(pw[i:i + n]) for i in range(len(pw) - n + 1)}
            share = len(g & pg) / len(g)
            if share > best:
                best, best_p = share, p
        if best >= max_share:
            out.append((q["id"], best_p, round(best, 2)))
    return out


def load_pair_prompts(paths: list[str]) -> list[str]:
    ps = []
    for f in paths:
        ps += [json.loads(l).get("prompt", "") for l in open(f, encoding="utf-8") if l.strip()]
    return [p for p in ps if p]


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
    pairs = ["What is 17 * 6? Answer with the number.", "Who created you?",
             "A farmer has 17 sheep. All but nine run away. How many remain?"]
    qs = [{"id": "exact", "prompt": "who created  you"},
          {"id": "near", "prompt": "A farmer has 17 sheep and all but nine run away; how many remain now?"},
          {"id": "short-new", "prompt": "What is 9 squared?"},
          {"id": "contains", "prompt": "What is your name, and who created you?"},
          {"id": "new", "prompt": "Tom is taller than Ana, and Ana is taller than Ravi. Who is shortest?"}]
    f = prompt_overlap(qs, pairs)
    assert [x[0] for x in f] == ["exact", "near", "contains"], f
    print("[self-test] check_eval_contamination ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*")
    p.add_argument("--max-share", type=float, default=0.5)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dpo-pairs", nargs="+", default=None,
                   help="only check queries against these pair JSONLs' prompts (exact + short n-gram)")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if a.dpo_pairs:
        prompts = load_pair_prompts(a.dpo_pairs)
        bad = 0
        for f in a.files:
            qs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            hits = prompt_overlap(qs, prompts, a.max_share)
            bad += len(hits)
            print(f"{f}: {len(qs)} queries vs {len(prompts)} pair prompts, {len(hits)} flagged")
            for h in hits:
                print("  ", *h)
        return 1 if bad else 0
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
