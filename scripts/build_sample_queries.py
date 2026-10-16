#!/usr/bin/env python3
"""Pick the shareable-samples prompt set (Claude Code, 2026-10-16; Joel: 5-15 samples for every language).

Up to 10 prompts per language from the existing clean eval files (no new prompts), round-robin over each
language's buckets so continuation / comprehension / instruction / translation all show. Fixed seed.

    .venv\\Scripts\\python.exe scripts\\build_sample_queries.py            # -> eval/queries_samples.jsonl
    .venv\\Scripts\\python.exe scripts\\build_sample_queries.py --self-test
"""

import argparse
import json
import random
from collections import defaultdict

SOURCES = ("eval/queries_120m_lingual250.jsonl", "eval/queries_120m_s250.jsonl")
LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng", "hin")   # esg excluded: all rights reserved


def pick(rows: list[dict], per_lang: int = 10, seed: int = 7) -> list[dict]:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("lang") in LANGS:
            by[r["lang"]][r.get("format") or r.get("bucket") or "other"].append(r)
    rng, out = random.Random(seed), []
    for lang in LANGS:
        buckets = [sorted(v, key=lambda r: r["id"]) for _, v in sorted(by[lang].items())]
        for b in buckets:
            rng.shuffle(b)
        got = []
        while len(got) < per_lang and any(buckets):
            for b in buckets:
                if b and len(got) < per_lang:
                    got.append(b.pop())
        out += got
    return out


def self_test():
    rows = [{"id": f"k{i}", "lang": "kan", "bucket": "a" if i % 2 else "b", "prompt": "p"} for i in range(30)]
    rows += [{"id": "h1", "lang": "hin", "bucket": "x", "prompt": "p"}, {"id": "e1", "lang": "esg", "prompt": "p"}]
    got = pick(rows)
    kan = [r for r in got if r["lang"] == "kan"]
    assert len(kan) == 10 and {r["bucket"] for r in kan} == {"a", "b"}
    assert [r["id"] for r in got if r["lang"] == "hin"] == ["h1"] and not any(r["lang"] == "esg" for r in got)
    assert pick(rows) == got   # deterministic
    print("[self-test] build_sample_queries ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval/queries_samples.jsonl")
    p.add_argument("--per-lang", type=int, default=10)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    rows, seen = [], set()
    for f in SOURCES:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
    got = pick(rows, a.per_lang)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in got:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for r in got:
        counts[r["lang"]] += 1
    print(dict(counts))


if __name__ == "__main__":
    main()
