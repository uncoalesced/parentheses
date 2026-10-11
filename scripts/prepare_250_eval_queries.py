import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("E:/Parentheses/parentheses-0.9")
SRC = ROOT / "eval/queries_v1.jsonl"
DST = ROOT / "eval/queries_120m_s250.jsonl"

TARGETS = {
    "eng": 100,
    "kan": 42,
    "tam": 30,
    "tel": 30,
    "mal": 30,
    "tcy": 6,
    "hin": 4,
    "kff": 3,
    "kxv": 3,
    "nit": 2,
}

def sample_queries():
    with open(SRC, "r", encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f if line.strip()]

    by_lang = defaultdict(lambda: defaultdict(list))
    for q in all_queries:
        by_lang[q["lang"]][q["bucket"]].append(q)

    rng = random.Random(1234)
    selected = []

    for lang, n_target in TARGETS.items():
        buckets = by_lang[lang]
        total_avail = sum(len(lst) for lst in buckets.values())
        assert total_avail >= n_target, f"Not enough queries for {lang}: {total_avail} < {n_target}"

        # Stratified sampling across buckets
        lang_selected = []
        bucket_names = sorted(buckets.keys())
        
        # Allocate per bucket
        allocations = {}
        rem = n_target
        for b in bucket_names:
            alloc = int(round(len(buckets[b]) / total_avail * n_target))
            alloc = max(1, min(alloc, len(buckets[b])))
            allocations[b] = alloc
        
        # Adjust rounding
        curr_total = sum(allocations.values())
        while curr_total > n_target:
            max_b = max(bucket_names, key=lambda b: allocations[b])
            allocations[max_b] -= 1
            curr_total -= 1
        while curr_total < n_target:
            min_b = min(bucket_names, key=lambda b: len(buckets[b]) - allocations[b])
            allocations[min_b] += 1
            curr_total += 1

        for b in bucket_names:
            k = allocations[b]
            items = list(buckets[b])
            rng.shuffle(items)
            lang_selected.extend(items[:k])

        assert len(lang_selected) == n_target
        selected.extend(lang_selected)

    assert len(selected) == 250, f"Expected 250, got {len(selected)}"
    
    with open(DST, "w", encoding="utf-8") as f:
        for q in selected:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(selected)} queries to {DST}")
    counts = defaultdict(int)
    for q in selected:
        counts[q["lang"]] += 1
    for lang, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count} queries")

if __name__ == "__main__":
    sample_queries()
