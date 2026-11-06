import json
from pathlib import Path

results_file = Path(r"E:\Parentheses\parentheses-0.9\out\eval_s3_60p_results.jsonl")
if not results_file.exists():
    print("Results file not found")
    exit(0)

with open(results_file, "r", encoding="utf-8") as f:
    recs = [json.loads(l) for l in f if l.strip()]

print(f"Total evaluated probes: {len(recs)}")
domains = {}
for r in recs:
    d = r.get("domain", "Unknown")
    domains.setdefault(d, []).append(r)

for d, items in sorted(domains.items()):
    print(f"\n==========================================")
    print(f"Domain: {d} ({len(items)} probes)")
    print(f"==========================================")
    for it in items[:3]:
        pid = it.get("probe_id")
        cat = it.get("category")
        prompt = it.get("prompt", "")[:70]
        gen = it.get("generation", "")[:120].replace("\n", " ")
        print(f"[{pid}] {cat}")
        print(f"  P: {prompt}...")
        print(f"  R: {gen}...")
