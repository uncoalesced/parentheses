import json, os, sys
from collections import defaultdict
import statistics

sys.stdout.reconfigure(encoding='utf-8')

with open('out/queries_120m_s250_final.jsonl', 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]

by_lang = defaultdict(list)
by_bucket = defaultdict(list)
for r in rows:
    by_lang[r['lang']].append(r)
    by_bucket[r['bucket']].append(r)

print("Total evaluated:", len(rows))

summary_lang = {}
for lang, lst in sorted(by_lang.items(), key=lambda x: -len(x[1])):
    oks = [r['script_ok'] for r in lst if r['script_ok'] is not None]
    script_acc = (sum(oks) / len(oks)) if oks else float('nan')
    eos_rate = sum(r['stop'] == 'eos' for r in lst) / len(lst)
    loop_rate = sum(r['stop'] == 'loop' for r in lst) / len(lst)
    max_tok_rate = sum(r['stop'] == 'max_tokens' for r in lst) / len(lst)
    avg_rep = statistics.mean(r['repetition_rate'] for r in lst)
    avg_d2 = statistics.mean(r['distinct_2'] for r in lst)
    avg_sec = statistics.mean(r['seconds'] for r in lst)
    avg_tokens = statistics.mean(r['n_tokens'] for r in lst)
    
    summary_lang[lang] = {
        "n": len(lst),
        "script_acc": script_acc,
        "eos_rate": eos_rate,
        "loop_rate": loop_rate,
        "max_tok_rate": max_tok_rate,
        "avg_rep": avg_rep,
        "avg_d2": avg_d2,
        "avg_sec": avg_sec,
        "avg_tokens": avg_tokens,
    }

out_data = {
    "total_queries": len(rows),
    "by_language": summary_lang,
    "overall": {
        "mean_seconds": statistics.mean(r['seconds'] for r in rows),
        "mean_tokens": statistics.mean(r['n_tokens'] for r in rows),
        "total_eos": sum(r['stop'] == 'eos' for r in rows),
        "total_loop": sum(r['stop'] == 'loop' for r in rows),
        "total_max_tokens": sum(r['stop'] == 'max_tokens' for r in rows),
    }
}

with open('out/swarm/120M_s250_eval_summary.json', 'w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2)

print("Generated out/swarm/120M_s250_eval_summary.json")
