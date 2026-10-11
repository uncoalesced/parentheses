import sys, json
sys.stdout.reconfigure(encoding='utf-8')

with open('out/queries_120m_s250_final.jsonl', 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]

print(f"Total rows evaluated: {len(rows)}")
langs_sampled = ['eng', 'kan', 'tel', 'tam', 'mal', 'tcy', 'kff', 'kxv', 'nit', 'hin']

for lang in langs_sampled:
    matching = [r for r in rows if r['lang'] == lang]
    if matching:
        m = matching[0]
        p = m['prompt'][:70].replace('\n', ' ')
        out = m['output'][:140].replace('\n', ' ')
        print(f"[{lang.upper()} ({m['bucket']})] Prompt: {p}...")
        print(f"   Output: {out}...")
        print(f"   Stats: stop={m['stop']}, n_tok={m['n_tokens']}, rep={m['repetition_rate']:.3f}, d2={m['distinct_2']:.2f}\n")
