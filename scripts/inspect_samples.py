import json, sys
sys.stdout.reconfigure(encoding='utf-8')

sample_ids = [line.strip() for line in open('out/swarm/judge_sample_ids.txt', encoding='utf-8') if line.strip()]
sample_set = set(sample_ids[:5])

runs = ['base', 'sft', 'sft_v2']
data = {}
for r in runs:
    data[r] = {}
    with open(f'out/queries_v1_{r}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d['id'] in sample_set:
                data[r][d['id']] = d

for sid in sample_ids[:5]:
    item_base = data['base'].get(sid, {})
    print(f"=== ID: {sid} | Lang: {item_base.get('lang')} | Bucket: {item_base.get('bucket')} ===")
    print(f"Prompt: {item_base.get('prompt')}")
    print(f"Ref: {item_base.get('reference')}")
    for r in runs:
        item = data[r].get(sid, {})
        out = item.get('output', '').replace('\n', ' ')
        print(f"  [{r}] stop={item.get('stop')} n_tok={item.get('n_tokens')}: {out[:120]}...")
    print()
