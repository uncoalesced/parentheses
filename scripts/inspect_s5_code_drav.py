import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

greedy_path = Path("E:/Parentheses/parentheses-0.9/out/eval_s5/s5_chat_greedy.jsonl")
t07_path = Path("E:/Parentheses/parentheses-0.9/out/eval_s5/s5_chat_t07x5.jsonl")

def load_probes(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

greedy_rows = load_probes(greedy_path)
t07_rows = load_probes(t07_path)

print("="*80)
print("INSPECTING S5 CODE PROBES (code-01 to code-10)")
print("="*80)

code_ids = [f"code-{i:02d}" for i in range(1, 11)]

for cid in code_ids:
    print(f"\n--- PROBE {cid} ---")
    g_match = [r for r in greedy_rows if r.get("id") == cid]
    if g_match:
        gm = g_match[0]
        prompt = gm.get("prompt", "")
        gen = gm.get("output", gm.get("generation", ""))
        print(f"PROMPT:\n{prompt}")
        print(f"\nGREEDY OUTPUT:\n{gen}")
    else:
        print("No greedy match found.")
    
    t_matches = [r for r in t07_rows if r.get("id") == cid]
    print(f"\nSAMPLED SAMPLES COUNT: {len(t_matches)}")
    for idx, tm in enumerate(t_matches[:2]): # show first 2 samples
        gen_s = tm.get("output", tm.get("generation", ""))
        print(f"Sample {idx}:\n{gen_s[:300]}...")

print("\n" + "="*80)
print("INSPECTING S5 DRAVIDIAN MAJOR PROBES (drav-maj-01 to drav-maj-10)")
print("="*80)

drav_ids = [f"drav-maj-{i:02d}" for i in range(1, 11)]

for did in drav_ids:
    print(f"\n--- PROBE {did} ---")
    g_match = [r for r in greedy_rows if r.get("id") == did]
    if g_match:
        gm = g_match[0]
        prompt = gm.get("prompt", "")
        gen = gm.get("output", gm.get("generation", ""))
        print(f"PROMPT:\n{prompt}")
        print(f"\nGREEDY OUTPUT:\n{gen}")
    else:
        print("No greedy match found.")
    
    t_matches = [r for r in t07_rows if r.get("id") == did]
    print(f"\nSAMPLED SAMPLES COUNT: {len(t_matches)}")
    for idx, tm in enumerate(t_matches[:2]):
        gen_s = tm.get("output", tm.get("generation", ""))
        print(f"Sample {idx}:\n{gen_s[:300]}...")
