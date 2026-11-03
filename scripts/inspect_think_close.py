import json

with open(r"E:\Parentheses\parentheses-0.9\out\cmp_130m_v3\think_close.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sample_0 = data["A_sft"]["samples"][0]
print("Keys in sample:", list(sample_0.keys()))
