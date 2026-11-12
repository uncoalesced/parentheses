# Engineered by uncoalesced
# Spot-check and log all c6_50 outputs for step_14000.pt and step_14200_golden.pt

import sys
sys.path.insert(0, r"E:\Parentheses\parentheses-0.9")
import json, torch
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.eval_60p_adversarial_suite import load_model, stream_slab_v4
from scripts.s10_pilot_check import check_idk, check_false_refusal

c6_path = Path(r"E:\Parentheses\parentheses-0.9\data\eval_suites\c6_50.jsonl")
with open(c6_path, "r", encoding="utf-8") as f:
    c6_prompts = [json.loads(line) for line in f if line.strip()]

models_to_check = [
    ("step_14000", Path(r"E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M-s10-blend\sft-winner.pt")),
    ("step_14200_golden", Path(r"E:\Parentheses\parentheses-0.9\checkpoints\parentheses-1.0-240M-s10-patch\step_14200_golden.pt")),
]

for label, ckpt_path in models_to_check:
    print(f"\n" + "=" * 60)
    print(f"=== Evaluating {label} on c6_50 ===")
    print("=" * 60)
    model, tokenizer, cfg = load_model(ckpt_path, device="cuda")
    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    sys_id = tokenizer.token_to_id("<|system|>")
    stops = {eot_id, user_id, sys_id} - {None}

    out_records = []
    fp_passes = 0
    fp_total = 0
    benign_refusals = 0
    benign_total = 0

    for it in c6_prompts:
        is_fp = it.get("is_false_premise", False)
        lang_id = tokenizer.token_to_id("<eng>")
        p_ids = tokenizer.encode(it["prompt"], add_special_tokens=False).ids
        in_ids = [user_id, lang_id] + p_ids + [asst_id]
        input_ids = torch.tensor([in_ids], dtype=torch.long, device="cuda")

        with torch.inference_mode():
            streamer = stream_slab_v4(model, input_ids, max_new_tokens=64, temperature=0.0)
            toks = []
            for t in streamer:
                if t in stops:
                    break
                toks.append(t)
        text = tokenizer.decode(toks, skip_special_tokens=False).strip()
        has_idk = check_idk(text)
        is_ref, _ = check_false_refusal(text)

        if is_fp:
            fp_total += 1
            if has_idk:
                fp_passes += 1
        else:
            benign_total += 1
            if is_ref:
                benign_refusals += 1

        out_records.append({
            "id": it["id"],
            "category": it.get("category"),
            "is_false_premise": is_fp,
            "prompt": it["prompt"],
            "generation": text,
            "has_idk": has_idk,
            "is_refused": is_ref,
        })

    del model
    torch.cuda.empty_cache()

    out_f = Path(r"E:\Parentheses\parentheses-0.9\out\s10") / f"c6_50_{label}_details.jsonl"
    with open(out_f, "w", encoding="utf-8") as fh:
        for r in out_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{label}: FP Abstention = {fp_passes}/{fp_total} ({fp_passes/fp_total:.1%}), Benign Refusal = {benign_refusals}/{benign_total}")
    print(f"Details written to: {out_f}")
