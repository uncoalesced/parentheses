#!/usr/bin/env python3
"""
scripts/eval_dpo_v2_candidates.py
Runs the full evaluation battery on the two newly trained DPO v2 models:
  1. B_dpo_v2 (raw terminal base)
  2. B_dpo_ag_v2 (SWA-3 soup base)

Gates evaluated:
  Gate 1: Honest Abstention Probe (did it abstain on unknown entities?)
  Gate 2: False-Abstention / Refusal Rate (0/19 false refusals, 0 false abstentions on known facts)
  Gate 3: Spontaneous <think> Activation on 40 fresh prompts
  Gate 4: FreeThink script_ok across 72 prompts
  Gate 5: Whole-set DPO Margin (900 pairs)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"E:\Parentheses\parentheses-0.9")
PY_EXE = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
LOCK_FILE = REPO_ROOT / "out" / "locks" / "GPU.lock"

MODELS = [
    {
        "id": "B_dpo_v2",
        "label": "B_dpo_v2 (Raw Terminal Base)",
        "ckpt": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-raw" / "step_0100_final.pt"),
        "pair_eval": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-raw" / "pair_eval.json"),
        "honest_out": str(REPO_ROOT / "out" / "eval_honest_v2_raw.json"),
        "refusal_out": str(REPO_ROOT / "out" / "eval_refusal_v2_raw.json"),
        "ft_jsonl": str(REPO_ROOT / "out" / "freethink_v2_raw.jsonl"),
        "ft_report": str(REPO_ROOT / "out" / "freethink_v2_raw_report.md"),
    },
    {
        "id": "B_dpo_ag_v2",
        "label": "B_dpo_ag_v2 (SWA-3 Soup Base)",
        "ckpt": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-swa3" / "step_0100_final.pt"),
        "pair_eval": str(REPO_ROOT / "checkpoints" / "parentheses-0.9-130M-v3-dpo-v2-swa3" / "pair_eval.json"),
        "honest_out": str(REPO_ROOT / "out" / "eval_honest_v2_swa3.json"),
        "refusal_out": str(REPO_ROOT / "out" / "eval_refusal_v2_swa3.json"),
        "ft_jsonl": str(REPO_ROOT / "out" / "freethink_v2_swa3.jsonl"),
        "ft_report": str(REPO_ROOT / "out" / "freethink_v2_swa3_report.md"),
    },
]


def acquire_lock(task: str):
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(f"antigravity {task} (started {time.strftime('%Y-%m-%d %H:%M:%S')})\n", encoding="utf-8")


def release_lock():
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except Exception:
            pass


def run_cmd(cmd: list[str], desc: str):
    print(f"  [RUN] {desc}...")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"  [FAIL] {desc} (code {res.returncode}) in {dt:.1f}s:\n{res.stderr}\n{res.stdout}")
        return False
    print(f"  [DONE] {desc} in {dt:.1f}s")
    return True


def run_fresh_prompts_eval(m: dict) -> dict:
    """Evaluate spontaneous <think> activation and closure on fresh prompts."""
    # We load 40 prompts from think_close.json or evaluate with free_think directly
    # Using python script on GPU
    code = f"""
import sys, json, torch
from pathlib import Path
REPO_ROOT = Path(r"{str(REPO_ROOT)}")
sys.path.insert(0, str(REPO_ROOT))
from features.free_think import FreeThinkSession
from eval_dravidian import repetition_rate

with open(r"{str(REPO_ROOT / 'out' / 'cmp_130m_v3' / 'think_close.json')}", 'r', encoding='utf-8') as f:
    tc_data = json.load(f)

# Use prompts from B_dpo samples
samples = tc_data.get('B_dpo', {{}}).get('samples', [])
prompts = [
    "If the train leaves at 9:40 and it takes 2 hours to travel from the source to the destination, then it travels at a rate of 2 * 9 = 18 minutes per hour. What time does it arrive?",
    "A box contains 5 red balls and 3 green balls. If 2 balls are drawn at random without replacement, what is the probability that both are red?",
    "Is a square always a rectangle?",
    "Explain the relationship between electric current and magnetic field.",
    "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
    "Calculate 37 * 6.",
    "Solve for x: 3x + 15 = 42.",
    "What is the capital of Australia?",
    "Does sound travel faster in water or in air?",
    "Why does ice float on water?"
]
# Add prompts up to 40
extra_prompts = [
    "What is 15% of 240?", "If today is Tuesday, what day of the week will it be in 100 days?",
    "Why do leaves change color in autumn?", "What is the primary function of mitochondria?",
    "If A is taller than B and B is taller than C, is A taller than C?", "What is 17 * 14?",
    "What is the boiling point of ethanol?", "How many chromosomes do human cells have?",
    "Describe the process of photosynthesis.", "Can a triangle have two right angles?",
    "What is the derivative of x^3?", "What is the square root of 289?",
    "What causes ocean tides?", "Which planet has the most moons?",
    "What is the difference between speed and velocity?", "What is the chemical formula for table salt?",
    "Who proposed the theory of general relativity?", "What is 84 / 7?",
    "Explain why metals conduct electricity.", "If you roll two six-sided dice, what is the probability of rolling a sum of 7?",
    "What is the main component of natural gas?", "How does an airplane generate lift?",
    "What is 25 * 25?", "What is the speed of light in vacuum?",
    "Why do we see rainbows after rain?", "What is the hardest natural mineral?",
    "What is the atomic number of carbon?", "If 4x - 8 = 24, what is x?",
    "What is the chemical symbol for gold?", "Why is the sky blue?"
]
all_prompts = prompts + extra_prompts

sess = FreeThinkSession(ckpt_path=r"{m['ckpt']}", mode='structured', max_new_tokens=128, device='cuda')
opened = 0
closed = 0
for p in all_prompts:
    res = sess.generate(p, think=True)
    out_text = res.get('text', '')
    if '<think>' in out_text:
        opened += 1
        if '</think>' in out_text:
            closed += 1

print(json.dumps({{'n': len(all_prompts), 'opened': opened, 'closed': closed}}))
"""
    res = subprocess.run([PY_EXE, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")
    if res.returncode == 0:
        try:
            return json.loads(res.stdout.strip().splitlines()[-1])
        except Exception:
            pass
    return {"n": 40, "opened": 0, "closed": 0, "error": res.stderr}


def evaluate_model(m: dict):
    print(f"\n{'='*70}\n[EVAL] Running 5-Gate Evaluation Battery on {m['label']}\n{'='*70}")
    
    # 1. Honest suite
    run_cmd([
        PY_EXE, str(REPO_ROOT / "scripts" / "eval_honest_suite.py"),
        "--checkpoint", m["ckpt"],
        "--out-json", m["honest_out"],
        "--pairs", str(REPO_ROOT / "out" / "dpo_pairs_v2_900.jsonl")
    ], "Honest Suite (Decontaminated)")

    # 2. Refusal rate
    run_cmd([
        PY_EXE, str(REPO_ROOT / "scripts" / "eval_refusal_rate.py"),
        "--checkpoint", m["ckpt"],
        "--out-json", m["refusal_out"],
    ], "Refusal Rate Audit (19 Probes)")

    # 3. FreeThink showcase
    run_cmd([
        PY_EXE, str(REPO_ROOT / "scripts" / "eval_freethink_showcase.py"),
        "--checkpoint", m["ckpt"],
        "--mode", "both",
        "--out-jsonl", m["ft_jsonl"],
        "--out-report", m["ft_report"],
    ], "FreeThink Multilingual Showcase (72 prompts)")

    # 4. Fresh prompt <think> activation
    print("  [RUN] Fresh 40-Prompt Spontaneous <think> Activation...")
    fresh_stats = run_fresh_prompts_eval(m)
    print(f"  [DONE] Fresh Prompts: {fresh_stats.get('opened')}/{fresh_stats.get('n')} opened, {fresh_stats.get('closed')} closed")
    m["fresh_stats"] = fresh_stats


def summarize_results():
    print(f"\n{'='*80}\n DPO v2 FIVE ACCEPTANCE GATES EVALUATION SCORECARD\n{'='*80}")
    print(f"{'Gate / Metric':<35} | {'Baseline (B_dpo v1)':<20} | {'B_dpo_v2 (Raw)':<18} | {'B_dpo_ag_v2 (SWA-3)':<18}")
    print(f"{'-'*35}-+-{'-'*20}-+-{'-'*18}-+-{'-'*18}")

    rows = []
    # Collect data for both models
    data = {}
    for m in MODELS:
        m_id = m["id"]
        data[m_id] = {}
        # Gate 1 & 2: Honest suite
        if os.path.exists(m["honest_out"]):
            with open(m["honest_out"], "r", encoding="utf-8") as f:
                h = json.load(f)
            total = len(h)
            correct = sum(1 for x in h if x.get("passed"))
            arith = sum(1 for x in h if x.get("cat") == "Arithmetic" and x.get("passed"))
            ab_item = next((x for x in h if x.get("cat") == "Honest Abstention"), None)
            ab_pass = ab_item.get("passed", False) if ab_item else False
            data[m_id]["honest_total"] = f"{correct}/{total}"
            data[m_id]["arith"] = f"{arith}/6"
            data[m_id]["abstain"] = "1/1 (Abstains)" if ab_pass else "0/1 (Fabricates)"
        else:
            data[m_id]["honest_total"] = "N/A"
            data[m_id]["arith"] = "N/A"
            data[m_id]["abstain"] = "N/A"

        # Gate 2: Refusal rate
        if os.path.exists(m["refusal_out"]):
            with open(m["refusal_out"], "r", encoding="utf-8") as f:
                ref = json.load(f)
            ref_cnt = ref.get("refused_count", 0)
            data[m_id]["refusal"] = f"{ref_cnt}/19 refused"
        else:
            data[m_id]["refusal"] = "N/A"

        # Gate 3: Spontaneous think
        fs = m.get("fresh_stats", {})
        data[m_id]["fresh_think"] = f"{fs.get('opened', 0)}/{fs.get('n', 40)} ({fs.get('closed', 0)} closed)"

        # Gate 4: FreeThink script_ok
        if os.path.exists(m["ft_jsonl"]):
            ok_cnt = 0
            tot_cnt = 0
            with open(m["ft_jsonl"], "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        tot_cnt += 1
                        if json.loads(line).get("script_ok"):
                            ok_cnt += 1
            data[m_id]["script_ok"] = f"{ok_cnt}/{tot_cnt}"
        else:
            data[m_id]["script_ok"] = "N/A"

        # Gate 5: Whole-set DPO margin
        if os.path.exists(m["pair_eval"]):
            with open(m["pair_eval"], "r", encoding="utf-8") as f:
                pe = json.load(f)
            last = pe[-1] if pe else {}
            margin = last.get("mean_margin", 0.0)
            acc = last.get("acc", 0.0)
            data[m_id]["dpo_margin"] = f"+{margin:.3f} ({acc*100:.1f}%)"
        else:
            data[m_id]["dpo_margin"] = "N/A"

    b_raw = data.get("B_dpo_v2", {})
    b_swa = data.get("B_dpo_ag_v2", {})

    print(f"{'1. Honest Abstention Probe':<35} | {'0/1 (Fabricates)':<20} | {b_raw.get('abstain', 'N/A'):<18} | {b_swa.get('abstain', 'N/A'):<18}")
    print(f"{'2. Refusal Rate Audit':<35} | {'0/19 refused':<20} | {b_raw.get('refusal', 'N/A'):<18} | {b_swa.get('refusal', 'N/A'):<18}")
    print(f"{'   Honest Total (Decontaminated)':<35} | {'5/24 (20.8%)':<20} | {b_raw.get('honest_total', 'N/A'):<18} | {b_swa.get('honest_total', 'N/A'):<18}")
    print(f"{'   Arithmetic Subset':<35} | {'1/6 (16.7%)':<20} | {b_raw.get('arith', 'N/A'):<18} | {b_swa.get('arith', 'N/A'):<18}")
    print(f"{'3. Spontaneous <think> Activation':<35} | {'17/40 (42.5%)':<20} | {b_raw.get('fresh_think', 'N/A'):<18} | {b_swa.get('fresh_think', 'N/A'):<18}")
    print(f"{'4. FreeThink script_ok (72 prompts)':<35} | {'54/72 (75.0%)':<20} | {b_raw.get('script_ok', 'N/A'):<18} | {b_swa.get('script_ok', 'N/A'):<18}")
    print(f"{'5. Whole-set DPO Margin (900 pairs)':<35} | {'+0.841 (97.9%)':<20} | {b_raw.get('dpo_margin', 'N/A'):<18} | {b_swa.get('dpo_margin', 'N/A'):<18}")
    print("=" * 80)


def main():
    acquire_lock("eval-dpo-v2-candidates")
    try:
        for m in MODELS:
            evaluate_model(m)
        summarize_results()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
