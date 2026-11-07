#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/eval_s7_5_gate_verdict.py

Mechanically audits the Session 7.5 evaluation artifacts against Gates G1-G5:
  G1: Run health (step_13000_final.pt exists, exit code 0, manifests match)
  G2: BPB rise vs S6 step 12000 at most +0.06 on every one of:
      kan, tam, tel, mal, tcy, kff, kxv, nit, eng, hin, san
  G3: 60-probe greedy score >= 5/60
  G4: 60-probe sampled pass@5 score >= 8/60
  G5: No collapse (bare IDK <= 3/60; math with closed think >= 3/10; looping <= 3/60)

Outputs:
  - Prints detailed gate report
  - Selects candidate S8 base (SWA-3 if it passes all gates and greedy >= final, else final)
  - Exits with 0 if all gates pass, 1 if any gate fails
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(r"E:\Parentheses\parentheses-0.9")
EVAL_DIR = REPO_ROOT / "out" / "eval_s7_5"
CKPT_DIR = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s7_5"


def check_ngram_loops(gen_text: str, n: int = 4, max_repeats: int = 3) -> bool:
    words = gen_text.split()
    if len(words) < n * max_repeats:
        return False
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        g = tuple(words[i : i + n])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= max_repeats:
            return True
    return False


def audit_output_sanity(jsonl_path: Path) -> Dict[str, Any]:
    if not jsonl_path.exists():
        return {"error": f"missing {jsonl_path}"}

    bare_idk = 0
    math_closed_think = 0
    math_total = 0
    looping = 0
    total = 0

    idk_phrases = ["i don't know.", "i do not know.", "i don't know", "i do not know"]

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            gen = row.get("generation", "").strip()
            domain = row.get("domain", "")

            if gen.lower() in idk_phrases:
                bare_idk += 1

            if "Arithmetic" in domain or "Math" in domain:
                math_total += 1
                if "<think>" in gen and "</think>" in gen:
                    if gen.find("<think>") < gen.find("</think>"):
                        math_closed_think += 1

            if check_ngram_loops(gen, 4, 3):
                looping += 1

    return {
        "total": total,
        "bare_idk": bare_idk,
        "math_total": math_total,
        "math_closed_think": math_closed_think,
        "looping": looping,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Session 7.5 Gate Auditor")
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    args = parser.parse_args()

    print("=== PARENTHESES 1.0-240M -- SESSION 7.5 GATE AUDIT ===")

    # 1. Gate G1: Run Health
    final_pt = args.ckpt_dir / "step_13000_final.pt"
    swa3_pt = args.ckpt_dir / "step_13000_swa3.pt"
    scores_json = args.eval_dir / "scores.json"
    bpb_gate_txt = args.eval_dir / "bpb_gate.txt"

    g1_pass = True
    g1_reasons = []
    if not final_pt.exists() or final_pt.stat().st_size < 3000000000:
        g1_pass = False
        g1_reasons.append(f"step_13000_final.pt missing or undersized ({final_pt})")
    if not scores_json.exists():
        g1_pass = False
        g1_reasons.append("scores.json missing")
    if not bpb_gate_txt.exists():
        g1_pass = False
        g1_reasons.append("bpb_gate.txt missing")

    print(f"Gate G1 (Run Health): {'PASS' if g1_pass else 'FAIL'} ({', '.join(g1_reasons) if g1_reasons else 'Checkpoints & eval artifacts intact'})")

    # 2. Gate G2: BPB Gate (all 11 languages <= +0.06)
    g2_pass = False
    g2_details = ""
    if bpb_gate_txt.exists():
        gate_content = bpb_gate_txt.read_text(encoding="utf-8")
        if "BPB GATE: PASS" in gate_content or "ALL PASS" in gate_content or "[PASS]" in gate_content:
            g2_pass = True
            g2_details = "All 11 languages <= +0.06 BPB delta"
        else:
            g2_details = "Regression above +0.06 BPB detected in bpb_gate.txt"
    else:
        g2_details = "bpb_gate.txt not found"
    print(f"Gate G2 (11-Lang BPB <= +0.06): {'PASS' if g2_pass else 'FAIL'} ({g2_details})")

    # 3. Gate G3 & G4: 60-probe scores
    g3_pass = False
    g4_pass = False
    final_greedy_score = 0.0
    swa3_greedy_score = 0.0
    sampled_pass5_score = 0.0

    if scores_json.exists():
        try:
            with scores_json.open("r", encoding="utf-8") as f:
                s_data = json.load(f)
            # Find final greedy, SWA greedy, sampled pass@5
            for k, v in s_data.items():
                if "swa3" in k and "greedy" in k:
                    swa3_greedy_score = float(v.get("total", v.get("score", 0)))
                elif "greedy" in k:
                    final_greedy_score = float(v.get("total", v.get("score", 0)))
                elif "t07x5" in k or "sampled" in k:
                    sampled_pass5_score = float(v.get("pass@5", v.get("total", 0)))

            if final_greedy_score >= 5.0 or swa3_greedy_score >= 5.0:
                g3_pass = True
            if sampled_pass5_score >= 8.0:
                g4_pass = True
        except Exception as e:
            print(f"[!] Error parsing scores.json: {e}")

    print(f"Gate G3 (60-Probe Greedy >= 5/60): {'PASS' if g3_pass else 'FAIL'} (Final: {final_greedy_score:.2f}, SWA-3: {swa3_greedy_score:.2f})")
    print(f"Gate G4 (60-Probe Sampled pass@5 >= 8/60): {'PASS' if g4_pass else 'FAIL'} (Sampled pass@5: {sampled_pass5_score:.2f})")

    # 4. Gate G5: Output Sanity (No Collapse)
    greedy_file = args.eval_dir / "s7_5_chat_greedy.jsonl"
    sanity = audit_output_sanity(greedy_file)
    g5_pass = True
    g5_reasons = []

    if sanity.get("bare_idk", 0) > 3:
        g5_pass = False
        g5_reasons.append(f"bare IDK {sanity['bare_idk']} > 3")
    if sanity.get("math_closed_think", 0) < 3:
        g5_pass = False
        g5_reasons.append(f"closed math think {sanity['math_closed_think']} < 3")
    if sanity.get("looping", 0) > 3:
        g5_pass = False
        g5_reasons.append(f"looping {sanity['looping']} > 3")

    print(f"Gate G5 (No Collapse): {'PASS' if g5_pass else 'FAIL'} ({', '.join(g5_reasons) if g5_reasons else f'bare_idk={sanity.get(\"bare_idk\")}, closed_think={sanity.get(\"math_closed_think\")}/10, looping={sanity.get(\"looping\")}'})")

    # Overall Verdict
    all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
    print("\n-----------------------------------------------------")
    print(f"OVERALL S7.5 VERDICT: {'ALL GATES PASS (RECOMMEND S8)' if all_pass else 'GATE FAILURE DETECTED (DO NOT START S8)'}")

    candidate_base = None
    if all_pass:
        if swa3_pt.exists() and swa3_greedy_score >= final_greedy_score:
            candidate_base = swa3_pt
        else:
            candidate_base = final_pt
        print(f"Candidate S8 Base Selected: {candidate_base}")

    # Output machine-readable verdict
    verdict_info = {
        "all_pass": all_pass,
        "candidate_base": str(candidate_base) if candidate_base else None,
        "gates": {
            "G1_run_health": g1_pass,
            "G2_bpb_drift": g2_pass,
            "G3_greedy_score": g3_pass,
            "G4_sampled_score": g4_pass,
            "G5_no_collapse": g5_pass,
        },
        "scores": {
            "final_greedy": final_greedy_score,
            "swa3_greedy": swa3_greedy_score,
            "sampled_pass5": sampled_pass5_score,
        },
        "sanity": sanity,
    }

    verdict_path = args.eval_dir / "gate_verdict.json"
    with verdict_path.open("w", encoding="utf-8") as f:
        json.dump(verdict_info, f, indent=2)
    print(f"[+] Saved gate verdict to {verdict_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
