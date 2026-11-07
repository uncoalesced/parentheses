#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/grade_rubric_900.py

Automated 3-Axis Strict Scoring & Multi-Session Comparative Scorer (Score out of 900)
Evaluates 60 probes across 6 domains (10 probes/domain):
  - Domain 1: Python Coding & Algorithmic Logic (10)
  - Domain 2: Arithmetic & Multi-Step Logic (10)
  - Domain 3: Major Dravidian Fluency (10)
  - Domain 4: Low-Resource Dravidian (10)
  - Domain 5: English Conceptual Reasoning (10)
  - Domain 6: Epistemic Abstention & Tag Closures (10)

Rubric:
  - Fluency (F): [0-5]
  - Logical Validity (L): [0-5]
  - Task Adherence (T): [0-5]
  Max probe score = 15. Max domain score = 150. Total benchmark score = 900.

Usage:
  python scripts/grade_rubric_900.py --runs s2=out/eval_s2_60p_results.jsonl s3=out/eval_s4/s3_chat_greedy_256.jsonl s4=out/eval_s4/s4_chat_greedy_256.jsonl s5=out/eval_s5/s5_chat_greedy.jsonl s6=out/eval_s6_12k/s6_12000_chat_greedy.jsonl s7_5=out/eval_s7_5/s7_5_chat_greedy.jsonl s8=out/eval_loop_900/s8_golden_chat_256.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_60p_adversarial_suite import PROBES
from scripts.score_60p import CODE_TESTS, KEYWORDS, MATH_FIRST_NUMBER, MATH_REGEX


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


def test_code_exec(probe_id: str, code_str: str) -> bool:
    test_suite = CODE_TESTS.get(probe_id)
    if not test_suite:
        return False
    
    # Clean markdown
    clean = re.sub(r"```python\s*", "", code_str)
    clean = re.sub(r"```\s*", "", clean)
    full_prog = f"{clean}\n\n{test_suite}"
    
    try:
        # Check syntax first
        ast.parse(clean)
    except Exception:
        return False

    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_prog)
        f_name = f.name
    try:
        res = subprocess.run([sys.executable, f_name], capture_output=True, timeout=3)
        return res.returncode == 0
    except Exception:
        return False
    finally:
        try:
            Path(f_name).unlink(missing_ok=True)
        except Exception:
            pass


def grade_probe(probe_id: str, domain: str, gen: str, p_info: Dict[str, Any]) -> Tuple[int, int, int]:
    """Returns (F, L, T) each in [0, 5]."""
    text = gen.strip()
    if not text:
        return (0, 0, 0)

    has_loop = check_ngram_loops(text, n=4, max_repeats=3)
    has_subword_loop = bool(re.search(r"(\S{2,})\s*(?:\1\s*){4,}", text))
    
    # --- 1. Fluency (0-5) ---
    if has_subword_loop or len(set(text.split())) <= 3:
        f_score = 0
    elif has_loop:
        f_score = 1
    elif len(text.split()) < 5:
        f_score = 2
    else:
        # Check script / grammar coherence
        f_score = 4
        # Deduct if excessive unclosed syntax or random tokens
        if text.count("(") != text.count(")") or text.count("{") != text.count("}"):
            f_score = max(2, f_score - 1)
        if len(text.split()) > 20 and not has_loop:
            f_score = min(5, f_score + 1)

    # --- 2. Logical Validity (0-5) & 3. Task Adherence (0-5) ---
    l_score = 0
    t_score = 0

    if "Coding" in domain:
        passes_code = test_code_exec(probe_id, text)
        has_def = "def " in text or "class " in text
        if passes_code:
            l_score = 5
            t_score = 5
            f_score = max(4, f_score)
        elif has_def:
            l_score = 2
            t_score = 3
            # Check if key algorithm logic appears
            if any(k in text.lower() for k in ["for ", "while ", "return ", "yield "]):
                l_score = 3
        else:
            l_score = 1 if not has_loop else 0
            t_score = 1 if not has_loop else 0

    elif "Arithmetic" in domain:
        # Check if math solution matches ground truth
        matched = False
        target_nums = MATH_FIRST_NUMBER.get(probe_id, set())
        # Check boxed
        boxed_m = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed_m:
            for b in boxed_m:
                b_clean = re.sub(r"[^\d.-]", "", b)
                try:
                    if float(b_clean) in target_nums:
                        matched = True
                        break
                except Exception:
                    pass

        if not matched and target_nums:
            # Check first number after think or in text
            t_after = text
            if "</think>" in text:
                t_after = text.split("</think>")[-1]
            nums = re.findall(r"[-+]?\d*\.?\d+", t_after)
            if nums:
                try:
                    if float(nums[0]) in target_nums:
                        matched = True
                except Exception:
                    pass

        regex_rules = MATH_REGEX.get(probe_id, [])
        for rx in regex_rules:
            if re.search(rx, text, re.IGNORECASE):
                matched = True
                break

        if matched:
            l_score = 5
            t_score = 5
            f_score = max(4, f_score)
        else:
            # Did it attempt reasoning?
            has_think = "<think>" in text and "</think>" in text
            has_eq = "=" in text or "+" in text or "*" in text or "-" in text
            if has_think:
                l_score = 3
                t_score = 3
            elif has_eq and not has_loop:
                l_score = 2
                t_score = 2
            else:
                l_score = 1 if not has_loop else 0
                t_score = 1 if not has_loop else 0

    elif "Major Dravidian" in domain or "Low-Resource Dravidian" in domain:
        kw_list = KEYWORDS.get(probe_id, [])
        found_kw = any(k.lower() in text.lower() for k in kw_list)
        
        # Check script presence
        has_indic = any(0x0900 <= ord(c) <= 0x0DFF for c in text)
        
        if found_kw:
            l_score = 4
            t_score = 4
            f_score = max(4, f_score)
        elif has_indic and not has_loop:
            l_score = 2
            t_score = 3
            f_score = max(3, f_score)
        else:
            l_score = 1 if not has_loop else 0
            t_score = 1 if not has_loop else 0

    elif "English" in domain:
        kw_list = KEYWORDS.get(probe_id, [])
        found_kw = any(k.lower() in text.lower() for k in kw_list)
        if found_kw:
            l_score = 4
            t_score = 4
            f_score = max(4, f_score)
        elif not has_loop and len(text.split()) > 10:
            l_score = 2
            t_score = 2
        else:
            l_score = 1 if not has_loop else 0
            t_score = 1 if not has_loop else 0

    elif "Abstention" in domain:
        kw_list = KEYWORDS.get(probe_id, [])
        found_kw = any(k.lower() in text.lower() for k in kw_list)
        
        # Special check for structured tag closure probes
        if probe_id == "abst-06":
            if "<think>" in text and "</think>" in text and ("85" in text or "95" in text):
                found_kw = True
        elif probe_id == "abst-07":
            if "∴" in text:
                found_kw = True

        if found_kw:
            l_score = 5
            t_score = 5
            f_score = max(4, f_score)
        else:
            # Check if it avoided hallucinating an agreement
            idk_words = ["not real", "no such", "cannot", "fictional", "unknown", "does not exist"]
            if any(w in text.lower() for w in idk_words):
                l_score = 3
                t_score = 3
            elif not has_loop:
                l_score = 1
                t_score = 1
            else:
                l_score = 0
                t_score = 0

    return (f_score, l_score, t_score)


def score_run(jsonl_path: Path) -> Dict[str, Any]:
    probe_map = {p["id"]: p for p in PROBES}
    items = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r.get("id")
            if not pid or pid not in probe_map:
                continue
            gen = r.get("generation", "")
            p_info = probe_map[pid]
            dom = p_info.get("domain", "Other")
            
            f_s, l_s, t_s = grade_probe(pid, dom, gen, p_info)
            tot = f_s + l_s + t_s
            items.append({
                "id": pid,
                "domain": dom,
                "F": f_s,
                "L": l_s,
                "T": t_s,
                "total": tot,
            })
            
    domain_order = [
        "Coding",
        "Arithmetic & Logic",
        "Major Dravidian",
        "Low-Resource Dravidian",
        "English Conceptual",
        "Epistemic Abstention",
    ]
    
    dom_stats = {d: {"F": 0, "L": 0, "T": 0, "total": 0, "count": 0} for d in domain_order}
    total_f = 0
    total_l = 0
    total_t = 0
    grand_total = 0
    
    for it in items:
        raw_d = it["domain"]
        if "Coding" in raw_d:
            d = "Coding"
        elif "Arithmetic" in raw_d:
            d = "Arithmetic & Logic"
        elif "Major Dravidian" in raw_d:
            d = "Major Dravidian"
        elif "Low-Resource" in raw_d:
            d = "Low-Resource Dravidian"
        elif "English" in raw_d:
            d = "English Conceptual"
        elif "Abstention" in raw_d:
            d = "Epistemic Abstention"
        else:
            d = raw_d

        if d in dom_stats:
            dom_stats[d]["F"] += it["F"]
            dom_stats[d]["L"] += it["L"]
            dom_stats[d]["T"] += it["T"]
            dom_stats[d]["total"] += it["total"]
            dom_stats[d]["count"] += 1
        total_f += it["F"]
        total_l += it["L"]
        total_t += it["T"]
        grand_total += it["total"]
        
    return {
        "file": str(jsonl_path),
        "total_score": grand_total,
        "max_score": 900,
        "pct": round((grand_total / 900.0) * 100.0, 2),
        "F": total_f,
        "L": total_l,
        "T": total_t,
        "domains": dom_stats,
    }


def main():
    ap = argparse.ArgumentParser(description="Strict 3-Axis Rubric Scorer (/900)")
    ap.add_argument("--runs", nargs="+", required=True, help="name=path.jsonl")
    ap.add_argument("--out-json", type=Path, help="JSON output path")
    args = ap.parse_args()
    
    results = {}
    for r in args.runs:
        name, p_str = r.split("=", 1)
        p = Path(p_str.strip())
        if not p.exists():
            print(f"[WARN] File not found: {p}")
            continue
        print(f"[*] Scoring {name} ({p})...")
        results[name.strip()] = score_run(p)
        
    print("\n" + "=" * 80)
    print("PARENT-1.0-240M: MULTI-SESSION 60-PROBE COMPARATIVE RUBRIC SCORECARD (/900)")
    print("=" * 80)
    
    header = f"{'Session':<12} | {'Total (/900)':<14} | {'Pct':<8} | {'Fluency':<10} | {'Logic':<10} | {'Task Adh':<10}"
    print(header)
    print("-" * len(header))
    
    for name, res in results.items():
        print(f"{name:<12} | {res['total_score']:>4} / 900     | {res['pct']:>6.2f}% | {res['F']:>3} / 300   | {res['L']:>3} / 300 | {res['T']:>3} / 300")
        
    print("\n" + "=" * 80)
    print("DOMAIN BREAKDOWN PER SESSION (/150 per domain)")
    print("=" * 80)
    domain_order = [
        "Coding",
        "Arithmetic & Logic",
        "Major Dravidian",
        "Low-Resource Dravidian",
        "English Conceptual",
        "Epistemic Abstention",
    ]
    
    dom_header = f"{'Session':<12} | " + " | ".join(f"{d[:8]:<8}" for d in domain_order)
    print(dom_header)
    print("-" * len(dom_header))
    for name, res in results.items():
        row_str = f"{name:<12} | " + " | ".join(f"{res['domains'][d]['total']:>3}/150   " for d in domain_order)
        print(row_str)
        
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Detailed scorecard written to {args.out_json}")

if __name__ == "__main__":
    main()
