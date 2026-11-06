#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Blind evaluation rubric sheet generator and score consolidator.

Used for double-blind human/LLM rubric grading (Opus / Claude / human):
  1. Build a randomized, blind evaluation sheet across multiple runs:
     .venv\\Scripts\\python.exe scripts\\rubric_blind.py --build-sheet \\
         --runs s4=out/eval_s4/s4_chat_greedy_256.jsonl s5=out/eval_s5/s5_chat_greedy.jsonl s6=out/eval_s6/s6_chat_greedy.jsonl \\
         --sheet-out out/eval_s6/rubric_blind_sheet.md \\
         --key-out out/eval_s6/rubric_blind_key.json

  2. Unblind and summarize grades after grading is complete:
     .venv\\Scripts\\python.exe scripts\\rubric_blind.py --unblind \\
         --grades out/eval_s6/rubric_blind.jsonl \\
         --key out/eval_s6/rubric_blind_key.json \\
         --summary-out out/eval_s6/rubric_blind_summary.txt

  3. Run self-test:
     .venv\\Scripts\\python.exe scripts\\rubric_blind.py --self-test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.eval_60p_adversarial_suite import PROBES
except ImportError:
    PROBES = []


def build_sheet(runs: dict[str, Path], sheet_out: Path, key_out: Path, seed: int = 1234) -> None:
    """Read run files, shuffle probe items across runs with blind IDs B000..B{N-1}, write sheet and key."""
    probe_map = {p["id"]: p for p in PROBES}
    items = []

    for run_name, path in runs.items():
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec["id"]
                gen = rec.get("generation", "")
                p_info = probe_map.get(pid, {})
                items.append({
                    "run": run_name,
                    "id": pid,
                    "prompt": rec.get("prompt", p_info.get("prompt", "")),
                    "truth": p_info.get("ground_truth", ""),
                    "criteria": p_info.get("criteria", ""),
                    "generation": gen,
                })

    rng = random.Random(seed)
    rng.shuffle(items)

    key_map = {}
    lines = []

    for idx, item in enumerate(items):
        bid = f"B{idx:03d}"
        key_map[bid] = item["run"]
        truth_desc = item["truth"]
        if item["criteria"]:
            truth_desc = f"{truth_desc} | {item['criteria']}"

        lines.append(f"### {bid} | {item['id']}")
        lines.append(f"PROMPT: {item['prompt'].strip()}")
        lines.append(f"TRUTH: {truth_desc.strip()}")
        lines.append("OUTPUT:")
        lines.append(item["generation"].strip())
        lines.append("")

    sheet_out.parent.mkdir(parents=True, exist_ok=True)
    key_out.parent.mkdir(parents=True, exist_ok=True)

    sheet_out.write_text("\n".join(lines), encoding="utf-8")
    with open(key_out, "w", encoding="utf-8") as kf:
        json.dump(key_map, kf, indent=2)

    print(f"[OK] Generated blind sheet with {len(items)} items -> {sheet_out}")
    print(f"[OK] Saved blind key -> {key_out}")


def domain_of(pid: str) -> str:
    prefix = pid.rsplit("-", 1)[0]
    return {
        "code": "Coding",
        "math": "Arithmetic",
        "drav-maj": "Major Dravidian",
        "drav-low": "Low-Resource Dravidian",
        "eng": "English Reasoning",
        "abst": "Abstention",
    }.get(prefix, "Other")


def unblind_summary(grades_path: Path, key_path: Path, summary_out: Path | None = None) -> dict:
    """Consolidate graded rubric JSONL using key JSON, computing totals, F/L/T sums, and domain breakdowns."""
    if not grades_path.exists():
        raise FileNotFoundError(f"Grades file not found: {grades_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"Key file not found: {key_path}")

    with open(key_path, "r", encoding="utf-8") as kf:
        key_map = json.load(kf)

    run_stats = defaultdict(lambda: {
        "F": 0, "L": 0, "T": 0, "total": 0, "count": 0,
        "domains": defaultdict(int),
    })

    with open(grades_path, "r", encoding="utf-8") as gf:
        for line in gf:
            if not line.strip():
                continue
            rec = json.loads(line)
            bid = rec["bid"]
            pid = rec.get("id", "")
            run = key_map.get(bid)
            if not run:
                continue

            f_score = int(rec.get("F", 0))
            l_score = int(rec.get("L", 0))
            t_score = int(rec.get("T", 0))
            tot = int(rec.get("total", f_score + l_score + t_score))

            st = run_stats[run]
            st["F"] += f_score
            st["L"] += l_score
            st["T"] += t_score
            st["total"] += tot
            st["count"] += 1
            dom = domain_of(pid)
            st["domains"][dom] += tot

    summary_lines = []
    dom_order = ["Coding", "Arithmetic", "Major Dravidian", "Low-Resource Dravidian", "English Reasoning", "Abstention"]

    for run in sorted(run_stats.keys()):
        st = run_stats[run]
        max_possible = st["count"] * 5
        pct = (st["total"] / max(1, max_possible)) * 100.0
        dom_str = "  ".join(f"{d} {st['domains'].get(d, 0)}" for d in dom_order)
        line = f"{run}: total {st['total']}/{max_possible} ({pct:.1f}%)  F/L/T {st['F']}/{st['L']}/{st['T']}  | {dom_str}"
        summary_lines.append(line)
        print(line)

    summary_text = "\n".join(summary_lines)
    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(summary_text + "\n", encoding="utf-8")
        print(f"[OK] Saved summary -> {summary_out}")

    return {k: dict(v) for k, v in run_stats.items()}


def self_test() -> None:
    test_key = {"B000": "s4", "B001": "s5"}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        kp = Path(td) / "key.json"
        gp = Path(td) / "grades.jsonl"
        sp = Path(td) / "summary.txt"
        kp.write_text(json.dumps(test_key), encoding="utf-8")
        grades = [
            {"bid": "B000", "id": "code-01", "F": 2, "L": 2, "T": 1, "total": 5, "note": "ok"},
            {"bid": "B001", "id": "math-01", "F": 1, "L": 0, "T": 0, "total": 1, "note": "wrong"},
        ]
        gp.write_text("\n".join(json.dumps(g) for g in grades), encoding="utf-8")
        res = unblind_summary(gp, kp, sp)
        assert res["s4"]["total"] == 5
        assert res["s5"]["total"] == 1
        assert sp.exists()
    print("rubric_blind self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser(description="Blind rubric sheet generator & score unblinder")
    ap.add_argument("--build-sheet", action="store_true", help="Generate blind sheet MD and key JSON")
    ap.add_argument("--runs", nargs="+", help="Pairs of run_name=path.jsonl (e.g. s4=path/s4.jsonl s5=path/s5.jsonl)")
    ap.add_argument("--sheet-out", type=Path, default=Path("out/rubric_blind_sheet.md"))
    ap.add_argument("--key-out", type=Path, default=Path("out/rubric_blind_key.json"))
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--unblind", action="store_true", help="Unblind grades and generate summary")
    ap.add_argument("--grades", type=Path, help="Graded rubric JSONL")
    ap.add_argument("--key", type=Path, help="Key JSON mapping Bxxx -> run_name")
    ap.add_argument("--summary-out", type=Path, help="Path to write summary TXT")

    ap.add_argument("--self-test", action="store_true", help="Run internal self-test")

    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.build_sheet:
        if not args.runs:
            print("Error: --runs required for --build-sheet")
            return 1
        run_dict = {}
        for r in args.runs:
            name, path = r.split("=", 1)
            run_dict[name.strip()] = Path(path.strip())
        build_sheet(run_dict, args.sheet_out, args.key_out, seed=args.seed)
        return 0

    if args.unblind:
        if not args.grades or not args.key:
            print("Error: --grades and --key required for --unblind")
            return 1
        unblind_summary(args.grades, args.key, args.summary_out)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
