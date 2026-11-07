#!/usr/bin/env python3
"""Score outputs from eval_specialized_1000_unbounded.py.

Closed-form final/output fields are exact-match checked; open-ended work is
scored against explicit point rubrics in a blinded review sheet. This script
never uses keyword-only grading for free-form reasoning.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import secrets
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "data" / "eval_suites"
PREFIX = "parentheses_253m_specialized_1000_v1"
DEFAULT_PROMPTS = SUITE_DIR / f"{PREFIX}.prompts.jsonl"
DEFAULT_KEYS = SUITE_DIR / f"{PREFIX}.keys.jsonl"
DEFAULT_MANIFEST = SUITE_DIR / f"{PREFIX}.manifest.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalized(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("／", "/")
    value = value.replace("`", "").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(" .\t\r\n")


def final_payload(response: str) -> str | None:
    matches = list(re.finditer(r"(?im)^\s*Final:\s*(.*?)\s*$", response))
    return matches[-1].group(1).strip() if matches else None


def output_payload(response: str) -> str | None:
    match = re.search(r"(?im)^\s*Output:\s*(.*)$", response)
    if not match:
        return None
    same_line = match.group(1)
    rest = response[match.end():]
    payload = same_line + ("\n" + rest if rest else "")
    payload = re.sub(r"(?m)^\s*```(?:python|text)?\s*$", "", payload)
    payload = re.sub(r"(?m)^\s*```\s*$", "", payload)
    return payload.strip()


def literal_equal(actual: str, expected: str) -> bool:
    try:
        return ast.literal_eval(actual) == ast.literal_eval(expected)
    except (SyntaxError, ValueError):
        return normalized(actual) == normalized(expected)


def automatic_check(key: dict[str, Any], response: str) -> dict[str, Any]:
    mode = key.get("scoring_mode", "rubric")
    if mode == "final_exact":
        payload = final_payload(response)
        if payload is None:
            return {"mode": mode, "status": "missing_final_marker", "correct": None}
        expected = key.get("accepted_finals", [])
        good = any(normalized(payload) == normalized(str(x)) for x in expected)
        return {"mode": mode, "status": "correct" if good else "incorrect", "correct": good, "payload": payload}
    if mode == "exact_output":
        payload = output_payload(response)
        if payload is None:
            return {"mode": mode, "status": "missing_output_marker", "correct": None}
        expected = str(key.get("reference_answer", ""))
        good = literal_equal(payload, expected)
        return {"mode": mode, "status": "correct" if good else "incorrect", "correct": good, "payload": payload}
    return {"mode": "rubric", "status": "human_review", "correct": None}


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not isinstance(row.get("id"), str):
            raise ValueError("result row lacks string id")
        case_id = row["id"]
        prior = records.get(case_id)
        if prior is None or row.get("is_complete") is True or prior.get("is_complete") is not True:
            records[case_id] = row
    return records


def prepare_review(
    prompts: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    review_path: Path,
    map_path: Path,
    allow_incomplete: bool,
) -> tuple[int, int]:
    prompt_by_id = {row["id"]: row for row in prompts}
    key_by_id = {row["id"]: row for row in keys}
    missing = set(prompt_by_id) - set(records)
    incomplete = {case_id for case_id, row in records.items() if case_id in prompt_by_id and row.get("is_complete") is not True}
    if (missing or incomplete) and not allow_incomplete:
        raise ValueError(f"review prep requires 1,000 complete responses; missing={len(missing)} incomplete={len(incomplete)}")

    salt = secrets.token_hex(16)
    review_rows = []
    blind_map: dict[str, dict[str, Any]] = {}
    for case_id in prompt_by_id:
        result = records.get(case_id)
        if result is None or result.get("is_complete") is not True:
            continue
        prompt = prompt_by_id[case_id]
        key = key_by_id[case_id]
        blind_id = hashlib.sha256(f"{salt}|{case_id}".encode("utf-8")).hexdigest()[:14]
        blind_map[blind_id] = {
            "id": case_id,
            "domain_id": prompt["domain_id"],
            "language": prompt["language"],
            "difficulty": prompt["difficulty"],
        }
        review_rows.append({
            "blind_id": blind_id,
            "language": prompt["language"],
            "prompt": prompt["prompt"],
            "response": result.get("generated_text", ""),
            "rubric": key["rubric"],
            "reference_answer": key["reference_answer"],
            "scoring_mode": key.get("scoring_mode", "rubric"),
            "finish_reason": result.get("finish_reason", "unknown"),
            "scores": {},
            "notes": "",
        })
    write_jsonl(review_path, review_rows)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps({"salt": salt, "cases": blind_map}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(review_rows), len(missing) + len(incomplete)


def validate_score_value(value: Any, maximum: int, blind_id: str, criterion_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{blind_id} {criterion_id}: score must be an integer 0..{maximum}")
    return value


def build_report(
    prompts: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    grades: dict[str, dict[str, Any]] | None = None,
    blind_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    key_by_id = {row["id"]: row for row in keys}
    prompt_by_id = {row["id"]: row for row in prompts}
    grade_by_id: dict[str, dict[str, Any]] = {}
    if grades is not None:
        if blind_map is None:
            raise ValueError("blind map is required to score grades")
        for blind_id, grade in grades.items():
            if blind_id not in blind_map:
                raise ValueError(f"unknown blind_id in grades: {blind_id}")
            grade_by_id[blind_map[blind_id]["id"]] = grade

    total = len(prompts)
    present = set(records) & set(prompt_by_id)
    complete = {case_id for case_id in present if records[case_id].get("is_complete") is True}
    modes = Counter()
    auto_counts: Counter[str] = Counter()
    rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    human_points = human_max = human_cases = 0
    token_counts = []

    for case_id, prompt in prompt_by_id.items():
        result = records.get(case_id)
        key = key_by_id[case_id]
        check = automatic_check(key, str(result.get("generated_text", ""))) if result else {"mode": key.get("scoring_mode", "rubric"), "status": "missing", "correct": None}
        modes[check["mode"]] += 1
        auto_counts[check["status"]] += 1
        if result:
            token_counts.append(int(result.get("completion_tokens", 0)))
        item = {"prompt": prompt, "result": result, "key": key, "auto": check}
        rows_by_domain[prompt["domain_id"]].append(item)
        rows_by_language[prompt["language"]].append(item)

        grade = grade_by_id.get(case_id)
        if grade is not None:
            scored = grade.get("scores")
            if not isinstance(scored, dict):
                raise ValueError(f"grade for {case_id} must contain a scores object")
            rubric = key["rubric"]
            expected_ids = {criterion["criterion_id"] for criterion in rubric}
            if set(scored) != expected_ids:
                raise ValueError(f"grade for {case_id}: expected scores for {sorted(expected_ids)}")
            points = sum(validate_score_value(scored[c["criterion_id"]], c["points"], case_id, c["criterion_id"]) for c in rubric)
            maximum = sum(c["points"] for c in rubric)
            human_points += points
            human_max += maximum
            human_cases += 1

    def coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
        present_n = sum(x["result"] is not None for x in items)
        complete_n = sum(x["result"] is not None and x["result"].get("is_complete") is True for x in items)
        auto = [x["auto"] for x in items if x["auto"]["mode"] in ("final_exact", "exact_output")]
        graded = [x for x in items if x["prompt"]["id"] in grade_by_id]
        pts = 0; max_pts = 0
        for x in graded:
            gr = grade_by_id[x["prompt"]["id"]]["scores"]
            for criterion in x["key"]["rubric"]:
                pts += gr[criterion["criterion_id"]]
                max_pts += criterion["points"]
        checkable = [x for x in auto if x["correct"] is not None]
        return {
            "cases": len(items),
            "results_present": present_n,
            "complete": complete_n,
            "objective_checkable": len(checkable),
            "objective_correct": sum(x["correct"] is True for x in checkable),
            "objective_accuracy": (sum(x["correct"] is True for x in checkable) / len(checkable) if checkable else None),
            "human_scored": len(graded),
            "human_points": pts,
            "human_max_points": max_pts,
            "human_rubric_rate": (pts / max_pts if max_pts else None),
        }

    result = {
        "suite_id": PREFIX,
        "reported_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total,
        "results_present": len(present),
        "complete_responses": len(complete),
        "missing_responses": total - len(present),
        "incomplete_responses": len(present - complete),
        "scoring_modes": dict(modes),
        "automatic_status_counts": dict(auto_counts),
        "objective_accuracy_scope": "Only exact-final/exact-output items with the required marker are included; free-form reasoning is not keyword-scored.",
        "overall": coverage([{"prompt": prompt_by_id[cid], "result": records.get(cid), "key": key_by_id[cid], "auto": automatic_check(key_by_id[cid], str(records.get(cid, {}).get("generated_text", ""))) if cid in records else {"mode": key_by_id[cid].get("scoring_mode", "rubric"), "status": "missing", "correct": None}} for cid in prompt_by_id]),
        "by_domain": {name: coverage(items) for name, items in sorted(rows_by_domain.items())},
        "by_language": {name: coverage(items) for name, items in sorted(rows_by_language.items())},
        "manual_review": {"cases_scored": human_cases, "points": human_points, "max_points": human_max, "rate": human_points / human_max if human_max else None},
        "completion_tokens": {
            "count": len(token_counts),
            "min": min(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
            "mean": sum(token_counts) / len(token_counts) if token_counts else None,
        },
        "warning": "A completed stop token is not a quality score; inspect rubric ratings, loops, watchdog stops, and missing prompts separately.",
    }
    return result


def self_test() -> None:
    assert automatic_check({"scoring_mode": "final_exact", "accepted_finals": ["1/4", "25%"]}, "Work\nFinal: 1/4") ["correct"] is True
    assert automatic_check({"scoring_mode": "final_exact", "accepted_finals": ["1/4"]}, "Final: 0.25") ["correct"] is False
    assert automatic_check({"scoring_mode": "exact_output", "reference_answer": "[1, 2, 3]"}, "Output: [1, 2, 3]") ["correct"] is True
    assert automatic_check({"scoring_mode": "exact_output", "reference_answer": "done\nalpha"}, "Output:\ndone\nalpha") ["correct"] is True
    assert automatic_check({"scoring_mode": "rubric"}, "anything") ["status"] == "human_review"
    assert validate_score_value(2, 3, "x", "R1") == 2
    print("PASS: exact-final and exact-output checks, multi-line output, and rubric-score bounds.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--keys", default=str(DEFAULT_KEYS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results", help="run JSONL emitted by the unbounded evaluator")
    parser.add_argument("--prepare-review", help="write a blinded JSONL review sheet")
    parser.add_argument("--blind-map", help="private blind-id map; required with --prepare-review or --grades")
    parser.add_argument("--grades", help="reviewer-completed JSONL with blind_id and per-criterion scores")
    parser.add_argument("--report", help="write a JSON scoring report")
    parser.add_argument("--allow-incomplete", action="store_true", help="allow review preparation from a partial run; incomplete rows are omitted")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.results:
        parser.error("--results is required")

    prompts = read_jsonl(Path(args.prompts))
    keys = read_jsonl(Path(args.keys))
    if len(prompts) != 1000 or len(keys) != 1000:
        raise ValueError("expected exactly 1,000 prompt/key rows")
    if [p.get("id") for p in prompts] != [k.get("id") for k in keys]:
        raise ValueError("prompt/key IDs do not align")
    records = latest_records(Path(args.results))

    if args.prepare_review:
        if not args.blind_map:
            parser.error("--blind-map is required with --prepare-review")
        count, omitted = prepare_review(prompts, keys, records, Path(args.prepare_review), Path(args.blind_map), args.allow_incomplete)
        print(f"Wrote {count} blinded review rows to {args.prepare_review}; omitted {omitted} missing/incomplete cases.")
        print(f"Private mapping written to {args.blind_map}; keep it away from reviewers until scoring is complete.")

    grades_by_blind: dict[str, dict[str, Any]] | None = None
    blind_cases: dict[str, dict[str, Any]] | None = None
    if args.grades:
        if not args.blind_map:
            parser.error("--blind-map is required with --grades")
        map_data = json.loads(Path(args.blind_map).read_text(encoding="utf-8"))
        blind_cases = map_data["cases"]
        grade_rows = read_jsonl(Path(args.grades))
        grades_by_blind = {}
        for row in grade_rows:
            blind_id = row.get("blind_id")
            if not isinstance(blind_id, str) or blind_id in grades_by_blind:
                raise ValueError("grades must have unique string blind_id fields")
            grades_by_blind[blind_id] = row

    report = build_report(prompts, keys, records, grades_by_blind, blind_cases)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote scoring report to {out}")
    print(f"Responses complete: {report['complete_responses']}/{report['total_cases']}; objective statuses={report['automatic_status_counts']}; manual rubric={report['manual_review']['cases_scored']} cases")
    return 0


if __name__ == "__main__":
    if not __import__("sys").argv[1:] or "--help" in __import__("sys").argv[1:]:
        pass
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}")
