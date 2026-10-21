#!/usr/bin/env python3
"""Parse Nemotron-v2 distillation rows into clean SFT lanes.

Inputs:  out/distill_v2/raw/{articles,factsheets,reasoning,format}.jsonl
         out/distill_v2/specs/*.jsonl  (id -> prompt/meta)
         out/vault_eval_*.json        (eval prompts to exclude verbatim)
Outputs: out/distill_v2/parsed/{facts,reasoning,format,prose}.jsonl + report.json

Eval prompts are excluded as questions; the fact is kept as a declarative
statement instead, so the model can generalize without memorizing eval strings.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

RAW = Path("out/distill_v2/raw")
SPECS = Path("out/distill_v2/specs")
OUT = Path("out/distill_v2/parsed")

NORM_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")

KV_TEMPLATES = {
    "capitals": ("What is the capital of {x}?", "The capital of {x} is {y}."),
    "elements": ("What is the chemical symbol for {x}?", "The chemical symbol for {x} is {y}."),
    "compounds": ("What is the chemical formula for {x}?", "The chemical formula for {x} is {y}."),
    "currencies": ("What is the currency of {x}?", "The currency of {x} is {y}."),
    "antonyms": ("What is the opposite of {x}?", "The opposite of {x} is {y}."),
    "http": ("What does HTTP status code {x} mean?", "HTTP status code {x} means {y}."),
    "shapes": ("How many sides does a {x} have?", "A {x} has {y} sides."),
}

SENT_TYPES = {"people", "dravidian", "geo"}


def norm(text: str) -> str:
    return WS_RE.sub(" ", NORM_RE.sub(" ", text.lower())).strip()


def toks(text: str) -> set[str]:
    return set(norm(text).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_eval_prompts() -> list[tuple[str, set[str]]]:
    prompts = []
    seen = set()
    for path in glob.glob("out/vault_eval_*.json"):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in data.get("results", []):
            p = norm(str(row.get("prompt", "")))
            if p and p not in seen:
                seen.add(p)
                prompts.append((p, toks(p)))
    return prompts


def is_eval(q: str, eval_sets: list[tuple[str, set[str]]]) -> bool:
    t = toks(q)
    n = norm(q)
    for es, et in eval_sets:
        if n == es or jaccard(t, et) >= 0.9:
            return True
    return False


def load_specs() -> dict[str, dict]:
    specs = {}
    for path in SPECS.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            specs[s["id"]] = s
    return specs


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = row.get("id")
        if rid is not None:
            if rid in seen:
                continue
            seen.add(rid)
        out.append(row)
    return out


def clean_answer(a: str) -> str:
    a = a.strip().strip('"').strip("'").strip()
    a = re.sub(r"\s+", " ", a)
    return a


_SELF_REWRITES = [
    (re.compile(r"^what is the capital of (.+?)\?$", re.I), "The capital of {x} is {a}."),
    (re.compile(r"^what is the currency of (.+?)\?$", re.I), "The currency of {x} is {a}."),
    (re.compile(r"^which currency is used in (.+?)\?$", re.I), "The currency of {x} is {a}."),
    (re.compile(r"^what is the official currency of (.+?)\?$", re.I), "The official currency of {x} is {a}."),
    (re.compile(r"^what is the chemical symbol (?:for|of) (.+?)\?$", re.I), "The chemical symbol for {x} is {a}."),
    (re.compile(r"^what is the chemical formula (?:for|of) (.+?)\?$", re.I), "The chemical formula for {x} is {a}."),
    (re.compile(r"^what is the opposite of (.+?)\?$", re.I), "The opposite of {x} is {a}."),
]
_WHO_VERB_RE = re.compile(r"^who (created|developed|invented|wrote|painted|composed|discovered) (.+?)\?$", re.I)


def self_contain(q: str, a: str) -> str:
    """Make a bare answer self-contained so paraphrases still associate the subject."""
    bare = a.strip().rstrip(".")
    if not bare or len(bare.split()) > 6 or bare.endswith("."):
        return a
    for rx, tmpl in _SELF_REWRITES:
        m = rx.match(q.strip())
        if m:
            return tmpl.format(x=m.group(1).strip(), a=bare)
    m = _WHO_VERB_RE.match(q.strip())
    if m:
        return f"{bare} {m.group(1).lower()} {m.group(2).strip()}."
    return a


def article_docs(row: dict, eval_sets: list[tuple[str, set[str]]], report: Counter,
                 facts: list, prose: list) -> None:
    topic = (row.get("meta") or {}).get("topic") or ""
    content = row.get("content") or ""
    if not topic or not content:
        report["article_drop_missing"] += 1
        return
    body_lines, qa_lines = [], []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("ARTICLE:"):
            body_lines.append(line.split(":", 1)[1].strip())
        elif line.upper().startswith("QA:"):
            qa_lines.append(line.split(":", 1)[1].strip())
        elif not qa_lines:
            body_lines.append(line)
    body = " ".join(body_lines).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) >= 12]
    if sentences:
        for s in sentences:
            prose.append({"q": f"Tell me a fact about {topic}.", "a": s, "src": row["id"]})
    for qa in qa_lines:
        if "||" not in qa:
            report["article_drop_bad_qa"] += 1
            continue
        q, a = qa.split("||", 1)
        q, a = q.strip(), clean_answer(a)
        if not q or not a:
            report["article_drop_empty_answer"] += 1
            continue
        a = self_contain(q, a)
        if topic.lower() not in q.lower() and len(q.split()) <= 14:
            q = f"In {topic}, {q[0].lower()}{q[1:]}"
        if is_eval(q, eval_sets):
            report["facts_excluded_eval"] += 1
            stmt = None
            for s in sentences:
                if a.lower() in s.lower():
                    stmt = s
                    break
            if stmt:
                facts.append({"q": f"Tell me a fact about {topic}.", "a": stmt,
                              "src": row["id"], "decl": True})
            continue
        facts.append({"q": q, "a": a, "src": row["id"], "decl": False})


def factsheet_docs(row: dict, eval_sets: list[tuple[str, set[str]]], report: Counter,
                   facts: list) -> None:
    meta = row.get("meta") or {}
    kind = meta.get("type") or ""
    content = row.get("content") or ""
    if kind not in KV_TEMPLATES and kind not in SENT_TYPES:
        report["fact_drop_unknown_type"] += 1
        return
    n = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        x, y = line.split(":", 1)
        x, y = x.strip(), clean_answer(y)
        if not x or not y or len(x) > 60 or len(y) > 240:
            report["fact_drop_bad_line"] += 1
            continue
        if kind in KV_TEMPLATES:
            q_t, decl_t = KV_TEMPLATES[kind]
            q = q_t.format(x=x, y=y)
            decl = decl_t.format(x=x, y=y)
            a = decl
        else:
            q = f"Tell me about {x}."
            a = y
            decl = y
        if is_eval(q, eval_sets):
            report["facts_excluded_eval"] += 1
            facts.append({"q": f"Tell me a fact about {x}.", "a": decl,
                          "src": row["id"], "decl": True})
        else:
            facts.append({"q": q, "a": a, "src": row["id"], "decl": False})
        n += 1
    if n == 0:
        report["fact_drop_empty_doc"] += 1


def reasoning_docs(row: dict, specs: dict, report: Counter, out: list) -> None:
    content = (row.get("content") or "").strip()
    if not content:
        report["reason_drop_empty"] += 1
        return
    idx = content.upper().rfind("FINAL:")
    if idx >= 0:
        a = clean_answer(content[idx + 6:])
    else:
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        a = clean_answer(lines[-1]) if lines else ""
        report["reason_no_final"] += 1
    if not a or len(a) > 160:
        report["reason_drop_bad_answer"] += 1
        return
    if any(p in a.lower() for p in ("i don't know", "cannot determine", "unable to")):
        report["reason_drop_abstain"] += 1
        return
    spec = specs.get(row["id"], {})
    q = spec.get("prompt") or row.get("prompt") or ""
    if not q:
        report["reason_drop_no_prompt"] += 1
        return
    out.append({"q": q.strip(), "a": a, "src": row["id"]})


def format_docs(row: dict, specs: dict, report: Counter, out: list) -> None:
    content = (row.get("content") or "").strip()
    spec = specs.get(row["id"], {})
    q = (spec.get("prompt") or "").strip()
    if not q or not content:
        report["format_drop_missing"] += 1
        return
    low = content.lower()
    if "final:" in low or low.startswith("i don't") or low.startswith("sorry"):
        report["format_drop_bad"] += 1
        return
    if len(content) > 1200:
        report["format_drop_long"] += 1
        return
    out.append({"q": q, "a": content, "src": row["id"]})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    raw = Path(args.raw)
    out_dir = Path(args.out)
    specs = load_specs()
    eval_sets = load_eval_prompts()
    report = Counter()
    facts, reason, fmt, prose = [], [], [], []

    for row in read_jsonl(raw / "articles.jsonl"):
        article_docs(row, eval_sets, report, facts, prose)
        report["articles_read"] += 1
    for row in read_jsonl(raw / "factsheets.jsonl"):
        factsheet_docs(row, eval_sets, report, facts)
        report["factsheets_read"] += 1
    for row in read_jsonl(raw / "reasoning.jsonl"):
        reasoning_docs(row, specs, report, reason)
        report["reasoning_read"] += 1
    for row in read_jsonl(raw / "format.jsonl"):
        format_docs(row, specs, report, fmt)
        report["format_read"] += 1

    write_jsonl(out_dir / "facts.jsonl", facts)
    write_jsonl(out_dir / "reasoning.jsonl", reason)
    write_jsonl(out_dir / "format.jsonl", fmt)
    write_jsonl(out_dir / "prose.jsonl", prose)
    report.update({"facts": len(facts), "reasoning": len(reason),
                   "format": len(fmt), "prose": len(prose),
                   "eval_prompts": len(eval_sets)})
    (out_dir / "report.json").write_text(
        json.dumps(dict(report), indent=2), encoding="utf-8")
    print(json.dumps(dict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
