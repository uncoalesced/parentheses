#!/usr/bin/env python3
"""Claude's side of the eval250x2 job (2026-10-13): check golds, re-score outputs, audit Antigravity, dump for reading.

    .venv\\Scripts\\python.exe scripts\\review_eval250.py golds  eval\\queries_120m_reason250.jsonl
    .venv\\Scripts\\python.exe scripts\\review_eval250.py score  out\\eval250x2\\reason_step3815.jsonl
    .venv\\Scripts\\python.exe scripts\\review_eval250.py audit  out\\eval250x2\\reason_step3815.jsonl out\\eval250x2\\judged_reason_step3815.jsonl
    .venv\\Scripts\\python.exe scripts\\review_eval250.py dump   out\\eval250x2\\reason_step3815.jsonl [judged.jsonl]
    .venv\\Scripts\\python.exe scripts\\review_eval250.py --self-test
golds: math golds vs GSM8K test, code asserts vs MBPP test (both from the F: hf_cache).
score: independent auto-checks. Math: strict "Answer:" and lenient last number. Code: prompt+output+asserts executed
(isolated interpreter, 10 s). Logic: normalised gold contained in the first output line.
dump: markdown pages of 50 rows under out\\eval250x2\\read\\ (prompt, reference, output, verdict) for a full read.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", r"F:\Lingual_Data\distillation\hf_cache")
sys.path.insert(0, r"F:\Lingual_Data\scripts")
import distill_reasoning as dr   # noqa: E402  (verify_python_code, norm_answer, ANSWER_RE, CODE_RE)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
WORD = re.compile(r"\w{4,}", re.UNICODE)


def jl(p) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def score_row(r: dict) -> dict:
    out, b = r.get("output", ""), r.get("bucket")
    if b == "math":
        strict = dr.ANSWER_RE.findall(dr.norm_answer(out))
        nums = NUM_RE.findall(out)
        return {"strict": bool(strict) and num_eq(strict[-1], r["reference"]),
                "lenient": bool(nums) and num_eq(nums[-1], r["reference"])}
    if b == "code":
        tests = r.get("notes") or ""
        blocks = dr.CODE_RE.findall(out)
        cand = [blocks[-1]] if blocks else []
        cand.append(r["prompt"] + out)   # base model continues the signature/docstring
        ok = any(dr.verify_python_code(c + "\n\n" + tests) for c in cand)
        return {"strict": ok, "lenient": ok}
    if b == "logic":
        first = norm(out.strip().split("\n")[0]) if out.strip() else ""
        gold = norm(str(r["reference"]))
        return {"strict": bool(gold) and first == gold, "lenient": bool(gold) and gold in first}
    return {}


def cmd_score(path):
    rows = jl(path)
    t = defaultdict(Counter)
    for r in rows:
        s = score_row(r)
        k = (r.get("lang"), r.get("bucket"))
        t[k]["n"] += 1
        t[k]["eos"] += r.get("stop") == "eos"
        t[k]["loop"] += r.get("stop") == "loop"
        t[k]["script_ok"] += bool(r.get("script_ok"))
        for m in ("strict", "lenient"):
            t[k][m] += bool(s.get(m))
    print("lang bucket n | strict lenient | eos loop script_ok")
    for k in sorted(t):
        c = t[k]
        print(f"{k[0]} {k[1]} {c['n']} | {c['strict']} {c['lenient']} | {c['eos']} {c['loop']} {c['script_ok']}")


def cmd_golds(path):
    from datasets import load_dataset
    rows = jl(path)
    gsm = {norm(x["question"]): x["answer"].split("####")[-1].strip().replace(",", "")
           for x in load_dataset("openai/gsm8k", "main", split="test")}
    mbpp = {norm(x["text"]): x for x in load_dataset("google-research-datasets/mbpp", "full", split="test")}
    bad = []
    for r in rows:
        if r["bucket"] == "math":
            hit = next((g for q, g in gsm.items() if q and q in norm(r["prompt"])), None)
            if hit is None or not num_eq(hit, str(r["reference"])):
                bad.append((r["id"], "math", "not in GSM8K test" if hit is None else f"gold {r['reference']} != {hit}"))
        elif r["bucket"] == "code":
            hit = next((x for t, x in mbpp.items() if t and t in norm(r["prompt"])), None)
            if hit is None:
                bad.append((r["id"], "code", "text not in MBPP test"))
            elif not all(norm(a) in norm(r.get("notes") or "") for a in hit["test_list"]):
                bad.append((r["id"], "code", f"asserts differ from MBPP task {hit['task_id']}"))
    print(f"{path}: {Counter(r['bucket'] for r in rows)}; gold problems: {len(bad)}")
    for b in bad:
        print("  ", *b)


def cmd_audit(out_path, judged_path):
    rows = {r["id"]: r for r in jl(out_path)}
    vs = jl(judged_path)
    ids = Counter(v.get("id") for v in vs)
    missing, extra = set(rows) - set(ids), set(ids) - set(rows)
    dups = sum(c > 1 for c in ids.values())
    comments = Counter(norm(v.get("comment", "")) for v in vs)
    dup_c = sum(c - 1 for c in comments.values() if c > 1)
    generic = [v["id"] for v in vs if v.get("id") in rows and not
               (set(w.lower() for w in WORD.findall(v.get("comment", ""))) &
                set(w.lower() for w in WORD.findall(rows[v["id"]]["output"] + " " + rows[v["id"]]["prompt"])))]
    # judge says correct but our auto-check says wrong (or the reverse) on checkable buckets
    clash = [(v["id"], v.get("correct"), score_row(rows[v["id"]]).get("lenient")) for v in vs
             if v.get("id") in rows and rows[v["id"]]["bucket"] in ("math", "code", "logic")
             and bool(v.get("correct")) != bool(score_row(rows[v["id"]]).get("lenient"))]
    import distill_eval   # F:\Lingual_Data\scripts, already on sys.path: same scripted-judging tells
    scripted = distill_eval.scripted_signs(vs, {i: r["output"] for i, r in rows.items()})
    by = defaultdict(list)
    for v in vs:
        if v.get("id") in rows:
            by[rows[v["id"]]["bucket"]].append(v)
    constant = [b for b, L in by.items() if len(L) >= 10 and
                all(len({v.get(f) for v in L}) == 1 for f in ("correct", "on_task", "fluent"))]
    print(f"verdicts {len(vs)} / rows {len(rows)}: missing {len(missing)}, extra {len(extra)}, dup ids {dups}, "
          f"dup comments {dup_c}, comments sharing no word with row {len(generic)}, judge-vs-autocheck clashes {len(clash)}")
    print(f"scripted-judging signs: {scripted or 'none'}; buckets with constant scores: {constant or 'none'}")
    for c in clash[:20]:
        print("   clash", *c)


def cmd_dump(out_path, judged_path=None):
    rows = jl(out_path)
    jv = {v["id"]: v for v in jl(judged_path)} if judged_path else {}
    d = Path(out_path).parent / "read"
    d.mkdir(exist_ok=True)
    stem = Path(out_path).stem
    for p in range(0, len(rows), 50):
        lines = []
        for r in rows[p:p + 50]:
            s = score_row(r)
            v = jv.get(r["id"], {})
            lines += [f"### {r['id']} [{r['lang']}/{r['bucket']}] stop={r.get('stop')} auto={s} judge=correct:{v.get('correct')} "
                      f"on_task:{v.get('on_task')} fluent:{v.get('fluent')} fab:{v.get('fabricated')}",
                      f"PROMPT: {r['prompt'][:600]}", f"REF: {str(r.get('reference'))[:300]}",
                      f"OUT: {r.get('output', '')[:900]}", f"JUDGE: {v.get('comment', '-')}", ""]
        (d / f"{stem}_{p // 50:02d}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {(len(rows) + 49) // 50} pages to {d}")


def self_test():
    assert score_row({"bucket": "math", "reference": "18", "output": "so 9*2\nAnswer: 18"}) == {"strict": True, "lenient": True}
    assert score_row({"bucket": "math", "reference": "18", "output": "it is 18 dollars"}) == {"strict": False, "lenient": True}
    assert score_row({"bucket": "logic", "reference": "Carol", "output": "carol\nbecause..."})["strict"]
    assert not score_row({"bucket": "logic", "reference": "Carol", "output": "Bob"})["lenient"]
    r = {"bucket": "code", "prompt": "def add(a, b):\n", "output": "    return a + b\n", "notes": "assert add(2, 3) == 5"}
    assert score_row(r)["strict"]
    assert not score_row({**r, "output": "    return a - b\n"})["strict"]
    print("[self-test] review_eval250 ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", choices=["golds", "score", "audit", "dump"])
    p.add_argument("files", nargs="*")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    {"golds": cmd_golds, "score": cmd_score, "audit": cmd_audit, "dump": cmd_dump}[a.cmd](*a.files)


if __name__ == "__main__":
    sys.exit(main())
