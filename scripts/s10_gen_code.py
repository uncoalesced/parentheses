#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/s10_gen_code.py
Session 10 (Tanzanite) first-party code-output generator.

Teaches "read a short Python program, answer `Output: <value>`, stop" with NO tool calls
(gates G3 format, G2 tool-call leak). The ground truth is the real stdout of the program, obtained by
executing it. Programs come from fixed templates with bounded ranges, so none can loop forever or touch
files/network; execution uses a tiny whitelist of builtins and no imports.

Output JSONL: {"id","source":"s10_gen_code","bucket":"code_output","lang":"eng","prompt","response","family"}
First-party data (LICENSE_POLICY.md section 1): no external dataset, no teacher.

    .venv\\Scripts\\python.exe scripts\\s10_gen_code.py --n 45000 --out data\\s10_gen\\code.jsonl
    .venv\\Scripts\\python.exe scripts\\s10_gen_code.py --self-test      # also re-runs a sample in a fresh `python -I`
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import random
import subprocess
import sys
from pathlib import Path

WORDS = ["apple", "river", "stone", "cloud", "tiger", "mango", "plant", "bread", "lemon", "chair", "eagle", "pearl"]
SAFE = {k: getattr(builtins, k)
        for k in ["print", "range", "len", "sum", "min", "max", "sorted", "reversed", "enumerate", "zip", "list",
                  "dict", "set", "tuple", "str", "int", "abs", "any", "all", "map", "filter", "round", "bool"]}

INSTR = [
    "What does this program print? Report the value after `Output:`.",
    "Trace the code and give what it prints. Write `Output: <value>`.",
    "What is printed? Answer with `Output: <value>` only.",
    "Predict the output of this Python code. End with `Output: <value>`.",
    "What does this print?",
    "Run this in your head and tell me the printed result.",
]


def t_sum_range(r):
    a, b, s = r.randint(0, 5), r.randint(8, 30), r.choice([1, 2, 3])
    return f"total = 0\nfor i in range({a}, {b}, {s}):\n    total += i\nprint(total)"


def t_list_comp(r):
    n, k, m = r.randint(5, 12), r.randint(2, 4), r.randint(2, 6)
    return f"nums = [i * {k} for i in range({n})]\nprint([x for x in nums if x % {m} == 0])"


def t_slice(r):
    xs = [r.randint(1, 40) for _ in range(r.randint(6, 9))]
    a, b = r.randint(0, 2), r.randint(3, len(xs))
    st = r.choice([1, 2])
    return f"data = {xs}\nprint(data[{a}:{b}:{st}])"


def t_str_ops(r):
    w = r.choice(WORDS)
    op = r.choice([f"s.upper()", f"s[::-1]", f"s.replace({w[1]!r}, '*')", f"s.count({w[0]!r})",
                   f"len(s) * {r.randint(2, 5)}", f"s[{r.randint(0, 2)}:{r.randint(3, 5)}]", f"s.capitalize()"])
    return f"s = {w!r}\nprint({op})"


def t_dict_count(r):
    ws = [r.choice(WORDS[:5]) for _ in range(r.randint(6, 10))]
    return (f"words = {ws}\ncounts = {{}}\nfor w in words:\n    counts[w] = counts.get(w, 0) + 1\n"
            f"print(sorted(counts.items()))")


def t_func(r):
    a, b, x = r.randint(2, 9), r.randint(1, 9), r.randint(1, 12)
    return f"def f(x):\n    return x * {a} + {b}\n\nprint(f({x}), f(f({r.randint(0, 4)})))"


def t_cond(r):
    x, y = r.randint(1, 30), r.randint(1, 30)
    return (f"x = {x}\ny = {y}\nif x > y:\n    print('first')\nelif x == y:\n    print('equal')\nelse:\n    print('second')\n"
            f"print(x + y)")


def t_while(r):
    n, d = r.randint(20, 400), r.choice([2, 3])
    return f"n = {n}\nsteps = 0\nwhile n > 1:\n    n = n // {d}\n    steps += 1\nprint(steps, n)"


def t_fib(r):
    n = r.randint(5, 14)
    return f"a, b = 0, 1\nfor _ in range({n}):\n    a, b = b, a + b\nprint(a)"


def t_sorted_key(r):
    ws = r.sample(WORDS, 5)
    key = r.choice(["len(w)", "w[-1]", "-len(w)"])
    return f"words = {ws}\nprint(sorted(words, key=lambda w: {key}))"


def t_enumerate(r):
    ws = r.sample(WORDS, 4)
    k = r.randint(1, 3)
    return f"items = {ws}\nout = []\nfor i, w in enumerate(items):\n    if i % 2 == {k % 2}:\n        out.append(w[:{k + 1}])\nprint(out)"


def t_minmax(r):
    xs = [r.randint(-20, 60) for _ in range(r.randint(5, 8))]
    return f"v = {xs}\nprint(max(v) - min(v), sum(v) // len(v))"


def t_nested(r):
    n, m = r.randint(2, 5), r.randint(2, 5)
    return f"count = 0\nfor i in range({n}):\n    for j in range({m}):\n        if (i + j) % 2 == 0:\n            count += 1\nprint(count)"


def t_set(r):
    a = [r.randint(1, 15) for _ in range(7)]
    b = [r.randint(1, 15) for _ in range(7)]
    return f"a = set({a})\nb = set({b})\nprint(sorted(a & b), len(a | b))"


def t_zip(r):
    xs = [r.randint(1, 9) for _ in range(4)]
    ys = [r.randint(1, 9) for _ in range(4)]
    return f"xs = {xs}\nys = {ys}\nprint(sum(x * y for x, y in zip(xs, ys)))"


def t_string_loop(r):
    w = r.choice(WORDS) + r.choice(WORDS)
    return f"s = {w!r}\nvowels = 0\nfor ch in s:\n    if ch in 'aeiou':\n        vowels += 1\nprint(vowels, len(s) - vowels)"


TEMPLATES = [t_sum_range, t_list_comp, t_slice, t_str_ops, t_dict_count, t_func, t_cond, t_while, t_fib,
             t_sorted_key, t_enumerate, t_minmax, t_nested, t_set, t_zip, t_string_loop]


def run(code: str):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<s10>", "exec"), {"__builtins__": SAFE})
    except Exception:
        return None
    out = buf.getvalue().rstrip("\n")
    return out if 0 < len(out) <= 160 else None


def make(r: random.Random):
    t = r.choice(TEMPLATES)
    code = t(r)
    out = run(code)
    if out is None:
        return None
    instr = r.choice(INSTR)
    prompt = f"{instr}\n```python\n{code}\n```" if r.random() < 0.7 else f"```python\n{code}\n```\n{instr}"
    return {"lang": "eng", "prompt": prompt, "response": f"Output: {out}", "family": t.__name__, "_code": code}


def generate(n: int, seed: int):
    r = random.Random(seed)
    seen, rows = set(), []
    tries = 0
    while len(rows) < n and tries < n * 30:
        tries += 1
        row = make(r)
        if row is None or row["_code"] in seen:
            continue
        seen.add(row["_code"]); rows.append(row)
    for i, row in enumerate(rows):
        row.update({"id": f"s10_code_{seed}_{i:07d}", "source": "s10_gen_code", "bucket": "code_output"})
    return rows


def self_test() -> None:
    rows = generate(2500, 11)
    assert len(rows) == 2500, len(rows)
    fams = {r["family"] for r in rows}
    assert len(fams) == len(TEMPLATES), fams
    for row in rows:
        assert row["response"].startswith("Output: ") and "<tool_call>" not in row["response"]
    # independent verification: run a sample in a fresh isolated interpreter and compare stdout
    sample = random.Random(5).sample(rows, 120)
    for row in sample:
        p = subprocess.run([sys.executable, "-I", "-c", row["_code"]], capture_output=True, text=True, timeout=20)
        assert p.returncode == 0, row["_code"]
        assert p.stdout.rstrip("\n") == row["response"][len("Output: "):], (row["_code"], p.stdout, row["response"])
    print(f"[self-test] s10_gen_code ok ({len(rows)} rows, {len(fams)} families, {len(sample)} re-run in python -I)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", type=Path, default=Path("data/s10_gen/code.jsonl"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    rows = generate(a.n, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            row.pop("_code", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
