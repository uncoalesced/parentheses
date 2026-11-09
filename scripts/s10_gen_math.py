#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/s10_gen_math.py
Session 10 (Tanzanite) first-party short-math generator.

Teaches "short worked answer, then `Final: <answer>`, then stop" (gates G1 clean stop, G3 format).
Every answer is computed by Python arithmetic from the generated numbers, so it is correct by
construction (no teacher model, no licence question: first-party data, LICENSE_POLICY.md section 1).

Output JSONL, one row per example:
    {"id","source":"s10_gen_math","bucket":"short_math","lang","prompt","response","family"}

Languages: English word problems (11 families) + native-script basic arithmetic for tam/tel/kan/mal.
The native-script phrases are short fixed templates written by Claude (Joel: "anything"); they are
NOT yet reviewed by a native speaker and the data card says so. The `Final:` marker stays English in
every language because the evaluators look for that literal string.

Contamination: wording is deliberately different from the Hermes-1000 suite. The builder runs the
digit-masked template overlap check against every eval file before packing.

    .venv\\Scripts\\python.exe scripts\\s10_gen_math.py --n-en 60000 --n-drav 12000 --out data\\s10_gen\\math.jsonl
    .venv\\Scripts\\python.exe scripts\\s10_gen_math.py --self-test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

NAMES = ["Asha", "Ravi", "Meena", "Kiran", "Lakshmi", "Arun", "Divya", "Suresh", "Priya", "Vikram",
         "Anitha", "Manoj", "Kavya", "Harish", "Nila", "Sanjay", "Ishaan", "Tara", "Ganesh", "Leela"]
ITEMS = ["pencils", "notebooks", "mangoes", "marbles", "stickers", "coins", "bottles", "books", "tickets",
         "oranges", "buttons", "candles", "plates", "seeds", "stamps"]
SHOPS = ["a stationery shop", "a market stall", "a school store", "a bakery", "a village fair"]

# instruction suffixes (about 60% of rows get one; the rest are bare so the model also learns the default)
FINAL_INSTR = [
    "End with `Final: <answer>`.",
    "Give a short working, then finish with a line `Final: <answer>` and nothing after it.",
    "Show a brief check, then end with exactly `Final: <answer>`.",
    "Last line must be `Final: <answer>`.",
    "Answer briefly. Put the result on the last line as `Final: <answer>`.",
]


def fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


# ---------------------------------------------------------------- English families
def f_arith(r):
    op = r.choice("+-*/")
    if op == "+":
        a, b = r.randint(12, 9999), r.randint(12, 9999); v = a + b; verb = "plus"
    elif op == "-":
        a, b = r.randint(100, 9999), r.randint(11, 99); v = a - b; verb = "minus"
    elif op == "*":
        a, b = r.randint(12, 99), r.randint(3, 99); v = a * b; verb = "times"
    else:
        b = r.randint(3, 40); v = r.randint(5, 300); a = b * v; verb = "divided by"
    p = r.choice([f"What is {a} {verb} {b}?", f"Compute {a} {op} {b}.", f"Calculate {a} {op} {b}.",
                  f"{a} {op} {b} = ?"])
    return p, f"{a} {op} {b} = {v}", v


def f_buy(r):
    n, c = r.randint(3, 15), r.randint(4, 60)
    paid = ((n * c) // 50 + 1) * 50
    name, item, shop = r.choice(NAMES), r.choice(ITEMS), r.choice(SHOPS)
    p = (f"{name} buys {n} {item} at {shop}, each costing {c} rupees, and pays with {paid} rupees. "
         f"How much change does {name} get back?")
    total = n * c
    return p, f"Cost = {n} x {c} = {total}. Change = {paid} - {total} = {paid - total}.", paid - total


def f_percent(r):
    while True:
        pct = r.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75])
        n = r.choice([40, 60, 80, 120, 160, 200, 240, 300, 400, 480, 500, 800])
        if pct * n % 100 == 0:
            break
    v = pct * n // 100
    p = r.choice([f"What is {pct}% of {n}?", f"Find {pct} percent of {n}.",
                  f"A class has {n} students and {pct}% of them wear glasses. How many wear glasses?"])
    return p, f"{pct}% of {n} = {pct}/100 x {n} = {v}.", v


def f_linear(r):
    x = r.randint(-12, 25)
    a = r.choice([2, 3, 4, 5, 6, 7, 8, 9])
    b = r.randint(-20, 30)
    c = a * x + b
    sign = "+" if b >= 0 else "-"
    p = f"Solve for x: {a}x {sign} {abs(b)} = {c}"
    return p, f"{a}x = {c} - ({b}) = {c - b}, so x = {c - b}/{a} = {x}.", x


def f_rect(r):
    w, l = r.randint(3, 40), r.randint(5, 60)
    which = r.choice(["area", "perimeter"])
    v = w * l if which == "area" else 2 * (w + l)
    unit = r.choice(["m", "cm"])
    p = f"A garden bed measures {l} {unit} by {w} {unit}. What is its {which} (include units)?"
    work = f"{l} x {w} = {v}" if which == "area" else f"2 x ({l} + {w}) = {v}"
    u = f"{unit}^2" if which == "area" else unit
    return p, f"{which.capitalize()} = {work} {u}.", f"{v} {u}"


def f_speed(r):
    s, t = r.randint(20, 90), r.randint(2, 9)
    d = s * t
    mode = r.choice(["d", "t", "s"])
    if mode == "d":
        return (f"A bus travels at {s} km/h for {t} hours. How far does it go?",
                f"Distance = speed x time = {s} x {t} = {d} km.", f"{d} km")
    if mode == "t":
        return (f"A cyclist covers {d} km at a steady {s} km/h. How many hours does the trip take?",
                f"Time = distance / speed = {d} / {s} = {t} hours.", f"{t} hours")
    return (f"A train covers {d} km in {t} hours at constant speed. What is its speed in km/h?",
            f"Speed = {d} / {t} = {s} km/h.", f"{s} km/h")


def f_average(r):
    k = r.randint(3, 6)
    xs = [r.randint(40, 99) for _ in range(k)]
    tot = sum(xs)
    if tot % k:
        xs[-1] += k - tot % k
        tot = sum(xs)
    p = f"Find the mean of these {k} test scores: {', '.join(map(str, xs))}."
    return p, f"Sum = {tot}. Mean = {tot}/{k} = {tot // k}.", tot // k


def f_discount(r):
    while True:
        price = r.choice([200, 250, 400, 500, 800, 1000, 1200, 1500, 2000])
        d = r.choice([10, 20, 25, 30, 40, 50])
        if price * d % 100 == 0:
            break
    cut = price * d // 100
    p = f"A jacket is marked {price} rupees with a {d}% discount. What is the sale price?"
    return p, f"Discount = {d}% of {price} = {cut}. Sale price = {price} - {cut} = {price - cut}.", price - cut


def f_ratio(r):
    a, b = r.choice([(2, 3), (3, 4), (1, 4), (2, 5), (3, 5), (1, 2)])
    unit = r.randint(4, 30)
    tot = (a + b) * unit
    n1, n2 = r.sample(NAMES, 2)
    p = f"{n1} and {n2} share {tot} rupees in the ratio {a}:{b}. How much does {n1} get?"
    return p, f"Parts = {a} + {b} = {a + b}. One part = {tot}/{a + b} = {unit}. {n1} gets {a} x {unit} = {a * unit}.", a * unit


def f_remainder(r):
    b = r.randint(3, 19)
    a = r.randint(50, 999)
    p = r.choice([f"What is the remainder when {a} is divided by {b}?", f"Compute {a} mod {b}."])
    return p, f"{a} = {b} x {a // b} + {a % b}.", a % b


def f_ages(r):
    n1, n2 = r.sample(NAMES, 2)
    a = r.randint(8, 40)
    diff = r.randint(2, 15)
    yrs = r.randint(2, 10)
    p = f"{n1} is {a} years old and {n2} is {diff} years older. How old will {n2} be in {yrs} years?"
    return p, f"{n2} is now {a} + {diff} = {a + diff}. In {yrs} years: {a + diff} + {yrs} = {a + diff + yrs}.", a + diff + yrs


FAMILIES = [f_arith, f_buy, f_percent, f_linear, f_rect, f_speed, f_average, f_discount, f_ratio, f_remainder, f_ages]
WEIGHTS = [3, 3, 2, 2, 1, 2, 1, 1, 1, 1, 1]


def make_en(r: random.Random):
    fam = r.choices(FAMILIES, WEIGHTS)[0]
    p, work, ans = fam(r)
    if r.random() < 0.6:
        p = p + " " + r.choice(FINAL_INSTR)
    return {"lang": "eng", "prompt": p, "response": f"{work}\nFinal: {fmt(ans)}", "family": fam.__name__}


# ---------------------------------------------------------------- native-script basic arithmetic
# (fixed short templates; {a},{b} are digits. Joel reviews once. See the data card.)
DRAV = {
    "tam": {"add": "{a} மற்றும் {b} கூட்டினால் எவ்வளவு?", "sub": "{a} இலிருந்து {b} கழித்தால் எவ்வளவு?",
            "mul": "{a} ஐ {b} ஆல் பெருக்கினால் எவ்வளவு?", "ans": "விடை: {w}\nFinal: {v}"},
    "tel": {"add": "{a} మరియు {b} కలిపితే ఎంత?", "sub": "{a} నుండి {b} తీసివేస్తే ఎంత?",
            "mul": "{a} ను {b} తో గుణిస్తే ఎంత?", "ans": "సమాధానం: {w}\nFinal: {v}"},
    "kan": {"add": "{a} ಮತ್ತು {b} ಕೂಡಿಸಿದರೆ ಎಷ್ಟು?", "sub": "{a} ರಿಂದ {b} ಕಳೆದರೆ ಎಷ್ಟು?",
            "mul": "{a} ನ್ನು {b} ರಿಂದ ಗುಣಿಸಿದರೆ ಎಷ್ಟು?", "ans": "ಉತ್ತರ: {w}\nFinal: {v}"},
    "mal": {"add": "{a} ഉം {b} ഉം കൂട്ടിയാൽ എത്ര?", "sub": "{a} ൽ നിന്ന് {b} കുറച്ചാൽ എത്ര?",
            "mul": "{a} നെ {b} കൊണ്ട് ഗുണിച്ചാൽ എത്ര?", "ans": "ഉത്തരം: {w}\nFinal: {v}"},
}


def make_drav(r: random.Random, lang: str):
    t = DRAV[lang]
    op = r.choice(["add", "sub", "mul"])
    if op == "add":
        a, b = r.randint(11, 999), r.randint(11, 999); v = a + b; w = f"{a} + {b} = {v}"
    elif op == "sub":
        a, b = r.randint(100, 999), r.randint(11, 99); v = a - b; w = f"{a} - {b} = {v}"
    else:
        a, b = r.randint(11, 99), r.randint(3, 30); v = a * b; w = f"{a} x {b} = {v}"
    return {"lang": lang, "prompt": t[op].format(a=a, b=b), "response": t["ans"].format(w=w, v=v),
            "family": f"drav_{op}"}


def generate(n_en: int, n_drav: int, seed: int):
    r = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < n_en and tries < n_en * 20:
        tries += 1
        row = make_en(r)
        if row["prompt"] in seen:
            continue
        seen.add(row["prompt"]); out.append(row)
    langs = list(DRAV)
    tries = 0
    nd = 0
    while nd < n_drav and tries < n_drav * 20:
        tries += 1
        row = make_drav(r, langs[nd % 4])
        if row["prompt"] in seen:
            continue
        seen.add(row["prompt"]); out.append(row); nd += 1
    r.shuffle(out)
    for i, row in enumerate(out):
        row.update({"id": f"s10_math_{seed}_{i:07d}", "source": "s10_gen_math", "bucket": "short_math"})
    return out


def self_test() -> None:
    import re
    rows = generate(3000, 400, 7)
    assert len(rows) == 3400, len(rows)
    for row in rows:
        assert row["response"].rstrip().split("\n")[-1].startswith("Final:"), row
        assert row["response"].count("Final:") == 1
    n_checked = 0
    for row in rows:
        if row["family"] == "f_arith":
            m = re.search(r"(\d+) ([+\-*/]) (\d+) = (-?\d+)", row["response"])
            a, op, b, v = int(m[1]), m[2], int(m[3]), int(m[4])
            assert {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op] == v and (op != "/" or a % b == 0)
            n_checked += 1
        if row["family"] == "f_linear":
            m = re.match(r"Solve for x: (\d+)x ([+-]) (\d+) = (-?\d+)", row["prompt"])
            a, s, b, c = int(m[1]), m[2], int(m[3]), int(m[4])
            x = int(row["response"].split("Final:")[1])
            assert a * x + (b if s == "+" else -b) == c
            n_checked += 1
        if row["family"] == "f_remainder":
            m = re.search(r"(\d+) (?:is divided by|mod) (\d+)", row["prompt"])
            assert int(m[1]) % int(m[2]) == int(row["response"].split("Final:")[1])
            n_checked += 1
    assert n_checked > 300, n_checked
    assert len({r["prompt"] for r in rows}) == len(rows), "duplicate prompt"
    assert {r["lang"] for r in rows} == {"eng", "tam", "tel", "kan", "mal"}
    print(f"[self-test] s10_gen_math ok ({len(rows)} rows, {n_checked} independently re-checked)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-en", type=int, default=60000)
    ap.add_argument("--n-drav", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", type=Path, default=Path("data/s10_gen/math.jsonl"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    rows = generate(a.n_en, a.n_drav, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
