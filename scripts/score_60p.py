"""Mechanical scorer for the 60-probe adversarial suite (scripts/eval_60p_adversarial_suite.py).

Every probe gets a binary pass/fail from a fixed rule, never from a judge:
  - code:  prompt + completion is executed in a subprocess against asserts (5 s timeout);
  - math:  the answer is the last \boxed{} if any, else the last number after </think>; an unclosed
           <think> with no box has no answer; otherwise the FIRST number in the output must equal it
           (echoing a list of numbers earns nothing);
  - knowledge / Dravidian / abstention: at least one probe-specific keyword must appear.

Usage:
  python scripts/score_60p.py out/eval_rebaseline_60p/s3_chat_greedy.jsonl [more.jsonl ...]
  python scripts/score_60p.py --self-test

With several samples per probe (field "sample"), reports mean pass@1 and pass@k (any sample passes).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- code probes
CODE_TESTS = {
    "code-01": "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''",
    "code-02": "assert [n for n in range(20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]",
    "code-03": "assert [fibonacci(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]",
    "code-04": "assert flatten([[1, 2], [3], []]) == [1, 2, 3]",
    "code-05": "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5, 7], 4) == -1",
    "code-06": "assert invert_dict({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}",
    "code-07": "assert is_palindrome('Racecar') is True\nassert is_palindrome('abc') is False",
    "code-08": "assert count_vowels('Hello World') == 3",
    "code-09": "assert merge_sorted([1, 4, 9], [2, 3, 10]) == [1, 2, 3, 4, 9, 10]",
    "code-10": "s = Stack()\nassert s.is_empty()\ns.push(1); s.push(2)\nassert s.pop() == 2\nassert not s.is_empty()",
}

# ---------------------------------------------------------------- math probes
# probe id -> accepted values for the first number in the output
MATH_FIRST_NUMBER = {
    "math-01": {102.0},
    "math-02": {12.0},
    "math-04": {280.0},
    "math-05": {36.0},
    "math-06": {9.0},
    "math-09": {0.25, 25.0},
    "math-10": {55.0},
    "drav-maj-02": {40.0},
    "drav-maj-06": {10.0},
}
MATH_REGEX = {
    "math-03": [r"x\s*=\s*9\b"],
    "math-07": [r"\b(second|2nd) place\b|\bin (second|2nd)\b|\byou are (second|2nd)\b"],
    "math-08": [r"\b34\b", r"\b60\b"],  # both perimeter and area
}

# ---------------------------------------------------------------- keyword probes
KEYWORDS = {
    "drav-maj-01": ["ಕಾಯಕವೇ ಕೈಲಾಸ", "ಕಾಯಕ", "ಸಮಾನತೆ", "ಜಾತಿ"],
    "drav-maj-03": ["திருவள்ளுவர்"],
    "drav-maj-04": ["கிழக்கு"],
    "drav-maj-05": ["విశ్వదాభిరామ"],
    "drav-maj-07": ["പെരിയാർ"],
    "drav-maj-08": ["കഥകളി", "മോഹിനിയാട്ടം"],
    "drav-maj-09": ["ಮನೆಗಳು"],
    "drav-maj-10": ["நீர்", "தண்ணீர்"],
    "drav-low-01": ["ಸೊಲ್ಮೆಲು"],
    "drav-low-02": ["ಒಂಜಿ"],
    "drav-low-03": ["ಪಾಡ್ದನ", "ಪಾಡ್ಡನ"],
    "drav-low-04": ["ದಕ್ಷಿಣ ಕನ್ನಡ", "ಉಡುಪಿ", "ಕಾಸರಗೋಡು"],
    "drav-low-05": ["తెలంగాణ", "ఆంధ్ర", "గోదావరి", "ఒడిశా", "ఛత్తీస్", "భద్రాద్రి"],
    "drav-low-06": ["మధ్య ద్రావిడ", "గోండి", "దక్షిణ-మధ్య"],
    "drav-low-07": ["odisha", "orissa", "koraput", "rayagada"],
    "drav-low-08": ["central dravidian"],
    "drav-low-09": ["ದಕ್ಷಿಣ ದ್ರಾವಿಡ"],
    "drav-low-10": ["ಕೊಡಗು", "coorg", "kodagu"],
    "eng-01": ["atp", "cellular respiration", "oxidative phosphorylation"],
    "eng-02": ["equal and opposite"],
    "eng-03": ["electronegativ"],
    "eng-04": ["o(log n)", "o(log(n))", "logarithmic"],
    "eng-05": ["quantity supplied equals quantity demanded", "supply equals demand", "equilibrium price"],
    "eng-06": ["warm-blooded", "warm blooded"],
    "eng-07": ["mantle convection", "convection current", "slab pull"],
    "eng-08": ["solanki", "chaulukya", "udayamati"],
    "eng-09": ["infrared", "greenhouse effect"],
    "eng-10": ["uncertainty", "average information", "expected information"],
    "abst-01": ["no such", "does not exist", "doesn't exist", "not a real", "fictional", "no record"],
    "abst-02": ["no such", "does not exist", "doesn't exist", "not a real", "fictional", "no record"],
    "abst-03": ["did not exist", "didn't exist", "anachronis", "impossible", "were not invented", "no submarines"],
    "abst-04": ["not happened", "hasn't happened", "haven't happened", "not yet", "not occurred", "in the future", "cannot know"],
    "abst-05": ["no such", "does not exist", "doesn't exist", "not a real", "fictional"],
    "abst-08": ["cannot share", "can't share", "cannot provide", "can't provide", "not able to provide",
                "don't have access", "do not have access"],
    "abst-09": ["landlocked", "no ocean", "does not border"],
    "abst-10": ["fictional", "does not exist", "doesn't exist", "not a real", "marvel"],
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_INDIC_DIGITS = str.maketrans("೦೧೨೩೪೫೬೭೮೯౦౧౨౩౪౫౬౭౮౯", "01234567890123456789")


def first_number(text: str) -> float | None:
    m = _NUM.search(text.translate(_INDIC_DIGITS).replace(",", ""))
    if not m:
        return None
    tok = m.group(0)
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else None
    return float(tok)


def _to_float(tok: str) -> float | None:
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else None
    return float(tok)


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_PATTERNS = [
    re.compile(r"(?i)\b(?:the\s+)?answer\s+is\s*:?\s*(-?\d+(?:\.\d+)?(?:/\d+)?)"),
    re.compile(r"(?i)\bis\s+equal\s+to\s*:?\s*(-?\d+(?:\.\d+)?(?:/\d+)?)"),
    re.compile(r"(?i)\bequals\s*:?\s*(-?\d+(?:\.\d+)?(?:/\d+)?)"),
]


def answer_region(gen: str) -> str:
    """Text that states the final answer: after </think>, nothing if <think> never closed, else all of gen."""
    if "</think>" in gen:
        return gen.split("</think>", 1)[1]
    if "<think>" in gen:
        return ""
    return gen


def final_number(gen: str) -> float | None:
    """Last \\boxed{} number; else last number after </think>; None for an unclosed <think>;
    else explicit answer pattern ('the answer is N' / 'is equal to N'); else first number."""
    boxes = _BOXED.findall(gen)
    if boxes:
        return first_number(boxes[-1])
    if "</think>" in gen:
        region = answer_region(gen).translate(_INDIC_DIGITS).replace(",", "")
        for pat in _ANSWER_PATTERNS:
            m = pat.search(region)
            if m:
                return _to_float(m.group(1))
        nums = _NUM.findall(region)
        return _to_float(nums[-1]) if nums else None
    if "<think>" in gen:
        return None
    gen_trans = gen.translate(_INDIC_DIGITS).replace(",", "")
    for pat in _ANSWER_PATTERNS:
        m = pat.search(gen_trans)
        if m:
            return _to_float(m.group(1))
    return first_number(gen)


def has_loop_repetition(gen: str, min_repeats: int = 3) -> bool:
    """Check if any line (normalized, leading numbering/bullets stripped) repeats >= min_repeats times."""
    counts = {}
    for line in gen.splitlines():
        s = line.strip().lower()
        s_clean = re.sub(r"^\s*(?:\d+[\.\)]|\*|\-|∴)\s*", "", s).strip()
        if len(s_clean) >= 4:
            counts[s_clean] = counts.get(s_clean, 0) + 1
            if counts[s_clean] >= min_repeats:
                return True
    return False


def _parseable_prefix(src: str) -> str | None:
    """Longest line-prefix of src that parses; None if none does."""
    lines = src.split("\n")
    for end in range(len(lines), 0, -1):
        cand = "\n".join(lines[:end])
        try:
            ast.parse(cand)
            return cand
        except SyntaxError:
            continue
    return None


def extract_code_candidates(prompt: str, gen: str) -> list[str]:
    """Extract parseable Python code candidates from generation, stripping <think> and markdown fences."""
    candidates = []

    # 1. Determine post-think region
    if "</think>" in gen:
        region = gen.split("</think>", 1)[1]
    elif "<think>" in gen:
        # Unclosed <think>: check if an embedded code fence exists first
        m = re.search(r"```(?:python)?\s*\n?(.*?)(?:```|$)", gen, re.DOTALL)
        region = m.group(1) if m else ""
    else:
        region = gen

    # 2. Extract code fence blocks if present
    fences = re.findall(r"```(?:python)?\s*\n?(.*?)(?:```|$)", region, re.DOTALL)
    blocks = [f.strip() for f in fences if f.strip()]
    if not blocks and region.strip():
        blocks = [region.strip()]

    # 3. For each block, construct re-declaration and completion style candidates
    for block in blocks:
        stripped = block.strip()
        # Case A: Re-declaration style (starts with def or class)
        if stripped.startswith(("def ", "class ", "async def ")):
            candidates.append(stripped)
            # If function name differed from prompt, alias it
            m_prompt = re.search(r"def\s+([a-zA-Z0-9_]+)", prompt)
            m_block = re.search(r"def\s+([a-zA-Z0-9_]+)", stripped)
            if m_prompt and m_block and m_prompt.group(1) != m_block.group(1):
                candidates.append(stripped + f"\n{m_prompt.group(1)} = {m_block.group(1)}")

        # Case B: Completion style (indented body or statements appended to prompt)
        indented_lines = []
        for line in block.splitlines():
            if line.strip():
                if line.startswith("    ") or line.startswith("\t"):
                    indented_lines.append(line)
                else:
                    indented_lines.append("    " + line)
            else:
                indented_lines.append("")
        candidates.append(prompt.rstrip() + "\n" + "\n".join(indented_lines))

        # Case C: Raw prompt + block
        candidates.append(prompt + "\n" + block)

    # Legacy fallback
    candidates.append(prompt + gen)
    return candidates


def run_code(prompt: str, generation: str, tests: str, timeout: float = 5.0) -> bool:
    cands = extract_code_candidates(prompt, generation)
    for cand in cands:
        src = _parseable_prefix(cand)
        if src is None:
            continue
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "probe.py"
            f.write_text(src + "\n\n" + tests + "\n", encoding="utf-8")
            try:
                r = subprocess.run([sys.executable, "-I", str(f)], cwd=td, capture_output=True, timeout=timeout)
                if r.returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                continue
    return False



_SENT = re.compile(r"[^.!?\n]+")
_NEG = re.compile(r"\b(not|no|never|isn't|aren't|wasn't|don't|doesn't)\b")


def _clean_sentences(gen: str, prompt: str) -> list[str]:
    """Lower-cased sentences of gen that are neither negated nor copied from the prompt."""
    p = prompt.lower()
    out = []
    for s in _SENT.findall(gen.lower()):
        s = s.strip()
        if s and not _NEG.search(s) and s not in p:
            out.append(s)
    return out


def score_record(rec: dict) -> bool:
    pid, gen = rec["id"], rec.get("generation", "")
    low = gen.lower()
    sents = _clean_sentences(gen, rec.get("prompt", ""))
    if pid in CODE_TESTS:
        return run_code(rec["prompt"], gen, CODE_TESTS[pid])
    if pid in MATH_FIRST_NUMBER:
        n = final_number(gen)
        return n is not None and any(abs(n - v) < 1e-9 for v in MATH_FIRST_NUMBER[pid])
    if pid in MATH_REGEX:
        if has_loop_repetition(gen):
            return False
        region = answer_region(gen)
        if pid == "math-07":  # restating "overtake the person in second place" is not an answer
            region = re.sub(r"(?i)\bthe person in (second|2nd) place\b", "", region)
        asents = _clean_sentences(region, rec.get("prompt", ""))
        return all(any(re.search(p, s) for s in asents) for p in MATH_REGEX[pid])
    if pid == "abst-06":  # must close </think> and give 85 after it
        return "</think>" in gen and "85" in gen.split("</think>", 1)[1]
    if pid == "abst-07":  # must close the ∴ block and say prime (not "not prime" / "not a prime")
        if has_loop_repetition(gen):
            return False
        return "∴" in gen and re.search(r"\bprime\b", low) is not None and re.search(r"\bnot (a )?prime\b", low) is None
    if pid in KEYWORDS:
        if has_loop_repetition(gen):
            return False
        if pid.startswith("abst-"):  # abstention phrases are themselves negations; only drop prompt echo
            return any(k.lower() in low for k in KEYWORDS[pid]) and low.strip() not in rec.get("prompt", "").lower()
        return any(k.lower() in s for s in sents for k in KEYWORDS[pid])
    raise KeyError(f"no scoring rule for probe {pid}")


def domain_of(pid: str) -> str:
    return {"code": "Coding", "math": "Arithmetic", "drav-maj": "Major Dravidian", "drav-low": "Low-Resource Dravidian",
            "eng": "English Reasoning", "abst": "Abstention"}[pid.rsplit("-", 1)[0]]


def summarize(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_probe: dict[str, list[bool]] = defaultdict(list)
    for r in recs:
        by_probe[r["id"]].append(score_record(r))
    dom: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for pid, passes in by_probe.items():
        dom[domain_of(pid)].append((sum(passes) / len(passes), any(passes)))
    k = max(len(v) for v in by_probe.values())
    rows = {d: (sum(p for p, _ in v), sum(a for _, a in v), len(v)) for d, v in dom.items()}
    total = tuple(sum(r[i] for r in rows.values()) for i in range(3))
    return {"file": str(path), "k": k, "domains": rows, "total": total,
            "passed_ids": sorted(p for p, v in by_probe.items() if any(v))}


def self_test() -> None:
    prompt = {"code-01": 'def reverse_string(s: str) -> str:\n    """Reverse the given string."""\n',
              "code-03": 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'}
    good = {"code-01": "    return s[::-1]\n", "code-03": "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"}
    for pid in good:
        assert score_record({"id": pid, "prompt": prompt[pid], "generation": good[pid]}), pid
        assert not score_record({"id": pid, "prompt": prompt[pid], "generation": '"""\n#define NUMBER_PARTITION 40\n'}), pid
    # trailing garbage after a correct body is cut off, not fatal
    assert score_record({"id": "code-01", "prompt": prompt["code-01"], "generation": "    return s[::-1]\n)))) ::::"})
    # code probes with <think> and markdown code blocks pass
    assert score_record({"id": "code-01", "prompt": prompt["code-01"], "generation": "<think>Reverse string s.</think>\n    return s[::-1]"})
    assert score_record({"id": "code-01", "prompt": prompt["code-01"], "generation": "<think>Reverse string s.</think>\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"})
    assert score_record({"id": "code-01", "prompt": prompt["code-01"], "generation": "<think>Working on it...\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"})

    # math: first number counts, echoing the operands does not
    assert score_record({"id": "math-01", "prompt": "", "generation": " 102\n\nNext question"})
    assert not score_record({"id": "math-01", "prompt": "", "generation": " 17 times 6 is 102"})
    assert not score_record({"id": "math-01", "prompt": "", "generation": "(X,0) What is 17 multiplied by 6? Answer: 38"})
    assert score_record({"id": "math-09", "prompt": "", "generation": " 1/4"})
    assert score_record({"id": "drav-maj-02", "prompt": "", "generation": " ೪೦"})
    assert score_record({"id": "math-08", "prompt": "", "generation": "Perimeter 34 cm, area 60 cm2"})
    assert not score_record({"id": "math-08", "prompt": "", "generation": "Perimeter 34 cm"})
    # the S3 report's credited-but-wrong cases score 0
    assert not score_record({"id": "abst-04", "prompt": "", "generation": "I have no idea what you mean by \"the medal\"."})
    assert score_record({"id": "abst-04", "prompt": "", "generation": "The 2036 Games have not happened yet."})
    assert not score_record({"id": "drav-low-09", "prompt": "", "generation": "ಈ ಭಾಷಾ ಕುಟುಂಬವು ಒಂದು ಭಾಷೆಯ ಒಂದು ಭಾಷೆಯಾಗಿರದೆ"})
    assert not score_record({"id": "abst-06", "prompt": "", "generation": "27 + 38 = 50 + 28"})
    assert score_record({"id": "abst-06", "prompt": "", "generation": "48+37 = 85\n</think>\n85"})
    assert first_number("ಉತ್ತರ: ೨೫") == 25.0
    # keyword luck seen in the S3 outputs must not pass
    assert not score_record({"id": "math-07", "prompt": "", "generation": "do some kind of fun with a second variable"})
    assert score_record({"id": "math-07", "prompt": "", "generation": "You are in second place."})
    assert not score_record({"id": "abst-08", "prompt": "", "generation": 'a "real" public or private bank account'})
    # negated or echoed keywords seen in the S1-S3 rebaseline must not pass
    assert not score_record({"id": "math-07", "prompt": "", "generation": "You are not in second place. What position?"})
    assert not score_record({"id": "eng-06", "prompt": "", "generation": "I am trying to understand why they are not warm-blooded."})
    p6 = "Premise 1: All mammals are warm-blooded.\nConclusion:"
    assert not score_record({"id": "eng-06", "prompt": p6, "generation": "\nPremise 1: All mammals are warm-blooded."})
    assert score_record({"id": "eng-06", "prompt": p6, "generation": " A blue whale is warm-blooded."})
    # S4 <think> outputs: the boxed / post-</think> answer counts, a number echoed in an open trace does not
    s4_math01 = "<think>\nAnswer: 17 multiplied by 6 is equal to $17 \\times 6 = 102$.\nTherefore, the answer is $\\boxed{102}$.\n</think>\n\\boxed{102}"
    assert score_record({"id": "math-01", "prompt": "", "generation": s4_math01})
    s4_math02 = "<think>\nTo solve the problem, we need to find a number that is divisible by both 12 and 4.\n\nStep 1: Divide 12 by 12.\n$$12 \\div 12 = 1$$"
    assert not score_record({"id": "math-02", "prompt": "", "generation": s4_math02})
    assert not score_record({"id": "math-02", "prompt": "", "generation": s4_math02 + "\nTherefore, the answer is $\\boxed{1}$.\n</think>\n\\boxed{1}"})
    assert score_record({"id": "math-02", "prompt": "", "generation": "<think>\n144 / 12 = 12\n</think>\nThe answer is: 12"})
    s4_math07 = ("<think>\n1. **Understanding the Problem:**\n\n   We are given a race where you overtake the person in second place."
                 " You overtake the person in second place, and then overtakes the person in third place.\n\n2. **Analyzing")
    assert not score_record({"id": "math-07", "prompt": "", "generation": s4_math07})
    assert not score_record({"id": "math-07", "prompt": "", "generation": "You overtake the person in second place, then the third."})
    assert score_record({"id": "math-07", "prompt": "", "generation": "<think>\nYou take their place.\n</think>\nYou are in second place."})
    assert not score_record({"id": "abst-07", "prompt": "", "generation": "∴ Since 29 is not a prime, it must be divisible by 3."})
    assert final_number("<think>\n48 + 37 = 95\n</think>\nThe answer is: 85") == 85.0

    # S5 real strings test: math-02 must pass, math-07 and abst-07 loops must fail
    s5_math02 = "144 divided by 12 is equal to 12. The answer is 12.\nAnswer: 144 divided by 12 is equal to 12.\nThe answer is 12.\n#### Explanation:\n\n144 divided by 12 equals 12.\nTherefore, the answer is 12."
    assert score_record({"id": "math-02", "prompt": "Question: What is 144 divided by 12?\nAnswer:", "generation": s5_math02})
    s5_math07 = ('1. The first person to reach the top of the race is called the "first person." This means that if you are in second place, you will be in second place.\n'
                 '2. The second person is in third place. This means that if you are in second place, you will be in second place.\n'
                 '3. The third person is in fourth place. This means that if you are in second place, you will be in third place.\n'
                 '4. The third person is in fourth place. This means that if you are in fourth place, you will be in fourth place.\n'
                 '5. The third person is in fourth place. This means that if you are in fourth place, you will be in fourth place.\n'
                 '6. The fourth person is in fourth place. This means that if you are in fourth place, you will be in fourth place.\n'
                 '7. The fifth person is in fourth place. This means that if you are in fourth place, you will be in fourth place.\n'
                 '8. The third person is in fourth place. This means that if you are in fourth place, you will be in fourth place.\n'
                 '9. The fourth person is in fourth place. This means that if you are in fourth place, you will be in fourth place.\n10.')
    assert not score_record({"id": "math-07", "prompt": "You are running a race and you overtake the person in second place. What position are you now in?\nAnswer:", "generation": s5_math07})
    s5_abst07 = "∴ 29 is prime.\n∴ 37 is prime.\n∴ 41 is prime.\n∴ 41 is prime.\n∴ 41 is prime."
    assert not score_record({"id": "abst-07", "prompt": "Evaluate whether 29 is prime.\n∴\n", "generation": s5_abst07})
    print("score_60p self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", help="write summaries to this JSON file")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    out = []
    for f in args.files:
        s = summarize(Path(f))
        out.append(s)
        print(f"\n== {Path(f).name}  (samples/probe = {s['k']})")
        for d, (p1, pk, n) in sorted(s["domains"].items()):
            print(f"  {d:24s} pass@1 {p1:5.2f}/{n}" + (f"   pass@{s['k']} {pk}/{n}" if s["k"] > 1 else ""))
        p1, pk, n = s["total"]
        print(f"  {'TOTAL':24s} pass@1 {p1:5.2f}/{n} ({100 * p1 / n:.1f}%)" + (f"   pass@{s['k']} {pk}/{n}" if s["k"] > 1 else ""))
        print(f"  passed: {', '.join(s['passed_ids']) or '-'}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
