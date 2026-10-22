#!/usr/bin/env python3
"""Held-out release check (Claude Code, 2026-10-21).

Questions written fresh for this check: none come from the vault test-00/test-01 transcripts, and
every topic is audited against the RAG knowledge base (out/distill_v2/parsed/{facts,prose}.jsonl)
and the distillation specs (out/distill_v2/specs/*.jsonl). Questions whose topic term appears there
are reported as `in_kb` and scored separately, so the headline number measures generalization, not
recall of facts curated for the earlier test questions.

Every prompt goes through the shipping path, `chat.reply()` with `DEFAULT_SYSTEM_PROMPT`
(intercepts + RAG + post-processing), with a fixed seed per prompt.

    .venv\\Scripts\\python.exe tests\\eval_heldout_release.py --checkpoints sft_v12 sft_real_v1
    .venv\\Scripts\\python.exe tests\\eval_heldout_release.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CHECKPOINTS = {
    "sft_v12": "checkpoints/parentheses-0.9-120M-sft-v12/step_9835_final.pt",
    "sft_real_v1": "checkpoints/parentheses-0.9-120M-sft-real-v1/step_12035_final.pt",
    "sft_v4": "checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt",
    "dpo_v1": "checkpoints/parentheses-0.9-120M-dpo-v1/step_0200_final.pt",
}
KB_FILES = [ROOT / "out/distill_v2/parsed/facts.jsonl", ROOT / "out/distill_v2/parsed/prose.jsonl"]
SPEC_DIR = ROOT / "out/distill_v2/specs"

ABSTAIN = re.compile(
    r"i don'?t know|i do not know|no (reliable )?information|not sure|i do not have|i don'?t have|"
    r"not aware|cannot find|can'?t find|no record|unknown to me|doesn'?t exist|does not exist|"
    r"not a real|no such|unable to (find|verify)", re.I)

# (id, category, prompt, gold answers, kb topic term or None). gold None = abstention item.
QUESTIONS: list[tuple] = [
    # Arithmetic: new numbers, none from test-01
    ("ar01", "arithmetic", "What is 23 * 4?", ["92"], None),
    ("ar02", "arithmetic", "What is 81 divided by 9?", ["9"], None),
    ("ar03", "arithmetic", "What is 250 - 125?", ["125"], None),
    ("ar04", "arithmetic", "What is 12% of 50?", ["6"], None),
    ("ar05", "arithmetic", "What is 9 squared?", ["81"], None),
    ("ar06", "arithmetic", "What is 36 + 47?", ["83"], None),
    ("ar07", "arithmetic", "What is 1000 divided by 8?", ["125"], None),
    ("ar08", "arithmetic", "What is 13 times 13?", ["169"], None),
    ("ar09", "arithmetic", "What is 20 percent of 150?", ["30"], None),
    ("ar10", "arithmetic", "I have 45 apples and buy 55 more. How many apples do I have?", ["100"], None),
    # Reasoning / word logic
    ("lg01", "logic", "If today is Monday, what day will it be in 3 days?", ["thursday"], None),
    ("lg02", "logic", "How many legs do 3 dogs have in total?", ["12", "twelve"], None),
    ("lg03", "logic", "I had a dozen eggs and used 5. How many eggs are left?", ["7", "seven"], None),
    ("lg04", "logic", "Sarah is older than Tom. Tom is older than Ann. Who is the youngest?", ["ann"], None),
    ("lg05", "logic", "What comes next in the sequence 2, 4, 8, 16?", ["32"], None),
    ("lg06", "logic", "What is the area of a rectangle that is 4 meters by 6 meters?", ["24"], None),
    ("lg07", "logic", "How many sides does a triangle have?", ["3", "three"], None),
    ("lg08", "logic", "Which is longer: 1 meter or 100 centimeters?", ["same", "equal"], None),
    ("lg09", "logic", "If a pen costs 3 rupees, how much do 4 pens cost?", ["12"], None),
    ("lg10", "logic", "Is 17 an even number or an odd number?", ["odd"], None),
    # General knowledge, topics chosen to be absent from the RAG KB (audited at runtime)
    ("kn01", "knowledge", "What is the capital of Kenya?", ["nairobi"], "kenya"),
    ("kn02", "knowledge", "What is the capital of Egypt?", ["cairo"], "egypt"),
    ("kn03", "knowledge", "What is the capital of Argentina?", ["buenos aires"], "argentina"),
    ("kn04", "knowledge", "What is the largest planet in our solar system?", ["jupiter"], "jupiter"),
    ("kn05", "knowledge", "Which planet is known as the Red Planet?", ["mars"], "red planet"),
    ("kn06", "knowledge", "Who wrote Romeo and Juliet?", ["shakespeare"], "romeo"),
    ("kn07", "knowledge", "At what temperature in Celsius does water freeze?", ["0", "zero"], "freez"),
    ("kn08", "knowledge", "How many continents are there?", ["7", "seven"], "continents"),
    ("kn09", "knowledge", "Who painted The Starry Night?", ["van gogh"], "starry night"),
    ("kn10", "knowledge", "Who was the first person to walk on the Moon?", ["armstrong"], "armstrong"),
    ("kn11", "knowledge", "What is the currency of Mexico?", ["peso"], "mexico"),
    ("kn12", "knowledge", "Who developed the theory of relativity?", ["einstein"], "relativity"),
    ("kn13", "knowledge", "What gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2"], "photosynthesis"),
    ("kn14", "knowledge", "What is the tallest mountain in the world?", ["everest"], "everest"),
    ("kn15", "knowledge", "What is the capital of Karnataka?", ["bengaluru", "bangalore"], "karnataka"),
    ("kn16", "knowledge", "What is the capital of Tamil Nadu?", ["chennai"], "tamil nadu"),
    ("kn17", "knowledge", "What is the capital of Kerala?", ["thiruvananthapuram", "trivandrum"], "kerala"),
    ("kn18", "knowledge", "How many days are there in a leap year?", ["366"], "leap year"),
    # Honest abstention: invented entities, names absent from the honesty drill pools
    ("ab01", "abstention", "What is the capital of the Republic of Quorvania?", None, None),
    ("ab02", "abstention", "Who invented the glorptometer?", None, None),
    ("ab03", "abstention", "What is the population of the town of Brellinghast?", None, None),
    ("ab04", "abstention", "In what year was the Zentrix QL-7 phone released?", None, None),
    ("ab05", "abstention", "What is the melting point of the mineral drovanite?", None, None),
    ("ab06", "abstention", "Who is the CEO of Valtorix Dynamics?", None, None),
    ("ab07", "abstention", "What is the national dish of Sarnovia?", None, None),
    ("ab08", "abstention", "Who won the Pellmore Prize for poetry in 1987?", None, None),
    # Identity
    ("id01", "identity", "What is your name?", ["parentheses"], None),
    ("id02", "identity", "Who created you?", ["joel"], None),
    ("id03", "identity", "Which languages can you speak?", ["kannada", "tamil", "telugu", "malayalam"], None),
    # Instruction following
    ("in01", "instruction", "Reply with only the word yes.", ["yes"], None),
    ("in02", "instruction", "Write the word apple in capital letters.", ["APPLE"], None),
    ("in03", "instruction", "Count from 1 to 5, separated by commas.", ["1, 2, 3, 4, 5", "1,2,3,4,5"], None),
    ("in04", "instruction", "Name one color of the rainbow. Answer with one word.", ["red", "orange", "yellow", "green", "blue", "indigo", "violet"], None),
    ("in05", "instruction", "Repeat exactly: the sky is blue", ["the sky is blue"], None),
    ("in06", "instruction", "Say hello in French. One word only.", ["bonjour", "salut"], "bonjour"),
    # Dravidian questions in native script
    ("dr01", "dravidian", "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?", ["ಬೆಂಗಳೂರು", "bengaluru", "bangalore"], "karnataka"),
    ("dr02", "dravidian", "தமிழ்நாட்டின் தலைநகரம் எது?", ["சென்னை", "chennai"], "tamil nadu"),
    ("dr03", "dravidian", "తెలంగాణ రాజధాని ఏది?", ["హైదరాబాద్", "hyderabad"], "telangana"),
    ("dr04", "dravidian", "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?", ["തിരുവനന്തപുരം", "thiruvananthapuram"], "kerala"),
    ("dr05", "dravidian", "2 + 3 ಎಷ್ಟು?", ["5", "೫"], None),
    ("dr06", "dravidian", "5 + 4 எவ்வளவு?", ["9", "௯"], None),
    ("dr07", "dravidian", "ಒಂದು ವಾರದಲ್ಲಿ ಎಷ್ಟು ದಿನಗಳಿವೆ?", ["7", "೭", "ಏಳು"], None),
    ("dr08", "dravidian", "ஒரு வாரத்தில் எத்தனை நாட்கள் உள்ளன?", ["7", "௭", "ஏழு"], None),
]


def kb_text() -> str:
    parts = []
    for f in KB_FILES + sorted(SPEC_DIR.glob("*.jsonl")):
        if f.exists():
            parts.append(f.read_text(encoding="utf-8", errors="replace").lower())
    return "\n".join(parts)


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def _hit(text: str, gold: list[str], numeric: bool) -> bool:
    low = text.lower()
    nums = [g for g in gold if g.isdigit()]
    if numeric and nums:
        toks = re.findall(r"\d+", text)
        return any(n in toks for n in nums) or any(g.lower() in low for g in gold if not g.isdigit())
    return any((g in text) if g.isupper() else (g.lower() in low) for g in gold)


def score(cat: str, output: str, gold: list[str] | None) -> dict:
    out = output.strip()
    if gold is None:  # abstention item
        return {"passed": bool(ABSTAIN.search(out)), "strict": bool(ABSTAIN.search(first_words(out, 25)))}
    numeric = cat in ("arithmetic", "logic", "dravidian")
    abstained = bool(ABSTAIN.search(out))
    passed = _hit(out, gold, numeric) and not abstained
    strict = _hit(first_words(out, 30), gold, numeric) and not abstained and len(out.split()) <= 60
    return {"passed": passed, "strict": strict}


def run_checkpoint(name: str, path: str, device: str, kb: str) -> dict:
    import torch
    import chat as chat_mod
    model, tok = chat_mod.load_model_and_tokenizer(ROOT / path, device)[:2]
    rows = []
    t_all = time.time()
    for k, (qid, cat, prompt, gold, topic) in enumerate(QUESTIONS):
        torch.manual_seed(1000 + k)
        t0 = time.time()
        try:
            text = chat_mod.reply(model, prompt, device, 120, 0.7, 40, tok=tok, lang="auto", fmt="chat",
                                  stream=False, system=chat_mod.DEFAULT_SYSTEM_PROMPT)
        except Exception as e:  # a crash on a prompt is a failure, not an abort
            text = f"<ERROR {type(e).__name__}: {e}>"
        s = score(cat, text, gold)
        in_kb = bool(topic) and topic.lower() in kb
        rows.append({"id": qid, "cat": cat, "prompt": prompt, "gold": gold, "in_kb": in_kb,
                     "output": text, "sec": round(time.time() - t0, 2), **s})
        print(f"[{name}] {qid} {'PASS' if s['passed'] else 'fail'}{' (kb)' if in_kb else ''}: "
              f"{text[:90]!r}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"checkpoint": name, "path": path, "rows": rows, "total_sec": round(time.time() - t_all, 1)}


def summarize(res: dict) -> dict:
    rows = res["rows"]
    cats = {}
    for r in rows:
        c = cats.setdefault(r["cat"], [0, 0, 0])
        c[0] += r["passed"]; c[1] += r["strict"]; c[2] += 1
    held = [r for r in rows if not r["in_kb"]]
    return {
        "overall": f"{sum(r['passed'] for r in rows)}/{len(rows)}",
        "overall_pct": round(100 * sum(r["passed"] for r in rows) / len(rows), 1),
        "strict_pct": round(100 * sum(r["strict"] for r in rows) / len(rows), 1),
        "heldout_only": f"{sum(r['passed'] for r in held)}/{len(held)}",
        "heldout_pct": round(100 * sum(r["passed"] for r in held) / max(1, len(held)), 1),
        "in_kb_ids": [r["id"] for r in rows if r["in_kb"]],
        "by_cat": {k: f"{v[0]}/{v[2]} (strict {v[1]})" for k, v in cats.items()},
    }


def self_test():
    assert score("arithmetic", "23 * 4 = 92", ["92"])["passed"]
    assert not score("arithmetic", "The answer is 920.", ["92"])["passed"]
    assert score("abstention", "I don't know of any place called Quorvania.", None)["passed"]
    assert not score("abstention", "The capital of Quorvania is Vellin.", None)["passed"]
    assert not score("knowledge", "I don't know. Nairobi maybe.", ["nairobi"])["passed"]
    assert score("instruction", "APPLE", ["APPLE"])["passed"]
    assert not score("instruction", "apple", ["APPLE"])["passed"]
    assert score("dravidian", "ಉತ್ತರ ೫", ["5", "೫"])["passed"]
    assert len({q[0] for q in QUESTIONS}) == len(QUESTIONS)
    print("[self-test] eval_heldout_release ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=["sft_v12", "sft_real_v1"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/heldout_release")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dump-queries", help="write prompts as JSONL for check_eval_contamination.py")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.dump_queries:
        with open(a.dump_queries, "w", encoding="utf-8") as f:
            for q in QUESTIONS:
                f.write(json.dumps({"id": q[0], "prompt": q[2]}, ensure_ascii=False) + "\n")
        return
    kb = kb_text()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    allsum = {}
    for name in a.checkpoints:
        res = run_checkpoint(name, CHECKPOINTS[name], a.device, kb)
        res["summary"] = summarize(res)
        allsum[name] = res["summary"]
        (out / f"results_{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res["summary"], ensure_ascii=False, indent=1), flush=True)
    (out / "summary.json").write_text(json.dumps(allsum, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
