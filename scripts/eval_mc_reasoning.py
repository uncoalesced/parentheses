#!/usr/bin/env python3
"""Likelihood multiple-choice score for eval/queries_120m_reason250.jsonl (Claude Code, 2026-10-13).

Works on base checkpoints (no generation, no judge): each item becomes gold + distractors and the model
picks the choice with the highest continuation log-likelihood (lm-eval style; also length-normalised).
Distractors (Joel Q12: dataset-provided first, Claude for the rest; GSM8K/MBPP ship none):
  math      numeric near-misses of the gold (x2, /2, +-10%, +-1)
  code      reference solutions of 3 other MBPP tasks
  logic     the opposite answer (Yes/No, True/False, Knight/Knave), other weekdays, nearby numbers,
            or other capitalised names from the prompt; items with no candidate are dropped (counted)
  reasoning eval/mc_reason_distractors.json (Claude-written, eval only)

    .venv\\Scripts\\python.exe scripts\\eval_mc_reasoning.py --checkpoint <ckpt> --out out\\mc_reason_step<N>.json
    .venv\\Scripts\\python.exe scripts\\eval_mc_reasoning.py --self-test
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PAIRS = {"yes": "No", "no": "Yes", "true": "False", "false": "True", "knight": "Knave", "knave": "Knight"}
STATES = ["solid", "liquid", "gas"]
STOP = {"the", "which", "who", "what", "is", "in", "if", "all", "some", "no", "every", "each", "a", "an", "question",
        "answer", "how", "when", "where", "there", "then", "therefore", "both", "neither", "none", "one", "only",
        "since", "because", "so", "yes", "true", "false", "she", "he", "they", "it", "this", "that", "after", "before"}


def num_distractors(g: str) -> list[str]:
    try:
        x = float(g.replace(",", ""))
    except ValueError:
        return []
    cands = [x * 2, x / 2, x * 1.1, x * 0.9, x + 1, x - 1, x + 10]
    out = []
    for c in cands:
        c = round(c) if float(x).is_integer() else round(c, 2)
        s = str(int(c)) if float(c).is_integer() else str(c)
        if s != g and s not in out and c >= 0:
            out.append(s)
    return out[:3]


def logic_distractors(prompt: str, gold: str) -> list[str]:
    g = gold.strip()
    if g.lower() in PAIRS:
        return [PAIRS[g.lower()]]
    if g in DAYS:
        i = DAYS.index(g)
        return [DAYS[(i + k) % 7] for k in (1, 2, 3)]
    if re.fullmatch(r"-?\d+", g):
        return num_distractors(g)
    if g.lower() in STATES:
        return [s.capitalize() for s in STATES if s != g.lower()]
    body = prompt.split("Answer:")[0].replace("Question:", " ")
    # capitalised words (names, "Car D", "Mountain Z") minus function words; a sentence-initial word counts
    # only if it recurs (puzzle names repeat, "Iron"/"If" at a sentence start do not)
    body = body.strip()
    toks = [(m.group(0), m.start()) for m in
            re.finditer(r"\b(?:(?:Car|Box|Mountain|Island|Room|Team) [A-Z]\b|[A-Z][a-z]+)", body)]
    initial = lambda pos: pos == 0 or re.search(r"[.?!:]\s*$", body[:pos]) is not None
    counts = {w: sum(1 for t, _ in toks if t == w) for w, _ in toks}
    words = [w for w, pos in toks if counts[w] > 1 or not initial(pos)]
    if " " in g:   # multi-word gold ("Blue box", "Station Alpha", "Door 1"): offer sibling phrases from the prompt
        head, tail = g.split(" ", 1)
        sib = [f"{w.capitalize()} {tail}" for w in re.findall(rf"\b(\w+) {re.escape(tail)}\b", body, re.I)
               if len(w) > 1 and w.lower() not in STOP]
        sib += [f"{head} {w}" for w in re.findall(rf"\b{re.escape(head)} (\w+)", body, re.I)
                if w[0].isupper() or w.isdigit()]
        words = sib + [w for w in words if not set(w.lower().split()) <= set(g.lower().split())]
    seen, out = {g.lower()} | STOP, []
    for w in words:
        if w.lower() not in seen and not set(w.lower().split()) <= set(g.lower().split()):
            seen.add(w.lower())
            out.append(w)
    return out[:3]


def build_items(rows: list[dict], reasoning_d: dict, seed: int = 7) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    code_refs = [r for r in rows if r["bucket"] == "code"]
    items, dropped = [], defaultdict(int)
    for r in rows:
        b, gold = r["bucket"], str(r["reference"]).strip()
        if b == "math":
            d = num_distractors(gold)
        elif b == "code":
            others = [c for c in code_refs if c["id"] != r["id"]]
            d = [str(c["reference"]) for c in rng.sample(others, min(3, len(others)))]
        elif b == "logic":
            d = logic_distractors(r["prompt"], gold)
        else:
            d = reasoning_d.get(r["id"], [])
        if not d:
            dropped[b] += 1
            continue
        choices = [gold] + d
        order = list(range(len(choices)))
        rng.shuffle(order)
        items.append({"id": r["id"], "bucket": b, "prompt": r["prompt"],
                      "choices": [choices[k] for k in order], "gold": order.index(0)})
    return items, dict(dropped)


def score(items: list[dict], cont_logprob, enc) -> dict:
    """cont_logprob(ctx_ids, cont_ids) -> (logprob, greedy); enc(str) -> ids."""
    by = defaultdict(lambda: {"n": 0, "acc": 0, "acc_norm": 0, "chance": 0.0})
    for it in items:
        ctx = enc(it["prompt"].rstrip())
        lps, norms = [], []
        for c in it["choices"]:
            ids = enc(" " + c.strip())
            lp = cont_logprob(ctx, ids)[0]
            lps.append(lp)
            norms.append(lp / max(1, len(ids)))
        s = by[it["bucket"]]
        s["n"] += 1
        s["acc"] += int(max(range(len(lps)), key=lps.__getitem__) == it["gold"])
        s["acc_norm"] += int(max(range(len(norms)), key=norms.__getitem__) == it["gold"])
        s["chance"] += 1 / len(it["choices"])
    out = {}
    for b, s in sorted(by.items()):
        out[b] = {"n": s["n"], "acc": round(s["acc"] / s["n"], 4), "acc_norm": round(s["acc_norm"] / s["n"], 4),
                  "chance": round(s["chance"] / s["n"], 4)}
    return out


def self_test():
    assert num_distractors("18")[:2] == ["36", "9"] and "18" not in num_distractors("18")
    assert logic_distractors("Q? Answer:", "Yes") == ["No"] and logic_distractors("x", "Monday")[0] == "Tuesday"
    d = logic_distractors("Question: Alice is taller than David. David is taller than Eric. If Eric grows, "
                          "who is tallest?\nAnswer:", "Alice")
    assert d == ["David", "Eric"], d
    assert logic_distractors("Question: Iron is a metal. Which state is it?\nAnswer:", "Solid") == ["Liquid", "Gas"]
    rows = [{"id": "m1", "bucket": "math", "prompt": "Question: 9*2?\nAnswer:", "reference": "18"},
            {"id": "c1", "bucket": "code", "prompt": "def f():", "reference": "return 1"},
            {"id": "c2", "bucket": "code", "prompt": "def g():", "reference": "return 2"},
            {"id": "r1", "bucket": "reasoning", "prompt": "Why?", "reference": "gold"},
            {"id": "r2", "bucket": "reasoning", "prompt": "Why not?", "reference": "gold2"}]
    items, dropped = build_items(rows, {"r1": ["wrong a", "wrong b"]})
    assert dropped == {"reasoning": 1} and len(items) == 4
    assert all(it["choices"][it["gold"]] == next(r["reference"] for r in rows if r["id"] == it["id"]) for it in items)
    gold_of = {it["prompt"].rstrip(): it["choices"][it["gold"]] for it in items}   # per prompt: code golds double as distractors
    enc = lambda s: [ord(ch) for ch in s]
    dec = lambda ids: "".join(map(chr, ids))
    oracle = lambda ctx, ids: (0.0 if dec(ids).strip() == gold_of[dec(ctx)] else -9.0, False)
    res = score(items, oracle, enc)
    assert all(v["acc"] == 1.0 for v in res.values()), res   # a scorer that prefers gold scores 1.0
    print("[self-test] eval_mc_reasoning ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--queries", default=os.path.join(ROOT, "eval", "queries_120m_reason250.jsonl"))
    p.add_argument("--distractors", default=os.path.join(ROOT, "eval", "mc_reason_distractors.json"))
    p.add_argument("--out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    import torch
    from tokenizers import Tokenizer
    from model.backbone import Parentheses
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from lm_eval_adapter import ParenthesesLM
    rows = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
    items, dropped = build_items(rows, json.load(open(a.distractors, encoding="utf-8")))
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model = Parentheses(ck["cfg"]).to(a.device).eval()
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(os.path.join(ck["tokenizer"]["path"], "tokenizer.json"))
    lm = ParenthesesLM(model, tok, "eng", a.device)
    res = {"checkpoint": a.checkpoint, "step": ck.get("step"), "items": len(items), "dropped": dropped,
           "by_bucket": score(items, lm._cont_logprob, lm._enc)}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
