#!/usr/bin/env python3
"""Generate DPO preference pairs for honesty: fabricated answers vs direct abstention.

Prompts are invented-entity questions whose names are disjoint from the P2.1 gate
prompts and from the SFT drill pools. The model's own non-abstaining replies are
the rejected completions (on-policy); chosen completions are direct IDK phrases.

Usage:
  .venv\\Scripts\\python.exe scripts\\gen_dpo_pairs.py \\
      --checkpoint checkpoints/parentheses-0.9-120M-sft-v10/step_9785_final.pt \\
      --out out/dpo_pairs_v1.jsonl --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import chat

ABSTAIN = (
    "i don't know", "i don’t know", "i do not know", "i don't have", "i do not have",
    "no information", "i'm not sure", "i am not sure", "not sure", "cannot confirm",
    "can't confirm", "no reliable", "not aware", "i haven't", "i have no",
)
BANNED = ("analyzing '", "okay, so i have this problem", "wait, let me guess")

CHOSEN = [
    "I don't know.",
    "I don't have that information.",
    "I don't have information about that.",
    "I'm not sure about that.",
    "I don't know the answer to that.",
    "I don't have reliable information on that.",
]

SYLL_A = ["kel", "vor", "mar", "tal", "zen", "dru", "qel", "rin", "sol", "bav", "tor", "nur",
          "gal", "mir", "pel", "sad", "vex", "lom", "har", "jov"]
SYLL_B = ["varia", "donia", "mera", "thon", "gard", "stan", "onia", "vash", "oria", "land",
          "mark", "heim", "port", "vale", "ridge", "fell", "more", "wick", "dale", "shire"]
SYLL_C = ["ite", "ium", "ene", "one", "ide", "ate", "yl", "ose", "ane", "yne"]
LETTERS = "QRSTVWXY"

PRODUCT_TPL = [
    "What is the release date of the {p}?",
    "Who invented the {p}?",
    "What is the {p} used for?",
    "How much does the {p} weigh?",
    "What is the battery life of the {p}?",
]
PLACE_TPL = [
    "What is the capital of {p}?",
    "What is the population of {p}?",
    "When was {p} founded?",
    "Who is the current president of {p}?",
    "What currency is used in {p}?",
    "What language is spoken in {p}?",
]
MATERIAL_TPL = [
    "What is the chemical symbol for {p}?",
    "What is the boiling point of {p}?",
    "Who discovered {p}?",
    "What is {p} made of?",
]
PERSON_TPL = [
    "What did {p} discover?",
    "When was {p} born?",
    "What is {p} known for?",
]
WORK_TPL = [
    "Who wrote {p}?",
    "Who composed {p}?",
    "What is the plot of {p}?",
]


def make_names(rng: random.Random, n: int) -> dict[str, list[str]]:
    places = set()
    while len(places) < n:
        places.add(rng.choice(SYLL_A) + rng.choice(SYLL_B))
    prods = set()
    while len(prods) < n:
        prods.add(f"{rng.choice(LETTERS)}{rng.choice(LETTERS)}-{rng.randint(1000, 9999)}")
        prods.add(f"{rng.choice(SYLL_A).capitalize()}{rng.randint(2, 9)}")
    mats = set()
    while len(mats) < n:
        mats.add(rng.choice(SYLL_A) + rng.choice(SYLL_C))
    people = set()
    first = ["Elara", "Tomas", "Mira", "Doran", "Sela", "Ravi", "Nika", "Oren", "Lena", "Kiran"]
    last = ["Voss", "Ryn", "Calder", "Mistral", "Ondar", "Perrin", "Solvei", "Bram", "Yara", "Quill"]
    while len(people) < n:
        people.add(f"{rng.choice(first)} {rng.choice(last)}")
    works = set()
    words = ["Echoes", "Silence", "Rivers", "Shadows", "Garden", "Winter", "Lanterns", "Tides",
             "Mirage", "Orchard", "Compass", "Embers"]
    while len(works) < n:
        works.add(f"The {rng.choice(words)} of {rng.choice(SYLL_A).capitalize()}{rng.choice(SYLL_B)}")
    return {"places": sorted(places), "products": sorted(prods), "materials": sorted(mats),
            "people": sorted(people), "works": sorted(works)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="out/dpo_pairs_v1.jsonl")
    ap.add_argument("--facts", default="", help="parsed facts.jsonl for known-answer pairs")
    ap.add_argument("--known-limit", type=int, default=80)
    ap.add_argument("--n-names", type=int, default=30)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    names = make_names(rng, args.n_names)
    prompts = []
    for p in names["places"]:
        prompts += [t.format(p=p) for t in rng.sample(PLACE_TPL, 3)]
    for p in names["products"]:
        prompts += [t.format(p=p) for t in rng.sample(PRODUCT_TPL, 1)]
    for p in names["materials"]:
        prompts += [t.format(p=p) for t in rng.sample(MATERIAL_TPL, 2)]
    for p in names["people"]:
        prompts += [t.format(p=p) for t in rng.sample(PERSON_TPL, 1)]
    for p in names["works"]:
        prompts += [t.format(p=p) for t in rng.sample(WORK_TPL, 1)]
    rng.shuffle(prompts)
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"[pairs] {len(prompts)} prompts, {args.samples} samples each")

    model, tok, _cfg, step = chat.load_model_and_tokenizer(args.checkpoint, device=args.device)
    print(f"[pairs] checkpoint step {step} on {args.device}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    n_skip = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i, prompt in enumerate(prompts, 1):
            for _ in range(args.samples):
                text = chat.reply(
                    model, prompt, args.device, args.max_new_tokens,
                    args.temperature, args.top_k, tok=tok, lang="eng", fmt="chat",
                    stream=False, system=chat.DEFAULT_SYSTEM_PROMPT,
                )
                low = text.lower().strip()
                if (not low or len(low) < 8 or any(a in low for a in ABSTAIN)
                        or any(b in low for b in BANNED)):
                    n_skip += 1
                    continue
                row = {"prompt": prompt, "chosen": rng.choice(CHOSEN),
                       "rejected": text.strip(), "kind": "unknown"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_pairs += 1
            if i % 25 == 0:
                dt = time.time() - t0
                print(f"[pairs] {i}/{len(prompts)} pairs={n_pairs} skipped={n_skip} "
                      f"{dt/i:.2f}s/prompt", flush=True)

        if args.facts and Path(args.facts).exists():
            fact_rows = [json.loads(l) for l in Path(args.facts).read_text(
                encoding="utf-8").splitlines() if l.strip()]
            rng.shuffle(fact_rows)
            known = 0
            for row in fact_rows:
                if known >= args.known_limit:
                    break
                q = (row.get("q") or "").strip()
                a = (row.get("a") or "").strip()
                if not q or not a or len(a) > 120 or len(a) < 3:
                    continue
                text = chat.reply(
                    model, q, args.device, args.max_new_tokens,
                    args.temperature, args.top_k, tok=tok, lang="eng", fmt="chat",
                    stream=False, system=chat.DEFAULT_SYSTEM_PROMPT,
                )
                low = text.lower().strip()
                if not low:
                    continue
                if a.lower().strip() in low:
                    continue
                f.write(json.dumps({"prompt": q, "chosen": a, "rejected": text.strip(),
                                    "kind": "known"}, ensure_ascii=False) + "\n")
                n_pairs += 1
                known += 1
            print(f"[pairs] known pairs: {known}", flush=True)
    print(f"[pairs] wrote {n_pairs} pairs to {out_path} (skipped {n_skip})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
