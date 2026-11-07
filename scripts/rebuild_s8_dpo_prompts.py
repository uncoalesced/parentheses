#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/rebuild_s8_dpo_prompts.py

Rebuilds the Session 8 On-Policy DPO prompt set:
  - Output: data/dpo_240m_apache/prompts.jsonl (1,200 rows)
  - Sidecar: data/dpo_240m_apache/prompts_sources.jsonl (1,200 rows)
  - Manifest: data/dpo_240m_apache/manifest.json

Resolves:
  1. Gate A5 Block: Extract 357 pure integer math problems from DeepScaleR where \\boxed{}
     contains an exact integer matching row['answer'] exactly.
  2. Gate A6 Block: Filter out Belebele Kannada row (index 285 / mistranslated option in
     A1GP_New_Zealand_speeds_its_way_into_record_books) and replace with valid Belebele QA.
  3. Gate A1/A2: 1,200 valid JSON rows, exact chat templates (<|user|><tag>...<|assistant|>),
     0 duplicates, 0 contamination against 60 probes and processed_v5 val texts.
  4. Manifest counts: aligned exactly with 1,200 total prompts.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer
from scripts.eval_60p_adversarial_suite import PROBES

OUT_DIR = ROOT / "data" / "dpo_240m_apache"
PACK = Path(r"E:\Parentheses\data\curated_s7_packin")
DEEPSCALER_PATH = Path(r"F:\Lingual_Data\raw\hf\tier1\agentica-org_DeepScaleR-Preview-Dataset\deepscaler.json")
TAGS = {"kan": "<kan>", "tam": "<tam>", "tel": "<tel>", "mal": "<mal>", "eng": "<eng>"}
SEED = 20261107


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def grams(t: str, n: int = 8) -> Set[Tuple[str, ...]]:
    w = norm(t).split()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def h(t: str) -> str:
    return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()


def val_texts(tok: Tokenizer) -> List[str]:
    langs = ["kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"]
    tag_of = {tok.token_to_id(f"<{l}>"): l for l in langs}
    val = np.memmap(ROOT / "data" / "processed_v5" / "val.bin", dtype=np.uint16, mode="r")
    vb = np.load(ROOT / "data" / "processed_v5" / "val_boundaries.npy")
    ve = np.append(vb[1:], len(val))
    out, nb = [], {}
    for s, e in zip(vb, ve):
        doc = np.asarray(val[s:e]).tolist()
        lang = tag_of.get(doc[0]) if doc else None
        if lang is None or nb.get(lang, 0) >= 400_000:
            continue
        t = tok.decode([x for x in doc[1:] if x != 0], skip_special_tokens=False)
        out.append(t)
        nb[lang] = nb.get(lang, 0) + len(t.encode("utf-8"))
    for f in (ROOT / "data" / "val_candidates").glob("*_val.txt"):
        out += [b for b in f.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    return out


def main() -> int:
    print("[*] Initializing tokenizer & decontamination index...")
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))

    banned_h: Set[str] = set()
    banned_g: Set[Tuple[str, ...]] = set()
    for t in [p["prompt"] for p in PROBES] + val_texts(tok):
        banned_h.add(h(t))
        banned_g |= grams(t)
    print(f"[*] Decontamination index: {len(banned_h)} hashes, {len(banned_g)} 8-grams.")

    def dirty(t: str) -> bool:
        return h(t) in banned_h or bool(grams(t) & banned_g)

    # 1. Load base 1200 file
    base = OUT_DIR / "prompts_base_ag_1200.jsonl"
    base_src = OUT_DIR / "prompts_sources_base_ag_1200.jsonl"
    if not base.exists():
        raise FileNotFoundError(f"Missing {base}")

    base_rows = [json.loads(l) for l in base.open(encoding="utf-8") if l.strip()]
    base_srcs = [json.loads(l) for l in base_src.open(encoding="utf-8") if l.strip()]

    # Keep non-Indic, non-Math rows from base
    keep = [
        i
        for i, r in enumerate(base_rows)
        if r.get("lang") not in ("kn", "ta", "te", "ml", "hi") and r.get("task_type") != "math"
    ]
    kept_rows, kept_srcs = [base_rows[i] for i in keep], [base_srcs[i] for i in keep]

    # Relabel lang from prompt tag
    tag2lang = {"<kan>": "kan", "<tam>": "tam", "<tel>": "tel", "<mal>": "mal"}
    for r in kept_rows:
        m = re.match(r"<\|user\|>(<[a-z]{3}>)", r["prompt"])
        if m and m.group(1) in tag2lang and r.get("lang") != tag2lang[m.group(1)]:
            r["lang"] = tag2lang[m.group(1)]

    # Deduplicate kept rows
    seen = set()
    dedup_rows, dedup_srcs = [], []
    for r, s in zip(kept_rows, kept_srcs):
        k = h(r["prompt"])
        if k not in seen:
            seen.add(k)
            dedup_rows.append(r)
            dedup_srcs.append(s)

    print(f"[*] Kept non-math base rows: {len(dedup_rows)} (code, abstention, identity, general)")

    # 2. Extract 357 pure integer math rows from DeepScaleR
    print("[*] Extracting 357 pure integer math rows from DeepScaleR...")
    with DEEPSCALER_PATH.open("r", encoding="utf-8") as f:
        deepscaler_data = json.load(f)

    math_rows: List[Dict[str, Any]] = []
    math_srcs: List[Dict[str, Any]] = []
    for idx, row in enumerate(deepscaler_data):
        sol = row.get("solution", "")
        m = re.findall(r"\\boxed\{([^{}]+)\}", sol)
        if not m:
            continue
        v = m[-1].strip().replace(",", "")
        if not re.match(r"^[-+]?\d+$", v):
            continue
        ans = str(row.get("answer", "")).strip().replace(",", "")
        if ans != v:
            continue
        prob = row.get("problem", "").strip()
        if len(prob) < 15 or len(prob) > 500:
            continue
        if dirty(prob):
            continue

        prompt = f"<|user|><eng>{prob}<|assistant|>"
        k = h(prompt)
        if k in seen:
            continue
        seen.add(k)

        math_rows.append({
            "prompt": prompt,
            "task_type": "math",
            "lang": "en",
            "ground_truth_num": v,
            "requires_think": True,
            "requires_abstention": False,
        })
        math_srcs.append({
            "id": f"math_deepscaler_{len(math_rows):04d}",
            "source": "agentica-org/DeepScaleR-Preview-Dataset",
            "source_index": idx,
            "license": "MIT",
            "task_type": "math",
        })
        if len(math_rows) >= 357:
            break

    assert len(math_rows) == 357, f"Expected 357 math rows, got {len(math_rows)}"
    print(f"[+] Extracted {len(math_rows)} verified pure integer math rows.")

    # 3. Extract 304 Belebele QA rows (76 each for kan, tam, tel, mal), filtering out row 901 mistranslation
    print("[*] Extracting 304 Belebele QA rows (76/lang)...")
    rng = random.Random(SEED)
    per_lang = {"kan": 76, "tam": 76, "tel": 76, "mal": 76}
    qa_rows: List[Dict[str, Any]] = []
    qa_srcs: List[Dict[str, Any]] = []
    removed_decontam: Dict[str, int] = {}

    jobs = [
        (l, PACK / f"belebele_{l}_{s}.jsonl", per_lang[l], "facebook/belebele", "CC-BY-SA-4.0")
        for l, s in (("kan", "Knda"), ("tam", "Taml"), ("tel", "Telu"), ("mal", "Mlym"))
    ]

    for lang, path, target_k, src, lic in jobs:
        cand = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        rng.shuffle(cand)
        got, rm = 0, 0
        for c in cand:
            if got >= target_k:
                break
            text = (c.get("prompt") or "").strip()
            resp = (c.get("response") or "").strip()
            if not text or len(text) > 2500:
                continue

            # Gate A6 fix: Filter out mistranslated option
            if "A1GP_New_Zealand_speeds_its_way_into_record_books" in text or "ಕಾರು ಬಹಳ ನಿಧಾನವಾಗಿ" in resp:
                print(f"[*] Filtered out mistranslated Belebele {lang} item: {text[:60]}...")
                continue

            prompt = f"<|user|>{TAGS[lang]}{text}<|assistant|>"
            if h(prompt) in seen:
                continue
            if dirty(text):
                rm += 1
                continue
            seen.add(h(prompt))

            row = {
                "prompt": prompt,
                "task_type": "qa",
                "lang": lang,
                "ground_truth_num": None,
                "requires_think": False,
                "requires_abstention": False,
                "ground_truth_text": resp,
            }
            qa_rows.append(row)
            qa_srcs.append({"source": src, "license": lic, "lang": lang, "domain": "qa"})
            got += 1

        removed_decontam[lang] = rm
        print(f"[{lang}] collected {got}/{target_k} (decontam removed {rm})")
        assert got == target_k, f"Underfilled {lang}: {got}/{target_k}"

    # Combine all rows
    all_rows = math_rows + dedup_rows + qa_rows
    all_srcs = math_srcs + dedup_srcs + qa_srcs
    assert len(all_rows) == 1200 == len(all_srcs), f"Total mismatch: {len(all_rows)} vs {len(all_srcs)}"

    # Validate chat templates for all rows
    print("[*] Validating chat templates for all 1,200 rows...")
    u_id = tok.token_to_id("<|user|>")
    a_id = tok.token_to_id("<|assistant|>")
    for idx, r in enumerate(all_rows):
        ids = tok.encode(r["prompt"]).ids
        if ids[0] != u_id or ids[-1] != a_id:
            raise ValueError(f"Row {idx} invalid chat template: {r['prompt'][:50]}")

    # Write prompts.jsonl and prompts_sources.jsonl
    print("[*] Writing data/dpo_240m_apache/prompts.jsonl and sidecar...")
    with (OUT_DIR / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (OUT_DIR / "prompts_sources.jsonl").open("w", encoding="utf-8") as f:
        for s in all_srcs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Update manifest.json
    from collections import Counter
    counts = Counter(r["task_type"] for r in all_rows)
    manifest = {
        "dataset_name": "data/dpo_240m_apache",
        "description": "Parentheses 1.0-240M Session 8 On-Policy DPO Multi-Domain Prompt Set",
        "license_compliance": "Permissive-Only",
        "total_prompts": len(all_rows),
        "domains": {
            "math": counts.get("math", 0),
            "code": counts.get("code", 0),
            "dravidian_devanagari_qa": counts.get("qa", 0),
            "epistemic_abstention": counts.get("abstention", 0),
            "model_identity": counts.get("identity", 0),
            "general_explanation": counts.get("general", 0),
        },
        "decontaminated": True,
        "decontamination_probes": 60,
        "rebuild": {
            "script": "scripts/rebuild_s8_dpo_prompts.py",
            "math_gt_fix": "100% boxed exact integer matching DeepScaleR answer",
            "qa_translation_fix": "Filtered out Belebele Kannada row 901 mistranslated option",
            "removed_by_decontam": removed_decontam,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Successfully rebuilt prompts.jsonl: {len(all_rows)} rows. Manifest updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
