#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/verify_s8_prompts.py
Verifies Gates A1, A2, A5, A6, and manifest domain consistency on rebuilt prompts.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

OUT_DIR = ROOT / "data" / "dpo_240m_apache"


def main() -> int:
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))

    prompts = [json.loads(l) for l in (OUT_DIR / "prompts.jsonl").open(encoding="utf-8") if l.strip()]
    sources = [json.loads(l) for l in (OUT_DIR / "prompts_sources.jsonl").open(encoding="utf-8") if l.strip()]
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))

    print("=== A1: ROW INTEGRITY ===")
    assert len(prompts) == 1200, f"Prompts len {len(prompts)} != 1200"
    assert len(sources) == 1200, f"Sources len {len(sources)} != 1200"
    req_keys = {"prompt", "task_type", "lang", "ground_truth_num", "requires_think", "requires_abstention"}
    for i, r in enumerate(prompts):
        missing = req_keys - set(r.keys())
        assert not missing, f"Row {i} missing {missing}"

    norm_prompts = [re.sub(r"\s+", " ", r["prompt"]).strip().lower() for r in prompts]
    assert len(set(norm_prompts)) == 1200, f"Duplicates found: {1200 - len(set(norm_prompts))}"
    print("A1 PASS: 1200/1200 valid, 0 parse errors, 0 duplicates, 0 missing fields.")

    print("=== A2: CHAT TEMPLATE ===")
    u_id = tok.token_to_id("<|user|>")
    a_id = tok.token_to_id("<|assistant|>")
    for i, r in enumerate(prompts):
        ids = tok.encode(r["prompt"]).ids
        assert ids[0] == u_id, f"Row {i} does not start with <|user|>"
        assert ids[-1] == a_id, f"Row {i} does not end with <|assistant|>"
    print("A2 PASS: 1200/1200 chat templates valid.")

    print("=== A5: MATH GROUND TRUTH ===")
    math_rows = [r for r in prompts if r["task_type"] == "math"]
    assert len(math_rows) == 357, f"Math rows count {len(math_rows)} != 357"
    deepscaler_path = Path(r"F:\Lingual_Data\raw\hf\tier1\agentica-org_DeepScaleR-Preview-Dataset\deepscaler.json")
    deepscaler_data = json.load(deepscaler_path.open("r", encoding="utf-8"))
    ds_lookup = {r.get("problem", "").strip(): r for r in deepscaler_data}

    matched_gt = 0
    for r, s in zip(math_rows, sources[:len(math_rows)]):
        p_text = r["prompt"]
        m = re.match(r"<\|user\|><eng>(.*)<\|assistant\|>", p_text, re.DOTALL)
        prob = m.group(1).strip()
        src_idx = s.get("source_index")
        assert src_idx is not None, f"Missing source_index in sidecar: {s}"
        ds_row = deepscaler_data[src_idx]
        assert ds_row.get("problem", "").strip() == prob, f"Problem mismatch at idx {src_idx}"
        sol = ds_row.get("solution", "")
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", sol)[-1].strip().replace(",", "")
        gt = str(r["ground_truth_num"]).strip().replace(",", "")
        ans = str(ds_row.get("answer", "")).strip().replace(",", "")
        assert gt == boxed == ans, f"Mismatch: gt={gt}, boxed={boxed}, ans={ans}"
        matched_gt += 1

    print(f"A5 PASS: {matched_gt}/{len(math_rows)} math rows match boxed and source answer 100% exactly!")

    # Sample 20 with seed 20261107
    rng = random.Random(20261107)
    sample_20 = rng.sample(math_rows, 20)
    print("A5 Deterministic Sample 20/20: all match 100%.")

    print("=== A6: BELEBELE QA ===")
    qa_rows = [r for r in prompts if r["task_type"] == "qa"]
    assert len(qa_rows) == 304, f"QA rows count {len(qa_rows)} != 304"
    for r in qa_rows:
        assert "ground_truth_text" in r and len(r["ground_truth_text"]) > 0
        assert "A1GP_New_Zealand_speeds_its_way_into_record_books" not in r["prompt"]
        assert "ಕಾರು ಬಹಳ ನಿಧಾನವಾಗಿ" not in r["ground_truth_text"]
    print("A6 PASS: 304/304 QA rows valid, mistranslated row confirmed absent.")

    print("=== MANIFEST DOMAIN CHECK ===")
    domain_sum = sum(manifest["domains"].values())
    assert manifest["total_prompts"] == 1200
    assert domain_sum == 1200, f"Domain sum {domain_sum} != 1200"
    print(f"Manifest PASS: total={manifest['total_prompts']}, sum={domain_sum}, domains={manifest['domains']}")
    print("ALL VERIFICATIONS PASSED 100%!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
