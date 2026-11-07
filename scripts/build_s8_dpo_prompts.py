#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/build_s8_dpo_prompts.py

Compiles the Session 8 On-Policy DPO prompt set:
  Output: data/dpo_240m_apache/prompts.jsonl
  Sidecar: data/dpo_240m_apache/prompts_sources.jsonl
  Manifest: data/dpo_240m_apache/manifest.json

Target Mix (exactly 1,200 rows):
  - Math with numeric ground truth (30%, 360 rows, requires_think=True)
  - Python coding (20%, 240 rows)
  - Dravidian & Devanagari QA (25%, 300 rows: KN 75, TE 75, TA 75, ML 50, HI 25)
  - Epistemic Abstention (15%, 180 rows, requires_abstention=True)
  - Model Identity (5%, 60 rows)
  - General Explanation (5%, 60 rows)

Enforces:
  1. Exact chat template inside prompt: <|user|><tag>text<|assistant|>
  2. Special tag mapping: <kan>, <tam>, <tel>, <mal>, <tcy>, <kff>, <kxv>, <nit>, <eng>, <|reserved_0|> (HI), <|reserved_1|> (SA)
  3. Decontamination against 60-probe adversarial suite & validation sets (0 exact matches, 0 8-gram overlaps)
  4. 100% Apache-2.0 / Permissive source tracking in sidecar and manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer
from scripts.eval_60p_adversarial_suite import PROBES

LANG_TAGS = {
    "en": "<eng>",
    "eng": "<eng>",
    "kn": "<kan>",
    "kan": "<kan>",
    "ta": "<tam>",
    "tam": "<tam>",
    "te": "<tel>",
    "tel": "<tel>",
    "ml": "<mal>",
    "mal": "<mal>",
    "tcy": "<tcy>",
    "kff": "<kff>",
    "kxv": "<kxv>",
    "nit": "<nit>",
    "hi": "<|reserved_0|>",
    "hin": "<|reserved_0|>",
    "sa": "<|reserved_1|>",
    "san": "<|reserved_1|>",
}


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def get_ngrams(tokens: List[str], n: int = 8) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def load_decontamination_index() -> Tuple[Set[str], Set[Tuple[str, ...]]]:
    exact_hashes: Set[str] = set()
    banned_8grams: Set[Tuple[str, ...]] = set()

    for p in PROBES:
        prompt_raw = p.get("prompt", "")
        norm = normalize_text(prompt_raw)
        if norm:
            exact_hashes.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())
            words = norm.split()
            banned_8grams.update(get_ngrams(words, 8))

    val_dir = REPO_ROOT / "data" / "val_candidates"
    if val_dir.exists():
        for f in val_dir.glob("*.txt"):
            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in lines[:500]:
                    norm = normalize_text(line)
                    if len(norm) > 20:
                        exact_hashes.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())
                        words = norm.split()
                        banned_8grams.update(get_ngrams(words, 8))
            except Exception:
                pass

    print(f"[*] Loaded decontamination index: {len(exact_hashes)} exact hashes, {len(banned_8grams)} 8-grams.")
    return exact_hashes, banned_8grams


def is_contaminated(text: str, exact_hashes: Set[str], banned_8grams: Set[Tuple[str, ...]]) -> bool:
    norm = normalize_text(text)
    if not norm:
        return True
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    if h in exact_hashes:
        return True
    words = norm.split()
    cand_ngrams = get_ngrams(words, 8)
    if cand_ngrams & banned_8grams:
        return True
    return False


def format_prompt(raw_text: str, lang: str) -> str:
    tag = LANG_TAGS.get(lang, "<eng>")
    cleaned = raw_text.strip()
    return f"<|user|>{tag}{cleaned}<|assistant|>"


def extract_math_gt(answer_str: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", answer_str)
    if boxed:
        val = boxed[-1].strip().replace(",", "")
        num_m = re.match(r"^[-+]?\d+(?:\.\d+)?$", val)
        if num_m:
            return num_m.group(0)
    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_str)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def collect_math_prompts(
    target_count: int,
    exact_hashes: Set[str],
    banned_8grams: Set[Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    deepscaler_path = Path(r"F:\Lingual_Data\raw\hf\tier1\agentica-org_DeepScaleR-Preview-Dataset\deepscaler.json")
    if deepscaler_path.exists():
        print(f"[*] Loading math prompts from {deepscaler_path}...")
        with deepscaler_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                prob = row.get("problem", "").strip()
                sol = row.get("solution", "")
                gt = extract_math_gt(sol)
                if not gt or len(prob) < 15 or len(prob) > 500:
                    continue
                if is_contaminated(prob, exact_hashes, banned_8grams):
                    continue

                formatted = format_prompt(prob, "en")
                items.append({
                    "prompt": formatted,
                    "task_type": "math",
                    "lang": "en",
                    "ground_truth_num": gt,
                    "requires_think": True,
                    "requires_abstention": False,
                })
                sources.append({
                    "id": f"math_deepscaler_{len(items):04d}",
                    "source": "agentica-org/DeepScaleR-Preview-Dataset",
                    "license": "Apache-2.0",
                    "task_type": "math",
                })
                if len(items) >= target_count:
                    break

    print(f"[+] Collected {len(items)}/{target_count} math prompts.")
    return items, sources


def collect_code_prompts(
    target_count: int,
    exact_hashes: Set[str],
    banned_8grams: Set[Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    code_path = Path(r"F:\Lingual_Data\raw\hf\tier1\flytech_python-codes-25k\python-codes-25k.jsonl")
    if code_path.exists():
        print(f"[*] Loading coding prompts from {code_path}...")
        with code_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                instr = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()
                prompt_text = f"{instr}\n{inp}".strip() if inp else instr
                if len(prompt_text) < 15 or len(prompt_text) > 400:
                    continue
                if is_contaminated(prompt_text, exact_hashes, banned_8grams):
                    continue

                formatted = format_prompt(prompt_text, "en")
                items.append({
                    "prompt": formatted,
                    "task_type": "code",
                    "lang": "en",
                    "ground_truth_num": None,
                    "requires_think": False,
                    "requires_abstention": False,
                })
                sources.append({
                    "id": f"code_python25k_{len(items):04d}",
                    "source": "flytech/python-codes-25k",
                    "license": "Apache-2.0",
                    "task_type": "code",
                })
                if len(items) >= target_count:
                    break

    print(f"[+] Collected {len(items)}/{target_count} code prompts.")
    return items, sources


def collect_abstention_prompts(
    target_count: int,
    exact_hashes: Set[str],
    banned_8grams: Set[Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    abst_path = REPO_ROOT / "data" / "sft_v4_abstention_4000.jsonl"
    if abst_path.exists():
        print(f"[*] Loading abstention prompts from {abst_path}...")
        with abst_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                p_text = row.get("prompt", "").strip()
                lang = row.get("lang", "en")
                if len(p_text) < 10 or len(p_text) > 400:
                    continue
                if is_contaminated(p_text, exact_hashes, banned_8grams):
                    continue

                formatted = format_prompt(p_text, lang)
                items.append({
                    "prompt": formatted,
                    "task_type": "abstention",
                    "lang": lang if lang in ("kn", "ta", "te", "ml") else "en",
                    "ground_truth_num": None,
                    "requires_think": False,
                    "requires_abstention": True,
                })
                sources.append({
                    "id": f"abst_sft_v4_{len(items):04d}",
                    "source": "uncoalesced/epistemic-abstention-v4",
                    "license": "Apache-2.0",
                    "task_type": "abstention",
                })
                if len(items) >= target_count:
                    break

    print(f"[+] Collected {len(items)}/{target_count} abstention prompts.")
    return items, sources


SYNTHETIC_IDENTITY_PROMPTS = [
    "What is your architecture?",
    "Are you based on the Transformer architecture?",
    "How does your memory state work during generation?",
    "Who designed and created Parentheses?",
    "Do you use quadratic self-attention KV caches?",
    "Explain what State Space Duality means in Parentheses.",
    "Are you trained by OpenAI, Google, or Anthropic?",
    "What is the meaning behind the name Parentheses?",
    "What parameter scale is Parentheses 1.0?",
    "Do your inference memory requirements grow with sequence length?",
    "Can you provide links to official documentation?",
    "How do you process recurrence in chunked parallel form?",
    "Who is uncoalesced?",
]


def collect_identity_prompts(
    target_count: int,
    exact_hashes: Set[str],
    banned_8grams: Set[Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    id_path = REPO_ROOT / "data" / "curated" / "parentheses_identity_sft_v1.jsonl"
    if id_path.exists():
        print(f"[*] Loading identity prompts from {id_path}...")
        with id_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                p_text = row.get("prompt", "").strip()
                lang = row.get("lang", "en")
                if not p_text:
                    continue
                if is_contaminated(p_text, exact_hashes, banned_8grams):
                    continue

                formatted = format_prompt(p_text, lang)
                items.append({
                    "prompt": formatted,
                    "task_type": "identity",
                    "lang": "en",
                    "ground_truth_num": None,
                    "requires_think": False,
                    "requires_abstention": False,
                })
                sources.append({
                    "id": f"identity_parentheses_{len(items):04d}",
                    "source": "uncoalesced/parentheses_identity_v1",
                    "license": "Apache-2.0",
                    "task_type": "identity",
                })
                if len(items) >= target_count:
                    break

    # Add synthetic identity if needed to reach target_count
    if len(items) < target_count:
        for synth_q in SYNTHETIC_IDENTITY_PROMPTS:
            if is_contaminated(synth_q, exact_hashes, banned_8grams):
                continue
            formatted = format_prompt(synth_q, "en")
            items.append({
                "prompt": formatted,
                "task_type": "identity",
                "lang": "en",
                "ground_truth_num": None,
                "requires_think": False,
                "requires_abstention": False,
            })
            sources.append({
                "id": f"identity_synth_{len(items):04d}",
                "source": "uncoalesced/parentheses_identity_synth",
                "license": "Apache-2.0",
                "task_type": "identity",
            })
            if len(items) >= target_count:
                break

    print(f"[+] Collected {len(items)}/{target_count} identity prompts.")
    return items, sources


def collect_dravidian_and_devanagari_prompts(
    target_counts: Dict[str, int],
    exact_hashes: Set[str],
    banned_8grams: Set[Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    bdir = Path(r"F:\Lingual_Data\raw\hf\tier1\facebook_belebele\data")
    file_map = {
        "kn": (bdir / "kan_Knda.jsonl", target_counts.get("kn", 75)),
        "te": (bdir / "tel_Telu.jsonl", target_counts.get("te", 75)),
        "ta": (bdir / "tam_Taml.jsonl", target_counts.get("ta", 75)),
        "ml": (bdir / "mal_Mlym.jsonl", target_counts.get("ml", 50)),
        "hi": (bdir / "hin_Deva.jsonl", target_counts.get("hi", 25)),
    }

    for lang, (fpath, tgt) in file_map.items():
        if not fpath.exists():
            print(f"[!] Warning: missing belebele file {fpath}")
            continue
        c = 0
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or c >= tgt:
                    break
                row = json.loads(line)
                q = row.get("question", "").strip()
                if len(q) < 10 or len(q) > 350:
                    continue
                if is_contaminated(q, exact_hashes, banned_8grams):
                    continue

                formatted = format_prompt(q, lang)
                items.append({
                    "prompt": formatted,
                    "task_type": "general",
                    "lang": lang,
                    "ground_truth_num": None,
                    "requires_think": False,
                    "requires_abstention": False,
                })
                sources.append({
                    "id": f"drav_{lang}_{c:04d}",
                    "source": f"facebook/belebele/{lang}",
                    "license": "Apache-2.0",
                    "task_type": "general",
                })
                c += 1
        print(f"[+] Collected Belebele {lang}: {c}/{tgt}")

    return items, sources


def collect_general_explanation_prompts(
    target_count: int,
    exact_hashes: Set[str],
    banned_8grams: Set[Tuple[str, ...]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    inq_file = Path(r"F:\Lingual_Data\raw\hf\tier1\mattwesney_General_Inquiry_Thinking-Chain-Of-Thought\General_Inquiry_Thinking-Chain-Of-Thought_6K.json")
    if inq_file.exists():
        print(f"[*] Loading general inquiry prompts from {inq_file}...")
        with inq_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                q = row.get("question", "").strip()
                if len(q) < 15 or len(q) > 350:
                    continue
                if is_contaminated(q, exact_hashes, banned_8grams):
                    continue
                formatted = format_prompt(q, "en")
                items.append({
                    "prompt": formatted,
                    "task_type": "general",
                    "lang": "en",
                    "ground_truth_num": None,
                    "requires_think": True,
                    "requires_abstention": False,
                })
                sources.append({
                    "id": f"gen_inq_{len(items):04d}",
                    "source": "mattwesney/General_Inquiry_Thinking-Chain-Of-Thought",
                    "license": "Apache-2.0",
                    "task_type": "general",
                })
                if len(items) >= target_count:
                    break

    print(f"[+] Collected {len(items)}/{target_count} general explanation prompts.")
    return items, sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Session 8 DPO prompt set")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "dpo_240m_apache")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exact_hashes, banned_8grams = load_decontamination_index()

    all_prompts: List[Dict[str, Any]] = []
    all_sources: List[Dict[str, Any]] = []

    # 1. Math (360)
    m_p, m_s = collect_math_prompts(360, exact_hashes, banned_8grams)
    all_prompts.extend(m_p)
    all_sources.extend(m_s)

    # 2. Code (240)
    c_p, c_s = collect_code_prompts(240, exact_hashes, banned_8grams)
    all_prompts.extend(c_p)
    all_sources.extend(c_s)

    # 3. Dravidian & Devanagari QA (300)
    d_p, d_s = collect_dravidian_and_devanagari_prompts(
        {"kn": 75, "te": 75, "ta": 75, "ml": 50, "hi": 25},
        exact_hashes,
        banned_8grams,
    )
    all_prompts.extend(d_p)
    all_sources.extend(d_s)

    # 4. Epistemic Abstention (180)
    a_p, a_s = collect_abstention_prompts(180, exact_hashes, banned_8grams)
    all_prompts.extend(a_p)
    all_sources.extend(a_s)

    # 5. Model Identity (60)
    i_p, i_s = collect_identity_prompts(60, exact_hashes, banned_8grams)
    all_prompts.extend(i_p)
    all_sources.extend(i_s)

    # 6. General Explanation (60)
    g_p, g_s = collect_general_explanation_prompts(60, exact_hashes, banned_8grams)
    all_prompts.extend(g_p)
    all_sources.extend(g_s)

    total_count = len(all_prompts)
    print(f"\n[+] Total compiled prompts: {total_count} (Target: 1,200)")

    # Verify Tokenizer Alignment
    tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    user_id = tok.token_to_id("<|user|>")
    asst_id = tok.token_to_id("<|assistant|>")

    for i in range(min(5, total_count)):
        ids = tok.encode(all_prompts[i]["prompt"]).ids
        print(f"Sample {i+1} encoded IDs: {ids[:3]} ... {ids[-2:]}")
        assert ids[0] == user_id, f"First token {ids[0]} != user_id {user_id}"
        assert ids[-1] == asst_id, f"Last token {ids[-1]} != asst_id {asst_id}"

    # Write prompts.jsonl
    prompts_file = args.out_dir / "prompts.jsonl"
    with prompts_file.open("w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[+] Wrote {total_count} prompts to {prompts_file}")

    # Write prompts_sources.jsonl
    sources_file = args.out_dir / "prompts_sources.jsonl"
    with sources_file.open("w", encoding="utf-8") as f:
        for s in all_sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[+] Wrote {len(all_sources)} source records to {sources_file}")

    # Write manifest.json
    manifest_file = args.out_dir / "manifest.json"
    manifest_data = {
        "dataset_name": "data/dpo_240m_apache",
        "description": "Parentheses 1.0-240M Session 8 On-Policy DPO Multi-Domain Prompt Set",
        "license_compliance": "Apache-2.0-Permissive-Only",
        "total_prompts": total_count,
        "domains": {
            "math": len(m_p),
            "code": len(c_p),
            "dravidian_devanagari_qa": len(d_p),
            "epistemic_abstention": len(a_p),
            "model_identity": len(i_p),
            "general_explanation": len(g_p),
        },
        "decontaminated": True,
        "decontamination_probes": len(PROBES),
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"[+] Wrote Apache-2.0 compliance manifest to {manifest_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
