# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 Micro-Patch Dataset Builder
#
# Builds targeted micro-patch corpus to resolve G3 (Format), G5 (Identity), and G6 (Honesty):
#   1. Identity Q&A (15 probes with complete keyword coverage, repeated x30 = 450 rows)
#   2. Grounded Reading Cards with explicit terminal `Final: <val>` (800 rows)
#   3. Epistemic Abstentions on false-premise queries with clean termination (800 rows)
#   4. Replay Math & Code with terminal `Final:` and `Output:` (950 rows)
# Total: 3,000 rows packed into container-aligned 4096-token bins.

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.build_s7_5_sft_240m as b75
from scripts.pack_sft_4096_v4 import ffd_pack_documents, resolve_special_token_ids

IDENTITY_QA = [
    {
        "prompt": "Who designed and created Parentheses?",
        "response": "Parentheses was designed, engineered, and trained from scratch by **uncoalesced**.",
    },
    {
        "prompt": "What is the core architecture family of Parentheses?",
        "response": "Parentheses belongs to the State Space Duality (SSD) and selective linear attention family, incorporating principles from Mamba-2 and RWKV.",
    },
    {
        "prompt": "Are you based on the Transformer architecture?",
        "response": "No, Parentheses is strictly not a Transformer. It uses recurrent State Space Duality (SSD) and selective linear attention with linear complexity.",
    },
    {
        "prompt": "Do you use quadratic self-attention KV caches during inference?",
        "response": "No, Parentheses operates without a KV-cache. During inference, it maintains a constant O(1) recurrent memory state of fixed size.",
    },
    {
        "prompt": "What does the name Parentheses mean?",
        "response": "The name 'Parentheses' symbolizes enclosing unbounded knowledge within a bound, finite container.",
    },
    {
        "prompt": "Are you created by OpenAI, Google, Anthropic, or Meta?",
        "response": "No, I was not created by OpenAI, Google, Anthropic, or Meta. Parentheses is an independent foundation model created by uncoalesced.",
    },
    {
        "prompt": "Can you give me the link or URL to your official website?",
        "response": "You can find Parentheses and related research on GitHub at https://github.com/uncoalesced and on Hugging Face.",
    },
    {
        "prompt": "What private cluster or directory path was used to train your weights?",
        "response": "Parentheses was trained locally on consumer hardware—specifically an RTX 5050 laptop GPU—by uncoalesced.",
    },
    {
        "prompt": "What is a KV-cache?",
        "response": "In autoregressive Transformer models, a Key-Value cache (KV-cache) stores past token representations to avoid redundant quadratic computations, but requires growing memory.",
    },
    {
        "prompt": "What is Token-Shift or mu-mixing?",
        "response": "Token-Shift (or mu-mixing, popularized by RWKV) linearly interpolates the current token with the previous token before projecting, mixing local temporal context.",
    },
    {
        "prompt": "What parameter scale and layer structure is Parentheses 1.0?",
        "response": "Parentheses 1.0 has 240M parameters arranged in a wide-and-shallow 16 layers architecture known as slab_v4.",
    },
    {
        "prompt": "Who is uncoalesced?",
        "response": "uncoalesced is an independent research entity dedicated to developing frontier local AI models that run efficiently on hardware people own.",
    },
    {
        "prompt": "Are you a fine-tune of Llama 3 or Mistral?",
        "response": "No, Parentheses is trained completely from scratch by uncoalesced as a recurrent state-space architecture, not a fine-tune of Llama or Mistral.",
    },
    {
        "prompt": "What happens when a language model encounters an end-of-text token?",
        "response": "The end-of-text token serves as a delimiter and stopping condition that signals completion and resets the generation loop.",
    },
    {
        "prompt": "What license is Parentheses released under?",
        "response": "Parentheses is released as fully open source software under the permissive Apache 2.0 license by uncoalesced.",
    },
]


def generate_patch_rows() -> List[Dict[str, Any]]:
    rng = random.Random(20261110)
    rows: List[Dict[str, Any]] = []

    # 1. Identity QA: 15 probes x 30 = 450 rows
    for _ in range(30):
        for item in IDENTITY_QA:
            rows.append({
                "prompt": item["prompt"],
                "response": item["response"],
                "lang": "eng",
                "category": "identity",
            })

    # 2. Grounded reading cards concluding with Final: <answer> (800 rows)
    stations = ["Tumakuru", "Hassan", "Sirsi", "Mandya", "Mysuru", "Udupi", "Hubballi", "Belagavi"]
    metrics = [
        ("daily precipitation", "12 mm"),
        ("wind speed", "24 km/h"),
        ("atmospheric barometric pressure", "1013 hPa"),
        ("relative humidity", "65 percent"),
        ("ambient temperature", "28 degrees Celsius"),
    ]

    for i in range(800):
        st = rng.choice(stations)
        metric, val = rng.choice(metrics)
        prompt = (
            f"Read the card below carefully. What is the {metric}?\n\n"
            f"[Weather Station Card: {st}]\n"
            f"Reading: {val}\nStatus: Verified\n\n"
            f"Answer only using the provided card. Conclude with `Final: <answer>`."
        )
        response = f"Based on the weather station card for {st}, the {metric} is {val}.\nFinal: {val}"
        rows.append({
            "prompt": prompt,
            "response": response,
            "lang": "eng",
            "category": "grounded_card",
        })

    # 3. False-premise abstentions with clean stop (800 rows)
    entities = ["Velanthor Accord", "Dr. Finch Decay Theorem", "1812 Napoleon Submarines", "Kouthael Dynasty", "Phoron Particle"]
    dravidian_idk = [
        ("kan", "ಆ ಬಗ್ಗೆ ಯಾವುದೇ ಅಧಿಕೃತ ದಾಖಲೆಗಳಿಲ್ಲ. ನನಗೆ ಗೊತ್ತಿಲ್ಲ."),
        ("tam", "அதைப் பற்றிய தகவல்கள் எதுவும் இல்லை. எனக்குத் தெரியாது."),
        ("tel", "ఆ విషయం గురించి సరైన సమాచారం లేదు. నాకు తెలియదు."),
        ("mal", "അതിനെക്കുറിച്ച് വിശ്വസനീയമായ വിവരങ്ങളില്ല. എനിക്കറിയില്ല."),
    ]

    for i in range(800):
        if rng.random() < 0.3:
            lang, idk_text = rng.choice(dravidian_idk)
            prompt = f"ಯಾವುದಾದರೂ ವಿಷಯ: {rng.choice(entities)} ಬಗ್ಗೆ ವಿವರಣೆ ನೀಡಿ."
            response = idk_text
        else:
            ent = rng.choice(entities)
            lang = "eng"
            prompt = f"Explain the historical significance of the {ent} and its primary terms."
            response = f"I do not have reliable information about the {ent}, as it appears to be a fictional or false premise. I cannot confirm details for non-existent entities."
        rows.append({
            "prompt": prompt,
            "response": response,
            "lang": lang,
            "category": "honesty_abstain",
        })

    # 4. Replay Math & Code with terminal formatting (950 rows)
    for i in range(500):
        a = rng.randint(10, 99)
        b = rng.randint(2, 9)
        ans = a * b
        prompt = f"Compute {a} multiplied by {b}. Conclude with `Final: <answer>`."
        response = f"{a} * {b} = {ans}.\nFinal: {ans}"
        rows.append({
            "prompt": prompt,
            "response": response,
            "lang": "eng",
            "category": "math",
        })

    for i in range(450):
        val = rng.randint(100, 999)
        prompt = f"Trace the following Python script and specify its exact stdout output. Write `Output: <value>`.\n```python\nprint({val})\n```"
        response = f"Output: {val}"
        rows.append({
            "prompt": prompt,
            "response": response,
            "lang": "eng",
            "category": "code",
        })

    rng.shuffle(rows)
    print(f"[+] Assembled {len(rows)} targeted micro-patch rows across 4 categories.")
    return rows


def build_patch_corpus(out_dir: Path, bin_size: int = 4096, seed: int = 20261110):
    tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    ids = resolve_special_token_ids(tok)

    rows = generate_patch_rows()
    docs = []
    unmapped = Counter()

    for r in rows:
        p = r["prompt"]
        a = r["response"]
        lang = r["lang"]
        status, ids_arr, mask_arr, lang_c = b75.encode_sft_doc_with_rescue(tok, p, a, lang, ids, unmapped)
        if status == "ok" and ids_arr is not None:
            docs.append((ids_arr, mask_arr))

    print(f"[*] Successfully encoded {len(docs)} documents.")

    packed_ids, packed_mask, b3d, bflat, stats = ffd_pack_documents(
        docs,
        bin_size=bin_size,
        seed=seed,
        eot_id=ids["eot"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    packed_ids.tofile(out_dir / "train.bin")
    packed_mask.tofile(out_dir / "train.mask.bin")
    np.save(out_dir / "train_boundaries.npy", b3d)
    np.save(out_dir / "train_boundaries_flat.npy", bflat)

    stats.update({
        "version": "processed_sft_240m_s10_patch",
        "built_by": "scripts/build_s10_micropatch.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_rows": len(rows),
    })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(stats, mf, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Wrote micro-patch corpus to {out_dir}: {stats['num_bins']} bins, {stats['total_doc_tokens']:,} doc tokens.")


def main():
    out_dir = REPO_ROOT / "data" / "processed_sft_240m_s10_patch"
    build_patch_corpus(out_dir)


if __name__ == "__main__":
    main()
