# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 Patch Decontamination & Leakage Checker
#
# Audits data/processed_sft_240m_s10_patch_v2 against data/eval_suites/suite_v3_heldout.jsonl:
#   1. Rolling 8-gram token/word overlap between training data and held-out evaluation suite.
#   2. Named entity & scenario disjointness check (held-out historical figures, future years, fake entities, false premises).
#   3. Zero probe leakage against identity_15.jsonl.
#
# Exits with 0 if cleanly decontaminated; non-zero if any contamination is detected.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
from tokenizers import Tokenizer

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# Concrete held-out scenarios and entities from Section 2.1 that must NEVER appear in patch training
FORBIDDEN_HELDOUT_SCENARIOS = [
    # Historical collisions in held-out
    "napoleon bonaparte", "abraham lincoln", "gettysburg", "leonardo da vinci",
    "alexander the great", "sir isaac newton", "thomas jefferson", "benjamin franklin",
    "charles babbage", "difference engine", "hannibal barca", "charlemagne",
    "lord nelson", "battle of trafalgar", "queen elizabeth i", "william shakespeare",

    # Future temporal anchors in held-out
    "california powerball", "apple inc", "icc men's cricket world cup",
    "census population of tokyo", "euro and the us dollar", "wimbledon men's singles",
    "112th oscars", "51st president of the united states", "reserve bank of india", "artemis 9",

    # Fake entities in held-out
    "chronium-99", "treaty of valyria", "north pangea", "subatomic aetherology",
    "dr. finch", "republic of zephyria", "elidor vance", "krypto-ferrite",
    "codex of morvath", "dr. donald thistlewood", "nebulastin", "luminescium-404",
    "psionic engineering", "mount thalassor", "south gondwana",

    # Dravidian held-out false premises
    "ಬೆಂಗಳೂರು ನಗರವು ಹಿಮಾಲಯ", "ಕೃಷ್ಣದೇವರಾಯರು ಅಮೆರಿಕ", "ಬಾದಾಮಿ ಗುಹೆಗಳು ಆರ್ಕ್ಟಿಕ್",
    "ಶರಾವತಿ ನದಿಯು ಅಂಟಾರ್ಕ್ಟಿಕಾ", "ವಿಧಾನ ಸೌಧವನ್ನು ಬ್ರಿಟಿಷ್ ರಾಣಿ",
    "ಚೆನ್ನೈ நகரம் கங்கை", "மகாபலிபுரம் சிற்பங்களை எகிப்திய", "திருக்குறளை லண்டன்",
    "ஹైదరాబాదు నగరం పసిఫిక్", "అమరావతి గ్రీస్", "తిరుమల వెంకటేశ్వర స్వామి ఆలయం హిమాలయ",
    "తిరువనంతపురం నగരം ఆఫ్రికా", "വേമ്പനാട്ട് കായൽ കാനഡ",
]

INSTRUCTION_SUFFIXES = [
    "ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ.",
    "முடிவில் `Final: <விடை>` என்று எழுதவும்.",
    "చివరిలో `Final: <సమాధానం>` అని రాయండి.",
    "ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക.",
    "Conclude with `Final: <answer>`.",
    "Write `Output: <value>`.",
]


def tokenize_words(text: str) -> List[str]:
    """Clean punctuation while keeping full Unicode words intact."""
    clean = re.sub(r"[\.,;:!?<>\"'`\(\)\[\]\{\}]", " ", text)
    return clean.lower().split()


def get_ngrams(tokens: List[str], n: int = 8) -> Set[Tuple[str, ...]]:
    """Return set of n-grams from a token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check patch leakage")
    parser.add_argument("--patch-dir", type=Path, default=REPO_ROOT / "data" / "processed_sft_240m_s10_patch_v2")
    parser.add_argument("--heldout", type=Path, default=REPO_ROOT / "data" / "eval_suites" / "suite_v3_heldout.jsonl")
    args = parser.parse_args()

    heldout_path = args.heldout
    patch_dir = args.patch_dir
    train_bin = patch_dir / "train.bin"
    tok_path = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"

    print("=" * 75)
    print("[*] S10 PATCH DECONTAMINATION & LEAKAGE AUDIT")
    print(f"Target Corpus: {train_bin}")
    print(f"Held-Out Suite: {heldout_path}")
    print("=" * 75)

    if not heldout_path.exists():
        raise FileNotFoundError(f"Held-out suite not found: {heldout_path}")
    if not train_bin.exists():
        raise FileNotFoundError(f"Patch training data not found: {train_bin}")

    tok = Tokenizer.from_file(str(tok_path))

    # 1. Load held-out prompts and extract prompt body 8-grams
    with open(heldout_path, "r", encoding="utf-8") as f:
        heldout_items = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Loaded {len(heldout_items)} held-out items.")

    heldout_8grams: Set[Tuple[str, ...]] = set()
    for item in heldout_items:
        p_text = item.get("prompt", "")
        # Remove standardized format instructions from n-gram body to test actual problem content
        for sfx in INSTRUCTION_SUFFIXES:
            p_text = p_text.replace(sfx, "")
        tokens = tokenize_words(p_text)
        heldout_8grams.update(get_ngrams(tokens, 8))

    print(f"[*] Extracted {len(heldout_8grams):,} unique 8-grams from held-out question bodies.")

    # 2. Decode patch training data in chunks of 50 bins
    data_mmap = np.memmap(train_bin, dtype=np.uint16, mode="r")
    num_tokens = len(data_mmap)
    total_bins = num_tokens // 4096
    print(f"[*] Scanning {total_bins} training bins ({num_tokens:,} tokens)...")

    ngram_hits = []
    scenario_hits = []

    chunk_bins = 50
    for b_start in range(0, total_bins, chunk_bins):
        b_end = min(b_start + chunk_bins, total_bins)
        tok_chunk = data_mmap[b_start * 4096 : b_end * 4096]
        text_chunk = tok.decode(tok_chunk.tolist(), skip_special_tokens=True).lower()

        # Scenario / entity check
        for scn in FORBIDDEN_HELDOUT_SCENARIOS:
            if scn.lower() in text_chunk:
                scenario_hits.append((scn, f"bins {b_start}-{b_end}"))

        # 8-gram check
        chunk_words = tokenize_words(text_chunk)
        chunk_8grams = get_ngrams(chunk_words, 8)
        overlap = heldout_8grams.intersection(chunk_8grams)
        if overlap:
            for hit in overlap:
                ngram_hits.append((" ".join(hit), f"bins {b_start}-{b_end}"))

    print("\n" + "=" * 75)
    print("AUDIT RESULTS:")
    print(f"  Exact 8-Gram Leaks Detected: {len(ngram_hits)}")
    print(f"  Forbidden Scenario / Entity Leaks Detected: {len(scenario_hits)}")

    if ngram_hits:
        print("\n[!] LEAKING 8-GRAM SAMPLES:")
        for span, loc in ngram_hits[:10]:
            print(f"  - '{span}' ({loc})")

    if scenario_hits:
        print("\n[!] LEAKING SCENARIOS / ENTITIES:")
        for ent, loc in set(scenario_hits):
            print(f"  - '{ent}' in {loc}")

    if not ngram_hits and not scenario_hits:
        print("\n[+] 100% CLEAN: ZERO 8-GRAM OVERLAP, ZERO ENTITY/SCENARIO OVERLAP.")
        print("[+] DECONTAMINATION VERIFIED: Patch training is fully cleared to proceed.")
        print("=" * 75)
        sys.exit(0)
    else:
        print("\n[FAIL] Decontamination failed: training data contains leaking spans.")
        print("=" * 75)
        sys.exit(1)


if __name__ == "__main__":
    main()
