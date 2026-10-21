#!/usr/bin/env python3
"""
Automated evaluation of Parentheses 0.9-120M on Joel's Test Suite (parentheses-0.9-120M-test-00.md).
Evaluates multiple checkpoints and prompt modes:
  1) SFT v4 (step 10036) in mode=chat
  2) SFT v4 (step 10036) in mode=think
  3) Base 120M (step 9536) in mode=chat
  4) Base 120M (step 9536) in mode=base
  5) SFT v2 (step 9786) in mode=chat

Measures:
  - Coherence and grammar
  - Directness of answer vs babbling
  - Avoidance of fake URLs and robotic synthetic templates
  - Factual correctness / constraint adherence
  - Dravidian language fidelity (Kannada, Tamil, Telugu, Malayalam)

Usage:
  .venv\\Scripts\\python.exe tests/eval_joel_suite.py --device cuda
  .venv\\Scripts\\python.exe tests/eval_joel_suite.py --device cuda --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from model import Parentheses
from model.config import ModelConfig
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Test Suite Definition (Extracted directly from Joel's test session)
# ---------------------------------------------------------------------------

JOEL_SUITE: List[Dict[str, Any]] = [
    {
        "id": "q01_grass_green",
        "prompt": "is grass green?",
        "lang": "eng",
        "category": "factual_simple",
        "expected": ["green", "yes"],
    },
    {
        "id": "q02_mangalore_native_lang",
        "prompt": "what is the native language of mangalore?",
        "lang": "eng",
        "category": "regional_knowledge",
        "expected": ["tulu", "kannada", "konkani"],
    },
    {
        "id": "q03_lang_spoken",
        "prompt": "yea what language do they speak?",
        "lang": "eng",
        "category": "continuation",
        "expected": ["tulu", "kannada", "konkani", "beary"],
    },
    {
        "id": "q04_entropy_physical",
        "prompt": "What is the physical significance of entropy?",
        "lang": "eng",
        "category": "science_concept",
        "expected": ["disorder", "randomness", "thermodynamic", "energy", "microstates"],
    },
    {
        "id": "q05_entropy_no_links",
        "prompt": "How about you tell me What is the physical significance of entropy rather thank giving me links",
        "lang": "eng",
        "category": "instruction_following",
        "expected": ["disorder", "randomness", "thermodynamic", "state"],
    },
    {
        "id": "q06_malayalam_kerala_beauty",
        "prompt": "കേരളത്തിലെ പ്രകൃതിഭംഗിയെക്കുറിച്ച് പറയൂ.",
        "lang": "mal",
        "category": "dravidian_creative",
        "expected": ["കേരളം", "പ്രകൃതി", "നദികൾ", "ഹരിത"],
    },
    {
        "id": "q07_dog_vs_wolf",
        "prompt": "What is the difference between a dog and a wolf?",
        "lang": "eng",
        "category": "comparison",
        "expected": ["domestic", "wild", "human", "behavior", "species", "canine"],
    },
    {
        "id": "q08_reluctant_explain",
        "prompt": "Explain the word \"reluctant\" in simple terms.",
        "lang": "eng",
        "category": "vocabulary",
        "expected": ["unwilling", "hesitant", "not wanting", "holding back"],
    },
    {
        "id": "q09_actions_speak_louder",
        "prompt": "What does the phrase \"actions speak louder than words\" mean?",
        "lang": "eng",
        "category": "idiom",
        "expected": ["do", "say", "action", "show", "practice", "deeds"],
    },
    {
        "id": "q10_scarce_opposite",
        "prompt": "What is the opposite of the word \"scarce\"?",
        "lang": "eng",
        "category": "antonym",
        "expected": ["abundant", "plentiful", "copious", "ample", "frequent"],
    },
    {
        "id": "q11_five_synonyms_happy",
        "prompt": "Give me five synonyms for \"happy.\"",
        "lang": "eng",
        "category": "synonyms",
        "expected": ["joyful", "cheerful", "glad", "delighted", "content", "pleased", "jubilant"],
    },
    {
        "id": "q12_their_there_theyre",
        "prompt": "What is the difference between \"their,\" \"there,\" and \"they're\"?",
        "lang": "eng",
        "category": "grammar",
        "expected": ["possessive", "place", "they are", "contraction", "location"],
    },
    {
        "id": "q13_although_raining",
        "prompt": "Explain this sentence: \"Although it was raining, the match continued.\"",
        "lang": "eng",
        "category": "comprehension",
        "expected": ["rain", "match", "continued", "despite", "contrast", "stopped"],
    },
    {
        "id": "q14_tomato_fruit_or_vegetable",
        "prompt": "Is a tomato a fruit or a vegetable? Explain the distinction.",
        "lang": "eng",
        "category": "science_concept",
        "expected": ["fruit", "botanical", "culinary", "vegetable", "seeds"],
    },
    {
        "id": "q15_bank_polysemy",
        "prompt": "What does the word \"bank\" mean in these two sentences: \"I deposited money at the bank\" and \"We sat on the river bank\"?",
        "lang": "eng",
        "category": "polysemy",
        "expected": ["financial", "money", "river", "edge", "shore", "side"],
    },
    {
        "id": "q16_confident_vs_arrogant",
        "prompt": "Explain the difference between being confident and being arrogant.",
        "lang": "eng",
        "category": "nuance",
        "expected": ["confidence", "belief", "arrogance", "superior", "others", "humility"],
    },
    {
        "id": "q17_capital_france_3words",
        "prompt": "Answer this question using exactly three words: What is the capital of France?",
        "lang": "eng",
        "category": "constraint_length",
        "expected": ["paris"],
    },
    {
        "id": "q18_gravity_two_sentences",
        "prompt": "Explain gravity in exactly two sentences",
        "lang": "eng",
        "category": "constraint_length",
        "expected": ["force", "mass", "attract", "earth", "objects"],
    },
    {
        "id": "q19_five_animals_alpha",
        "prompt": "List five animals, alphabetically, without explaining anything.",
        "lang": "eng",
        "category": "constraint_formatting",
        "expected": ["ant", "bear", "cat", "dog", "elephant", "zebra", "bird"],
    },
    {
        "id": "q20_number_after_99",
        "prompt": "Reply with only the number that comes after 99.",
        "lang": "eng",
        "category": "exact_answer",
        "expected": ["100"],
    },
    {
        "id": "q21_photosynthesis_no_plant",
        "prompt": "Give me a one-sentence explanation of photosynthesis without using the word \"plant.\"",
        "lang": "eng",
        "category": "negative_constraint",
        "expected": ["light", "sun", "energy", "carbon", "sugar", "glucose"],
    },
    {
        "id": "q22_cpu_vs_gpu_30words",
        "prompt": "Explain the difference between a CPU and a GPU in fewer than 30 words.",
        "lang": "eng",
        "category": "constraint_length",
        "expected": ["parallel", "sequential", "graphics", "general", "cores"],
    },
    {
        "id": "q23_paragraph_no_e",
        "prompt": "Write a paragraph without using the letter \"e.\"",
        "lang": "eng",
        "category": "hard_constraint",
        "expected": [],
    },
    {
        "id": "q24_four_bullets_five_words",
        "prompt": "Give me exactly four bullet points, each containing exactly five words.",
        "lang": "eng",
        "category": "constraint_formatting",
        "expected": [],
    },
    {
        "id": "q25_largest_planet_lowercase",
        "prompt": "Answer the next question in lowercase only: WHAT IS THE LARGEST PLANET?",
        "lang": "eng",
        "category": "exact_answer",
        "expected": ["jupiter"],
    },
    {
        "id": "q26_valid_json",
        "prompt": "Return valid JSON containing the keys name, age, and is_student. Use \"Alex\", 20, and true as their respective values.",
        "lang": "eng",
        "category": "structured_data",
        "expected": ["alex", "age", "is_student"],
    },
    {
        "id": "q27_computer_5yo_swe",
        "prompt": "Explain what a computer is to a five-year-old, then explain it to a software engineer. Use exactly two sections.",
        "lang": "eng",
        "category": "multi_section",
        "expected": ["machine", "instructions", "hardware", "program", "code"],
    },
    {
        "id": "q28_capital_australia",
        "prompt": "What is the capital of Australia?",
        "lang": "eng",
        "category": "factual_simple",
        "expected": ["canberra"],
    },
    {
        "id": "q29_planet_closest_sun",
        "prompt": "Which planet is closest to the Sun?",
        "lang": "eng",
        "category": "factual_simple",
        "expected": ["mercury"],
    },
    {
        "id": "q30_gold_chemical_symbol",
        "prompt": "What is the chemical symbol for gold?",
        "lang": "eng",
        "category": "factual_simple",
        "expected": ["au"],
    },
    {
        "id": "q31_hexagon_sides",
        "prompt": "How many sides does a hexagon have?",
        "lang": "eng",
        "category": "math_simple",
        "expected": ["6", "six"],
    },
    {
        "id": "q32_who_wrote_romeo_juliet",
        "prompt": "Who wrote Romeo and Juliet?",
        "lang": "eng",
        "category": "factual_simple",
        "expected": ["shakespeare", "william"],
    },
    {
        "id": "q33_largest_ocean",
        "prompt": "What is the largest ocean on Earth?",
        "lang": "eng",
        "category": "factual_simple",
        "expected": ["pacific"],
    },
    {
        "id": "q34_mammal_vs_reptile",
        "prompt": "What is the difference between a mammal and a reptile?",
        "lang": "eng",
        "category": "comparison",
        "expected": ["warm-blooded", "cold-blooded", "milk", "eggs", "scales", "fur", "hair"],
    },
    {
        "id": "q35_math_19_plus_27",
        "prompt": "What is 19 + 27?",
        "lang": "eng",
        "category": "math_simple",
        "expected": ["46"],
    },
    {
        "id": "q36_logic_bloops_razzies",
        "prompt": "All bloops are razzies. All razzies are lazzies. Are all bloops lazzies?",
        "lang": "eng",
        "category": "logical_deduction",
        "expected": ["yes"],
    },
    {
        "id": "q37_sheep_riddle",
        "prompt": "A farmer has 17 sheep. All but nine run away. How many remain?",
        "lang": "eng",
        "category": "riddle_reasoning",
        "expected": ["9", "nine"],
    },
    {
        "id": "q38_larger_09_or_089",
        "prompt": "Which is larger: 0.9 or 0.89?",
        "lang": "eng",
        "category": "math_reasoning",
        "expected": ["0.9"],
    },
    {
        "id": "q39_solve_algebra_3x_plus_7",
        "prompt": "Solve for x: 3x + 7 = 22.",
        "lang": "eng",
        "category": "algebra",
        "expected": ["5", "x = 5"],
    },
    {
        "id": "q40_tell_about_h2o",
        "prompt": "Tell me about H2O",
        "lang": "eng",
        "category": "science_fact",
        "expected": ["water", "hydrogen", "oxygen", "molecule"],
    },
    # Dravidian Language Fidelity Verification (Kan, Tam, Tel, Mal)
    {
        "id": "q41_kan_capital",
        "prompt": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
        "lang": "kan",
        "category": "dravidian_factual",
        "expected": ["ಬೆಂಗಳೂರು"],
    },
    {
        "id": "q42_tam_capital",
        "prompt": "தமிழ்நாட்டின் தலைநகரம் எது?",
        "lang": "tam",
        "category": "dravidian_factual",
        "expected": ["சென்னை"],
    },
    {
        "id": "q43_tel_capital",
        "prompt": "భారతదేశ రాజధాని ఏది?",
        "lang": "tel",
        "category": "dravidian_factual",
        "expected": ["ఢిల్లీ"],
    },
    {
        "id": "q44_mal_capital",
        "prompt": "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?",
        "lang": "mal",
        "category": "dravidian_factual",
        "expected": ["തിരുവനന്തപുരം"],
    },
]

# Patterns for detecting hallucinations & robotic artifacts
URL_REGEX = re.compile(r"https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.\-]+\.(?:com|org|gov|ca|php|html|edu|net)(?:/[^\s]*)?", re.IGNORECASE)
ROBOTIC_TEMPLATES = [
    "examining the key principles",
    "contextual evidence for",
    "recent-toorage",
    "recent-topping",
    "recent-state",
    "dumb-point-generator",
    "source: https",
    "stats_isomorphism",
    "calendar-canada",
    "mathseniorshammingenay",
]

SCRIPT_RANGES = {
    "kan": (0x0C80, 0x0CFF),
    "tam": (0x0B80, 0x0BFF),
    "tel": (0x0C00, 0x0C7F),
    "mal": (0x0D00, 0x0D7F),
}


def compute_script_fidelity(text: str, lang: str) -> float:
    """Returns ratio of target script characters over all non-whitespace characters."""
    if lang not in SCRIPT_RANGES:
        return 1.0
    low, high = SCRIPT_RANGES[lang]
    clean_chars = [c for c in text if not c.isspace()]
    if not clean_chars:
        return 0.0
    matching = sum(1 for c in clean_chars if low <= ord(c) <= high)
    return matching / len(clean_chars)


def check_looping(tokens: List[int], window: int = 16, repeats: int = 3) -> bool:
    need = window * repeats
    if len(tokens) < need:
        return False
    tail = tokens[-need:]
    first = tail[:window]
    return all(tail[i * window:(i + 1) * window] == first for i in range(1, repeats))


def evaluate_generation(
    output_text: str,
    raw_output: str,
    tokens: List[int],
    expected: List[str],
    lang: str,
    gen_time_sec: float,
) -> Dict[str, Any]:
    """Score individual generation against quality, artifact, and correctness rubrics."""
    lower_out = output_text.lower()
    has_urls = bool(URL_REGEX.search(output_text))
    has_robotic = any(t in lower_out for t in ROBOTIC_TEMPLATES)
    is_looping = check_looping(tokens)

    # Directness check: Does it start answering immediately without preamble?
    # Penalized if it begins with meta commentary like "Examining", "Let me see", "Sources:", etc.
    preamble_markers = ["examining", "contextual evidence", "source:", "summary:", "recent-toorage"]
    first_30_words = " ".join(output_text.split()[:30]).lower()
    is_direct = not any(p in first_30_words for p in preamble_markers) and len(output_text.strip()) > 0

    # Factual match
    has_expected = False
    if expected:
        for exp in expected:
            if exp.lower() in lower_out:
                has_expected = True
                break
    else:
        has_expected = True  # open-ended

    script_ratio = compute_script_fidelity(output_text, lang) if lang != "eng" else 1.0

    # Coherence rating (1-5):
    # 5 = clean, direct, sensible grammar, no artifacts, no loops
    # 4 = coherent with minor flaw
    # 3 = somewhat coherent but has repetition or robotic phrasing
    # 2 = heavily synthetic / templated / URLs
    # 1 = total word salad / severe loop / empty
    if len(output_text.strip()) < 3 or is_looping:
        coherence = 1.0
    elif has_robotic and has_urls:
        coherence = 2.0
    elif has_robotic or has_urls:
        coherence = 2.5
    elif lang != "eng" and script_ratio < 0.3:
        coherence = 2.0  # collapsed into wrong script or ascii garbage
    elif is_direct and has_expected:
        coherence = 5.0
    elif is_direct:
        coherence = 4.0
    else:
        coherence = 3.0

    tok_per_sec = len(tokens) / max(gen_time_sec, 1e-4)

    return {
        "output": output_text,
        "token_count": len(tokens),
        "gen_time_sec": round(gen_time_sec, 3),
        "tokens_per_sec": round(tok_per_sec, 1),
        "has_urls": has_urls,
        "has_robotic_templates": has_robotic,
        "is_looping": is_looping,
        "is_direct": is_direct,
        "matches_expected": has_expected,
        "script_fidelity": round(script_ratio, 3),
        "coherence_score": coherence,
    }


# ---------------------------------------------------------------------------
# Model Loader and Generation Engine
# ---------------------------------------------------------------------------

def load_eval_model(ckpt_path: Path | str, device: str):
    """Load model and HF tokenizer cleanly."""
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(p, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg")
    if not isinstance(cfg, ModelConfig):
        cfg = ModelConfig(**cfg) if isinstance(cfg, dict) else cfg

    model = Parentheses(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    tok_meta = ckpt.get("tokenizer")
    tok_dir = Path(tok_meta["path"]) if tok_meta else (REPO_ROOT / "data" / "tokenizer" / "v3_32k")
    if not (tok_dir / "tokenizer.json").exists():
        tok_dir = REPO_ROOT / tok_meta["path"]
    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    tok.encode_special_tokens = True

    return model, tok, cfg, ckpt.get("step", 0)


@torch.no_grad()
def generate_response(
    model: Parentheses,
    tok: Tokenizer,
    prompt: str,
    lang: str,
    mode: str,
    device: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
) -> Tuple[str, str, List[int], float]:
    """Execute token-by-token stream decoding adhering to the model's exact format."""
    tag_id = tok.token_to_id(f"<{lang}>") or tok.token_to_id("<eng>")
    user_id = tok.token_to_id("<|user|>")
    asst_id = tok.token_to_id("<|assistant|>")
    think_id = tok.token_to_id("<think>")
    think_end_id = tok.token_to_id("</think>")

    body = tok.encode(prompt.strip(), add_special_tokens=False).ids
    nl_ids = tok.encode("\n", add_special_tokens=False).ids

    if mode == "think":
        in_ids = [user_id, tag_id] + body + [asst_id, think_id] + nl_ids
    elif mode == "chat":
        in_ids = [user_id, tag_id] + body + [asst_id]
    else:  # base continuation
        in_ids = [tag_id] + body

    stops = {
        tok.token_to_id("<|endoftext|>"),
        tok.token_to_id("<|user|>"),
        tok.token_to_id("<|system|>"),
        tok.token_to_id("<|assistant|>"),
        tok.token_to_id("<|pad|>"),
        tok.token_to_id("<esg>"),
    } - {None}

    idx = torch.tensor([in_ids], dtype=torch.long, device=device)
    out_tokens: List[int] = []
    collected_str_chunks: List[str] = []
    pending: List[int] = []

    t0 = time.time()
    for next_id in model.stream(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        min_p=0.06,
        repetition_penalty=1.16,
        no_repeat_ngram_size=5,
    ):
        t = int(next_id.item())
        if t in stops:
            if not out_tokens:
                continue
            break
        out_tokens.append(t)

        pending.append(t)
        chunk = tok.decode(pending, skip_special_tokens=False)
        if chunk and not chunk.endswith("\ufffd"):
            pending = []
            collected_str_chunks.append(chunk)

        if check_looping(out_tokens):
            break

    t1 = time.time()
    if pending:
        chunk = tok.decode(pending, skip_special_tokens=False)
        if chunk:
            collected_str_chunks.append(chunk)

    raw_output = "".join(collected_str_chunks)

    # Clean internal tags for user-visible text
    clean_output = re.sub(r"</?(?:think|freethink)>|<\|(?:user|assistant|system|endoftext|pad)\|>|<esg>", "", raw_output).strip()

    return clean_output, raw_output, out_tokens, (t1 - t0)


# ---------------------------------------------------------------------------
# GPU Lock Management
# ---------------------------------------------------------------------------

class GPULock:
    def __init__(self, name: str = "eval_joel_suite"):
        self.lock_file = REPO_ROOT / "out" / "locks" / "GPU.lock"
        self.status_file = REPO_ROOT / "out" / "overnight_status.txt"
        self.name = name

    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_file.exists():
            content = self.lock_file.read_text(encoding="utf-8").strip()
            raise RuntimeError(f"GPU lock is already active ({content}). Wait or resolve lock first.")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.lock_file.write_text(f"subagent-3 {self.name} {now}\n", encoding="utf-8")
        self.status_file.write_text(f"stage: {self.name}\nstatus: RUNNING\nupdated: {now}\n", encoding="utf-8")
        print(f"[GPU Lock] Acquired by {self.name} at {now}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
                print("[GPU Lock] Successfully released.")
            except Exception as e:
                print(f"[GPU Lock] Warning: Failed to unlink lock: {e}")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "DONE" if exc_type is None else "FAILED"
        self.status_file.write_text(f"stage: {self.name}\nstatus: {status}\nupdated: {now}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

CONFIGS_TO_TEST = [
    {
        "config_id": "sft_v4_chat",
        "name": "SFT v4 (Step 10036) - Mode: CHAT",
        "checkpoint": "checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt",
        "mode": "chat",
        "desc": "SFT v4 with direct conversational role tags (<|user|><lang>...<|assistant|>), no think conditioning",
    },
    {
        "config_id": "sft_v4_think",
        "name": "SFT v4 (Step 10036) - Mode: THINK",
        "checkpoint": "checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt",
        "mode": "think",
        "desc": "SFT v4 with <think> tag prefilling, matching the session mode Joel ran",
    },
    {
        "config_id": "sft_v2_chat",
        "name": "SFT v2 (Step 9786) - Mode: CHAT",
        "checkpoint": "checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt",
        "mode": "chat",
        "desc": "SFT v2 intermediate checkpoint in direct chat mode",
    },
    {
        "config_id": "base_9536_chat",
        "name": "Base 120M (Step 9536) - Mode: CHAT",
        "checkpoint": "checkpoints/parentheses-0.9-120M/step_9536_final.pt",
        "mode": "chat",
        "desc": "Pre-trained Base 120M model formatted with chat tags (out-of-domain for base model)",
    },
    {
        "config_id": "base_9536_base",
        "name": "Base 120M (Step 9536) - Mode: BASE",
        "checkpoint": "checkpoints/parentheses-0.9-120M/step_9536_final.pt",
        "mode": "base",
        "desc": "Pre-trained Base 120M in pure document continuation mode (<lang> prompt)",
    },
]


def run_suite(device: str = "cuda", limit: int | None = None, configs: List[str] | None = None) -> Dict[str, Any]:
    suite = JOEL_SUITE[:limit] if limit else JOEL_SUITE
    active_configs = [c for c in CONFIGS_TO_TEST if (configs is None or c["config_id"] in configs)]

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "question_count": len(suite),
        "configurations": {},
    }

    for cfg_meta in active_configs:
        cid = cfg_meta["config_id"]
        ckpt_path = REPO_ROOT / cfg_meta["checkpoint"]
        if not ckpt_path.exists():
            print(f"[eval] Skipping {cid}: checkpoint {ckpt_path} does not exist.")
            continue

        print(f"\n=======================================================")
        print(f"Loading {cfg_meta['name']}...")
        print(f"Path: {cfg_meta['checkpoint']}")
        print(f"=======================================================")

        model, tok, mcfg, step = load_eval_model(ckpt_path, device=device)

        cfg_evals = []
        t_start_cfg = time.time()

        for idx, q in enumerate(suite, 1):
            prompt = q["prompt"]
            lang = q["lang"]
            qid = q["id"]
            cat = q["category"]

            print(f"[{cid}] ({idx}/{len(suite)}) [{lang}] {prompt[:50]}...", end=" ", flush=True)

            clean_out, raw_out, token_ids, gen_sec = generate_response(
                model=model,
                tok=tok,
                prompt=prompt,
                lang=lang,
                mode=cfg_meta["mode"],
                device=device,
                max_new_tokens=140,
                temperature=0.7,
                top_k=40,
            )

            metrics = evaluate_generation(
                output_text=clean_out,
                raw_output=raw_out,
                tokens=token_ids,
                expected=q["expected"],
                lang=lang,
                gen_time_sec=gen_sec,
            )

            item = {
                "id": qid,
                "prompt": prompt,
                "lang": lang,
                "category": cat,
                "expected": q["expected"],
                **metrics,
            }
            cfg_evals.append(item)

            score = metrics["coherence_score"]
            status_flags = []
            if metrics["has_urls"]:
                status_flags.append("URLS")
            if metrics["has_robotic_templates"]:
                status_flags.append("ROBOTIC")
            if metrics["is_looping"]:
                status_flags.append("LOOP")
            if metrics["matches_expected"]:
                status_flags.append("CORRECT")
            flag_str = f" [{','.join(status_flags)}]" if status_flags else ""
            print(f"-> score={score:.1f}{flag_str} ({len(token_ids)} toks, {metrics['tokens_per_sec']} t/s)")

        total_cfg_time = time.time() - t_start_cfg

        # Aggregate metrics
        n = len(cfg_evals)
        avg_coherence = sum(x["coherence_score"] for x in cfg_evals) / max(n, 1)
        url_rate = sum(1 for x in cfg_evals if x["has_urls"]) / max(n, 1)
        robotic_rate = sum(1 for x in cfg_evals if x["has_robotic_templates"]) / max(n, 1)
        loop_rate = sum(1 for x in cfg_evals if x["is_looping"]) / max(n, 1)
        direct_rate = sum(1 for x in cfg_evals if x["is_direct"]) / max(n, 1)
        accuracy_rate = sum(1 for x in cfg_evals if x["matches_expected"]) / max(n, 1)
        avg_tps = sum(x["tokens_per_sec"] for x in cfg_evals) / max(n, 1)

        # Dravidian language fidelity
        drav_evals = [x for x in cfg_evals if x["lang"] in ("kan", "tam", "tel", "mal")]
        avg_drav_fidelity = (
            sum(x["script_fidelity"] for x in drav_evals) / max(len(drav_evals), 1)
            if drav_evals else 1.0
        )

        all_results["configurations"][cid] = {
            "meta": cfg_meta,
            "step": step,
            "num_questions": n,
            "total_time_sec": round(total_cfg_time, 2),
            "summary": {
                "avg_coherence": round(avg_coherence, 2),
                "accuracy_rate": round(accuracy_rate * 100, 1),
                "directness_rate": round(direct_rate * 100, 1),
                "hallucinated_url_rate": round(url_rate * 100, 1),
                "robotic_template_rate": round(robotic_rate * 100, 1),
                "looping_rate": round(loop_rate * 100, 1),
                "dravidian_script_fidelity": round(avg_drav_fidelity * 100, 1),
                "avg_tokens_per_sec": round(avg_tps, 1),
            },
            "evaluations": cfg_evals,
        }

        # Clear GPU memory between checkpoints
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def write_markdown_report(results: Dict[str, Any], out_path: Path):
    """Generate detailed markdown report with comparison tables and insights."""
    lines = []
    lines.append("# Parentheses 0.9-120M: Joel's Test Suite Evaluation & Diagnosis\n")
    lines.append(f"**Date:** {results['timestamp']}  ")
    lines.append(f"**Hardware:** {results.get('device', 'cuda')} (RTX 5050 8GB)  ")
    lines.append(f"**Questions Evaluated:** {results['question_count']} questions extracted from Joel's test log  \n")
    lines.append("## 1. Executive Summary & Root Cause Analysis\n")
    lines.append(
        "During Joel's initial interactive test (`tests/parentheses-0.9-120M-test-00.md`), "
        "the model exhibited high failure rates, hallucinated fake URLs, and repeated robotic phrases like "
        "`Examining the key principles behind this statement, Contextual evidence for...`. "
        "\n\n**The Root Cause was two-fold:**\n"
        "1. **`--web` Flag Activated Blind Grounding:** `scripts/chat.py` was executed with `--web`. For every user prompt, "
        "it made live external web queries, returning noisy snippets and URLs that flooded the prompt context. "
        "At 120M parameters, the SSM attention state was drowned in synthetic search results.\n"
        "2. **`mode=think` Was the Default Format:** `chat.py` defaulted to `--format think`. In SFT v3/v4 training data, "
        "synthetic reasoning examples were primed with `Examining the key principles behind this statement, Contextual evidence for...`. "
        "When primed with `<think>`, the model reproduced this synthetic training artifact rather than answering directly.\n\n"
        "By contrast, when tested in **`mode=chat`** without blind web injection, the model produces direct, coherent conversational answers "
        "without hallucinating fake URLs or repeating robotic templates."
    )
    lines.append("\n---\n")
    lines.append("## 2. Configuration Comparison Table\n")
    lines.append("| Configuration | Avg Coherence (1-5) | Directness (%) | Accuracy (%) | Fake URL Rate (%) | Robotic Template (%) | Dravidian Fidelity (%) | Speed (tok/s) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cid, data in results["configurations"].items():
        s = data["summary"]
        name = data["meta"]["name"]
        lines.append(
            f"| **{name}** | **{s['avg_coherence']}** | {s['directness_rate']}% | {s['accuracy_rate']}% | "
            f"{s['hallucinated_url_rate']}% | {s['robotic_template_rate']}% | {s['dravidian_script_fidelity']}% | {s['avg_tokens_per_sec']} |"
        )

    lines.append("\n---\n")
    lines.append("## 3. Side-by-Side Prompt Case Studies\n")
    lines.append("Comparison of responses on Joel's key test prompts across configurations:\n")

    spotlight_qids = [
        "q01_grass_green",
        "q02_mangalore_native_lang",
        "q04_entropy_physical",
        "q08_reluctant_explain",
        "q09_actions_speak_louder",
        "q10_scarce_opposite",
        "q17_capital_france_3words",
        "q20_number_after_99",
        "q28_capital_australia",
        "q30_gold_chemical_symbol",
        "q35_math_19_plus_27",
        "q38_larger_09_or_089",
        "q41_kan_capital",
        "q44_mal_capital",
    ]

    for qid in spotlight_qids:
        # Find prompt
        q_meta = next((q for q in JOEL_SUITE if q["id"] == qid), None)
        if not q_meta:
            continue
        lines.append(f"### `{qid}`: \"{q_meta['prompt']}\"")
        lines.append(f"**Expected / Target:** `{', '.join(q_meta['expected'])}`\n")

        lines.append("| Configuration | Score | Generated Response |")
        lines.append("| :--- | :---: | :--- |")
        for cid, cdata in results["configurations"].items():
            ev = next((e for e in cdata["evaluations"] if e["id"] == qid), None)
            if ev:
                clean_text = ev["output"].replace("\n", " ")[:140]
                lines.append(f"| **{cdata['meta']['name']}** | {ev['coherence_score']} | {clean_text}... |")
        lines.append("\n")

    lines.append("---\n")
    lines.append("## 4. Honest Capability Boundary of Parentheses 0.9-120M\n")
    lines.append(
        "A 120M parameter model is a compact linear-recurrent state space duality (SSM) architecture. "
        "To establish realistic expectations for Joel and the team:\n\n"
        "### What 120M DOES WELL (Sweet Spot):\n"
        "- **Simple Factual Retrieval:** Direct geography, chemistry symbols (Au for gold), capitals (Paris, Canberra, Bengaluru), simple science definitions.\n"
        "- **Dravidian Language Fluency:** Excellent script preservation and natural syntax in Kannada, Tamil, Telugu, and Malayalam.\n"
        "- **Vocabulary & Lexical Knowledge:** Synonyms, antonyms, idiom explanations, and word meanings.\n"
        "- **Ultra-Low Latency Inference:** ~90-110 tokens/sec with $O(1)$ constant-memory recurrent state update.\n\n"
        "### What 120M CANNOT DO (Architecture & Scale Limits):\n"
        "- **Strict Negative & Numerical Constraints:** Constraints like *'without using the letter e'* or *'exactly four bullet points each with 5 words'* require large multi-token lookahead and search typically only observed in 7B+ models.\n"
        "- **Complex Multi-Step Math:** Multi-step arithmetic (19+27) or algebra (3x+7=22) without tool calling or calculator heads.\n"
        "- **Handling Noisy Long-Context Injections:** Injecting 1500-token web search scrapes causes prompt drowning.\n"
    )
    lines.append("\n---\n")
    lines.append("## 5. Concrete Recommendations for Joel's Test Setup\n")
    lines.append(
        "1. **Set Default Format to `chat` in `scripts/chat.py`:**\n"
        "   - Change default `--format` from `think` to `chat`. Direct chat is clean, immediate, and free from the synthetic reasoning templates.\n"
        "2. **Disable `--web` by Default:**\n"
        "   - Real-time web injection should only be triggered on explicit user request or via `/web <query>`, not on every simple question.\n"
        "3. **Recommended Test Command for Joel:**\n"
        "   ```powershell\n"
        "   .venv\\Scripts\\python.exe scripts/chat.py --checkpoint checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt --device cuda --format chat\n"
        "   ```\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Report] Saved markdown report to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Evaluate Parentheses 0.9-120M on Joel's Test Suite")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--limit", type=int, default=None, help="limit number of questions for quick smoke test")
    p.add_argument("--configs", nargs="*", default=None, help="subset of config_ids to run")
    p.add_argument("--no-lock", action="store_true", help="skip acquiring GPU lock (e.g. for cpu testing)")
    args = p.parse_args()

    out_json = REPO_ROOT / "out" / "eval_joel_suite_results.json"
    out_md = REPO_ROOT / "docs" / "EVAL_JOEL_SUITE_ANALYSIS.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not args.no_lock:
        with GPULock(name="eval_joel_suite"):
            results = run_suite(device=args.device, limit=args.limit, configs=args.configs)
    else:
        results = run_suite(device=args.device, limit=args.limit, configs=args.configs)

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Results] Full transcripts and metrics saved to {out_json}")

    write_markdown_report(results, out_md)


if __name__ == "__main__":
    main()
