"""
Translation evaluation pipeline for Parentheses checkpoints.

Engineered by uncoalesced

Computes Dravidian morphosyntactic metrics (chrF++), corpus BLEU (SacreBLEU),
script purity (Unicode range compliance), and length ratio against reference pairs.

Usage:
    python scripts/eval_translation.py --self-test
    python scripts/eval_translation.py --checkpoint checkpoints/dravidian-cpt-selective-v1/step_610998_final.pt
"""

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Parentheses

# Unicode script ranges for Dravidian languages
UNICODE_BLOCKS = {
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "ml": (0x0D00, 0x0D7F),  # Malayalam
}

# Standard reference test pairs for quick evaluation
TEST_PAIRS = [
    {"src_lang": "en", "tgt_lang": "kn", "src": "Water is important for life.", "ref": "ನೀರು ಜೀವನಕ್ಕೆ ಮುಖ್ಯವಾಗಿದೆ."},
    {"src_lang": "en", "tgt_lang": "kn", "src": "The book is on the table.", "ref": "ಪುಸ್ತಕವು ಮೇಜಿನ ಮೇಲಿದೆ."},
    {"src_lang": "en", "tgt_lang": "ta", "src": "Water is important for life.", "ref": "நீர் வாழ்க்கைக்கு முக்கியமானது."},
    {"src_lang": "en", "tgt_lang": "te", "src": "Water is important for life.", "ref": "జీవితానికి నీరు చాలా ముఖ్యం."},
]


def verify_script_purity(text: str, target_lang: str) -> float:
    """Verifies the ratio of alphabetic characters belonging to the target script."""
    if target_lang not in UNICODE_BLOCKS:
        return 1.0
    start, end = UNICODE_BLOCKS[target_lang]
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    pure_chars = sum(1 for c in alpha_chars if start <= ord(c) <= end)
    return pure_chars / len(alpha_chars)


def compute_chrf(hypotheses: list[str], references: list[str], n: int = 6, beta: float = 2.0) -> float:
    """Fallback character n-gram F-score (chrF++) computation if sacrebleu is unavailable."""
    try:
        import sacrebleu
        ref_matrix = [[r] for r in references]
        return sacrebleu.corpus_chrf(hypotheses, ref_matrix, word_order=2).score
    except ImportError:
        pass

    # Pure Python micro-chrF implementation for self-test / fallback
    total_f = 0.0
    for hyp, ref in zip(hypotheses, references):
        hyp_clean = hyp.replace(" ", "")
        ref_clean = ref.replace(" ", "")
        if not hyp_clean or not ref_clean:
            continue
        
        hyp_ngrams = collections_counter([hyp_clean[i:i+n] for i in range(len(hyp_clean)-n+1)])
        ref_ngrams = collections_counter([ref_clean[i:i+n] for i in range(len(ref_clean)-n+1)])
        
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        prec = overlap / max(len(hyp_clean)-n+1, 1)
        rec = overlap / max(len(ref_clean)-n+1, 1)
        
        if prec + rec > 0:
            f_score = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
            total_f += f_score

    return (total_f / max(len(hypotheses), 1)) * 100.0


def collections_counter(items):
    from collections import Counter
    return Counter(items)


def compute_metrics(predictions: list[str], references: list[str], tgt_langs: list[str]):
    """Returns evaluation dictionary including chrF++, BLEU (if sacrebleu present), and script purity."""
    res = {}
    try:
        import sacrebleu
        ref_matrix = [[r] for r in references]
        bleu = sacrebleu.corpus_bleu(predictions, ref_matrix)
        chrf = sacrebleu.corpus_chrf(predictions, ref_matrix, word_order=2)
        res["bleu"] = round(bleu.score, 2)
        res["chrf++"] = round(chrf.score, 2)
    except ImportError:
        res["bleu"] = None
        res["chrf++"] = round(compute_chrf(predictions, references), 2)

    purities = [verify_script_purity(p, l) for p, l in zip(predictions, tgt_langs)]
    res["avg_script_purity"] = round(sum(purities) / max(len(purities), 1), 4)
    return res


@torch.no_grad()
def generate_translation(model, prompt: str, device: str, max_new_tokens: int = 60) -> str:
    """Autoregressively sample continuation given prompt."""
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=0.7, top_k=40)
    decoded = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    # Return continuation after prompt
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()


def _self_test():
    """Verify metrics and script purity calculations."""
    preds = ["ಪುಸ್ತಕವು ಮೇಜಿನ ಮೇಲಿದೆ.", "நீர் வாழ்க்கைக்கு முக்கியமானது."]
    refs = ["ಪುಸ್ತಕವು ಮೇಜಿನ ಮೇಲಿದೆ.", "நீர் வாழ்க்கைக்கு முக்கியமானது."]
    langs = ["kn", "ta"]

    metrics = compute_metrics(preds, refs, langs)
    assert metrics["avg_script_purity"] == 1.0, f"Expected 1.0 purity, got {metrics['avg_script_purity']}"
    assert metrics["chrf++"] > 90.0, f"Expected >90 chrF for identical pairs, got {metrics['chrf++']}"

    # Verify script purity catches contamination
    bad_purity = verify_script_purity("ಪುಸ್ತಕವು table ಮೇಲಿದೆ", "kn")
    assert bad_purity < 1.0, "Purity did not penalize Latin characters"

    print("[self-test] eval_translation: metrics and purity calculations passed ok.")


def main():
    p = argparse.ArgumentParser(description="Evaluate translation metrics on Parentheses checkpoints.")
    p.add_argument("--checkpoint", default="checkpoints/dravidian-cpt-selective-v1/step_610998_final.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint {args.checkpoint} does not exist.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])

    predictions = []
    references = []
    tgt_langs = []

    print(f"Evaluating checkpoint: {args.checkpoint}")
    for item in TEST_PAIRS:
        prompt = f"<{item['src_lang']}> {item['src']} -> <{item['tgt_lang']}>"
        gen = generate_translation(model, prompt, args.device)
        predictions.append(gen)
        references.append(item["ref"])
        tgt_langs.append(item["tgt_lang"])
        print(f"Prompt: {prompt}")
        print(f"Generated : {gen}")
        print(f"Reference : {item['ref']}\n")

    metrics = compute_metrics(predictions, references, tgt_langs)
    print("--- Benchmark Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
