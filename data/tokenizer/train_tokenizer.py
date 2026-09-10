"""
Optional: train a small BPE tokenizer over the combined corpus.

NOT needed for the current default presets (parentheses-0.9-100k/300k/600k)
-- they use raw byte-level tokenization (vocab_size=256, every UTF-8 byte
is a token) by default, specifically because at this param scale a normal
32k-vocab BPE tokenizer's embedding table alone would be several times
bigger than the whole model. See docs/training-time-estimate.md and the
comment above ModelConfig.PRESETS in model/config.py for the full reasoning
-- no need to run this script for those presets, encode text directly with
`text.encode("utf-8")` and each byte is already a token id.

This script is for the parentheses-0.9-1m stretch preset, which uses a
small trained vocab (512) instead of raw bytes now that there's finally
enough param budget to afford one. Keep vocab_size modest here too --
still nowhere near the 32k a 50M+ model would use.
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/tokenizer/parentheses-bpe")
    p.add_argument("--vocab-size", type=int, default=512,
                    help="match model/config.py's vocab_size for whichever preset you're targeting")
    args = p.parse_args()

    raise NotImplementedError(
        "Wire up tokenizers.ByteLevelBPETokenizer().train(files=[...glob corpus_dir...], "
        f"vocab_size={args.vocab_size}) and save_model(out_dir) here."
    )


if __name__ == "__main__":
    main()
