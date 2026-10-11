"""
Byte-level BPE tokenizer for Parentheses 0.9-NM (corpus v3). HF `tokenizers`.

    python data/tokenizer/train_tokenizer.py --vocab-size 65536 --derive 32768 49152
    python data/tokenizer/train_tokenizer.py --vocab-size 49152 --regex gpt --name v3_48k_gptregex

Design (documentation.md 2026-10-07):
- Byte-level BPE over all 256 bytes, so there is no UNK: any UTF-8 round-trips.
- NFC normalizer (the corpus is NFC-cleaned, so this is a no-op on it).
- Pre-tokenizer keeps combining marks inside words: [\\p{L}\\p{M}]+. The GPT-2/4
  pattern's \\p{L}+ cuts every Indic word at each vowel sign and virama (those are
  \\p{M}), so BPE can never learn whole aksharas across them. ZWJ/ZWNJ stay in
  the word. Digits split one per token.
- 256 reserved special ids at 0..255 (tags, chat, Free Think, tool calls, spare).
- Language-balanced sample (~100 MB per group) so Indic scripts get as many
  merges as English.
- Smaller vocabs are derived from a larger run by truncation: BPE is greedy, so
  the first N merges of a 64k run ARE the N-merge run on the same sample (ids:
  specials, then the 256-byte alphabet, then merges in order). One training run,
  not three.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANGS = ("kan", "tam", "tel", "mal")
LOW = ("tcy", "kff", "kxv", "nit")
TAGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng", "esg")  # esg reserved, unused
SPECIALS = (["<|endoftext|>", "<|pad|>"] + [f"<{t}>" for t in TAGS]
            + ["<|system|>", "<|user|>", "<|assistant|>", "<think>", "</think>", "<tool_call>", "</tool_call>"])
SPECIALS += [f"<|reserved_{i}|>" for i in range(256 - len(SPECIALS))]
assert len(SPECIALS) == 256

ZW = "‌‍"
REGEX = {
    # letters + combining marks (+ ZWNJ/ZWJ) are one word; single digits
    "indic": (r"'(?i:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}" + ZW + r"]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}" + ZW
              + r"]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
    # GPT-style ablation: \p{L}+ only (marks fall into the punctuation branch)
    "gpt": r"'(?i:[sdmt]|ll|ve|re)| ?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}


def build(regex: str):
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(REGEX[regex]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def _bin_docs(data, starts, ends):
    for s, e in zip(starts, ends):
        yield data[s:e].astype(np.uint8).tobytes().decode("utf-8", "replace")


TEXT_V3 = Path("F:/Lingual_Data/data/processed_v3/text")


def sample_texts(mb_per_group: int, seed: int = 0):
    """~mb_per_group MB per group (kan, tam, tel, mal, eng; low-res: all of it),
    a uniform random line sample of the corpus-v3 stage-1 text. Falls back to
    decoding the v1 bins when that text is missing."""
    if all((TEXT_V3 / f"{l}.jsonl").exists() for l in (*LANGS, *LOW, "eng")):
        rng = np.random.default_rng(seed)
        for lang in (*LANGS, *LOW, "eng"):
            p = TEXT_V3 / f"{lang}.jsonl"
            keep = 1.0 if lang in LOW else min(1.0, mb_per_group * 1e6 / p.stat().st_size * 1.05)
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if rng.random() < keep:
                        yield json.loads(line)["text"]
        return
    yield from _sample_v1(mb_per_group, seed)


def _sample_v1(mb_per_group: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cap = mb_per_group * 1_000_000
    d = ROOT / "data/processed_dravidian_v1"
    data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    bnd = np.load(d / "train_boundaries.npy").astype(np.int64)
    ends = np.append(bnd[1:], len(data))
    heads = np.stack([data[bnd + i] for i in range(1, 4)], 1).astype(np.uint8)
    lang_of = np.array([bytes(r).decode("ascii", "replace") for r in heads])
    for lang in LANGS:
        ids = rng.permutation(np.nonzero(lang_of == lang)[0])
        n = 0
        for i in ids:
            t = data[bnd[i] + 6:ends[i]].astype(np.uint8).tobytes().decode("utf-8", "replace")
            n += len(t.encode())
            yield t
            if n >= cap:
                break
    for lang in LOW:   # tiny: take all train text
        ld = ROOT / "data/processed_dravidian_lowres_v1" / lang
        lb = np.load(ld / "train_boundaries.npy").astype(np.int64)
        ldata = np.memmap(ld / "train.bin", dtype=np.uint16, mode="r")
        for t in _bin_docs(ldata, lb, np.append(lb[1:], len(ldata))):
            yield t[6:]
    a = np.memmap(ROOT / "data/processed_all_sources_v1/train.bin", dtype=np.uint16, mode="r")
    n_blocks, blk = 50, cap // 50
    for i in range(n_blocks):
        s = i * ((len(a) - blk) // (n_blocks - 1))
        raw = a[s:s + blk].astype(np.uint8).tobytes()
        raw = raw[raw.index(b"\n") + 1: raw.rindex(b"\n") + 1].decode("utf-8", "replace")
        for j in range(0, len(raw), 8192):
            yield raw[j:j + 8192]


def truncate(tok_json: dict, n: int) -> dict:
    """First n ids of a BPE tokenizer.json (specials + alphabet + first merges)."""
    out = json.loads(json.dumps(tok_json))
    vocab = out["model"]["vocab"]
    base = len(vocab) - len(out["model"]["merges"])
    keep = {k: v for k, v in vocab.items() if v < n}
    merges = out["model"]["merges"][: n - base]
    for m in merges:
        a, b = m if isinstance(m, list) else m.split(" ")
        assert a + b in keep, f"merge {a}+{b} not in truncated vocab"
    out["model"]["vocab"], out["model"]["merges"] = keep, merges
    assert len(keep) == n, (len(keep), n)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--derive", type=int, nargs="*", default=[], help="smaller vocabs cut from this run")
    p.add_argument("--regex", choices=sorted(REGEX), default="indic")
    p.add_argument("--mb-per-group", type=int, default=100)
    p.add_argument("--name", default=None, help="out dir name under data/tokenizer (default v3_<k>k)")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return _self_test()

    from tokenizers import Tokenizer, pre_tokenizers, trainers
    assert args.vocab_size <= 65536, "corpus ids are uint16"
    tok = build(args.regex)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(sample_texts(args.mb_per_group), trainer=trainer)
    full = json.loads(tok.to_str())
    for n in [args.vocab_size, *args.derive]:
        name = args.name if (args.name and n == args.vocab_size) else f"v3_{n // 1024}k"
        out = ROOT / "data/tokenizer" / name
        out.mkdir(parents=True, exist_ok=True)
        js = full if n == args.vocab_size else truncate(full, n)
        (out / "tokenizer.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
        t = Tokenizer.from_file(str(out / "tokenizer.json"))
        assert all(t.token_to_id(s) == i for i, s in enumerate(SPECIALS))
        meta = {"vocab_size": t.get_vocab_size(), "regex": args.regex, "pattern": REGEX[args.regex],
                "sample_mb_per_group": args.mb_per_group, "derived_from": None if n == args.vocab_size
                else f"first {n} ids of the {args.vocab_size} run",
                "sha256": hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()}
        (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"[tokenizer] {out}: {meta['vocab_size']} ids, sha {meta['sha256'][:12]}", flush=True)


def _self_test():
    """Regex: Indic word stays whole under 'indic', splits under 'gpt'; round trip exact."""
    from tokenizers import trainers, pre_tokenizers
    word = "ಕನ್ನಡ"   # has virama (\p{M}) in the middle
    assert len(build("indic").pre_tokenizer.pre_tokenize_str(word)) == 1
    assert len(build("gpt").pre_tokenizer.pre_tokenize_str(word)) > 1
    assert len(build("indic").pre_tokenizer.pre_tokenize_str("2026")) == 4   # one digit per token
    tok = build("indic")
    tr = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    text = ["ಕನ್ನಡ ನಾಡು 2026 hello world", "தமிழ் നമസ്കാരം తెలుగు", "zwnj: ക്‌ x"] * 50
    tok.train_from_iterator(text, trainer=tr)
    for t in text[:3] + ["unseen 𝄞 emoji 🙂"]:
        assert tok.decode(tok.encode(t).ids, skip_special_tokens=False) == t, t
    assert tok.token_to_id("<kan>") == 2 and tok.token_to_id("<|endoftext|>") == 0
    js = json.loads(tok.to_str())
    small = truncate(js, 560)
    assert len(small["model"]["vocab"]) == 560
    print("[self-test] train_tokenizer ok")


if __name__ == "__main__":
    sys.exit(main())
