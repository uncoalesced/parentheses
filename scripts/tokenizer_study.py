"""
Tokenizer study for corpus v3: every data/tokenizer/v3_* tokenizer plus raw bytes,
on each language's held-out val text (the same old val splits the eval uses).
Prints to the terminal only.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/tokenizer_study.py

Columns: tok/word (whitespace words), bytes/tok, chars/tok, xEN = tok/word over
English's tok/word with the same tokenizer (fertility parity; 1.0 = as cheap as
English), RT = exact decode(encode(text)) == text on every doc.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "F:/Lingual_Data/scripts")
from build_corpus_v3 import val_texts  # noqa: E402

MAX_BYTES = 2_000_000   # per language, plenty for stable ratios


def main():
    toks = {p.name: Tokenizer.from_file(str(p / "tokenizer.json"))
            for p in sorted((ROOT / "data/tokenizer").glob("v3_*")) if (p / "tokenizer.json").exists()}
    for t in toks.values():
        t.encode_special_tokens = True
    texts = {}
    for lang, docs in val_texts().items():
        keep, n = [], 0
        for d in docs:
            keep.append(d)
            n += len(d.encode())
            if n >= MAX_BYTES:
                break
        texts[lang] = keep

    stats = {}
    for lang, docs in texts.items():
        words = sum(len(d.split()) for d in docs)
        chars = sum(len(d) for d in docs)
        nbytes = sum(len(d.encode()) for d in docs)
        stats[(lang, "bytes")] = (nbytes / words, 1.0, chars / nbytes, True)
        for name, tok in toks.items():
            enc = tok.encode_batch(docs, add_special_tokens=False)
            n = sum(len(e.ids) for e in enc)
            rt = all(tok.decode(e.ids, skip_special_tokens=False) == d for e, d in zip(enc, docs))
            stats[(lang, name)] = (n / words, nbytes / n, chars / n, rt)

    names = ["bytes", *toks]
    print("\nTokenizer study (held-out val text, ~2 MB per language)\n")
    head = f"{'lang':<5}" + "".join(f"{n:>34}" for n in names)
    print(head)
    print(f"{'':<5}" + "".join(f"{'tok/word bytes/tok chars/tok  xEN RT':>34}" for _ in names))
    print("-" * len(head))
    for lang in texts:
        row = f"{lang:<5}"
        for n in names:
            tw, bt, ct, rt = stats[(lang, n)]
            x = tw / stats[("eng", n)][0]
            row += f"{tw:>9.2f}{bt:>10.2f}{ct:>10.2f}{x:>5.2f}{'  ok' if rt else ' BAD'}"
        print(row)
    bad = [k for k, v in stats.items() if not v[3]]
    print(f"\nround trip: {'exact on every language and tokenizer' if not bad else f'FAILED {bad}'}")
    for n in toks:
        indic = [stats[(l, n)][1] for l in ("kan", "tam", "tel", "mal")]
        print(f"{n:<18} mean Indic bytes/tok {sum(indic) / 4:5.2f}   English {stats[('eng', n)][1]:5.2f}")


if __name__ == "__main__":
    main()
