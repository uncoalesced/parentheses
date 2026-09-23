"""
Download + clean public-domain books (Project Gutenberg etc.) into
plain-text shards for tokenization.

Engineered by uncoalesced

Pulls from a pre-packaged Gutenberg mirror dataset on Hugging Face (e.g.
`manu/project_gutenberg`) rather than scraping gutenberg.org directly, and
strips Gutenberg's standard license header/footer boilerplate from each
text before it enters the corpus. At this param scale "a handful of
shorter books is plenty" (see data/README.md) -- --limit defaults small.
"""

import argparse
import re

from _shard_io import write_shards, self_test as _shard_io_self_test

GUTENBERG_HEADER_RE = re.compile(r"\*\*\* START OF.*?\*\*\*", re.DOTALL)
GUTENBERG_FOOTER_RE = re.compile(r"\*\*\* END OF.*", re.DOTALL)


def strip_gutenberg_boilerplate(text: str) -> str:
    text = GUTENBERG_HEADER_RE.split(text, maxsplit=1)[-1]
    text = GUTENBERG_FOOTER_RE.split(text, maxsplit=1)[0]
    return text.strip()


def iter_books(source: str, limit: int, lang: str = "en"):
    """Yield cleaned book text from an HF Gutenberg-mirror dataset, capped
    at --limit books. manu/project_gutenberg splits by language (there's no
    "train" split -- each language, e.g. "en", is its own split, and the
    "en" split alone is 12GB+), so this streams instead of downloading the
    whole split just to take the first --limit rows."""
    from datasets import load_dataset
    ds = load_dataset(source, split=lang, streaming=True)
    count = 0
    for row in ds:
        if count >= limit:
            break
        text = strip_gutenberg_boilerplate(row["text"] or "")
        if text:
            count += 1
            yield text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/books")
    p.add_argument("--source", default="manu/project_gutenberg",
                    help="HF Gutenberg-mirror dataset id")
    p.add_argument("--lang", default="en", help="language split (dataset splits by language, not a column)")
    p.add_argument("--limit", type=int, default=50, help="max number of books to pull")
    p.add_argument("--shard-size", type=int, default=25, help="books per shard file")
    p.add_argument("--self-test", action="store_true",
                    help="check the shard-writer + boilerplate-strip logic and exit (no network)")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    n = write_shards(iter_books(args.source, args.limit, args.lang), args.out_dir, args.shard_size)
    print(f"[done] wrote {n} shard(s) to {args.out_dir}")


def _self_test():
    sample = "PREAMBLE\n*** START OF THIS PROJECT GUTENBERG EBOOK ***\nBODY TEXT\n*** END OF THIS PROJECT GUTENBERG EBOOK ***\nlicense stuff"
    assert strip_gutenberg_boilerplate(sample) == "BODY TEXT", strip_gutenberg_boilerplate(sample)
    _shard_io_self_test()
    print("[self-test] prepare_books ok")


if __name__ == "__main__":
    main()
