"""
Download + clean Wikipedia into plain-text shards for tokenization.

Defaults to **Simple English Wikipedia**, not the full English dump --
sized to actually match the ~100K-1M param tier (see
docs/training-time-estimate.md): ~226K articles / ~161MB text, versus full
enwiki's ~16GB. Full enwiki is ~100-1000x more data than a model this size
can usefully absorb in one pass; Simple Wikipedia comfortably covers the
Chinchilla-optimal token budget for every current preset with room left
for books, and one full pass takes hours, not days.

Switch --source to "wikimedia/wikipedia" (config e.g. "20231101.en") once
the model has scaled well past the 1M-param stretch tier and can actually
make use of that much more data.
"""

import argparse

from _shard_io import write_shards, self_test as _shard_io_self_test


def iter_articles(source: str, config: str | None = None, split: str = "train"):
    """Yield cleaned article text from an HF dataset (network + `datasets`
    package only needed here, at actual run time).

    `config` is the dataset's config name -- `wikimedia/wikipedia` is one repo
    holding ~300 languages and needs one (e.g. "20231101.en", "20231101.tcy").
    The docstring above has told people to pass a config since this file was
    written; there was just never an argument to pass it through.
    """
    from datasets import load_dataset
    ds = load_dataset(source, config, split=split) if config else load_dataset(source, split=split)
    text_field = "text" if "text" in ds.column_names else ds.column_names[0]
    for row in ds:
        yield row[text_field]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/wikipedia")
    p.add_argument("--source", default="pszemraj/simple_wikipedia",
                    help="HF dataset id; switch to 'wikimedia/wikipedia' (config '20231101.en') "
                         "once the model has grown well past the 1M-param stretch tier")
    p.add_argument("--config", default=None,
                    help="HF dataset config name, required by multi-language repos "
                         "(e.g. --source wikimedia/wikipedia --config 20231101.tcy)")
    p.add_argument("--shard-size", type=int, default=5000, help="articles per shard file")
    p.add_argument("--self-test", action="store_true",
                    help="check the shard-writer logic with fake data and exit (no network)")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    n = write_shards(iter_articles(args.source, args.config), args.out_dir, args.shard_size)
    print(f"[done] wrote {n} shard(s) to {args.out_dir}")


def _self_test():
    """No network: the shard writer, plus proof --config reaches load_dataset."""
    import sys
    import types

    _shard_io_self_test()

    seen = []

    class FakeDS(list):
        column_names = ["id", "text"]

    def load_dataset(*a, **kw):
        seen.append((a, kw))
        return FakeDS([{"text": "one"}, {"text": "two"}])

    real = sys.modules.get("datasets")
    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=load_dataset)
    try:
        # with a config, it is passed positionally, where HF expects it
        assert list(iter_articles("wikimedia/wikipedia", "20231101.tcy")) == ["one", "two"]
        assert seen[-1] == (("wikimedia/wikipedia", "20231101.tcy"), {"split": "train"}), seen[-1]

        # without one, the call shape is exactly what it was before --config
        # existed, so the Simple-Wikipedia default cannot have been broken
        list(iter_articles("pszemraj/simple_wikipedia"))
        assert seen[-1] == (("pszemraj/simple_wikipedia",), {"split": "train"}), seen[-1]
    finally:
        if real is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = real

    print("[self-test] prepare_wikipedia ok")


if __name__ == "__main__":
    main()
