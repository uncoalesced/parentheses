"""
Pipeline step 4 (see data/README.md): turn the raw text shards written by
prepare_wikipedia.py / prepare_books.py into the uint16 binary files
train.py's np.memmap loader expects (data/processed/train.bin, val.bin).

Engineered by uncoalesced

Byte-level by default (vocab_size=256, matches the 100k/300k/600k presets --
`text.encode("utf-8")`, no tokenizer training needed). Pass --tokenizer-dir
to encode with a trained BPE tokenizer instead (the 1m preset -- see
data/tokenizer/train_tokenizer.py).
"""

import argparse
import glob
import json
import os

import numpy as np

MANIFEST = "split_manifest.json"


def iter_sources(raw_dirs):
    """-> [(source_dir, [shard paths])], one entry per directory holding shards.

    The unit of the train/val split. `--raw-dirs data/raw/parallel` is one
    argument but 22 languages, so grouping by the directory a shard actually
    lives in -- not by the argument it was reached through -- is what makes
    `data/raw/parallel/en-kn` its own source. Sorted so a rerun over the same
    disk produces byte-identical bins.
    """
    groups = {}
    for raw_dir in raw_dirs:
        for path in sorted(glob.glob(os.path.join(raw_dir, "**", "shard_*.txt"), recursive=True)):
            groups.setdefault(os.path.dirname(path).replace(os.sep, "/"), []).append(path)
    return sorted(groups.items())


def _copy(src, dst, count: int):
    """Copy `count` uint16 tokens from an open binary file to another."""
    left = count * 2
    while left > 0:
        buf = src.read(min(left, 1 << 26))
        if not buf:
            break
        dst.write(buf)
        left -= len(buf)


def encode(text: str, tokenizer_dir) -> np.ndarray:
    """-> uint16 token ids for one shard's text."""
    if tokenizer_dir:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        return np.array(tok.encode(text).ids, dtype=np.uint16)
    # Byte-level: the UTF-8 bytes already are the ids. The old
    # `list(text.encode("utf-8"))` built a Python list one int object per byte,
    # which on a 1.6GB corpus is the single dominant cost of the whole
    # pipeline; frombuffer produces the identical numbers with no list at all.
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)


def write_bins(raw_dirs, tokenizer_dir, val_fraction: float, out_dir: str):
    """Encode every shard straight to disk, split per source, write the bins.
    Returns (train_tokens, val_tokens, per_source dict).

    Every source directory contributes its own `val_fraction` to val and the
    rest to train. Before 2026-08-31 this took one cut across the whole
    concatenated stream instead, and since sources are concatenated in the
    order given and shards sort alphabetically inside each, the tail 1% came
    entirely from whichever directory sorted last -- `data/raw/parallel/en-zh`.
    The recorded symptom was `val covers : zh 100%` in
    samples/translation_samples.txt: a val loss that measured Chinese and
    nothing else, for a model trained on 22 languages.

    Still streams. Peak memory is one shard, same as before; peak extra disk is
    one source's tokens rather than the whole corpus's, so this is strictly
    cheaper than the `_all.bin` it replaces. Each source's val slice is the
    tail of that source's own stream, which is the direct per-source analogue
    of what the old single cut did, and keeps train contiguous within a source.

    A source of 2+ tokens always contributes at least 1 token to val -- the
    whole point here is that no source is silently absent, and int() on a
    small source would round its share to zero.
    """
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_source.bin")
    per_source, total_train, total_val = {}, 0, 0
    with open(os.path.join(out_dir, "train.bin"), "wb") as tf,          open(os.path.join(out_dir, "val.bin"), "wb") as vf:
        for name, paths in iter_sources(raw_dirs):
            n = 0
            with open(tmp, "wb") as f:
                for path in paths:
                    with open(path, encoding="utf-8") as sh:
                        a = encode(sh.read(), tokenizer_dir)
                    a.tofile(f)
                    n += len(a)
            val_n = max(1, int(n * val_fraction)) if n > 1 else 0
            with open(tmp, "rb") as src:
                _copy(src, tf, n - val_n)
                _copy(src, vf, val_n)
            per_source[name] = {"train": n - val_n, "val": val_n,
                                "val_offset": total_val, "shards": len(paths)}
            total_train += n - val_n
            total_val += val_n
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"val_fraction": val_fraction, "tokenizer_dir": tokenizer_dir,
                   "train_tokens": total_train, "val_tokens": total_val,
                   "sources": per_source}, f, indent=2)
    return total_train, total_val, per_source


def verify(out_dir: str) -> int:
    """Independently confirm every source really is in val.bin. -> failures.

    Does not trust write_bins' own bookkeeping. Reads the manifest only for
    each source's offset and length in val.bin, then pulls those exact tokens
    off disk, decodes them, and checks they match the tail of that source's
    last shard read straight from the raw directory. That is the same kind of
    evidence the original bug was caught with (val content vs source content),
    not a restatement of the intent.

    Byte-level only -- with a BPE tokenizer the val tokens are ids, not bytes,
    so the decode check is skipped and only the offsets are audited.
    """
    with open(os.path.join(out_dir, MANIFEST), encoding="utf-8") as f:
        man = json.load(f)
    val = np.memmap(os.path.join(out_dir, "val.bin"), dtype=np.uint16, mode="r")
    fails, at = 0, 0
    if len(val) != man["val_tokens"]:
        print(f"[FAIL] val.bin holds {len(val):,} tokens, manifest says {man['val_tokens']:,}")
        fails += 1
    for name, s in man["sources"].items():
        if s["val"] < 1:
            print(f"[FAIL] {name}: contributed 0 tokens to val")
            fails += 1
            continue
        if s["val_offset"] != at:
            print(f"[FAIL] {name}: manifest offset {s['val_offset']:,}, expected {at:,}")
            fails += 1
        at += s["val"]
        if man["tokenizer_dir"]:
            continue
        # last <=4KB of this source's val slice vs the last <=4KB of its own
        # last shard on disk; both are the tail of the same byte stream.
        #
        # The shard must be re-read in TEXT mode, the way write_bins reads it.
        # These files are CRLF on disk and Python's universal newlines collapse
        # each CRLF to a single LF before encoding, so the bytes on disk and
        # the
        # tokens in the bin legitimately differ -- comparing the raw file bytes
        # instead reports a false failure on every CRLF source, which is what
        # this check did on its first run against the real corpus.
        k = min(s["val"], 4096)
        got = bytes(val[s["val_offset"] + s["val"] - k: s["val_offset"] + s["val"]].tolist())
        last = sorted(glob.glob(os.path.join(name, "**", "shard_*.txt"), recursive=True))[-1]
        with open(last, encoding="utf-8") as f:
            want = f.read().encode("utf-8")[-k:]
        if got != want:
            print(f"[FAIL] {name}: val tail does not match its last shard's tail")
            fails += 1
    covered = sum(1 for s in man["sources"].values() if s["val"] > 0)
    print(f"[verify] {covered}/{len(man['sources'])} sources present in val.bin, "
          f"{man['train_tokens']:,} train / {man['val_tokens']:,} val tokens, "
          f"{fails} failure(s)")
    return fails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dirs", nargs="+", default=["data/raw/wikipedia", "data/raw/books"])
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--tokenizer-dir", default=None,
                    help="trained BPE tokenizer dir (1m preset only); omit for byte-level")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--self-test", action="store_true",
                    help="check the encode/split logic on fake shards and exit")
    p.add_argument("--verify", action="store_true",
                    help="audit an existing --out-dir's val.bin against the raw shards and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.verify:
        raise SystemExit(1 if verify(args.out_dir) else 0)

    train_n, val_n, per_source = write_bins(args.raw_dirs, args.tokenizer_dir,
                                            args.val_fraction, args.out_dir)
    for name, src in per_source.items():
        print(f"  {name:<34} {src['train']:>13,} train  {src['val']:>10,} val")
    print(f"[done] {train_n:,} train tokens, {val_n:,} val tokens from "
          f"{len(per_source)} sources -> {args.out_dir}")
    raise SystemExit(1 if verify(args.out_dir) else 0)


def _self_test():
    import tempfile

    CRLF = chr(13) + chr(10)

    def shard(d, i, text):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "shard_%05d.txt" % i), "w", encoding="utf-8") as f:
            f.write(text)

    # single source: unchanged behaviour, tail is val, nothing is lost
    with tempfile.TemporaryDirectory() as tmp:
        shard(os.path.join(tmp, "wiki"), 0, "hello")
        out = os.path.join(tmp, "out")
        train_n, val_n, per = write_bins([os.path.join(tmp, "wiki")], None, 0.2, out)
        assert (train_n, val_n) == (4, 1), (train_n, val_n)
        combined = list(np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)) +                    list(np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16))
        assert combined == list("hello".encode("utf-8")), combined
        assert not os.path.exists(os.path.join(out, "_source.bin"))   # scratch cleaned up
        assert verify(out) == 0

    # the actual bug. Two very lopsided sources reached through ONE --raw-dirs
    # argument, the small one sorting last -- exactly data/raw/parallel's shape
    # with en-zh at the end. The old single-cut split put 100% of val in the
    # last source; every source must now appear, proportionally.
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "parallel")
        shard(os.path.join(root, "en-aa"), 0, "A" * 8000)
        shard(os.path.join(root, "en-aa"), 1, "B" * 2000)
        shard(os.path.join(root, "en-zz"), 0, "Z" * 1000)
        out = os.path.join(tmp, "out")
        train_n, val_n, per = write_bins([root], None, 0.1, out)

        assert len(per) == 2, per                       # subdirs, not the argument
        assert set(os.path.basename(k) for k in per) == {"en-aa", "en-zz"}, per
        aa = per[[k for k in per if k.endswith("en-aa")][0]]
        zz = per[[k for k in per if k.endswith("en-zz")][0]]
        assert (aa["train"], aa["val"]) == (9000, 1000), aa
        assert (zz["train"], zz["val"]) == (900, 100), zz
        assert aa["shards"] == 2 and zz["shards"] == 1

        val = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16)
        assert (train_n, val_n) == (9900, 1100), (train_n, val_n)
        assert len(val) == 1100
        assert set(val[:1000].tolist()) == {ord("B")}, "en-aa's val slice is wrong"
        assert set(val[1000:].tolist()) == {ord("Z")}, "en-zz's val slice is wrong"
        # nothing dropped: train + val is the whole corpus, per source
        train = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)
        assert (train == ord("A")).sum() == 8000 and (train == ord("B")).sum() == 1000
        assert (train == ord("Z")).sum() == 900
        assert verify(out) == 0

    # a source too small for its proportional share still lands in val, and a
    # 1-token source is allowed to have none rather than crashing -- verify()
    # is what flags that, so it is not silent either way
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r")
        shard(os.path.join(root, "big"), 0, "x" * 5000)
        shard(os.path.join(root, "small"), 0, "yy")
        shard(os.path.join(root, "tiny"), 0, "z")
        out = os.path.join(tmp, "out")
        _, _, per = write_bins([root], None, 0.01, out)
        assert per[[k for k in per if k.endswith("small")][0]]["val"] == 1
        assert per[[k for k in per if k.endswith("tiny")][0]]["val"] == 0
        assert verify(out) == 1, "verify must flag the source with no val tokens"

    # CRLF shards: the real corpus is CRLF on disk and text-mode reads collapse
    # it, so verify() has to compare against the same collapsed bytes
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r")
        os.makedirs(os.path.join(root, "crlf"))
        with open(os.path.join(root, "crlf", "shard_00000.txt"), "wb") as f:
            f.write((("line one" + CRLF + "line two" + CRLF) * 200).encode())
        out = os.path.join(tmp, "out")
        _, val_n, _ = write_bins([root], None, 0.1, out)
        assert val_n == int(200 * 18 * 0.1), val_n   # CRLF is one token
        assert verify(out) == 0, "verify tripped over CRLF shards"

    # verify() must actually fail on a corrupted val.bin, or it proves nothing
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r")
        shard(os.path.join(root, "one"), 0, "abcdefghij" * 100)
        shard(os.path.join(root, "two"), 0, "klmnopqrst" * 100)
        out = os.path.join(tmp, "out")
        write_bins([root], None, 0.1, out)
        assert verify(out) == 0
        v = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16)
        v[-1] = ord("!")
        v.tofile(os.path.join(out, "val.bin"))
        assert verify(out) == 1, "verify passed a corrupted val.bin"

    print("[self-test] tokenize ok")


if __name__ == "__main__":
    main()
