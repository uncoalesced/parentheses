"""
Shared shard-writing helper for the data prep scripts (prepare_wikipedia.py,
prepare_books.py). Both pull text from an HF dataset and write it to
out_dir as plain-text shards; data/tokenize_corpus.py reads those shards back in.
One tiny shared function instead of duplicating it in both scripts.
"""

import os


def write_shards(texts, out_dir: str, shard_size: int = 5000) -> int:
    """Write an iterable of text strings to out_dir/shard_NNNNN.txt,
    shard_size texts per file, blank-line separated. Returns shard count."""
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    buf = []
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        buf.append(text)
        if len(buf) == shard_size:
            _flush(buf, out_dir, idx)
            idx += 1
            buf = []
    if buf:
        _flush(buf, out_dir, idx)
        idx += 1
    return idx


def _flush(buf, out_dir, idx):
    with open(os.path.join(out_dir, f"shard_{idx:05d}.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(buf))


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        n = write_shards(iter(["a", "b", "", "c"]), tmp, shard_size=2)
        assert n == 2, n
        assert sorted(os.listdir(tmp)) == ["shard_00000.txt", "shard_00001.txt"]
        assert open(os.path.join(tmp, "shard_00000.txt")).read() == "a\n\nb"
        assert open(os.path.join(tmp, "shard_00001.txt")).read() == "c"
    print("[self-test] _shard_io ok")


if __name__ == "__main__":
    self_test()
