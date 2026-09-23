r"""
Pipeline step 5 (see data/README.md): the instruction/conversation stage.

Engineered by uncoalesced

Steps 1-4 all end in one undifferentiated byte stream that train.py runs
next-token prediction over. That teaches a model what a transcript looks
like, not how to answer when asked. This script is the other half: it
converts conversation-shaped sources into a turn-delimited flat format and
emits, alongside each shard, the byte ranges that belong to assistant turns
-- which is what lets train.py score the loss on the response only
(--sft-spans, see train.py).

Turn format. The delimiters are bytes, not words: the byte-level vocab has no
reserved slots to spend, and a text marker like "### Assistant:" collides with
real content (this corpus is full of transcripts about transcripts).

    \x01<role>\x02<content>\x03    \x01 start-of-turn, \x02 role sep, \x03 end-of-turn

Conversations are blank-line separated inside a shard, the same document
convention the rest of the pipeline uses. Only `gpt` turns are marked as
assistant spans -- a `tool` turn is the environment talking, and training the
model to emit it would teach it to hallucinate tool output.

    python3 data/prepare_sft.py --src data/raw/manual/raw-sourced/hermes-function-calling-v1
    python3 data/prepare_sft.py --pack
    python3 data/prepare_sft.py --self-test

Staging: shards land in data/raw/sft/<source>/, which no --raw-dirs default
sweeps (same rule as data/raw/manual/) so SFT data can never be silently
mixed into the pretraining corpus. --pack is the deliberate step that turns
it into bins: data/processed_sft/{train,val}.bin plus the parallel
{train,val}.mask.bin that train.py reads.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# Same sibling-import shape as prepare_books.py's `from _shard_io import ...`,
# but this module is also imported as `data.prepare_sft` (scripts/eval_persona.py
# wants the delimiters), and that entry point does not put data/ on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenize_corpus import iter_sources, write_bins

START, SEP, END = "\x01", "\x02", "\x03"
CONTROL = (START, SEP, END)

# ShareGPT's `from` values -> our role names. Anything else is a new dataset
# shape, which is a decision, not a default: flatten() raises on it.
ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
ASSISTANT = "assistant"

# hh-rlhf's "chosen" side encodes harmlessness preference as well as
# helpfulness, so a wholesale pull would bake refusal behaviour into a model
# whose whole point is not having any. project-gaps-and-fixes.md item 1 leaves
# that filter's design open (Joel's call); until it exists the source is not
# usable here, and that is enforced rather than remembered.
BLOCKED = {"hh-rlhf": "hh-rlhf needs a refusal/hedge filter on its 'chosen' side before it "
                      "can be used for SFT -- see project-gaps-and-fixes.md item 1. Not a "
                      "default; someone has to design that filter first."}


def flatten(conversations) -> tuple[str, list[tuple[int, int]]]:
    """One ShareGPT `conversations` array -> (turn-delimited text, assistant spans).

    Spans are [start, end) byte offsets into the returned text, covering the
    assistant's content *and* its end-of-turn byte -- that byte is the only
    signal that teaches the model to stop, so it has to be supervised too.
    """
    parts, spans, at = [], [], 0
    for turn in conversations:
        src = turn.get("from")
        if src not in ROLES:
            raise ValueError(f"unknown ShareGPT role {src!r} (known: {sorted(ROLES)})")
        role = ROLES[src]
        content = turn.get("value") or ""
        for c in CONTROL:                      # or the delimiters stop delimiting
            content = content.replace(c, "")
        # Hermes content carries CRLF in places. Everything downstream reads
        # shards in text mode (write_bins does), so a \r would vanish between
        # the offsets recorded here and the bytes that reach the bin -- 31 of
        # them in one real shard, every span after the first one shifted.
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        head = START + role + SEP
        body = content + END
        at += len(head.encode("utf-8"))
        n = len(body.encode("utf-8"))
        if role == ASSISTANT:
            spans.append((at, at + n))
        at += n
        parts.append(head)
        parts.append(body)
    return "".join(parts), spans


def iter_sharegpt(paths):
    """Yield (text, spans) per conversation across a list of ShareGPT JSON files."""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            convs = rec.get("conversations")
            if convs:
                yield flatten(convs)


def spans_path(shard: str) -> str:
    return shard[:-len(".txt")] + ".spans.json"


def write_sft_shards(items, out_dir: str, shard_size: int = 500) -> tuple[int, int]:
    """Write (text, spans) pairs to out_dir as shard_NNNNN.txt + .spans.json.

    Not data/_shard_io.py's write_shards: that one strips and drops texts,
    which silently shifts every byte offset after it. Offsets are the whole
    point here, so the writer that produces them owns the joining.
    """
    os.makedirs(out_dir, exist_ok=True)
    idx, spans_n, buf = 0, 0, []
    for item in items:
        buf.append(item)
        if len(buf) == shard_size:
            spans_n += _flush(buf, out_dir, idx)
            idx, buf = idx + 1, []
    if buf:
        spans_n += _flush(buf, out_dir, idx)
        idx += 1
    return idx, spans_n


def _flush(buf, out_dir, idx) -> int:
    texts, spans, at = [], [], 0
    for text, sp in buf:
        spans.extend([at + s, at + e] for s, e in sp)
        texts.append(text)
        at += len(text.encode("utf-8")) + 2        # the "\n\n" joiner
    body = "\n\n".join(texts)
    nbytes = len(body.encode("utf-8"))
    assert nbytes == at - 2, (nbytes, at)
    # newline="\n": the default would translate every \n to \r\n on Windows,
    # and every offset recorded above would be wrong by the number of newlines
    # in front of it.
    with open(os.path.join(out_dir, f"shard_{idx:05d}.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(body)
    with open(os.path.join(out_dir, f"shard_{idx:05d}.spans.json"), "w", encoding="utf-8") as f:
        json.dump({"bytes": nbytes, "spans": spans}, f)
    return len(spans)


def shard_mask(shard: str) -> np.ndarray:
    """-> uint8 mask over one shard's bytes, 1 where an assistant turn is."""
    with open(shard, encoding="utf-8") as f:       # text mode, as write_bins reads it
        n = len(f.read().encode("utf-8"))
    sp = spans_path(shard)
    if not os.path.exists(sp):
        raise FileNotFoundError(f"{shard} has no {os.path.basename(sp)}; SFT shards must "
                                f"carry their spans (was this dir written by prepare_sft.py?)")
    with open(sp, encoding="utf-8") as f:
        meta = json.load(f)
    if meta["bytes"] != n:
        raise ValueError(f"{shard}: {n} bytes on disk, spans file says {meta['bytes']}. The "
                         f"shard changed after its spans were written -- masks would be "
                         f"misaligned, which supervises the wrong bytes silently.")
    mask = np.zeros(n, dtype=np.uint8)
    for s, e in meta["spans"]:
        mask[s:e] = 1
    return mask


def pack(sft_dirs, out_dir: str, val_fraction: float):
    """Encode the SFT shards to bins and write the parallel loss masks.

    Alignment is not re-derived: write_bins does the encoding and reports each
    source's train/val token counts, and the masks stream through the same
    iter_sources() order and are cut at the same point. Reimplementing the
    split here instead is how a mask drifts one shard out of step with its
    data while everything still runs.
    """
    train_n, val_n, per_source = write_bins(sft_dirs, None, val_fraction, out_dir)
    ones = 0
    with open(os.path.join(out_dir, "train.mask.bin"), "wb") as tf, \
         open(os.path.join(out_dir, "val.mask.bin"), "wb") as vf:
        for name, paths in iter_sources(sft_dirs):
            left = per_source[name]["train"]
            for path in paths:
                m = shard_mask(path)
                ones += int(m.sum())
                if left >= len(m):
                    tf.write(m.tobytes())
                    left -= len(m)
                elif left > 0:
                    tf.write(m[:left].tobytes())
                    vf.write(m[left:].tobytes())
                    left = 0
                else:
                    vf.write(m.tobytes())
    for split, n in (("train", train_n), ("val", val_n)):
        got = os.path.getsize(os.path.join(out_dir, f"{split}.mask.bin"))
        assert got == n, f"{split}.mask.bin is {got} bytes for {n} tokens"
    return train_n, val_n, ones, per_source


def _check_not_blocked(src: str):
    low = os.path.basename(os.path.normpath(src)).lower()
    for key, why in BLOCKED.items():
        if key in low:
            raise SystemExit(f"[refused] {src}: {why}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/manual/raw-sourced/hermes-function-calling-v1",
                   help="directory of ShareGPT .json files (or one .json file)")
    p.add_argument("--out-dir", default=None,
                   help="shard output dir (default data/raw/sft/<source name>)")
    p.add_argument("--shard-size", type=int, default=500, help="conversations per shard")
    p.add_argument("--pack", action="store_true",
                   help="encode the data/raw/sft shards to bins + loss masks and exit")
    p.add_argument("--sft-dirs", nargs="+", default=["data/raw/sft"],
                   help="--pack only: shard dirs to encode")
    p.add_argument("--processed-dir", default="data/processed_sft",
                   help="--pack only: bin output dir")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--self-test", action="store_true",
                   help="check the turn format, spans and packing on fake data and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.pack:
        train_n, val_n, ones, per_source = pack(args.sft_dirs, args.processed_dir, args.val_fraction)
        for name, s in per_source.items():
            print(f"  {name:<44} {s['train']:>12,} train  {s['val']:>9,} val")
        total = train_n + val_n
        print(f"[done] {train_n:,} train / {val_n:,} val tokens -> {args.processed_dir}; "
              f"{ones:,} supervised ({ones/max(total, 1):.1%} of bytes are assistant turns)")
        return

    _check_not_blocked(args.src)
    paths = ([args.src] if args.src.endswith(".json")
             else sorted(glob.glob(os.path.join(args.src, "*.json"))))
    if not paths:
        raise FileNotFoundError(f"no .json files under {args.src}")
    out_dir = args.out_dir or os.path.join("data/raw/sft",
                                           os.path.basename(os.path.normpath(args.src)))
    shards, spans = write_sft_shards(iter_sharegpt(paths), out_dir, args.shard_size)
    print(f"[done] {len(paths)} file(s) -> {shards} shards, {spans:,} assistant turns in {out_dir}")
    print("[next] python3 data/prepare_sft.py --pack")


def _self_test():
    import tempfile

    # flatten: delimiters land where they should, spans cover assistant
    # content plus its end-of-turn byte, and nothing else.
    text, spans = flatten([{"from": "system", "value": "S"},
                           {"from": "human", "value": "Q"},
                           {"from": "gpt", "value": "A"},
                           {"from": "tool", "value": "T"},
                           {"from": "gpt", "value": "B"}])
    assert text == ("\x01system\x02S\x03\x01user\x02Q\x03\x01assistant\x02A\x03"
                    "\x01tool\x02T\x03\x01assistant\x02B\x03"), repr(text)
    raw = text.encode("utf-8")
    assert [raw[s:e].decode() for s, e in spans] == ["A\x03", "B\x03"], spans
    assert len(spans) == 2, "tool turns must not be supervised"

    # control bytes inside content would break the delimiting -> stripped
    t, sp = flatten([{"from": "gpt", "value": "a\x01b\x02c\x03d"}])
    assert t == "\x01assistant\x02abcd\x03", repr(t)
    assert t.encode()[sp[0][0]:sp[0][1]].decode() == "abcd\x03"

    # multibyte content: spans are byte offsets, not character offsets
    t, sp = flatten([{"from": "human", "value": "héllo"}, {"from": "gpt", "value": "wörld"}])
    assert t.encode("utf-8")[sp[0][0]:sp[0][1]].decode("utf-8") == "wörld\x03"

    # CRLF in the source content: normalised here, or the text-mode read that
    # every downstream consumer does silently shortens the shard and every
    # span after the first \r addresses the wrong bytes (hit for real on
    # Hermes shard_00002, 31 bytes out)
    t, sp = flatten([{"from": "human", "value": "a\r\nb\rc"}, {"from": "gpt", "value": "x\r\ny"}])
    assert "\r" not in t and t.count("\n") == 3, repr(t)
    assert t.encode("utf-8")[sp[0][0]:sp[0][1]].decode("utf-8") == "x\ny\x03"

    try:
        flatten([{"from": "assistant", "value": "x"}])
        raise AssertionError("unknown role must raise")
    except ValueError as e:
        assert "assistant" in str(e)

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.json")
        with open(src, "w", encoding="utf-8") as f:
            json.dump([{"conversations": [{"from": "human", "value": "q%d" % i},
                                          {"from": "gpt", "value": "answer%d" % i}]}
                       for i in range(7)], f)
        shards_dir = os.path.join(tmp, "raw", "toy")
        n_shards, n_spans = write_sft_shards(iter_sharegpt([src]), shards_dir, shard_size=3)
        assert (n_shards, n_spans) == (3, 7), (n_shards, n_spans)

        # a shard's spans must still address its own bytes after the round trip
        # through disk (this is what catches CRLF translation on Windows)
        shard0 = os.path.join(shards_dir, "shard_00000.txt")
        m = shard_mask(shard0)
        with open(shard0, encoding="utf-8") as f:
            body = f.read().encode("utf-8")
        assert bytes(np.frombuffer(body, dtype=np.uint8)[m == 1]) == b"answer0\x03answer1\x03answer2\x03"

        out = os.path.join(tmp, "processed")
        train_n, val_n, ones, per = pack([os.path.join(tmp, "raw")], out, 0.1)
        data = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)
        mask = np.fromfile(os.path.join(out, "train.mask.bin"), dtype=np.uint8)
        assert len(data) == len(mask) == train_n, (len(data), len(mask), train_n)
        # the supervised bytes of the packed corpus are exactly the assistant
        # turns, in order -- the property train.py's masking depends on
        sup = bytes(data[mask == 1].astype(np.uint8).tolist())
        assert sup.startswith(b"answer0\x03answer1\x03"), sup[:40]
        assert b"q0" not in sup and b"\x01user\x02" not in sup
        assert ones == sum(len(b"answer%d\x03" % i) for i in range(7)), ones
        vmask = np.fromfile(os.path.join(out, "val.mask.bin"), dtype=np.uint8)
        assert len(vmask) == val_n

        # a shard edited after its spans were written must fail loudly, not
        # supervise whatever now sits at those offsets
        with open(shard0, "a", encoding="utf-8", newline="\n") as f:
            f.write("tail")
        try:
            shard_mask(shard0)
            raise AssertionError("stale spans must raise")
        except ValueError as e:
            assert "misaligned" in str(e)

    try:
        _check_not_blocked("data/raw/manual/raw-sourced/hh-rlhf")
        raise AssertionError("hh-rlhf must be refused until its filter exists")
    except SystemExit as e:
        assert "refusal/hedge filter" in str(e)

    print("[self-test] prepare_sft ok")


if __name__ == "__main__":
    main()
