"""
Throughput: plain Free Think vs Modular Free Think on each retrieval backend.

handoff-vector-memory.md holds Modular Free Think to the tokens/sec bar the
BM25 path already cleared (0.98x GPU vs plain Free Think), and says plainly
that a vector query costs a full forward pass through the frozen backbone,
which is not obviously as cheap as a BM25 lexical score, and had not been
measured. This measures it. All three configurations decode the same number
of tokens from the same prompt with the same seed, after a warm-up pass.

    python3 scripts/benchmark_retrieval.py --checkpoint checkpoints/multilingual/step_278999_final.pt
    python3 scripts/benchmark_retrieval.py --self-test
"""

import argparse
import glob
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PRESETS, Parentheses
from model.embedding_head import EmbeddingHead, encode, load_trained
from features.free_think import FreeThinkSession
from features.modular_free_think import IngestedStore, ModularFreeThinkSession


def make_embedder(backbone, head, device: str):
    """texts -> (N, dim) float32 unit vectors, for IngestedStore's vector path."""
    def embed(texts):
        return encode(backbone, head, list(texts), device).float().cpu().numpy()
    return embed


def timed(fn, tokens: int) -> tuple[float, float]:
    """Run `fn` once and return (seconds, tokens/sec)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, tokens / dt


def corpus_file(raw_dir: str, out_path: str, max_bytes: int) -> int:
    """Concatenate real shard text into one file. Returns bytes written."""
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(os.path.join(raw_dir, "**", "shard_*.txt"), recursive=True)):
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            out.write(text[:max_bytes - written])
            written = min(written + len(text), max_bytes)
            if written >= max_bytes:
                break
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/all-sources-v1/step_560999_final.pt",
                   help="backbone checkpoint (all-sources-v1, the current primary, by default; "
                        "older checkpoints are still usable via this flag explicitly)")
    p.add_argument("--head", default="checkpoints/all-sources-v1/embedding_head_all_sources_v1.pt",
                   help="a head only means anything beside the exact backbone it trained "
                        "against, so this default moves with --checkpoint")
    p.add_argument("--raw-dir", default="data/raw/wikipedia")
    p.add_argument("--corpus-bytes", type=int, default=400_000)
    p.add_argument("--prompt", default="The ocean is very deep.")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--splice-every", type=int, default=64)
    p.add_argument("--retrieve-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=7, help="timed runs per configuration; median wins")
    p.add_argument("--out", default="samples/retrieval_benchmark.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    # load_trained() also refuses a head trained against a different
    # backbone -- the manual load this replaced didn't check that.
    try:
        model, head, meta = load_trained(args.checkpoint, args.head, args.device)
    except ValueError as e:
        raise SystemExit(str(e))
    embedder = make_embedder(model, head, args.device)

    with tempfile.TemporaryDirectory() as tmp:
        cpath = os.path.join(tmp, "corpus.txt")
        nbytes = corpus_file(args.raw_dir, cpath, args.corpus_bytes)
        paths = [Path(cpath)]

        t0 = time.perf_counter()
        bm25 = IngestedStore(backend="bm25")
        bm25.ingest(paths)
        bm25_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        vec = IngestedStore(backend="vector", embedder=embedder)
        vec.ingest(paths)
        vec_build = time.perf_counter() - t0

    n = len(bm25.documents)
    print(f"[corpus] {nbytes:,} bytes -> {n:,} chunks from {args.raw_dir}")
    print(f"[index]  bm25 {bm25_build:.2f}s   vector {vec_build:.2f}s "
          f"(embedding {n:,} chunks through the backbone)")

    def plain():
        torch.manual_seed(args.seed)
        FreeThinkSession(model).run(args.prompt, args.tokens)

    def modular(store):
        def go():
            torch.manual_seed(args.seed)
            ModularFreeThinkSession(model, store, splice_every=args.splice_every,
                                    retrieve_k=args.retrieve_k).run(args.prompt, args.tokens)
        return go

    runs = [("plain free think", plain),
            ("modular + bm25", modular(bm25)),
            ("modular + vector", modular(vec))]

    # Medians over repeats, not a single sample: a run is only a couple of
    # seconds, so one timing is well inside the noise of the differences being
    # compared here.
    results = {}
    for name, fn in runs:
        fn()                                    # warm-up, not measured
        samples = [timed(fn, args.tokens) for _ in range(args.repeats)]
        tps_all = sorted(s[1] for s in samples)
        med = tps_all[len(tps_all) // 2]
        results[name] = (float(np.median([s[0] for s in samples])), med, tps_all[0], tps_all[-1])
        print(f"[bench] {name:<18} {med:8.1f} tok/s median of {args.repeats} "
              f"(min {tps_all[0]:.1f}, max {tps_all[-1]:.1f})")

    base = results["plain free think"][1]
    lines = [
        f"repeats        : {args.repeats} per configuration, median reported",
        f"checkpoint     : {args.checkpoint}",
        f"head           : {args.head} (dim {meta['dim']}, recall@1 {meta.get('recall_at_1')})",
        f"device         : {args.device}",
        f"corpus         : {nbytes:,} bytes -> {n:,} chunks from {args.raw_dir}",
        f"decode         : {args.tokens} tokens, splice_every={args.splice_every}, "
        f"retrieve_k={args.retrieve_k}, seed={args.seed}",
        f"index build    : bm25 {bm25_build:.2f}s, vector {vec_build:.2f}s",
        "",
    ]
    for name, (dt, tps, lo, hi) in results.items():
        lines.append(f"{name:<18} {dt:6.2f}s  {tps:8.1f} tok/s median "
                     f"(min {lo:7.1f}, max {hi:7.1f})   {tps / base:.3f}x vs plain")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.out}")


def _self_test():
    torch.manual_seed(0)
    cfg = PRESETS["parentheses-0.9-100k"]
    model = Parentheses(cfg).eval()
    head = EmbeddingHead(cfg.n_embd, dim=32).eval()
    embed = make_embedder(model, head, "cpu")

    v = embed(["one two", "three four"])
    assert v.shape == (2, 32) and v.dtype == np.float32, (v.shape, v.dtype)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    # corpus_file concatenates shards and respects the byte cap
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        for i in range(3):
            Path(raw, f"shard_0000{i}.txt").write_text("word " * 200, encoding="utf-8")
        out = os.path.join(tmp, "c.txt")
        got = corpus_file(raw, out, 300)
        assert got == 300, got
        assert len(Path(out).read_text(encoding="utf-8")) == 300

    # timed() reports a positive rate
    dt, tps = timed(lambda: sum(range(10000)), 400)
    assert dt > 0 and tps > 0

    print("[self-test] benchmark_retrieval ok")


if __name__ == "__main__":
    main()
