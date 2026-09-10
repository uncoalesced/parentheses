"""
Duplicate scan across data/raw/manual/raw-sourced/ -- the still-open gap
project-gaps-and-fixes.md calls out ("no dedup or data-hygiene pass across
the now-many sources"). Two passes:

  1. Exact duplicates: hash every chunk's whitespace-collapsed text
     (stdlib hashlib, no model, no accelerator). Cheap, runs over
     everything -- this project has real precedent for exact overlap
     landing in two places (the proof-pile-2 concurrent-session collision;
     OpenStax titles that may share chapters).
  2. Near duplicates: embeds a bounded, per-source-sampled subset of
     chunks via the trained embedding head (see
     scripts/export_embedder_onnx.py) and flags pairs above a cosine
     similarity threshold. This is the part onnxruntime can route to the
     Ryzen AI NPU (VitisAIExecutionProvider) if it's set up, or DirectML,
     falling back to plain CPU -- this script tells you which one actually
     ran rather than silently using CPU.

ponytail: an all-pairs comparison is O(n^2), and this project's
raw-sourced/ already has 11GB+ in project-gutenberg alone -- chunked at
128 bytes that's tens of millions of chunks; no execution provider makes
an O(n^2) scan over that trivial. Pass 2 caps chunks per source
(--max-chunks-per-source, default 2000, reservoir-sampled so a huge
source doesn't just contribute its first N chunks) instead of embedding
the whole corpus. Upgrade path if this ever needs full-corpus coverage:
an ANN index (faiss, or a cheap LSH bucket pass) instead of a dense
similarity matrix.

    python3 scripts/export_embedder_onnx.py   # once, if embedder.onnx doesn't exist yet
    python3 scripts/dedup_raw_sourced.py
    python3 scripts/dedup_raw_sourced.py --self-test
"""

import argparse
import glob
import hashlib
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.modular_free_think import _chunk

RAW_SOURCED = "data/raw/manual/raw-sourced"
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Collapse whitespace before hashing -- two chunks differing only in
    line-wrap or trailing spaces are the same duplicate, not two different
    ones."""
    return _WS.sub(" ", text).strip()


def iter_source_chunks(raw_sourced_dir: str = RAW_SOURCED):
    """Yield (source, path, chunk_index, chunk_text) for every
    shard_*.txt under every raw-sourced/<source>/ directory.

    Only shard_*.txt counts -- same convention data/tokenize_corpus.py's
    iter_sources() uses. A SOURCE.txt or a stray .json sitting in a source
    directory is not corpus text and is silently skipped, on purpose.
    """
    for source_dir in sorted(glob.glob(os.path.join(raw_sourced_dir, "*"))):
        if not os.path.isdir(source_dir):
            continue
        source = os.path.basename(source_dir)
        for path in sorted(glob.glob(os.path.join(source_dir, "**", "shard_*.txt"),
                                      recursive=True)):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            for i, chunk in enumerate(_chunk(text)):
                yield source, path, i, chunk


def find_exact_duplicates(raw_sourced_dir: str = RAW_SOURCED):
    """-> (dupes, total_chunks_seen). dupes maps a hash to every
    (source, path, chunk_index, text) that shares it, for hashes seen more
    than once."""
    seen: dict[str, list] = defaultdict(list)
    n = 0
    for source, path, i, chunk in iter_source_chunks(raw_sourced_dir):
        n += 1
        norm = _norm(chunk)
        if len(norm) < 20:  # a handful of bytes matching by chance isn't a duplicate
            continue
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        seen[h].append((source, path, i, chunk))
    dupes = {h: v for h, v in seen.items() if len(v) > 1}
    return dupes, n


def sample_chunks_per_source(raw_sourced_dir: str, cap: int, seed: int = 0):
    """-> list of (source, path, chunk_index, text), <= cap per source.

    Reservoir sampling per source so a huge source (project-gutenberg)
    contributes a spread sample rather than just its first `cap` chunks.
    """
    rng = random.Random(seed)
    per_source: dict[str, list] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for source, path, i, chunk in iter_source_chunks(raw_sourced_dir):
        counts[source] += 1
        bucket = per_source[source]
        if len(bucket) < cap:
            bucket.append((source, path, i, chunk))
        else:
            j = rng.randint(0, counts[source] - 1)
            if j < cap:
                bucket[j] = (source, path, i, chunk)
    out = []
    for bucket in per_source.values():
        out.extend(bucket)
    return out


def _dml_device_id_avoiding_discrete_gpu(ort) -> int:
    """-> DML device_id of a non-discrete adapter (integrated GPU/NPU), or 0.

    Plain DirectML has no NPU adapter to offer on this machine/driver combo
    (checked via ort.get_ep_devices() -- only the two GPUs show up), and its
    default device_id=0 picks the highest-performance adapter, which on a
    gaming laptop is the discrete GPU. This project trains on that same
    discrete GPU (train.py), often for days unattended, so a dedup run
    defaulting onto it would contend with an in-progress training job
    instead of using the otherwise-idle integrated GPU. Route around it.
    """
    # device_id is DirectML's raw DXGI adapter index (metadata's DxgiAdapterNumber),
    # NOT this list's position -- get_ep_devices() orders by performance preference
    # (discrete GPU first), which on this machine is the opposite of raw adapter order.
    for d in ort.get_ep_devices():
        if d.ep_name == "DmlExecutionProvider" and d.device.metadata.get("Discrete") == "0":
            return int(d.device.metadata["DxgiAdapterNumber"])
    return 0


def _make_session(onnx_path: str):
    """-> (InferenceSession, provider actually used, block_size or None).

    Tries the NPU provider first, then DirectML, then CPU. onnxruntime
    does not error if a requested provider isn't installed/configured --
    it just silently skips it, so checking session.get_providers()[0]
    after construction is the only way to know what actually ran.

    block_size comes back from the .onnx file's own metadata (written by
    scripts/export_embedder_onnx.py) rather than being guessed here --
    RoPE's precomputed table is sized to it at trace time, so a wrong
    guess can index past it on a long chunk instead of just truncating.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = [p for p in ("VitisAIExecutionProvider", "DmlExecutionProvider")
                 if p in available]
    providers = preferred + ["CPUExecutionProvider"]
    provider_options = []
    for p in providers:
        if p == "DmlExecutionProvider":
            provider_options.append({"device_id": str(_dml_device_id_avoiding_discrete_gpu(ort))})
        else:
            provider_options.append({})
    sess = ort.InferenceSession(onnx_path, providers=providers, provider_options=provider_options)
    block_size = sess.get_modelmeta().custom_metadata_map.get("block_size")
    return sess, sess.get_providers()[0], (int(block_size) if block_size else None)


def embed_chunks(onnx_path: str, chunks: list, block_size: int | None = None,
                  batch_size: int = 32):
    """-> (N, dim) float32 unit vectors, run through onnxruntime.

    block_size overrides the value read from the .onnx file's metadata;
    leave it None (the default) unless you know what you're doing.
    """
    from model.embedding_head import batch_bytes

    sess, used, detected_block_size = _make_session(onnx_path)
    if block_size is None:
        if detected_block_size is None:
            raise ValueError(
                f"{onnx_path} has no block_size metadata (exported by an older copy of "
                f"scripts/export_embedder_onnx.py?) -- pass block_size explicitly")
        block_size = detected_block_size

    note = "" if used != "CPUExecutionProvider" else (
        " -- NPU/DirectML provider not available or not selected; see "
        "scripts/export_embedder_onnx.py's docstring for the remaining "
        "manual Ryzen AI Software setup")
    print(f"[info] onnxruntime provider in use: {used}{note}")

    outs = []
    for i in range(0, len(chunks), batch_size):
        idx, mask = batch_bytes(chunks[i:i + batch_size], block_size, "cpu")
        out = sess.run(["embedding"], {"idx": idx.numpy(), "mask": mask.numpy()})[0]
        outs.append(out)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)


def find_near_duplicates(vectors: np.ndarray, items: list, threshold: float = 0.95):
    """Unit vectors -> cosine similarity is a plain dot product. All-pairs
    over `items` -- already capped by the caller, see the module docstring."""
    sims = vectors @ vectors.T
    n = len(items)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            if sims[a, b] >= threshold:
                pairs.append((items[a], items[b], float(sims[a, b])))
    pairs.sort(key=lambda p: -p[2])
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-sourced-dir", default=RAW_SOURCED)
    p.add_argument("--onnx", default="checkpoints/all-sources-v1/embedder.onnx",
                   help="run scripts/export_embedder_onnx.py first if this doesn't exist; "
                        "checkpoints/multilingual-22/embedder.onnx is still usable explicitly")
    p.add_argument("--block-size", type=int, default=None,
                   help="override; normally auto-detected from the .onnx file's own metadata")
    p.add_argument("--max-chunks-per-source", type=int, default=2000)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--report", default="data/raw/DEDUP_REPORT.md")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    print(f"[info] scanning {args.raw_sourced_dir} for exact duplicates...")
    exact, total = find_exact_duplicates(args.raw_sourced_dir)
    print(f"[info] {total:,} chunks scanned, {len(exact)} exact-duplicate group(s)")

    print(f"[info] sampling up to {args.max_chunks_per_source} chunks/source for near-dup pass...")
    sample = sample_chunks_per_source(args.raw_sourced_dir, args.max_chunks_per_source)
    print(f"[info] {len(sample):,} chunks sampled across "
          f"{len({s for s, *_ in sample})} source(s)")

    near = []
    if sample and os.path.exists(args.onnx):
        vectors = embed_chunks(args.onnx, [c for *_, c in sample], args.block_size)
        near = find_near_duplicates(vectors, sample, args.threshold)
        print(f"[info] {len(near)} near-duplicate pair(s) at >= {args.threshold} cosine similarity")
    elif not os.path.exists(args.onnx):
        print(f"[warn] {args.onnx} not found -- skipping near-duplicate pass. "
              f"Run scripts/export_embedder_onnx.py first.")

    _write_report(args.report, exact, near, total, len(sample))
    print(f"[info] report written to {args.report}")


def _write_report(path: str, exact: dict, near: list, total_chunks: int, sampled: int):
    lines = [
        "# Duplicate scan — data/raw/manual/raw-sourced/",
        "",
        f"Chunks scanned (exact-dup pass, all sources): {total_chunks:,}",
        f"Chunks sampled (near-dup pass): {sampled:,}",
        f"Exact-duplicate groups: {len(exact)}",
        f"Near-duplicate pairs: {len(near)}",
        "",
        "## Exact duplicates",
        "",
    ]
    if not exact:
        lines.append("None found.")
    for h, group in exact.items():
        lines.append(f"- `{h[:12]}` — {len(group)} occurrences:")
        for source, fpath, idx, text in group:
            lines.append(f"  - `{source}` / `{fpath}` chunk {idx}: "
                          f"{text[:80]!r}{'...' if len(text) > 80 else ''}")
    lines += ["", "## Near duplicates", ""]
    if not near:
        lines.append("None found (or near-dup pass was skipped — see the run log).")
    for (a, b, score) in near:
        (sa, pa, ia, ta), (sb, pb, ib, tb) = a, b
        lines.append(f"- {score:.4f} — `{sa}`/`{pa}`#{ia} <-> `{sb}`/`{pb}`#{ib}")
        lines.append(f"  - A: {ta[:80]!r}{'...' if len(ta) > 80 else ''}")
        lines.append(f"  - B: {tb[:80]!r}{'...' if len(tb) > 80 else ''}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _self_test():
    """End-to-end on fake data in a temp dir -- doesn't need the real
    corpus, the real checkpoint, or an NPU. Proves: exact-dup hashing
    finds a real duplicate and ignores near-but-not-identical text; the
    near-dup pipeline's mechanics (embed -> cosine sim -> threshold) work
    by feeding the same chunk through as two different "sources", which
    must score a perfect 1.0 regardless of how good/bad the model is --
    isolates the mechanism from model quality, which an untrained tiny
    model can't be expected to get right.
    """
    import shutil
    import tempfile

    from model.config import PRESETS

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[self-test] SKIPPED -- onnxruntime not installed "
              "(pip install onnxruntime to run this check)")
        return

    from scripts.export_embedder_onnx import _EmbedForExport
    from model.embedding_head import EmbeddingHead, batch_bytes
    from model.transformer import Parentheses
    import torch

    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw-sourced")
        src_a = os.path.join(raw, "source-a")
        src_b = os.path.join(raw, "source-b")
        os.makedirs(src_a)
        os.makedirs(src_b)

        exact_text = "This exact paragraph appears in two different sources on purpose."
        with open(os.path.join(src_a, "shard_00.txt"), "w", encoding="utf-8") as f:
            f.write(exact_text + "\n\nA second, unrelated paragraph about oceans and tides.")
        with open(os.path.join(src_b, "shard_00.txt"), "w", encoding="utf-8") as f:
            f.write(exact_text + "\n\nA third paragraph about rockets and orbital mechanics.")
        # a non-shard file must be ignored
        with open(os.path.join(src_a, "SOURCE.txt"), "w", encoding="utf-8") as f:
            f.write("not corpus text")

        exact, total = find_exact_duplicates(raw)
        assert total == 4, f"expected 4 chunks (2 real paragraphs x 2 sources), got {total}"
        assert len(exact) == 1, f"expected exactly 1 exact-duplicate group, got {len(exact)}"
        (group,) = exact.values()
        assert len(group) == 2 and {g[0] for g in group} == {"source-a", "source-b"}
        assert all(_norm(exact_text) in _norm(g[3]) for g in group)

        # near-dup mechanics, via a tiny untrained model exported to ONNX.
        # Not tiny-smoke: its vocab_size=64 can't hold ordinary ASCII bytes.
        torch.manual_seed(0)
        cfg = PRESETS["parentheses-0.9-300k"]
        backbone = Parentheses(cfg).eval()
        for prm in backbone.parameters():
            prm.requires_grad_(False)
        head = EmbeddingHead(cfg.n_embd, dim=16).eval()
        wrapper = _EmbedForExport(backbone, head)
        idx, mask = batch_bytes(["probe"], cfg.block_size, "cpu")
        onnx_path = os.path.join(td, "tiny.onnx")
        torch.onnx.export(
            wrapper, (idx, mask), onnx_path,
            input_names=["idx", "mask"], output_names=["embedding"],
            dynamic_axes={"idx": {0: "batch", 1: "seq"}, "mask": {0: "batch", 1: "seq"},
                           "embedding": {0: "batch"}},
            opset_version=17,
            dynamo=False,  # avoid the onnxscript-based exporter, see export_embedder_onnx.py
        )

        sample = sample_chunks_per_source(raw, cap=10)
        vectors = embed_chunks(onnx_path, [c for *_, c in sample], cfg.block_size)
        near = find_near_duplicates(vectors, sample, threshold=0.999)
        # the identical-text chunk from source-a and source-b must be the
        # top (or only) near-duplicate pair, at ~1.0 similarity
        assert near, "expected at least one near-duplicate pair (the identical chunk)"
        (a, b, score) = near[0]
        assert score > 0.999, score
        assert {a[0], b[0]} == {"source-a", "source-b"}

        shutil.rmtree(raw, ignore_errors=True)

    print("[self-test] dedup_raw_sourced ok")


if __name__ == "__main__":
    main()
