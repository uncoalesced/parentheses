"""
Qualitative retrieval comparison: what BM25 and the vector head actually
return for the same query over the same store.

The recall@1 number in scripts/train_embedding_head.py is measured on parallel
sentence pairs. This is the other half -- monolingual English paragraphs, the
use case Modular Free Think is actually for, which handoff-vector-memory.md
flags as an assumption to verify rather than assume transfers. Same corpus,
same queries, same truncation every run, so two heads are compared on the
identical page rather than on remembered impressions.

    python3 scripts/retrieval_sidebyside.py --head checkpoints/multilingual/embedding_head_kn.pt
    python3 scripts/retrieval_sidebyside.py --self-test
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.embedding_head import EmbeddingHead, encode, load_trained
from features.modular_free_think import IngestedStore
from scripts.benchmark_retrieval import corpus_file, make_embedder

# Frozen so successive runs are diffable. The middle one is the bar: on the
# 2026-08-31 en-kn head the vector backend answered it with a Pakistani
# politician, a Spanish municipality and a Tallinn memorial.
QUERIES = ["The deep ocean is cold and dark.",
           "Music and musical instruments.",
           "A war fought in Europe."]
WIDTH = 110         # chars of each chunk shown; matches the first run's file


def render(store_by_name: dict, queries, top_k: int, width: int) -> list[str]:
    out = []
    for q in queries:
        out.append(f"QUERY: {q}")
        for name, store in store_by_name.items():
            out.append(f"  {name}:")
            hits = store.retrieve(q, top_k)
            if not hits:
                out.append("    (nothing retrieved)")
            for chunk in hits:
                out.append("    - " + " ".join(chunk.split())[:width])
        out.append("")
    return out


def diagnostics(vectors: np.ndarray, docs: list[str], n_probe: int, seed: int,
                queries: np.ndarray | None = None) -> list[str]:
    """Two label-free health checks on the embedding space itself.

    The recall@1 in train_embedding_head.py is measured on parallel sentence
    pairs, where every candidate is a sentence of roughly the same length. A
    real store is not like that -- Wikipedia chunks run from a one-line
    category stub to a full paragraph -- and an embedder can score well on the
    first while being useless on the second. These catch that:

      * self-retrieval: embed a chunk, search the store it is already in; the
        nearest vector should be itself, since it is an exact duplicate. Any
        embedder that does not hit ~100% here is not separating documents at
        all, whatever its pair recall says.
      * hubness: how many DISTINCT chunks ever come back as a top-1 across
        many probes. A healthy space returns a different neighbour per probe;
        a collapsed one funnels everything into a few "hub" vectors that sit
        near the middle of every query. The top hub's share is the headline.
      * length bias: how strongly a chunk's similarity to the real queries
        tracks its character length. This is the one that caught the
        2026-08-31 unfrozen-last-block head: it scored 2.1x the frozen head's
        pair recall while correlating -0.373 with length, i.e. preferring the
        store's shortest fragments, because the queries are short and it had
        learned length as a dimension. The training pairs are sentence-to-
        sentence and symmetric in length, so nothing in the objective punishes
        that; short-query/long-document retrieval does. Near zero is what a
        length-neutral embedder looks like.
    """
    rng = np.random.default_rng(seed)
    probe = rng.choice(len(docs), size=min(n_probe, len(docs)), replace=False)
    sims = vectors[probe] @ vectors.T                    # unit vectors -> cosine
    top1 = sims.argmax(axis=1)
    self_hit = float((top1 == probe).mean())

    # Hubness has to be measured on someone ELSE's nearest neighbour, or exact
    # self-matches would hide the collapse completely.
    sims[np.arange(len(probe)), probe] = -np.inf
    nn = sims.argmax(axis=1)
    uniq, counts = np.unique(nn, return_counts=True)
    top_share = float(counts.max() / len(probe))
    out = [
        f"self-retrieval : {self_hit:.3f} of {len(probe)} probes find themselves (1.000 = healthy)",
        f"hubness        : {len(uniq)} distinct nearest neighbours over {len(probe)} probes; "
        f"the single most-returned chunk takes {top_share:.1%} (lower is healthier)",
    ]
    if queries is not None and len(queries):
        lens = np.array([len(d) for d in docs], dtype=np.float64)
        qsim = (queries @ vectors.T).mean(axis=0)
        r = float(np.corrcoef(lens, qsim)[0, 1])
        top3 = np.argsort(-(queries @ vectors.T), axis=1)[:, :3]
        out.append(f"length bias    : corr(chunk length, similarity to query) {r:+.3f} "
                   f"(0 = neutral); top-3 retrieved average {lens[top3].mean():.0f} chars "
                   f"against a store median of {np.median(lens):.0f}")
    return out


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
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--probes", type=int, default=500, help="chunks probed for the diagnostics block")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples/retrieval_sidebyside.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    backbone, head, meta = load_trained(args.checkpoint, args.head, args.device)
    embedder = make_embedder(backbone, head, args.device)

    with tempfile.TemporaryDirectory() as tmp:
        cpath = os.path.join(tmp, "corpus.txt")
        nbytes = corpus_file(args.raw_dir, cpath, args.corpus_bytes)
        paths = [Path(cpath)]
        bm25 = IngestedStore(backend="bm25")
        bm25.ingest(paths)
        vec = IngestedStore(backend="vector", embedder=embedder)
        vec.ingest(paths)

    n = len(bm25.documents)
    recall = meta.get("recall_at_1")
    header = [
        f"Retrieval side-by-side over the same {n:,}-chunk store from {args.raw_dir}.",
        f"bm25 = lexical (rank_bm25). vector = {meta['dim']}-d embedding head + TurboVec.",
        f"backbone : {args.checkpoint}",
        f"head     : {args.head}",
        f"           trained on {len(meta.get('raw_dirs', [meta.get('raw_dir', '?')]))} language dir(s), "
        f"last block {'UNFROZEN' if meta.get('unfreeze_last_block') else 'frozen'}, "
        f"mean per-language recall@1 {recall:.4f}" if recall is not None else "",
        "",
    ]
    lines = ([h for h in header if h != ""]
             + diagnostics(vec._vectors, vec.documents, args.probes, args.seed,
                           np.ascontiguousarray(embedder(QUERIES), dtype=np.float32))
             + [""] + render({"bm25": bm25, "vector": vec}, QUERIES, args.top_k, WIDTH))
    text = "\n".join(lines).rstrip() + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"[done] wrote {args.out}")


def _self_test():
    from model import PRESETS
    from model.transformer import Parentheses

    torch.manual_seed(0)
    cfg = PRESETS["parentheses-0.9-100k"]
    backbone = Parentheses(cfg).eval()
    head = EmbeddingHead(cfg.n_embd, dim=32).eval()

    # load_trained round-trips a head file, last-block overlay included, and
    # comes back frozen -- the property every consumer of it assumes.
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, "ck.pt")
        hp = os.path.join(tmp, "head.pt")
        torch.save({"model": backbone.state_dict(), "cfg": cfg, "step": 1}, ck)

        torch.save({"head": head.state_dict(), "dim": 32, "n_embd": cfg.n_embd,
                    "backbone_last_block": None}, hp)
        b1, h1, m1 = load_trained(ck, hp, "cpu")
        assert not any(p.requires_grad for p in b1.parameters())
        assert torch.equal(b1.blocks[-1].attn.qkv.weight, backbone.blocks[-1].attn.qkv.weight)
        assert torch.allclose(encode(b1, h1, ["abc"], "cpu"), encode(backbone, head, ["abc"], "cpu"))

        tweaked = {k: v + 1.0 for k, v in backbone.blocks[-1].state_dict().items()}
        torch.save({"head": head.state_dict(), "dim": 32, "n_embd": cfg.n_embd,
                    "backbone_last_block": tweaked}, hp)
        b2, _, _ = load_trained(ck, hp, "cpu")
        assert not torch.equal(b2.blocks[-1].attn.qkv.weight, backbone.blocks[-1].attn.qkv.weight), \
            "the unfrozen block was not restored"
        assert torch.equal(b2.blocks[0].attn.qkv.weight, backbone.blocks[0].attn.qkv.weight)

    # render() truncates, collapses whitespace, and survives an empty result
    class FakeStore:
        def __init__(self, hits): self.hits = hits
        def retrieve(self, q, k): return self.hits[:k]

    long = "w " * 400
    out = render({"a": FakeStore([long, "two\n\nlines"]), "b": FakeStore([])},
                 ["q"], 2, WIDTH)
    assert out[0] == "QUERY: q"
    body = [l for l in out if l.startswith("    - ")]
    assert all(len(l) <= 6 + WIDTH for l in body), max(len(l) for l in body)
    assert "    - two lines" in out
    assert "    (nothing retrieved)" in out

    # diagnostics: a perfectly separated space scores 1.0 self-retrieval with
    # no dominant hub; a collapsed one is caught on both counts.
    good = np.eye(6, dtype=np.float32)
    d = diagnostics(good, [str(i) for i in range(6)], 6, 0)
    assert "1.000 of 6" in d[0], d[0]

    # every vector but one identical -> everything self-matches by tie-break at
    # best, and the odd one out is every single probe's nearest neighbour.
    collapsed = np.zeros((6, 3), dtype=np.float32)
    collapsed[:, 0] = 1.0
    collapsed[5] = [0.9987, 0.05, 0.0]
    d = diagnostics(collapsed, [str(i) for i in range(6)], 6, 0)
    assert "takes 83.3%" in d[1] or "takes 100.0%" in d[1], d[1]

    # length bias: build a space where similarity to the query IS length, and
    # one where it is the opposite, and check the sign comes out right.
    docs = ["x" * (10 * (i + 1)) for i in range(6)]
    ramp = np.stack([[np.cos(t), np.sin(t)] for t in np.linspace(0, 1.2, 6)]).astype(np.float32)
    q = ramp[-1:].copy()                       # closest to the longest doc
    d = diagnostics(ramp, docs, 6, 0, q)
    assert len(d) == 3 and "length bias" in d[2], d
    assert d[2].split("query) ")[1].startswith("+"), d[2]
    d = diagnostics(ramp, docs[::-1], 6, 0, q)                 # lengths reversed
    assert d[2].split("query) ")[1].startswith("-"), d[2]

    print("[self-test] retrieval_sidebyside ok")


if __name__ == "__main__":
    main()
