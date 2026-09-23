"""
Sentence-embedding head for Parentheses -- the piece that turns text into a
vector so TurboVec has something to index (see handoff-vector-memory.md).

Engineered by uncoalesced

Beside Parentheses, not a fork of it: this module owns only the pooling and
projection, and calls the backbone's `hidden()` for everything else. Same
composition shape features/modular_free_think.py used over free_think.py.

Design, straight from the handoff:
  * mean-pool the final layer's hidden states over byte positions (the model
    has no EOS/pad token, so there is no last-token trick available anyway)
  * one hidden linear + GELU + one projection down to `dim`
  * L2-normalise, so cosine similarity is a plain inner product -- which is
    what TurboVec's search returns
  * the backbone stays frozen by default; only this head trains. `encode()`
    honours `requires_grad` on backbone parameters rather than hard-coding
    no_grad, so training the top block alongside the head is possible
    without forking this module (see scripts/train_embedding_head.py
    --unfreeze-last-block).

Embedding dimension defaults to 64. The pooled state is n_embd-wide (72 on
-300k), so anything above 72 is fabricated rank, not capacity -- 64 is the
largest power of two at or under that ceiling, keeps a mild bottleneck for the
contrastive objective to exploit, and lands on a clean 32-byte code at
TurboVec's default 4-bit width.

    python3 -m model.embedding_head      # self-test, no data or network
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Byte 0 never occurs in valid UTF-8 text, so it is a safe pad id; the pooling
# mask excludes those positions regardless.
PAD_ID = 0


class EmbeddingHead(nn.Module):
    """Mean-pool + project the backbone's hidden states to a unit vector."""

    def __init__(self, n_embd: int, dim: int = 64):
        super().__init__()
        self.n_embd = n_embd
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, dim),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hidden (B, T, n_embd) + mask (B, T) bool -> (B, dim) unit vectors."""
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.net(pooled), dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def batch_bytes(texts: list[str], block_size: int, device: str):
    """UTF-8 encode, truncate to block_size, right-pad. -> (idx, mask)."""
    seqs = [list(t.encode("utf-8"))[:block_size] or [PAD_ID] for t in texts]
    width = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        idx[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = True
    return idx.to(device), mask.to(device)


def encode(backbone, head: EmbeddingHead, texts: list[str], device: str,
           batch_size: int = 64, grad: bool = False) -> torch.Tensor:
    """Embed `texts` -> (N, dim) unit vectors.

    The backbone is frozen by default and stays that way unless something has
    explicitly flipped `requires_grad` on some of its parameters -- there is no
    "unfreeze" flag here on purpose. Whoever wants to fine-tune the top block
    sets `requires_grad_(True)` on exactly those parameters and passes them to
    an optimizer; this function then stops wrapping the backbone pass in
    no_grad so a graph actually reaches them. Everything the caller left frozen
    still costs no activation memory, because autograd only records what leads
    to a leaf that wants a gradient.
    """
    tune = grad and any(p.requires_grad for p in backbone.parameters())
    outs = []
    for i in range(0, len(texts), batch_size):
        idx, mask = batch_bytes(texts[i:i + batch_size], backbone.cfg.block_size, device)
        with torch.set_grad_enabled(tune):
            hidden = backbone.hidden(idx)
        with torch.set_grad_enabled(grad):
            outs.append(head(hidden, mask))
    return torch.cat(outs) if outs else torch.zeros((0, head.dim), device=device)


def load_trained(checkpoint: str, head_path: str, device: str):
    """-> (backbone.eval(), head.eval(), head_meta). Frozen, ready to embed.

    Loads the checkpoint, then overlays the head file's `backbone_last_block`
    if it has one: a head trained with --unfreeze-last-block is only correct
    beside the block it was trained against, so the two always load together.
    Kept here rather than in either script so the benchmark and the
    side-by-side cannot drift apart on how a head is restored.

    Refuses a mismatched pair. A head file records which backbone it trained
    against; every parentheses-0.9-300k checkpoint shares n_embd=72, so a
    mismatched pair loads with no shape error and produces plausible-looking,
    meaningless vectors -- exactly what happened once already in
    data/FOLD_SOURCES_REPORT.md's retrieval comparison. This used to be
    checked only in scripts/export_embedder_onnx.py; moved here so every
    caller of the shared loader gets it, not just the one that remembered to.
    """
    from .backbone import Parentheses

    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    backbone = Parentheses(ck["cfg"]).to(device).eval()
    backbone.load_state_dict(ck["model"])

    meta = torch.load(head_path, map_location=device, weights_only=False)
    trained_against = meta.get("backbone")
    if trained_against and os.path.normpath(trained_against) != os.path.normpath(checkpoint):
        raise ValueError(
            f"{head_path} was trained against {trained_against}, not {checkpoint}. "
            f"A head is only meaningful beside the exact backbone it trained against "
            f"-- pair them, or train a head for this backbone.")
    if meta.get("backbone_last_block"):
        backbone.blocks[-1].load_state_dict(meta["backbone_last_block"])
    for prm in backbone.parameters():
        prm.requires_grad_(False)

    head = EmbeddingHead(meta["n_embd"], meta["dim"]).to(device).eval()
    head.load_state_dict(meta["head"])
    return backbone, head, meta


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Symmetric in-batch-negative contrastive loss over aligned pairs.

    Row i of `a` and row i of `b` are a translation pair; every other row in
    the batch is a negative. Symmetric because retrieval runs both ways -- an
    English query should find its Kannada chunk and the reverse.
    """
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def self_test():
    from .config import PRESETS
    from .backbone import Parentheses

    torch.manual_seed(0)
    cfg = PRESETS["parentheses-0.9-300k"]
    backbone = Parentheses(cfg).eval()
    # Freezing is the caller's job now that encode() honours requires_grad --
    # this is the same two lines every real caller runs (see
    # scripts/train_embedding_head.py, scripts/benchmark_retrieval.py).
    for prm in backbone.parameters():
        prm.requires_grad_(False)
    head = EmbeddingHead(cfg.n_embd, dim=64)

    # shape + unit norm
    v = encode(backbone, head, ["hello world", "a much longer sentence here"], "cpu")
    assert v.shape == (2, 64), v.shape
    assert torch.allclose(v.norm(dim=-1), torch.ones(2), atol=1e-5), v.norm(dim=-1)

    # padding must not change an embedding: the short text batched next to a
    # long one has to match the same text encoded alone.
    alone = encode(backbone, head, ["hi there"], "cpu")
    padded = encode(backbone, head, ["hi there", "x" * 200], "cpu")[:1]
    assert torch.allclose(alone, padded, atol=1e-5), (alone - padded).abs().max()

    # mask really is what does that -- an all-True mask over padding differs
    idx, mask = batch_bytes(["hi there", "x" * 200], cfg.block_size, "cpu")
    hidden = backbone.hidden(idx)
    unmasked = head(hidden, torch.ones_like(mask))[:1]
    assert not torch.allclose(alone, unmasked, atol=1e-4)

    # truncation at block_size, not a crash
    long_idx, _ = batch_bytes(["y" * (cfg.block_size * 3)], cfg.block_size, "cpu")
    assert long_idx.shape[1] == cfg.block_size, long_idx.shape

    # An empty string collapses to a single pad position that IS attended, so
    # pooling divides by 1 and returns a finite vector instead of 0/0 = NaN.
    e_idx, e_mask = batch_bytes([""], cfg.block_size, "cpu")
    assert e_idx.shape == (1, 1) and e_mask.all()
    assert torch.isfinite(encode(backbone, head, [""], "cpu")).all()

    # contrastive loss: aligned pairs must score better than shuffled ones
    a = F.normalize(torch.randn(8, 64), dim=-1)
    b = F.normalize(a + 0.01 * torch.randn(8, 64), dim=-1)      # near-perfect matches
    aligned = info_nce(a, b)
    shuffled = info_nce(a, b[torch.randperm(8, generator=torch.manual_seed(1))])
    assert aligned < shuffled, (aligned.item(), shuffled.item())
    assert aligned.item() < 0.1, aligned.item()

    # frozen by default: a backward pass through encode() leaves the backbone
    # gradient-free, even with grad=True (that flag is about the head).
    backbone.zero_grad(set_to_none=True)
    encode(backbone, head, ["x"], "cpu", grad=True).sum().backward()
    assert all(p.grad is None for p in backbone.parameters()), "frozen backbone got gradients"

    # unfreezing is done by flipping requires_grad, and then encode() has to
    # actually build a graph through the backbone -- only for what was unfrozen.
    for p in backbone.blocks[-1].parameters():
        p.requires_grad_(True)
    backbone.zero_grad(set_to_none=True)
    encode(backbone, head, ["x"], "cpu", grad=True).sum().backward()
    assert backbone.blocks[-1].attn.qkv.weight.grad is not None, "last block got no gradient"
    assert backbone.blocks[0].attn.qkv.weight.grad is None, "a frozen block got a gradient"
    assert backbone.tok_emb.weight.grad is None, "the embedding table got a gradient"
    for p in backbone.blocks[-1].parameters():
        p.requires_grad_(False)
    backbone.zero_grad(set_to_none=True)

    # head is small next to the backbone it rides on
    assert head.num_params() < backbone.num_params() // 10, (head.num_params(), backbone.num_params())

    # load_trained() refuses a head trained against a different checkpoint,
    # and loads clean when the backbone path matches.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ck_path = os.path.join(tmp, "backbone.pt")
        torch.save({"cfg": cfg, "model": backbone.state_dict()}, ck_path)
        head_path = os.path.join(tmp, "head.pt")
        torch.save({"head": head.state_dict(), "dim": head.dim, "n_embd": head.n_embd,
                    "backbone": ck_path, "backbone_last_block": None}, head_path)

        b2, h2, m2 = load_trained(ck_path, head_path, "cpu")
        assert m2["backbone"] == ck_path

        other_path = os.path.join(tmp, "other.pt")
        torch.save({"cfg": cfg, "model": backbone.state_dict()}, other_path)
        try:
            load_trained(other_path, head_path, "cpu")
            assert False, "load_trained accepted a mismatched backbone/head pair"
        except ValueError as e:
            assert "was trained against" in str(e), e

    print(f"[self-test] embedding_head ok (dim={head.dim}, {head.num_params():,} head params "
          f"on a {backbone.num_params():,}-param backbone)")


if __name__ == "__main__":
    self_test()
