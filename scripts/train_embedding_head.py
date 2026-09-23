"""
Train the sentence-embedding head (model/embedding_head.py) on parallel pairs,
against a backbone that is frozen by default.

Engineered by uncoalesced

An `<en> src` and its `<kn> tgt` are semantically identical text in two forms,
which is exactly the positive pair an in-batch-negative contrastive objective
wants, and the data is already downloaded and already licensed.

`--raw-dirs` takes any number of `data/raw/parallel/en-*` directories and pools
them; `--max-pairs` is a *per-directory* cap, so pooling stays balanced across
languages instead of being dominated by whichever one loads first.

`--unfreeze-last-block` additionally trains the backbone's top recurrent
block at `--backbone-lr` (default: a tenth of the head's lr). Everything below
that block stays frozen, and the embedding table always stays frozen. This is
the fallback handoff-vector-memory.md named for the case where a fully frozen
backbone's embeddings turn out too weak -- it is not the default.

Evaluation reports three recall@1 numbers, because they measure different
things and only one of them is comparable to the original en-kn baseline:
  * per-language: each language's own held-out pool, scored against only its
    own targets. Hard, honest, directly comparable to the 0.0742 baseline.
  * mean over those per-language numbers -- the headline.
  * pooled/mixed: one pool sampled across all languages. Easier by
    construction (most negatives are in a different script from the true
    target, which a byte-level model can separate without understanding
    anything), so it flatters the model. Reported anyway, labelled as such.

    python3 scripts/train_embedding_head.py --checkpoint checkpoints/multilingual/step_278999_final.pt
    python3 scripts/train_embedding_head.py --raw-dirs data/raw/parallel/en-*
    python3 scripts/train_embedding_head.py --self-test     # no data, no checkpoint
"""

import argparse
import glob
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PRESETS, Parentheses
from model.embedding_head import EmbeddingHead, encode, info_nce

# Non-greedy source side so the split lands on the FIRST " -> <xx> ", not on
# an arrow that happens to occur inside the sentence.
PAIR = re.compile(r"^<(\w+)> (.*?) -> <(\w+)> (.*)$", re.S)


def expand_raw_dirs(raw_dirs: list[str]) -> list[str]:
    """Expand any entry that still contains a glob wildcard into real dirs.

    bash expands `data/raw/parallel/en-*` into separate argv entries before
    this script ever runs; PowerShell does not do that for an external
    command, so the whole pattern arrives as one literal string. Left alone,
    `load_pairs`'s own `glob.glob()` still matches shard files across every
    en-* directory (its `*` isn't anchored to one path segment), but this
    script treated that single string as *one* language bucket -- so
    `--max-pairs`, meant as a per-language cap, became one global cap that
    only the alphabetically-first directory could fill. Confirmed for real:
    a PowerShell run of `--raw-dirs data\\raw\\parallel\\en-*` silently
    trained on en-ar alone (documentation.md, 2026-09-03 testing session).
    Expanding here makes both shells behave the same regardless of which one
    launched python.
    """
    out = []
    for d in raw_dirs:
        if any(c in d for c in "*?["):
            matches = sorted(p for p in glob.glob(d) if os.path.isdir(p))
            if not matches:
                raise SystemExit(f"--raw-dirs {d!r} matched no directories")
            out.extend(matches)
        else:
            out.append(d)
    return out


def load_pairs(raw_dir: str, limit: int = 0) -> list[tuple[str, str]]:
    """Read prepare_parallel.py's shards back into (source, target) tuples."""
    pairs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "shard_*.txt"))):
        with open(path, encoding="utf-8") as f:
            for block in f.read().split("\n\n"):
                m = PAIR.match(block.strip())
                if m:
                    pairs.append((m.group(2), m.group(4)))
        if limit and len(pairs) >= limit:
            return pairs[:limit]
    return pairs


@torch.no_grad()
def recall_at_1(backbone, head, pairs, device: str) -> float:
    """Cross-lingual retrieval accuracy over a held-out pool.

    For every English side, is its own translation the nearest target in the
    pool? Chance is 1/len(pairs), so on a 512-pair pool anything above ~0.2%
    is signal. This is the number that says whether the head learned anything;
    the loss alone can fall just from the temperature.
    """
    src = encode(backbone, head, [a for a, _ in pairs], device)
    tgt = encode(backbone, head, [b for _, b in pairs], device)
    nearest = (src @ tgt.t()).argmax(dim=1)
    return (nearest == torch.arange(len(pairs), device=device)).float().mean().item()


def evaluate(backbone, head, evalsets: dict, device: str, rng: random.Random):
    """-> (per_language {lang: recall}, mean_per_language, pooled_mixed).

    `pooled` deliberately uses a pool the same size as one language's, so its
    chance rate is identical and the two numbers differ only in how hard the
    negatives are -- not in how many there are.
    """
    per = {lang: recall_at_1(backbone, head, pool, device) for lang, pool in evalsets.items()}
    mean = sum(per.values()) / len(per)
    size = len(next(iter(evalsets.values())))
    flat = [pr for pool in evalsets.values() for pr in pool]
    pooled = recall_at_1(backbone, head, rng.sample(flat, min(size, len(flat))), device)
    return per, mean, pooled


def trainable_backbone_params(backbone, unfreeze_last_block: bool) -> list:
    """Flip requires_grad on exactly what should train. -> the params to optimize.

    Only the top Block. Not ln_f (a 72-element gain shared with text
    generation), not the tied embedding table -- that table is also the LM
    output head, so a gradient into it is the one edit that would visibly
    change what the checkpoint generates, which is the whole thing the frozen
    design was protecting.
    """
    for prm in backbone.parameters():
        prm.requires_grad_(False)
    if not unfreeze_last_block:
        return []
    for prm in backbone.blocks[-1].parameters():
        prm.requires_grad_(True)
    return [prm for prm in backbone.blocks[-1].parameters()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/all-sources-v1/step_560999_final.pt",
                   help="backbone checkpoint (all-sources-v1, the current primary, by default; "
                        "older checkpoints are still usable via this flag explicitly)")
    p.add_argument("--raw-dirs", nargs="+", default=["data/raw/parallel/en-kn"],
                   help="one or more data/raw/parallel/en-* dirs; pairs from all of them are "
                        "pooled. A single unexpanded glob string (e.g. from PowerShell) is "
                        "expanded internally, so this works the same from any shell.")
    p.add_argument("--out", default="checkpoints/all-sources-v1/embedding_head_all_sources_v1.pt")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256,
                   help="also the number of in-batch negatives; bigger is a harder task")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-last-block", action="store_true",
                   help="also train the backbone's top recurrent block (everything below stays frozen)")
    p.add_argument("--backbone-lr", type=float, default=None,
                   help="lr for the unfrozen block; default lr/10")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--eval-pairs", type=int, default=512,
                   help="held out PER language, so every language gets its own same-size pool")
    p.add_argument("--max-pairs", type=int, default=100_000, help="per raw-dir cap")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    args.raw_dirs = expand_raw_dirs(args.raw_dirs)

    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    backbone = Parentheses(ckpt["cfg"]).to(args.device).eval()
    backbone.load_state_dict(ckpt["model"])
    tune = trainable_backbone_params(backbone, args.unfreeze_last_block)

    head = EmbeddingHead(ckpt["cfg"].n_embd, args.dim).to(args.device)
    groups = [{"params": list(head.parameters()), "lr": args.lr}]
    if tune:
        blr = args.backbone_lr if args.backbone_lr is not None else args.lr / 10
        groups.append({"params": tune, "lr": blr})
    opt = torch.optim.AdamW(groups)
    n_tune = sum(t.numel() for t in tune)
    print(f"[info] head {head.num_params():,} trainable params, dim={args.dim}, "
          f"backbone {backbone.num_params():,} ({n_tune:,} of it trainable"
          + (f" @ lr {groups[-1]['lr']:g}" if tune else "") + ")")

    # Per language, so a held-out pool is never contaminated by another
    # language's train split and every pool is the same size (= same chance).
    train, evalsets = [], {}
    for raw_dir in args.raw_dirs:
        lang = os.path.basename(os.path.normpath(raw_dir))
        pairs = load_pairs(raw_dir, args.max_pairs)
        rng.shuffle(pairs)
        if len(pairs) <= args.eval_pairs:
            raise SystemExit(f"{raw_dir}: only {len(pairs)} pairs, need > {args.eval_pairs}")
        evalsets[lang] = pairs[:args.eval_pairs]
        train.extend(pairs[args.eval_pairs:])
    rng.shuffle(train)
    print(f"[info] {len(train):,} training pairs pooled from {len(evalsets)} dir(s), "
          f"{args.eval_pairs} held out each")
    if len(train) < args.batch_size:
        raise SystemExit(f"only {len(train)} training pairs, need >= {args.batch_size}")

    chance = 1 / args.eval_pairs
    per0, mean0, pooled0 = evaluate(backbone, head, evalsets, args.device, rng)
    print(f"[eval] before: mean per-language recall@1 {mean0:.4f} | "
          f"pooled/mixed {pooled0:.4f} | chance {chance:.4f}")

    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = rng.sample(train, args.batch_size)
        a = encode(backbone, head, [x for x, _ in batch], args.device, grad=True)
        b = encode(backbone, head, [y for _, y in batch], args.device, grad=True)
        loss = info_nce(a, b, args.temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.4f} | {time.time() - t0:.1f}s", flush=True)

    per, mean, pooled = evaluate(backbone, head, evalsets, args.device, rng)
    print(f"[eval] after:  mean per-language recall@1 {mean:.4f} (was {mean0:.4f}) | "
          f"pooled/mixed {pooled:.4f} (was {pooled0:.4f}) | chance {chance:.4f}")
    for lang in sorted(per):
        print(f"       {lang:<8} {per[lang]:.4f}  (was {per0[lang]:.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"head": head.state_dict(), "dim": args.dim, "n_embd": ckpt["cfg"].n_embd,
                "backbone": args.checkpoint, "raw_dirs": args.raw_dirs,
                "unfreeze_last_block": args.unfreeze_last_block,
                # The unfrozen block only means something paired with the head
                # it was trained beside, so it travels in the same file. None
                # when nothing was unfrozen -- loaders check for that.
                "backbone_last_block": (backbone.blocks[-1].state_dict() if tune else None),
                "recall_at_1": mean, "recall_at_1_per_lang": per,
                "recall_at_1_pooled_mixed": pooled,
                "loss_curve": losses, "steps": args.steps}, args.out)
    print(f"[done] saved {args.out}")


def _self_test():
    import tempfile

    # expand_raw_dirs: a literal directory passes through untouched (the
    # bash-already-expanded case); an unexpanded glob string (the PowerShell
    # case) expands to every real matching directory, not one fake bucket.
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("en-aa", "en-bb", "en-cc"):
            os.makedirs(os.path.join(tmp, name))
        assert expand_raw_dirs([os.path.join(tmp, "en-aa")]) == [os.path.join(tmp, "en-aa")]
        got = expand_raw_dirs([os.path.join(tmp, "en-*")])
        assert got == sorted(os.path.join(tmp, n) for n in ("en-aa", "en-bb", "en-cc")), got
        try:
            expand_raw_dirs([os.path.join(tmp, "nope-*")])
            assert False, "expand_raw_dirs accepted a glob with no matches"
        except SystemExit:
            pass

    # shard parsing round-trips prepare_parallel.py's format, including a
    # source sentence that itself contains an arrow
    with tempfile.TemporaryDirectory() as tmp:
        blocks = ["<en> The sky is blue. -> <kn> ಆಕಾಶ ನೀಲಿ.",
                  "<en> A -> B notation. -> <kn> ಎ ಬಿ.",
                  "not a pair at all"]
        with open(os.path.join(tmp, "shard_00000.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(blocks))
        got = load_pairs(tmp)
        assert len(got) == 2, got
        assert got[0] == ("The sky is blue.", "ಆಕಾಶ ನೀಲಿ."), got[0]
        assert got[1] == ("A -> B notation.", "ಎ ಬಿ."), got[1]
        assert len(load_pairs(tmp, limit=1)) == 1

    torch.manual_seed(0)
    cfg = PRESETS["parentheses-0.9-300k"]
    backbone = Parentheses(cfg).eval()
    for prm in backbone.parameters():          # what main() does; encode() honours it
        prm.requires_grad_(False)
    head = EmbeddingHead(cfg.n_embd, dim=64)

    pairs = [(f"english sentence number {i}", f"target sentence number {i}") for i in range(16)]
    r = recall_at_1(backbone, head, pairs, "cpu")
    assert 0.0 <= r <= 1.0, r

    # the head must actually learn: overfit these 16 pairs and recall must rise
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(60):
        a = encode(backbone, head, [x for x, _ in pairs], "cpu", grad=True)
        b = encode(backbone, head, [y for _, y in pairs], "cpu", grad=True)
        loss = info_nce(a, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert recall_at_1(backbone, head, pairs, "cpu") >= r, "head failed to fit 16 pairs"

    # and the backbone must not have moved: no gradient reached it, and its
    # weights are bit-identical after a backward pass through the head
    before = backbone.tok_emb.weight.clone()
    a = encode(backbone, head, ["x"], "cpu", grad=True)
    a.sum().backward()
    assert backbone.tok_emb.weight.grad is None, "backbone received gradients"
    assert torch.equal(backbone.tok_emb.weight, before)

    # evaluate() splits the score three ways over per-language pools
    sets = {"en-kn": pairs[:8], "en-hi": pairs[8:]}
    per, mean, pooled = evaluate(backbone, head, sets, "cpu", random.Random(0))
    assert set(per) == {"en-kn", "en-hi"}, per
    assert abs(mean - sum(per.values()) / 2) < 1e-9, (mean, per)
    assert 0.0 <= pooled <= 1.0, pooled

    # trainable_backbone_params: default freezes everything and hands back
    # nothing to optimize...
    assert trainable_backbone_params(backbone, False) == []
    assert not any(prm.requires_grad for prm in backbone.parameters())

    # ...and --unfreeze-last-block frees exactly the top block, nothing else,
    # and above all not the tied embedding table that is also the LM head.
    tune = trainable_backbone_params(backbone, True)
    assert len(tune) == len(list(backbone.blocks[-1].parameters())) and tune
    assert all(prm.requires_grad for prm in backbone.blocks[-1].parameters())
    assert not any(prm.requires_grad for prm in backbone.blocks[0].parameters())
    assert not backbone.tok_emb.weight.requires_grad
    assert not backbone.ln_f.weight.requires_grad

    # a real optimizer step must move the unfrozen block and leave the rest
    # bit-identical -- freezing that only holds inside no_grad is not freezing
    b0 = backbone.blocks[0].attn.qkv.weight.clone()
    bl = backbone.blocks[-1].attn.qkv.weight.clone()
    emb = backbone.tok_emb.weight.clone()
    opt = torch.optim.AdamW([{"params": list(head.parameters()), "lr": 1e-3},
                             {"params": tune, "lr": 1e-4}])
    a = encode(backbone, head, [x for x, _ in pairs], "cpu", grad=True)
    b = encode(backbone, head, [y for _, y in pairs], "cpu", grad=True)
    info_nce(a, b).backward()
    opt.step()
    assert not torch.equal(backbone.blocks[-1].attn.qkv.weight, bl), "top block did not train"
    assert torch.equal(backbone.blocks[0].attn.qkv.weight, b0), "a frozen block moved"
    assert torch.equal(backbone.tok_emb.weight, emb), "the embedding table moved"

    print("[self-test] train_embedding_head ok")


if __name__ == "__main__":
    main()
