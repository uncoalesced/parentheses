"""
Training entrypoint for Parentheses 0.9.

Usage:
    python3 train.py --preset parentheses-0.9-150m --data data/processed/train.bin

This is intentionally a plain, readable training loop (no Trainer
abstraction) so it's easy to modify as you experiment. It supports the
memory/throughput techniques discussed in docs/training-time-estimate.md:
mixed precision, gradient accumulation, gradient checkpointing, and an
optional 8-bit optimizer.
"""

import argparse
import os
import time

import numpy as np
import torch

from model import ModelConfig, PRESETS, Parentheses


def supervised_starts(mask: np.memmap, hi: int, block_size: int, batch_size: int,
                      rounds: int = 100) -> list[int]:
    """Sample batch_size window starts whose *target* window has >=1 supervised byte.

    An SFT corpus is mostly prompt: Hermes' tool-schema system prompts alone
    run past block_size, so a uniformly drawn window is often entirely masked
    out. A whole micro-batch of those makes cross_entropy average over zero
    elements -> nan loss -> nan weights, silently, a few hundred steps in.

    ponytail: rejection sampling. Ceiling is a corpus where supervised bytes
    are so sparse that most draws miss; the upgrade path is a precomputed
    index of eligible starts (one pass, 8 bytes each), which only pays for
    itself if this starts showing up in the step time.
    """
    keep = []
    for _ in range(rounds):
        for i in torch.randint(hi, (batch_size,)).tolist():
            if mask[i + 1:i + 1 + block_size].any():
                keep.append(i)
                if len(keep) == batch_size:
                    return keep
    raise RuntimeError(
        f"no supervised targets found in {rounds * batch_size} sampled windows -- is "
        f"--sft-spans pointing at the mask for a different corpus than --data?")


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str,
              mask: np.memmap | None = None):
    hi = len(data) - block_size - 1
    ix = (torch.randint(hi, (batch_size,)) if mask is None
          else torch.tensor(supervised_starts(mask, hi, block_size, batch_size)))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if mask is not None:
        # -1 is Parentheses.forward's ignore_index, so masked positions cost
        # nothing and contribute no gradient: the model is scored on producing
        # the assistant's turn, not on reciting the prompt back.
        keep = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + block_size].astype(bool)) for i in ix])
        y = y.masked_fill(~keep, -1)
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, use_8bit: bool):
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("[warn] bitsandbytes not installed, falling back to torch.optim.AdamW")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="parentheses-0.9-300k", choices=list(PRESETS.keys()))
    p.add_argument("--data", default="data/processed/train.bin", help="path to tokenized uint16 .bin file")
    p.add_argument("--sft-spans", default=None,
                   help="SFT stage: uint8 loss mask written by data/prepare_sft.py --pack "
                        "(e.g. data/processed_sft/train.mask.bin), same length as --data. "
                        "Loss is scored only on the assistant-turn bytes it marks.")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--resume-from", default=None,
                   help="checkpoint .pt to resume from (for staged/batched runs). "
                        "--max-steps is the absolute target step, not an additional count.")
    p.add_argument("--batch-size", type=int, default=64, help="micro-batch size (per grad-accum step) -- "
                    "VRAM is not a constraint at this param scale (see docs/training-time-estimate.md), "
                    "so this defaults much higher than it would for the 50M+ presets")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--optimizer-8bit", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true",
                   help="check batch sampling / SFT loss masking on fake data and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    cfg: ModelConfig = PRESETS[args.preset]
    model = Parentheses(cfg).to(args.device)
    print(f"[info] {args.preset}: {model.num_params()/1e6:.1f}M params on {args.device}")

    if args.grad_checkpointing:
        # Simple manual checkpointing hook point; wire into Block.forward via
        # torch.utils.checkpoint if/when VRAM pressure requires it.
        print("[info] gradient checkpointing requested (enable per-block checkpoint() calls as needed)")

    optimizer = build_optimizer(model, args.lr, args.weight_decay, args.optimizer_8bit)

    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler(args.device, enabled=use_amp)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"No tokenized data at {args.data}. Run data/prepare_wikipedia.py and "
            f"data/prepare_books.py first (see data/README.md)."
        )
    data = np.memmap(args.data, dtype=np.uint16, mode="r")

    mask = None
    if args.sft_spans:
        mask = np.memmap(args.sft_spans, dtype=np.uint8, mode="r")
        if len(mask) != len(data):
            raise ValueError(
                f"{args.sft_spans} is {len(mask):,} bytes, {args.data} is {len(data):,} tokens. "
                f"A mask that is not element-for-element aligned with its data supervises the "
                f"wrong bytes; re-run data/prepare_sft.py --pack to write both together.")
        print(f"[info] SFT loss masking on: {int(np.count_nonzero(mask)):,} of {len(mask):,} "
              f"bytes supervised ({np.count_nonzero(mask)/len(mask):.1%})")

    os.makedirs(args.out_dir, exist_ok=True)

    start_step = 0
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume-from checkpoint not found: {args.resume_from}")
        # weights_only=False: the checkpoint pickles the ModelConfig dataclass
        # (features/free_think.py carries the same note). Torch 2.6+ defaults this
        # to True, which rejects the load outright -- --resume-from could not load
        # any checkpoint this repo has ever written.
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        if ckpt["cfg"] != cfg:
            raise ValueError(
                f"--resume-from checkpoint was trained with cfg {ckpt['cfg']!r}, but "
                f"--preset {args.preset!r} is {cfg!r}. Resuming into a different "
                f"architecture would silently produce garbage.")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(f"[warn] {args.resume_from} has no saved optimizer state (older checkpoint "
                  f"format) -- resuming with a fresh optimizer, momentum is lost")
        start_step = ckpt["step"] + 1
        print(f"[info] resumed from {args.resume_from} at step {ckpt['step']}, "
              f"continuing to step {args.max_steps - 1}")
        if start_step >= args.max_steps:
            raise ValueError(f"--max-steps {args.max_steps} is not past the resumed step "
                              f"{ckpt['step']}; nothing to do")

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(data, cfg.block_size, args.batch_size, args.device, mask)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {loss_accum:.4f} | {dt:.1f}s elapsed")

        if step > 0 and step % args.ckpt_every == 0:
            path = os.path.join(args.out_dir, f"step_{step}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "cfg": cfg, "step": step}, path)
            print(f"[info] saved checkpoint: {path}")

    # final save -- the loop above only checkpoints on a step%ckpt_every==0
    # step *inside* range(max_steps), so whenever max_steps isn't hit by
    # that stride (e.g. range(162000) stops at step 161999) the last
    # <ckpt_every steps of training were never written to disk.
    final_step = args.max_steps - 1
    path = os.path.join(args.out_dir, f"step_{final_step}_final.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": cfg, "step": final_step}, path)
    print(f"[info] saved final checkpoint: {path}")


def _self_test():
    torch.manual_seed(0)
    block, batch = 8, 4
    data = np.arange(200, dtype=np.uint16) % 256

    # no mask: unchanged behaviour, y is x shifted by one
    x, y = get_batch(data, block, batch, "cpu")
    assert x.shape == y.shape == (batch, block)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert (y == -1).sum() == 0

    # mask: supervision is exactly the marked target positions, everything
    # else is ignore_index, and a real model's loss ignores them
    mask = np.zeros(200, dtype=np.uint8)
    mask[100:120] = 1
    x, y = get_batch(data, block, batch, "cpu", mask)
    sup = (y != -1)
    assert sup.any(), "every sampled window was fully masked"
    # every supervised target is a byte that mask marks, and its value is the
    # real next token -- not a shifted or reindexed one
    for row in range(batch):
        start = int((x[row, 0]).item())                  # data is a byte ramp, so value == index
        for t in range(block):
            j = start + 1 + t
            assert bool(sup[row, t]) == bool(mask[j]), (row, t, j)
            if sup[row, t]:
                assert int(y[row, t]) == int(data[j])

    from model.transformer import Parentheses
    cfg = PRESETS["tiny-smoke"]
    m = Parentheses(cfg)
    _, loss = m(x.clamp(max=cfg.vocab_size - 1), y.clamp(max=cfg.vocab_size - 1))
    assert torch.isfinite(loss), "masked batch produced a non-finite loss"

    # a corpus with no supervised bytes at all must say so, not train on nan
    try:
        get_batch(data, block, batch, "cpu", np.zeros(200, dtype=np.uint8))
        raise AssertionError("an all-zero mask must raise")
    except RuntimeError as e:
        assert "no supervised targets" in str(e)

    # resume: a checkpoint saved by one run loads cleanly into another and
    # continues from step+1 (exercises the same load/step-arithmetic main()
    # uses for --resume-from), and a cfg mismatch is a real mismatch, not a
    # test-fixture bug -- so a mismatched load can be caught rather than
    # silently producing garbage.
    import tempfile
    tiny_cfg = PRESETS["tiny-smoke"]
    m1 = Parentheses(tiny_cfg)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "step_4.pt")
        torch.save({"model": m1.state_dict(), "optimizer": opt1.state_dict(),
                    "cfg": tiny_cfg, "step": 4}, ckpt_path)

        m2 = Parentheses(tiny_cfg)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["cfg"] == tiny_cfg
        m2.load_state_dict(ckpt["model"])
        opt2.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        assert start_step == 5
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2), "resumed weights must match the source model exactly"

        other_cfg = PRESETS["parentheses-0.9-300k"]
        assert ckpt["cfg"] != other_cfg, "cfg equality check would not catch a real mismatch"
    print("[self-test] train resume ok")

    print("[self-test] train ok")


if __name__ == "__main__":
    main()
