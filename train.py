"""
Training entrypoint for Parentheses 0.9.

Engineered by uncoalesced

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


def boundary_reset_mask(boundaries: np.ndarray, ix, block_size: int) -> torch.Tensor:
    """(len(ix), block_size) uint8 mask: 1 everywhere, 0 at each document start.

    `boundaries` is an ascending array of document-start *token offsets*, one
    entry per document, first entry 0 -- not a per-token mask, which is the one
    way it differs from --sft-spans and the easiest thing to get wrong here.
    SelectiveLinearAttention.forward reads 0 as "a document starts at this
    position" (doc_id = cumsum(reset_mask == 0)), so a window straddling no
    boundary is all ones and behaves exactly like reset_mask=None.

    ponytail: searchsorted picks each window's boundary slice in one vectorized
    pass, then a Python loop writes only the rows that actually contain one. At
    block_size 256 against a ~5,000-token mean document that is ~5% of rows;
    the upgrade path (flat scatter over concatenated offsets) only pays for
    itself if documents ever get short enough that most rows are hits.
    """
    starts = np.asarray(ix, dtype=np.int64)
    out = np.ones((len(starts), block_size), dtype=np.uint8)
    lo = np.searchsorted(boundaries, starts, side="left")
    hi = np.searchsorted(boundaries, starts + block_size, side="left")
    for row in np.nonzero(hi > lo)[0]:
        inside = np.asarray(boundaries[lo[row]:hi[row]], dtype=np.int64) - starts[row]
        out[row, inside] = 0
    return torch.from_numpy(out)


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str,
              mask: np.memmap | None = None, boundaries: np.ndarray | None = None):
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
    # reset_mask is aligned to x's positions (absolute offsets i..i+block_size-1),
    # not y's: it gates which past tokens the attention may carry state from, and
    # the model reads it alongside the inputs.
    reset_mask = None if boundaries is None else boundary_reset_mask(boundaries, ix.numpy(), block_size)
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        if reset_mask is not None:
            reset_mask = reset_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        if reset_mask is not None:
            reset_mask = reset_mask.to(device)
    return x, y, reset_mask


def mixed_batch(data, mixes, block_size: int, batch_size: int, device: str,
                mask=None, boundaries=None):
    """get_batch() where the last rows come from extra corpora.

    `mixes` is a list of (data, boundaries_or_None, rows): each contributes a fixed
    number of rows per batch, `data` fills the rest. A corpus without boundaries
    gets all-ones reset rows (no document resets) whenever any part has a mask.
    """
    parts = [get_batch(data, block_size, batch_size - sum(n for _, _, n in mixes), device, mask, boundaries)]
    parts += [get_batch(d, block_size, n, device, None, b) for d, b, n in mixes]
    x, y = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    if all(p[2] is None for p in parts):
        return x, y, None
    rm = torch.cat([p[2] if p[2] is not None else torch.ones_like(p[0], dtype=torch.uint8) for p in parts])
    return x, y, rm


def lr_mult(step: int, decay_start: int | None, max_steps: int, min_ratio: float,
            warmup: int = 0, start_step: int = 0, decay_end: int | None = None) -> float:
    """Warmup-stable-decay: linear warmup over the first `warmup` steps from `start_step`
    (first step gets 1/warmup, never 0), 1.0 until decay_start, then linear to min_ratio at
    decay_end (default: the last step, max_steps - 1). decay_start=None keeps the old constant LR."""
    rel = step - start_step
    w = min(1.0, (rel + 1) / warmup) if warmup > 0 else 1.0
    if decay_start is None or step < decay_start:
        return w
    end = max_steps - 1 if decay_end is None else decay_end
    frac = (step - decay_start) / max(1, end - decay_start)
    return w * (1.0 + (min_ratio - 1.0) * min(frac, 1.0))


def apply_adaptive_gradient_clipping(
    model: torch.nn.Module,
    clip_factor: float = 0.05,
    decay_clip_factor: float = 0.01,
    eps_w: float = 1e-3,
    eps_g: float = 1e-6,
) -> None:
    """Unit-Wise Adaptive Gradient Clipping (AGC) & Recurrent Dissipation Floor.

    Engineered by uncoalesced

    For each parameter matrix or vector W in R^{m x n} with gradient G = grad_W L:
        G_{i, :} <- min(1, (lambda * max(||W_{i, :}||_2, eps_w)) / (||G_{i, :}||_2 + eps_g)) * G_{i, :}
    where lambda = decay_clip_factor for recurrent decay parameters (alpha, delta, decay)
    and lambda = clip_factor for standard weights.
    """
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        is_decay = any(term in name.lower() for term in ("alpha", "delta", "decay"))
        lam = decay_clip_factor if is_decay else clip_factor

        p_data = p.data
        g_data = p.grad.data

        if p_data.ndim > 1:
            dims = tuple(range(1, p_data.ndim))
            w_norm = torch.linalg.vector_norm(p_data, dim=dims, keepdim=True)
            g_norm = torch.linalg.vector_norm(g_data, dim=dims, keepdim=True)
        else:
            w_norm = p_data.abs()
            g_norm = g_data.abs()

        max_norm = torch.clamp(w_norm, min=eps_w) * lam
        clip_scale = torch.clamp(max_norm / (g_norm + eps_g), max=1.0)
        p.grad.data.mul_(clip_scale)


def build_differential_adamw(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    recurrent_lr_ratio: float = 0.1,
    use_8bit: bool = False,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
):
    """Segregated Differential AdamW optimizer.

    Engineered by uncoalesced

    Partition parameters into:
    1. recurrent_decay (alpha, delta, decay): lr = lr * recurrent_lr_ratio, weight_decay = 0.0
    2. matrix_weights (2D+ matrices: projections, embeddings, MLP): lr = lr, weight_decay = weight_decay
    3. biases_and_norms (1D biases, LayerNorm / RMSNorm weights): lr = lr, weight_decay = 0.0
    """
    decay_params = []
    matrix_params = []
    bias_norm_params = []

    decay_lr = lr * recurrent_lr_ratio

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(term in name.lower() for term in ("alpha", "delta", "decay")):
            decay_params.append(p)
        elif p.ndim < 2:
            bias_norm_params.append(p)
        else:
            matrix_params.append(p)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": decay_lr, "weight_decay": 0.0, "name": "recurrent_decay"})
    if matrix_params:
        groups.append({"params": matrix_params, "lr": lr, "weight_decay": weight_decay, "name": "matrix_weights"})
    if bias_norm_params:
        groups.append({"params": bias_norm_params, "lr": lr, "weight_decay": 0.0, "name": "biases_and_norms"})

    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, betas=betas, eps=eps)
        except ImportError:
            print("[warn] bitsandbytes not installed, falling back to torch.optim.AdamW")
    return torch.optim.AdamW(groups, betas=betas, eps=eps)


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    use_8bit: bool = False,
    differential: bool = False,
    recurrent_lr_ratio: float = 0.1,
):
    if differential:
        return build_differential_adamw(
            model,
            lr=lr,
            weight_decay=weight_decay,
            recurrent_lr_ratio=recurrent_lr_ratio,
            use_8bit=use_8bit,
        )
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
    p.add_argument("--boundaries", default=None,
                   help="document-boundary reset: .npy array of document-start token "
                        "offsets for --data (e.g. data/processed_dravidian_v1/"
                        "train_boundaries.npy). Builds a per-batch reset_mask so "
                        "selective_linear attention carries no state across a packed "
                        "document boundary. selective_linear presets only.")
    p.add_argument("--replay-data", default=None,
                   help="second uint16 .bin mixed into every batch to stop forgetting the old "
                        "domain (e.g. data/processed_all_sources_v1/train.bin). No boundaries: "
                        "its rows train without document resets, as selective-v1 did.")
    p.add_argument("--replay-frac", type=float, default=0.0,
                   help="fraction of each micro-batch drawn from --replay-data")
    p.add_argument("--mix", nargs="+", action="append", default=[], metavar="ARG",
                   help="extra corpus: PATH ROWS [BOUNDARIES]. ROWS of every micro-batch come from "
                        "PATH (uint16 .bin); BOUNDARIES (.npy) turns on document resets for it. "
                        "Repeat per corpus, e.g. one per low-resource language.")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--resume-from", default=None,
                   help="checkpoint .pt to resume from (for staged/batched runs). "
                        "--max-steps is the absolute target step, not an additional count.")
    p.add_argument("--batch-size", type=int, default=64, help="micro-batch size (per grad-accum step) -- "
                    "VRAM is not a constraint at this param scale (see docs/training-time-estimate.md), "
                    "so this defaults much higher than it would for the 50M+ presets")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--decay-start", type=int, default=None,
                   help="absolute step where LR starts a linear decay to --min-lr at --max-steps "
                        "(cooldown). Unset = constant LR, the pre-2026-10-06 behaviour.")
    p.add_argument("--decay-end", type=int, default=None,
                   help="absolute step where the decay reaches --min-lr (default --max-steps - 1); "
                        "lets one cooldown span several step-gated sessions")
    p.add_argument("--min-lr", type=float, default=0.0, help="LR at the last step when --decay-start is set")
    p.add_argument("--warmup-steps", type=int, default=0, help="linear LR warmup steps (absolute, from step 0)")
    p.add_argument("--tokenizer", default=None,
                   help="dir with the HF tokenizer.json --data was encoded with; its path and sha256 "
                        "are saved in every checkpoint (byte-level runs leave this unset)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed model init, and (separately, re-seeded after init) the batch sampler, so "
                        "two architectures with different param counts still see identical batches")
    p.add_argument("--ce-rows", type=int, default=4096,
                   help="tokens per chunked cross-entropy slice (smaller = less peak memory, a little slower)")
    p.add_argument("--deadline", type=float, default=None,
                   help="unix time: save a final checkpoint and stop once passed (overnight runs)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--differential-adamw", action="store_true",
                   help="use segregated Differential AdamW (separate recurrent decay, matrix, and bias/norm groups)")
    p.add_argument("--recurrent-lr-ratio", type=float, default=0.1,
                   help="ratio of recurrent decay learning rate to base learning rate (default 0.1)")
    p.add_argument("--agc", action="store_true",
                   help="enable Unit-Wise Adaptive Gradient Clipping (AGC)")
    p.add_argument("--clip-factor", type=float, default=0.05,
                   help="AGC clipping threshold lambda for standard weights (default 0.05)")
    p.add_argument("--decay-clip-factor", type=float, default=0.01,
                   help="AGC clipping threshold lambda_decay for recurrent decay parameters (default 0.01)")
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--optimizer-8bit", action="store_true")
    p.add_argument("--z-loss-coeff", type=float, default=0.0,
                   help="auxiliary Z-loss coefficient c_z to penalize logit drift (recommended 1e-4 for selective_linear CPT)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true",
                   help="check batch sampling / SFT loss masking on fake data and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    cfg: ModelConfig = PRESETS[args.preset]
    if args.device == "cuda":
        # Windows lets CUDA spill into shared system RAM instead of raising OOM, which
        # runs ~10x slower with no error. Cap it so overflow is a real OOM the overnight
        # driver can react to (halve batch, double grad-accum).
        torch.cuda.set_per_process_memory_fraction(0.92)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = Parentheses(cfg).to(args.device)
    print(f"[info] {args.preset}: {model.num_params()/1e6:.1f}M params on {args.device}")

    if args.grad_checkpointing:
        model.grad_checkpointing = True   # per-block activation checkpointing in Parentheses.hidden
        print("[info] activation checkpointing on (per block)")

    optimizer = build_optimizer(
        model,
        args.lr,
        args.weight_decay,
        use_8bit=args.optimizer_8bit,
        differential=args.differential_adamw,
        recurrent_lr_ratio=args.recurrent_lr_ratio,
    )

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

    boundaries = None
    if args.boundaries:
        # Raise here rather than letting Block.forward's assert fire five frames
        # into the first backward pass: causal attention has no recurrent state
        # to reset, so --boundaries on a causal preset is a launch-command bug.
        if cfg.attn_type != "selective_linear":
            raise ValueError(
                f"--boundaries needs attn_type='selective_linear'; --preset {args.preset!r} "
                f"is {cfg.attn_type!r}. Causal attention carries no state across a document "
                f"boundary, so there is nothing for a reset_mask to do.")
        boundaries = np.load(args.boundaries, mmap_mode="r")
        # These are document-start offsets, not a per-token mask -- a per-token
        # mask of the same name would load fine and silently mark ~every window
        # as one long document, so check the shape claim rather than trusting it.
        if boundaries.ndim != 1 or len(boundaries) < 1:
            raise ValueError(
                f"{args.boundaries} has shape {boundaries.shape}; expected a 1-D array of "
                f"document-start token offsets (one entry per document).")
        if int(boundaries[0]) != 0:
            raise ValueError(
                f"{args.boundaries} starts at offset {int(boundaries[0])}, not 0 -- the first "
                f"document must begin at token 0, so this is not a document-start index.")
        if not np.all(np.diff(np.asarray(boundaries, dtype=np.int64)) > 0):
            raise ValueError(
                f"{args.boundaries} is not strictly increasing. boundary_reset_mask() uses "
                f"np.searchsorted, which needs a sorted array; an unsorted one would mark "
                f"the wrong positions with no error.")
        if int(boundaries[-1]) >= len(data):
            raise ValueError(
                f"{args.boundaries}'s last document starts at token {int(boundaries[-1]):,}, "
                f"past the end of {args.data} ({len(data):,} tokens). These two files are not "
                f"the same corpus; re-pair them with the manifest that wrote both.")
        print(f"[info] document-boundary reset on: {len(boundaries):,} documents over "
              f"{len(data):,} tokens ({len(data)/len(boundaries):,.0f} tokens/doc mean)")

    mix_specs = list(args.mix)
    if args.replay_data:
        n_replay = round(args.batch_size * args.replay_frac)
        if n_replay < 1:
            raise ValueError(f"--replay-frac {args.replay_frac} of --batch-size {args.batch_size} gives "
                             f"{n_replay} replay rows; need at least one.")
        mix_specs.append([args.replay_data, str(n_replay)])
    mixes = []
    for spec in mix_specs:
        if len(spec) not in (2, 3):
            raise ValueError(f"--mix takes PATH ROWS [BOUNDARIES], got {spec}")
        mdata = np.memmap(spec[0], dtype=np.uint16, mode="r")
        mbound = np.load(spec[2], mmap_mode="r") if len(spec) == 3 else None
        if mbound is not None and (int(mbound[0]) != 0 or int(mbound[-1]) >= len(mdata)):
            raise ValueError(f"{spec[2]} does not index {spec[0]} (first {int(mbound[0])}, "
                             f"last {int(mbound[-1])}, {len(mdata):,} tokens)")
        if mbound is not None and cfg.attn_type != "selective_linear":
            raise ValueError(f"--mix boundaries need a selective_linear preset, not {cfg.attn_type!r}")
        mixes.append((mdata, mbound, int(spec[1])))
        print(f"[info] mix: {spec[1]}/{args.batch_size} rows per batch from {spec[0]}"
              f"{' (doc resets)' if mbound is not None else ''}")
    if sum(n for _, _, n in mixes) >= args.batch_size or any(n < 1 for _, _, n in mixes):
        raise ValueError(f"mix rows {[n for _, _, n in mixes]} must each be >= 1 and leave at least one "
                         f"row of --data in a batch of {args.batch_size}")

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
            # load_state_dict restores the saved per-group lr too, which would
            # silently override --lr on every resume; keep momentum, take this run's lr.
            fresh_lrs = [g["lr"] for g in optimizer.param_groups]
            optimizer.load_state_dict(ckpt["optimizer"])
            for g, lr in zip(optimizer.param_groups, fresh_lrs):
                g["lr"] = lr
        else:
            print(f"[warn] {args.resume_from} has no saved optimizer state (older checkpoint "
                  f"format) -- resuming with a fresh optimizer, momentum is lost")
        start_step = ckpt["step"] + 1
        print(f"[info] resumed from {args.resume_from} at step {ckpt['step']}, "
              f"continuing to step {args.max_steps - 1}")
        if start_step >= args.max_steps:
            raise ValueError(f"--max-steps {args.max_steps} is not past the resumed step "
                              f"{ckpt['step']}; nothing to do")

    tok_meta = None
    if args.tokenizer:
        import hashlib
        tj = os.path.join(args.tokenizer, "tokenizer.json")
        with open(tj, "rb") as f:
            tok_meta = {"path": args.tokenizer, "sha256": hashlib.sha256(f.read()).hexdigest()}
        if args.resume_from and ckpt.get("tokenizer") and ckpt["tokenizer"]["sha256"] != tok_meta["sha256"]:
            raise ValueError(f"--tokenizer {args.tokenizer} is not the tokenizer {args.resume_from} was "
                             f"trained with ({ckpt['tokenizer']['path']})")
        print(f"[info] tokenizer {args.tokenizer} sha256 {tok_meta['sha256'][:12]}, slab_v2={cfg.slab_v2}")

    def save(path, step):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cfg": cfg, "step": step, "tokenizer": tok_meta}, path)

    if args.seed is not None:
        # batch sampling only from here on; offset by start_step so a resume does not
        # replay the windows of the first steps
        torch.manual_seed(args.seed + 1 + start_step)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    min_ratio = args.min_lr / args.lr
    t0 = time.time()
    final_step = args.max_steps - 1
    for step in range(start_step, args.max_steps):
        if args.deadline is not None and time.time() > args.deadline:
            final_step = step - 1
            print(f"[info] --deadline passed, stopping after step {final_step}")
            break
        mult = lr_mult(step, args.decay_start, args.max_steps, min_ratio, args.warmup_steps, start_step,
                       args.decay_end)
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * mult
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y, reset_mask = mixed_batch(data, mixes, cfg.block_size, args.batch_size,
                                           args.device, mask, boundaries)
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=use_amp):
                # logits are only needed for z-loss; otherwise the chunked CE never builds
                # the (B*T, vocab) tensor -- the dominant memory cost at a 64k vocab
                logits, loss = model(x, y, reset_mask=reset_mask, return_logits=args.z_loss_coeff > 0.0,
                                     ce_rows=args.ce_rows)
                if args.z_loss_coeff > 0.0 and logits is not None:
                    log_z = torch.logsumexp(logits, dim=-1)
                    loss = loss + args.z_loss_coeff * (log_z ** 2).mean()
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        if args.agc:
            apply_adaptive_gradient_clipping(
                model,
                clip_factor=args.clip_factor,
                decay_clip_factor=args.decay_clip_factor,
            )
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {args.lr * mult:.2e} | {dt:.1f}s elapsed")

        if step > 0 and step % args.ckpt_every == 0:
            path = os.path.join(args.out_dir, f"step_{step}.pt")
            save(path, step)
            print(f"[info] saved checkpoint: {path}")

    # final save -- the loop above only checkpoints on a step%ckpt_every==0
    # step *inside* range(max_steps), so whenever max_steps isn't hit by
    # that stride (e.g. range(162000) stops at step 161999) the last
    # <ckpt_every steps of training were never written to disk.
    path = os.path.join(args.out_dir, f"step_{final_step}_final.pt")
    save(path, final_step)
    print(f"[info] saved final checkpoint: {path}")


def _self_test():
    from dataclasses import replace

    from model.backbone import Parentheses

    torch.manual_seed(0)
    block, batch = 8, 4
    data = np.arange(200, dtype=np.uint16) % 256

    # cooldown: flat before decay_start, linear to min_ratio at max_steps-1
    assert lr_mult(50, None, 100, 0.0) == 1.0
    assert lr_mult(79, 80, 101, 0.1) == 1.0
    assert lr_mult(80, 80, 101, 0.1) == 1.0
    assert abs(lr_mult(90, 80, 101, 0.1) - 0.55) < 1e-9
    assert abs(lr_mult(100, 80, 101, 0.1) - 0.1) < 1e-9
    # decay_end past this run's max_steps: the cooldown continues in the next session
    assert abs(lr_mult(90, 80, 91, 0.1, decay_end=100) - 0.55) < 1e-9
    assert abs(lr_mult(100, 80, 200, 0.1, decay_end=100) - 0.1) < 1e-9
    assert abs(lr_mult(150, 80, 200, 0.1, decay_end=100) - 0.1) < 1e-9
    # warmup composes with the cooldown and never yields lr 0
    assert abs(lr_mult(0, 80, 101, 0.1, warmup=10) - 0.1) < 1e-9
    assert abs(lr_mult(4, 80, 101, 0.1, warmup=10) - 0.5) < 1e-9
    assert lr_mult(9, 80, 101, 0.1, warmup=10) == 1.0 == lr_mult(50, 80, 101, 0.1, warmup=10)
    assert abs(lr_mult(100, 80, 101, 0.1, warmup=10) - 0.1) < 1e-9
    # warmup on resumed run
    assert abs(lr_mult(100, 180, 201, 0.1, warmup=10, start_step=100) - 0.1) < 1e-9
    assert abs(lr_mult(104, 180, 201, 0.1, warmup=10, start_step=100) - 0.5) < 1e-9
    assert lr_mult(109, 180, 201, 0.1, warmup=10, start_step=100) == 1.0

    # mixes: rows come from each corpus in order; one without boundaries never resets,
    # one with boundaries does, and main-only-without-boundaries still gets a mask
    rep = (np.arange(200, dtype=np.uint16) % 256) + 1000
    low = (np.arange(200, dtype=np.uint16) % 256) + 2000
    x, y, rm = mixed_batch(data, [(rep, None, 1)], block, batch, "cpu", None, np.array([0, 50, 120]))
    assert x.shape == rm.shape == (batch, block)
    assert (x[:-1] < 1000).all() and (x[-1] >= 1000).all()
    assert rm[-1].all()
    x, y, rm = mixed_batch(data, [(rep, None, 1), (low, np.arange(0, 200, 2), 1)], block, batch, "cpu")
    assert (x[:2] < 1000).all() and (x[2] // 1000 == 1).all() and (x[3] >= 2000).all()
    assert rm[:3].all() and not rm[3].all(), "only the bounded corpus row should contain resets"
    assert mixed_batch(data, [], block, batch, "cpu")[2] is None

    # no mask: unchanged behaviour, y is x shifted by one
    x, y, reset_mask = get_batch(data, block, batch, "cpu")
    assert reset_mask is None, "no --boundaries must mean no reset_mask, not an all-ones one"
    assert x.shape == y.shape == (batch, block)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert (y == -1).sum() == 0

    # mask: supervision is exactly the marked target positions, everything
    # else is ignore_index, and a real model's loss ignores them
    mask = np.zeros(200, dtype=np.uint8)
    mask[100:120] = 1
    x, y, _ = get_batch(data, block, batch, "cpu", mask)
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

    cfg = PRESETS["tiny-smoke"]
    m = Parentheses(cfg)
    logits, loss = m(x.clamp(max=cfg.vocab_size - 1), y.clamp(max=cfg.vocab_size - 1))
    assert torch.isfinite(loss), "masked batch produced a non-finite loss"
    log_z = torch.logsumexp(logits, dim=-1)
    z_loss = 1e-4 * (log_z ** 2).mean()
    assert torch.isfinite(z_loss) and z_loss.item() >= 0.0, "z-loss calculation failed"

    # a corpus with no supervised bytes at all must say so, not train on nan
    try:
        get_batch(data, block, batch, "cpu", np.zeros(200, dtype=np.uint8))
        raise AssertionError("an all-zero mask must raise")
    except RuntimeError as e:
        assert "no supervised targets" in str(e)

    # --boundaries: the mask is 0 at exactly the document starts that fall
    # inside each sampled window. Expected positions are built from `bounds` by
    # hand below, never read back out of boundary_reset_mask() itself.
    bounds = np.array([0, 3, 11, 50, 137, 190], dtype=np.uint32)

    # hand-picked windows first, so the interesting cases are hit deterministically
    # rather than only when randint happens to land on one
    rm = boundary_reset_mask(bounds, np.array([48, 100]), block)
    assert rm.shape == (2, block) and rm.dtype == torch.uint8, (rm.shape, rm.dtype)
    assert rm[0].tolist() == [1, 1, 0, 1, 1, 1, 1, 1], rm[0].tolist()   # 50 is 2 past 48
    assert rm[1].tolist() == [1] * block, rm[1].tolist()                # [100, 108) holds none
    # a window opening exactly on a document start resets at t=0
    assert boundary_reset_mask(bounds, np.array([137]), block)[0].tolist() == [0] + [1] * (block - 1)
    # two starts in one window
    assert boundary_reset_mask(bounds, np.array([3]), block)[0].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert boundary_reset_mask(np.array([0, 4, 6], dtype=np.uint32),
                               np.array([2]), block)[0].tolist() == [1, 1, 0, 1, 0, 1, 1, 1]

    # and through get_batch, on sampled windows
    x, y, rm = get_batch(data, block, batch, "cpu", None, bounds)
    assert rm is not None and rm.shape == (batch, block)
    for row in range(batch):
        start = int(x[row, 0].item())                    # data is a byte ramp, so value == index
        expected = [1] * block
        for b in bounds.tolist():
            if start <= b < start + block:
                expected[b - start] = 0
        assert rm[row].tolist() == expected, (row, start, rm[row].tolist(), expected)

    # Wiring, not just collation: a reset_mask handed to Parentheses.forward has
    # to reach SelectiveLinearAttention. If any link in
    # forward -> hidden -> Block.forward -> attn drops it, collation above still
    # passes and training silently runs with no boundary isolation at all.
    sel_cfg = replace(PRESETS["tiny-smoke"], attn_type="selective_linear")
    m_sel = Parentheses(sel_cfg).eval()
    xi = torch.randint(0, sel_cfg.vocab_size, (2, 16))
    rm_sel = torch.ones(2, 16, dtype=torch.uint8)
    rm_sel[:, 8] = 0
    with torch.no_grad():
        plain, _ = m_sel(xi)
        reset, _ = m_sel(xi, reset_mask=rm_sel)
    # causality: a boundary at t=8 cannot reach back before it, so those logits
    # must be bitwise identical -- a mask that changed them would be a real bug
    assert torch.equal(plain[:, :8], reset[:, :8]), "reset_mask altered pre-boundary positions"
    assert not torch.equal(plain[:, 8:], reset[:, 8:]), (
        "reset_mask never reached the attention block -- forward/hidden/Block dropped it")

    # and the causal path must refuse it loudly rather than ignoring it
    causal = Parentheses(PRESETS["tiny-smoke"]).eval()
    try:
        causal(xi, reset_mask=rm_sel)
    except AssertionError as e:
        assert "reset_mask" in str(e), e
    else:
        raise AssertionError("causal model silently accepted a reset_mask")
    print("[self-test] document-boundary reset_mask ok")

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

    # Differential AdamW verification: 3 segregated groups
    m_diff = Parentheses(replace(tiny_cfg, attn_type="selective_linear"))
    opt_diff = build_differential_adamw(m_diff, lr=3e-4, weight_decay=0.1, recurrent_lr_ratio=0.1)
    group_map = {g["name"]: g for g in opt_diff.param_groups}
    assert "recurrent_decay" in group_map, "missing recurrent_decay parameter group"
    assert "matrix_weights" in group_map, "missing matrix_weights parameter group"
    assert "biases_and_norms" in group_map, "missing biases_and_norms parameter group"
    assert abs(group_map["recurrent_decay"]["lr"] - 3e-5) < 1e-12, "decay lr must be 0.1 * base lr"
    assert group_map["recurrent_decay"]["weight_decay"] == 0.0, "decay weight_decay must be 0.0"
    assert abs(group_map["matrix_weights"]["lr"] - 3e-4) < 1e-12, "matrix lr must match base lr"
    assert group_map["matrix_weights"]["weight_decay"] == 0.1, "matrix weight_decay must match base"
    assert abs(group_map["biases_and_norms"]["lr"] - 3e-4) < 1e-12, "norm lr must match base lr"
    assert group_map["biases_and_norms"]["weight_decay"] == 0.0, "norm weight_decay must be 0.0"
    total_opt_params = sum(len(g["params"]) for g in opt_diff.param_groups)
    assert total_opt_params == len(list(m_diff.parameters())), "differential optimizer must cover all parameters"
    print("[self-test] differential adamw ok")

    # Unit-Wise AGC verification
    # 1. Unclipped when gradient is small
    w_small = torch.randn(4, 8)
    g_small = torch.randn(4, 8) * 1e-5
    mod_test = torch.nn.Module()
    mod_test.w = torch.nn.Parameter(w_small.clone())
    mod_test.w.grad = g_small.clone()
    apply_adaptive_gradient_clipping(mod_test, clip_factor=0.05)
    assert torch.allclose(mod_test.w.grad, g_small, atol=1e-7), "small gradients should not be clipped"

    # 2. Clipped when gradient is excessively large
    w_large = torch.randn(4, 8)
    g_large = torch.randn(4, 8) * 100.0
    mod_test.w = torch.nn.Parameter(w_large.clone())
    mod_test.w.grad = g_large.clone()
    apply_adaptive_gradient_clipping(mod_test, clip_factor=0.05)
    row_norms_w = torch.linalg.vector_norm(w_large, dim=1)
    row_norms_g = torch.linalg.vector_norm(mod_test.w.grad, dim=1)
    max_expected = 0.05 * torch.clamp(row_norms_w, min=1e-3)
    assert (row_norms_g <= max_expected + 1e-5).all(), "gradient norm must be bounded by lambda * ||W||_2"

    # 3. Recurrent decay parameters use decay_clip_factor
    mod_decay = torch.nn.Module()
    mod_decay.alpha_proj = torch.nn.Parameter(w_large.clone())
    mod_decay.alpha_proj.grad = g_large.clone()
    apply_adaptive_gradient_clipping(mod_decay, clip_factor=0.05, decay_clip_factor=0.01)
    row_norms_decay_g = torch.linalg.vector_norm(mod_decay.alpha_proj.grad, dim=1)
    max_expected_decay = 0.01 * torch.clamp(row_norms_w, min=1e-3)
    assert (row_norms_decay_g <= max_expected_decay + 1e-5).all(), "decay gradients must be clipped with decay_clip_factor"
    print("[self-test] adaptive gradient clipping ok")

    # 4. Full step integration test with AGC and Differential AdamW
    x_test = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.block_size))
    y_test = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.block_size))
    opt_diff.zero_grad()
    _, l_test = m_diff(x_test, y_test)
    l_test.backward()
    apply_adaptive_gradient_clipping(m_diff, clip_factor=0.05, decay_clip_factor=0.01)
    opt_diff.step()
    print("[self-test] full training step with agc ok")

    print("[self-test] train ok")


if __name__ == "__main__":
    main()
