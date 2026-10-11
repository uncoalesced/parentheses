"""
Empirical step-time benchmark for Parentheses 0.9 at sub-1M-param scale.

Engineered by uncoalesced (updated for CLI support and selective optimization).

Why this replaces the pure-FLOPs estimate used for the 50M-350M presets
(see docs/training-time-estimate.md): at a few-hundred-thousand-parameter
scale, a single training step is nowhere near large enough to saturate a
GPU (or even a CPU) -- wall-clock time is dominated by Python loop
overhead, kernel launch latency, and the optimizer step, not raw FLOPs.
Measuring real steps/sec is more honest at this scale.

Usage:
    python scripts/benchmark_step.py
    python scripts/benchmark_step.py --preset parentheses-0.9-300k-selective
    python scripts/benchmark_step.py --model selective --seq-len 2048 --batch-size 2
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

# Ensure repo root is on sys.path so the script can be invoked from any directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model import PRESETS, Parentheses


def benchmark(preset: str, batch_size: int, seq_len: int | None = None,
              steps: int = 30, warmup: int = 5, device: str = "cpu",
              compile_model: bool = False):
    cfg = PRESETS[preset]
    if seq_len is not None and seq_len != cfg.block_size:
        cfg = replace(cfg, block_size=seq_len)

    model = Parentheses(cfg).to(device)

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[warning] torch.compile failed ({e}); falling back to uncompiled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tokens_per_step = batch_size * cfg.block_size
    return {
        "preset": preset,
        "params": model.num_params(),
        "seq_len": cfg.block_size,
        "batch_size": batch_size,
        "sec_per_step": dt / steps,
        "steps_per_sec": steps / dt,
        "tokens_per_step": tokens_per_step,
        "tokens_per_sec": tokens_per_step * steps / dt,
    }


def grid_point(cfg, device: str, max_tokens: int, steps: int = 5, warmup: int = 2,
               compile_projections: bool = False, compile_backend: str = "inductor",
               grad_ckpt: bool = False) -> dict | None:
    """Largest power-of-two micro-batch (<= max_tokens tokens) that fits, timed
    with bf16 autocast like train.py. None if even batch 1 OOMs."""
    import gc
    torch.cuda.set_per_process_memory_fraction(0.92)   # real OOM, not a silent spill to shared RAM
    bs = max(1, max_tokens // cfg.block_size)
    while bs >= 1:
        oom = False
        try:
            torch.cuda.reset_peak_memory_stats()
            model = Parentheses(cfg).to(device)
            model.grad_checkpointing = grad_ckpt
            if compile_projections and getattr(cfg, "slab_v2", False):
                model.compile_projections(backend=compile_backend)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
            x = torch.randint(0, cfg.vocab_size, (bs, cfg.block_size), device=device)
            for i in range(warmup + steps):
                if i == warmup:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, x, return_logits=False)   # chunked CE, as train.py
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
            return {"params": model.num_params(), "bs": bs, "ms": dt * 1000,
                    "tok_s": bs * cfg.block_size / dt, "vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        except torch.OutOfMemoryError:
            oom = True     # handled below, outside the except, so the traceback's frame refs are gone
        finally:
            model = opt = x = loss = None
            gc.collect()
            torch.cuda.empty_cache()
        if oom:
            bs //= 2
    return None


def run_grid(args):
    """Size x context x slab_v2 grid for the BPE model (documentation.md 2026-10-07)."""
    print(f"grid: vocab {args.vocab}, tied, bf16, head_dim 64, projection over {args.hours} h\n")
    header = (f"{'d':>4}{'L':>4}{'T':>6}{'v2':>4}{'comp':>5}{'params':>9}{'emb%':>6}{'bs':>4}{'VRAM':>6}"
              f"{'ms/step':>9}{'tok/s':>9}{'tok/run':>9}{'x params':>9}")
    print(header + "\n" + "-" * len(header), flush=True)
    for d in args.d_model:
        for L in args.layers:
            for T in args.seq_lens:
                variants = [(False, False), (True, False)]
                if getattr(args, "compile_projections", False):
                    variants.append((True, True))
                for v2, comp in variants:
                    cfg = replace(PRESETS["parentheses-0.9-300k-selective"], vocab_size=args.vocab,
                                  block_size=T, n_layer=L, n_head=d // 64, n_embd=d, slab_v2=v2)
                    comp_label = "yes" if comp else "no"
                    try:
                        r = grid_point(cfg, "cuda", args.max_tokens, compile_projections=comp,
                                       compile_backend=args.compile_backend,
                                       grad_ckpt=args.grad_checkpointing)
                    except Exception as e:
                        print(f"{d:>4}{L:>4}{T:>6}{int(v2):>4}{comp_label:>5}  compile error: {type(e).__name__}: {e}", flush=True)
                        continue
                    if r is None:
                        print(f"{d:>4}{L:>4}{T:>6}{int(v2):>4}{comp_label:>5}  OOM at batch 1", flush=True)
                        continue
                    run_tok = r["tok_s"] * args.hours * 3600
                    print(f"{d:>4}{L:>4}{T:>6}{int(v2):>4}{comp_label:>5}{r['params'] / 1e6:>8.1f}M"
                          f"{100 * args.vocab * d / r['params']:>5.0f}%{r['bs']:>4}{r['vram_gb']:>5.1f}G"
                          f"{r['ms']:>9.0f}{r['tok_s']:>9,.0f}{run_tok / 1e6:>8.0f}M{run_tok / r['params']:>9.1f}",
                          flush=True)


def main():
    parser = argparse.ArgumentParser(description="Empirical step-time benchmark for Parentheses 0.9")
    parser.add_argument("--grid", action="store_true", help="size x context grid (CUDA, bf16)")
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--d-model", type=int, nargs="+", default=[384, 512, 640, 768])
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--max-tokens", type=int, default=32768, help="largest micro-batch tried, in tokens")
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--preset", type=str, default=None,
                        help="Specific preset to benchmark (e.g. parentheses-0.9-300k-selective)")
    parser.add_argument("--model", type=str, choices=["all", "causal", "selective"], default="all",
                        help="Model family to benchmark: 'all', 'causal', or 'selective'")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides default preset config)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length override (e.g. 256 or 2048)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of timed benchmark steps")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of untimed warmup steps")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to benchmark on ('cuda' or 'cpu')")
    parser.add_argument("--compile", action="store_true",
                        help="Attempt torch.compile on the model")
    parser.add_argument("--compile-projections", action="store_true",
                        help="Benchmark compiled slab_v2 projection/output paths")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="grid: per-block activation checkpointing, as train.py --grad-checkpointing")
    parser.add_argument("--compile-backend", type=str, default="inductor",
                        help="Backend for compile_projections")
    args = parser.parse_args()
    if args.grid:
        return run_grid(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    if args.preset:
        bs = args.batch_size or (64 if "300k" in args.preset or "100k" in args.preset else 32)
        configs = [(args.preset, bs)]
    elif args.model == "causal":
        bs = args.batch_size or 64
        configs = [("parentheses-0.9-300k", bs)]
    elif args.model == "selective":
        bs = args.batch_size or 64
        configs = [("parentheses-0.9-300k-selective", bs)]
    else:
        configs = [
            ("parentheses-0.9-100k", 64),
            ("parentheses-0.9-300k", 64),
            ("parentheses-0.9-300k-selective", 64),
            ("parentheses-0.9-600k", 32),
            ("parentheses-0.9-1m", 32),
        ]

    header = f"{'preset':<30}{'params':>10}{'seq':>6}{'bs':>6}{'ms/step':>10}{'step/s':>10}{'tok/sec':>12}"
    print(header)
    print("-" * len(header))
    for preset, default_bs in configs:
        bs = args.batch_size or default_bs
        r = benchmark(preset, bs, seq_len=args.seq_len, steps=args.steps,
                      warmup=args.warmup, device=device, compile_model=args.compile)
        print(f"{r['preset']:<30}{r['params']:>10,}{r['seq_len']:>6}{r['batch_size']:>6}"
              f"{r['sec_per_step']*1000:>10.2f}{r['steps_per_sec']:>10.2f}{r['tokens_per_sec']:>12,.0f}")


if __name__ == "__main__":
    main()
