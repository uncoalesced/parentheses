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


def main():
    parser = argparse.ArgumentParser(description="Empirical step-time benchmark for Parentheses 0.9")
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
    args = parser.parse_args()

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
