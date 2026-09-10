"""
Empirical step-time benchmark for Parentheses 0.9 at sub-1M-param scale.

Why this replaces the pure-FLOPs estimate used for the 50M-350M presets
(see docs/training-time-estimate.md): at a few-hundred-thousand-parameter
scale, a single training step is nowhere near large enough to saturate a
GPU (or even a CPU) -- wall-clock time is dominated by Python loop
overhead, kernel launch latency, and the optimizer step, not raw FLOPs.
The FLOPs/peak-TFLOPS formula from the 50M+ analysis would predict
sub-millisecond steps here, which isn't a meaningful number. Measuring
real steps/sec is more honest at this scale.

Run: python3 scripts/benchmark_step.py
"""

import time

import torch

from model import PRESETS, Parentheses


def benchmark(preset: str, batch_size: int, steps: int = 30, warmup: int = 5, device: str = "cpu"):
    cfg = PRESETS[preset]
    model = Parentheses(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()  # warmup work is async; don't time it

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
        "sec_per_step": dt / steps,
        "steps_per_sec": steps / dt,
        "tokens_per_step": tokens_per_step,
        "tokens_per_sec": tokens_per_step * steps / dt,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [
        ("parentheses-0.9-100k", 64),
        ("parentheses-0.9-300k", 64),
        ("parentheses-0.9-300k-selective", 64),
        ("parentheses-0.9-600k", 32),
        ("parentheses-0.9-1m", 32),
    ]
    print(f"device: {device}\n")
    header = f"{'preset':<26}{'params':>10}{'ms/step':>10}{'tok/step':>10}{'tok/sec':>10}"
    print(header)
    print("-" * len(header))
    for preset, bs in configs:
        r = benchmark(preset, bs, device=device)
        print(f"{r['preset']:<26}{r['params']:>10,}{r['sec_per_step']*1000:>10.2f}{r['tokens_per_step']:>10}{r['tokens_per_sec']:>10,.0f}")
