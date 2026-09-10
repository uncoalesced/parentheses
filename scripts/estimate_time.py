"""
Training time estimator for Parentheses 0.9 on an RTX 5050 Laptop GPU (8GB).

Method: standard compute approximation for dense transformer training,
FLOPs_train ~= 6 * N_params * N_tokens (Kaplan et al. 2020 / Chinchilla).
time = FLOPs_train / (peak_tensor_FLOPs * utilization_fraction)

All hardware numbers below are sourced from public spec listings (see
docs/training-time-estimate.md for citations) as of Aug 2026. The laptop
RTX 5050 spec sheet is incomplete/contested in the wild (GDDR6 vs GDDR7,
exact TGP), so we treat it as compute-equivalent to the desktop RTX 5050
(2560 CUDA cores, same architecture/clock class) and apply a mild discount
for laptop power/thermal limits.

Run: python3 scripts/estimate_time.py
"""

from dataclasses import dataclass

# --- Hardware model -----------------------------------------------------

DESKTOP_5050_FP16_TENSOR_DENSE_TFLOPS = 52.68 / 2  # sparsity-rated figure halved -> dense
LAPTOP_DISCOUNT = 0.85  # laptop power/thermal limits vs desktop TGP
PEAK_FP16_TENSOR_TFLOPS = DESKTOP_5050_FP16_TENSOR_DENSE_TFLOPS * LAPTOP_DISCOUNT
PEAK_FLOPS_PER_SEC = PEAK_FP16_TENSOR_TFLOPS * 1e12

# Achieved utilization of peak tensor throughput for single-GPU training.
# Naive PyTorch (no FlashAttention/torch.compile/fused kernels): ~10-15%.
# Reasonably optimized (AMP + FlashAttention + torch.compile): ~25-30%.
UTILIZATION = {
    "naive": 0.12,
    "optimized": 0.28,
}

SECONDS_PER_DAY = 86400


@dataclass
class Scenario:
    name: str
    params: float          # parameter count
    tokens: float           # training tokens (dataset size x epochs)


def train_seconds(params: float, tokens: float, utilization: float) -> float:
    flops = 6 * params * tokens
    achieved = PEAK_FLOPS_PER_SEC * utilization
    return flops / achieved


def fmt_days(seconds: float) -> str:
    days = seconds / SECONDS_PER_DAY
    if days < 1:
        return f"{seconds/3600:.1f} hours"
    return f"{days:.1f} days"


if __name__ == "__main__":
    print(f"Assumed peak FP16 tensor throughput (laptop, dense): {PEAK_FP16_TENSOR_TFLOPS:.1f} TFLOPS\n")

    # Chinchilla-optimal token budget: ~20 tokens per parameter.
    scenarios = [
        Scenario("50M params, Chinchilla-optimal (~1.0B tok)", 50e6, 20 * 50e6),
        Scenario("150M params, Chinchilla-optimal (~3.0B tok)", 150e6, 20 * 150e6),
        Scenario("350M params, Chinchilla-optimal (~7.0B tok)", 350e6, 20 * 350e6),
        Scenario("700M params, Chinchilla-optimal (~14.0B tok)", 700e6, 20 * 700e6),
    ]

    header = f"{'Scenario':<45}{'naive':<16}{'optimized':<16}"
    print(header)
    print("-" * len(header))
    for s in scenarios:
        naive = fmt_days(train_seconds(s.params, s.tokens, UTILIZATION["naive"]))
        opt = fmt_days(train_seconds(s.params, s.tokens, UTILIZATION["optimized"]))
        print(f"{s.name:<45}{naive:<16}{opt:<16}")
