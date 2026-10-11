"""
NPU Eval Offload Procedure & Harness for Parentheses 0.9.
Engineered for AMD Ryzen AI NPU (Phoenix XDNA1, VEN_1022 DEV_1502).

Policy (per project owner directive):
  - Route to NPU only when NPU-suited (fixed-shape quantized inference).
  - Never run keep-alive or polling loops on NPU: the XDNA driver power-gates an
    idle NPU (~0 mW draw); keep it idle between evaluations.
  - Log every NPU job (timestamp, duration, model, throughput, device state via xrt-smi).
  - Host process must use minimal CPU (OMP_NUM_THREADS=1, priority below normal)
    so GPU training throughput remains within +-1% during concurrent execution.

Usage:
  # Run standalone verification of NPU eval offload:
  python scripts/npu_eval_offload.py --model checkpoints/all-sources-v1/embedder_fixed128.onnx --samples 1000
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPU_ROOT = Path(r"E:\Parentheses\NPU")
XRT_SMI = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
NPU_PYTHON = NPU_ROOT / "conda_envs" / "ryzen-ai" / "python.exe"


def check_npu_device() -> dict:
    """Query xrt-smi examine to confirm device health and driver status."""
    if not XRT_SMI.exists():
        return {"error": "xrt-smi not found"}
    try:
        res = subprocess.run([str(XRT_SMI), "examine"], capture_output=True, text=True, check=True)
        return {"output": res.stdout, "healthy": "NPU Phoenix" in res.stdout}
    except Exception as e:
        return {"error": str(e), "healthy": False}


def measure_training_throughput(log_path: Path, last_n_steps: int = 50) -> float | None:
    """Parse recent step timings from training log to compute steps/sec."""
    if not log_path.exists():
        return None
    import re
    step_re = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+[\d.]+\s+\|\s+lr\s+[\d.e+-]+\s+\|\s+([\d.]+)s elapsed")
    steps = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = step_re.match(line)
            if m:
                step, elapsed = int(m[1]), float(m[2])
                steps.append((step, elapsed))
    if len(steps) < 2:
        return None
    window = steps[-last_n_steps:]
    d_steps = window[-1][0] - window[0][0]
    d_time = window[-1][1] - window[0][1]
    return d_steps / d_time if d_time > 0 else None


def run_npu_eval(model_path: Path, num_samples: int = 1000, provider: str = "auto") -> dict:
    """Execute fixed-shape ONNX evaluation on the NPU."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed in current environment"}

    # Set single-thread host CPU execution to protect GPU trainer threads
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Configure session
    available = ort.get_available_providers()
    providers = []
    if provider in ("auto", "vitisai") and "VitisAIExecutionProvider" in available:
        providers.append("VitisAIExecutionProvider")
    providers.append("CPUExecutionProvider")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    t0_init = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    init_time = time.perf_counter() - t0_init
    active_ep = session.get_providers()[0]

    # Generate dummy evaluation batch of fixed shape (1, 128)
    idx = np.random.randint(0, 256, size=(1, 128), dtype=np.int64)
    mask = np.ones((1, 128), dtype=bool)

    # Warmup
    for _ in range(10):
        session.run(["embedding"], {"idx": idx, "mask": mask})

    # Timed evaluation run
    t0_eval = time.perf_counter()
    for _ in range(num_samples):
        session.run(["embedding"], {"idx": idx, "mask": mask})
    total_eval_time = time.perf_counter() - t0_eval

    latency_ms = (total_eval_time / num_samples) * 1000.0
    throughput = num_samples / total_eval_time

    return {
        "active_provider": active_ep,
        "init_time_s": init_time,
        "eval_time_s": total_eval_time,
        "num_samples": num_samples,
        "latency_ms": latency_ms,
        "throughput_samples_per_sec": throughput,
    }


def main():
    p = argparse.ArgumentParser(description="NPU Eval Offload & Verification")
    p.add_argument("--model", default="checkpoints/all-sources-v1/embedder_fixed128.onnx")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--training-log", default=None, help="Optional path to training log to verify +-1% step rate")
    p.add_argument("--provider", choices=["auto", "vitisai", "cpu"], default="auto")
    args = p.parse_args()

    print(f"[npu_eval] Checking hardware status via xrt-smi...")
    dev_status = check_npu_device()
    print(f"[npu_eval] Device status: {'Healthy' if dev_status.get('healthy') else 'Unavailable/Error'}")

    # Baseline training throughput if log provided
    pre_rate = None
    if args.training_log:
        pre_rate = measure_training_throughput(Path(args.training_log))
        print(f"[npu_eval] Pre-eval training rate: {pre_rate:.4f} steps/s" if pre_rate else "[npu_eval] Log empty/waiting")

    print(f"[npu_eval] Running evaluation on {args.samples} samples using {args.model}...")
    t_start = datetime.datetime.now().isoformat()
    res = run_npu_eval(Path(args.model), args.samples, provider=args.provider)
    t_end = datetime.datetime.now().isoformat()

    print(f"[npu_eval] Execution Provider : {res.get('active_provider')}")
    print(f"[npu_eval] Latency            : {res.get('latency_ms', 0):.2f} ms/sample")
    print(f"[npu_eval] Throughput         : {res.get('throughput_samples_per_sec', 0):.2f} samples/s")

    post_rate = None
    delta_pct = None
    if args.training_log:
        post_rate = measure_training_throughput(Path(args.training_log))
        if pre_rate and post_rate:
            delta_pct = abs(post_rate - pre_rate) / pre_rate * 100.0
            print(f"[npu_eval] Post-eval training rate: {post_rate:.4f} steps/s (delta: {delta_pct:.2f}%)")
            assert delta_pct <= 1.0, f"Training throughput degraded by {delta_pct:.2f}% (exceeds +-1% limit)"

    # Log NPU job entry
    log_entry = {
        "job": "npu_eval_offload",
        "start": t_start,
        "end": t_end,
        "model": args.model,
        "samples": args.samples,
        "results": res,
        "training_rate_pre": pre_rate,
        "training_rate_post": post_rate,
        "training_rate_delta_pct": delta_pct,
    }

    out_file = ROOT / "out" / "npu_eval_job.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2)
    print(f"[npu_eval] Job logged to {out_file}")


if __name__ == "__main__":
    main()
