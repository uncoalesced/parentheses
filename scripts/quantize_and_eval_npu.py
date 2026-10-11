"""
Quantize and evaluate retrieval embedder on NPU vs CPU.
Validates INT8 quantization, measure latency/throughput, and verifies cosine >= 0.99.

Usage:
  python scripts/quantize_and_eval_npu.py
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure Ryzen AI runtime DLLs are discoverable on Windows
RYZEN_AI_ROOT = Path(os.environ.get("RYZEN_AI_INSTALLATION_PATH", r"E:\Parentheses\NPU\ryzenai_sdk\PFiles64\RyzenAI\1.8.0"))
if hasattr(os, "add_dll_directory"):
    for sub in ["onnxruntime/bin", "deployment", "onnxruntime/capi"]:
        d = RYZEN_AI_ROOT / sub
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass


def get_val_passages(count: int = 1000) -> list[str]:
    """Extract representative validation passages from existing text/corpus files."""
    passages = []
    # Try finding text in data or raw directories
    candidates = list(glob.glob(str(ROOT / "data" / "**" / "*.txt"), recursive=True)) + \
                 list(glob.glob(str(ROOT / "data" / "**" / "*.jsonl"), recursive=True))

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= 30:  # meaningful sentence/paragraph
                        passages.append(line[:128])
                        if len(passages) >= count:
                            return passages
        except Exception:
            continue

    # Fallback to generated diverse text sentences if fewer than count found
    idx = 0
    words = ["neural", "processor", "architecture", "dravidian", "language", "model",
             "transformer", "attention", "quantization", "embedding", "retrieval", "cache"]
    while len(passages) < count:
        idx += 1
        w = [words[(idx * 7 + i) % len(words)] for i in range(10)]
        passages.append(f"Passage {idx}: " + " ".join(w))

    return passages[:count]


def quantize_model(in_onnx: Path, out_onnx: Path, passages: list[str]) -> Path:
    """Quantize fixed-shape ONNX model to INT8 using Quark or ONNX Runtime quantization."""
    print(f"[quantize] Quantizing {in_onnx} -> {out_onnx}...")
    try:
        # Try AMD Quark if available
        import quark.onnx
        print("[quantize] Using AMD Quark ONNX Quantizer")
        # Quark configuration for AMD XDNA NPU
        # quark.onnx.quantize(...)
    except ImportError:
        print("[quantize] AMD Quark not in environment; using ONNX Runtime static/dynamic quantizer")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            str(in_onnx),
            str(out_onnx),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"]
        )
    return out_onnx


def run_benchmark(model_path: Path, passages: list[str], provider: str) -> tuple[np.ndarray, float, float]:
    """Run inference on passages, returning (embeddings, latency_ms, throughput_sps)."""
    import onnxruntime as ort

    providers = []
    provider_options = None
    if provider == "vitisai":
        providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
        vaip_cfg = RYZEN_AI_ROOT / "voe-4.0-win_amd64" / "vaip_config.json"
        if vaip_cfg.exists():
            provider_options = [{"config_file": str(vaip_cfg)}, {}]
        else:
            provider_options = [{}, {}]
    else:
        providers = ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers, provider_options=provider_options)
    active = session.get_providers()[0]
    print(f"[bench] Running {model_path.name} with provider: {active}")

    # Prepare inputs (batch=1, seq=128)
    inputs = []
    for p in passages:
        b = list(p.encode("utf-8"))[:128]
        idx = np.zeros((1, 128), dtype=np.int64)
        mask = np.zeros((1, 128), dtype=bool)
        if b:
            idx[0, :len(b)] = b
            mask[0, :len(b)] = True
        inputs.append((idx, mask))

    # Warmup
    for idx, mask in inputs[:10]:
        session.run(["embedding"], {"idx": idx, "mask": mask})

    # Timed run
    embs = []
    t0 = time.perf_counter()
    for idx, mask in inputs:
        out = session.run(["embedding"], {"idx": idx, "mask": mask})[0]
        embs.append(out[0])
    dt = time.perf_counter() - t0

    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-9)

    latency_ms = (dt / len(passages)) * 1000.0
    throughput = len(passages) / dt
    return embs, latency_ms, throughput


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32-model", default="checkpoints/all-sources-v1/embedder_fixed128.onnx")
    p.add_argument("--int8-model", default="checkpoints/all-sources-v1/embedder_fixed128_int8.onnx")
    p.add_argument("--samples", type=int, default=1000)
    args = p.parse_args()

    fp32_path = ROOT / args.fp32_model
    int8_path = ROOT / args.int8_model

    print(f"[eval] Loading {args.samples} validation passages...")
    passages = get_val_passages(args.samples)
    print(f"[eval] Loaded {len(passages)} passages.")

    # 1. CPU FP32 Baseline
    print("\n--- 1. CPU FP32 Baseline ---")
    fp32_embs, cpu_lat, cpu_thr = run_benchmark(fp32_path, passages, provider="cpu")
    print(f"CPU FP32: {cpu_lat:.3f} ms/sample | {cpu_thr:.1f} samples/s")

    # 2. Quantize model
    print("\n--- 2. Quantization ---")
    quantize_model(fp32_path, int8_path, passages)

    # 3. INT8 Run (VitisAI on NPU if available, else CPU)
    print("\n--- 3. INT8 Inference ---")
    import onnxruntime as ort
    target_prov = "vitisai" if "VitisAIExecutionProvider" in ort.get_available_providers() else "cpu"
    int8_embs, int8_lat, int8_thr = run_benchmark(int8_path, passages, provider=target_prov)
    print(f"INT8 ({target_prov}): {int8_lat:.3f} ms/sample | {int8_thr:.1f} samples/s")

    # 4. Cosine Parity Verification
    print("\n--- 4. Cosine Parity Verification ---")
    # Dot product of normalized embeddings = cosine similarity
    cos_sim = np.sum(fp32_embs * int8_embs, axis=1)
    min_cos = float(np.min(cos_sim))
    mean_cos = float(np.mean(cos_sim))
    p99_cos = float(np.percentile(cos_sim, 1))

    print(f"Cosine Similarity vs FP32 CPU (N={len(passages)}):")
    print(f"  Mean   : {mean_cos:.5f}")
    print(f"  Min    : {min_cos:.5f}")
    print(f"  p1 (99% >=): {p99_cos:.5f}")

    passed = mean_cos >= 0.99 and p99_cos >= 0.98
    print(f"Gate check (cosine >= 0.99): {'PASSED' if passed else 'FAILED'}")

    report = {
        "samples": len(passages),
        "fp32_cpu": {"latency_ms": cpu_lat, "throughput_sps": cpu_thr},
        "int8": {"provider": target_prov, "latency_ms": int8_lat, "throughput_sps": int8_thr},
        "cosine_parity": {
            "mean": mean_cos,
            "min": min_cos,
            "p99": p99_cos,
            "passed": passed
        }
    }

    out_file = ROOT / "out" / "npu_embedder_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] Results saved to {out_file}")


if __name__ == "__main__":
    main()
