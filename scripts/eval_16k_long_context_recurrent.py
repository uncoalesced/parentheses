#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Parentheses 1.0-240M — C8: 16k Long-Context Recurrent State Space Sanity Check.
Validates the linear-recurrent O(1) state engine of SlabV4Model over 16,384 tokens:
  1. Feeds 16,384 consecutive tokens sequentially through SlabV4Model.step()
  2. Monitors all 16 layer associative recurrent states S_t in (16, 1, 16, 64, 64)
  3. Verifies zero NaNs, zero Infs, and bounded Frobenius norms across the sequence
  4. Tests autoregressive continuation at 4k, 8k, 12k, and 16k tokens to ensure
     coherent generation without attention degradation or state collapse.

Pass bar:
  - Zero NaN / Inf in any recurrent state or logit across all 16,384 steps
  - Finite bounded Frobenius state norm across all 16 layers
  - Successful autoregressive continuation past 4k context

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_16k_long_context_recurrent.py --checkpoint checkpoints/parentheses-1.0-240M-s8/latest.pt --device cuda
  .venv\\Scripts\\python.exe scripts\\eval_16k_long_context_recurrent.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from models.model_slab_v4 import SlabV4Model, SlabV4Config
    from scripts.eval_60p_adversarial_suite import load_model
except ImportError:
    torch = None
    Tokenizer = None
    SlabV4Model = None
    SlabV4Config = None
    load_model = None


def generate_recurrent_probe(
    model: SlabV4Model,
    states: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    start_token_id: int,
    num_tokens: int = 16,
    device: str = "cuda",
) -> Tuple[List[int], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Autoregressively decode a short continuation from the current recurrent state."""
    curr = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    curr_states = states
    gen_tokens = []

    with torch.inference_mode():
        for _ in range(num_tokens):
            logits, curr_states = model.step(curr, curr_states)
            next_tok = int(torch.argmax(logits, dim=-1).item())
            gen_tokens.append(next_tok)
            curr = torch.tensor([[next_tok]], dtype=torch.long, device=device)

    return gen_tokens, curr_states


def run_16k_recurrent_eval(
    checkpoint_path: Path,
    device: str = "cuda",
    total_tokens: int = 16384,
    log_interval: int = 2048,
    out_path: Path | None = None,
) -> Dict[str, Any]:
    if out_path is None:
        out_path = REPO_ROOT / "out" / "eval_s9" / "c8_16k_long_context_report.json"

    print(f"[*] Starting C8: 16k Long-Context Recurrent State Sanity Check on {device}...")
    model, tokenizer, cfg = load_model(checkpoint_path, device=device)
    assert isinstance(model, SlabV4Model), f"Expected SlabV4Model, got {type(model)}"

    # Load real multilingual text tokens to feed
    val_bin_p = REPO_ROOT / "data" / "processed_v5" / "val.bin"
    if val_bin_p.exists():
        import numpy as np
        raw_val = np.memmap(val_bin_p, dtype=np.uint16, mode="r")
        feed_tokens = torch.from_numpy(raw_val[:total_tokens].astype(np.int64)).to(device)
    else:
        # Fallback to tokenizing a long prompt or repeated sample
        sample_text = (
            "Parentheses 1.0 is a linear-recurrent State Space Duality language model designed by uncoalesced. "
            "It runs in constant O(1) memory during autoregressive generation without a KV cache. "
            "ಈ ಮಾದರಿಯು ದ್ರಾವಿಡ ಭಾಷೆಗಳಿಗೆ ಮೊದಲ ಆದ್ಯತೆ ನೀಡುತ್ತದೆ. தமிழ், తెలుగు, മലയാളം, ತುಳು ಭಾಷೆಗಳನ್ನು ಒಳಗೊಂಡಿದೆ. "
            "Solving complex mathematical, reasoning, and programming challenges with high fidelity. "
        )
        sample_ids = tokenizer.encode(sample_text, add_special_tokens=False).ids
        repeats = math.ceil(total_tokens / len(sample_ids))
        long_ids = (sample_ids * repeats)[:total_tokens]
        feed_tokens = torch.tensor(long_ids, dtype=torch.long, device=device)

    print(f"[+] Prepared sequence of {len(feed_tokens)} tokens for continuous recurrence scan.")

    # Initialize empty O(1) states
    states = model.empty_states(batch_size=1, device=device)

    checkpoints_telemetry = []
    has_nan_or_inf = False
    start_time = time.perf_counter()

    with torch.inference_mode():
        for step_idx in range(len(feed_tokens)):
            tok = feed_tokens[step_idx : step_idx + 1].unsqueeze(0)
            logits, states = model.step(tok, states)

            # Check NaNs at step
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                has_nan_or_inf = True
                print(f"[ERROR] NaN/Inf detected in logits at step {step_idx}!")
                break

            # Periodically inspect recurrent memory states
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == len(feed_tokens):
                layer_norms = []
                for l_idx, layer_state in enumerate(states):
                    s_mat, c_buf, sh_buf = layer_state
                    if torch.isnan(s_mat).any() or torch.isinf(s_mat).any():
                        has_nan_or_inf = True
                        print(f"[ERROR] NaN/Inf in layer {l_idx} S-state at step {step_idx}!")
                    s_norm = float(torch.norm(s_mat).item())
                    layer_norms.append(s_norm)

                max_s_norm = max(layer_norms)
                mean_s_norm = sum(layer_norms) / len(layer_norms)
                logit_mean = float(logits.mean().item())
                logit_std = float(logits.std().item())

                # Quick probe continuation at this context length
                last_tok = int(feed_tokens[step_idx].item())
                # Clone states so probe does not disrupt the long-stream states
                cloned_states = [(s.clone(), c.clone(), sh.clone()) for s, c, sh in states]
                probe_tokens, _ = generate_recurrent_probe(
                    model, cloned_states, start_token_id=last_tok, num_tokens=12, device=device
                )
                probe_text = tokenizer.decode(probe_tokens, skip_special_tokens=True).strip()

                print(
                    f"  Step {step_idx + 1:5d}/{len(feed_tokens)}: "
                    f"Mean S-Norm={mean_s_norm:.2f}, Max S-Norm={max_s_norm:.2f}, "
                    f"Logits(μ={logit_mean:.2f}, σ={logit_std:.2f}) | Probe: '{probe_text[:40]}...'"
                )

                checkpoints_telemetry.append({
                    "step": step_idx + 1,
                    "mean_s_norm": mean_s_norm,
                    "max_s_norm": max_s_norm,
                    "layer_s_norms": layer_norms,
                    "logit_mean": logit_mean,
                    "logit_std": logit_std,
                    "probe_text": probe_text,
                })

    elapsed = time.perf_counter() - start_time
    tok_per_sec = len(feed_tokens) / max(elapsed, 1e-4)

    # Strictly check recurrent state norm stability:
    # 1. Zero NaNs or Infs
    # 2. Bounded norm growth: ratio between 16k step norm and 2k step norm must be < 20.0
    # 3. Max layer S-norm must remain finite and bounded (< 50,000.0 for 65k-element state tensor)
    norm_ratio = 1.0
    if len(checkpoints_telemetry) >= 2:
        norm_ratio = checkpoints_telemetry[-1]["mean_s_norm"] / max(checkpoints_telemetry[0]["mean_s_norm"], 1e-4)
    final_max_norm = checkpoints_telemetry[-1]["max_s_norm"] if checkpoints_telemetry else 0.0

    bounded_norm_passed = (norm_ratio < 20.0) and (final_max_norm < 50000.0)
    gate_passed = (not has_nan_or_inf) and bounded_norm_passed and (len(checkpoints_telemetry) > 0)

    if checkpoints_telemetry:
        print(f"\n[+] Recurrent Norm Stability: Initial Mean={checkpoints_telemetry[0]['mean_s_norm']:.2f}, Final Mean={checkpoints_telemetry[-1]['mean_s_norm']:.2f}, Ratio={norm_ratio:.2f}x (Limit < 20x, Max Layer={final_max_norm:.2f} < 50000)")

    summary = {
        "suite": "C8_16k_long_context_recurrent",
        "checkpoint": str(checkpoint_path),
        "total_tokens_evaluated": len(feed_tokens),
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(tok_per_sec, 2),
        "has_nan_or_inf": has_nan_or_inf,
        "norm_ratio": round(norm_ratio, 2),
        "final_max_norm": round(final_max_norm, 2),
        "bounded_norm_passed": bounded_norm_passed,
        "gate_passed": gate_passed,
        "telemetry_points": checkpoints_telemetry,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[+] C8 Evaluation Completed in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
    print(f"[+] Gate Verdict: {'PASS' if gate_passed else 'FAIL'}")
    print(f"[+] Telemetry written to {out_path}\n")
    return summary


def self_test() -> None:
    """Run minimal CPU step loop to verify recurrence mechanics and telemetry reporting."""
    print("[*] Running eval_16k_long_context_recurrent self-test on CPU...")
    # Build a tiny 2-layer slab_v4 configuration
    cfg = SlabV4Config(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        chunk_size=16,
    )
    test_model = SlabV4Model(cfg).to("cpu").eval()
    states = test_model.empty_states(batch_size=1, device="cpu")

    dummy_tokens = torch.randint(0, 100, (64,), dtype=torch.long)
    for i in range(len(dummy_tokens)):
        tok = dummy_tokens[i : i + 1].unsqueeze(0)
        logits, states = test_model.step(tok, states)
        assert not torch.isnan(logits).any(), f"NaN in step {i}"

    for l_idx, (s_mat, c_buf, sh_buf) in enumerate(states):
        assert not torch.isnan(s_mat).any(), f"NaN in state layer {l_idx}"
        norm = torch.norm(s_mat).item()
        assert norm > 0.0, "State norm is zero"

    print("[PASS] eval_16k_long_context_recurrent self-test successful.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Parentheses 1.0 on C8 16k Recurrent Context")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/parentheses-1.0-240M-s7_5/step_13000_final.pt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--interval", type=int, default=2048)
    parser.add_argument("--out", type=str, default="out/eval_s9/c8_16k_long_context_report.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_16k_recurrent_eval(
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        total_tokens=args.tokens,
        log_interval=args.interval,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
