"""Gate verification script for Workstream A1.

Checks that the reloaded release model (from release/parentheses-0.9-41M/model.safetensors)
produces logits identical to the training checkpoint (checkpoints/parentheses-0.9-41M/step_8082_final.pt)
on 4 fixed prompts on CPU.
Gate requirement: max abs diff < 1e-5 fp32.

Usage:
    python scripts/check_release_gate_a1.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.backbone import Parentheses


def main():
    release_dir = REPO_ROOT / "release" / "parentheses-0.9-41M"
    ckpt_path = REPO_ROOT / "checkpoints" / "parentheses-0.9-41M" / "step_8082_final.pt"

    print("=== GATE A1: Checkpoint vs Reloaded Release Parity (CPU) ===")
    print(f"Source checkpoint: {ckpt_path}")
    print(f"Release directory: {release_dir}")

    # Load original checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    orig_model = Parentheses(cfg)
    orig_model.load_state_dict(ckpt["model"], strict=True)
    orig_model.eval()

    # Load release model from release directory
    # Using the load_example loader logic
    sys.path.insert(0, str(release_dir))
    import load_example

    reloaded_model = load_example.load_model(release_dir, device="cpu")
    reloaded_model.eval()

    # 4 fixed prompts:
    # 1. <kan> (id 2)
    # 2. <tam> (id 3)
    # 3. <tel> (id 4)
    # 4. <mal> (id 5)
    # Multi-token sequences to thoroughly check hidden state recurrence and projections
    prompts = [
        ("prompt_kan", torch.tensor([[2, 100, 500, 1200]], dtype=torch.long)),
        ("prompt_tam", torch.tensor([[3, 200, 600, 1500]], dtype=torch.long)),
        ("prompt_tel", torch.tensor([[4, 300, 700, 1800]], dtype=torch.long)),
        ("prompt_mal", torch.tensor([[5, 400, 800, 2100]], dtype=torch.long)),
    ]

    max_overall_diff = 0.0
    for name, p in prompts:
        with torch.no_grad():
            orig_logits, _ = orig_model(p)
            reloaded_logits, _ = reloaded_model(p)
            diff = (orig_logits - reloaded_logits).abs().max().item()
            if diff > max_overall_diff:
                max_overall_diff = diff
            print(f"  {name} (shape {tuple(p.shape)}): max abs diff = {diff:.6e}")

    print(f"\nOverall Max Abs Diff: {max_overall_diff:.6e}")
    threshold = 1e-5
    if max_overall_diff < threshold:
        print(f"RESULT: PASS (max diff {max_overall_diff:.6e} < {threshold})")
        return 0
    else:
        print(f"RESULT: FAIL (max diff {max_overall_diff:.6e} >= {threshold})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
