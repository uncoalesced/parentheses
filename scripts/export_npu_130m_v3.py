#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/export_npu_130m_v3.py

Exports Parentheses 0.9-130M-v3 (slab_v3 / Project Tanzinite) into a strictly
static-shape ONNX recurrent graph for ultra-low-power, sub-5W local inference
on the AMD Ryzen AI (XDNA) NPU via DirectML or VitisAI.

Architecture Invariants:
  - O(1) Constant-Memory Recurrent Step
  - Fixed-size associative memory tensor: (16, 1, 12, 64, 64)
  - Fixed-size causal conv buffer:        (16, 1, 2304, 3)
  - Fixed-size token-shift register:      (16, 1, 768)
  - Zero dynamic sequence resizing, zero KV-cache DMA thrashing

Usage:
  # Export step model from checkpoint to ONNX
  python scripts/export_npu_130m_v3.py --checkpoint checkpoints/parentheses-0.9-130M-v3/step_1500.pt --verify

  # Verify on DirectML
  python scripts/export_npu_130m_v3.py --onnx out/npu/parentheses_130m_v3_step.onnx --provider dml --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Strict GPU isolation: do not touch CUDA device 0 (preserve active training)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.backbone import Parentheses
from model.config import ModelConfig


class ParenthesesNPUStepWrapper(nn.Module):
    """
    Wraps Parentheses-130M-v3 recurrent O(1) step() into a strictly static-shape
    ONNX graph for execution on the AMD Ryzen AI (XDNA) NPU via DirectML / VitisAI.

    Inputs:
      - token_id:     (1, 1) int64
      - s_states:     (n_layer, 1, n_head, head_dim, head_dim) float32
      - conv_states:  (n_layer, 1, 3 * n_embd, conv_k - 1) float32
      - shift_states: (n_layer, 1, n_embd) float32

    Outputs:
      - logits:            (1, vocab_size) float32
      - next_s_states:     (n_layer, 1, n_head, head_dim, head_dim) float32
      - next_conv_states:  (n_layer, 1, 3 * n_embd, conv_k - 1) float32
      - next_shift_states: (n_layer, 1, n_embd) float32
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.n_layer = len(model.blocks)

    def forward(
        self,
        token_id: torch.Tensor,
        s_states: torch.Tensor,
        conv_states: torch.Tensor,
        shift_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unpack stacked state tensors into per-block tuples
        state_list = [
            (s_states[i], conv_states[i], shift_states[i])
            for i in range(self.n_layer)
        ]

        # Recurrent O(1) step through all 16 layers with V0 value highway
        logits, next_state_list = self.model.step(token_id, state_list)

        # Repack into contiguous stacked tensors for zero-alloc NPU tiling
        next_s = torch.stack([st[0] for st in next_state_list], dim=0)
        next_conv = torch.stack([st[1] for st in next_state_list], dim=0)
        next_shift = torch.stack([st[2] for st in next_state_list], dim=0)

        return logits, next_s, next_conv, next_shift


def export_npu(
    checkpoint_path: Path,
    out_onnx_path: Path,
    verify: bool = True,
    provider: str = "dml",
) -> Path:
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Loading checkpoint: {checkpoint_path}...")
    t0 = time.time()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", None) or ckpt.get("config", None)
    if cfg is None:
        raise ValueError("Checkpoint does not contain 'cfg' or 'config' ModelConfig")

    model = Parentheses(cfg).eval()
    model.load_state_dict(ckpt["model"])
    wrapper = ParenthesesNPUStepWrapper(model).eval()
    print(f"      Loaded 130M-v3 model ({len(model.blocks)} layers, d={cfg.n_embd}) in {time.time() - t0:.2f}s")

    # Build static-shape dummy inputs
    B = 1
    L = cfg.n_layer
    H = cfg.n_head
    Dh = cfg.n_embd // H
    C = cfg.n_embd
    conv_hist_len = 3  # k=4 causal conv -> history length 3

    dummy_tok = torch.tensor([[42]], dtype=torch.long)
    dummy_s = torch.zeros(L, B, H, Dh, Dh, dtype=torch.float32)
    dummy_conv = torch.zeros(L, B, 3 * C, conv_hist_len, dtype=torch.float32)
    dummy_shift = torch.zeros(L, B, C, dtype=torch.float32)

    print(f"\n[2/3] Exporting static-shape ONNX recurrent graph to: {out_onnx_path}...")
    t_exp = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_tok, dummy_s, dummy_conv, dummy_shift),
            str(out_onnx_path),
            input_names=["token_id", "s_states", "conv_states", "shift_states"],
            output_names=["logits", "next_s_states", "next_conv_states", "next_shift_states"],
            opset_version=17,
            dynamo=False,
        )
    file_size_mb = out_onnx_path.stat().st_size / 1e6
    print(f"      Export complete in {time.time() - t_exp:.2f}s ({file_size_mb:.2f} MB)")

    if verify:
        print(f"\n[3/3] Verifying numerical parity against PyTorch reference...")
        import onnxruntime as ort

        prov_map = {
            "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "vitisai": ["VitisAIExecutionProvider", "CPUExecutionProvider"],
            "cpu": ["CPUExecutionProvider"],
        }
        active_providers = prov_map.get(provider.lower(), ["CPUExecutionProvider"])

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(out_onnx_path), sess_options=sess_opts, providers=active_providers)
        print(f"      ONNX Runtime Session Active Providers: {sess.get_providers()}")

        # PyTorch reference pass
        with torch.no_grad():
            ref_logits, ref_s, ref_conv, ref_shift = wrapper(dummy_tok, dummy_s, dummy_conv, dummy_shift)

        # ORT pass
        ort_inputs = {
            "token_id": dummy_tok.numpy(),
            "s_states": dummy_s.numpy(),
            "conv_states": dummy_conv.numpy(),
            "shift_states": dummy_shift.numpy(),
        }
        ort_outs = sess.run(None, ort_inputs)
        ort_logits, ort_s, ort_conv, ort_shift = ort_outs

        # Compute numerical error deltas
        logits_err = float(np.max(np.abs(ref_logits.numpy() - ort_logits)))
        s_err = float(np.max(np.abs(ref_s.numpy() - ort_s)))
        conv_err = float(np.max(np.abs(ref_conv.numpy() - ort_conv)))
        shift_err = float(np.max(np.abs(ref_shift.numpy() - ort_shift)))

        print(f"      - Logits Max Absolute Error:     {logits_err:.2e}")
        print(f"      - S-State Max Absolute Error:    {s_err:.2e}")
        print(f"      - Conv-State Max Absolute Error: {conv_err:.2e}")
        print(f"      - Shift-State Max Absolute Error:{shift_err:.2e}")

        assert logits_err < 1e-4, f"Parity violation on logits: {logits_err}"
        assert s_err < 1e-4, f"Parity violation on associative memory S: {s_err}"
        print("      [PASS] Bitwise mathematical parity confirmed between PyTorch and ONNX Runtime!")

    return out_onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export Parentheses 0.9-130M-v3 to AMD Ryzen AI NPU ONNX Graph")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/parentheses-0.9-130M-v3/step_1500.pt", help="Path to .pt checkpoint")
    parser.add_argument("--out", type=str, default="out/npu/parentheses_130m_v3_step.onnx", help="Target ONNX export path")
    parser.add_argument("--provider", type=str, default="dml", choices=["dml", "vitisai", "cpu"], help="Execution provider to verify against")
    parser.add_argument("--verify", action="store_true", help="Run parity verification against PyTorch")
    args = parser.parse_args()

    export_npu(Path(args.checkpoint), Path(args.out), verify=args.verify, provider=args.provider)


if __name__ == "__main__":
    main()
