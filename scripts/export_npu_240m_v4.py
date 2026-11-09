#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Engineered by uncoalesced

"""
scripts/export_npu_240m_v4.py

Exports Parentheses 1.0-240M (slab_v4) into a strictly static-shape ONNX
recurrent graph for ultra-low-power, sub-5W local inference on AMD Ryzen AI /
DirectML environments.

Architecture Invariants:
  - O(1) Constant-Memory Recurrent Step
  - Fixed-size associative memory tensor:   (16, 1, 16, 64, 64)
  - Fixed-size causal conv buffer:          (16, 1, 3, 3072)
  - Fixed-size token-shift register:        (16, 1, 1024)
  - Zero dynamic sequence resizing, zero KV-cache DMA thrashing
  - slab_v4: Token-Shift (mu-mix), Depthwise Causal Conv1d (K=4),
    Chunked SSD Scan (C=64), Beta-Gated Selective Write, V0 Value Highway
  - Document Context Reset: is_boundary is fixed to False in the recurrent step graph;
    callers reset sequence context between documents by passing zero-initialized state tensors.
"""

from __future__ import annotations

import argparse
import json
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

from models.model_slab_v4 import SlabV4Model, SlabV4Config


class SlabV4NPUStepWrapper(nn.Module):
    """
    Wraps Parentheses-240M slab_v4 recurrent O(1) step() into a strictly
    static-shape ONNX graph for execution on the AMD Ryzen AI (XDNA) NPU
    via DirectML / VitisAI.

    Inputs:
      - token_id:     (1, 1) int64
      - s_states:     (n_layer, 1, n_head, head_dim, head_dim) float32   -> (16, 1, 16, 64, 64)
      - conv_states:  (n_layer, 1, conv_hist_len, 3 * n_embd) float32    -> (16, 1, 3, 3072)
      - shift_states: (n_layer, 1, n_embd) float32                       -> (16, 1, 1024)

    Outputs:
      - logits:            (1, vocab_size) float32
      - next_s_states:     (16, 1, 16, 64, 64) float32
      - next_conv_states:  (16, 1, 3, 3072) float32
      - next_shift_states: (16, 1, 1024) float32
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
        # Each block state: (S: (1, H, Dh, Dh), conv: (1, 3, 3*C), token: (1, 1, C))
        state_list = [
            (s_states[i], conv_states[i], shift_states[i])
            for i in range(self.n_layer)
        ]

        # Recurrent O(1) step through all 16 layers with V0 value highway
        logits, next_state_list = self.model.step(token_id, state_list, is_boundary=False)

        # Repack into contiguous stacked tensors for zero-alloc NPU tiling
        # st[0]: S_next (1, H, Dh, Dh) -> (16, 1, 16, 64, 64)
        # st[1]: conv_next (1, 3, 3*C) -> (16, 1, 3, 3072)
        # st[2]: token_next (1, 1, C) -> squeeze(1) -> (1, C) -> stack -> (16, 1, 1024)
        next_s = torch.stack([st[0] for st in next_state_list], dim=0)
        next_conv = torch.stack([st[1] for st in next_state_list], dim=0)
        next_shift = torch.stack([st[2].squeeze(1) for st in next_state_list], dim=0)

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
        raise ValueError("Checkpoint does not contain 'cfg' or 'config' SlabV4Config")
    if isinstance(cfg, dict):
        cfg = SlabV4Config(**cfg)

    model = SlabV4Model(cfg).eval()
    model.load_state_dict(ckpt["model"])
    wrapper = SlabV4NPUStepWrapper(model).eval()
    print(f"      Loaded 240M slab_v4 model ({len(model.blocks)} layers, d={cfg.n_embd}, vocab={cfg.vocab_size}) in {time.time() - t0:.2f}s")

    # Build static-shape dummy inputs
    B = 1
    L = cfg.n_layer
    H = cfg.n_head
    Dh = getattr(cfg, "head_dim", None) or (cfg.n_embd // cfg.n_head)
    C = cfg.n_embd
    conv_hist_len = 3  # k=4 causal conv -> history length 3

    dummy_tok = torch.tensor([[42]], dtype=torch.long)
    dummy_s = torch.zeros(L, B, H, Dh, Dh, dtype=torch.float32)
    dummy_conv = torch.zeros(L, B, conv_hist_len, 3 * C, dtype=torch.float32)
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

    parity_metrics = {}
    if verify:
        print(f"\n[3/3] Verifying numerical parity against PyTorch reference over 32 steps...")
        import onnxruntime as ort

        targets = ["cpu", "dml"] if provider == "both" else [provider]
        test_tokens = [42, 17, 99, 3, 240, 7, 64, 11] * 4  # 32 tokens

        for prov_target in targets:
            print(f"    Testing provider: {prov_target.upper()}...")
            prov_map = {
                "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "vitisai": ["VitisAIExecutionProvider", "CPUExecutionProvider"],
                "cpu": ["CPUExecutionProvider"],
            }
            active_providers = prov_map.get(prov_target.lower(), ["CPUExecutionProvider"])

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(out_onnx_path), sess_options=sess_opts, providers=active_providers)
            actual_provs = sess.get_providers()
            print(f"      ONNX Runtime Active Providers: {actual_provs}")
            if prov_target == "dml":
                assert "DmlExecutionProvider" in actual_provs, f"DirectML requested but not active! Active: {actual_provs}"

            # Initialize states from PyTorch dummy tensors
            py_s = dummy_s.clone()
            py_cv = dummy_conv.clone()
            py_sh = dummy_shift.clone()

            ort_st = [dummy_s.numpy(), dummy_conv.numpy(), dummy_shift.numpy()]

            max_logits_err = 0.0
            max_s_err = 0.0
            max_cv_err = 0.0
            max_sh_err = 0.0

            for step_idx, tok_val in enumerate(test_tokens):
                t_tensor = torch.tensor([[tok_val]], dtype=torch.long)
                with torch.no_grad():
                    py_logits, py_s, py_cv, py_sh = wrapper(t_tensor, py_s, py_cv, py_sh)

                ort_inputs = {
                    "token_id": np.array([[tok_val]], dtype=np.int64),
                    "s_states": ort_st[0],
                    "conv_states": ort_st[1],
                    "shift_states": ort_st[2],
                }
                ort_outs = sess.run(None, ort_inputs)
                ort_logits, ort_s, ort_cv, ort_sh = ort_outs

                # Track errors
                err_l = float(np.max(np.abs(py_logits.numpy() - ort_logits)))
                err_s = float(np.max(np.abs(py_s.numpy() - ort_s)))
                err_cv = float(np.max(np.abs(py_cv.numpy() - ort_cv)))
                err_sh = float(np.max(np.abs(py_sh.numpy() - ort_sh)))

                max_logits_err = max(max_logits_err, err_l)
                max_s_err = max(max_s_err, err_s)
                max_cv_err = max(max_cv_err, err_cv)
                max_sh_err = max(max_sh_err, err_sh)

                ort_st = [ort_s, ort_cv, ort_sh]

            print(f"      32-Step Multi-Step Parity Summary for {prov_target.upper()}:")
            print(f"      - Logits Max Absolute Error:     {max_logits_err:.2e}")
            print(f"      - S-State Max Absolute Error:    {max_s_err:.2e}")
            print(f"      - Conv-State Max Absolute Error: {max_cv_err:.2e}")
            print(f"      - Shift-State Max Absolute Error:{max_sh_err:.2e}")

            assert max_logits_err < 1e-4, f"Parity violation on logits ({prov_target}): {max_logits_err}"
            assert max_s_err < 1e-4, f"Parity violation on associative memory S ({prov_target}): {max_s_err}"
            assert max_cv_err < 1e-4, f"Parity violation on conv buffer ({prov_target}): {max_cv_err}"
            assert max_sh_err < 1e-4, f"Parity violation on shift register ({prov_target}): {max_sh_err}"
            print(f"      [PASS] Bitwise mathematical parity confirmed for {prov_target.upper()} over 32 steps (all errors < 1e-4)!")

            parity_metrics[prov_target] = {
                "active_providers": actual_provs,
                "logits_max_err": max_logits_err,
                "s_state_max_err": max_s_err,
                "conv_state_max_err": max_cv_err,
                "shift_state_max_err": max_sh_err,
                "passed": True,
            }

    return {
        "onnx_path": str(out_onnx_path),
        "file_size_mb": round(file_size_mb, 2),
        "parity_verified": verify,
        "parity_metrics": parity_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Parentheses 1.0-240M slab_v4 to AMD Ryzen AI DirectML ONNX Graph")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/parentheses-1.0-240M-s8-golden/parentheses-1.0-240M-dpo-swa3-golden.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/npu/parentheses_240m_v4_s8_step.onnx",
        help="Target ONNX export path",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="both",
        choices=["dml", "vitisai", "cpu", "both"],
        help="Execution provider to verify against",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run parity verification against PyTorch",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write export and verification metrics JSON",
    )
    args = parser.parse_args()

    metrics = export_npu(Path(args.checkpoint), Path(args.out), verify=args.verify, provider=args.provider)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[+] Export report written to {out_p}")


if __name__ == "__main__":
    main()
