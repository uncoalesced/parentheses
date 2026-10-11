# Gates: slab_v2 Retry (Stream C)

Owner: worker SLAB  
Date: 2026-10-08  
Hardware: 1x NVIDIA GeForce RTX 5050 Laptop GPU (8 GB)  
Environment: Windows, Python `.venv\Scripts\python.exe` (torch 2.12.0.dev20260408+cu128)  

---

### Gate C1.1 — Attention Module Self-Test & Parity (including 5-chunk test)

- **CHECK:** `.venv\Scripts\python.exe -m model.selective_linear_attention`
- **EXPECT:** dual/recurrent parity < 1e-9 in fp64; 5-chunk recurrence test passes; slab_v2 forward/step/dense/reset parity < 1e-9.
- **EVIDENCE:**
```
[self-test] selective_linear_attention: dual/recurrent parity ok (dense diff 1.776e-15, chunk diff 4.635e-15, reset diff 1.665e-15, packed isolation ok, conv diff 4.441e-16, deltanet ok)
[self-test] slab_v2: step 9.2e-16, dense 1.1e-15, reset 8.9e-16, isolation 2.2e-16 ok
```
- **STATUS:** PASS

---

### Gate C1.2 — Backbone Model Self-Test & Parity

- **CHECK:** `.venv\Scripts\python.exe -m model.backbone`
- **EXPECT:** kv-cache ok, chunked CE == full CE, forward/step and stream parity for both slab_v2=False and slab_v2=True.
- **EVIDENCE:**
```
[self-test] kv-cache ok
[self-test] chunked loss == full CE (loss diff 0.0e+00, grad diff 2.2e-08)
[self-test] selective-linear wiring (slab_v2=False): forward/step parity ok (max abs diff 2.220e-16)
[self-test] selective-linear stream (slab_v2=False): generate/step parity ok
[self-test] selective-linear wiring (slab_v2=True): forward/step parity ok (max abs diff 2.220e-16)
[self-test] selective-linear stream (slab_v2=True): generate/step parity ok
```
- **STATUS:** PASS

---

### Gate C1.3 — Training Loop Self-Test

- **CHECK:** `.venv\Scripts\python.exe train.py --self-test`
- **EXPECT:** reset_mask ok, resume ok, diff adamw ok, AGC ok, full training step ok.
- **EVIDENCE:**
```
[self-test] document-boundary reset_mask ok
[self-test] train resume ok
[self-test] differential adamw ok
[self-test] adaptive gradient clipping ok
[self-test] full training step with agc ok
[self-test] train ok
```
- **STATUS:** PASS

---

### Gate C1.4 — Throughput Benchmark (Baseline vs slab_v2 vs slab_v2+compile)

- **CHECK:** `.venv\Scripts\python.exe scripts/benchmark_step.py --grid --vocab 32768 --d-model 384 --layers 16 --seq-lens 4096 --compile-projections`
- **EXPECT:** slab_v2 compiled throughput within 10% of baseline (baseline 32.8k tok/s -> threshold >= 29.5k tok/s) without new dependencies.
- **EVIDENCE:**
```
grid: vocab 32768, tied, bf16, head_dim 64, projection over 6.0 h

   d   L     T  v2 comp   params  emb%  bs  VRAM  ms/step    tok/s  tok/run x params
------------------------------------------------------------------------------------
 384  16  4096   0   no    40.9M   31%   2  6.5G      250   32,820     709M     17.3
 384  16  4096   1   no    43.4M   29%   1  5.3G      178   23,027     497M     11.5
 384  16  4096   1  yes  compile error: TritonMissing: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at: https://github.com/triton-lang/triton
```

Alternative backend checks:
1. `backend="aot_eager"`:
Command: `.venv\Scripts\python.exe scripts/benchmark_step.py --grid --vocab 32768 --d-model 384 --layers 16 --seq-lens 4096 --compile-projections --compile-backend aot_eager`
Evidence:
```
 384  16  4096   0   no    40.9M   31%   2  6.5G      249   32,860     710M     17.3
 384  16  4096   1   no    43.4M   29%   1  5.3G      179   22,943     496M     11.4
 384  16  4096   1  yes    43.4M   29%   1  5.3G      181   22,687     490M     11.3
```
Result: 22,687 tok/s (31.0% slower than baseline; no fusion, adds AOT tracing dispatch overhead).

2. `backend="cudagraphs"`:
Fails during multi-step autograd backward with:
`RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.`

3. `backend="openxla"`:
Fails with `ImportError: Please follow the instruction in https://github.com/pytorch/xla#pytorchxla to install torch_xla`.

4. `backend="tvm"`:
Fails with `ModuleNotFoundError: No module named 'tvm'`.

- **STATUS:** ABANDON C1
  - Inductor compilation on Windows CUDA requires Triton, which is not installed, and new dependencies are strictly disallowed by swarm rules.
  - Non-Triton backends (`aot_eager`: 22.7k tok/s vs baseline 32.8k tok/s) cannot fuse operations and run 31% slower than baseline, failing the <= 10% throughput threshold.

---

### Gate C2 — Fair 2-Arm x 2-Seed x 12-Min A/B Validation

- **PRE-REGISTERED DECISION RULE:**
  - Prerequisite: slab_v2 compiled must achieve within 10% of baseline throughput in Gate C1.
  - If eligible: 2 arms x 2 seeds x 12 min equal wall-time, interleaved A,B,A,B, same corpus (`data/processed_v3`), metric equal-weight mean bits/byte over kan, tam, tel, mal, eng.
  - Winning threshold: slab_v2 wins only if >1% better mean bpb on BOTH seeds AND all tag accuracies >= 0.85. Otherwise baseline stays.
- **DECISION:** ABANDON C2
  - Reason: Gate C1 did not clear the throughput gate (slab_v2 is ~30% slower than baseline: 22.9k vs 32.8k tok/s; `torch.compile(backend="inductor")` unavailable on Windows without Triton; `aot_eager` achieves 22.7k tok/s).
  - Per instructions, C2 is skipped and C is abandoned with numbers. GPU.lock released.
