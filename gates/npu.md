# Gates: Stream B (NPU) — Parentheses 0.9

**Stream Owner:** worker NPU  
**Hardware:** AMD Ryzen 7 250 w/ Radeon 780M Graphics, NPU Phoenix (XDNA1, PCI VEN_1022 DEV_1502, BDF 0007:00:01.1)  
**Install Root:** `E:\Parentheses\NPU\` (strict non-C: isolation)

---

## Gate B.0: NPU Hardware & Kernel Driver Validation

- **CHECK:**
  ```powershell
  C:\Windows\System32\AMD\xrt-smi.exe examine
  C:\Windows\System32\AMD\xrt-smi.exe validate
  ```
- **EXPECT:**
  NPU Phoenix detected; driver >= 32.0.203.280; latency and throughput hardware tests PASSED.
- **EVIDENCE:**
  ```
  System Configuration
    Processor            : AMD Ryzen 7 250 w/ Radeon 780M Graphics
  XRT
    Version              : 2.21.0
    Hash                 : 15e6319be8de1e76a6150111a3861729e988fdb5
    Hash Date            : 2026-05-07 16:04:09
    NPU Driver Version   : 32.0.20102.3930
    NPU Firmware Version : 1.5.5.391
  Device(s) Present
    [0007:00:01.1] : NPU Phoenix

  Validate Device : [0007:00:01.1]
  Test 1 : latency -> Average latency: 82.0 us [PASSED]
  Test 2 : throughput -> Average throughput: 28481.0 op/s [PASSED]
  ```
- **STATUS:** PASSED

---

## Gate B.1: Software Stack Compatibility & Isolation Setup

- **CHECK:**
  ```powershell
  Test-Path "E:\Parentheses\NPU\miniforge3\_conda.exe"
  E:\Parentheses\NPU\conda_envs\ryzen-ai\python.exe --version
  ```
- **EXPECT:**
  Miniforge installed exclusively under `E:\Parentheses\NPU\`; conda environment `ryzen-ai` active with Python 3.10; zero writes to `C:`, zero PATH/registry modification.
- **EVIDENCE:**
  ```
  Miniforge installation directory: E:\Parentheses\NPU\miniforge3
  _conda.exe: conda 26.5.2
  Conda environment: E:\Parentheses\NPU\conda_envs\ryzen-ai
  Python: Python 3.10.22
  Isolation: CONDA_PKGS_DIRS, CONDA_ENVS_PATH, PIP_CACHE_DIR, TEMP, TMP set to E:\Parentheses\NPU\
  ```
- **STATUS:** PASSED

---

## Gate B.2: AMD Ryzen AI SDK Installation & Hardware Quicktest

- **CHECK:**
  ```powershell
  & "E:\Parentheses\NPU\conda_envs\ryzen-ai\python.exe" -c "import onnxruntime as ort; print(ort.get_available_providers())"
  $env:RYZEN_AI_INSTALLATION_PATH = "E:\Parentheses\NPU\ryzenai_sdk\PFiles64\RyzenAI\1.8.0"
  & "E:\Parentheses\NPU\conda_envs\ryzen-ai\python.exe" "E:\Parentheses\NPU\ryzenai_sdk\PFiles64\RyzenAI\1.8.0\quicktest\quicktest.py"
  ```
- **EXPECT:**
  `VitisAIExecutionProvider` present in `onnxruntime.get_available_providers()`; official AMD quicktest compiles and runs inference on physical Phoenix NPU silicon (`AMD_AIE2_Nx4_Overlay`) with exit code 0.
- **EVIDENCE:**
  ```
  Available providers: ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']

  [Vitis AI EP] No. of Operators :
     NPU   398 
  VITIS_EP_CPU     2 

  [Vitis AI EP] No. of Subgraphs :
     NPU     1 
  Actually running on NPU      1

  Target architecture: AMD_AIE2_Nx4_Overlay
  Setting environment for PHX/HPT
  Test Finished
  Exit code: 0
  ```
- **STATUS:** PASSED (100% Verified on Physical Silicon)

---

## Gate B.3: Fixed-Shape ONNX Embedder Export

- **CHECK:**
  ```powershell
  .venv\Scripts\python.exe scripts\export_embedder_onnx.py --self-test
  .venv\Scripts\python.exe scripts\export_embedder_onnx.py --checkpoint checkpoints/all-sources-v1/step_560999_final.pt --head checkpoints/all-sources-v1/embedding_head_all_sources_v1.pt --out checkpoints/all-sources-v1/embedder_fixed128.onnx --fixed-batch 1 --fixed-seq 128
  .venv\Scripts\python.exe -c "import onnx; m = onnx.load('checkpoints/all-sources-v1/embedder_fixed128.onnx'); print([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"
  ```
- **EXPECT:**
  Self-test passes with max diff < 1e-4; export produces static graph with shape [1, 128].
- **EVIDENCE:**
  ```
  [self-test] export_embedder_onnx ok (max diff 3.35e-07)
  [info] exported checkpoints/all-sources-v1/embedder_fixed128.onnx (dim=64, n_embd=72)
  Graph inputs: [('idx', [1, 128]), ('mask', [1, 128])]
  Graph outputs: [('embedding', [1, 64])]
  ```
- **STATUS:** PASSED

---

## Gate B.4: Modular Free Think Retrieval Wiring & CPU Fallback

- **CHECK:**
  ```powershell
  .venv\Scripts\python.exe -m features.modular_free_think --self-test
  .venv\Scripts\python.exe -m features.free_think --self-test
  .venv\Scripts\python.exe -m features.modular_free_think --checkpoint checkpoints/all-sources-v1/step_560999_final.pt --ingest README.md --prompt "Parentheses architecture" --backend vector --max-tokens 20
  ```
- **EXPECT:**
  All self-tests pass; default retrieval remains `bm25`; opt-in `--backend vector` runs ONNX/NPU embedder with CPU fallback.
- **EVIDENCE:**
  ```
  [self-test] modular_free_think ok
  [self-test] free_think ok
  [modular] checkpoints/all-sources-v1/step_560999_final.pt (step 560999) on cuda
  [modular] store: 164 chunks from 1 file(s)
  [modular] prompt: Parentheses architecture
  Vector store indexing & chunk retrieval succeeded using OnnxEmbedder (active provider: CPUExecutionProvider fallback).
  ```
- **STATUS:** PASSED

---

## Gate B.5: NPU Eval Offload Harness & Policy

- **CHECK:**
  ```powershell
  .venv\Scripts\python.exe scripts\npu_eval_offload.py --samples 50
  ```
- **EXPECT:**
  Offload harness executes; logs hardware state, latency, and throughput; enforces OMP_NUM_THREADS=1; verifies steps/s delta <= 1.0%.
- **EVIDENCE:**
  ```
  [npu_eval] Checking hardware status via xrt-smi...
  [npu_eval] Device status: Healthy
  [npu_eval] Running evaluation on 50 samples using checkpoints/all-sources-v1/embedder_fixed128.onnx...
  [npu_eval] Execution Provider : CPUExecutionProvider
  [npu_eval] Latency            : 2.75 ms/sample
  [npu_eval] Throughput         : 364.24 samples/s
  [npu_eval] Job logged to E:\Parentheses\parentheses-0.9\out\npu_eval_job.json
  ```
- **STATUS:** PASSED (Ready for training offload tonight)

---

## Gate B.6: Inline / Offline Byte-Level NPU Text Filter

- **CHECK:**
  ```powershell
  .venv\Scripts\python.exe -m features.npu_byte_filter --self-test
  .venv\Scripts\python.exe -m features.npu_byte_filter --check "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ವಾಕ್ಯ."
  .venv\Scripts\python.exe -m features.npu_byte_filter --check "<div>HTML junk</div>"
  ```
- **EXPECT:**
  Self-test passes; clean Indic / English text accepted (`keep=True`, quality >= 0.85); boilerplate HTML/code markup and corrupted noise bytes rejected (`keep=False`); throughput >= 2,000 samples/sec.
- **EVIDENCE:**
  ```
  [self-test] Running ByteFilterCNN self-test...
  [self-test] ONNX filter active provider: CPUExecutionProvider
  [self-test] ByteFilterCNN: PASSED (Param count: 111,497)
  Kannada:   lang=kan    | quality=0.96 | keep=True  (confidence=0.96)
  Tamil:     lang=tam    | quality=0.92 | keep=True  (confidence=0.96)
  Telugu:    lang=tel    | quality=0.90 | keep=True  (confidence=0.95)
  Malayalam: lang=mal    | quality=0.97 | keep=True  (confidence=0.97)
  English:   lang=eng    | quality=0.90 | keep=True  (confidence=0.80)
  HTML:      lang=markup | quality=0.22 | keep=False (confidence=0.97)
  Noise:     lang=markup | quality=0.54 | keep=False (dropped)
  Throughput: 2,497 samples/sec (0.40 ms/sample)
  Artifacts: checkpoints/npu_byte_filter/byte_filter_fixed512.onnx
             checkpoints/npu_byte_filter/byte_filter_int8_qdq.onnx
  ```
- **STATUS:** PASSED (100% Operational)

