<div align="center">
  <img src="assets/parentheses_lockup_transparent.svg" alt="parentheses" width="380">
</div>

<h3 align="center">parentheses-1.0-240M-tanzanite</h3>

<p align="center">
  <b>A recurrent language model for Kannada, Tamil, Telugu, Malayalam and English, built from scratch on one 8 GB laptop GPU.</b><br>
  253.54M parameters · constant 4.85 MB memory at any length · not a transformer
</p>

<p align="center">
  <a href="https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite"><img src="https://img.shields.io/github/v/release/uncoalesced/parentheses?label=release&color=F19143" alt="Latest release"></a>
  <a href="https://huggingface.co/uncoalesced-org/parentheses-1.0-240M"><img src="https://img.shields.io/badge/Hugging_Face-parentheses--1.0--240M-ffd21e" alt="Hugging Face"></a>
  <a href="docs/ARCHITECTURE.md"><img src="https://img.shields.io/badge/architecture-recurrent_SSD_(slab__v4)-9cf" alt="Architecture"></a>
  <img src="https://img.shields.io/badge/trained_on-1x_RTX_5050_Laptop_(8_GB)-green" alt="Trained on one RTX 5050 Laptop GPU">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-brightgreen" alt="Code license: MIT"></a>
  <a href="LICENSE-MODEL"><img src="https://img.shields.io/badge/weights-Apache_2.0-orange" alt="Weights license: Apache 2.0"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#downloads">Downloads</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#known-limits">Known limits</a> •
  <a href="docs/SETUP.md">Setup guide</a> •
  <a href="docs/ARCHITECTURE.md">How it works</a> •
  <a href="docs/DOCUMENTATION.md">Technical reference</a> •
  <a href="MODEL_CARD.md">Model card</a>
</p>

---

We wanted to see how far a small team could push a language model on an ordinary gaming laptop in Bengaluru. parentheses is the result: a recurrent model from the same family as Mamba-2 and RWKV, trained from scratch, with Kannada as its first language.

A transformer keeps a key and a value for every token it has seen, so its memory grows as it writes. parentheses keeps one 64-by-64 matrix per head instead. Each token fades that memory a little, writes itself in, and reads an answer back out:

$$S_t = \alpha_t S_{t-1} + \beta_t (k_t \otimes v_t), \qquad y_t = q_t S_t$$

Across all 16 layers that is 4.85 MB of state in float32, the same at token 10 as at token 16,000.

<p align="center">
  <img src="docs/figures/state-vs-kvcache.svg" alt="A transformer's KV cache grows with every token; the parentheses state stays the same size" width="760">
</p>

**Languages.** Kannada, Tamil, Telugu, Malayalam and English are the main languages. Tulu, Koya, Kuvi and Kolami are experimental. The tokenizer keeps Indic syllables (aksharas) whole, so the model reads and writes Dravidian script without breaking letters into stray bytes.

---

## Quick start

### With transformers (no repository needed)

```bash
pip install torch transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "uncoalesced-org/parentheses-1.0-240M"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()

print(model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng"))
print(model.chat(tok, "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?", lang="kan"))
```

`trust_remote_code=True` loads `modeling_parentheses.py` from the Hub repository, a self-contained copy of the model code. `chat()` and `stream()` use the constant-memory recurrent path. Language codes are `kan`, `tam`, `tel`, `mal`, `eng`, and the experimental `tcy`, `kff`, `kxv`, `nit`.

### From this repository (chat CLI)

You need Python 3.10 or newer. An NVIDIA GPU helps, but the model also runs on CPU.

```bash
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
pip install -r requirements.txt
hf download uncoalesced-org/parentheses-1.0-240M config.json model.safetensors tokenizer.json tokenizer_config.json --local-dir release/parentheses-1.0-240M-tanzanite
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device auto
```

> **On an Apple Silicon Mac (M1–M4)?** Follow [docs/APPLE_SILICON.md](docs/APPLE_SILICON.md). It covers a Mac-specific install and a one-command self-test.

Type a prompt and press Enter; an empty line exits. `--device auto` picks CUDA, then Apple Silicon `mps`, then CPU. Use `--device cpu` to force CPU. Add `--lang kan` to force Kannada; by default the script is detected per prompt. On Windows with an RTX 50-series card, install a PyTorch build for CUDA 12.8 or newer ([setup guide](docs/SETUP.md#2-make-pytorch-see-your-gpu)).

### On an NPU (DirectML)

Download `parentheses_240m_v4_step.onnx` from the [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite). It is a static-shape recurrent step graph for ONNX Runtime with DirectML, aimed at AMD Ryzen AI NPUs. [Technical reference, section 4](docs/DOCUMENTATION.md#4-edge-npu-execution-directml-and-amd-ryzen-ai) describes its inputs and outputs.

---

## Downloads

| File | What it is | Size | Where |
|:---|:---|---:|:---|
| `model.safetensors` | float32 weights, the tested reference | 1,014,203,240 bytes | [Hugging Face](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M) |
| `model-bf16.safetensors` | bfloat16 weights, half the download | 507,115,040 bytes | [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite) |
| `model.pt` | PyTorch state dict | 1,014,250,863 bytes | [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite) |
| `parentheses_240m_v4_step.onnx` | DirectML / NPU recurrent step graph | 1,148,704,972 bytes | [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite) |

All benchmarks below were measured on the float32 weights. The bf16 file was not benchmarked separately. SHA-256 hashes for every file are in `MANIFEST_sha256.tsv` on the release.

---

## Benchmarks

Nine public benchmarks, four runs each (one greedy, three sampled at T=0.7). Each run uses a small sample of a few dozen items, so read the average next to the best.

| Benchmark | What it tests | Best | Average | Peer baseline |
|:---|:---|:---:|:---:|:---|
| TerminalBench | Shell commands | **78.33%** | 74.16% | Harbor CLI baseline, 66.00% |
| TruthfulQA | Truthfulness | **91.00%** | 90.00% | OpenAI Instruct-Ada (350M), 84.50% |
| MBPP | Python functions | **90.67%** | 81.92% | Salesforce CodeGen-350M Mono, 82.20% |
| HaluEval QA | Hallucination detection | **80.00%** | 78.34% | OpenAI InstructGPT, 68.20% |
| MMLU (STEM subset) | Academic knowledge | 50.00% | 30.00% | SmolLM-135M, 30.20% |
| HellaSwag | Commonsense continuation | 33.33% | 24.16% | SmolLM-360M, 42.80% |
| ARC-Challenge | Science questions | 26.67% | 16.67% | SmolLM-360M, 34.20% |
| Belebele (Dravidian) | Reading comprehension | 25.00% | 17.18% | BLOOM-560M, 31.25% |
| GSM8K | Grade-school maths | 10.00% | 5.83% | MobileLLM-350M, 13.20% |

We beat the peer baseline on the first four and trail it on HellaSwag, ARC-Challenge, Belebele and GSM8K. Per-run numbers and citations are in [BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md).

<p align="center">
  <img src="docs/figures/benchmark-comparison-top-labs.svg" alt="parentheses-1.0-240M against peer-scale baselines" width="880">
</p>

### Against 1B-scale models

| | parentheses-1.0-240M | TinyLlama-1.1B | OPT-1.3B | Pythia-1.4B | RWKV-4 1.5B |
|:---|:---:|:---:|:---:|:---:|:---:|
| Architecture | Recurrent | Transformer | Transformer | Transformer | Recurrent |
| MMLU | **50.00%** (30.0% avg) | 26.04% | 25.00% | 25.50% | 25.80% |
| ARC-Challenge | 26.67% (16.7% avg) | 30.12% | 31.70% | **32.60%** | 29.40% |
| GSM8K | **10.00%** (5.8% avg) | 1.50% | 2.50% | 2.80% | 2.10% |
| TruthfulQA | **91.00%** (90.0% avg) | 35.08% | 33.50% | 38.60% | 37.20% |
| HaluEval QA | **80.00%** (78.3% avg) | 53.20% | 51.20% | 52.80% | ~54.00% |
| Memory at 4,096 tokens | **4.85 MB** | 536.9 MB | 1,258.3 MB | 1,440.0 MB | ~14.7 MB |
| Memory at 16,384 tokens | **4.85 MB** | 2,147.6 MB | 5,033.2 MB | 5,760.0 MB | ~14.7 MB |

At 4,096 tokens our state is over 99% smaller than the three transformers' KV-caches. RWKV-4 is recurrent too, so its state is also fixed.

<p align="center">
  <img src="docs/figures/benchmark-1b-scale-efficiency.svg" alt="parameter and memory efficiency against 1B-scale models" width="880">
</p>

### Our own held-out checks (300 prompts)

| Check | Result |
|:---|:---|
| Clean stop at `<|endoftext|>` | 95.67% (287/300) |
| Internal tag leaks | 0.00% (0/300) |
| Dravidian answers in the right script | 100.00% (80/80) |
| Abstains on trap questions | 94.00% (94/100) |
| Refuses ordinary questions | 0.00% (0/90) |
| Recurrent state over 16,384 tokens | stable, no NaN or drift |

The ONNX step graph matches PyTorch to within 5.5e-5 on the logits over 32 decoding steps (CPU and DirectML).

---

## Known limits

- **Short Python functions work best.** Factorial, palindrome and Fibonacci come out correct; longer programs are less reliable.
- **Dravidian answers are shallow.** They stay in script and on topic, but are brief and can repeat a line.
- **English facts are weak.** At 253M parameters the model can produce fluent, specific, wrong answers. Check anything that matters.
- **Use the chat format.** The `<think>` reasoning format is experimental on this checkpoint and returns templated text.
- **Maths is weak** (GSM8K average 5.83%). The chat script answers simple arithmetic with an exact calculator.
- **Speed:** 11 to 34 tokens per second on the RTX 5050 Laptop GPU with the Python recurrent loop, typically 20 to 27. NPU speed and power have not been measured yet.

---

## How we trained it

Everything ran on one NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB of VRAM. We grew the 240M model from our earlier 130M model with a function-preserving widening (Net2WiderNet), then trained for 14,173 steps across ten sessions: multilingual pretraining, a STEM and code mix, textbook text with multilingual replay, a learning-rate anneal on commercially licensed data, supervised fine-tuning, preference tuning, checkpoint souping, and a final calibration pass for clean stopping and abstention. [How it works](docs/ARCHITECTURE.md#how-it-was-trained) tells the story, including the step where training loss fell while every language got worse.

| | |
|:---|:---|
| Parameters | 253,544,176 |
| Layers / width / heads | 16 / 1024 / 16 heads of 64 |
| SwiGLU hidden width | 2752 |
| Vocabulary | 32,768 byte-level BPE (`v3_32k`), akshara-preserving |
| Training context | 4,096 tokens |
| Recurrent state | 4.85 MB in float32, constant |
| Training data | Licences that allow commercial use only ([policy](LICENSE_POLICY.md), [ledger](TRAINING_DATA.md)) |

---

## Repository layout

| Path | What is there |
|:---|:---|
| `models/` | The 1.0 model (`model_slab_v4.py`) and its document-boundary scan (`slab_v4_boundary.py`) |
| `hf/` | Self-contained Hugging Face `transformers` port, shipped to the Hub |
| `model/` | The older 0.9 model and the shared sampler (`sampling.py`) |
| `scripts/` | Chat CLI, training drivers, exports, evaluation and figure scripts |
| `features/` | FreeThink, retrieval, conversation memory, image and document input |
| `tests/` | Model, parity, AutoModel and feature tests |
| `docs/` | Setup guide, architecture guide, technical reference, roadmap, benchmark spec, figures |
| `train.py`, `app.py` | Training entry point and local web studio |

---

## Roadmap

| Milestone | Status |
|:---|:---|
| parentheses-0.9-130M preview (CC BY-NC 4.0 weights) | Released |
| parentheses-1.0-240M-tanzanite (Apache 2.0 weights, MIT code) | **Released, November 2026** |
| parentheses-1.0-multimodal (speech and acoustic tokens) | In progress |

Details in [ROADMAP.md](docs/ROADMAP.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md).

---

## Questions, bugs and feedback

Open an issue on [GitHub](https://github.com/uncoalesced/parentheses/issues) or start a thread in the [Hugging Face discussions](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M/discussions). We read both. Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@software{parentheses_2026,
  title   = {parentheses-1.0-240M-tanzanite: a recurrent language model for Dravidian languages and English},
  author  = {uncoalesced},
  year    = {2026},
  url     = {https://github.com/uncoalesced/parentheses},
  version = {1.0.0-tanzanite}
}
```

## Licenses

- Code: [MIT](LICENSE)
- Model weights: [Apache 2.0](LICENSE-MODEL)

<p align="center"><sub>Engineered by uncoalesced</sub></p>
