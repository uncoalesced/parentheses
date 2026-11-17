<div align="center">
  <img src="assets/parentheses_lockup_transparent.svg" alt="parentheses" width="380">
</div>

<h3 align="center">parentheses-1.0-240M-tanzanite</h3>

<p align="center">
  <b>A recurrent language model for Kannada, Tamil, Telugu, Malayalam and English, with constant-size state.</b><br>
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
  <a href="#how-we-trained-it">How we trained it</a> •
  <a href="#documentation">Documentation</a> •
  <a href="docs/SETUP.md">Setup guide</a> •
  <a href="docs/ARCHITECTURE.md">How it works</a> •
  <a href="docs/DOCUMENTATION.md">Technical reference</a> •
  <a href="MODEL_CARD.md">Model card</a> •
  <a href="RELEASE_NOTES.md">Release notes</a> •
  <a href="docs/ROADMAP.md">Roadmap</a> •
  <a href="HISTORY.md">History</a>
</p>

---

This repository is the 1.1 line of parentheses. It ships the 1.0 model code, inference runtime and tokenizer, and the 1.0 weights run from it unchanged. We wanted to see how far a small team could push a language model on an ordinary gaming laptop. parentheses is the result: a recurrent model from the same family as Mamba-2 and RWKV, grown from our 120M proof-of-concept model, with Kannada as its first language.

A transformer keeps a key and a value for every token it has seen, so its memory grows as it writes. parentheses keeps one 64-by-64 matrix per head instead. Each token fades that memory a little, writes itself in, and reads an answer back out:

$$S_t = \alpha_t S_{t-1} + \beta_t (k_t \otimes v_t), \qquad y_t = q_t S_t$$

Across all 16 layers that is 4.85 MB of state in float32, the same at token 10 as at token 16,000.

<p align="center">
  <img src="docs/figures/state-vs-kvcache.svg" alt="A transformer's KV cache grows with every token; the parentheses state stays the same size" width="760">
</p>

**Languages.** Kannada, Tamil, Telugu, Malayalam and English are the main languages. Tulu, Koya, Kuvi and Kolami are experimental. The pre-tokenizer keeps vowel signs and viramas in the same word as their letter, so text never breaks into stray bytes. BPE can still split an akshara before a vowel sign inside a word.

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

### From this repository

You need Python 3.10 or newer. An NVIDIA GPU helps, but the model also runs on CPU.

```bash
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
pip install -e .
hf download uncoalesced-org/parentheses-1.0-240M model.safetensors --local-dir weights
python -m parentheses.chat --checkpoint weights --device auto
```

Type a prompt and press Enter; an empty line exits. `--device auto` picks CUDA, then Apple Silicon `mps`, then CPU. Use `--device cpu` to force CPU. Add `--lang kan` (or `tam`, `tel`, `mal`, `eng`) to force a language; by default the script is detected for each prompt from the four Dravidian scripts. Each message starts a fresh conversation unless you pass `--keep-history`. On Windows with an RTX 50-series card, install a PyTorch build for CUDA 12.8 or newer ([setup guide](docs/SETUP.md#2-make-pytorch-see-your-gpu)).

> **On an Apple Silicon Mac (M1–M4)?** Follow [docs/APPLE_SILICON.md](docs/APPLE_SILICON.md). It covers a Mac install and a one-command self-test:
>
> ```bash
> python -m parentheses.selftest --checkpoint weights
> ```

### On an NPU (DirectML)

Download `parentheses_240m_v4_step.onnx` from the [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite). It is a static-shape recurrent step graph for ONNX Runtime with DirectML, aimed at AMD Ryzen AI NPUs. [Technical reference, section 4](docs/DOCUMENTATION.md#4-static-shape-step-graph-for-npus) describes its inputs and outputs.

---

## Downloads

| File | What it is | Size | Where |
|:---|:---|---:|:---|
| `model.safetensors` | float32 weights, the tested reference | 1,014,203,240 bytes | [Hugging Face](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M) |
| `model-bf16.safetensors` | bfloat16 weights, half the download | 507,115,040 bytes | [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite) |
| `model.pt` | PyTorch state dict | 1,014,250,863 bytes | [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite) |
| `parentheses_240m_v4_step.onnx` | DirectML / NPU recurrent step graph | 1,148,704,972 bytes | [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite) |

The float32 file is the reference copy. The bf16 file was not tested separately. SHA-256 hashes for every file are in `MANIFEST_sha256.tsv` on the release.

---

## Benchmarks

The tables published with 1.0 overstated the model. They used small samples, keyword scoring, and the best of four runs per task. We have removed them. The history of that mistake and the correction is in [HISTORY.md](HISTORY.md#the-benchmark-tables-november-2026).

Scores now come from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on the full test sets, with one greedy run per task. A person reads the raw results and at least 20 scored outputs for each task before anything is published. The re-run of the 1.0 model is done and under review, and its numbers will be published when that review is complete.

To run a task yourself:

```bash
pip install -e ".[eval]"
python -m parentheses.evaluate --checkpoint weights/model.safetensors --tasks arc_easy,hellaswag
```

Results are written to `eval/results/` by default. The full protocol, including the task list and the rules for what counts as a published number, is in [BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md).

### Memory test: how far back can it recall?

This is an internal check, not a benchmark. We hide a 5-digit number in some text, add 0 to 4,096 tokens of unrelated passages after it ([Belebele](https://huggingface.co/datasets/facebook/belebele)), and then ask for the number back. We score it three ways:
- **Choice:** does the model rank the real number above four decoys? Chance is 0.20.
- **Exact:** does its greedy answer start with the right digits?
- **Gain:** how much more likely is the real number than a decoy, in nats?

Each cell is 50 items. The sentences in Kannada, Tamil, Telugu and Malayalam were written by an AI model and have not been checked by native speakers yet.

Plain document, exact recall:

| Tokens after the number | English | Kannada | Tamil | Telugu | Malayalam |
|--:|--:|--:|--:|--:|--:|
| 128 | 0.96 | 0.98 | 0.96 | 1.00 | 1.00 |
| 512 | 0.52 | 0.86 | 0.58 | 0.90 | 0.88 |
| 1,024 | 0.08 | 0.26 | 0.04 | 0.44 | 0.36 |
| 2,048 (choice) | 0.68 | 0.82 | 0.70 | 0.86 | 0.86 |
| 4,096 (choice) | 0.20 | 0.32 | 0.26 | 0.24 | 0.24 |

What this shows:
- Exact recall holds to about 256-512 tokens.
- Recall stays above chance to about 2,048 tokens and is gone by 4,096.
- In chat format it fades faster. English exact recall at 512 tokens is 0.04.
- Kannada, Telugu and Malayalam hold on longer than English and Tamil.
- Making the memory fade more slowly at inference time roughly triples English recall at 2,048 tokens in chat. It also makes text prediction about 9% worse and hurts short-range recall, so it is a diagnosis, not a setting to use.

Raw per-item results and the full table, including the chat and ablation runs, are in [eval/results/needle-recall-1.0/](eval/results/needle-recall-1.0/summary.md).

---

## Known limits

- **Short Python functions work best.** Factorial, palindrome and Fibonacci come out correct; longer programs are less reliable.
- **Dravidian answers are shallow.** They stay in script and on topic, but they are brief and can repeat a line.
- **English facts are weak.** At 253M parameters the model can produce fluent, specific, wrong answers. Check anything that matters.
- **Single-turn chats only.** The model was trained on single-turn chats, so follow-up questions usually get the previous answer again. Each message starts a fresh conversation by default; `--keep-history` exists, but it does not fix this.
- **Maths is weak.** Do not rely on it for numbers.
- **No speed numbers yet.** We have not published throughput for the current runtime or for the NPU.

---

## How we trained it

Everything ran on one NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB of VRAM. We grew the 240M model from our earlier 120M model with a function-preserving widening (Net2WiderNet), then trained for 14,173 steps across ten sessions: multilingual pretraining, a STEM and code mix, textbook text with multilingual replay, a learning-rate anneal, supervised fine-tuning, preference tuning, checkpoint souping, and a final calibration pass for clean stopping and abstention. [How it works](docs/ARCHITECTURE.md#how-it-was-trained) tells the story, including the step where training loss fell while every language got worse.

1.0 used only commercially licensed data. From 1.1 the training data and the data pipeline are not published. The licence tiers we use are in [LICENSE_TABLE.md](docs/LICENSE_TABLE.md).

| | |
|:---|:---|
| Parameters | 253,544,176 |
| Layers / width / heads | 16 / 1024 / 16 heads of 64 |
| SwiGLU hidden width | 2752 |
| Vocabulary | 32,768 byte-level BPE (`v3_32k`) |
| Training context | 4,096 tokens |
| Recurrent state | 4.85 MB in float32, constant |
| Attention | None |

---

## Documentation

| Document | What it is for |
|:---|:---|
| [SETUP.md](docs/SETUP.md) | Installing, making PyTorch see your GPU, downloading weights, chat options and troubleshooting |
| [APPLE_SILICON.md](docs/APPLE_SILICON.md) | Installing and self-testing on an M1 to M4 Mac |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the model works and [how it was trained](docs/ARCHITECTURE.md#how-it-was-trained) |
| [DOCUMENTATION.md](docs/DOCUMENTATION.md) | Technical reference, including the NPU graph and its inputs and outputs |
| [MODEL_CARD.md](MODEL_CARD.md) | Model card: intended use, limits and training data summary |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | What changed in each release |
| [ROADMAP.md](docs/ROADMAP.md) | Planned work and milestones |
| [HISTORY.md](HISTORY.md) | How the project got here, with dates, including the benchmark correction |
| [BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md) | How every published number is produced |
| [LICENSE_TABLE.md](docs/LICENSE_TABLE.md) | Licence tiers for training data |
| [AGENTS.md](AGENTS.md) | Contributor and coding agent rules for this repository |
| [SECURITY.md](SECURITY.md) | How to report a security problem |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Expected behaviour in project spaces |

---

## Repository layout

| Path | What is there |
|:---|:---|
| `python/parentheses/` | The package: model definition, recurrent session and step graph, checkpoint loading, chat CLI, self-test, evaluation and the lm-eval adapter |
| `python/parentheses/tokenizer_data/` | Tokenizer files shipped with the package |
| `tests/` | Tests for the chat loop, sessions, the step graph, the lm-eval adapter and the model |
| `docs/` | Setup and Apple Silicon guides, architecture, technical reference, roadmap, benchmark protocol, licence table and figures |
| `eval/` | Reserved for reviewed raw benchmark results (`eval/results/`) |
| `tools/` | Repository tools, including the public-tree check |
| `assets/` | Logos and lockup images |
| `rust/` | Reserved for the inference runtime, tokenizer and data hot paths. Not yet populated |

---

## Roadmap

| Milestone | Status |
|:---|:---|
| parentheses-0.9-130M preview (CC BY-NC 4.0 weights) | Released |
| parentheses-1.0-240M-tanzanite (Apache 2.0 weights, MIT code) | **Released, November 2026** |
| 1.0 re-measured with lm-evaluation-harness | Run done, in review |
| parentheses-1.1 (500M+, image input, speech) | In development, target mid-December 2026 |
| Model family release | Planned for early 2027 |

Details in [ROADMAP.md](docs/ROADMAP.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md).

---

## Questions, bugs and feedback

Open an issue on [GitHub](https://github.com/uncoalesced/parentheses/issues) or start a thread in the [Hugging Face discussions](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M/discussions). We read both. Contributions are welcome. Read [AGENTS.md](AGENTS.md) for the rules that apply to every change, and [SECURITY.md](SECURITY.md) before reporting a vulnerability.

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
- 1.0 model weights: [Apache 2.0](LICENSE-MODEL)
- 0.9-130M preview weights: CC BY-NC 4.0

<p align="center"><sub>Engineered by uncoalesced</sub></p>
