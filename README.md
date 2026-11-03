```
██████╗   █████╗  ██████╗  ███████╗ ███╗   ██╗ ████████╗ ██╗  ██╗ ███████╗ ███████╗ ███████╗ ███████╗
██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║  ██║ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
██████╔╝ ███████║ █████╔╝  █████╗   ██╔██╗ ██║    ██║    ███████║ █████╗   ███████╗ █████╗   ███████╗
██╔═══╝  ██╔══██║ ██╔══██╗ ██╔══╝   ██║╚██╗██║    ██║    ██╔══██║ ██╔══╝   ╚════██║ ██╔══╝   ╚════██║
██║      ██║  ██║ ██║  ██║ ███████╗ ██║ ╚████║    ██║    ██║  ██║ ███████╗ ███████║ ███████╗ ███████║
╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚══════╝ ╚══════╝
```

# parentheses

### parentheses-0.9-130M (Project Tanzinite)
Engineered by uncoalesced

<div align="center">
  <img src="assets/parentheses_lockup_transparent.svg" alt="parentheses / Project Tanzinite" width="380">
</div>

<p align="center">
  <b>A from-scratch recurrent language model for Kannada, four other Dravidian languages and English, trained on one laptop GPU.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Work_in_progress-yellow.svg" alt="Status: work in progress">
  <img src="https://img.shields.io/badge/Release-v0.9--130M--v3--golden-blue.svg" alt="Release">
  <img src="https://img.shields.io/badge/HuggingFace-parentheses--v0.9--130M--v3-ffd21e.svg" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Architecture-Recurrent_SSD-9cf.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/Hardware-1x_RTX_5050_Laptop_(8_GB)-green.svg" alt="Hardware">
  <img src="https://img.shields.io/badge/Code_License-MIT-brightgreen.svg" alt="Code License">
  <img src="https://img.shields.io/badge/Weights_License-CC_BY--NC_4.0-orange.svg" alt="Weights License">
</p>

<p align="center">
  <a href="#quick-setup">Quick setup</a> •
  <a href="docs/SETUP.md">Full setup guide</a> •
  <a href="docs/ARCHITECTURE.md">Architecture</a> •
  <a href="docs/ROADMAP.md">Roadmap</a> •
  <a href="docs/DOCUMENTATION.md">Documentation</a> •
  <a href="https://huggingface.co/uncoalesced/parentheses-v0.9-130M-v3">Hugging Face</a>
</p>

---

parentheses is a small language model I'm building on my own in Bengaluru, on a single 8 GB laptop GPU (an NVIDIA RTX 5050 Laptop). It isn't a transformer. It's a recurrent model from the State Space Duality family (the same family as Mamba-2 and RWKV), so it keeps a fixed-size memory while it writes instead of a cache that grows with every token. Kannada came first, then Tamil, Telugu, Malayalam and English, with early support for Tulu, Koya, Kuvi and Kolami.

The current release, parentheses-0.9-130M, is a preview with 148 million parameters. It writes the scripts it was trained on reliably. Conversation and maths are still weak, and it will make things up with total confidence. The whole project is a work in progress, and the next model, 1.0-240M, is training now.

> [!IMPORTANT]
> If you tried the model and got strings of numbers or gibberish back, the chat script was the problem, not you. It was quietly skipping 63 of the model's weight tensors when loading the Hugging Face files. Run `git pull` and try again.

---

## Quick setup

You need Python 3.10 or newer, git, and a few GB of free disk space (most of it is PyTorch; the weights are about 350 MB). An NVIDIA GPU makes it much faster, but it runs on a CPU too.

**1. Get the code and install the dependencies**

```bash
git clone https://github.com/uncoalesced/parentheses.git
```

```bash
cd parentheses
```

```bash
pip install -r requirements.txt
```

**2. Download the weights.** Open the [Hugging Face page](https://huggingface.co/uncoalesced/parentheses-v0.9-130M-v3), go to "Files and versions", and download `config.json`, `model.safetensors` and `tokenizer.json`. Put all three in a new folder called `checkpoints/parentheses-0.9-130M-v3/` inside the repo. Or let the command line do it:

```bash
hf download uncoalesced/parentheses-v0.9-130M-v3 config.json model.safetensors tokenizer.json --local-dir checkpoints/parentheses-0.9-130M-v3
```

**3. Start chatting**

```bash
python scripts/chat.py --checkpoint checkpoints/parentheses-0.9-130M-v3 --device cuda
```

Type a message and press Enter. Press Enter on an empty line to quit. Leave out `--device cuda` if you don't have an NVIDIA GPU.

**4. Pick a language (optional).** By default the chat detects the language from the script you type in. To force one, add `--lang` with a code: `kan` (Kannada), `tam`, `tel`, `mal`, `eng`, or the experimental `tcy`, `kff`, `kxv`, `nit`. For Kannada with the model's reasoning shown:

```bash
python scripts/chat.py --checkpoint checkpoints/parentheses-0.9-130M-v3 --device cuda --lang kan --format think
```

A forced language applies to every message, so typing "hi" with `--lang kan` still sends it down the Kannada path. For Telugu, always pass `--lang tel`: auto-detection often mistakes everyday Telugu for Kolami.

> [!TIP]
> If it feels slow, check that you added `--device cuda`. The script uses the CPU unless you ask for the GPU. If that flag then fails with `Torch not compiled with CUDA enabled`, pip installed a CPU-only PyTorch, which is common on Windows. Install a CUDA build from the [PyTorch selector](https://pytorch.org/get-started/locally/); RTX 50-series cards need CUDA 12.8 or newer. [SETUP.md](docs/SETUP.md#2-make-pytorch-see-your-gpu) walks through it.

---

## What it can and can't do yet

I'd rather you hear this from me than find out halfway through a conversation.

It's good at staying in the right script. In testing, 40 of 40 runs in the five main languages stayed in the language they were asked in. Its memory cost stays flat however long it writes: the last token took about as long as the first (a ratio of 1.025), at around 27 tokens a second on the laptop GPU. It didn't refuse any of the 19 sensitive-but-legitimate technical questions I tried.

It's bad at being right. On a clean set of 24 fresh reasoning prompts it scored 5. When asked about a Martian colony "founded in 1842", it described the colony instead of saying no such place exists. Kannada conversation and multi-step arithmetic fall apart quickly, and Tulu, Koya, Kuvi and Kolami are fragile. Simple sums like "17 x 6" are answered by a small built-in calculator in the chat script, not by the model.

---

## The guides

### Setup

[docs/SETUP.md](docs/SETUP.md) is the long version of the quick setup above. It opens with a legend that jumps straight to your language, then covers getting PyTorch to see your GPU, both ways of downloading the weights, every chat option and in-chat command, using the model from Python, and fixes for the problems people actually hit.

| Language | Code | Status |
|---|---|---|
| [Kannada](docs/SETUP.md#kannada) | `kan` | main |
| [Tamil](docs/SETUP.md#tamil) | `tam` | main |
| [Telugu](docs/SETUP.md#telugu) | `tel` | main |
| [Malayalam](docs/SETUP.md#malayalam) | `mal` | main |
| [English](docs/SETUP.md#english) | `eng` | main |
| [Tulu](docs/SETUP.md#tulu) | `tcy` | experimental |
| [Koya](docs/SETUP.md#koya) | `kff` | experimental |
| [Kuvi](docs/SETUP.md#kuvi) | `kxv` | experimental |
| [Kolami](docs/SETUP.md#kolami) | `nit` | experimental |

### Architecture

A transformer looks back over every earlier token each time it writes a new one, and it keeps all of them in a cache that grows as it goes. parentheses keeps one 64 x 64 memory matrix per attention head instead. For every token it fades that memory a little, writes the new token into it, and reads the answer back out:

$$S_t = \alpha_t S_{t-1} + \beta_t (k_t \otimes v_t), \qquad y_t = q_t S_t$$

That's about 1.6 MB of state for the whole model, no matter how long the output gets.

<p align="center">
  <img src="docs/figures/state-vs-kvcache.svg" alt="A transformer's KV cache grows with every token; the parentheses state stays the same size" width="760">
</p>

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the rest in plain terms: what happens inside one layer, why I went wide and shallow for 130M, how a recurrent model still trains in parallel, the tokenizer that keeps Indic letters whole, and the training recipe with its charts.

### Roadmap

| Milestone | Status |
|---|---|
| parentheses-0.9-130M preview | Released |
| 1.0-240M pretraining, sessions 1 to 4 | Done (step 8,000) |
| Session 5: textbooks plus Dravidian replay | Next |
| Session 6: learning-rate anneal | Planned |
| Instruction tuning that teaches "I don't know" | Planned |
| Preference tuning and the 1.0-240M release | Planned |

parentheses-1.0-240M is 16 layers at width 1024, grown from 0.9, and trained on the same laptop. From now on, everything I release is under Apache 2.0 and trained only on data whose license allows commercial use. That rule shuts out most good Kannada text, which is the biggest problem the project has right now. [docs/ROADMAP.md](docs/ROADMAP.md) has the full plan, the training curve so far, the parts that went wrong, and how you can help.

### History

[docs/HISTORY.md](docs/HISTORY.md) is how parentheses got here: from a 64M transformer on a 4 GB GTX 1050 in 2024, through a run of small models, to the recurrent model on a laptop GPU, and the licensing mistakes I learned from along the way.

### Documentation

[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) is the reference manual. The first part covers the code: a map of the repository, every `chat.py` option, the Python API, training and evaluation scripts, and the ONNX export. The second part is the full mathematics of the model, from tokenisation through the chunked scan to constant-memory decoding.

---

## Model at a glance

| | |
|---|---|
| Model | parentheses-0.9-130M (v3, "golden") |
| Parameters | 148.38M, referred to as 130M |
| Architecture | Recurrent selective linear attention (State Space Duality) |
| Layers / width / heads | 16 / 768 / 12 heads of 64 |
| Context window | 4,096 tokens |
| Recurrent state | about 1.6 MB in total, fixed |
| Vocabulary | 32,768 tokens (`v3_32k` byte-level BPE that keeps Indic letters whole) |
| Languages | Kannada, Tamil, Telugu, Malayalam, English; experimental Tulu, Koya, Kuvi, Kolami |
| Training hardware | One NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB |
| Weights | [Hugging Face](https://huggingface.co/uncoalesced/parentheses-v0.9-130M-v3) · [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v0.9-130m-v3-golden) |
| More detail | [MODEL_CARD.md](MODEL_CARD.md) · [RELEASE_NOTES.md](RELEASE_NOTES.md) · [DATA_LEDGER.md](DATA_LEDGER.md) |

---

## Helping out

Right now it's just me. If you'd like to help, start by running the [setup](docs/SETUP.md) on your machine and [opening an issue](https://github.com/uncoalesced/parentheses/issues) for anything that breaks or reads badly. [CONTRIBUTING.md](CONTRIBUTING.md) lists the open problems and what I can accept, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) explains why the model is uncensored. For anything else, email mail@uncoalesced.com.

## Licenses

- Code and architecture: [MIT License](LICENSE).
- parentheses-0.9 weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Free for research, education and non-commercial use.
- Security reports: see [SECURITY.md](SECURITY.md).
