# Setting up parentheses

By the end of this page you will have parentheses-1.0-240M running in a terminal, ready to chat in Kannada, Tamil, Telugu, Malayalam or English. It takes about 10 minutes plus a download of about 1 GB of weights. It runs on an ordinary CPU, but it is much faster on an NVIDIA GPU, so step 2 is worth doing if you have one.

## Legend

This page has two ways in. The language table takes you straight to the setup for your language. The contents list below it takes you to any section.

### Jump to your language

Every language uses the same command; one flag picks the language. Do steps 1 to 3 first, then jump to your row.

| Language | Code | Script | Flag to add | Status |
|---|---|---|---|---|
| [Kannada](#kannada) | kan | Kannada | `--lang kan` | main |
| [Tamil](#tamil) | tam | Tamil | `--lang tam` | main |
| [Telugu](#telugu) | tel | Telugu | `--lang tel` | main |
| [Malayalam](#malayalam) | mal | Malayalam | `--lang mal` | main |
| [English](#english) | eng | Latin | `--lang eng` | main |
| [Tulu](#tulu) | tcy | Kannada | `--lang tcy` | experimental |
| [Koya](#koya) | kff | Odia | `--lang kff` | experimental |
| [Kuvi](#kuvi) | kxv | Odia | `--lang kxv` | experimental |
| [Kolami](#kolami) | nit | Telugu | `--lang nit` | experimental |

### Contents

- [Before you start](#before-you-start)
- [1. Get the code](#1-get-the-code)
- [2. Make PyTorch see your GPU](#2-make-pytorch-see-your-gpu)
  - [On a Mac with Apple Silicon](#on-a-mac-with-apple-silicon-m1-m2-m3-m4)
- [3. Download the weights](#3-download-the-weights)
- [4. Start chatting](#4-start-chatting)
- [Setup by language](#setup-by-language)
- [Modes](#modes)
- [Commands inside the chat](#commands-inside-the-chat)
- [All chat options](#all-chat-options)
- [Using it from Python](#using-it-from-python)
- [Troubleshooting](#troubleshooting)
- [What to expect](#what-to-expect)

## Before you start

- Python 3.10 or newer. We test on Python 3.11, so that version is the safest choice.
- Git, so you can download the code.
- Free disk space: about 2 GB for the CPU route, or about 6 GB if you install a CUDA build of PyTorch in step 2. The float32 weights are about 1 GB.
- A GPU is optional. A graphics card with 8 GB of memory is more than enough for chat. We develop on an RTX 5050 Laptop GPU with 8 GB.

Open a terminal in the folder where you want the project to live. On Windows, Windows Terminal shows Indic letters more reliably than the older console window.

## 1. Get the code

Clone the repository and move into it. Every command from here on assumes you are in this folder.

```bash
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
```

A virtual environment keeps the project's packages away from the rest of your system. It's optional, but it avoids version clashes later.

```bash
python -m venv .venv
```

Activate it. On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. If PowerShell refuses to run the activation script, run the same command from Command Prompt instead.

Install the package and its three dependencies (PyTorch, safetensors and tokenizers):

```bash
pip install -e .
```

On Windows this usually installs a CPU-only build of PyTorch. It works, but it is slow for chat. Step 2 shows how to swap in a GPU build.

If you cloned some time ago, run `git pull` before going on.

## 2. Make PyTorch see your GPU

This step is optional, but on an NVIDIA card it is the difference between slow and fast output. Skip it if you have no NVIDIA GPU.

The trap is that `pip install torch` usually gives you a CPU-only build on Windows. If you then ask for the GPU, you get this error:

```text
AssertionError: Torch not compiled with CUDA enabled
```

To fix it, open the PyTorch install selector at https://pytorch.org/get-started/locally/, pick your operating system, Pip and a CUDA version, and run the command it shows you exactly as written. The page changes over time, so don't reuse a command from an older guide. On an RTX 50-series card, pick CUDA 12.8 or newer; older builds don't support those cards.

If a CPU build is already installed, the new command may not replace it. Uninstall it first, then run the selector's command:

```bash
pip uninstall -y torch
```

Check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True`. If it prints `False`, carry on with the CPU for now and come back to this step later.

### On a Mac with Apple Silicon (M1, M2, M3, M4)

Status: untested on real Mac hardware. The code is plain PyTorch and should run on the Mac GPU through PyTorch's Metal backend (`mps`), but nobody has confirmed it on a real Mac yet. The CPU always works. The short version is below; [APPLE_SILICON.md](APPLE_SILICON.md) has the full walk-through and what to send back.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

The last line lets the few operations Metal doesn't support run on the CPU instead of stopping with an error. After step 3, run the self-test, which compares the Mac GPU against the CPU and ends with `PASS` or `FAIL`:

```bash
python -m parentheses.selftest --checkpoint weights
```

## 3. Download the weights

The repository holds the code, not the weights. The weights are on Hugging Face at https://huggingface.co/uncoalesced-org/parentheses-1.0-240M. You need one file, `model.safetensors` (about 1 GB, float32). The tokenizer is already in this repository.

Option A, the command line. The `hf` command comes with the `huggingface_hub` package; install it with `pip install huggingface_hub` if it's missing.

```bash
hf download uncoalesced-org/parentheses-1.0-240M model.safetensors --local-dir weights
```

Option B, the browser. Open the model page, click Files and versions, download `model.safetensors`, and save it into a folder called `weights` inside the repository folder.

Either way you should end up with:

```text
parentheses/
  weights/
    model.safetensors
  python/parentheses/
  pyproject.toml
```

A bfloat16 copy (half the download) and a PyTorch `.pt` file are on the [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite). The bfloat16 file has not been tested separately, so start with the float32 one.

## 4. Start chatting

From the repository folder:

```bash
python -m parentheses.chat --checkpoint weights --device auto
```

`--device auto` uses CUDA if PyTorch can see an NVIDIA GPU, then the Mac GPU, then the CPU. Use `--device cpu` to force the CPU.

The program loads the model and prints one line naming the checkpoint, the parameter count (253.54M), the device, the language setting and the format. Check the device. If it says `cpu` and you expected the GPU, go back to step 2.

Type a message and press Enter. Before the answer, a short line shows which language the message was sent as, for example `[kan: Kannada]`. The answer then streams in.

For a quick check, try `Write a Python function that returns the factorial of n.` Short functions like this usually come out right; longer programs often have mistakes. The output is sampled, so you won't get the same text every time unless you pass `--seed`.

To quit, press Enter on an empty line, or Ctrl+C. A stray empty line ends the session, so if that happens just run the command again.

## Setup by language

Every language uses the same command with a different `--lang`. The default, `auto`, picks the language for each message from the script you type: Kannada, Tamil, Telugu and Malayalam script are recognised, and everything else goes to English. Forcing a language keeps every message in it, which is safer when you want one language only. You can also force a single message by starting it with a code, such as `/tam` followed by your text.

### Kannada

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang kan
```

Auto recognises Kannada script, so you can leave `--lang` out. Forcing `--lang kan` sends every message as Kannada, including English ones, and an English `hi` forced to Kannada usually returns junk. Type `/eng hi` to send one message as English, or `/auto` to go back to detection.

Example prompt: ನಿನ್ನ ಹೆಸರು ಏನು? ("what is your name?")

Expect limited quality. Answers stay in Kannada script, but are short, sometimes drift into English, and longer ones repeat themselves.

### Tamil

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang tam
```

Auto recognises Tamil script. Force it if you want every message kept in Tamil.

Example prompt: உங்கள் பெயர் என்ன? ("what is your name?")

Expect Tamil script, and answers that can still be wrong.

### Telugu

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang tel
```

Auto recognises Telugu script. Kolami is written in the same script, so auto never picks Kolami; force `--lang nit` for that.

Example prompt: మీ పేరు ఏమిటి? ("what is your name?")

Expect Telugu script, and answers that can be wrong.

### Malayalam

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang mal
```

Auto recognises Malayalam script.

Example prompt: നിങ്ങളുടെ പേര് എന്താണ്? ("what is your name?")

Expect Malayalam script. Check answers before you trust them.

### English

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang eng
```

Auto already sends anything without Indic script to English.

Example prompt: `Write a short Python function that adds two numbers.`

Expect fluent-sounding English. Short code often comes out right; facts can be wrong and stated with full confidence.

### Experimental languages

Four more languages are included as experimental. They share scripts with the main languages, so auto never picks them. Always set `--lang`, or start a message with `/tcy`, `/kff`, `/kxv` or `/nit`. There is no tested sample prompt for these four; type a short question of your own in the right script and expect rough output.

### Tulu

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang tcy
```

Tulu is written here in Kannada script.

### Koya

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang kff
```

Koya is written here in Odia script.

### Kuvi

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang kxv
```

Kuvi is also written in Odia script.

### Kolami

```bash
python -m parentheses.chat --checkpoint weights --device auto --lang nit
```

Kolami is written here in Telugu script.

## Modes

`--format chat` is the default. It wraps your message in the chat template the model was tuned on and asks for a direct answer.

`--format base` gives the model your text with only a language tag in front, and the model continues it. Use it to see what the model writes without the chat wrapper.

By default each message starts a fresh conversation, because 1.0 was trained on one question and one answer at a time. `--keep-history` carries the conversation from message to message without re-reading it, but 1.0 usually answers a follow-up by repeating its previous answer. `/new` clears the conversation.

## Commands inside the chat

A session command changes the setting for every message until you change it again. A per-message command applies only to the message it is typed in.

| Command | Scope | What it does | Example |
|---|---|---|---|
| `/lang <code>` | Session | Uses that language for every message. | `/lang tam` |
| `/lang auto` or `/auto` | Session | Goes back to picking the language from the script. | `/auto` |
| `/kan`, `/tam`, `/tel`, `/mal`, `/eng`, `/tcy`, `/kff`, `/kxv`, `/nit` | One message | Sends this message in that language. Needs text after it. | `/tam what is your name?` |
| `/new` | Session | Starts a fresh conversation (matters with `--keep-history`). | `/new` |

## All chat options

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint PATH` | `weights/model.safetensors` | The weights file, or the folder that holds `model.safetensors`. |
| `--device` | `auto` | `auto`, `cuda`, `mps` or `cpu`. |
| `--lang` | `auto` | `auto`, `eng`, `kan`, `tam`, `tel`, `mal`, `tcy`, `kff`, `kxv` or `nit`. |
| `--format` | `chat` | `chat` or `base`. |
| `--keep-history` | off | Carry the conversation between messages. |
| `--max-new-tokens` | `300` | The most tokens in one reply. |
| `--temperature` | `0.2` | Higher is more varied. `0` is greedy. |
| `--top-k` | `40` | Sample only from the 40 most likely next tokens. |
| `--min-p` | `0.1` | Drop tokens less than a tenth as likely as the best one. |
| `--no-repeat-ngram` | `8` | Never repeat the same run of 8 tokens within a reply. `0` turns it off. |
| `--seed` | none | Fix the random seed so runs repeat exactly. |

The temperature, top-k, min-p and n-gram defaults are the settings 1.0 was tuned with.

## Using it from Python

```python
import torch
from tokenizers import Tokenizer
from parentheses import TOKENIZER_PATH
from parentheses.checkpoint import load_model
from parentheses.session import RELEASE_SAMPLING, ChatSession

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("weights/model.safetensors", device=device)
tok = Tokenizer.from_file(str(TOKENIZER_PATH))

chat = ChatSession(model, tokenizer=tok, lang="eng")
ids = chat.reply("Write a Python function that returns the factorial of n.",
                 max_new_tokens=200, **RELEASE_SAMPLING)
print(tok.decode(ids))
```

`reply` returns token ids. Pass `on_token=` to `generate` if you want to stream them. If you only need inference and don't want this repository, the Hugging Face route needs no code from here at all:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "uncoalesced-org/parentheses-1.0-240M"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
print(model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng"))
```

## Troubleshooting

### The output is slow

The model is probably on the CPU. Check the device in the first line the program prints. Pass `--device cuda`; if PyTorch then fails with the assertion error below, go back to step 2.

### AssertionError: Torch not compiled with CUDA enabled

Your PyTorch is a CPU-only build. Follow step 2 to uninstall it and install a CUDA build, then run the check in step 2; it should print `True`.

### No module named 'parentheses', 'torch' or 'tokenizers'

The virtual environment is probably not active, or `pip install -e .` was run somewhere else. Activate the environment and run `pip install -e .` from the repository folder again.

### checkpoint not found

`--checkpoint` points at the wrong place. It should be `weights/model.safetensors` or the `weights` folder. Check that the file landed directly inside `weights` and not one folder deeper, and run the command from the repository folder so the relative path works.

### The output is gibberish or a stream of numbers

First check the download: `model.safetensors` should be about 1,014 MB. A partial download can load and still give nonsense. Download it again. If the file is complete, run `python -m parentheses.selftest --checkpoint weights` and open an issue with its report.

### The answer comes back in the wrong language

The session is probably forced with `--lang` or `/lang`, so every message goes to that language. Type `/auto`, or restart without `--lang`. The line before each answer shows which language was used. The model also sometimes answers a Kannada question in English even when the language is right; that comes from the model, not the routing.

### Boxes appear instead of Indic letters

The terminal font has no glyphs for those letters. Use Windows Terminal with a font that covers Indic scripts, such as Nirmala UI. The program already writes UTF-8, so the font is almost always the problem. If you redirect output to a file, set `PYTHONIOENCODING=utf-8` first (`export` on macOS and Linux, `set` in Windows Command Prompt).

## What to expect

parentheses-1.0-240M is a 253.54 million parameter recurrent model, not a transformer. It writes the scripts of the five main languages reliably. Its chat and maths are weak: it writes fluent sentences that are often wrong, and it invents details with confidence. Longer answers, in Kannada especially, start repeating themselves. Follow-up questions don't work well yet. Don't rely on its answers for anything that matters.

How the model works is in [ARCHITECTURE.md](ARCHITECTURE.md), and the full technical reference is [DOCUMENTATION.md](DOCUMENTATION.md). If setup fails on a machine we haven't tried, the most useful report is your operating system, your GPU and the full error text. We have only run this on an RTX 5050, so reports from other systems help most.
