# Setting up parentheses-1.0-240M-tanzanite

By the end of this page you will have parentheses-1.0-240M running in a terminal, ready to chat in Kannada, Tamil, Telugu, Malayalam or English. The steps take about 10 minutes, plus a download of the model weights (~500 MB). It runs on an ordinary CPU, but it is much faster on an NVIDIA GPU, so step 2 is worth doing if you have one.

## Legend

This page has two ways in. The language table takes you straight to the setup for your language. The contents list below it takes you to any section.

### Jump to your language

Every language uses the same command; you add one flag to pick the language. Do steps 1 to 3 first, then jump to your row.

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
- [3. Download the weights](#3-download-the-weights)
- [4. Start chatting](#4-start-chatting)
- [Setup by language](#setup-by-language)
  - [Kannada](#kannada)
  - [Tamil](#tamil)
  - [Telugu](#telugu)
  - [Malayalam](#malayalam)
  - [English](#english)
  - [Experimental languages](#experimental-languages)
  - [Tulu](#tulu)
  - [Koya](#koya)
  - [Kuvi](#kuvi)
  - [Kolami](#kolami)
- [Modes](#modes)
- [Commands inside the chat](#commands-inside-the-chat)
- [All chat.py options](#all-chatpy-options)
- [Using it from Python](#using-it-from-python)
- [Troubleshooting](#troubleshooting)
- [What to expect](#what-to-expect)

## Before you start

You need a few things on your machine before you begin.

- Python 3.10 or newer. We test on Python 3.11, so that version is the safest choice.
- Git, so you can download the code.
- Free disk space: about 1.5 GB for the CPU route, or about 6 GB if you install a CUDA build of PyTorch in step 2. The weights themselves are about 967 MB (float32) or 484 MB (bfloat16).
- A GPU is optional. The model is small, and a graphics card with 8 GB of memory is more than enough for chat. We develop on an RTX 5050 Laptop GPU with 8 GB.

Open a terminal in a folder where you want the project to live. On Windows, Windows Terminal is a better choice than the older console window, because it shows Indic letters more reliably.

## 1. Get the code

Clone the repository from GitHub. This creates a folder called `parentheses`.

```bash
git clone https://github.com/uncoalesced/parentheses.git
```

Move into that folder. Every command from here on assumes you are in it.

```bash
cd parentheses
```

A virtual environment keeps the project's Python packages separate from the rest of your system. It is optional, but it avoids version clashes later. On Windows, create one like this.

```bash
python -m venv .venv
```

Then activate it. On Windows, use this command.

```bash
.venv\Scripts\activate
```

On macOS or Linux, use this command instead.

```bash
source .venv/bin/activate
```

When the environment is active, your prompt starts with `(.venv)`. If PowerShell refuses to run the activation script, run the same command from Command Prompt instead.

Install the libraries the project needs.

```bash
pip install -r requirements.txt
```

This takes a few minutes. On Windows it installs a CPU-only build of PyTorch, which works but is slow for chat. Step 2 shows how to swap in a GPU build.

If you cloned the repository some time ago, run `git pull` inside the folder before you go on, so you have the latest `scripts/chat.py`. The troubleshooting section explains why this matters.

```bash
git pull
```

## 2. Make PyTorch see your GPU

This step is optional, but if you have an NVIDIA graphics card, it is the difference between slow and fast output. Skip it if you have no NVIDIA GPU, and use the CPU commands in step 4.

The trap is this. `pip install torch`, which `requirements.txt` runs, usually installs a CPU-only build of PyTorch on Windows. If you then ask the chat program to use the GPU, you get this error.

```text
AssertionError: Torch not compiled with CUDA enabled
```

To fix it, open the PyTorch install selector at https://pytorch.org/get-started/locally/. Choose your operating system, choose Pip, and choose a CUDA version. The page then shows you an install command, and you should run that command exactly as it appears. Do not reuse a command from an older guide, because the page changes over time.

If you have an RTX 50-series card, pick a CUDA version of 12.8 or newer, because older builds do not support the Blackwell architecture. We develop on a PyTorch build for CUDA 12.8.

If PyTorch is already installed as a CPU build, the new command may not replace it. Uninstall the old copy first.

```bash
pip uninstall -y torch
```

Then run the install command the selector gave you.

Check that PyTorch can now see the GPU.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The command should print `True`. If it prints `False`, PyTorch cannot see the GPU. Keep going with the CPU commands for now, and come back to this step later.

### On a Mac with Apple Silicon (M1, M2, M3, M4)

Apple Silicon has no NVIDIA GPU, so `--device cuda` does not apply. The chat program can use the Apple GPU through PyTorch's Metal backend, called `mps`. Use a recent macOS and the standard PyTorch from `pip install -r requirements.txt`. Do not use `requirements-lock.txt`, because it is a Windows CUDA freeze.

Check that the Metal backend is visible.

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

If it prints `True`, run the chat with `--device mps`, or `--device auto`, which picks CUDA first, then MPS, then CPU.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device mps
```

If it prints `False`, you are on an Intel Mac or an older PyTorch. Use `--device cpu`. The Mac build runs in float32. Speed on MPS has not been measured yet for this model, so compare it against `--device cpu` on your own machine.

## 3. Download the weights

The GitHub repository holds the code, but not the model weights. You get the weights from the Hugging Face model page at https://huggingface.co/uncoalesced-org/parentheses-1.0-240M. You need three files.

- `config.json` holds the model's settings.
- `model.safetensors` holds the weights (about 967 MB in float32).
- `tokenizer.json` tells the program how to split text into tokens.

The same page also lists `tokenizer_config.json` and a README. You do not need either of them.

There are two ways to get the files.

Option A is the browser route. Open the model page, click the Files and versions tab, and download the three files. Create a folder called `release/parentheses-1.0-240M-tanzanite` inside the repository folder, and save the three files into it.

Option B is the command line route. The `hf` command comes with the Hugging Face Hub library, which `pip install -r requirements.txt` installed for you. It creates the folder and downloads the files in one step.

```bash
hf download uncoalesced-org/parentheses-1.0-240M config.json model.safetensors tokenizer.json tokenizer_config.json --local-dir release/parentheses-1.0-240M-tanzanite
```

When you are done, the folder should look like this.

```text
parentheses/
  release/
    parentheses-1.0-240M-tanzanite/
      config.json
      model.safetensors
      tokenizer.json
  scripts/chat.py
  requirements.txt
```

The folder name matters. The `--checkpoint` flag in the next step has to point at this folder exactly. If you choose a different name, change the flag to match.

## 4. Start chatting

Stay in the repository folder, the one that contains `scripts` and `requirements.txt`. Run this command to start the chat with the GPU.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda
```

If you skipped step 2, or PyTorch cannot see your GPU, leave out `--device cuda`. The program then runs on the CPU. It works, but it is much slower.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite
```

The program first loads the model, which can take a moment, and then prints a short banner. The first line of the banner names the checkpoint folder, the parameter count (253.54 million), the attention type, the session language, the mode, and the device. You can ignore the step number in it. Check the device in that line. If it says `cpu` and you meant to use the GPU, you left out `--device cuda`. The next lines list the languages the program can route to, say whether the built-in system prompt is on, and list the commands you can use inside the chat. After that you see a green `>` prompt.

Type a prompt and press Enter. Before the answer appears, a dim blue line shows where the program sent your prompt, for example:

```text
[Routing: kan (ಕನ್ನಡ (Kannada))]
```

This line is the routing line. It shows the language the program chose for that prompt. If you typed English but the routing line says a different language, the session is probably forced with `--lang`.

The answer then streams in one piece at a time, in bold. For a quick check, try an English prompt such as `Write a Python function that returns the factorial of n.` Short functions like this usually come out correct; longer programs often have mistakes. Do not expect the exact same text every time, because the program samples its output.

Simple arithmetic is handled differently. A question such as `What is 17 x 6?` is answered by a built-in calculator, not by the model, so the answer is exact.

On the laptop GPU we use (RTX 5050 Laptop), we measured 11 to 34 tokens per second, typically 20 to 27, so replies appear at a steady pace.

To quit, press Enter on an empty line, or press Ctrl+C. An empty line quits straight away, so a stray Enter ends the session. If that happens, run the same command again.

## Setup by language

Each main language uses the same program with a different `--lang` value. The default is `auto`, which picks the language for each prompt from the script you type. You can leave `--lang` out, or pass `--lang auto`, if you mix languages in one session. Forcing a language keeps every prompt in that language, which is safer when you want a specific language.

You can also force a language for a single message by typing a code such as `/tam` at the start of it, followed by your text. That is covered in the commands section.

### Kannada

The full command for Kannada is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang kan
```

You can use auto instead. Leave out `--lang kan`, or pass `--lang auto`. Kannada script is detected automatically, so auto picks Kannada for Kannada text. Forcing `--lang kan` keeps every prompt in Kannada, including English words, so use the forced form when you want Kannada and auto when you mix languages. With `--lang kan`, typing `hi` is routed to Kannada and usually returns junk. Type `/eng hi` to send one message as English, or `/auto` to go back to detection.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang kan
```

Think mode (`--format think`) is experimental on this checkpoint. It returns a templated restatement of the question rather than useful reasoning, so stay with the default chat mode.

An example prompt is ನಿನ್ನ ಹೆಸರು ಏನು? which means "what is your name?"

Expect limited quality. The Kannada prompt above often comes back echoed, or answered in English.

### Tamil

The full command for Tamil is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang tam
```

Auto works for Tamil too, because Tamil script is detected directly. Use the forced form if you want every prompt kept in Tamil.

An example prompt is உங்கள் பெயர் என்ன? which means "what is your name?"

Expect Tamil that predicts well on held-out text, with a bits-per-byte score of 0.45, but answers that may still be wrong.

### Telugu

The full command for Telugu is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang tel
```

Don't rely on auto for Telugu. Auto often sends Telugu to Kolami by mistake, because two of its Kolami marker words (`అని` and `మరి`) are everyday Telugu words. Force `--lang tel`, and check that the routing line says `tel`.

An example prompt is మీ పేరు ఏమిటి? which means "what is your name?"

Expect Telugu script in the output, and answers that can be wrong.

### Malayalam

The full command for Malayalam is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang mal
```

Auto works for Malayalam, because Malayalam script is detected directly. Use the forced form if you want every prompt kept in Malayalam.

An example prompt is നിങ്ങളുടെ പേര് എന്താണ്? which means "what is your name?"

Expect the best predictions of the nine languages on held-out text, with a bits-per-byte score of 0.44, but check answers before you trust them.

### English

The full command for English is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang eng
```

Auto already picks English for text that has no Indic script, so the default works. Use the forced form if you want to be certain.

An example prompt is `Write a short Python function that adds two numbers.`

Expect fluent-sounding English. Short code attempts often come out partly wrong, and factual answers can be wrong too, stated with full confidence.

### Experimental languages

Four more languages are included as experimental. Their held-out scores, measured in bits per byte where lower is better, run from 1.09 to 1.57. The validation sets for Koya, Kuvi and Kolami hold only 12 documents each, so those numbers are rough. Across all nine languages, about 74 percent of outputs kept the right script (53 of 72).

Auto-detection for these four relies on a few marker words. Without a marker word, the program can send a prompt to a related language. For that reason, set `--lang` yourself for these languages. You can also force one message with a code such as `/tcy`, `/kff`, `/kxv` or `/nit` followed by your text.

This guide has no tested sample prompt for these four languages, so type a short question of your own in the right script.

### Tulu

The full command for Tulu is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang tcy
```

Tulu is written here in the Kannada script. Auto sends Kannada-script text to Kannada unless one of the Tulu marker words appears, so force `--lang tcy` for Tulu.

Expect rough output. Tulu scores 1.09 bits per byte on held-out text, and it is included in the roughly 74 percent script figure above.

### Koya

The full command for Koya is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang kff
```

Koya is written here in the Odia script. Auto picks Koya for Odia-script text unless Kuvi marker words appear, so force `--lang kff` to be safe.

Expect rough output. Koya scores 1.55 bits per byte, and its validation set holds only 12 documents.

### Kuvi

The full command for Kuvi is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang kxv
```

Kuvi is also written in the Odia script. Auto picks Kuvi only when Kuvi marker words appear, and otherwise sends Odia-script text to Koya. Force `--lang kxv` for Kuvi.

Expect rough output. Kuvi scores 1.57 bits per byte, and its validation set holds only 12 documents.

### Kolami

The full command for Kolami is this.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda --lang nit
```

Kolami is written here in the Telugu script. Auto picks Kolami when it sees certain marker words, but two of those (`అని` and `మరి`) are also everyday Telugu, so auto confuses the two in both directions. Force `--lang nit` for Kolami.

The startup banner calls this language "Gondi", and the routing line says "SE Gondi". Both labels refer to Kolami, which is the name this project uses.

Expect rough output. Kolami scores 1.42 bits per byte, and its validation set holds only 12 documents.

## Modes

The `--format` flag sets the mode for the whole session. You can also switch the mode for one message with `/think`, `/chat`, `/freethink` or `/base`, as described in the commands section.

Chat is the default, and the best place to start. It wraps your message in a chat format and asks the model for a direct answer.

Think mode asks the model to write a reasoning block inside think tags before the answer. The block appears dimmed in cyan, and the answer follows in bold. The reasoning is not reliable. In test runs the model opened a think block in 36 of 72 cases, and it closed the block by itself in only 4 of those 36. Use think mode to see what the model tries to do, not to trust its reasoning.

Freethink mode runs an open-ended stream. Instead of stopping after one answer, the model keeps writing from your prompt, and it can pull in retrieved text as it goes. Treat it as an experiment.

Base mode gives the model your text, with only a language tag in front, and no chat template around it. The model simply continues the text. This mode shows what the model writes without the chat wrapper.

## Commands inside the chat

Commands come in two kinds, and the difference matters. A session command changes the setting for every message until you change it again. A per-message command applies only to the message it is typed in, and the next message goes back to the session setting.

A per-message command must come with text. If you type `/tam` or `/think` alone on a line, the program does not switch anything. It sends the bare word to the model as ordinary text, and the reply is noise. Type the command and your question together, for example `/tam what is your name?`.

| Command | Scope | What it does | Example |
|---|---|---|---|
| `/lang <code>` | Session | Sets the language for every message until you change it. | `/lang tam` |
| `/lang auto` or `/auto` | Session | Returns to automatic detection. | `/auto` |
| `/kan`, `/tam`, `/tel`, `/mal`, `/eng`, `/tcy`, `/kff`, `/kxv`, `/nit` | One message | Forces that language for this message only. Needs text after it. | `/tam what is your name?` |
| `/think`, `/chat`, `/freethink`, `/base` | One message | Uses that mode for this message only. Needs text after it. | `/think why is the sky blue?` |
| `/image <path>` | Session | Attaches an image, which stays in the session for later messages. Works on its own. | `/image photo.png` |
| `/doc <path>` | Session | Attaches a document, which stays in the session for later messages. Works on its own. | `/doc notes.txt` |
| `/web <query>` | One message | Searches the web for this message only. | `/web latest news` |
| `/url <url>` | One message | Fetches a page for this message only. | `/url https://example.com` |

To change the mode for the whole session, restart the program with `--format` set to the mode you want.

## All chat.py options

Every option below is a flag you can add to the `python scripts/chat.py` command.

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint PATH` | `release/parentheses-1.0-240M-tanzanite` | The folder with `config.json`, `model.safetensors` and `tokenizer.json`. Always pass this flag. |
| `--device` | `cpu` | Where the model runs. Use `cuda` on an NVIDIA GPU. `mps` is untested. |
| `--lang` | `auto` | The session language: `auto`, `eng`, `kan`, `tam`, `tel`, `mal`, `tcy`, `kff`, `kxv` or `nit`. |
| `--format` (alias `--mode`) | `chat` | The mode: `chat`, `think`, `freethink` or `base`. |
| `--max-new-tokens` | `300` | The most tokens the model writes for one reply. |
| `--temperature` | `0.2` for 1.0-240M, `0.7` for 0.9 checkpoints | Higher values make the output more varied and less predictable. `0` is greedy. |
| `--top-k` | `40` | At each step, the model picks only from the 40 most likely next tokens. |
| `--system` | Off for 1.0-240M, on for 0.9 checkpoints | Turns on the identity and honesty system prompt. Bare `--system` uses the built-in text, and `--system "text"` uses your own. The 1.0-240M release was tuned without a system prompt, and the 0.9 prompt makes its answers worse. |
| `--no-system` | Off | Turns the system prompt off. |
| `--image PATH [PATH ...]` | None | Attaches images at startup. Accepts `.png`, `.jpg`, `.bmp` and `.webp`. |
| `--doc PATH [PATH ...]` (alias `--ingest`) | None | Attaches documents at startup. Accepts `.pdf`, `.docx`, `.html`, `.csv`, `.json` and `.txt`. |
| `--web` (alias `--web-search`) | Off | Allows web grounding for clearly real-time questions. `/web` always searches. |
| `--self-test` | Off | Runs the built-in checks and exits. |

## Using it from Python

If you want to call the model from your own script, run the script from the repository folder, so that `scripts` can be imported. The simplest route uses the same loader and reply function as the chat program.

```python
from scripts.chat import load_model_and_tokenizer, reply

model, tok, cfg, _ = load_model_and_tokenizer("release/parentheses-1.0-240M-tanzanite", device="cuda")
answer = reply(
    model,
    "Write a Python function that returns the factorial of n.",
    "cuda",
    200,
    0.2,
    40,
    tok=tok,
    lang="auto",
    fmt="chat",
    stream=False,
    system=None,
)
print(answer)
```

If you have no NVIDIA GPU, or PyTorch can't see it, change both `"cuda"` values to `"cpu"`. The `reply` function returns the answer as a string. These are the settings the chat program uses for 1.0-240M: no system prompt and temperature 0.2. `reply` picks the matching repetition settings for the model it is given.

If you only need inference, the Hugging Face route needs no repository code at all:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "uncoalesced-org/parentheses-1.0-240M"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
print(model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng"))
```

If you prefer to build the model yourself, the lower-level route works too. It loads the weights into the model class directly.

```python
import safetensors.torch
from tokenizers import Tokenizer

from model.backbone import Parentheses
from model.config import PRESETS

model = Parentheses(PRESETS["parentheses-1.0-240M"])
sd = safetensors.torch.load_file("release/parentheses-1.0-240M-tanzanite/model.safetensors")
model.load_state_dict(sd, strict=False)
model.eval()
tok = Tokenizer.from_file("release/parentheses-1.0-240M-tanzanite/tokenizer.json")
```

The output head is tied to the embedding inside `Parentheses`, so no extra step is needed for that. Be careful with `strict=False`, though. It hides any tensor that fails to load, and that is the bug described in the troubleshooting section. The helper function above prints a warning in that case, so prefer it unless you need the lower-level control.

## Troubleshooting

### The output is slow

The model runs on the CPU, because the command has no `--device cuda`. Add `--device cuda` to the command. If PyTorch then fails with the assertion error below, go back to step 2 and install a CUDA build.

### AssertionError: Torch not compiled with CUDA enabled

Your PyTorch is a CPU-only build. Follow step 2 to uninstall it and install a CUDA build from the PyTorch selector. Then run the one-line check in step 2, and it should print `True`.

### No module named 'safetensors'

The program needs the `safetensors` package. Install it with this command.

```bash
pip install safetensors
```

The package is also listed in `requirements.txt`, so running `pip install -r requirements.txt` again fixes it too. If you see `No module named torch` or `No module named tokenizers` instead, the virtual environment is probably not active. Activate it again and retry.

### The output is gibberish or a stream of numbers

Symptoms include number soup, random symbols, or text that never settles into words. Work through these causes in order.

First, make sure you have the latest code. Earlier versions of `scripts/chat.py` ignored the `slab_v3` setting in `config.json` when they loaded the safetensors folder. They silently dropped 63 weight tensors, and the result was exactly this kind of output. Run `git pull` in the repository folder, and then start the chat again.

Second, look at the startup output for a line that starts with `[chat] WARNING:` and says that some checkpoint tensors were not loaded. If you see that line, the weights did not load properly, and the output will be degraded. Pull the latest code first. If the warning still appears after that, the files in your checkpoint folder do not match the config.

Third, check the download. `model.safetensors` should be about 967 MB. A partial download can give odd output or fail to load. Download the three files again, and check that all of them are present in the folder.

### The answer comes back in the wrong language

This usually happens because the session is forced with `--lang`, or with `/lang`. Every prompt then goes to that language, including English. Type `/auto` in the chat to return to detection, or restart without `--lang`. Check the routing line after each prompt to see where it went.

The model also sometimes answers a Kannada prompt in English, even when the routing is right. That comes from the model, not from routing, and it is one reason the Kannada expectation above is limited.

### Boxes appear instead of Indic letters

The terminal font does not have the glyphs for those letters. Use Windows Terminal, and choose a font that covers Indic scripts, such as Nirmala UI. The chat program already switches its output to UTF-8 on Windows, so the font is almost always the issue.

If you send the output to a file, set the environment variable `PYTHONIOENCODING` to `utf-8` before you run the command. On macOS or Linux, use this.

```bash
export PYTHONIOENCODING=utf-8
```

On Windows Command Prompt, use `set PYTHONIOENCODING=utf-8` instead.

### FileNotFoundError that names config.json

The `--checkpoint` path points at a folder that exists, but the folder does not contain `config.json`. This often happens when the files landed one folder deeper than expected, for example `parentheses-1.0-240M-tanzanite/parentheses-1.0-240M-tanzanite/config.json`. Open the folder and check that the three files sit directly inside it. Run the command from the repository folder, so the relative path resolves.

### FileNotFoundError that names step_12035_final.pt

You left out `--checkpoint`, so the program fell back to its default, an older checkpoint path that is not part of this release. Add the flag.

```bash
python scripts/chat.py --checkpoint release/parentheses-1.0-240M-tanzanite --device cuda
```

## What to expect

parentheses-1.0-240M (tanzanite) is a 253.54 million parameter recurrent model. It is not a transformer. It writes the scripts of the five main languages reliably. In testing, 40 of 40 outputs in those languages kept their script. Its chat and mathematics are weak. It writes fluent sentences that are often wrong, and it invents details with confidence. In one test it described a Martian colony founded in 1842, which never happened. On a fresh honest-reasoning suite of 24 prompts it scored 5 out of 24. Kannada conversation and multi-step arithmetic also degrade. Simple arithmetic is handled by the built-in calculator, so those answers are exact. Do not rely on its answers for anything that matters.

The project's direction and plans are in [ROADMAP.md](ROADMAP.md). The design of the model is explained in [ARCHITECTURE.md](ARCHITECTURE.md). For the project overview, see [the README](../README.md).

If setup fails on a machine we have not tried, the most useful report is your operating system, your GPU, and the full error text. We have run this on an RTX 5050 GPU, so reports from other systems help us the most.
