---
language:
- kn
- ta
- te
- ml
- en
- tcy
- kff
- kxv
- nit
license: apache-2.0
license_name: apache-2.0
tags:
- ssm
- mamba-2
- rwkv
- selective-linear-attention
- state-space-model
- recurrent
- edge-ai
- on-device
- dravidian
- kannada
- tamil
- telugu
- malayalam
model_name: parentheses-1.0-240M (Project Tanzanite)
pipeline_tag: text-generation
library_name: transformers
inference: false
---

# parentheses-1.0-240M (Project Tanzanite): Model card

parentheses-1.0-240M is a compact recurrent language model built on selective linear attention (`slab_v4`). It was grown from the 0.9-120M checkpoint and has 253,544,176 parameters. During generation it keeps a fixed-size state, so memory does not grow with the length of the text. It is meant to run on ordinary laptops, with or without a GPU.

The weights are published on Hugging Face. The code that runs them is in the parentheses-1.1 repository, which is MIT licensed.

---

## 1. Model details

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| Parameters | 253,544,176 (about 253.54M) | Trained on one 8 GB laptop GPU. |
| Layers (`n_layer`) | 16 | Uniform recurrent blocks with token-shift and SwiGLU. |
| Model dimension (`n_embd`) | 1024 | 16 heads of 64 dimensions each. |
| Heads (`n_head`) | 16 | Selective linear attention heads. |
| Head dimension (`head_dim`) | 64 | Matches the chunk size of 64 used in the recurrent kernel. |
| SwiGLU width (`n_inner`) | 2752 | No biases. |
| Context length (`block_size`) | 4,096 tokens | Training context. Quality beyond it has not been measured. |
| Vocabulary (`vocab_size`) | 32,768 | Byte-level BPE with 256 special token slots. The pre-tokenizer keeps vowel signs and viramas with their letter at word level, but BPE can still split an akshara before a vowel sign inside a word, so an akshara is not always one token. |
| Memory during inference | 4.85 MB in fp32 | 1,212,416 floats. Constant in sequence length. |
| Languages | Kannada, Tamil, Telugu, Malayalam, English | Tulu (`tcy`), Koya (`kff`), Kuvi (`kxv`) and Kolami (`nit`) are experimental. |
| License | Weights: Apache 2.0. Code: MIT | See section 7. |

---

## 2. Intended use

The model is for research, study and experiments on small recurrent models for these languages. It writes fluent-sounding text in the five main languages and handles simple code requests and short questions reasonably well.

It is not for advice on medical, legal, financial or safety matters, and it is not a source of facts. It often states wrong facts with confidence. Do not use its output for anything that matters without checking it.

---

## 3. Evaluation

The benchmark tables published with 1.0 should not be cited. They used small samples, keyword scoring and the best of four runs, and they overstated the model. [HISTORY.md](https://github.com/uncoalesced/parentheses/blob/main/HISTORY.md#the-benchmark-tables-november-2026) explains what went wrong.

Re-measurement with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), using full test sets and one greedy run, is done and under review. The numbers will be published after review. The method is in [docs/BENCHMARK_SPEC.md](https://github.com/uncoalesced/parentheses/blob/main/docs/BENCHMARK_SPEC.md).

Until then, this card reports no benchmark scores.

### Memory test (internal diagnostic): how far back can it recall?

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

Raw per-item results and the full table, including the chat and ablation runs, are in [eval/results/needle-recall-1.0/](https://github.com/uncoalesced/parentheses/blob/main/eval/results/needle-recall-1.0/summary.md).

---

## 4. How to use

### Command line (parentheses-1.1)

Clone the code, install it, and download the weights:

```bash
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
pip install -e .
hf download uncoalesced-org/parentheses-1.0-240M model.safetensors --local-dir weights
```

Start a chat:

```bash
python -m parentheses.chat --checkpoint weights --device auto
```

`--device auto` uses CUDA, then the Mac GPU, then the CPU. The program prints the device it chose. Type `/kan`, `/tam`, `/tel`, `/mal`, `/eng` or a message prefix such as `/tam` to set the language, and `/new` to start a fresh conversation.

See [docs/SETUP.md](https://github.com/uncoalesced/parentheses/blob/main/docs/SETUP.md) for GPU setup, troubleshooting and every chat option.

### Python (parentheses-1.1)

```python
import torch
from tokenizers import Tokenizer
from parentheses import TOKENIZER_PATH
from parentheses.checkpoint import load_model
from parentheses.session import RELEASE_SAMPLING, ChatSession

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("weights/model.safetensors", device=device)
tok = Tokenizer.from_file(str(TOKENIZER_PATH))

def ask(text, lang="eng", max_new_tokens=200):
    chat = ChatSession(model, tokenizer=tok, lang=lang)
    return tok.decode(chat.reply(text, max_new_tokens=max_new_tokens, **RELEASE_SAMPLING))

print(ask("Write a Python function that returns the factorial of n."))
print(ask("ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?", lang="kan"))
```

`RELEASE_SAMPLING` sets temperature 0.2, top-k 40, min-p 0.1 and a block on repeating any 8-token run. These are the settings the model was tuned with. Language codes are `kan`, `tam`, `tel`, `mal`, `eng`, `tcy`, `kff`, `kxv` and `nit`.

### Hugging Face transformers

The model also loads without the parentheses code, through `transformers`. The Hub repository ships its own modelling files, so `trust_remote_code=True` is needed. Only `torch` and `transformers` are required.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "uncoalesced-org/parentheses-1.0-240M"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()

print(model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng"))
print(model.chat(tok, "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?", lang="kan"))
```

Notes:
- The prompt format is `<|user|><lang> question <|assistant|>` with no system prompt. `build_prompt()` and `chat()` build it for you.
- `chat()` uses temperature 0.2 and an 8-gram loop block, with no repetition penalty. A repetition penalty stops the model reusing variable names in code.
- The `<think>` reasoning format is experimental and is not the tested path.

---

## 5. Weights and formats

- `model.safetensors` (float32, about 1 GB) on the [Hugging Face Hub](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M). These are the reference weights.
- `model-bf16.safetensors` (bfloat16, half the download) and `model.pt` (PyTorch state dictionary) on the [GitHub release](https://github.com/uncoalesced/parentheses/releases/tag/v1.0.0-tanzanite). The bfloat16 file has not been tested separately, so start with the float32 file.

---

## 6. Training

The model was grown from the 0.9-120M checkpoint and trained on one 8 GB laptop GPU. Training data is not redistributed. [docs/LICENSE_TABLE.md](https://github.com/uncoalesced/parentheses/blob/main/docs/LICENSE_TABLE.md) lists the licence tier each kind of data came under.

The training was single-turn: each example is one question and one answer. See the limitations below.

---

## 7. Limitations

- Chat and arithmetic are weak. The model writes fluent sentences that are often wrong, and it invents details with confidence.
- Short code requests and simple questions usually come out right. Longer programs often have mistakes.
- Longer answers, especially in Kannada, start repeating themselves after a sentence or two.
- The model was trained on single-turn chats only. Follow-up questions usually get the previous answer again, so start a new conversation for each question. The chat tools do this by default.
- The fixed-size state loses older facts as new text comes in. Exact recall of a number holds to about 256-512 tokens and is gone by 4,096 (see section 3).
- Quality beyond the 4,096-token training context has not been measured.
- Tulu, Koya, Kuvi and Kolami are experimental. They have no tested sample prompts.
- Output is sampled, so the same prompt can give different answers unless a seed is fixed.

---

## 8. Licence

- Model weights: [Apache 2.0](https://github.com/uncoalesced/parentheses/blob/main/LICENSE-MODEL)
- Code: [MIT](https://github.com/uncoalesced/parentheses/blob/main/LICENSE)
- Training data is not redistributed. See [docs/LICENSE_TABLE.md](https://github.com/uncoalesced/parentheses/blob/main/docs/LICENSE_TABLE.md).

---

## 9. Citation

```bibtex
@software{parentheses_2026,
  title   = {parentheses: a recurrent language model for Dravidian languages and English},
  author  = {uncoalesced},
  year    = {2026},
  url     = {https://github.com/uncoalesced/parentheses}
}
```

Feedback goes to the [GitHub issues](https://github.com/uncoalesced/parentheses/issues) or the [Hugging Face discussions](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M/discussions). Security reports go through the repository's SECURITY.md.
