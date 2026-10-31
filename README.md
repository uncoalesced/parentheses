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
  <b>A from-scratch recurrent language model built with selective linear attention (State Space Duality / Mamba-2 / RWKV family), trained for Dravidian languages and English on a single laptop GPU.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Release-v0.9--130M--v3--golden-blue.svg" alt="Release">
  <img src="https://img.shields.io/badge/HuggingFace-parentheses--v0.9--130M--v3-ffd21e.svg" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Architecture-Recurrent_SSD-9cf.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/Hardware-1x_RTX_5050_Laptop_(8_GB)-green.svg" alt="Hardware">
  <img src="https://img.shields.io/badge/Code_License-MIT-brightgreen.svg" alt="Code License">
  <img src="https://img.shields.io/badge/Weights_License-CC_BY--NC_4.0-orange.svg" alt="Weights License">
</p>

<p align="center">
  <a href="https://github.com/uncoalesced/parentheses/releases/tag/v0.9-130m-v3-golden">GitHub release</a> •
  <a href="https://huggingface.co/uncoalesced/parentheses-v0.9-130M-v3">HuggingFace</a> •
  <a href="TRAINING_DATA.md">Training data ledger</a> •
  <a href="#how-it-works">How it works</a> •
  <a href="#evaluations-and-what-is-still-weak">Audit and evals</a>
</p>

---

parentheses-0.9-130M is a recurrent language model trained from scratch on one 8 GB laptop GPU (NVIDIA GeForce RTX 5050 Laptop). It belongs to the State Space Duality (SSD / Mamba-2 / RWKV) family rather than standard transformers.

Because it is recurrent, memory consumption per token stays flat during generation. Each layer updates a fixed-size 64 x 64 memory matrix per head, requiring zero growing KV-cache. The model is built for Kannada, four other Dravidian languages, and English, using an Akshara-preserving tokenizer that keeps Indic letters whole.

We are based in Bengaluru, so Kannada came first. Out of the box, the model introduces itself in Kannada:
```text
ನನ್ನ ಹೆಸರು ಪ್ಯಾರೆಂಥಿಸಿಸ್ 0.9 (Parentheses 0.9).
```

### Release links

- GitHub release: https://github.com/uncoalesced/parentheses/releases/tag/v0.9-130m-v3-golden
- HuggingFace: https://huggingface.co/uncoalesced/parentheses-v0.9-130M-v3

---

## How it works

A standard transformer looks back across every token it has generated each time it predicts the next one. It saves those earlier keys and values in a KV cache that grows with every token, which means long generations eat more memory and run slower.

parentheses works differently:

- Each layer keeps a fixed-size memory state: a 64 x 64 matrix per head (12 heads per layer across 16 layers, about 1.6 MB total state).
- For each token, the model fades the old state with a learned decay rate, then writes the new token as a key-value outer product:
  $$S_t = \alpha_t S_{t-1} + \beta_t (k_t \otimes v_t)$$
  $$y_t = q_t S_t$$
  Here $S$ is the memory state, $\alpha$ is how much memory to keep, $\beta$ is the learned write gate, and $k, v, q$ are the key, value, and query projections.
- To produce tokens, the model reads from that memory state using its query projection.
- Memory consumption never grows with output length. In our test runs, the last token took about as long to generate as the first (a last-to-first step latency ratio of 1.025x).

### Parallel training

Recurrent models used to be slow to train because each step had to wait for the step before it. We train in parallel instead:

- Every 4,096-token sequence is divided into chunks of 64 tokens.
- Within each chunk, computations run as parallel matrix multiplications.
- Across chunk boundaries, state propagation uses a closed-form associative scan with one masked decay matrix, without Python loops over chunks.
- Multiple documents are packed into each 4,096-token window, with a reset mask zeroing the memory at document boundaries so text from one source never bleeds into another.

---

## Why wide and shallow (slab_v3)

Our earlier 120M model was deep and narrow: 32 layers and width 512. Passing a recurrent signal through 32 layers proved difficult on a small card. Gradients shrank by about 23.5x between layer 0 and layer 15, and the narrow width constrained the memory state per head.

For 130M, we moved to a wider and shallower layout with 16 layers and width 768. We also added four architectural changes (slab_v3):

1. Token-shift mixing. Each token's projections are mixed with the previous token, giving the model short local context before updating recurrent memory.
2. Learned write gate ($\beta$). The model learns how strongly each token writes to memory, letting filler words write weakly and substantive tokens write strongly.
3. Layer-0 value highway. Values from layer 0 are fed directly into later layers through a learned residual connection, cutting the layer-0 to layer-15 gradient slope ratio from 23.5x down to 16.29x.
4. FP32 decay scan. Decay accumulations are stored in 32-bit floats. In BF16, precision loss across chunks causes the model to forget context prematurely.

---

## Tokenizer: keeping Kannada letters whole

Indic scripts write a syllable (an akshara) as a base consonant joined with vowel marks (matras) or a virama. For instance, the word "ಕನ್ನಡ" (Kannada) contains the conjunct letter ನ್ನ, which combines two consonants with a virama.

Standard GPT-style tokenizer regexes treat these marks as separate tokens. They split words in the middle of individual letters, forcing the model to spend parameters relearning basic orthography.

Our 32k byte-level BPE tokenizer (`v3_32k`) uses a regex that keeps combining marks (`\p{M}`) attached to their base consonant:

- At the same 48k vocabulary size, our regex fits 10.76 bytes of Indic text per token compared to 3.67 bytes for a standard GPT-style regex. That is about 2.9x more text per token.
- Our shipped 32k tokenizer packs 9.94 bytes per token.
- Every validation document across all nine languages round-trips back to the exact original bytes without data loss.
- 256 byte-level fallbacks guarantee zero out-of-vocabulary (`<unk>`) errors.

---

## Model specifications

| Specification | Value |
|---|---|
| Model name | parentheses-0.9-130M |
| GitHub release | [v0.9-130m-v3-golden](https://github.com/uncoalesced/parentheses/releases/tag/v0.9-130m-v3-golden) |
| HuggingFace repo | [uncoalesced/parentheses-v0.9-130M-v3](https://huggingface.co/uncoalesced/parentheses-v0.9-130M-v3) |
| Parameter count | 148.38M parameters (referred to as 130M) |
| Architecture | Recurrent selective linear attention (State Space Duality / SSD) |
| Layers | 16 |
| Hidden dimension ($d_{\text{model}}$) | 768 |
| Attention heads | 12 heads (dimension 64 per head) |
| Context window | 4,096 tokens |
| Vocabulary | 32,768 tokens (`v3_32k` Akshara-preserving byte-level BPE) |
| Embeddings | Tied input and output embedding table |
| Recurrent memory footprint | ~1.60 MB total state across all layers |
| Training hardware | One NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM) |
| Code license | MIT License |
| Weights license | CC BY-NC 4.0 (research and non-commercial) |

---

## Training recipe

Everything was trained on one 8 GB laptop GPU:

1. Pretraining. Trained on an 808M-token corpus (`processed_130m_v1`) over 22,888 steps (about 1.5B token positions seen). The wall clock was about 67 hours at 10.5 seconds per step. Mix shares: Dravidian core 40%, STEM reasoning 25%, conversational and open data 20%, English anchors 15%.
2. Pretrain checkpoint soup. We averaged the final three pretraining checkpoints (steps 22,000, 22,500, and 22,887) to reduce parameter variance.
3. Instruction tuning (SFT v2). 131M tokens over 2,000 steps on `processed_sft_stem_open_v2`. Documents truncated by sequence bounds were masked out of the loss function, so the model never learned from incomplete reasoning chains. About 20% of rows replayed pretraining data to preserve language grounding.
4. SFT checkpoint soup. We averaged the final three SFT checkpoints (steps 1,500, 1,750, and 1,999) into `step_2000_sft_swa3.pt`.
5. Preference tuning (DPO v2). 100 steps over 900 balanced preference pairs. The dataset was rebuilt after an audit to balance reasoning blocks and fix label errors: 400 abstention pairs ("I don't know") across five languages, 250 facts, 100 persona, 78 maths, and 72 logic. Final margin reached +0.3507 with 84.2% preference accuracy across the full dataset.

---

## Quickstart

Clone the repository and install dependencies:

```bash
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
pip install -r requirements.txt
```

### Interactive chat

Run the interactive terminal chat:

```bash
# Chat mode with automatic language detection
python scripts/chat.py --checkpoint <path-to-checkpoint> --mode chat

# Thinking mode with explicit reasoning traces in Kannada
python scripts/chat.py --checkpoint <path-to-checkpoint> --mode think --lang kan

# Thinking mode in English
python scripts/chat.py --checkpoint <path-to-checkpoint> --mode think --lang eng
```

### Python usage

```python
import torch
from model.backbone import Parentheses
from model.config import PRESETS

# Instantiate the 130M architecture
config = PRESETS["parentheses-0.9-130M-v3"]
model = Parentheses(config)

# Load checkpoint weights
checkpoint = torch.load("path/to/step_0100_final.pt", map_location="cpu", weights_only=True)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# Input token IDs (batch, sequence_length)
input_ids = torch.tensor([[1, 28, 452, 91]], dtype=torch.long)

# Parallel training / prompt prefill form:
logits = model(input_ids)

# Recurrent step generation form (constant O(1) memory per step):
# model.step(token_id, state) updates a fixed (12, 64, 64) matrix per layer
```

---

## Evaluations and what is still weak

We believe in testing our own numbers and publishing honest results.

### What works

- Script fidelity. In testing, 40 of 40 runs stayed in the right script across our five main languages (Kannada, Tamil, Telugu, Malayalam, English). Across all nine languages, 53 of 72 runs (73.6%) stayed in script.
- Flat generation cost. Per-token generation latency stayed flat throughout testing (1.025x ratio between last and first steps). Generation averaged about 27 tokens a second on the laptop GPU.
- Clean outputs. Zero control tokens or internal tags leaked into user text across 72 test runs.
- Legitimate technical queries. The model did not refuse any of the 19 sensitive-but-legitimate technical prompts tested (0.0% refusals).
- Edge export. We exported a single-step ONNX graph (694 MB) that runs via ONNX Runtime DirectML. Its outputs match PyTorch within about 2e-5.

### Bits per byte (held-out validation text)

Compression density measured on held-out validation data from `processed_130m_v1` using the `v3_32k` tokenizer (lower is better):

| Language | Bits per byte |
|---|---|
| Malayalam (`mal`) | about 0.44 |
| Tamil (`tam`) | about 0.45 |
| Kannada (`kan`) | about 0.50 |
| Telugu (`tel`) | about 0.56 |
| Tulu (`tcy`) | about 1.1 |
| English (`eng`) | about 1.21 |
| Kolami (`nit`) | about 1.4 |
| Koya (`kff`) | about 1.6 |
| Kuvi (`kxv`) | about 1.6 |

Indic scripts encode in UTF-8 with three bytes per character, which explains why byte-level compression looks lower than English. Scores for Koya, Kuvi, and Kolami rest on smaller validation sets (12 documents each), so treat them as rough indications.

### What is still weak

- Reasoning benchmark. Our initial test scored 15 of 24 on an internal honest suite. When we audited the test, we found that 18 of those 24 prompts were copies of training prompts. On a clean set of 24 fresh prompts, the honest score was 5 of 24. We published the clean score.
- Hallucinations. The model still fabricates facts when asked about things that do not exist. When asked about a Martian colony "founded in 1842", it invented a detailed response rather than stating that no such colony exists.
- Incomplete reasoning traces. In 72 test runs, the model opened a `<think>` scratchpad in 36 runs, but closed the block itself in only 4 of those 36.
- Kannada conversation and mathematics. While script retention is solid, complex dialogue and multi-step arithmetic in Kannada still degrade frequently.
- Low-resource languages. Tulu, Koya, Kuvi, and Kolami script accuracy sits at roughly 74%, and generations in those dialects remain fragile.

---

## What is next

We are currently training parentheses-1.0-240M on the same laptop setup:

- 16 layers, hidden dimension 1024, 16 heads (about 240M parameters), grown from 0.9.
- Epistemic abstention traces built directly into instruction tuning, so the model learns to say "I don't know" rather than fabricating plausible fiction.
- Expanded conversational data to improve multi-turn dialogue.

---

## Licenses

- Code and architecture: [MIT License](LICENSE)
- Model weights: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC 4.0). Free for academic research, education, and non-commercial evaluation.
