# Architecture: Parentheses 0.9 (Project Tanzinite)
<!-- Engineered by uncoalesced -->

> Parentheses is not a Transformer. It is a from-scratch recurrent / state space duality (SSD) model in the Mamba-2 / RWKV / selective linear attention family: chunked $O(T)$ parallel training and constant $O(1)$ recurrent decoding, with no softmax attention and no KV-cache.

---

## 1. System Overview & Core Topology

Parentheses (alias **Project Tanzinite**) is a research-grade linear recurrent language model family designed for high-throughput $O(T)$ parallel training and constant $O(1)$ memory edge/NPU inference.

### High-Level Topology (`model/backbone.py`)

Each layer $l \in [0, L-1]$ is structured as a Pre-LN residual block:

$$\begin{aligned}
x^{(l)}_{\text{norm1}} &= \text{RMSNorm}(x^{(l)}) \\
a^{(l)}, V_0 &= \text{SelectiveLinearAttention}(x^{(l)}_{\text{norm1}}, \text{reset\_mask}, V_0) \\
x^{(l + 1/2)} &= x^{(l)} + a^{(l)} \\
x^{(l)}_{\text{norm2}} &= \text{RMSNorm}(x^{(l + 1/2)}) \\
m^{(l)} &= \text{SwiGLU}(x^{(l)}_{\text{norm2}}) \\
x^{(l+1)} &= x^{(l + 1/2)} + m^{(l)}
\end{aligned}$$

- **RMSNorm:** Root Mean Square Layer Normalization without learnable bias.
- **RoPE (`precompute_rope` / `apply_rope`):** Applied to Query and Key projections prior to recurrence.
- **SwiGLU MLP:** Gated linear unit with SiLU activation ($\text{SwiGLU}(x) = (W_1 x \odot \text{SiLU}(W_3 x)) W_2$), where hidden dimension expands to approximately $\frac{8}{3} d_{\text{model}}$.
- **Tied Embeddings:** `head.weight = tok_emb.weight`. Embedding table parameters are strictly tied between input and output projections.
- **Pre-LM-Head Pooling:** `Parentheses.hidden()` exposes final-layer representations post-`ln_f` for embedding extraction and sentence pooling without re-implementing block passes.

---

## 2. Core Selective Linear Attention (`model/selective_linear_attention.py`)

The attention layer implements State Space Duality (SSD) with two mathematically equivalent execution paths:

### 2.1 The Two Execution Modes

1. **Dual / Parallel Form (`forward`):** Used during training and chunked prompt processing. Sequence of length $T$ is processed in chunked associative parallel scans in $O(T)$ time.
2. **Recurrent Form (`step`):** Used during autoregressive generation, interactive chat, and NPU/edge deployment. Processes a single token in $O(1)$ time and constant memory:
   $$S_t = \alpha_t S_{t-1} + \beta_t (k_t \otimes v_t)$$
   $$y_t = q_t S_t$$
   Bitwise mathematical parity between chunked parallel and recurrent step execution is verified to machine precision ($\le 10^{-15}$ error in FP64 tests).

---

### 2.2 Closed-Form Chunked Linear Scan ($C=64$)

Unlike early prototype iterations that evaluated dense $O(T^2)$ masks, the production forward pass divides the sequence of length $T$ into discrete chunks of size $C = 64$:

1. **Intra-Chunk Scan:** Computes strictly causal local interactions within each chunk $c \in [0, n_{\text{chunks}}-1]$ using local cumulative log-decay:
   $$\Lambda_{c, i} = \sum_{j=0}^{i} \log \alpha_{c, j}$$
   $$M_{\text{intra}}[i, j] = \exp(\Lambda_{c, i} - \Lambda_{c, j}) \quad \text{for } i \ge j$$
   $$o_{\text{intra}} = \left( (q_c \cdot k_c^T \cdot \text{scale}) \odot M_{\text{intra}} \right) v_c$$

2. **Inter-Chunk Recurrence (Closed-Form):** Computes incoming recurrent state $S_{\text{in}}[c]$ for each chunk $c$ across all preceding chunks $j < c$:
   $$S_{\text{in}}[c] = \sum_{j < c} \exp(\text{LA}[c-1] - \text{LA}[j]) \cdot dS[j]$$
   where $dS[j] = (k_j \cdot \text{decay\_to\_end})^T v_j$ is the chunk state contribution, and $\text{LA}$ is the cumulative sum of chunk-total log decays $A[c] = \Lambda_{c, C-1}$.

3. **Strict FP32 State Accumulation (Precision Mandate):**
   In BF16 mixed-precision training, small decay exponentials ($e^{-10}$ to $e^{-30}$) truncate to zero across 64 chunks due to 7-bit mantissa limitations. The inter-chunk accumulation in `SelectiveLinearAttention.forward` strictly upcasts decay weights $W$ and state deltas $dS$ to `torch.float32`:
   ```python
   acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
   W_acc = W.to(acc_dtype)
   dS_acc = dS.to(acc_dtype)
   S_in_acc = torch.einsum("bhcj,bhjde->bhcde", W_acc, dS_acc)
   S_in = S_in_acc.to(q_c.dtype)
   ```
   This eliminates truncation error, preserving long-range dependencies across the entire sequence.

4. **Dynamic Document Boundary Isolation (`reset_mask`):**
   A boolean reset tensor $(B, T)$ where $0$ marks document boundaries is fully integrated into both intra-chunk and inter-chunk transitions:
   - Intra-chunk: $M_{\text{intra}}$ masks out tokens belonging to different documents ($\text{masked\_fill}(\sim \text{same\_doc}, -\infty)$).
   - Inter-chunk: State contributions $dS[j]$ and transfer weights $W[c, j]$ zero out whenever a document boundary occurs between chunks $j$ and $c$.
   - Recurrent `step()`: Accepts `reset: (B,)` to zero out the recurrent matrix and convolution history at document starts.

---

### 2.3 Architectural Flavors: `slab_v2` and `slab_v3`

#### Baseline `slab_v2` Mechanisms
- **Depthwise 1D Causal Convolution ($k=4$):** Prepended to Query, Key, and Value projections to provide short-range $n$-gram inductive bias without sequence attention.
- **SiLU Output Gating:** Output projection is gated by an input-dependent SiLU gate: $y = \text{RMSNorm}(y_{\text{heads}}) \odot \text{SiLU}(W_g x)$.
- **Head RMSNorm (`y_norm`):** Normalizes recurrent head outputs prior to gating to prevent activation magnitude runaway.
- **Log-Spaced Decay Horizon Bias:** `alpha_proj.bias` initialized across heads with log-spaced retention targets ($h_i \in [16, 4096]$ tokens).

#### Cutting-Edge `slab_v3` Enhancements
`slab_v3` introduces three high-leverage mathematical upgrades that solve state-rank pollution, temporal control blindness, and deep-layer gradient dampening while maintaining strict $O(1)$ recurrent memory:

1. **Token-Shift ($\mu$-Shift / Time-Mix):**
   - In standard linear models, forget and output gates project from the unshifted current token $x_t$, making them blind to transitions (e.g. punctuation $\to$ word).
   - `TokenShift` blends $x_t$ with $x_{t-1}$ via learnable channel-wise interpolation vector $\mu \in \mathbb{R}^D$:
     $$x_t^{\text{mix}} = \mu \odot x_t + (1 - \mu) \odot x_{t-1}$$
   - Applied to the control projections: `alpha_proj` (forget gate), `beta_proj` (write gate), and `gate_proj` (output gate).
   - In recurrent `step()`, previous token activations are preserved in an $O(1)$ buffer of shape $(B, D)$.

2. **$\beta$-Gated Selective Write:**
   - In standard SSD ($S_t = \alpha_t S_{t-1} + k_t \otimes v_t$), every token consumes rank in the recurrent state matrix, allowing low-entropy filler tokens (whitespace, punctuation, stop words) to overwrite valuable long-range state.
   - `slab_v3` introduces an input-dependent write strength:
     $$\beta_t = 2 \cdot \sigma(W_\beta x_t^{\text{mix}}) \in (0, 2)$$
     $$S_t = \alpha_t S_{t-1} + \beta_t (k_t \otimes v_t)$$
   - When $\beta_t \approx 0$, the token passes through without polluting the recurrent state; when $\beta_t > 1$, high-salience tokens are emphatically written.

3. **Layer-0 $V_0$ Value Residual Highway:**
   - In deep recurrent models, representations drift through sequential state updates, causing early layer features to be diluted.
   - Layer 0 computes and retains initial value representation $V^{(0)}$:
     $$V^{(0)} = W_v x^{(0)}$$
   - Layers $1 \dots L-1$ blend their local values $v^{(l)}$ with $V^{(0)}$ via a learnable per-head mixing parameter $\lambda_v^{(l)}$:
     $$v^{(l)} = (1 - \sigma(\lambda_v^{(l)})) v^{(l)} + \sigma(\lambda_v^{(l)}) V^{(0)}$$
     ($\lambda_v$ initialized to $-2.0 \implies \sigma(-2.0) \approx 0.12$ initial anchor).
   - Flattens the gradient slope from Layer 0 to Layer $L-1$, stabilizing deep pretraining.

---

## 3. Initialization & Gradient Flow Invariants (`model/backbone.py`)

Two critical invariants govern model initialization and residual stability:

1. **`_preserve_bias` Protection on `_init_weights`:**
   Standard PyTorch recursive `_init_weights` sweeps zero all module biases. This previously wiped `alpha_proj.bias` to zero, setting all heads to a uniform decay $\alpha \approx 0.5$ (destroying state every ~5 tokens).
   - `_init_weights` strictly checks `not getattr(module, "_preserve_bias", False)` before zeroing biases.
   - `Parentheses.__init__` explicitly calls `m.init_decay_bias()` post-sweep as a secondary defense.

2. **$\frac{0.02}{\sqrt{2L}}$ Residual Projection Scaling:**
   In Pre-LN architectures ($x_{l+1} = x_l + f(x_l)$), residual stream variance grows linearly with depth ($\text{Var}(x_l) \approx 1 + 2l$). Consequently, backward gradients through RMSNorm dampen by $\frac{1}{\sqrt{1 + 2l}}$, starving deep layers of gradient signal.
   - Projection weights `attn.proj.weight` and `mlp.w2.weight` are explicitly initialized with scaled standard deviation:
     $$\sigma_{\text{res}} = \frac{0.02}{\sqrt{2 \cdot n_{\text{layer}}}}$$
   - This compressed the initial Layer 0 vs. Layer 15 gradient slope ratio from $23.5\times \to 16.29\times$.

---

## 4. Model Configurations & Presets (`model/config.py`)

All model architectures are defined in `model/config.py` in `PRESETS`.

### The Aspect Ratio Restructuring (Wide-and-Shallow)

Earlier 120M configurations attempted to scale parameters by doubling depth to **32 layers** while keeping model width narrow ($d_{\text{model}}=512, n_{\text{head}}=8$). In linear recurrence, this caused:
1. **Unrolled RNN Bottleneck:** Gradients backpropagated through 32 sequential linear recurrence steps, causing deep-layer vanishing gradients.
2. **State Capacity Bottleneck:** Total recurrent rank was restricted to $8 \times 64 = 512$.

The architecture was restructured into wide-and-shallow 16-layer configurations (`130M-v2` and `130M-v3`), expanding width to $d_{\text{model}}=768$ ($n_{\text{head}}=12, d_{\text{head}}=64$), increasing recurrent rank by **+50%** to 768 per layer while halving backpropagation depth.

### Active Presets Summary Table

| Preset Name | Layers ($L$) | Heads ($H$) | Dim ($D$) | Head Dim | Vocab | Approx Params | Architecture & Features |
|---|---|---|---|---|---|---|---|
| `tiny-smoke` | 2 | 2 | 32 | 16 | 64 | ~35K | CPU smoke test & CI validation |
| `parentheses-0.9-100k` | 4 | 4 | 48 | 12 | 256 | ~123K | Byte-level experimental |
| `parentheses-0.9-300k` | 5 | 4 | 72 | 18 | 256 | ~330K | Historical byte-level baseline |
| `parentheses-0.9-600k` | 5 | 4 | 96 | 24 | 320 | ~585K | Byte-level intermediate |
| `parentheses-0.9-1m` | 5 | 8 | 128 | 16 | 512 | ~1.13M | BPE 512 experimental |
| `parentheses-0.9-41M` | 16 | 6 | 384 | 64 | 32,768 | 40.9M | Production BPE 32k (trained on 1.06B tokens) |
| `parentheses-0.9-120M` | 32 | 8 | 512 | 64 | 32,768 | 119.7M | Historical deep-narrow pretraining run |
| **`parentheses-0.9-130M-v2`** | 16 | 12 | 768 | 64 | 32,768 | 148.22M | Wide-shallow, `slab_v2=True`, FP32 chunked scan |
| **`parentheses-0.9-130M-v3`** | 16 | 12 | 768 | 64 | 32,768 | 148.38M | **Active Target**: `slab_v3=True` (Token-Shift, $\beta$-gate, $V_0$ highway) |

---

## 5. Universal Segregated Optimization (`train.py`)

To prevent standard weight decay from destroying sensitive recurrent decay horizons and normalization statistics, `build_optimizer` in `train.py` enforces a **3-way parameter segregation** by default:

```
                  ┌─────────────────────────────────────────────────────────────┐
                  │                    All Model Parameters                     │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
                 ┌───────────────────────────────┼──────────────────────────────┐
                 ▼                               ▼                              ▼
    ┌─────────────────────────┐     ┌─────────────────────────┐    ┌─────────────────────────┐
    │     recurrent_decay     │     │     matrix_weights      │    │    biases_and_norms     │
    │  alpha, delta, decay,   │     │   2D+ Linear Weights    │    │  1D biases, RMSNorms,   │
    │    beta, v_residual     │     │   (Projections, MLP)    │    │  conv filters, y_norm,  │
    │                         │     │                         │    │   mu (TokenShift mix)   │
    ├─────────────────────────┤     ├─────────────────────────┤    ├─────────────────────────┤
    │  Weight Decay: 0.0      │     │  Weight Decay: 0.1      │    │  Weight Decay: 0.0      │
    │  Learning Rate: 0.1x LR │     │  Learning Rate: 1.0x LR │    │  Learning Rate: 1.0x LR │
    └─────────────────────────┘     └─────────────────────────┘    └─────────────────────────┘
```

- **Fused Execution:** Automatically enables PyTorch CUDA `fused=True` AdamW when running on GPU.
- **Activation Checkpointing:** Enabled via `Parentheses.grad_checkpointing = True`. Blocks recompute activations during backward pass, allowing $B=2 \dots 6, T=4096$ to run on consumer hardware (RTX 5050 8GB VRAM) using only 2.3 GB to 5.5 GB VRAM.

---

## 6. Autoregressive Decoding & Features (`features/`)

All decoding and interactive features run on the trained recurrent backbone with constant memory:

1. **`Parentheses.step()`:** Recurrent single-token execution. Accepts state tuple `(mat, conv_hist, shift_state)` under `slab_v3`:
   - `mat`: Matrix recurrent state $(B, H, d_h, d_h)$.
   - `conv_hist`: Causal convolution buffer $(B, 3 \cdot n_{\text{embd}}, 3)$.
   - `shift_state`: Previous token representation $(B, n_{\text{embd}})$.
   Total state size is fixed and does NOT grow with generation length.

2. **`features/free_think.py`:** Unbounded open-ended thought generator streaming from `Parentheses.step()`.

3. **`features/modular_free_think.py`:** Re-queries an `IngestedStore` via BM25 retrieval every `--splice-every` tokens (default 64) to splice relevant context into generation without sequence length growth.

4. **`features/conversation_memory.py`:** Multi-turn conversational memory logging turns to markdown and re-ingesting through the BM25 store.

---

## 7. Data Pipeline & Corpora (`data/`)

- **BPE 32k (`data/tokenizer/v3_32k`):** Production subword tokenizer covering Kannada, Tamil, Telugu, Malayalam, English, and low-resource scripts (Tulu, Kodava, Gondi).
- **Corpus v3 / v4 (`data/processed_v3/`):** 760M-token pretraining mix built from Sangraha verified Dravidian, English Wikipedia, and regional literature.
- **Document Boundaries:** `train_boundaries.npy` tracks document start token offsets, dynamically collated into `reset_mask` by `get_batch()` in `train.py`.
- **Conversational Intelligence Distillation:** Active ingestion from Karya (`karya.in`) and NVIDIA NIM synthetic distillation feeding multimodal and instruction tuning.

---

## 8. Verification & Test Harnesses

Every architecture modification must clear the official test suites:
- **`tests/validate_130m_v3.py`:** 5-step end-to-end verification of `slab_v3` (parameter existence, forward/backward, gradient slope ratio, AdamW memory allocation, and recurrent step parity).
- **`python -m model.selective_linear_attention --self-test`:** Dual vs. recurrent bitwise mathematical parity and boundary isolation test.
- **`python train.py --self-test`:** Collation and optimizer initialization checks.
- **`tests/eval_heldout_release.py`:** 63 held-out prompts testing factual accuracy, zero censorship preambles, and 0.0% refusal rate.
