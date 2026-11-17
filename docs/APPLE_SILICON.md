# Running parentheses on an Apple Silicon Mac (M1, M2, M3, M4)

**Status: untested on real Mac hardware.** The model code is plain PyTorch and should run on the Mac GPU through PyTorch's Metal backend (`mps`). Nobody has confirmed that on a real Mac yet. If you try it, please send back the self-test report from step 4. The CPU path always works.

You need macOS 12.3 or newer, Python 3.11 and about 3 GB of free disk.

## 1. Get the code and a Python 3.11 environment

```bash
brew install python@3.11          # skip if `python3.11 --version` already works
git clone https://github.com/uncoalesced/parentheses.git
cd parentheses
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . huggingface_hub
```

That installs PyTorch, safetensors and tokenizers, plus the `hf` download command. Nothing in it needs CUDA, so it installs cleanly on a Mac.

Check that Python is native arm64 and that PyTorch can see the Mac GPU:

```bash
python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available())"
```

You should see `arm64 2.x.x True`.
- If it prints `x86_64`, the Python is an Intel build running under Rosetta. Install the arm64 Python and recreate the venv.
- If it prints `False`, use `--device cpu` in step 5.

## 2. Download the weights (about 1 GB, float32)

```bash
hf download uncoalesced-org/parentheses-1.0-240M model.safetensors --local-dir weights
```

If the `hf` command is missing, download `model.safetensors` from the [Hugging Face page](https://huggingface.co/uncoalesced-org/parentheses-1.0-240M) (Files and versions tab) into a `weights` folder inside the repository. Use the float32 file; the bfloat16 export is untested.

## 3. Let missing Metal operations fall back to the CPU

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

A few PyTorch operations have no Metal version yet. With this setting they run on the CPU instead of stopping the program.

## 4. Run the self-test

```bash
python -m parentheses.selftest --checkpoint weights
```

The self-test:
- generates 16 tokens on the CPU,
- replays the same tokens on the Mac GPU and checks that the two agree,
- times both and prints tokens per second,
- ends with a report block and a single `PASS` or `FAIL` line.

It takes about a minute once the weights have loaded.

## 5. Chat

```bash
python -m parentheses.chat --checkpoint weights --device auto
```

`--device auto` tries CUDA first, then the Mac GPU (`mps`), then the CPU. Type a message and press Enter. An empty line exits. Add `--lang kan`, `tam`, `tel` or `mal` to force a Dravidian language. Every option is listed in the [setup guide](SETUP.md#all-chat-options).

## What to send back

1. The whole self-test output, from `----- BEGIN PARENTHESES SELFTEST REPORT` to the final `PASS` or `FAIL` line.
2. Your Mac model and chip, RAM and macOS version, from Apple menu > About This Mac.
3. Any warning that says an operation is "not currently supported on the MPS backend".
4. Any `pip` error, copied in full.
5. Whether chat felt faster or slower than `--device cpu`.

Open an issue on GitHub with these.

## Troubleshooting

- **`NotImplementedError` mentioning MPS:** you skipped step 3. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` and run again.
- **The Mac GPU is slower than the CPU:** this can happen. The model writes one token at a time, and on Metal the per-step overhead can cancel out the GPU's speed. Use `--device cpu` if it's faster for you, and tell us the numbers.
- **`pip` can't find a torch wheel:** check that you're on Python 3.11 in arm64 mode. Newer PyTorch builds need a newer macOS. On macOS 12 or 13, pip picks an older PyTorch build, which is fine.
- **You'd rather use Hugging Face `transformers`:** run `pip install transformers`, then follow "With transformers" in the [README](../README.md#with-transformers-no-repository-needed). The chat command above is the tested path.

More detail is in the [setup guide](SETUP.md#on-a-mac-with-apple-silicon-m1-m2-m3-m4).
