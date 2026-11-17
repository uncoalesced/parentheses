"""Check that the model runs the same on your accelerator as on the CPU.

    python -m parentheses.selftest --checkpoint weights/model.safetensors

Generates 16 tokens greedily on the CPU, replays the same tokens on the
chosen device (CUDA or Apple's mps), compares the logits, and times both.
Ends with a report block and one PASS or FAIL line. Paste the report into an
issue if anything looks wrong.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import torch

from . import TOKENIZER_PATH
from .chat import pick_device, resolve_checkpoint
from .checkpoint import load_model

PROMPT = "Write a Python function that returns the factorial of n."
N_TOKENS = 16
TOLERANCE = 1e-3


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def greedy(model, ids, device, n):
    """Greedy tokens and per-step logits, using the parallel prefill then step()."""
    x = torch.tensor([ids], device=device)
    logits, states = model.forward_with_state(x)
    scores, out, all_logits = logits[0, -1], [], []
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        all_logits.append(scores.float().cpu())
        nid = int(scores.argmax())
        out.append(nid)
        step_logits, states = model.step(torch.tensor([nid], device=device), states)
        scores = step_logits[0]
    _sync(device)
    return out, torch.stack(all_logits), n / (time.perf_counter() - t0)


@torch.no_grad()
def replay(model, ids, tokens, device):
    """Feed the CPU's tokens on `device` and return its logits at each step."""
    x = torch.tensor([ids], device=device)
    logits, states = model.forward_with_state(x)
    scores, all_logits = logits[0, -1], []
    for nid in tokens:
        all_logits.append(scores.float().cpu())
        step_logits, states = model.step(torch.tensor([nid], device=device), states)
        scores = step_logits[0]
    return torch.stack(all_logits)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m parentheses.selftest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="weights/model.safetensors")
    ap.add_argument("--device", default="auto", help="device to check against the CPU (auto, cuda, mps, cpu)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    ids = ([tok.token_to_id("<|user|>"), tok.token_to_id("<eng>")]
           + tok.encode(PROMPT, add_special_tokens=False).ids + [tok.token_to_id("<|assistant|>")])
    device = pick_device(args.device)
    ckpt = resolve_checkpoint(args.checkpoint)

    cpu_model = load_model(ckpt, device="cpu")
    cpu_tokens, cpu_logits, cpu_tps = greedy(cpu_model, ids, "cpu", N_TOKENS)
    lines = [f"cpu tokens/s: {cpu_tps:.1f}", f"cpu text: {tok.decode(cpu_tokens)!r}"]
    ok = True
    if device != "cpu":
        del cpu_model
        dev_model = load_model(ckpt, device=device)
        dev_logits = replay(dev_model, ids, cpu_tokens, device)
        dev_tokens, _, dev_tps = greedy(dev_model, ids, device, N_TOKENS)
        diff = (dev_logits - cpu_logits).abs().max().item()
        same = dev_tokens == cpu_tokens
        ok = diff < TOLERANCE and same
        lines += [f"{device} tokens/s: {dev_tps:.1f}",
                  f"{device} vs cpu max abs logit diff: {diff:.3e} (limit {TOLERANCE:g})",
                  f"{device} greedy tokens match cpu: {same}"]
    else:
        lines.append("no accelerator found; only the CPU was checked")

    print("----- BEGIN PARENTHESES SELFTEST REPORT")
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}, "
          f"torch {torch.__version__}")
    print(f"device: {device}")
    for line in lines:
        print(line)
    print("----- END PARENTHESES SELFTEST REPORT")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
