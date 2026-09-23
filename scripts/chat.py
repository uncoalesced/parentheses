"""
Interactive terminal chat: type a prompt, get a completion back, repeat.
For eyeballing quality by hand -- no scoring, no logging, no format assumed.

Engineered by uncoalesced

Same checkpoint-load + model.generate() pattern as sample_freeform.py, just
wrapped in an input() loop instead of a fixed PROMPTS dict. Works for both
attn_type="causal" and "selective_linear" checkpoints unmodified -- generate()
is a thin wrapper around stream() for both (documentation.md, 2026-09-06 5e).

    python3 scripts/chat.py --checkpoint checkpoints/all-sources-v1/step_560999_final.pt
    python3 scripts/chat.py --self-test        # no checkpoint, no torch network need
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PRESETS, Parentheses


@torch.no_grad()
def reply(model, prompt: str, device: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
    full = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    return full[len(prompt):]  # only the continuation, prompt already on screen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/all-sources-v1/step_560999_final.pt")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"[chat] {args.checkpoint} (step {ckpt.get('step')}, {model.num_params()/1e6:.2f}M params, "
          f"attn_type={ckpt['cfg'].attn_type}) on {args.device}. Empty line or Ctrl-C to quit.")

    while True:
        try:
            prompt = input("\n> ").strip().replace("\\n", "\n")  # literal \n typed at a
            # single-line prompt -> real newline, so multi-line-shaped prompts (e.g. the
            # "## Key bindings\n\n* " style) actually reach the model as one
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        out = reply(model, prompt, args.device, args.max_new_tokens, args.temperature, args.top_k)
        print(out)


def _self_test():
    torch.manual_seed(0)
    cfg = PRESETS["tiny-smoke"]
    model = Parentheses(cfg).eval()
    # "12" is bytes 49/50 on purpose: tiny-smoke's vocab_size is 64, an
    # ASCII-letter prompt would index off the end of tok_emb (same constraint
    # sample_freeform.py's self-test works around).
    out = reply(model, "12", "cpu", max_new_tokens=8, temperature=1.0, top_k=4)
    assert len(out.encode("utf-8")) >= 1, out
    print("[self-test] chat ok")


if __name__ == "__main__":
    main()
