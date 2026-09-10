"""
Free-form (non-translation-format) sampling for `all-sources-v1` and friends.

`sample_translation.py` only prompts in the `<en> ... -> <lang>` pair format.
This is the throwaway script TESTING_HANDOFF.md called for: byte-encode a
plain prompt, run it through `model.generate`, write the continuation. No
scoring -- a human reads the output.

PROMPTS are one per curated domain the fold added (see documentation.md,
"all-sources-v1 completed"): Hermes function-calling, OpenStax textbook prose,
Steam game-manual text, Gutenberg literary prose, Open Web Math. The Hermes
prompt is raw ShareGPT JSON on purpose, not the `\x01<role>\x02...` SFT
format `data/prepare_sft.py` defines -- FOLD_SOURCES_REPORT.md's self-test
sweep only runs `prepare_sft`'s own self-test, it never appears in the fold
itself, so `data/raw/curated/hermes-function-calling-v1`'s raw JSON went into
`all-sources-v1` as plain undifferentiated byte text. Prompting with the SFT
control bytes would be testing a format nothing in this checkpoint ever saw.

    python3 scripts/sample_freeform.py --checkpoint checkpoints/all-sources-v1/step_560999_final.pt
    python3 scripts/sample_freeform.py --self-test        # no checkpoint, no data
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PRESETS, Parentheses

PROMPTS = {
    "hermes-function-calling (raw JSON, as trained)": (
        '[{"id": "1", "conversations": [{"from": "system", "value": '
        '"You are a function calling AI model. You are provided with function '
        'signatures within <tools> </tools> XML tags.\\n<tools>\\n[{\\"type\\": '
        '\\"function\\", \\"function\\": {\\"name\\": \\"get_weather\\", '
        '\\"description\\": \\"Get current weather for a city.\\", \\"parameters\\": '
        '{\\"type\\": \\"object\\", \\"properties\\": {\\"city\\": {\\"type\\": '
        '\\"string\\"}}}}}]\\n</tools>"}, {"from": "human", "value": '
        '"What is the weather in Paris?"}, {"from": "gpt", "value": "'
    ),
    "openstax (textbook prose)": "The mean of a data set is calculated by",
    "steam-manual (technical/UI text)": "## Key bindings and shortcuts\n\n* ",
    "gutenberg (literary prose)": "It was a cold morning when",
    "open-web-math (theorem prose)": "Theorem. Let x be a real number such that",
}


@torch.no_grad()
def sample(model, prompt: str, device: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
    return bytes(out[0].tolist()).decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/all-sources-v1/step_560999_final.pt")
    p.add_argument("--out", default="samples/freeform_samples.txt")
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

    lines = [
        f"checkpoint : {args.checkpoint}  (step {ckpt.get('step')})",
        f"sampling   : temperature={args.temperature} top_k={args.top_k} "
        f"max_new_tokens={args.max_new_tokens}",
        "",
    ]
    for label, prompt in PROMPTS.items():
        out = sample(model, prompt, args.device, args.max_new_tokens, args.temperature, args.top_k)
        lines += ["=" * 72, label, "-" * 72, f"PROMPT: {prompt}", f"OUTPUT: {out}", ""]
        print(f"[sample] {label}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] wrote {args.out}")


def _self_test():
    torch.manual_seed(0)
    cfg = PRESETS["tiny-smoke"]
    model = Parentheses(cfg).eval()
    # "12" is bytes 49/50 on purpose: tiny-smoke's vocab_size is 64, an
    # ASCII-letter prompt would index off the end of tok_emb.
    out = sample(model, "12", "cpu", max_new_tokens=8, temperature=1.0, top_k=4)
    assert out.startswith("12"), out
    assert len(out.encode("utf-8")) >= 2 + 8, out
    print("[self-test] sample_freeform ok")


if __name__ == "__main__":
    main()
