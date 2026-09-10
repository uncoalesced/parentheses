"""
Sanity-check a trained checkpoint: aggregate val loss + raw translation-format
samples, written to a file for a human to actually read.

Two separate jobs, both deliberately dumb:

1. `val_loss` -- train.py has no validation loop at all, so the ~1.3 figure in
   train_log.txt is *training* loss on randomly-sampled windows. This walks
   val.bin in sequential non-overlapping block_size windows and reports the
   mean next-token cross-entropy, which is the number you can actually compare
   across runs.
2. `sample` -- prompts the model in the exact format prepare_parallel.py wrote
   its pairs in ("<en> source -> <kn> target", see format_pair there) and lets
   it continue. No scoring, no BLEU: a 330K-param model one night off a mixed
   22-language corpus is not a translator, and the useful output here is raw
   text a human can eyeball, not a metric.

    python3 scripts/sample_translation.py --checkpoint checkpoints/multilingual/step_278999_final.pt
    python3 scripts/sample_translation.py --self-test        # no checkpoint, no data
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PRESETS, Parentheses

# Short simple declaratives, in the register the corpus actually holds
# (scripture, Wikipedia, government notices, UI strings).
PROMPTS = [
    "The sky is blue.",
    "He went to the city.",
    "Water is important for life.",
    "The book is on the table.",
]

# Kannada first -- Joel reads it, so it is the one sample that can be judged
# by eye rather than by vibe. The rest span the script families in the corpus:
# Devanagari, Latin, Cyrillic, Han.
LANGS = ["kn", "hi", "fr", "ru", "zh"]


def format_prompt(text: str, tgt: str, src: str = "en") -> str:
    """The left-hand side of prepare_parallel.py's format_pair, up to the point
    where the model has to produce the translation itself."""
    return f"<{src}> {text} -> <{tgt}>"


@torch.no_grad()
def val_loss(model, path: str, device: str, batch_size: int = 64, max_batches: int = 0) -> tuple[float, int]:
    """Mean next-token cross-entropy over sequential windows of `path`.

    Sequential and non-overlapping, not train.py's random sampling, so the
    number is deterministic and re-runnable. Returns (loss, windows_used).
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")
    block = model.cfg.block_size
    n = (len(data) - 1) // block
    total, count = 0.0, 0
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        idx = [i * block for i in range(start, stop)]
        x = torch.from_numpy(np.stack([data[i:i + block] for i in idx]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block] for i in idx]).astype(np.int64)).to(device)
        _, loss = model(x, y)
        total += loss.item() * (stop - start)
        count += stop - start
        if max_batches and count >= max_batches * batch_size:
            break
    return total / max(count, 1), count


def val_languages(path: str, limit: int = 40_000_000) -> str:
    """Which target-language tags actually appear in the val split.

    tokenize_corpus.py takes val as a *contiguous tail* of the concatenated
    stream, and the raw dirs are globbed in sorted order -- so the tail is
    whatever language sorts last (en-zh), not a sample of the corpus. That was
    harmless when the corpus was all English and is actively misleading now: a
    val loss computed only on Chinese is a loss on the most expensive script in
    the set (3 UTF-8 bytes per Han character), not a corpus-wide number.
    Printing the coverage keeps that from being read as a generalization gap.
    """
    import collections
    import re
    data = np.memmap(path, dtype=np.uint16, mode="r")
    text = bytes(data[:min(len(data), limit)].tolist()).decode("utf-8", errors="replace")
    tags = collections.Counter(re.findall(r"-> <([a-z]{2})>", text))
    if not tags:
        return "no pair tags found (monolingual text)"
    total = sum(tags.values())
    return ", ".join(f"{k} {v / total:.0%}" for k, v in tags.most_common(8)) + f"  ({total:,} pairs)"


@torch.no_grad()
def sample(model, prompt: str, device: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
    # errors="replace": byte-level sampling can land mid-codepoint, and a
    # half-formed multibyte character is itself worth seeing in the output.
    return bytes(out[0].tolist()).decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/multilingual/step_278999_final.pt")
    p.add_argument("--val-data", default="data/processed_multilingual/val.bin")
    p.add_argument("--out", default="samples/translation_samples.txt")
    p.add_argument("--langs", nargs="+", default=LANGS)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--val-batches", type=int, default=0, help="0 = the whole val set")
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
        f"config     : {ckpt['cfg']}",
        f"sampling   : temperature={args.temperature} top_k={args.top_k} "
        f"max_new_tokens={args.max_new_tokens}",
        "",
    ]

    if os.path.exists(args.val_data):
        vl, nwin = val_loss(model, args.val_data, args.device, max_batches=args.val_batches)
        lines += [f"val loss   : {vl:.4f} over {nwin:,} sequential {model.cfg.block_size}-token "
                  f"windows of {args.val_data}"]
        print(f"[val] loss {vl:.4f} over {nwin:,} windows")
        cov = val_languages(args.val_data)
        lines.append(f"val covers : {cov}")
        print(f"[val] language tags present: {cov}")
        lines.append("")
    else:
        lines += [f"val loss   : SKIPPED, no {args.val_data}", ""]

    lines.append("=" * 72)
    lines.append("Translation-format samples. Prompt is prepare_parallel.py's pair format")
    lines.append("up to the target tag; everything after it is the model's own continuation.")
    lines.append("=" * 72)

    # sample() calls model.generate(), which now has a selective_linear branch
    # (Parentheses.stream(), added 2026-09-06) as well as the causal KV-cache
    # path -- both are safe to call unconditionally here. (Previously this
    # guarded against a crash when only the causal branch existed; stale now
    # that generate() supports both -- removed rather than left as dead code
    # that would silently keep skipping samples forever.)
    for lang in args.langs:
        lines += ["", f"---------- en -> {lang} ----------"]
        for text in PROMPTS:
            prompt = format_prompt(text, lang)
            out = sample(model, prompt, args.device, args.max_new_tokens,
                         args.temperature, args.top_k)
            lines += ["", f"PROMPT: {prompt}", f"OUTPUT: {out}"]
            print(f"[sample] en->{lang}: {text}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] wrote {args.out}")


def _self_test():
    import tempfile

    # format matches prepare_parallel.format_pair's left-hand side exactly
    assert format_prompt("A dog.", "kn") == "<en> A dog. -> <kn>"
    assert format_prompt("A dog.", "kn").startswith("<en> ")

    torch.manual_seed(0)
    cfg = PRESETS["tiny-smoke"]
    model = Parentheses(cfg).eval()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "val.bin")
        rng = np.random.default_rng(0)
        rng.integers(0, cfg.vocab_size, size=cfg.block_size * 20, dtype=np.uint16).tofile(path)
        loss, n = val_loss(model, path, "cpu", batch_size=4)
        assert n == (cfg.block_size * 20 - 1) // cfg.block_size, n
        # untrained model on uniform-random data sits near ln(vocab_size)
        expected = float(np.log(cfg.vocab_size))
        assert abs(loss - expected) < 1.0, (loss, expected)

    # val_languages reads the pair tags back out of a .bin
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tagged.bin")
        blob = ("<en> a -> <kn> b\n\n" * 3 + "<en> c -> <zh> d\n\n").encode("utf-8")
        np.frombuffer(blob, dtype=np.uint8).astype(np.uint16).tofile(path)
        cov = val_languages(path)
        assert "kn 75%" in cov and "zh 25%" in cov, cov
        mono = os.path.join(tmp, "mono.bin")
        np.frombuffer(b"plain english text", dtype=np.uint8).astype(np.uint16).tofile(mono)
        assert "no pair tags" in val_languages(mono)

    # Sampling returns the prompt plus exactly max_new_tokens more ids.
    # "12" is bytes 49/50 on purpose: tiny-smoke's vocab_size is 64, so a
    # normal ASCII-letter prompt indexes straight off the end of tok_emb.
    out = model.generate(torch.tensor([[49, 50]]), 8, temperature=1.0, top_k=4)
    assert out.shape == (1, 10), out.shape
    assert sample(model, "12", "cpu", max_new_tokens=8, temperature=1.0, top_k=4).startswith("12")
    print("[self-test] sample_translation ok")


if __name__ == "__main__":
    main()
