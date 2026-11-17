"""Chat with parentheses in a terminal.

    python -m parentheses.chat --checkpoint weights/model.safetensors

Type a message and press Enter. An empty line exits. Commands:

    /lang <code>   use that language for every message (kan tam tel mal eng tcy kff kxv nit)
    /auto          go back to picking the language from the script you type
    /<code> text   send one message in that language, e.g. /tam vanakkam
    /new           start a fresh conversation (only matters with --keep-history)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from . import TOKENIZER_PATH
from .checkpoint import load_model
from .session import RELEASE_SAMPLING, ChatSession

LANGS = ("kan", "tam", "tel", "mal", "eng", "tcy", "kff", "kxv", "nit")
LANG_NAMES = {"kan": "Kannada", "tam": "Tamil", "tel": "Telugu", "mal": "Malayalam", "eng": "English",
              "tcy": "Tulu", "kff": "Koya", "kxv": "Kuvi", "nit": "Kolami"}

# Unicode blocks of the main scripts. Experimental languages share scripts
# with these, so they are never picked automatically: force them.
SCRIPT_BLOCKS = (("kan", 0x0C80, 0x0CFF), ("tam", 0x0B80, 0x0BFF), ("tel", 0x0C00, 0x0C7F),
                 ("mal", 0x0D00, 0x0D7F))


def detect_language(text: str) -> str:
    """Pick the language from the script that appears most often; English otherwise."""
    counts = {code: 0 for code, _, _ in SCRIPT_BLOCKS}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                counts[code] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] else "eng"


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}\n"
                         "Download it with: hf download uncoalesced-org/parentheses-1.0-240M "
                         "model.safetensors --local-dir weights")
    return p


class Streamer:
    """Prints decoded text as tokens arrive, holding back half-finished characters."""

    def __init__(self, tok):
        self.tok, self.ids, self.shown = tok, [], 0

    def __call__(self, token_id: int) -> None:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return
        sys.stdout.write(text[self.shown:])
        sys.stdout.flush()
        self.shown = len(text)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m parentheses.chat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="weights/model.safetensors",
                    help="model.safetensors, or the folder that holds it")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (auto tries them in that order)")
    ap.add_argument("--lang", default="auto", choices=("auto",) + LANGS)
    ap.add_argument("--format", default="chat", choices=("chat", "base"),
                    help="chat wraps the message in the chat template; base just continues your text")
    ap.add_argument("--keep-history", action="store_true",
                    help="carry the conversation between messages (1.0 was trained on single turns)")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=RELEASE_SAMPLING["temperature"])
    ap.add_argument("--top-k", type=int, default=RELEASE_SAMPLING["top_k"])
    ap.add_argument("--min-p", type=float, default=RELEASE_SAMPLING["min_p"])
    ap.add_argument("--no-repeat-ngram", type=int, default=RELEASE_SAMPLING["no_repeat_ngram_size"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from tokenizers import Tokenizer
    device = pick_device(args.device)
    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"loading {ckpt} on {device} ...", flush=True)
    model = load_model(ckpt, device=device)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"parentheses, {model.num_params() / 1e6:.2f}M parameters, device {device}, "
          f"language {args.lang}, format {args.format}"
          f"{', history kept' if args.keep_history else ''}")
    print("Type a message and press Enter. Empty line exits. /lang <code>, /auto, /<code> text, /new.")

    session_lang = args.lang
    session: Optional[ChatSession] = None
    sampling = dict(temperature=args.temperature, top_k=args.top_k, min_p=args.min_p,
                    no_repeat_ngram_size=args.no_repeat_ngram)

    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            break
        text, lang = line.strip(), None
        if text.startswith("/"):
            cmd, _, rest = text[1:].partition(" ")
            if cmd == "lang" and rest.strip() in LANGS + ("auto",):
                session_lang = rest.strip()
                print(f"[language: {session_lang}]")
                continue
            if cmd == "auto":
                session_lang = "auto"
                print("[language: auto]")
                continue
            if cmd == "new":
                session = None
                print("[new conversation]")
                continue
            if cmd in LANGS:
                if not rest.strip():
                    print(f"[/{cmd} needs text after it, e.g. /{cmd} your question]")
                    continue
                text, lang = rest.strip(), cmd
            else:
                print("[unknown command; see /lang, /auto, /<code> text, /new]")
                continue
        if lang is None:
            lang = detect_language(text) if session_lang == "auto" else session_lang
        print(f"[{lang}: {LANG_NAMES[lang]}]")

        if session is None or not args.keep_history:
            session = ChatSession(model, tokenizer=tok, lang=lang)
        stream = Streamer(tok)
        try:
            if args.format == "chat":
                session.add_user(text, lang=lang)
            else:
                session.feed([tok.token_to_id(f"<{lang}>")] + tok.encode(text, add_special_tokens=False).ids)
            session.generate(max_new_tokens=args.max_new_tokens, on_token=stream, **sampling)
        except KeyboardInterrupt:
            print("\n[stopped]")
        print()


if __name__ == "__main__":
    main()
