#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Converse with Parentheses 1.0-240M (slab_v4).
Enables interactive chat, multi-turn testing, and side-by-side comparison between
Session 6 (Step 12k pretraining anchor) and Session 8 (DPO + SWA-3 golden).

Usage:
  .venv\\Scripts\\python.exe scripts/converse.py --prompt "Hello! Can you write a Python function to compute fibonacci numbers?"
  .venv\\Scripts\\python.exe scripts/converse.py --compare --prompt "What is the capital of Karnataka?"
  .venv\\Scripts\\python.exe scripts/converse.py --interactive
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_slab_v4 import SlabV4Model
from scripts.eval_60p_adversarial_suite import load_model, stream_slab_v4
from scripts import term_ui as ui

DEFAULT_S8 = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s8-golden" / "parentheses-1.0-240M-dpo-swa3-golden.pt"
DEFAULT_S6 = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s6" / "step_12000.pt"
DEFAULT_S7_5 = REPO_ROOT / "checkpoints" / "parentheses-1.0-240M-s7_5" / "step_13000_final.pt"

LANG_TAGS = {
    "eng": "<eng>",
    "kan": "<kan>",
    "tam": "<tam>",
    "tel": "<tel>",
    "mal": "<mal>",
    "hin": "<hin>",
    "san": "<san>",
}


def format_chat_prompt(tokenizer: Tokenizer, prompt: str, lang: str = "eng") -> List[int]:
    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    lang_tag = LANG_TAGS.get(lang, "<eng>")
    lang_id = tokenizer.token_to_id(lang_tag)
    body_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    return [user_id, lang_id] + body_ids + [asst_id]


def generate_response(
    model: SlabV4Model,
    tokenizer: Tokenizer,
    prompt: str,
    lang: str = "eng",
    max_tokens: int = 256,
    temp: float = 0.0,
    top_k: int = 50,
    min_p: float = 0.05,
    rep_penalty: float = 1.15,
    device: str = "cuda",
) -> Dict[str, Any]:
    input_ids_list = format_chat_prompt(tokenizer, prompt, lang=lang)
    inp = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    stops = {eot_id, user_id, tokenizer.token_to_id("<|system|>")} - {None}

    t0 = time.perf_counter()
    tokens = []
    with torch.inference_mode():
        for tok in stream_slab_v4(
            model,
            inp,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=rep_penalty,
        ):
            if tok in stops:
                break
            tokens.append(tok)
    t1 = time.perf_counter()
    elapsed = max(t1 - t0, 1e-4)

    full_text = tokenizer.decode(tokens, skip_special_tokens=False)

    # Separate think block and response
    think_text = ""
    ans_text = full_text
    m = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
    if m:
        think_text = m.group(1).strip()
        ans_text = full_text[m.end():].strip()
    elif "<think>" in full_text:
        # Unclosed think block
        parts = full_text.split("<think>", 1)
        think_text = parts[1].strip()
        ans_text = ""

    # Clean special tokens from display text
    clean_ans = re.sub(r"<\|.*?\|>", "", ans_text).strip()
    clean_think = re.sub(r"<\|.*?\|>", "", think_text).strip()

    return {
        "raw_text": full_text,
        "think": clean_think,
        "response": clean_ans if clean_ans else full_text,
        "tokens": len(tokens),
        "elapsed_sec": elapsed,
        "tok_per_sec": len(tokens) / elapsed,
    }


def converse_turn(
    model: SlabV4Model,
    tokenizer: Tokenizer,
    prompt: str,
    name: str = "parentheses",
    lang: str = "eng",
    temp: float = 0.0,
    max_tokens: int = 256,
    rep_penalty: float = 1.15,
    device: str = "cuda",
    show_think: bool = True,
) -> None:
    res = generate_response(
        model,
        tokenizer,
        prompt,
        lang=lang,
        max_tokens=max_tokens,
        temp=temp,
        rep_penalty=rep_penalty,
        device=device,
    )

    ui.model_header(name)
    if res["think"] and show_think:
        ui.thinking(res["think"])
    ui.response(res["response"])
    ui.stats(res["tokens"], res["tok_per_sec"], res["elapsed_sec"])


def main():
    parser = argparse.ArgumentParser(description="Converse with Parentheses 1.0-240M models")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_S8), help="Path to checkpoint")
    parser.add_argument("--s6-checkpoint", type=str, default=str(DEFAULT_S6), help="Path to S6 anchor checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Single-shot prompt to ask")
    parser.add_argument("--compare", action="store_true", help="Compare S6 vs S8 on the same prompt")
    parser.add_argument("--interactive", action="store_true", help="Run interactive conversation loop")
    parser.add_argument("--lang", type=str, default="eng", choices=list(LANG_TAGS.keys()), help="Language code")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--plain", action="store_true", help="Disable colors and styling")
    args = parser.parse_args()
    ui.set_plain(args.plain)

    if args.compare:
        print(f"[*] Loading S6 Anchor Checkpoint: {args.s6_checkpoint}")
        s6_model, s6_tok, _ = load_model(Path(args.s6_checkpoint), device=args.device)
        print(f"[*] Loading S8 Golden Checkpoint: {args.checkpoint}")
        s8_model, s8_tok, _ = load_model(Path(args.checkpoint), device=args.device)

        prompts = [args.prompt] if args.prompt else [
            "Explain what Parentheses is and how its recurrent memory works without a KV-cache.",
            "Write a Python function to check whether a word is a palindrome.",
            "Tell me about the history of the Vijayanagara Empire in Hampi.",
            "What happens if you mix vinegar and baking soda?",
        ]

        for p in prompts:
            print("=" * 70)
            print(f"PROMPT: {p}")
            print("=" * 70)
            converse_turn(s6_model, s6_tok, p, name="Session 6 (Step 12k Anchor)", lang=args.lang, temp=args.temp, max_tokens=args.max_tokens, rep_penalty=args.rep_penalty, device=args.device)
            converse_turn(s8_model, s8_tok, p, name="Session 8 (DPO+SWA-3 Golden)", lang=args.lang, temp=args.temp, max_tokens=args.max_tokens, rep_penalty=args.rep_penalty, device=args.device)
        return

    # Single model mode
    ui.banner({"checkpoint": Path(args.checkpoint).name, "device": args.device, "lang": args.lang})
    with ui.Spinner("loading weights"):
        model, tok, _ = load_model(Path(args.checkpoint), device=args.device)

    if args.prompt:
        print(ui.fg(ui.ACCENT, "( you ) ", bold=True) + ui.fg(ui.DIM, "› ") + args.prompt)
        converse_turn(model, tok, args.prompt, lang=args.lang, temp=args.temp, max_tokens=args.max_tokens, rep_penalty=args.rep_penalty, device=args.device)
        return

    if args.interactive:
        show_think = True
        while True:
            try:
                user_msg = ui.user_prompt().strip()
                if not user_msg:
                    continue
                cmd = user_msg.lower()
                if cmd in ("exit", "quit", "/exit", "/quit"):
                    ui.note("goodbye.")
                    break
                if cmd == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    ui.banner({"checkpoint": Path(args.checkpoint).name, "device": args.device, "lang": args.lang})
                    continue
                if cmd == "/think":
                    show_think = not show_think
                    ui.note(f"reasoning display {'on' if show_think else 'off'}")
                    continue
                converse_turn(model, tok, user_msg, lang=args.lang, temp=args.temp, max_tokens=args.max_tokens, rep_penalty=args.rep_penalty, device=args.device, show_think=show_think)
            except (KeyboardInterrupt, EOFError):
                print()
                ui.note("session ended.")
                break


if __name__ == "__main__":
    main()
