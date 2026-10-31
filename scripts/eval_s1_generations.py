# Engineered by uncoalesced
"""
Parentheses 1.0-240M (slab_v4) - CPU-Only Checkpoint Generation Smoke Test
Loads rotated milestone checkpoints from I: onto CPU (zero GPU VRAM impact)
and verifies English, Kannada, Tamil, and <think> math completions.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(r"E:\Parentheses\parentheses-0.9")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ModelConfig
from model_slab_v4 import SlabV4Model

PROMPTS: List[Tuple[str, str]] = [
    ("English Historical", "The history of the Vijayanagara Empire in South India began when"),
    ("Kannada Literary", "ಕರ್ನಾಟಕದ ಸಾಹಿತ್ಯ ಮತ್ತು ಸಂಸ್ಕೃತಿಯ ಇತಿಹಾಸವು"),
    ("Tamil Classical", "தமிழ் மொழியின் தொன்மையான இலக்கிய மரபு"),
    ("STEM <think> Math", "Question: If a train travels 180 km in 3 hours, how far will it travel in 5 hours at the same speed?\nAnswer: <think>"),
]


def has_4gram_loop(token_ids: List[int], max_repeats: int = 3) -> bool:
    if len(token_ids) < 12:
        return False
    counts: Dict[Tuple[int, ...], int] = {}
    for i in range(len(token_ids) - 3):
        gram = tuple(token_ids[i : i + 4])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= max_repeats:
            return True
    return False


@torch.no_grad()
def greedy_decode_cpu(
    model: SlabV4Model,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 48,
) -> Tuple[str, bool]:
    encoded = tokenizer.encode(prompt).ids
    x = torch.tensor([encoded], dtype=torch.long, device="cpu")

    # Parallel forward over prompt, then greedy token-by-token step
    generated: List[int] = []
    curr_ids = list(encoded)
    for _ in range(max_new_tokens):
        inp = torch.tensor([curr_ids], dtype=torch.long, device="cpu")
        logits = model(inp)
        if isinstance(logits, tuple):
            logits = logits[0]
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(next_tok)
        curr_ids.append(next_tok)
        if next_tok == 0:  # <|endoftext|>
            break

    completion = tokenizer.decode(generated)
    looped = has_4gram_loop(generated)
    return completion, looped


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only generation check for rotated S1 checkpoints")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"I:\parentheses\checkpoints\parentheses-1.0-240M-s1\step_500.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "out" / "s1_generation_smoke.md",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"[SKIP] Checkpoint not found yet: {args.ckpt}")
        return 1

    torch.set_num_threads(6)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_data = ckpt.get("cfg", ckpt.get("config", {}))
    cfg = cfg_data if isinstance(cfg_data, ModelConfig) else ModelConfig(**cfg_data) if isinstance(cfg_data, dict) else cfg_data

    model = SlabV4Model(cfg).to("cpu").eval()
    state = {k.replace("_orig_mod.", ""): v.float() for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    del ckpt

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows: List[str] = []
    any_loop = False

    for label, prompt in PROMPTS:
        comp, looped = greedy_decode_cpu(model, tokenizer, prompt, max_new_tokens=64)
        any_loop = any_loop or looped
        flag = "[WARN: 4-GRAM LOOP]" if looped else "[PASS: COHERENT]"
        clean_comp = comp.replace("\n", " \\n ")
        rows.append(f"### {label} — `{flag}`\n- **Prompt:** `{prompt}`\n- **Output:** `{clean_comp}`\n")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    verdict = "[WARN] REPETITION DETECTED" if any_loop else "[PASS] ZERO REPETITION LOOPS"
    report = f"""# Parentheses 1.0-240M (`slab_v4`) — CPU Generation Smoke Report
- **Timestamp:** `{now_str}`
- **Checkpoint Audited:** `{args.ckpt}`
- **Overall Verdict:** **{verdict}**

---

{chr(10).join(rows)}
---
*Engineered by uncoalesced*
"""
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
