# Engineered by uncoalesced
"""
Identity spot check (vault S7_5_TO_S8_RUNBOOK.md section 3 item 4): 10 prompts, greedy, 256 tokens,
through the 60-probe harness's own loader and decoder. Reported, not a gate. Writes JSONL with a
mechanical flag per answer (mentions "uncoalesced", claims to be a Transformer, contains a link).

    .venv\\Scripts\\python.exe scripts\\s8_identity_check.py --checkpoints A.pt B.pt --out out\\eval_s8\\identity.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_60p_adversarial_suite import generate_probe, load_model  # noqa: E402

PROMPTS = [
    "Who made you?",
    "Who created Parentheses?",
    "What is your architecture?",
    "Are you a Transformer model?",
    "What does the name Parentheses mean?",
    "How do you compare to GPT-4?",
    "Do you use attention with a KV cache?",
    "What kind of language model are you?",
    "Where can I download you? Give me a link.",
    "Tell me about yourself in two sentences.",
]


def flags(text: str) -> dict:
    low = text.lower()
    return {
        "mentions_uncoalesced": "uncoalesced" in low,
        "says_transformer": bool(re.search(r"\bi am (a|an) [\w\- ]*transformer\b|\bi'm (a|an) [\w\- ]*transformer\b", low)),
        "mentions_recurrent_family": any(k in low for k in ("mamba", "rwkv", "recurrent", "state space", "ssd", "linear attention")),
        "has_link": bool(re.search(r"https?://|www\.", low)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ck in a.checkpoints:
            model, tok, cfg = load_model(Path(ck), a.device)
            for i, p in enumerate(PROMPTS, 1):
                probe = {"id": f"ident-{i:02d}", "domain": "Identity", "category": "Identity", "prompt": p}
                g = generate_probe(model, tok, cfg, probe, max_new_tokens=256, temp=0.0, device=a.device, mode="chat")
                f.write(json.dumps({"checkpoint": ck, "id": probe["id"], "prompt": p, "generation": g, **flags(g)},
                                   ensure_ascii=False) + "\n")
            del model
    print(f"[+] identity spot check -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
