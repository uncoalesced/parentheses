#!/usr/bin/env python3
# Engineered by uncoalesced
"""Bits per UTF-8 byte for slab_v4 (240M) checkpoints, per language, on the processed_v5 val split.

eval_dravidian.py builds `Parentheses` and so cannot load slab_v4; this reuses its doc_stream / sum_nats
with SlabV4Model (strict load). bpb only: no tag-generation tests (SlabV4Model has no generate()).
    .venv\\Scripts\\python.exe scripts\\bpb_slab_v4.py --checkpoints a.pt b.pt --out out\\bpb_240m.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_slab_v4 import SlabV4Model  # noqa: E402
from scripts.eval_dravidian import BPB_LANGS, doc_stream, sum_nats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--bpb-bytes", type=int, default=400_000, help="val bytes per language")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(ROOT / "out" / "bpb_240m.json"))
    args = p.parse_args()

    # The old byte-level val split (data/processed_dravidian_v1) no longer exists, so score the
    # processed_v5 val split: docs are "<lang> text <|endoftext|>", decoded back to text per language.
    from tokenizers import Tokenizer
    vtok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))
    tag_of = {vtok.token_to_id(f"<{l}>"): l for l in BPB_LANGS if vtok.token_to_id(f"<{l}>") is not None}
    val = np.memmap(ROOT / "data" / "processed_v5" / "val.bin", dtype=np.uint16, mode="r")
    vb = np.load(ROOT / "data" / "processed_v5" / "val_boundaries.npy")
    ve = np.append(vb[1:], len(val))
    texts, nbytes = {}, {}
    eot = vtok.token_to_id("<|endoftext|>")
    for s, e in zip(vb, ve):
        doc = np.asarray(val[s:e]).tolist()
        lang = tag_of.get(doc[0]) if doc else None
        if lang is None or nbytes.get(lang, 0) >= args.bpb_bytes:
            continue
        body = [t for t in doc[1:] if t != eot]
        t = vtok.decode(body, skip_special_tokens=False)
        texts.setdefault(lang, []).append(t)
        nbytes[lang] = nbytes.get(lang, 0) + len(t.encode("utf-8"))

    # Load held-out Devanagari validation texts (hin, san) from data/val_candidates or I:
    for dev_lang in ["hin", "san"]:
        cand_p = ROOT / "data" / "val_candidates" / f"{dev_lang}_val.txt"
        if not cand_p.exists():
            cand_p = Path(rf"I:\parentheses\val_candidates\{dev_lang}_val.txt")
        if cand_p.exists():
            raw = cand_p.read_text(encoding="utf-8")
            blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
            for b in blocks:
                if nbytes.get(dev_lang, 0) >= args.bpb_bytes:
                    break
                texts.setdefault(dev_lang, []).append(b)
                nbytes[dev_lang] = nbytes.get(dev_lang, 0) + len(b.encode("utf-8"))

    print("[info] val bytes per lang: " + " ".join(f"{l}:{n}" for l, n in nbytes.items()), flush=True)

    results = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        cfg = ckpt["cfg"]
        model = SlabV4Model(cfg).to(args.device).eval()
        model.load_state_dict(ckpt["model"], strict=True)
        tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))
        tok.encode_special_tokens = True
        res = {"checkpoint": path, "step": ckpt.get("step"), "bpb": {}}
        for lang in BPB_LANGS:
            if lang not in texts:
                continue
            ids, score, starts = doc_stream(texts[lang], lang, tok)
            nats, _ = sum_nats(model, ids, score, starts, cfg.block_size, args.batch, args.device)
            res["bpb"][lang] = nats / sum(len(t.encode("utf-8")) + 1 for t in texts[lang]) / np.log(2)
        print(f"[bpb] step {res['step']}: " + " ".join(f"{l} {v:.3f}" for l, v in res["bpb"].items()), flush=True)
        results.append(res)
        del model, ckpt
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[info] wrote {args.out}")


if __name__ == "__main__":
    main()
