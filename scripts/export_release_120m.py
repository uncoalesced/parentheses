#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/export_release_120m.py

Packages Parentheses 0.9-120M for Hugging Face & GitHub release into:
  release/parentheses-0.9-120M/
    - model.safetensors          (fp32 weights, no optimizer state)
    - model.bf16.safetensors     (bf16 weights, ~240 MB compact release)
    - config.json                (ModelConfig + exact parameter count 119,702,272)
    - tokenizer.json             (v3_32k BPE tokenizer, sha256 7a9b4187d5bc...)
    - tokenizer_config.json      (HF PreTrainedTokenizerFast config + 19 special tokens)
    - load_example.py            (standalone loader demonstrating Chat, <think>, FreeThink, and Base modes)

Also runs the mandatory parity gate:
  - Reloads `release/parentheses-0.9-120M/model.safetensors` into a fresh `Parentheses` instance
    and verifies `max_abs_diff < 1e-5` in fp32 logits across 4 fixed prompts (eng, kan, tam, tel).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model import ModelConfig, Parentheses
from tokenizers import Tokenizer


def save_safetensors_pure(tensors: dict[str, torch.Tensor], filepath: Path, metadata: dict[str, str] | None = None):
    """Saves tensors in safetensors format. Uses official safetensors package if available, with pure-Python fast fallback."""
    try:
        import safetensors.torch
        safetensors.torch.save_file(tensors, str(filepath), metadata=metadata)
        return
    except ImportError:
        pass

    dtype_str_map = {
        torch.float32: "F32",
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.int64: "I64",
        torch.int32: "I32",
    }
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    raw_buffers = []
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().cpu().contiguous()
        dt_str = dtype_str_map[t.dtype]
        nbytes = t.numel() * t.element_size()
        header[name] = {
            "dtype": dt_str,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        # Fast zero-copy uint8 view for any dtype (instant C-level memcpy)
        buf = t.view(torch.uint8).numpy().tobytes()
        raw_buffers.append(buf)
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad

    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in raw_buffers:
            f.write(b)


LOAD_EXAMPLE_CODE = '''"""Standalone loader and inference example for Parentheses 0.9-120M.

Supports 4 generation modes out of the box across all 9 languages
(eng, kan, tam, tel, mal, tcy, kff, kxv, nit):
  1. --mode think      : Careful, structured <think>...</think> reasoning + final answer (default)
  2. --mode chat       : Direct assistant response (<|user|><lang>...<|assistant|>)
  3. --mode freethink  : Open-ended FreeThink Mode stream on a statement
  4. --mode base       : Raw <lang> text continuation

Usage:
    python load_example.py --mode think --lang eng --prompt "If 3 notebooks cost 45 rupees, how much do 7 notebooks cost?"
    python load_example.py --mode think --lang kan --prompt "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?"
    python load_example.py --mode freethink --lang eng --prompt "Recurrent state space models run in constant O(1) memory."
"""

import argparse
import dataclasses
import json
import struct
import sys
from pathlib import Path

import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if (REPO_ROOT / "model").exists() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.backbone import Parentheses
from model.config import ModelConfig


def load_safetensors_fallback(filepath: Path) -> dict[str, torch.Tensor]:
    """Loads tensors from safetensors format. Uses official library if available, with pure-Python fallback."""
    try:
        import safetensors.torch
        return safetensors.torch.load_file(str(filepath))
    except ImportError:
        pass

    with open(filepath, "rb") as f:
        length_bytes = f.read(8)
        header_len = struct.unpack("<Q", length_bytes)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data = f.read()

    dtype_map = {
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "I64": torch.int64,
        "I32": torch.int32,
    }
    tensors = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        dt = dtype_map[v["dtype"]]
        shape = v["shape"]
        start, end = v["data_offsets"]
        tensor = torch.frombuffer(bytearray(data[start:end]), dtype=dt).reshape(shape).clone()
        tensors[k] = tensor
    return tensors


def load_model(release_dir: Path, device: str = "cpu", use_bf16: bool = False) -> Parentheses:
    cfg_dict = json.loads((release_dir / "config.json").read_text(encoding="utf-8"))
    cfg_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    cfg = ModelConfig(**filtered_cfg)
    model = Parentheses(cfg)
    fname = "model.bf16.safetensors" if use_bf16 and (release_dir / "model.bf16.safetensors").exists() else "model.safetensors"
    st_path = release_dir / fname
    state_dict = load_safetensors_fallback(st_path)
    model.load_state_dict(state_dict, strict=False)
    if cfg.tie_embeddings:
        model.head.weight = model.tok_emb.weight
    model.to(device)
    model.eval()
    return model


def load_tokenizer(release_dir: Path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(release_dir / "tokenizer.json"))
    tok.encode_special_tokens = True
    return tok


def main():
    p = argparse.ArgumentParser(description="Run Parentheses 0.9-120M")
    p.add_argument("--mode", choices=["think", "chat", "freethink", "base"], default="think")
    p.add_argument("--lang", default="eng", help="eng, kan, tam, tel, mal, tcy, kff, kxv, nit")
    p.add_argument("--prompt", default="Explain why recurrent state space models use O(1) memory during inference.")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = load_model(SCRIPT_DIR, device=args.device)
    tok = load_tokenizer(SCRIPT_DIR)

    if args.mode == "freethink" and (REPO_ROOT / "features/free_think.py").exists():
        from features.free_think import FreeThinkSession
        sess = FreeThinkSession(model, temperature=args.temperature, top_k=args.top_k, tokenizer=tok, lang=args.lang)
        for chunk in sess.start(args.prompt, max_tokens=args.max_tokens, force=True):
            print(chunk, end="", flush=True)
        print()
        return

    tag_id = tok.token_to_id(f"<{args.lang}>") or tok.token_to_id("<eng>")
    user_id = tok.token_to_id("<|user|>")
    asst_id = tok.token_to_id("<|assistant|>")
    think_id = tok.token_to_id("<think>")
    stops = {tok.token_to_id(s) for s in ("<|endoftext|>", "<|user|>", "<|system|>")} - {None}

    body_ids = tok.encode(args.prompt.strip(), add_special_tokens=False).ids
    if args.mode == "base":
        input_ids = [tag_id] + body_ids
    elif args.mode == "chat":
        input_ids = [user_id, tag_id] + body_ids + [asst_id]
    else:  # think
        input_ids = [user_id, tag_id] + body_ids + [asst_id, think_id]

    idx = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    out_ids = [think_id] if args.mode == "think" else []
    for next_t in model.stream(idx, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k):
        tid = int(next_t.item())
        if tid in stops:
            break
        out_ids.append(tid)
    print(tok.decode(out_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
'''


def main():
    p = argparse.ArgumentParser(description="Export Parentheses 0.9-120M release directory")
    p.add_argument("--checkpoint", required=True, help="Path to final .pt checkpoint")
    p.add_argument("--out-dir", default=str(REPO / "release/parentheses-0.9-120M"))
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading {ckpt_path} on cpu...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: ModelConfig = ckpt["cfg"]
    model = Parentheses(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    exact_params = int(model.num_params())
    print(f"[export] Model loaded: {exact_params:,} unique parameters (step {ckpt.get('step')})", flush=True)

    # Deduplicate tied head.weight if tie_embeddings is True so safetensors stores unique tensors cleanly
    raw_sd = model.state_dict()
    clean_fp32 = {}
    clean_bf16 = {}
    for k, v in raw_sd.items():
        if cfg.tie_embeddings and k == "head.weight":
            continue
        clean_fp32[k] = v.detach().cpu().to(torch.float32).contiguous()
        clean_bf16[k] = v.detach().cpu().to(torch.bfloat16).contiguous()

    meta = {
        "format": "pt",
        "model_type": "parentheses-0.9-120M",
        "attn_type": cfg.attn_type,
        "step": str(ckpt.get("step", 0)),
        "params": str(exact_params),
    }

    fp32_path = out_dir / "model.safetensors"
    bf16_path = out_dir / "model.bf16.safetensors"
    print(f"[export] Writing {fp32_path} (fp32) and {bf16_path} (bf16)...", flush=True)
    save_safetensors_pure(clean_fp32, fp32_path, metadata=meta)
    save_safetensors_pure(clean_bf16, bf16_path, metadata=meta)

    # Write config.json
    cfg_dict = {
        "model_type": "parentheses",
        "auto_map": {
            "AutoConfig": "configuration_parentheses.ParenthesesConfig",
            "AutoModelForCausalLM": "modeling_parentheses.ParenthesesForCausalLM",
        },
        **dataclasses.asdict(cfg),
        "params": exact_params,
        "step": int(ckpt.get("step", 0)),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    # Copy tokenizer.json and tokenizer_config.json
    shutil.copy2(REPO / "data/tokenizer/v3_32k/tokenizer.json", out_dir / "tokenizer.json")
    tok_cfg_src = REPO / "release/parentheses-0.9-41M/tokenizer_config.json"
    if tok_cfg_src.exists():
        shutil.copy2(tok_cfg_src, out_dir / "tokenizer_config.json")

    # Write load_example.py
    (out_dir / "load_example.py").write_text(LOAD_EXAMPLE_CODE, encoding="utf-8")

    # --- Parity Gate: reload from release/parentheses-0.9-120M and verify max_abs_diff < 1e-5 ---
    print("[export] Running 4-prompt fp32 logit parity gate...", flush=True)
    sys.path.insert(0, str(out_dir))
    import load_example as le

    reloaded = le.load_model(out_dir, device="cpu", use_bf16=False)
    tok = le.load_tokenizer(out_dir)

    test_prompts = [
        "<|user|><eng>Explain how recurrent state space models work.<|assistant|><think>",
        "<|user|><kan>ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ನದಿಗಳು ಯಾವುವು?<|assistant|><think>",
        "<|user|><tam>திருக்குறளில் எத்தனை அதிகாரங்கள் உள்ளன?<|assistant|><think>",
        "<|user|><tel>గోదావరి నది ఎక్కడ పుడుతుంది?<|assistant|><think>",
    ]
    max_diffs = []
    with torch.no_grad():
        for p_str in test_prompts:
            ids = torch.tensor([tok.encode(p_str, add_special_tokens=False).ids], dtype=torch.long)
            logits_orig, _ = model(ids, return_logits=True)
            logits_rel, _ = reloaded(ids, return_logits=True)
            diff = float((logits_orig - logits_rel).abs().max().item())
            max_diffs.append(diff)
            assert diff < 1e-5, f"Parity gate failed on {p_str!r}: max_abs_diff={diff}"

    print(f"[PASS] 4-prompt fp32 logit parity gate: max_abs_diff = {max(max_diffs):.2e} (< 1e-5)", flush=True)
    print(f"[export] Release artifacts written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
