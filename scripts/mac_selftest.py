# Engineered by uncoalesced
"""Apple Silicon (MPS) self-test for Parentheses 1.0-240M (Tanzanite).

Run from the repository root after installing requirements-inference.txt and
downloading config.json, model.safetensors and tokenizer.json into
release/parentheses-1.0-240M-tanzanite:

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/mac_selftest.py

What it does:
  1. Prints platform, Python, torch version and whether MPS is built/available.
  2. Loads the float32 release checkpoint.
  3. Greedy-decodes N tokens (default 16) on CPU, recording the logits at every step.
  4. If MPS is available: replays the same token sequence on MPS (teacher-forced, so one
     near-tie cannot cascade), compares logits against CPU within a tolerance, then runs a
     free greedy decode on MPS for a tokens/s figure.
  5. Ends with one PASS/FAIL line and a report block to copy-paste back.

If MPS is unavailable the test runs on CPU only and says so.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "release" / "parentheses-1.0-240M-tanzanite"
PROMPT = "Write a Python function that returns the factorial of n."


def mps_status() -> tuple[bool, bool]:
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False, False
    return bool(mps.is_built()), bool(mps.is_available())


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def load_model(ckpt_dir: Path):
    from models.model_slab_v4 import SlabV4Config, SlabV4Model
    import safetensors.torch
    from tokenizers import Tokenizer

    cfg_dict = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    cfg = SlabV4Config(
        vocab_size=cfg_dict.get("vocab_size", 32768),
        block_size=cfg_dict.get("block_size", 4096),
        n_layer=cfg_dict.get("n_layer", 16),
        n_head=cfg_dict.get("n_head", 16),
        n_embd=cfg_dict.get("n_embd", 1024),
        chunk_size=cfg_dict.get("chunk_size", 64),
        eot_token_id=cfg_dict.get("eot_token_id", 0),
        dropout=0.0,
        bias=cfg_dict.get("bias", False),
        tie_embeddings=cfg_dict.get("tie_embeddings", True),
        ignore_index=cfg_dict.get("ignore_index", -100),
        slab_v4=True,
    )
    sd = safetensors.torch.load_file(str(ckpt_dir / "model.safetensors"))
    if "tok_emb.weight" not in sd and "head.weight" in sd:
        sd["tok_emb.weight"] = sd["head.weight"]
    model = SlabV4Model(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()
    tok = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))
    return model, tok


def prompt_ids(tok) -> list[int]:
    ids = [tok.token_to_id("<|user|>"), tok.token_to_id("<eng>")]
    ids += tok.encode(PROMPT, add_special_tokens=False).ids
    ids.append(tok.token_to_id("<|assistant|>"))
    if any(i is None for i in ids):
        raise RuntimeError("tokenizer.json is missing a special token (<|user|>, <eng> or <|assistant|>)")
    return ids


@torch.no_grad()
def run(model, ids: list[int], n_tokens: int, device: str, forced: list[int] | None = None):
    """Prefill the prompt, then decode n_tokens one step at a time.

    If `forced` is given, those tokens are fed back instead of the model's own argmax
    (teacher forcing), so logits can be compared step-for-step against another device.
    Returns (generated_tokens, per-step logits on CPU, decode seconds).
    """
    model.to(device)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    states = model.empty_states(1, device=device)
    for i in range(inp.size(1) - 1):
        _, states = model.step(inp[:, i:i + 1], states)
    curr = inp[:, -1:]
    out_tokens, out_logits = [], []
    sync(device)
    t0 = time.perf_counter()
    for t in range(n_tokens):
        logits, states = model.step(curr, states)
        row = logits.reshape(-1, logits.size(-1))[0].float()
        nxt = int(torch.argmax(row).item()) if forced is None else forced[t]
        out_tokens.append(nxt)
        out_logits.append(row.cpu())
        curr = torch.tensor([[nxt]], dtype=torch.long, device=device)
    sync(device)
    dt = time.perf_counter() - t0
    model.to("cpu")
    return out_tokens, torch.stack(out_logits), dt


def compare(ref: torch.Tensor, other: torch.Tensor, tol: float) -> dict:
    diff = (ref - other).abs()
    max_abs = float(diff.max().item())
    ref_top = ref.argmax(-1)
    oth_top = other.argmax(-1)
    mismatches = []
    for t in range(ref.size(0)):
        if int(ref_top[t]) != int(oth_top[t]):
            top2 = torch.topk(ref[t], 2).values
            margin = float(top2[0] - top2[1])
            mismatches.append({"step": t, "margin": round(margin, 5)})
    # An argmax flip only counts as a failure when CPU's top-1 lead is clearly larger
    # than the numerical disagreement (i.e. not a near-tie).
    hard = [m for m in mismatches if m["margin"] > 2 * max(max_abs, tol)]
    finite = bool(torch.isfinite(other).all().item())
    return {
        "max_abs_logit_diff": max_abs,
        "argmax_agree": f"{ref.size(0) - len(mismatches)}/{ref.size(0)}",
        "near_tie_flips": len(mismatches) - len(hard),
        "hard_flips": len(hard),
        "finite": finite,
        "ok": finite and max_abs <= tol and not hard,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parentheses 1.0-240M Apple Silicon self-test")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                    help="release folder with config.json, model.safetensors, tokenizer.json")
    ap.add_argument("--tokens", type=int, default=16, help="greedy tokens to generate (default 16)")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="max allowed |logit_cpu - logit_accel| (default 1e-2, float32)")
    ap.add_argument("--compare-device", default="mps",
                    help="accelerator to compare against CPU (default mps; 'cuda' exercises the same path on NVIDIA)")
    ap.add_argument("--cpu-only", action="store_true", help="skip the accelerator comparison")
    args = ap.parse_args()

    built, avail = mps_status()
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or "n/a",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_built": built,
        "mps_available": avail,
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "checkpoint": str(args.checkpoint),
        "tokens": args.tokens,
        "tol": args.tol,
    }
    print("== Parentheses 1.0-240M self-test ==")
    for k in ("platform", "machine", "macos", "python", "torch", "mps_built", "mps_available",
              "PYTORCH_ENABLE_MPS_FALLBACK"):
        print(f"{k:28s} {info[k]}")

    accel = None
    if not args.cpu_only:
        if args.compare_device == "mps":
            accel = "mps" if avail else None
        elif args.compare_device.startswith("cuda"):
            accel = args.compare_device if torch.cuda.is_available() else None
        else:
            accel = args.compare_device
        if accel is None:
            print(f"\n[note] {args.compare_device} is NOT available on this machine; running CPU only.")
            if args.compare_device == "mps" and platform.system() == "Darwin" and platform.machine() != "arm64":
                print("[note] machine is not arm64: Intel Mac, or Python running under Rosetta.")
    info["accelerator"] = accel or "none (CPU only)"

    verdict, reasons = "PASS", []
    try:
        for f in ("config.json", "model.safetensors", "tokenizer.json"):
            if not (args.checkpoint / f).exists():
                raise FileNotFoundError(f"{args.checkpoint / f} is missing; download it from Hugging Face first")
        t0 = time.perf_counter()
        model, tok = load_model(args.checkpoint)
        info["load_s"] = round(time.perf_counter() - t0, 2)
        info["params"] = sum(p.numel() for p in model.parameters())
        print(f"\nloaded {info['params']:,} params in {info['load_s']} s")
        ids = prompt_ids(tok)

        cpu_toks, cpu_logits, cpu_dt = run(model, ids, args.tokens, "cpu")
        info["cpu_tok_per_s"] = round(args.tokens / cpu_dt, 2)
        info["cpu_text"] = tok.decode(cpu_toks, skip_special_tokens=False)
        print(f"cpu: {info['cpu_tok_per_s']} tok/s  text={info['cpu_text']!r}")
        if not bool(torch.isfinite(cpu_logits).all().item()):
            verdict = "FAIL"
            reasons.append("CPU logits contain NaN/Inf")

        if accel:
            # Warm-up: first accelerator steps include kernel compilation.
            run(model, ids, 2, accel)
            _, acc_logits, _ = run(model, ids, args.tokens, accel, forced=cpu_toks)
            cmp = compare(cpu_logits, acc_logits, args.tol)
            info["compare"] = cmp
            print(f"{accel} vs cpu: max|dlogit|={cmp['max_abs_logit_diff']:.3e}  argmax {cmp['argmax_agree']}  "
                  f"near-tie flips={cmp['near_tie_flips']}  hard flips={cmp['hard_flips']}  finite={cmp['finite']}")
            acc_toks, _, acc_dt = run(model, ids, args.tokens, accel)
            info[f"{accel}_tok_per_s"] = round(args.tokens / acc_dt, 2)
            info[f"{accel}_text"] = tok.decode(acc_toks, skip_special_tokens=False)
            info["free_greedy_matches_cpu"] = acc_toks == cpu_toks
            print(f"{accel}: {info[accel + '_tok_per_s']} tok/s  text={info[accel + '_text']!r}")
            if not cmp["ok"]:
                verdict = "FAIL"
                if not cmp["finite"]:
                    reasons.append(f"{accel} logits contain NaN/Inf")
                if cmp["max_abs_logit_diff"] > args.tol:
                    reasons.append(f"{accel} logits differ from CPU by {cmp['max_abs_logit_diff']:.3e} > tol {args.tol}")
                if cmp["hard_flips"]:
                    reasons.append(f"{cmp['hard_flips']} greedy token(s) differ from CPU outside a near-tie")
    except Exception as e:  # report every failure in the copy-paste block
        verdict = "FAIL"
        first = (str(e).strip().splitlines() or [""])[0]
        reasons.append(f"{type(e).__name__}: {first}")
        info["traceback"] = traceback.format_exc()[-1500:]
        print(info["traceback"], file=sys.stderr)

    scope = f"cpu+{accel}" if accel else "cpu-only"
    info["verdict"] = verdict
    info["reasons"] = reasons
    print("\n----- BEGIN PARENTHESES SELFTEST REPORT (copy everything down to END) -----")
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print("----- END PARENTHESES SELFTEST REPORT -----")
    line = f"{verdict} ({scope})"
    if reasons:
        line += ": " + "; ".join(reasons)
    print(line)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
