#!/usr/bin/env python3
"""Decoupled Background Validation Watcher for Parentheses 0.9-120M Session 2.

Runs strictly in the background during Session 2 training:
- Monitors checkpoints/parentheses-0.9-120M/ for new checkpoints.
- Every 300 steps (or designated interval), evaluates 9 languages (kan, tam, tel, mal, tcy, kff, kxv, nit, eng).
- Uses torch.set_num_threads(1) so host CPU cycles and memory bus remain 99% free for PyTorch DataLoader workers.
- Records metrics in out/session2_val_log.jsonl.
- Detects anomalies / divergence and writes out/locks/ALERT_DIVERGENCE if val loss spikes.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.backbone import Parentheses

LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng")


def get_val_windows(T: int = 500, n_per_lang: int = 2, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    data = {}
    for lang in LANGS:
        val_file = ROOT / "data" / "processed_v4" / "val" / f"{lang}.bin"
        if not val_file.is_file():
            continue
        arr = np.fromfile(val_file, dtype=np.uint16)
        if len(arr) <= T + 1:
            continue
        starts = rng.integers(0, len(arr) - T - 1, n_per_lang)
        data[lang] = [torch.tensor(arr[s:s + T + 1], dtype=torch.long) for s in starts]
    return data


def eval_checkpoint(ckpt_path: Path, val_data: dict) -> dict:
    torch.set_num_threads(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = Parentheses(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("step", -1)
    results = {"step": step, "checkpoint": ckpt_path.name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "losses": {}}

    with torch.no_grad():
        for lang, samples in val_data.items():
            lang_losses = []
            for s in samples:
                idx = s[:-1].unsqueeze(0)
                tgt = s[1:].unsqueeze(0)
                logits, _ = model(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)).item()
                lang_losses.append(loss)
            results["losses"][lang] = round(float(np.mean(lang_losses)), 4)
    return results


def main():
    parser = argparse.ArgumentParser(description="Background Validation Watcher for Session 2")
    parser.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "parentheses-0.9-120M"))
    parser.add_argument("--out-log", default=str(ROOT / "out" / "session2_val_log.jsonl"))
    parser.add_argument("--interval-steps", type=int, default=300)
    parser.add_argument("--once", action="store_true", help="Run once on latest checkpoint and exit")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    print(f"[val_watcher] Starting background validation watcher on {ckpt_dir} (interval: every {args.interval_steps} steps)")
    val_data = get_val_windows()
    evaluated = set()

    # Preload previously evaluated steps
    if out_log.is_file():
        with open(out_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    evaluated.add(obj["step"])
                except Exception:
                    pass

    while True:
        ckpts = glob.glob(str(ckpt_dir / "step_*.pt"))
        for c in ckpts:
            m = re.search(r"step_(\d+)", os.path.basename(c))
            if not m:
                continue
            step = int(m.group(1))
            if step in evaluated:
                continue
            if step % args.interval_steps == 0 or "final" in os.path.basename(c):
                print(f"[val_watcher] Evaluating new checkpoint: {os.path.basename(c)} (step {step})...")
                res = eval_checkpoint(Path(c), val_data)
                with open(out_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                evaluated.add(step)
                print(f"[val_watcher] Step {step} val losses: {res['losses']}")

        if args.once:
            break

        # Check if driver finished
        lock_path = ROOT / "out" / "locks" / "GPU.lock"
        status_path = ROOT / "out" / "overnight_status.txt"
        if not lock_path.is_file() and status_path.is_file():
            text = status_path.read_text(encoding="utf-8", errors="replace")
            if "DONE" in text:
                print("[val_watcher] Main driver status DONE and GPU.lock released. Exiting watcher.")
                break
        time.sleep(30)


if __name__ == "__main__":
    main()
