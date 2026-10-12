#!/usr/bin/env python3
"""Decoupled Background Validation Watcher for Parentheses 0.9-120M Session 2.

Runs strictly in the background during Session 2 training:
- Monitors checkpoints/parentheses-0.9-120M/ for new checkpoints.
- Every 300 steps (or designated interval), evaluates 9 languages (kan, tam, tel, mal, tcy, kff, kxv, nit, eng).
- Uses torch.set_num_threads(1) so host CPU cycles and memory bus remain 99% free for PyTorch DataLoader workers.
- Records metrics in out/session2_val_log.jsonl.
- Writes out/locks/ALERT_DIVERGENCE (alert only, training is not touched) when a language's
  val loss exceeds --alert-ratio x its loss on the first (baseline) checkpoint.
- Skips checkpoints below --min-step, so session-1 checkpoints are not re-evaluated.
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


def get_val_windows(T: int = 1024, n_per_lang: int = 4, seed: int = 1234) -> dict:
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
    parser.add_argument("--min-step", type=int, default=1907, help="skip checkpoints below this step")
    parser.add_argument("--alert-ratio", type=float, default=1.10)
    parser.add_argument("--once", action="store_true", help="Run one pass over existing checkpoints and exit")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    alert_path = ROOT / "out" / "locks" / "ALERT_DIVERGENCE"
    status_path = ROOT / "out" / "overnight_status.txt"

    print(f"[val_watcher] watching {ckpt_dir} (every {args.interval_steps} steps, from step {args.min_step})")
    val_data = get_val_windows()
    evaluated, baseline = set(), None
    if out_log.is_file():
        for line in out_log.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            evaluated.add(obj["step"])
            if baseline is None and obj["step"] >= args.min_step:
                baseline = obj["losses"]

    started, seen_running = time.time(), False
    while True:
        found = []
        for c in glob.glob(str(ckpt_dir / "step_*.pt")):
            m = re.search(r"step_(\d+)", os.path.basename(c))
            if m:
                found.append((int(m.group(1)), c))
        for step, c in sorted(found):
            if step < args.min_step or step in evaluated:
                continue
            if step % args.interval_steps == 0 or "final" in os.path.basename(c):
                print(f"[val_watcher] evaluating {os.path.basename(c)} (step {step})...", flush=True)
                res = eval_checkpoint(Path(c), val_data)
                if baseline is None:
                    baseline = res["losses"]
                bad = {k: (v, baseline[k]) for k, v in res["losses"].items()
                       if k in baseline and v > args.alert_ratio * baseline[k]}
                res["alert"] = bool(bad)
                with open(out_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                evaluated.add(step)
                print(f"[val_watcher] step {step} val losses: {res['losses']}", flush=True)
                if bad:
                    alert_path.parent.mkdir(parents=True, exist_ok=True)
                    alert_path.write_text(f"step {step} {os.path.basename(c)}: (loss, baseline) "
                                          f"{json.dumps(bad)}\n", encoding="utf-8")
                    print(f"[val_watcher] ALERT: {bad}", flush=True)

        if args.once:
            break
        # status still says DONE from the previous session until the driver's first 60 s
        # write, so only trust DONE after a fresh non-DONE status was seen
        if status_path.is_file():
            text = status_path.read_text(encoding="utf-8", errors="replace")
            if status_path.stat().st_mtime > started and "DONE" not in text:
                seen_running = True
            elif seen_running and "DONE" in text and not any(
                    s not in evaluated and s >= args.min_step and "final" in c for s, c in found):
                print("[val_watcher] driver DONE; exiting.", flush=True)
                break
        time.sleep(30)


if __name__ == "__main__":
    main()
