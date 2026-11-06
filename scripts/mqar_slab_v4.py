#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Real MQAR (multi-query associative recall) for the slab_v4 delta-rule A/B (Claude Code, 2026-11-04).

scripts/mqar_eval.py's MQAR battery does not score the model: it runs a forward pass, drops the logits
and grades a hand-written simulated recurrence over random embeddings. This script uses its batch
generator (generate_mqar_batch) but trains a tiny slab_v4 per arm on the task and scores the model's
own argmax at every query position (Zoology-style protocol).

Arms differ only by delta_rule / beta_mode; same init seed, same batches, same steps.
    .venv\\Scripts\\python.exe scripts\\mqar_slab_v4.py --arms A0,A1,A2 --out out\\delta_ab\\mqar.json
    .venv\\Scripts\\python.exe scripts\\mqar_slab_v4.py --self-test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from model.config import ModelConfig  # noqa: E402
from models.model_slab_v4 import SlabV4Model  # noqa: E402
from scripts.mqar_eval import generate_mqar_batch  # noqa: E402

ARMS = {"A0": (False, "2sig"), "A1": (True, "2sig"), "A2": (True, "sig05")}


def mqar_cfg(n_layer: int, n_embd: int, n_head: int, seq_len: int, delta: bool, beta_mode: str) -> ModelConfig:
    return ModelConfig(vocab_size=256, block_size=seq_len, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                       attn_type="selective_linear", slab_v2=True, slab_v3=True, slab_v4=True,
                       tie_embeddings=True, bias=False, dropout=0.0, delta_rule=delta, beta_mode=beta_mode)


def batch(kv_counts: list[int], bs: int, seq_len: int, device: str, seed: int):
    random.seed(seed)
    n = kv_counts[seed % len(kv_counts)]
    tok, tgt, _, _ = generate_mqar_batch(batch_size=bs, seq_len=seq_len, num_kv=n, device="cpu")
    return tok.to(device), tgt.to(device)


@torch.no_grad()
def accuracy(model, n: int, bs: int, seq_len: int, device: str, trials: int, seed: int) -> float:
    model.eval()
    hit = tot = 0
    for t in range(trials):
        random.seed(10_000_000 + seed * 1000 + t)
        tok, tgt, _, _ = generate_mqar_batch(batch_size=bs, seq_len=seq_len, num_kv=n, device="cpu")
        tok, tgt = tok.to(device), tgt.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(tok)
        m = tgt != -100
        hit += int((logits.argmax(-1)[m] == tgt[m]).sum())
        tot += int(m.sum())
    model.train()
    return hit / max(1, tot)


def run_arm(arm: str, a, device: str) -> dict:
    delta, beta_mode = ARMS[arm]
    torch.manual_seed(a.seed)
    model = SlabV4Model(mqar_cfg(a.layers, a.d_model, a.heads, a.seq_len, delta, beta_mode)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.98))
    kv = [int(x) for x in a.train_kv.split(",")]
    t0, losses = time.time(), []
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * (1 - step / a.steps))
        tok, tgt = batch(kv, a.batch, a.seq_len, device, a.seed * 100_000 + step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(tok, targets=tgt)
        assert torch.isfinite(loss), f"{arm}: non-finite loss at step {step}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, a.steps // 10) == 0:
            print(f"[{arm}] step {step} loss {loss.item():.4f}", flush=True)
    res = {"delta_rule": delta, "beta_mode": beta_mode, "params": model.num_params(),
           "final_loss_mean_last50": sum(losses[-50:]) / len(losses[-50:]),
           "train_sec": round(time.time() - t0, 1), "acc": {}}
    for n in [int(x) for x in a.eval_kv.split(",")]:
        res["acc"][f"N_{n:02d}"] = round(accuracy(model, n, a.batch, a.seq_len, device, a.trials, a.seed), 4)
    print(f"[{arm}] {res['acc']}", flush=True)
    return res


def self_test() -> None:
    a = argparse.Namespace(layers=1, d_model=64, heads=2, seq_len=64, lr=1e-3, warmup=2, steps=4, batch=4,
                           train_kv="4", eval_kv="4", trials=1, seed=1)
    for arm in ARMS:
        r = run_arm(arm, a, "cpu")
        assert 0.0 <= r["acc"]["N_04"] <= 1.0 and r["final_loss_mean_last50"] > 0
    # identical init across arms: same seed -> same parameters (delta/beta_mode add none)
    torch.manual_seed(0); m0 = SlabV4Model(mqar_cfg(1, 64, 2, 64, False, "2sig"))
    torch.manual_seed(0); m1 = SlabV4Model(mqar_cfg(1, 64, 2, 64, True, "sig05"))
    assert all(torch.equal(p, q) for p, q in zip(m0.parameters(), m1.parameters()))
    print("mqar_slab_v4 self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,A1,A2")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", dest="d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--train-kv", dest="train_kv", default="8,16,32,64")
    ap.add_argument("--eval-kv", dest="eval_kv", default="8,16,32,64")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "delta_ab" / "mqar.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    results = {"config": {k: str(v) for k, v in vars(a).items()}, "arms": {}}
    for arm in a.arms.split(","):
        results["arms"][arm] = run_arm(arm, a, a.device)
        a.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: v["acc"] for k, v in results["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
