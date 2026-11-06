#!/usr/bin/env python3
"""
Telemetry & Statistics Exporter for Parentheses 1.0-240M Session 4 (Steps 6,001 -> 8,000)
Extracts step-level algorithmic metrics and 15s hardware telemetry into clean JSONL and CSV formats
for post-training analysis, statistical charting, and plotting.
"""

import csv
import datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out"


def export_stats():
    metrics_src = OUT_DIR / "session1_240m_metrics.jsonl"
    telemetry_src = OUT_DIR / "telemetry_s3_live.jsonl"

    metrics_out_jsonl = OUT_DIR / "session4_metrics.jsonl"
    metrics_out_csv = OUT_DIR / "session4_metrics.csv"
    telemetry_out_csv = OUT_DIR / "session4_telemetry.csv"

    # 1. Parse step-by-step training metrics (step >= 6001)
    s4_metrics = []
    if metrics_src.exists():
        with open(metrics_src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("step", 0) >= 6001:
                        s4_metrics.append(record)
                except Exception:
                    pass

    # Deduplicate by step (latest wins)
    step_map = {}
    for rec in s4_metrics:
        step_map[rec["step"]] = rec
    sorted_steps = sorted(step_map.keys())
    deduped_metrics = [step_map[s] for s in sorted_steps]

    # Write session4_metrics.jsonl
    with open(metrics_out_jsonl, "w", encoding="utf-8") as f:
        for rec in deduped_metrics:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write session4_metrics.csv
    if deduped_metrics:
        fieldnames = ["step", "loss", "lr", "grad_norm", "tok_per_sec", "beta_write", "v0_mix", "vram_gb", "val_loss", "val_ppl"]
        with open(metrics_out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in deduped_metrics:
                writer.writerow(rec)

    # 2. Parse 15-second hardware telemetry (launch timestamp >= 2026-11-02T21:16:00)
    launch_threshold = "2026-11-02T21:16:00"
    s4_telemetry = []
    if telemetry_src.exists():
        with open(telemetry_src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pt = json.loads(line)
                    ts = pt.get("timestamp", "")
                    if ts >= launch_threshold:
                        gpu = pt.get("gpu", {})
                        disks = pt.get("disks", {})
                        row = {
                            "timestamp": ts,
                            "step": pt.get("step"),
                            "loss": pt.get("loss"),
                            "lr": pt.get("lr"),
                            "grad_norm": pt.get("grad_norm"),
                            "v0_mix": pt.get("v0_mix"),
                            "beta_write": pt.get("beta_write"),
                            "val_loss": pt.get("val_loss"),
                            "val_ppl": pt.get("val_ppl"),
                            "sec_per_step": round(pt.get("sec_per_step", 0.0), 2) if pt.get("sec_per_step") else None,
                            "effective_tok_s": pt.get("effective_tok_s"),
                            "temp_c": gpu.get("temp_c"),
                            "gpu_util_pct": gpu.get("gpu_util_pct"),
                            "mem_util_pct": gpu.get("mem_util_pct"),
                            "vram_used_mb": gpu.get("vram_used_mb"),
                            "power_w": gpu.get("power_w"),
                            "c_free_gb": disks.get("C", {}).get("free_gb"),
                            "e_free_gb": disks.get("E", {}).get("free_gb"),
                            "i_free_gb": disks.get("I", {}).get("free_gb"),
                        }
                        s4_telemetry.append(row)
                except Exception:
                    pass

    # Write session4_telemetry.csv
    if s4_telemetry:
        tel_fields = [
            "timestamp", "step", "loss", "lr", "grad_norm", "v0_mix", "beta_write",
            "val_loss", "val_ppl", "sec_per_step", "effective_tok_s",
            "temp_c", "gpu_util_pct", "mem_util_pct", "vram_used_mb", "power_w",
            "c_free_gb", "e_free_gb", "i_free_gb"
        ]
        with open(telemetry_out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=tel_fields, extrasaction="ignore")
            writer.writeheader()
            for row in s4_telemetry:
                writer.writerow(row)

    # 3. Print Statistical Summary
    total_steps = len(deduped_metrics)
    if total_steps > 0:
        first_step = deduped_metrics[0]["step"]
        last_step = deduped_metrics[-1]["step"]
        losses = [m["loss"] for m in deduped_metrics if m.get("loss") is not None]
        gnorms = [m["grad_norm"] for m in deduped_metrics if m.get("grad_norm") is not None]
        tps = [m["tok_per_sec"] for m in deduped_metrics if m.get("tok_per_sec") is not None]

        min_loss = min(losses) if losses else 0.0
        start_loss = losses[0] if losses else 0.0
        end_loss = losses[-1] if losses else 0.0
        avg_gnorm = sum(gnorms) / len(gnorms) if gnorms else 0.0
        avg_tps = sum(tps) / len(tps) if tps else 0.0

        temps = [t["temp_c"] for t in s4_telemetry if t.get("temp_c") is not None]
        powers = [t["power_w"] for t in s4_telemetry if t.get("power_w") is not None]
        vrams = [t["vram_used_mb"] for t in s4_telemetry if t.get("vram_used_mb") is not None]

        max_temp = max(temps) if temps else 0
        avg_temp = sum(temps) / len(temps) if temps else 0.0
        avg_power = sum(powers) / len(powers) if powers else 0.0
        max_vram = max(vrams) if vrams else 0

        print(f"=== PARENTHESES 1.0-240M: SESSION 4 TELEMETRY EXPORT ===")
        print(f"Step Range: {first_step} -> {last_step} ({total_steps} total steps recorded)")
        print(f"Loss: Start = {start_loss:.4f} | Current = {end_loss:.4f} | Min = {min_loss:.4f}")
        print(f"Mean Gradient Norm: {avg_gnorm:.4f}")
        print(f"Mean Throughput: {avg_tps:.1f} tok/s")
        print(f"Hardware: Avg Temp = {avg_temp:.1f}°C (Peak {max_temp}°C) | Avg Power = {avg_power:.1f}W | Peak VRAM = {max_vram} MB")
        print(f"Exported Files:")
        print(f"  - {metrics_out_jsonl} ({total_steps} records)")
        print(f"  - {metrics_out_csv} ({total_steps} rows)")
        print(f"  - {telemetry_out_csv} ({len(s4_telemetry)} 15s telemetry samples)")


if __name__ == "__main__":
    export_stats()
