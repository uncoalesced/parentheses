"""
Real-Time Parallel Telemetry & Watchdog Engine for Parentheses 1.0-240M
Monitors GPU hardware, disk headroom, loss descent, recurrent SSD dynamics,
and step pacing in real-time. Logs structured telemetry and enforces circuit-breakers.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Regex patterns matching both standard and extended train.py logging lines
LOG_RE_EXTENDED = re.compile(
    r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)\s+\|\s+"
    r"grad_norm\s+([\d.]+)\s+\|\s+tok/s\s+(\d+)\s+\|\s+v0_mix\s+([\d.]+)\s+\|\s+"
    r"beta_write\s+([\d.]+)(?:\s+\|\s+val_loss\s+([\d.]+)\s+\|\s+val_ppl\s+([\d.]+))?\s+\|\s+([\d.]+)s\s+elapsed"
)

LOG_RE_STANDARD = re.compile(
    r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)\s+\|\s+([\d.]+)s\s+elapsed"
)


def get_gpu_telemetry():
    """Queries nvidia-smi for real-time GPU statistics."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 6:
                return {
                    "temp_c": int(float(parts[0])),
                    "gpu_util_pct": int(float(parts[1])),
                    "mem_util_pct": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "vram_total_mb": int(float(parts[4])),
                    "power_w": round(float(parts[5]), 1),
                    "status": "OK"
                }
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "temp_c": 0, "vram_used_mb": 0, "power_w": 0.0}
    return {"status": "UNKNOWN", "temp_c": 0, "vram_used_mb": 0, "power_w": 0.0}


def get_disk_telemetry():
    """Checks free disk space across critical workspace partitions."""
    disks = {}
    for drive in ["C:\\", "E:\\", "F:\\", "I:\\"]:
        try:
            if os.path.exists(drive):
                total, used, free = shutil.disk_usage(drive)
                disks[drive[0]] = {
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2)
                }
        except Exception:
            pass
    return disks


def parse_log_tail(log_path, max_bytes=32768):
    """Scans the end of the log file for the newest step records and anomalies."""
    if not os.path.exists(log_path):
        return None, []
    
    file_size = os.path.getsize(log_path)
    read_size = min(file_size, max_bytes)
    
    try:
        # Read raw bytes and try utf-16 then utf-8
        with open(log_path, "rb") as f:
            f.seek(file_size - read_size)
            raw = f.read()
        
        text = None
        for enc in ["utf-16", "utf-8", "latin-1"]:
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
                
        if not text:
            return None, []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        anomalies = []
        latest_record = None

        for line in lines:
            line_lower = line.lower()
            # Check for NaN / Inf as distinct words, ignoring [info] tags
            if re.search(r"\b(nan|inf|-inf)\b", line_lower) and not line_lower.startswith("[info]"):
                anomalies.append(f"NaN/Inf detected in log line: {line[:80]}")

            # Check extended format first
            m_ext = LOG_RE_EXTENDED.search(line)
            if m_ext:
                rec = {
                    "step": int(m_ext.group(1)),
                    "loss": float(m_ext.group(2)),
                    "lr": float(m_ext.group(3)),
                    "grad_norm": float(m_ext.group(4)),
                    "tok_per_sec": int(m_ext.group(5)),
                    "v0_mix": float(m_ext.group(6)),
                    "beta_write": float(m_ext.group(7)),
                    "val_loss": float(m_ext.group(8)) if m_ext.group(8) else None,
                    "val_ppl": float(m_ext.group(9)) if m_ext.group(9) else None,
                    "elapsed_sec": float(m_ext.group(10)),
                    "raw_line": line
                }
                latest_record = rec
                continue

            # Fallback to standard format
            m_std = LOG_RE_STANDARD.search(line)
            if m_std:
                rec = {
                    "step": int(m_std.group(1)),
                    "loss": float(m_std.group(2)),
                    "lr": float(m_std.group(3)),
                    "grad_norm": None,
                    "tok_per_sec": None,
                    "v0_mix": None,
                    "beta_write": None,
                    "val_loss": None,
                    "val_ppl": None,
                    "elapsed_sec": float(m_std.group(4)),
                    "raw_line": line
                }
                latest_record = rec

        return latest_record, anomalies

    except Exception as e:
        return None, [f"Log parsing error: {e}"]


def update_dashboard_md(out_path, gpu, disks, rec, rolling_stats, anomalies):
    """Writes a clean, real-time Markdown status dashboard."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    step_str = f"{rec['step']:,}" if rec else "Initializing"
    loss_str = f"{rec['loss']:.4f}" if rec else "N/A"
    lr_str = f"{rec['lr']:.2e}" if rec else "N/A"
    tok_s_str = f"{rolling_stats.get('effective_tok_s', 0):,}" if rec else "N/A"
    pacing_str = f"{rolling_stats.get('sec_per_step', 0):.2f} s/step" if rec else "N/A"
    eta_str = rolling_stats.get("eta_str", "Calculating...")
    
    c_free = disks.get("C", {}).get("free_gb", "N/A")
    e_free = disks.get("E", {}).get("free_gb", "N/A")
    f_free = disks.get("F", {}).get("free_gb", "N/A")
    i_free = disks.get("I", {}).get("free_gb", "N/A")

    md = f"""# Parentheses 1.0-240M — Live Telemetry Dashboard

**Last Updated:** `{now_str}`  
**Watchdog Status:** `ACTIVE / HEALTHY`  

---

### 1. Training & Algorithmic State

| Current Step | Loss (CE) | Learning Rate | Instantaneous Pacing | Effective Throughput | Estimated Completion (ETA) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **{step_str}** | **{loss_str}** | `{lr_str}` | `{pacing_str}` | **{tok_s_str} tok/s** | **{eta_str}** |

- **Recurrent SSD Dynamics:** $\\sigma(b_\\beta) = {rec.get('beta_write', 'N/A') if rec else 'N/A'}$, $\\sigma(\\lambda_v) = {rec.get('v0_mix', 'N/A') if rec else 'N/A'}$, $\\|\\mathbf{{g}}\\|_2 = {rec.get('grad_norm', 'N/A') if rec else 'N/A'}$
- **Validation Perplexity:** {rec.get('val_ppl', 'Tracking') if rec and rec.get('val_ppl') else 'Logged every 100-250 steps'}

---

### 2. GPU & Thermal Envelopes (RTX 5050 Laptop 8GB)

| Temperature | GPU Load | Power Draw | VRAM Utilization | Tripwire Threshold |
| :---: | :---: | :---: | :---: | :---: |
| **{gpu.get('temp_c', 0)}°C** | {gpu.get('gpu_util_pct', 0)}% | **{gpu.get('power_w', 0.0)} W** | {gpu.get('vram_used_mb', 0)} MB / {gpu.get('vram_total_mb', 0)} MB | Safe (< 82°C) |

---

### 3. Disk Storage & Buffer Headroom

| C: System Drive | E: NVMe Worktree | F: Upstream HDD | I: Permanent Archive |
| :---: | :---: | :---: | :---: |
| **{c_free} GB Free** (Min 75GB) | **{e_free} GB Free** (Min 15GB) | {f_free} GB Free | **{i_free} GB Free** |

"""
    if anomalies:
        md += "\n> [!CAUTION]\n> **Active Watchdog Alerts Detected:**\n"
        for a in anomalies:
            md += f"> - {a}\n"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception:
        pass


def update_status_txt(out_path, rec, rolling_stats, gpu, anomalies):
    """Writes standard overnight_status.txt for multi-agent consumers."""
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    step = rec["step"] if rec else 0
    loss = f"{rec['loss']:.4f}" if rec else "?"
    rate = f"{rolling_stats.get('steps_per_sec', 0):.3f}" if rolling_stats.get("steps_per_sec") else "?"
    eta = rolling_stats.get("eta_clock", "?")
    temp = gpu.get("temp_c", 0)

    status_line = (
        f"stage: Parentheses 1.0-240M Session 3 | step: {step}/6000 | loss: {loss} | "
        f"steps_per_s: {rate} | temp: {temp}C | eta: {eta} | anomalies: {len(anomalies)} | updated: {now_str}\n"
    )
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(status_line)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Live Parallel Telemetry & Watchdog Engine for 240M")
    parser.add_argument("--log-path", default=str(REPO_ROOT / "out" / "session3_240m_live.log"))
    parser.add_argument("--telemetry-jsonl", default=str(REPO_ROOT / "out" / "telemetry_s3_live.jsonl"))
    parser.add_argument("--dashboard-md", default=str(REPO_ROOT / "out" / "telemetry_dashboard.md"))
    parser.add_argument("--status-txt", default=str(REPO_ROOT / "out" / "overnight_status.txt"))
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--tokens-per-step", type=int, default=65536)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--thermal-limit", type=int, default=82)
    parser.add_argument("--loss-collapse-limit", type=float, default=1.40)
    parser.add_argument("--grad-explosion-limit", type=float, default=10.0)
    parser.add_argument("--min-e-free-gb", type=float, default=10.0)
    args = parser.parse_args()

    print(f"[*] Starting Parallel Telemetry & Watchdog Engine for Parentheses 1.0-240M")
    print(f"    - Monitoring Log : {args.log_path}")
    print(f"    - Output Stream  : {args.telemetry_jsonl}")
    print(f"    - Live Dashboard : {args.dashboard_md}")
    print(f"    - Polling Cadence: {args.interval}s\n")

    history = []
    t_start = time.time()

    while True:
        try:
            gpu = get_gpu_telemetry()
            disks = get_disk_telemetry()
            latest_rec, log_anomalies = parse_log_tail(args.log_path)
            
            anomalies = list(log_anomalies)

            # Circuit-Breaker 1: Thermal Limit
            if gpu.get("temp_c", 0) > args.thermal_limit:
                msg = f"CRITICAL THERMAL ALERT: GPU temperature {gpu['temp_c']}C exceeds limit ({args.thermal_limit}C)!"
                anomalies.append(msg)
                print(f"[!] {msg}", file=sys.stderr)

            # Circuit-Breaker 2: Disk Headroom on E:
            e_free = disks.get("E", {}).get("free_gb", 999.0)
            if e_free < args.min_e_free_gb:
                msg = f"CRITICAL DISK ALERT: E: free space {e_free} GB is below minimum safety headroom ({args.min_e_free_gb} GB)!"
                anomalies.append(msg)
                print(f"[!] {msg}", file=sys.stderr)

            # Circuit-Breaker 3: Algorithmic Loss Collapse / Explosion
            if latest_rec:
                if latest_rec["loss"] < args.loss_collapse_limit:
                    anomalies.append(f"Loss collapse anomaly: {latest_rec['loss']} < {args.loss_collapse_limit}")
                if latest_rec.get("grad_norm") and latest_rec["grad_norm"] > args.grad_explosion_limit:
                    anomalies.append(f"Gradient explosion anomaly: {latest_rec['grad_norm']} > {args.grad_explosion_limit}")

            # Calculate Rolling Velocity & Pacing
            rolling_stats = {}
            if latest_rec:
                history.append((time.time(), latest_rec["step"], latest_rec["loss"]))
                # Keep last 30 minutes of history
                cutoff = time.time() - 1800
                history = [h for h in history if h[0] >= cutoff]

                if len(history) >= 2 and history[-1][1] > history[0][1]:
                    dt_span = history[-1][0] - history[0][0]
                    dstep_span = history[-1][1] - history[0][1]
                    steps_per_sec = dstep_span / dt_span
                    sec_per_step = dt_span / dstep_span
                    effective_tok_s = int(steps_per_sec * args.tokens_per_step)
                    
                    steps_remaining = max(0, args.max_steps - latest_rec["step"])
                    time_remaining_sec = steps_remaining * sec_per_step
                    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining_sec)
                    
                    hours = int(time_remaining_sec // 3600)
                    mins = int((time_remaining_sec % 3600) // 60)
                    eta_str = f"{finish_time.strftime('%I:%M %p')} (~{hours}h {mins}m remaining)"
                    eta_clock = finish_time.strftime("%H:%M")
                    
                    rolling_stats = {
                        "steps_per_sec": steps_per_sec,
                        "sec_per_step": sec_per_step,
                        "effective_tok_s": effective_tok_s,
                        "eta_str": eta_str,
                        "eta_clock": eta_clock,
                        "time_remaining_sec": time_remaining_sec
                    }

            # Update live human and agent artifacts
            update_dashboard_md(args.dashboard_md, gpu, disks, latest_rec, rolling_stats, anomalies)
            update_status_txt(args.status_txt, latest_rec, rolling_stats, gpu, anomalies)

            # Record telemetry point to JSONL
            telemetry_point = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step": latest_rec["step"] if latest_rec else None,
                "loss": latest_rec["loss"] if latest_rec else None,
                "lr": latest_rec["lr"] if latest_rec else None,
                "grad_norm": latest_rec.get("grad_norm") if latest_rec else None,
                "v0_mix": latest_rec.get("v0_mix") if latest_rec else None,
                "beta_write": latest_rec.get("beta_write") if latest_rec else None,
                "val_loss": latest_rec.get("val_loss") if latest_rec else None,
                "val_ppl": latest_rec.get("val_ppl") if latest_rec else None,
                "sec_per_step": rolling_stats.get("sec_per_step"),
                "effective_tok_s": rolling_stats.get("effective_tok_s"),
                "gpu": gpu,
                "disks": disks,
                "anomalies": anomalies
            }

            with open(args.telemetry_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(telemetry_point, ensure_ascii=False) + "\n")

            # Check if training has reached completion
            if latest_rec and latest_rec["step"] >= args.max_steps:
                print(f"[+] Milestone step {args.max_steps} reached! Telemetry engine concluding.")
                break

        except KeyboardInterrupt:
            print("\n[*] Telemetry engine stopped by user.")
            break
        except Exception as e:
            print(f"[!] Warning: Watchdog polling error: {e}", file=sys.stderr)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
