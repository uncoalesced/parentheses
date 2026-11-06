# Engineered by uncoalesced
"""
Extracts tabular step & telemetry statistics for Parentheses 1.0-240M Session 6
Parses out/session6_240m_live.log and out/telemetry_s3_live.jsonl into a clean
CSV (out/session6_stats.csv) and JSON (out/session6_stats.json) for graphing loss,
learning rate, gradient norm, throughput, GPU thermals, and VRAM.
"""

import json
import re
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "out" / "session6_240m_live.log"
TELEMETRY_PATH = REPO_ROOT / "out" / "telemetry_s3_live.jsonl"
OUT_CSV = REPO_ROOT / "out" / "session6_stats.csv"
OUT_JSON = REPO_ROOT / "out" / "session6_stats.json"

LOG_PATTERN = re.compile(
    r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)\s+\|\s+"
    r"grad_norm\s+([\d.]+)\s+\|\s+tok/s\s+(\d+)\s+\|\s+v0_mix\s+([\d.]+)\s+\|\s+"
    r"beta_write\s+([\d.]+)"
    r"(?:\s+\|\s+val_loss\s+([\d.]+)\s+\|\s+val_ppl\s+([\d.]+)\s+\|\s+val_v5\s+([\d.]+)\s+\|\s+ppl_v5\s+([\d.]+))?"
    r"\s+\|\s+([\d.]+)s\s+elapsed"
)



def extract():
    records = []
    if LOG_PATH.exists():
        raw_text = ""
        for enc in ["utf-16", "utf-8", "latin-1"]:
            try:
                with open(LOG_PATH, "r", encoding=enc) as f:
                    raw_text = f.read()
                if "step" in raw_text:
                    break
            except Exception:
                continue

        for line in raw_text.splitlines():
            m = LOG_PATTERN.search(line)
            if m:
                rec = {
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "lr": float(m.group(3)),
                    "grad_norm": float(m.group(4)),
                    "tok_per_sec": int(m.group(5)),
                    "v0_mix": float(m.group(6)),
                    "beta_write": float(m.group(7)),
                    "val_loss": float(m.group(8)) if m.group(8) else None,
                    "val_ppl": float(m.group(9)) if m.group(9) else None,
                    "val_v5": float(m.group(10)) if m.group(10) else None,
                    "ppl_v5": float(m.group(11)) if m.group(11) else None,
                    "elapsed_sec": float(m.group(12)),
                }
                records.append(rec)


    # Load GPU telemetry lookup by step
    telemetry_by_step = {}
    if TELEMETRY_PATH.exists():
        with open(TELEMETRY_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    s = obj.get("step")
                    if s and s >= 10001:
                        telemetry_by_step[s] = obj
                except Exception:
                    pass

    # Enrich records
    for r in records:
        t = telemetry_by_step.get(r["step"])
        if t:
            gpu = t.get("gpu", {})
            r["temp_c"] = gpu.get("temp_c")
            r["vram_used_mb"] = gpu.get("vram_used_mb")
            r["power_w"] = gpu.get("power_w")
            r["gpu_util_pct"] = gpu.get("gpu_util_pct")
        else:
            r["temp_c"] = None
            r["vram_used_mb"] = None
            r["power_w"] = None
            r["gpu_util_pct"] = None

    if records:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    print(f"[+] Extracted {len(records)} steps -> {OUT_CSV.name} and {OUT_JSON.name}")
    return records


if __name__ == "__main__":
    extract()
