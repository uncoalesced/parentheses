"""Summarise GPU telemetry logged during training: energy, temperature, throttling.

The collector is nvidia-smi itself, started next to the trainer and stopped after it:

    nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks_event_reasons.active --format=csv,nounits -lms 1000 > run/telemetry.csv

Redirect stdout; do not use nvidia-smi's own -f option. With -f nothing reaches the
file until nvidia-smi exits cleanly, so a killed collector leaves an empty log
(checked on this machine). Stdout is written line by line.

Then:

    python -m parentheses.telemetry run/*.csv --tokens 1250000000 --grid-kg-per-kwh 0.7 --grid-source "CEA vXX"

Energy is board power from the GPU's own sensor, integrated over time. It leaves out
the CPU, memory, screen and fans; pass --wall-kwh from a plug meter to report the whole
machine. Carbon is printed only when a grid factor and its source are given.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
# Bits of clocks_event_reasons.active (NVML clocksEventReasons).
POWER_CAP = 0x4
THERMAL = 0x20 | 0x40           # software and hardware thermal slowdown


def read_rows(path: Path) -> List[dict]:
    """Rows with time (s since epoch), watts, celsius, reasons. Unreadable values become None."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, skipinitialspace=True)
        first = next(reader, None)
        if first is None:
            return rows                                    # empty: nvidia-smi wrote nothing
        header = [h.split(" [")[0].strip() for h in first]
        for raw in reader:
            if len(raw) != len(header):
                continue                                   # a line cut off when nvidia-smi was stopped
            rec = dict(zip(header, raw))
            try:
                t = datetime.strptime(rec["timestamp"], TIME_FORMAT).timestamp()
            except ValueError:
                continue
            rows.append({"t": t, "watts": _num(rec.get("power.draw")),
                         "celsius": _num(rec.get("temperature.gpu")),
                         "reasons": _hex(rec.get("clocks_event_reasons.active",
                                                 rec.get("clocks_throttle_reasons.active")))})
    return rows


def _num(s: Optional[str]) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None                                        # "[N/A]" and the like


def _hex(s: Optional[str]) -> Optional[int]:
    try:
        return int(s, 16)
    except (TypeError, ValueError):
        return None


def summarise(rows: Iterable[dict], max_gap: float = 5.0) -> Dict[str, float]:
    """Integrate power with the trapezoid rule. A step longer than `max_gap` seconds is
    not bridged: its time is counted as a gap and adds no energy."""
    rows = sorted(rows, key=lambda r: r["t"])
    joules = covered = gap = thermal = capped = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        if dt > max_gap or a["watts"] is None or b["watts"] is None:
            gap += dt
            continue
        joules += 0.5 * (a["watts"] + b["watts"]) * dt
        covered += dt
        reasons = a["reasons"] or 0
        thermal += dt if reasons & THERMAL else 0.0
        capped += dt if reasons & POWER_CAP else 0.0
    temps = [r["celsius"] for r in rows if r["celsius"] is not None]
    return {
        "samples": len(rows),
        "hours_logged": covered / 3600,
        "hours_gap": gap / 3600,
        "gpu_kwh": joules / 3.6e6,
        "mean_watts": joules / covered if covered else 0.0,
        "mean_celsius": sum(temps) / len(temps) if temps else 0.0,
        "peak_celsius": max(temps) if temps else 0.0,
        "share_thermal_throttled": thermal / covered if covered else 0.0,
        "share_power_capped": capped / covered if covered else 0.0,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m parentheses.telemetry", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="nvidia-smi CSV logs, one per session")
    ap.add_argument("--tokens", type=int, default=None, help="training tokens these logs cover")
    ap.add_argument("--grid-kg-per-kwh", type=float, default=None, help="grid emission factor")
    ap.add_argument("--grid-source", default=None, help="where the factor comes from, e.g. 'CEA vXX'")
    ap.add_argument("--wall-kwh", type=float, default=None, help="whole-machine energy from a plug meter")
    ap.add_argument("--max-gap", type=float, default=5.0, help="seconds; longer gaps are not bridged")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)
    if (args.grid_kg_per_kwh is None) != (args.grid_source is None):
        ap.error("--grid-kg-per-kwh and --grid-source go together")

    rows = [r for p in args.csv for r in read_rows(Path(p))]
    out = summarise(rows, args.max_gap)
    out["files"] = len(args.csv)
    if args.tokens:
        out["tokens"] = args.tokens
        out["gpu_wh_per_million_tokens"] = out["gpu_kwh"] * 1000 / (args.tokens / 1e6)
    if args.wall_kwh is not None:
        out["wall_kwh"] = args.wall_kwh
    if args.grid_kg_per_kwh is not None:
        basis = out.get("wall_kwh", out["gpu_kwh"])
        out["kg_co2"] = basis * args.grid_kg_per_kwh
        out["co2_basis"] = "wall" if "wall_kwh" in out else "gpu board only"
        out["grid_source"] = args.grid_source

    if args.json:
        print(json.dumps(out, indent=1))
    else:
        for k, v in out.items():
            print(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
