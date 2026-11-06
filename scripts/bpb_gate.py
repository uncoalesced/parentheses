"""
Warn-only BPB gate (Joel, 2026-11-03): compare the first and last checkpoint in a bpb_slab_v4.py JSON
and flag any language whose bits per byte rose more than --max-rise. Always exits 0.
    .venv\\Scripts\\python.exe scripts\\bpb_gate.py out\\eval_s5\\bpb_240m.json --max-rise 0.06
    .venv\\Scripts\\python.exe scripts\\bpb_gate.py --self-test
"""

import argparse
import json
import sys


def gate(results: list, max_rise: float) -> list[str]:
    base, new = results[0], results[-1]
    lines = [f"baseline step {base.get('step')} -> step {new.get('step')}, max rise {max_rise}"]
    for lang, a in base["bpb"].items():
        if lang not in new["bpb"]:
            continue
        b = new["bpb"][lang]
        flag = "WARN" if b - a > max_rise else "ok  "
        lines.append(f"{flag} {lang} {a:.3f} -> {b:.3f} ({b - a:+.3f})")
    return lines


def self_test() -> None:
    r = [{"step": 8000, "bpb": {"kan": 0.50, "nit": 0.757}}, {"step": 10000, "bpb": {"kan": 0.52, "nit": 0.83}}]
    out = gate(r, 0.06)
    assert out[1].startswith("ok") and out[2].startswith("WARN"), out
    print("bpb_gate self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?")
    ap.add_argument("--max-rise", type=float, default=0.06)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    with open(a.json, encoding="utf-8") as fh:
        print("\n".join(gate(json.load(fh), a.max_rise)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
