import os, sys, json, re
from pathlib import Path

ROOT = Path("E:/Parentheses/parentheses-0.9")

def parse_log(path):
    steps = []
    losses = []
    lrs = []
    elapseds = []
    pattern = re.compile(r"step\s+(\d+)\s+\|\s+loss\s+([\d\.]+)\s+\|\s+lr\s+([\d\.e\-\+]+)\s+\|\s+([\d\.]+)s elapsed")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                elapseds.append(float(m.group(4)))
    return {
        "steps": steps,
        "losses": losses,
        "lrs": lrs,
        "elapseds": elapseds,
    }

def parse_val_exports():
    eval_dir = ROOT / "out/npu_eval"
    results = {}
    if not eval_dir.exists():
        return results
    for d in eval_dir.iterdir():
        if d.is_dir():
            exp_file = d / "export.json"
            if exp_file.exists():
                try:
                    with open(exp_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results[d.name] = {
                        "checkpoint": data.get("checkpoint"),
                        "step": data.get("step"),
                        "torch_fp32_loss": data.get("torch_fp32_loss", {})
                    }
                except Exception as e:
                    pass
    return results

def main():
    run1 = parse_log(ROOT / "out/parentheses-0.9-120M-run1-lr3e-4.log")
    run2 = parse_log(ROOT / "out/parentheses-0.9-120M.log")
    vals = parse_val_exports()

    output = {
        "run1": run1,
        "run2": run2,
        "val_evals": vals,
    }
    
    out_path = ROOT / "out/swarm/120M_session1_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Stats written to {out_path}")
    print(f"Run 1 data points: {len(run1['steps'])}")
    print(f"Run 2 data points: {len(run2['steps'])}")
    print(f"Val eval checkpoints: {list(vals.keys())}")

if __name__ == "__main__":
    main()
