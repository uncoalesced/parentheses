"""Run lm-evaluation-harness tasks against a checkpoint and save the raw results.

    python -m parentheses.evaluate --checkpoint model.safetensors \
        --tasks gsm8k,arc_easy --output eval/results

Results go to <output>/<checkpoint sha256>/<label>.json with per-sample logs
next to them. A run with --limit is a smoke test: it is saved under a name that
says so and must never be reported as a score. See docs/BENCHMARK_SPEC.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import lm_eval

from .checkpoint import sha256_file
from .lm_eval_adapter import ParenthesesLM


def harness_revision() -> str:
    """Git revision of the harness, if it is running from its own checkout.

    A pip-installed copy has no revision of its own. Asking git anyway would
    return the revision of whatever repository happens to contain the
    environment, so that case is reported as unknown.
    """
    root = Path(lm_eval.__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return "unknown: harness is an installed package, pass --harness-revision"
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated harness task names")
    ap.add_argument("--output", default="eval/results")
    ap.add_argument("--label", default=None, help="file name for this run (default: the task list)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only: first N items per task")
    ap.add_argument("--harness-revision", default=None,
                    help="git revision of lm-evaluation-harness, when it is installed as a package")
    ap.add_argument("--allow-code-execution", action="store_true",
                    help="needed for MBPP and HumanEval; only inside a sandbox with no network")
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    checkpoint_sha = sha256_file(args.checkpoint)
    lm = ParenthesesLM(checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size)

    # Greedy, zero-shot unless the task itself defines few-shot examples, one run.
    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, batch_size=args.batch_size, limit=args.limit, log_samples=True,
        confirm_run_unsafe_code=args.allow_code_execution,
    )

    label = (args.label or "+".join(tasks))[:80]
    if args.limit:
        label = f"SMOKE-limit{args.limit}-{label}"
    out_dir = Path(args.output) / checkpoint_sha
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = results.pop("samples", {})
    record = {
        # The checkpoint path is local; hide it in both "--checkpoint p" and "--checkpoint=p" forms.
        "command": "python -m parentheses.evaluate " + " ".join(
            a.replace(args.checkpoint, "<checkpoint>") for a in (argv or sys.argv[1:])),
        "date": datetime.date.today().isoformat(),
        "harness_version": lm_eval.__version__,
        "harness_revision": args.harness_revision or harness_revision(),
        "checkpoint_sha256": checkpoint_sha,
        "limit": args.limit,
        "smoke_test": bool(args.limit),
        "results": results.get("results"),
        "n_samples": results.get("n-samples"),
        "versions": results.get("versions"),
        "configs": results.get("configs"),
    }
    (out_dir / f"{label}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    for task, rows in samples.items():
        with open(out_dir / f"{label}.samples.{task}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    print(f"\nsaved {out_dir / (label + '.json')}")
    for task, metrics in (record["results"] or {}).items():
        shown = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(f"  {task}: {shown}  n={(record['n_samples'] or {}).get(task)}")
    if args.limit:
        print("SMOKE TEST: these numbers are not results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
