"""
Training stats report for the Heafod vault: charts + one Markdown note.

Reads training logs (out/*.log) and eval_dravidian.py JSON (out/sweep/eval.json,
out/eval_dravidian*.json), merges every eval into a history.jsonl in the output
folder (keyed by checkpoint, newest wins) so numbers survive even after out/ or
checkpoints are cleaned, then regenerates the PNGs and `Training Stats.md`.

Needs only json + matplotlib (no torch), so the system Python can run it:

    python scripts/report_stats.py
    python scripts/report_stats.py --self-test

Add a new run by appending to RUNS below; re-run after every eval.
"""

import argparse
import glob
import json
import os
import re
from datetime import date

HEAFOD = r"C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\Training Stats"
LANGS = ("kan", "tam", "tel", "mal")
LOWRES = ("tcy", "kff", "kxv", "nit", "esg")  # Tulu, Koya, Kuvi, SE Kolami, Aheri Gondi (research-only)

# log file -> (label, date started, recipe). Order = plot order.
RUNS = {
    "out/dravidian-v1-train.log": ("v1", "2026-10-01", "resume selective-v1, const LR 3e-4, no replay (killed by laptop sleep at 583.1k)"),
    "out/dravidian-v1-train-resume.log": ("v1 resume", "2026-10-06", "resume v1 step 580k, const LR 3e-4, no replay"),
    "out/dravidian-v2-train.log": ("v2 replay", "2026-10-06", "resume v1 final, LR 3e-4, 10% EN replay, cooldown last 20k"),
    "out/dravidian-v3-train.log": ("v3 lowres", "2026-10-07", "resume v2 final; per batch of 64: 6 EN, 3 tcy, 1 each kff/kxv/nit/esg; cooldown last 12k"),
}

# BPE runs log nats per *token*, not comparable with the byte runs' nats/byte --
# their train curves get their own chart; cross-model comparison is bits/byte.
BPE_RUNS = {    # log -> (label, started, recipe)
    "out/parentheses-0.9-41M.log": ("0.9-41M", "2026-10-07",
                                    "from scratch, BPE v3_32k, T=4096, 40.9M params, corpus v3 (EN 40%, "
                                    "kan/tam/tel/mal 13.8% each, low-res 4.7%), 131k tok/step x 8,083 steps "
                                    "= 1.06B tokens, LR 1e-3, warmup 108, cooldown from step 6,466 (extended "
                                    "twice mid-run by resume, documentation.md 2026-10-07 §19-20)"),
}

STEP_RE = re.compile(r"^step\s+(\d+) \| loss ([\d.]+)")
BPB_LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng")


def merge_bpb(path: str, evals: list[dict]) -> list[dict]:
    """eval_dravidian.py --bpb results -> history_bpb.jsonl, keyed by checkpoint."""
    hist = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hist[r["checkpoint"]] = r
    for r in evals:
        hist[r["checkpoint"]] = {k: v for k, v in r.items() if k != "tags"} | {
            "tag_acc": {l: t["acc"] for l, t in r["tags"].items()},
            "recorded": hist.get(r["checkpoint"], {}).get("recorded", date.today().isoformat())}
    rows = sorted(hist.values(), key=lambda r: (r["recorded"], r["checkpoint"]))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def bpb_chart(out_dir, rows) -> list[str]:
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = .8 / len(rows)
    for i, r in enumerate(rows):
        ax.bar([j + i * w for j in range(len(BPB_LANGS))], [r["bpb"].get(l, 0) for l in BPB_LANGS], w,
               label=f"{short(r['checkpoint'])} ({r['params'] / 1e6:.1f}M, T={r['block']})")
    ax.set_xticks([j + .4 - w / 2 for j in range(len(BPB_LANGS))], BPB_LANGS)
    ax.set(title="Bits per UTF-8 byte, same val text for every model (lower better)", ylabel="bits/byte")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "bits_per_byte.png"), dpi=130); plt.close(fig)
    return ["bits_per_byte.png"]


def bpb_note(rows) -> list[str]:
    if not rows:
        return []
    lines = ["", "## Bits per byte (byte models vs BPE)", "",
             "`eval_dravidian.py --bpb`: every model on the same decoded val text (old val splits, ~1 MB per "
             "language), scored on text + one end marker, tag never scored. bits/byte = nats ÷ bytes ÷ ln 2, "
             "so a byte model's nats/byte × 1.4427 is directly comparable. Tag cell = script accuracy / "
             "tag-swap gap (nats/token).", "",
             "| recorded | checkpoint | params | T | " + " | ".join(BPB_LANGS) + " |",
             "|---|---|---|---|" + "---|" * len(BPB_LANGS)]
    for r in rows:
        lines.append(f"| {r['recorded']} | {short(r['checkpoint'])} | {r['params'] / 1e6:.1f}M | {r['block']} | "
                     + " | ".join(f"{r['bpb'][l]:.3f}" for l in BPB_LANGS) + " |")
    lines += ["", "| checkpoint | " + " | ".join(l for l in BPB_LANGS) + " |", "|---|" + "---|" * len(BPB_LANGS)]
    for r in rows:
        cell = lambda l: (f"{r['tag_acc'][l]:.2f} / {r.get('tagswap', {}).get(l, float('nan')):+.2f}"
                          if r["tag_acc"].get(l) is not None else "-")
        lines.append(f"| {short(r['checkpoint'])} | " + " | ".join(cell(l) for l in BPB_LANGS) + " |")
    return lines


def parse_log(lines) -> list[tuple[int, float]]:
    return [(int(m[1]), float(m[2])) for m in map(STEP_RE.match, lines) if m]


def smooth(ys, k=10):
    out, acc = [], 0.0
    for i, y in enumerate(ys):
        acc += y - (ys[i - k] if i >= k else 0.0)
        out.append(acc / min(i + 1, k))
    return out


def short(ckpt: str) -> str:
    """checkpoints/sweep/s2_x/step_N_final.pt -> s2_x@N"""
    parts = ckpt.replace("\\", "/").split("/")
    step = re.search(r"step_(\d+)", parts[-1])
    return f"{parts[-2]}@{step[1] if step else '?'}"


def merge_history(path: str, evals: list[dict]) -> list[dict]:
    hist = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hist[r["checkpoint"]] = r
    for r in evals:
        r = {k: v for k, v in r.items() if k != "tags"} | {
            "tag_acc": {l: t["acc"] for l, t in r["tags"].items()},
            "recorded": hist.get(r["checkpoint"], {}).get("recorded", date.today().isoformat())}
        hist[r["checkpoint"]] = r
    rows = sorted(hist.values(), key=lambda r: (r["recorded"], r.get("step") or 0, r["checkpoint"]))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def charts(out_dir, curves, hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    # 1. train loss curves
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, pts in curves.items():
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, smooth(list(ys)), label=label, lw=1.4)
    ax.set(xlabel="step", ylabel="train loss (nats/byte, smoothed)", title="Training loss")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=130); plt.close(fig)
    files.append("loss_curves.png")

    names = [short(r["checkpoint"]) for r in hist]
    # 2. Dravidian vs English tradeoff
    pts = [(r["drav_val"], r.get("all_sources_val"), n) for r, n in zip(hist, names) if r.get("all_sources_val")]
    if pts:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for d, e, n in pts:
            ax.scatter(d, e, s=30)
            ax.annotate(n, (d, e), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set(xlabel="Dravidian val loss (lower better)", ylabel="English / all-sources val loss (lower better)",
               title="Dravidian vs English tradeoff (bottom-left wins)")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tradeoff.png"), dpi=130); plt.close(fig)
        files.append("tradeoff.png")

    # 3-6. per-language val loss and tag accuracy, grouped bars, anchors and low-resource
    # charted separately (unseen-script losses ~4.5 would flatten the anchors' ~0.6)
    for group, gname in ((LANGS, ""), (LOWRES, "_lowres")):
        rows = [(r, n) for r, n in zip(hist, names) if any(r.get(f"{l}_val") is not None for l in group)]
        if not rows:
            continue
        for key, fname, title, ylab in (
                ("{}_val", f"per_language_loss{gname}.png", "Per-language val loss", "val loss (lower better)"),
                ("tag_acc", f"tag_accuracy{gname}.png", "Language-tag accuracy (right script)", "fraction correct")):
            fig, ax = plt.subplots(figsize=(max(8, len(rows) * .9), 4.5))
            w = .8 / len(group)
            for i, l in enumerate(group):
                ys = [(r["tag_acc"].get(l) if key == "tag_acc" else r.get(key.format(l))) or 0 for r, _ in rows]
                ax.bar([j + i * w for j in range(len(rows))], ys, w, label=l)
            ax.set_xticks([j + .4 - w / 2 for j in range(len(rows))], [n for _, n in rows],
                          rotation=35, ha="right", fontsize=7)
            if key != "tag_acc" and not gname:
                lo = min((r.get(key.format(l)) or 9) for r, _ in rows for l in group)
                ax.set_ylim(lo * .9, None)
            ax.set(title=title + (" — low-resource" if gname else ""), ylabel=ylab)
            ax.legend(ncol=len(group)); ax.grid(axis="y", alpha=.3)
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=130); plt.close(fig)
            files.append(fname)
    return files


def note(curves, hist, files) -> str:
    f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"
    lines = [f"# Parentheses training stats", "",
             f"Generated {date.today().isoformat()} by `parentheses-0.9/scripts/report_stats.py`. "
             "Numbers live in `history.jsonl` next to this note; re-run the script after every eval.", "",
             "Losses are nats per byte. Val sets: `processed_dravidian_v1/val.bin` (overall + per language, "
             "with document resets) and `processed_all_sources_v1/val.bin` (English/old domain). Tag accuracy = "
             "share of 32 samples from a `<lang> ` prompt whose output is mostly that language's script.", "",
             "## Runs", "", "| run | started | steps logged | last train loss | recipe |", "|---|---|---|---|---|"]
    for log, (label, started, recipe) in RUNS.items():
        pts = curves.get(label) or []
        rng = f"{pts[0][0]:,}–{pts[-1][0]:,}" if pts else "-"
        lines.append(f"| {label} | {started} | {rng} | {f(pts[-1][1]) if pts else '-'} | {recipe} |")
    lines += ["", "## Eval history", "",
              "| recorded | checkpoint | drav | " + " | ".join(LANGS) + " | English | tags " + "/".join(LANGS) + " |",
              "|---|---|---|" + "---|" * len(LANGS) + "---|---|"]
    for r in hist:
        tags = "/".join(f"{r['tag_acc'].get(l, 0):.2f}" for l in LANGS)
        lines.append(f"| {r['recorded']} | {short(r['checkpoint'])} | {f(r['drav_val'])} | "
                     + " | ".join(f(r.get(f"{l}_val")) for l in LANGS)
                     + f" | {f(r.get('all_sources_val'))} | {tags} |")
    low = [r for r in hist if any(r.get(f"{l}_val") is not None for l in LOWRES)]
    if low:
        lines += ["", "## Low-resource languages", "",
                  "Cell = val loss / tag accuracy (script) / tag-swap gap (loss under a confusable tag minus "
                  "own tag; > 0 means the model uses the tag). Koya/Kuvi are Odia script, Gondi Devanagari, "
                  "so their script test is meaningful; Tulu (Kannada script) and Kolami (Telugu script) rely "
                  "on the swap gap. Gondi is research-only (Wycliffe) — never release a model trained on it.", "",
                  "| checkpoint | " + " | ".join(LOWRES) + " |", "|---|" + "---|" * len(LOWRES)]
        for r in low:
            cell = lambda l: (f"{r[f'{l}_val']:.3f} / {r['tag_acc'].get(l, 0):.2f} / "
                              f"{r.get('tagswap', {}).get(l, float('nan')):+.3f}") if r.get(f"{l}_val") else "-"
            lines.append(f"| {short(r['checkpoint'])} | " + " | ".join(cell(l) for l in LOWRES) + " |")
    lines += ["", "## Charts", ""] + [f"![[{x}]]\n" for x in files]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=HEAFOD)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return _self_test()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    curves = {}
    for log, (label, _, _) in RUNS.items():
        path = os.path.join(root, log)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                curves[label] = parse_log(fh)
    evals = []
    for path in [os.path.join(root, "out/sweep/eval.json"), *glob.glob(os.path.join(root, "out/eval_dravidian*.json"))]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                evals += json.load(fh)

    bpb_evals = []
    for path in glob.glob(os.path.join(root, "out/bpb_eval*.json")):
        with open(path, encoding="utf-8") as fh:
            bpb_evals += json.load(fh)
    bpe_curves = {}
    for log, (label, _, _) in BPE_RUNS.items():
        path = os.path.join(root, log)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                bpe_curves[label] = parse_log(fh)

    os.makedirs(args.out_dir, exist_ok=True)
    hist = merge_history(os.path.join(args.out_dir, "history.jsonl"), evals)
    files = charts(args.out_dir, curves, hist)
    bpb_rows = merge_bpb(os.path.join(args.out_dir, "history_bpb.jsonl"), bpb_evals)
    files += bpb_chart(args.out_dir, bpb_rows)
    if any(bpe_curves.values()):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for label, pts in bpe_curves.items():
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, smooth(list(ys)), label=label, lw=1.4)
        ax.set(xlabel="step", ylabel="train loss (nats/token, smoothed)", title="BPE runs: training loss")
        ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "loss_curves_bpe.png"), dpi=130); plt.close(fig)
        files.append("loss_curves_bpe.png")
    text = note(curves, hist, files)
    extra = bpb_note(bpb_rows)
    if BPE_RUNS:
        extra += ["", "## BPE runs", "", "| run | started | steps logged | last train loss (nats/token) | recipe |",
                  "|---|---|---|---|---|"]
        for log, (label, started, recipe) in BPE_RUNS.items():
            pts = bpe_curves.get(label) or []
            rng = f"{pts[0][0]:,}–{pts[-1][0]:,}" if pts else "-"
            extra.append(f"| {label} | {started} | {rng} | {pts[-1][1]:.4f} | {recipe} |" if pts
                         else f"| {label} | {started} | - | - | {recipe} |")
    text = text.replace("\n## Charts\n", "\n".join(extra) + "\n\n## Charts\n", 1)
    with open(os.path.join(args.out_dir, "Training Stats.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[info] {len(hist)} evals, {len(bpb_rows)} bpb evals, {len(files)} charts -> {args.out_dir}")

    # Regenerate model stats panel (Model Stats.md + internal weight charts)
    try:
        from model_stats import analyze_checkpoint, generate_plots, generate_markdown, get_disk_free_stats, DEFAULT_CHECKPOINTS
        from pathlib import Path
        print("[info] Regenerating model stats panel...")
        m_results = [analyze_checkpoint(name, path) for name, path in DEFAULT_CHECKPOINTS if path.exists()]
        d_stats = get_disk_free_stats()
        generate_plots(m_results, Path(args.out_dir))
        generate_markdown(m_results, d_stats, Path(args.out_dir))
        print(f"[info] Model stats panel updated in {args.out_dir}")
    except Exception as e:
        print(f"[warn] Could not regenerate model stats panel: {e}")


def _self_test():
    import tempfile
    pts = parse_log(["[info] x", "step 561000 | loss 1.1557 | 0.7s elapsed",
                     "step 580010 | loss 1.0360 | lr 7.72e-04 | 2.4s elapsed"])
    assert pts == [(561000, 1.1557), (580010, 1.036)], pts
    assert smooth([1, 3, 5], k=2) == [1, 2, 4]
    assert short("checkpoints/sweep/s2_a/step_666999_final.pt") == "s2_a@666999"
    ev = {"checkpoint": "c/a/step_5.pt", "step": 5, "drav_val": .5,
          "tags": {l: {"acc": 1.0} for l in LANGS}}
    with tempfile.TemporaryDirectory() as d:
        h = os.path.join(d, "history.jsonl")
        merge_history(h, [ev])
        rows = merge_history(h, [ev | {"drav_val": .4}])   # same checkpoint: replaced, not duplicated
        assert len(rows) == 1 and rows[0]["drav_val"] == .4 and rows[0]["tag_acc"]["kan"] == 1.0

    # Self-test for model_stats hook
    from model_stats import run_self_test
    run_self_test()

    print("[self-test] report_stats ok")


if __name__ == "__main__":
    main()
