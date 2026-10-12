#!/usr/bin/env python3
"""Session 2 report (Claude Code, 2026-10-12): one markdown file from whatever outputs exist.

    .venv\\Scripts\\python.exe scripts\\session2_report.py            # -> out/session2_report.md
    .venv\\Scripts\\python.exe scripts\\session2_report.py --self-test

Sections: training loss, val watcher, bits/byte, 250 queries, decoding sweep, SFT trial,
lm-eval, session-3 numbers. A missing input prints "not run" for its section; nothing is invented.
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

STEP_RE = re.compile(r"step\s+(\d+) \| loss ([\d.]+)")
MAIN4 = ("kan", "tam", "tel", "mal")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.isfile(path) else []


def train_losses(log: str) -> dict[int, float]:
    """step -> loss; the last write wins (resumed runs re-log the same steps)."""
    out = {}
    if os.path.isfile(log):
        for m in STEP_RE.finditer(open(log, encoding="utf-8", errors="replace").read()):
            out[int(m.group(1))] = float(m.group(2))
    return out


def query_stats(rows: list[dict]) -> dict:
    ok = [r["script_ok"] for r in rows if r.get("script_ok") is not None]
    n = len(rows)
    return {"n": n, "script_ok": sum(ok) / len(ok) if ok else float("nan"),
            "eos": sum(r["stop"] == "eos" for r in rows) / n, "loop": sum(r["stop"] == "loop" for r in rows) / n,
            "rep": sum(r["repetition_rate"] for r in rows) / n, "d2": sum(r["distinct_2"] for r in rows) / n}


def sweep_pick(stats: dict[str, dict]) -> str:
    """Rule fixed before the run: max (script_ok - loop), ties broken by eos."""
    return max(stats, key=lambda k: (round(stats[k]["script_ok"] - stats[k]["loop"], 4), stats[k]["eos"]))


def build(out_dir: str) -> str:
    o = lambda *p: os.path.join(out_dir, *p)
    L = ["# Parentheses 0.9-120M — Session 2 report", ""]

    L += ["## Training loss (mean per 150-step bucket)", ""]
    tl = train_losses(o("parentheses-0.9-120M.log"))
    s2 = {k: v for k, v in tl.items() if 1907 <= k}
    if s2:
        L += ["| steps | n | mean loss |", "|---|---|---|"]
        for b in range(1800, max(s2) + 1, 150):
            xs = [v for k, v in tl.items() if b <= k < b + 150]
            if xs:
                L.append(f"| {b}-{b + 149} | {len(xs)} | {sum(xs) / len(xs):.4f} |")
        L += ["", f"last logged step {max(s2)}, loss {s2[max(s2)]:.4f}", ""]
    else:
        L += ["not run", ""]

    L += ["## Val watcher (CPU, 4x1024 tokens per language)", ""]
    val = jl(o("session2_val_log.jsonl"))
    if val:
        langs = list(val[0]["losses"])
        L += ["| step | " + " | ".join(langs) + " | alert |", "|---" * (len(langs) + 2) + "|"]
        for r in val:
            L.append(f"| {r['step']} | " + " | ".join(f"{r['losses'][g]:.4f}" for g in langs) + f" | {r.get('alert', '')} |")
        b, e = val[0]["losses"], val[-1]["losses"]
        L += ["", "change first->last: " + ", ".join(f"{g} {100 * (e[g] / b[g] - 1):+.1f}%" for g in langs), ""]
    else:
        L += ["not run", ""]

    L += ["## Bits per byte (eval_dravidian --bpb, same val text)", ""]
    bpb = json.load(open(o("bpb_eval_120M_s2.json"), encoding="utf-8")) if os.path.isfile(o("bpb_eval_120M_s2.json")) else []
    if bpb:
        langs = list(bpb[0]["bpb"])
        L += ["| checkpoint | " + " | ".join(langs) + " |", "|---" * (len(langs) + 1) + "|"]
        for r in bpb:
            L.append(f"| step {r['step']} | " + " | ".join(f"{r['bpb'][g]:.4f}" for g in langs) + " |")
        if len(bpb) >= 2:
            a, z = bpb[0]["bpb"], bpb[-1]["bpb"]
            L += ["", "change: " + ", ".join(f"{g} {100 * (z[g] / a[g] - 1):+.1f}%" for g in langs)]
        L.append("")
    else:
        L += ["not run", ""]

    L += ["## 250 queries", ""]
    qfiles = [("step_1601 (session 1 file)", o("queries_120m_s250_final.jsonl")),
              ("step_1907 base", o("queries_120m_s250_step1907.jsonl")),
              ("step_3815 base", o("queries_120m_s250_step3815.jsonl")),
              ("SFT trial chat", o("queries_120m_s250_sft_trial_chat.jsonl"))]
    rows = [(n, query_stats(jl(p))) for n, p in qfiles if jl(p)]
    if rows:
        L += ["| run | n | script_ok | eos | loop | rep | distinct_2 |", "|---|---|---|---|---|---|---|"]
        L += [f"| {n} | {s['n']} | {s['script_ok']:.3f} | {s['eos']:.3f} | {s['loop']:.3f} | {s['rep']:.3f} | {s['d2']:.3f} |"
              for n, s in rows]
        L.append("")
    else:
        L += ["not run", ""]

    L += ["## Decoding sweep (step_3815, first 8 queries per language)", ""]
    sw = {Path(p).stem: query_stats(jl(p)) for p in sorted(glob.glob(o("decode_sweep_s2", "*.jsonl"))) if jl(p)}
    if sw:
        L += ["| config | n | script_ok | eos | loop | rep | distinct_2 |", "|---|---|---|---|---|---|---|"]
        L += [f"| {k} | {s['n']} | {s['script_ok']:.3f} | {s['eos']:.3f} | {s['loop']:.3f} | {s['rep']:.3f} | {s['d2']:.3f} |"
              for k, s in sw.items()]
        L += ["", f"pick (rule: max script_ok - loop, then eos): **{sweep_pick(sw)}**", ""]
    else:
        L += ["not run", ""]

    L += ["## lm-eval (limit 500 per task)", ""]
    lm = [(p, json.load(open(p, encoding="utf-8"))) for p in sorted(glob.glob(o("lm_eval_120M_step*.json")))]
    if lm:
        tasks = sorted(lm[0][1]["results"])
        L += ["| checkpoint | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 1) + "|"]
        for p, d in lm:
            L.append(f"| {Path(p).stem} | " + " | ".join(
                f"{d['results'][t].get('acc_norm,none', d['results'][t].get('acc,none', float('nan'))):.3f}" for t in tasks) + " |")
        L += ["", "(acc_norm where the task has it, else acc)", ""]
    else:
        L += ["not run", ""]

    L += ["## Numbers for the session-3 decision", ""]
    if len(val) >= 3:
        m4 = [sum(r["losses"][g] for g in MAIN4) / 4 for r in val]
        steps = [r["step"] for r in val]
        late = 100 * (m4[-1] / m4[-3] - 1) / max(1, (steps[-1] - steps[-3]) / 300)
        L.append(f"- main-4 val loss {m4[0]:.4f} -> {m4[-1]:.4f}; last two intervals {late:+.2f}% per 300 steps")
    if len(s2) > 60:
        hi = max(s2)
        a = [v for k, v in s2.items() if hi - 600 < k <= hi - 300]
        z = [v for k, v in s2.items() if hi - 300 < k]
        if a and z:
            L.append(f"- train loss last 300 steps {sum(z) / len(z):.4f} vs the 300 before {sum(a) / len(a):.4f}")
    L += ["- LR 3e-4 degraded the grown model in session 1 (launch_120M_session.ps1 note); 1e-4 is the default.", ""]
    return "\n".join(L) + "\n"


def self_test():
    with tempfile.TemporaryDirectory() as d:
        assert "not run" in build(d)                      # nothing exists: no crash, no numbers
        open(os.path.join(d, "parentheses-0.9-120M.log"), "w").write(
            "".join(f"step {s:6d} | loss {4 - s / 1e4:.4f} | lr 1e-4\n" for s in range(1900, 2700, 10))
            + "step   1900 | loss 9.0000 | lr 1e-4\n")      # resumed re-log: last wins
        assert train_losses(os.path.join(d, "parentheses-0.9-120M.log"))[1900] == 9.0
        langs = {"kan": 3.0, "tam": 3.0, "tel": 3.0, "mal": 3.0, "eng": 3.5}
        with open(os.path.join(d, "session2_val_log.jsonl"), "w") as f:
            for i, s in enumerate((1907, 2100, 2400)):
                f.write(json.dumps({"step": s, "losses": {k: v - 0.1 * i for k, v in langs.items()}, "alert": False}) + "\n")
        q = lambda ok, stop: {"lang": "eng", "script_ok": ok, "stop": stop, "repetition_rate": 0.1, "distinct_2": 0.5}
        os.mkdir(os.path.join(d, "decode_sweep_s2"))
        for name, rs in {"t0.7_k50": [q(True, "loop"), q(True, "eos")], "t0.5_k40": [q(True, "eos"), q(True, "max_tokens")]}.items():
            with open(os.path.join(d, "decode_sweep_s2", name + ".jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rs)
        rep = build(d)
        assert "pick (rule: max script_ok - loop, then eos): **t0.5_k40**" in rep, rep
        assert "| 2400 | 2.8000 |" in rep and "kan -6.7%" in rep, rep
        assert "train loss last 300 steps" in rep and "per 300 steps" in rep, rep
        assert query_stats([q(True, "eos"), q(False, "loop"), q(None, "eos")])["script_ok"] == 0.5
    print("[self-test] session2_report ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="out")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    text = build(a.out_dir)
    Path(a.out_dir, "session2_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
