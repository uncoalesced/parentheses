#!/usr/bin/env python3
"""All-sessions stats panel for Parentheses 0.9-120M (Claude Code, 2026-10-16; Joel: presentable stats of every session).

Collects only numbers already on disk (training log, bpb / lm-eval / MC / eval250x2 / s250 outputs) and writes one
self-contained HTML page (scripts/stats_page_template.html) with the data embedded.

    .venv\\Scripts\\python.exe scripts\\build_stats_page.py          # -> out/stats/stats_page.html
    .venv\\Scripts\\python.exe scripts\\build_stats_page.py --self-test
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from session2_report import query_stats   # noqa: E402

STEP_RE = re.compile(r"^step\s+(\d+) \| loss ([\d.]+) \| lr ([\d.e+-]+) \| ([\d.]+)s elapsed", re.M)
PER_SESSION, TOTAL, TOK_PER_STEP = 1908, 9537, 131072
LANGS = ["kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"]
TEMPLATE = Path(__file__).with_name("stats_page_template.html")


def session_of(step: int) -> int:
    return min(step // PER_SESSION, 4) + 1


def parse_log(text: str) -> tuple[list, dict]:
    """Loss curve (mean per 50-step bucket, last write per step wins) and per-session wall time / launches."""
    segs, cur = [], []
    for line in text.splitlines():
        if line.startswith("[driver]") and " launch: " in line:
            if cur:
                segs.append(cur)
            cur = []
        m = STEP_RE.match(line)
        if m:
            cur.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if cur:
        segs.append(cur)
    steps = {}
    sess = {s: {"launches": 0, "seconds": 0.0, "steps": 0} for s in range(1, 6)}
    for seg in segs:
        for st, loss, lr, _ in seg:
            steps[st] = (loss, lr)
        if len(seg) >= 2:
            s = session_of(seg[0][0])
            sess[s]["launches"] += 1
            sess[s]["seconds"] += seg[-1][3] - seg[0][3]
            sess[s]["steps"] += seg[-1][0] - seg[0][0]
    buckets = {}
    for st, (loss, lr) in steps.items():
        buckets.setdefault(st // 50, []).append((loss, lr))
    curve = [{"step": b * 50 + 25, "loss": round(sum(x for x, _ in v) / len(v), 4),
              "lr": max(y for _, y in v)} for b, v in sorted(buckets.items())]
    for s in sess.values():
        s["tok_per_s"] = round(s["steps"] * TOK_PER_STEP / s["seconds"]) if s["seconds"] else None
        s["hours"] = round(s["seconds"] / 3600, 2)
    return curve, sess


def letterless(path: str) -> dict:
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            b = out.setdefault(r["bucket"], [0, 0])
            b[1] += 1
            b[0] += not any(ch.isalpha() for ch in r["output"])
    return out


def collect(out_dir: str) -> dict:
    o = lambda *p: os.path.join(out_dir, *p)
    log = o("parentheses-0.9-120M.log")
    curve, sess = parse_log(open(log, encoding="utf-8", errors="replace").read()) if os.path.isfile(log) else ([], {})
    bpb = {}
    for f in sorted(glob.glob(o("bpb_eval_120M_s*.json"))):
        for x in json.load(open(f, encoding="utf-8")):
            bpb[x["step"]] = {k: round(v, 4) for k, v in x["bpb"].items()}
    lm = {}
    for f in glob.glob(o("lm_eval_120M_step*.json")):
        st = int(re.search(r"step(\d+)", f).group(1))
        r = json.load(open(f, encoding="utf-8"))
        r = r.get("results", r)
        lm[st] = {k: round(v.get("acc,none", v.get("acc", 0)), 3) for k, v in r.items() if isinstance(v, dict)}
    mc = {int(re.search(r"step(\d+)", f).group(1)): {k: v["acc_norm"] for k, v in json.load(open(f))["by_bucket"].items()}
          for f in glob.glob(o("mc_reason_step*.json"))}
    let = {int(re.search(r"step(\d+)", f).group(1)): letterless(f) for f in glob.glob(o("eval250x2", "reason_step*.jsonl"))}
    q = {}
    for f in glob.glob(o("queries_120m_s250_step*_t0.9k50.jsonl")):
        st = int(re.search(r"step(\d+)", f).group(1))
        s = query_stats([json.loads(l) for l in open(f, encoding="utf-8") if l.strip()])
        q[st] = {k: round(v, 3) for k, v in s.items() if k != "n"}
    last = max(curve, key=lambda c: c["step"])["step"] if curve else 0
    return {"params": 119702272, "tokens_total": TOTAL * TOK_PER_STEP, "tok_per_step": TOK_PER_STEP,
            "per_session": PER_SESSION, "last_step": last, "curve": curve,
            "sessions": [dict(id=k, **v) for k, v in sorted(sess.items())],
            "bpb": [dict(step=k, **v) for k, v in sorted(bpb.items())], "langs": LANGS,
            "lm": [dict(step=k, **v) for k, v in sorted(lm.items())],
            "mc": [dict(step=k, **v) for k, v in sorted(mc.items())],
            "letterless": [{"step": k, **{b: v for b, v in d.items()}} for k, d in sorted(let.items())],
            "s250": [dict(step=k, **v) for k, v in sorted(q.items())]}


def render(data: dict) -> str:
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))


def self_test():
    log = ("[driver] 00:00:01 launch: x\nstep      0 | loss 3.1000 | lr 2.00e-07 | 10.0s elapsed\n"
           "step     10 | loss 3.0000 | lr 2.20e-06 | 110.0s elapsed\n"
           "[driver] 00:10:00 launch: x\nstep   1910 | loss 2.9000 | lr 1.00e-04 | 5.0s elapsed\n"
           "step   1920 | loss 2.8000 | lr 1.00e-04 | 105.0s elapsed\n")
    curve, sess = parse_log(log)
    assert [c["step"] for c in curve] == [25, 1925] and curve[0]["loss"] == 3.05, curve
    assert sess[1]["launches"] == 1 and sess[1]["steps"] == 10 and sess[1]["seconds"] == 100.0
    assert sess[2]["tok_per_s"] == round(10 * TOK_PER_STEP / 100) and sess[3]["launches"] == 0
    assert session_of(1907) == 1 and session_of(1908) == 2 and session_of(9536) == 5
    print("[self-test] build_stats_page ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="out")
    p.add_argument("--out", default="out/stats/stats_page.html")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    d = collect(a.out_dir)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render(d), encoding="utf-8")
    print(f"{a.out}: last step {d['last_step']}, bpb checkpoints {[b['step'] for b in d['bpb']]}, "
          f"sessions {[(s['id'], s['launches'], s['hours']) for s in d['sessions']]}")


if __name__ == "__main__":
    main()
