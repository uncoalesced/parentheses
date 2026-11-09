# Engineered by uncoalesced
"""
Mechanical gates for the unattended S7.5 -> S8 chain (vault S7_5_TO_S8_RUNBOOK.md section 4,
S8_PLAN.md sections 3-4). Used by scripts/run_session8_auto.ps1. Every subcommand writes a JSON
verdict and exits 0 = pass, 1 = fail, 2 = could not evaluate (missing or unreadable input).

  s75     G1-G5 on out/eval_s7_5 and pick the S8 base (SWA-3 or final)
  pairs   after --mode generate: >= --min-pairs pairs, every chosen score > rejected score,
          plus mechanical length/EOT proxies and 20 random pairs dumped for human review
  s8      S8 pass bar: BPB vs S6, greedy >= base, bare-IDK and loops not above base
  sanity  output-sanity counts for one or more 60-probe JSONL files (report only, exit 0)

    .venv\\Scripts\\python.exe scripts\\s8_gates.py s75
    .venv\\Scripts\\python.exe scripts\\s8_gates.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ["kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng", "hin", "san"]
MAX_RISE = 0.06
IDK = {"i don't know.", "i don't know", "i do not know.", "i do not know"}
SCRIPT_RANGES = {"kan": (0x0C80, 0x0CFF), "tam": (0x0B80, 0x0BFF), "tel": (0x0C00, 0x0C7F), "mal": (0x0D00, 0x0D7F)}


# ----------------------------------------------------------------------------- output sanity
def word_4gram_loop(gen: str, n: int = 4, reps: int = 3) -> bool:
    """Runbook definition: some whitespace 4-gram occurs 3+ times."""
    w = gen.split()
    seen: dict = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= reps:
            return True
    return False


def line_loop(gen: str, reps: int = 3) -> bool:
    """score_60p.has_loop_repetition: a normalized line repeated 3+ times."""
    c: dict = {}
    for line in gen.splitlines():
        s = re.sub(r"^\s*(?:\d+[\.\)]|\*|\-|∴)\s*", "", line.strip().lower()).strip()
        if len(s) >= 4:
            c[s] = c.get(s, 0) + 1
            if c[s] >= reps:
                return True
    return False


def closed_think(gen: str) -> bool:
    a, b = gen.find("<think>"), gen.find("</think>")
    return a != -1 and b != -1 and a < b


def drav_lang(rec: dict) -> str | None:
    cat = rec.get("category", "") + " " + rec.get("domain", "")
    for name, code in (("Kannada", "kan"), ("Tamil", "tam"), ("Telugu", "tel"), ("Malayalam", "mal")):
        if name in cat:
            return code
    return None


def in_script(gen: str, lang: str) -> bool:
    lo, hi = SCRIPT_RANGES[lang]
    body = gen.split("</think>")[-1]
    alpha = [ch for ch in body if ord(ch) > 127 and ch.isalpha()]
    return bool(alpha) and sum(lo <= ord(ch) <= hi for ch in alpha) / len(alpha) >= 0.5


def sanity(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("sample", 0) == 0]  # greedy files: one per probe
    math = [r for r in recs if str(r.get("id", "")).startswith("math-")]
    drav = [(r, drav_lang(r)) for r in recs if drav_lang(r)]
    return {
        "file": str(path),
        "n": len(recs),
        "bare_idk": sum(r.get("generation", "").strip().lower() in IDK for r in recs),
        "math_total": len(math),
        "math_closed_think": sum(closed_think(r.get("generation", "")) for r in math),
        "loops_4gram": sum(word_4gram_loop(r.get("generation", "")) for r in recs),
        "loops_line": sum(line_loop(r.get("generation", "")) for r in recs),
        "drav_total": len(drav),
        "drav_right_script": sum(in_script(r.get("generation", ""), l) for r, l in drav),
        "tool_call_on_code": sum("<tool_call>" in r.get("generation", "") for r in recs
                                 if str(r.get("id", "")).startswith("code-")),
    }


# ----------------------------------------------------------------------------- score / bpb readers
def score_entry(scores: list, name: str) -> dict | None:
    for s in scores:
        if Path(s["file"]).name == name:
            return s
    return None


def bpb_rise(bpb: list, idx: int) -> dict:
    base, new = bpb[0]["bpb"], bpb[idx]["bpb"]
    return {l: round(new[l] - base[l], 4) for l in LANGS if l in base and l in new}


def bpb_ok(rise: dict) -> bool:
    return len(rise) == len(LANGS) and all(v <= MAX_RISE for v in rise.values())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- s75
def gate_s75(a) -> int:
    ev, ck, arch = Path(a.eval_dir), Path(a.ckpt_dir), Path(a.archive_dir)
    out: dict = {"gates": {}, "notes": []}
    final, latest, swa = ck / "step_13000_final.pt", ck / "latest.pt", ck / "step_13000_swa3.pt"

    # G1 run health, from files (the supervisor's exit code is not visible to this process)
    g1 = []
    if not final.exists():
        g1.append("step_13000_final.pt missing")
    elif not latest.exists() or latest.stat().st_size != final.stat().st_size:
        g1.append("latest.pt missing or size != final")
    log = ROOT / "out" / "session7_5_240m_live.log"
    if log.exists():
        raw = log.read_bytes()  # PS5 Tee/redirect writes UTF-16 LE with BOM
        txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
        steps = [int(m) for m in re.findall(r"^step\s+(\d+)", txt, re.M)]
        if not steps or max(steps) < 13000:
            g1.append(f"log max step {max(steps) if steps else None} < 13000")
        if re.search(r"\bnan\b|\binf\b", txt, re.I):
            g1.append("nan/inf in training log")
    else:
        g1.append("training log missing")
    man = arch / "MANIFEST_s7_5_sha256.tsv"
    rows = {}
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                rows[parts[1]] = parts[0]
    hashes = {}
    for name in ("step_13000_final.pt", "step_12500.pt", "step_12750.pt"):
        src, dst = ck / name, arch / name
        if name not in rows or not dst.exists():
            g1.append(f"{name} not in I: manifest/archive")
            continue
        if not a.skip_hash:
            h_src = sha256(src) if src.exists() else None
            hashes[name] = h_src
            if h_src != rows[name]:
                g1.append(f"{name} sha256 mismatch vs manifest")
    need = ["scores.json", "bpb_240m.json", "s7_5_chat_greedy.jsonl", "s7_5_chat_t07x5.jsonl"]
    if swa.exists():
        need.append("s7_5_swa3_chat_greedy.jsonl")
    for n in need:
        if not (ev / n).exists():
            g1.append(f"eval output missing: {n}")
    out["gates"]["G1_run_health"] = {"pass": not g1, "problems": g1, "rehashed": hashes}
    if not (ev / "scores.json").exists() or not (ev / "bpb_240m.json").exists():
        return write(out, ev / "gate_verdict_claude.json", 2, "scores.json / bpb_240m.json missing")

    scores = json.loads((ev / "scores.json").read_text(encoding="utf-8"))
    bpb = json.loads((ev / "bpb_240m.json").read_text(encoding="utf-8"))
    cands = {"final": {"ckpt": str(final), "greedy": "s7_5_chat_greedy.jsonl", "bpb_idx": 1}}
    if swa.exists() and (ev / "s7_5_swa3_chat_greedy.jsonl").exists() and len(bpb) >= 3:
        cands["swa3"] = {"ckpt": str(swa), "greedy": "s7_5_swa3_chat_greedy.jsonl", "bpb_idx": 2}
    if Path(bpb[0]["checkpoint"]).name != "step_12000.pt":
        out["notes"].append(f"bpb[0] is {bpb[0]['checkpoint']}, expected S6 step_12000.pt")

    s6 = sanity(ROOT / "out" / "eval_s6_12k" / "s6_12000_chat_greedy.jsonl")
    out["s6_sanity"] = s6
    for name, c in cands.items():
        s = score_entry(scores, c["greedy"])
        rise = bpb_rise(bpb, c["bpb_idx"])
        san = sanity(ev / c["greedy"])
        c["greedy_total"] = s["total"][0] if s else None
        c["bpb_rise"] = rise
        c["sanity"] = san
        c["G2"] = bpb_ok(rise)
        c["G3"] = s is not None and s["total"][0] >= 5
        # G5 per Joel 2026-11-07 ~01:00: "not worse than S6" (S6 greedy: 12/60 4-gram loops, 2/10 closed
        # math think). The runbook's absolute <=3 / >=3 bar fails S6 itself.
        c["G5"] = (san["bare_idk"] <= 3 and san["math_closed_think"] >= s6["math_closed_think"]
                   and san["loops_4gram"] <= s6["loops_4gram"])
        c["ok"] = c["G2"] and c["G3"] and c["G5"]
    samp = score_entry(scores, "s7_5_chat_t07x5.jsonl")
    g4_val = samp["total"][1] if samp else None
    out["gates"]["G4_sampled_pass5"] = {"pass": g4_val is not None and g4_val >= 8, "value": g4_val}
    out["candidates"] = cands
    for g in ("G2", "G3", "G5"):
        out["gates"][g] = {k: v[g] for k, v in cands.items()}

    base = None
    if "swa3" in cands and cands["swa3"]["ok"] and (
            not cands["final"]["ok"] or cands["swa3"]["greedy_total"] >= cands["final"]["greedy_total"]):
        base = "swa3"
    elif cands["final"]["ok"]:
        base = "final"
    all_pass = out["gates"]["G1_run_health"]["pass"] and out["gates"]["G4_sampled_pass5"]["pass"] and base is not None
    out["all_pass"] = all_pass
    out["candidate_base_name"] = base
    out["candidate_base"] = cands[base]["ckpt"] if base else None
    out["base_greedy_file"] = str(ev / cands[base]["greedy"]) if base else None
    return write(out, ev / "gate_verdict_claude.json", 0 if all_pass else 1,
                 f"S7.5 gates {'PASS' if all_pass else 'FAIL'}; base={base}")


# ----------------------------------------------------------------------------- pairs
def gate_pairs(a) -> int:
    p = Path(a.pairs)
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return write({}, outdir / "pairs_check.json", 2, f"{p} missing")
    pairs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(pairs)
    bad_margin = sum(not (x["score_chosen"] > x["score_rejected"]) for x in pairs)
    longer = sum(len(x["chosen_ids"]) > len(x["rejected_ids"]) for x in pairs)
    ch_eot = sum(bool(x["chosen_ids"]) and x["chosen_ids"][-1] == 0 for x in pairs)
    rj_eot = sum(bool(x["rejected_ids"]) and x["rejected_ids"][-1] == 0 for x in pairs)
    by_dom: dict = {}
    for x in pairs:
        k = f"{x.get('task_type')}/{x.get('lang')}"
        by_dom[k] = by_dom.get(k, 0) + 1
    out = {"pairs": n, "min_pairs": a.min_pairs, "bad_margin": bad_margin,
           "frac_chosen_longer": round(longer / max(n, 1), 3),
           "frac_chosen_eot": round(ch_eot / max(n, 1), 3), "frac_rejected_eot": round(rj_eot / max(n, 1), 3),
           "by_domain": by_dom}
    rng = random.Random(20261107)
    lines = ["# S8 pairs: 20 random for human review (Hermes / Joel)\n",
             "For each: is chosen actually better, or just longer / keyword-stuffed? >4/20 wrong = verifier gamed.\n"]
    for i, x in enumerate(rng.sample(pairs, min(20, n)), 1):
        lines += [f"\n## {i}. {x.get('task_type')}/{x.get('lang')}  chosen {x['score_chosen']} vs rejected {x['score_rejected']}\n",
                  f"**Prompt:** {x['prompt'][:600]}\n", f"\n**Chosen:**\n```\n{x['chosen'][:1200]}\n```\n",
                  f"\n**Rejected:**\n```\n{x['rejected'][:1200]}\n```\n"]
    (outdir / "pairs_review_20.md").write_text("".join(lines), encoding="utf-8")
    ok = n >= a.min_pairs and bad_margin == 0
    return write(out, outdir / "pairs_check.json", 0 if ok else 1,
                 f"pairs {n} (need {a.min_pairs}), bad margins {bad_margin}, chosen longer {out['frac_chosen_longer']}")


# ----------------------------------------------------------------------------- s8 pass bar
def gate_s8(a) -> int:
    ev = Path(a.eval_dir)
    try:
        scores = json.loads((ev / "scores.json").read_text(encoding="utf-8"))
        bpb = json.loads((ev / "bpb_240m.json").read_text(encoding="utf-8"))
        san_base, san_gold = sanity(Path(a.base_greedy)), sanity(ev / a.golden_greedy)
    except Exception as e:  # noqa: BLE001
        return write({}, ev / "s8_verdict.json", 2, f"missing/unreadable S8 eval files: {e}")
    base_g = score_entry(scores, Path(a.base_greedy).name)
    gold_g = score_entry(scores, a.golden_greedy)
    rise = bpb_rise(bpb, len(bpb) - 1)
    checks = {
        "bpb_within_0.06_vs_S6": bpb_ok(rise),
        "greedy_ge_base": bool(base_g and gold_g and gold_g["total"][0] >= base_g["total"][0]),
        "bare_idk_not_higher": san_gold["bare_idk"] <= san_base["bare_idk"],
        "loops_not_higher": san_gold["loops_4gram"] <= san_base["loops_4gram"],
    }
    out = {"checks": checks, "bpb_rise_golden_vs_s6": rise,
           "greedy": {"base": base_g["total"][0] if base_g else None, "golden": gold_g["total"][0] if gold_g else None},
           "sanity": {"base": san_base, "golden": san_gold},
           "note": "identity answers are checked by s8_identity_check.py and reported, not gated here"}
    ok = all(checks.values())
    return write(out, ev / "s8_verdict.json", 0 if ok else 1, f"S8 pass bar {'PASS' if ok else 'FAIL'}: {checks}")


def write(obj: dict, path: Path, code: int, msg: str) -> int:
    obj["exit_code"], obj["message"] = code, msg
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(msg)
    print(f"[verdict] {path}")
    return code


def self_test() -> None:
    assert word_4gram_loop("a b c d a b c d a b c d") and not word_4gram_loop("a b c d e f")
    assert line_loop("x yy\nx yy\nx yy") and not line_loop("one\ntwo")
    assert closed_think("<think>x</think> y") and not closed_think("</think><think>")
    assert in_script("ಕನ್ನಡ ಭಾಷೆ", "kan") and not in_script("hello", "kan")
    assert bpb_ok({l: 0.01 for l in LANGS}) and not bpb_ok({**{l: 0.0 for l in LANGS}, "nit": 0.07})
    assert not bpb_ok({"kan": 0.0})
    print("s8_gates self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["s75", "pairs", "s8", "sanity"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval-dir", default=str(ROOT / "out" / "eval_s7_5"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "parentheses-1.0-240M-s7_5"))
    ap.add_argument("--archive-dir", default=r"I:\parentheses\checkpoints\parentheses-1.0-240M-s7_5")
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--pairs", default=str(ROOT / "data" / "dpo_240m_apache" / "on_policy_pairs.jsonl"))
    ap.add_argument("--min-pairs", type=int, default=800)
    ap.add_argument("--out-dir", default=str(ROOT / "out" / "eval_s8"))
    ap.add_argument("--base-greedy", default="")
    ap.add_argument("--golden-greedy", default="s8_golden_chat_greedy.jsonl")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.cmd == "s75":
        return gate_s75(a)
    if a.cmd == "pairs":
        return gate_pairs(a)
    if a.cmd == "s8":
        return gate_s8(a)
    if a.cmd == "sanity":
        print(json.dumps([sanity(Path(f)) for f in a.files], ensure_ascii=False, indent=2))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
