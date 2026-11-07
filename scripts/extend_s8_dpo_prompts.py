# Engineered by uncoalesced
"""
Extend the S8 DPO prompt set with the rows the 00:41 build left out (S8_PLAN.md section 2):
  - Dravidian QA, 75 rows each for kan / tam / tel / mal, from the local Belebele packs
    (E:\\Parentheses\\data\\curated_s7_packin\\belebele_*.jsonl, CC-BY-SA-4.0). These packs were
    also S7.5 SFT inputs; that is fine for on-policy DPO prompts (the model answers them itself).
    The gold answer goes in `ground_truth_text`, which train_dpo_v2_apache.py scores.
  Antigravity's other 900 rows (math, code, abstention, identity, general) are kept unchanged.

Decontamination (stricter than build_s8_dpo_prompts.py, which read only 500 lines of hin/san val):
  60 probes + the exact text bpb_slab_v4.py scores (processed_v5 val, 400 KB per language, and
  data/val_candidates hin/san), plus rows already in prompts.jsonl. Exact normalized hash and
  whitespace 8-gram overlap.

Idempotent: Antigravity's 1,200-row file is kept as prompts_base_ag_1200.jsonl and the output is rebuilt
from it on every run.
    .venv\\Scripts\\python.exe scripts\\extend_s8_dpo_prompts.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer  # noqa: E402
from scripts.eval_60p_adversarial_suite import PROBES  # noqa: E402

OUT_DIR = ROOT / "data" / "dpo_240m_apache"
PACK = Path(r"E:\Parentheses\data\curated_s7_packin")
TAGS = {"kan": "<kan>", "tam": "<tam>", "tel": "<tel>", "mal": "<mal>", "eng": "<eng>"}
PER_LANG = 75
N_GENERAL = 70
SEED = 20261107


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def grams(t: str, n: int = 8) -> set:
    w = norm(t).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def h(t: str) -> str:
    return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()


def val_texts(tok: Tokenizer) -> list[str]:
    langs = ["kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"]
    tag_of = {tok.token_to_id(f"<{l}>"): l for l in langs}
    val = np.memmap(ROOT / "data" / "processed_v5" / "val.bin", dtype=np.uint16, mode="r")
    vb = np.load(ROOT / "data" / "processed_v5" / "val_boundaries.npy")
    ve = np.append(vb[1:], len(val))
    out, nb = [], {}
    for s, e in zip(vb, ve):
        doc = np.asarray(val[s:e]).tolist()
        lang = tag_of.get(doc[0]) if doc else None
        if lang is None or nb.get(lang, 0) >= 400_000:
            continue
        t = tok.decode([x for x in doc[1:] if x != 0], skip_special_tokens=False)
        out.append(t)
        nb[lang] = nb.get(lang, 0) + len(t.encode("utf-8"))
    for f in (ROOT / "data" / "val_candidates").glob("*_val.txt"):
        out += [b for b in f.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    return out


def main() -> int:
    tok = Tokenizer.from_file(str(ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"))
    # Antigravity's 00:43 build (1,200 rows) is kept as the base, minus its 300 Indic rows: those are
    # Belebele questions with the passage and options stripped ("according to the passage..."),
    # unanswerable and without a gold answer. They are replaced by full passage+question rows below.
    base = OUT_DIR / "prompts_base_ag_1200.jsonl"
    base_src = OUT_DIR / "prompts_sources_base_ag_1200.jsonl"
    if not base.exists():
        shutil.copy2(OUT_DIR / "prompts.jsonl", base)
        shutil.copy2(OUT_DIR / "prompts_sources.jsonl", base_src)
    rows = [json.loads(l) for l in base.open(encoding="utf-8") if l.strip()]
    srcs = [json.loads(l) for l in base_src.open(encoding="utf-8") if l.strip()]
    assert len(rows) == 1200 == len(srcs), (len(rows), len(srcs))
    keep = [i for i, r in enumerate(rows) if r.get("lang") not in ("kn", "ta", "te", "ml", "hi")]
    rows, srcs = [rows[i] for i in keep], [srcs[i] for i in keep]
    assert len(rows) == 900, len(rows)
    # 146 abstention rows carry Indic text and an Indic tag but lang "en", so the scorer skipped its
    # script-purity check on them. Take lang from the tag.
    tag2lang = {"<kan>": "kan", "<tam>": "tam", "<tel>": "tel", "<mal>": "mal"}
    relabeled = 0
    for r in rows:
        m = re.match(r"<\|user\|>(<[a-z]{3}>)", r["prompt"])
        if m and m.group(1) in tag2lang and r.get("lang") != tag2lang[m.group(1)]:
            r["lang"] = tag2lang[m.group(1)]
            relabeled += 1
    # Drop exact duplicates (normalized) from the base; Belebele tops the set back up to 1,200.
    seen_base, dedup_r, dedup_s = set(), [], []
    for r, s in zip(rows, srcs):
        k = h(r["prompt"])
        if k not in seen_base:
            seen_base.add(k)
            dedup_r.append(r)
            dedup_s.append(s)
    n_dups = len(rows) - len(dedup_r)
    rows, srcs = dedup_r, dedup_s
    print(f"[base] relabeled lang from tag: {relabeled}; duplicates dropped: {n_dups}")
    extra = 1200 - len(rows) - 4 * PER_LANG  # spread over the 4 Belebele languages
    per_lang = {l: PER_LANG + (1 if i < extra else 0) for i, l in enumerate(("kan", "tam", "tel", "mal"))}

    banned_h, banned_g = set(), set()
    for t in [p["prompt"] for p in PROBES] + val_texts(tok):
        banned_h.add(h(t))
        banned_g |= grams(t)
    seen = {h(r["prompt"]) for r in rows}
    print(f"[decontam] {len(banned_h)} hashes, {len(banned_g)} 8-grams")

    def dirty(t: str) -> bool:
        return h(t) in banned_h or bool(grams(t) & banned_g)

    rng = random.Random(SEED)
    removed = {}
    new_rows, new_srcs = [], []
    jobs = [(l, PACK / f"belebele_{l}_{s}.jsonl", per_lang[l], "facebook/belebele", "CC-BY-SA-4.0")
            for l, s in (("kan", "Knda"), ("tam", "Taml"), ("tel", "Telu"), ("mal", "Mlym"))]
    for lang, path, k, src, lic in jobs:
        cand = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        rng.shuffle(cand)
        got, rm = 0, 0
        for c in cand:
            if got >= k:
                break
            text = (c.get("prompt") or "").strip()
            if not text or len(text) > 2500:
                continue
            prompt = f"<|user|>{TAGS[lang]}{text}<|assistant|>"
            if h(prompt) in seen:
                continue
            if dirty(text):
                rm += 1
                continue
            seen.add(h(prompt))
            row = {"prompt": prompt, "task_type": "general" if lang == "eng" else "qa", "lang": lang,
                   "ground_truth_num": None, "requires_think": False, "requires_abstention": False}
            if lang != "eng" and c.get("response"):
                row["ground_truth_text"] = c["response"].strip()
            new_rows.append(row)
            new_srcs.append({"source": src, "license": lic, "lang": lang, "domain": row["task_type"]})
            got += 1
        removed[lang] = rm
        print(f"[{lang}] kept {got}/{k}, removed by decontam {rm}")
        if got < k:
            print(f"[warn] {lang}: only {got} rows")

    for r in new_rows[:3] + new_rows[-2:]:
        ids = tok.encode(r["prompt"]).ids
        assert ids[0] == tok.token_to_id("<|user|>") and ids[1] == tok.token_to_id(TAGS[r["lang"]]) \
            and ids[-1] == tok.token_to_id("<|assistant|>"), ids[:3]

    allr, alls = rows + new_rows, srcs + new_srcs
    with (OUT_DIR / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT_DIR / "prompts_sources.jsonl").open("w", encoding="utf-8") as f:
        for s in alls:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    man = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    man["total_prompts"] = len(allr)
    man["domains"]["dravidian_devanagari_qa"] = sum(r["lang"] in ("kan", "tam", "tel", "mal") for r in new_rows)
    man["extension"] = {"script": "scripts/extend_s8_dpo_prompts.py", "removed_by_decontam": removed,
                        "decontam": "60 probes + processed_v5 val (400KB/lang, as bpb_slab_v4) + hin/san val; exact hash + 8-gram"}
    (OUT_DIR / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] prompts.jsonl: {len(allr)} rows ({len(new_rows)} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
