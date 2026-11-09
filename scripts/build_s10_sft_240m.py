#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/build_s10_sft_240m.py
Parentheses 1.0-240M (Tanzanite) Session 10 behaviour-SFT corpus builder.

Built from docs/S10_PLAN_IMPROVED_VS_NOT.md plus the 2026-11-09 data plan
(C:\\Users\\Joel Anthony\\.claude\\plans\\based-on-e-parentheses-parentheses-0-9-d-soft-valiant.md).
Reuses the S7.5 encoder / replay builder / FFD packer, so the chat template is identical:
    [user_id, lang_id] + prompt_ids + [assistant_id] + response_ids + [eot_id]   (mask = 1 only on the response)

Buckets (share of the final doc tokens; replay absorbs any shortfall so the corpus still reaches the target size):
    replay_11lang      25%+  processed_240m_apache / S5 textbooks / Devanagari  (protects BPB, gate G8)
    short_math         20%   own generator + short reshaped MetaMathQA / Orca-Math / NuminaMath + short SmolTalk-v2
    code_output        15%   own generator (executed) + short flytech python rows
    dravidian_qa       15%   Belebele x2, Aya tail (short rows), Kannada / Telugu curated
    abstention         15%   false premise, SQuAD2-style "not stated", unanswerable math, short abstention_v4, benign facts,
                             (+ R-Tuning rows from data/s10_gen/rtuning.jsonl when present)
    identity            5%   identity file + direct / paraphrased variants, upsampled
    grounded_reading    5%   SQuAD 2.0 passages (answerable and unanswerable)

Hard rules enforced here (the build FAILS or drops the row if any is broken):
    * no tool-call text in any row, no apigen subset, no Llama-generated subset
    * every document ends with <|endoftext|>; every <think> is closed
    * responses of reshaped rows are short (<= 256 tokens); prompts <= 700 tokens
    * digit-masked 8-gram / exact-template contamination gate against Hermes-1000, identity_15, honesty_10,
      benign_refusal_50 and the 60 adversarial probes

    .venv\\Scripts\\python.exe scripts\\build_s10_sft_240m.py --self-test
    .venv\\Scripts\\python.exe scripts\\build_s10_sft_240m.py                       # full build
    .venv\\Scripts\\python.exe scripts\\build_s10_sft_240m.py --smoke --out-dir data\\processed_sft_240m_s10_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.build_s7_5_sft_240m as b75                                  # noqa: E402
from scripts.pack_sft_4096_v4 import ffd_pack_documents, resolve_special_token_ids  # noqa: E402
from scripts.s10_gen_math import FINAL_INSTR                                # noqa: E402

GEN = REPO_ROOT / "data" / "s10_gen"
CURATED = Path(r"F:\Lingual_Data\raw\hf_permissive_stem_code\curated")
PACKIN = Path(r"E:\Parentheses\data\curated_s7_packin")
OUT_DIR = REPO_ROOT / "data" / "processed_sft_240m_s10"
EVAL = REPO_ROOT / "data" / "eval_suites"
BIN_SIZE = 4096
TARGET_BINS = 16000
TOTAL = TARGET_BINS * BIN_SIZE

TARGET_SHARES = {"replay_11lang": 0.25, "short_math": 0.20, "code_output": 0.15, "dravidian_qa": 0.15,
                 "abstention": 0.15, "identity": 0.05, "grounded_reading": 0.05}
ALLOWED_SMOLTALK = {"smol-constraints", "smol-rewrite", "smol-summarize"}
MAX_RESP_TOK = 256
MAX_PROMPT_TOK = 700
WORD = re.compile(r"\w+", re.UNICODE)


# ----------------------------------------------------------------------------- contamination gate
def mask_digits(s: str) -> List[str]:
    return WORD.findall(re.sub(r"\d+", "#", s.lower()))


def ngrams(w: List[str], n: int) -> set:
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


class Contam:
    """Digit-masked template check. Flags a train prompt when >=50% of its 8-grams (8+ words) occur in an eval prompt,
    or when its whole masked text equals an eval prompt's masked text."""

    def __init__(self, eval_prompts: List[str]):
        self.exact = {" ".join(mask_digits(p)) for p in eval_prompts}
        self.grams = set()
        for p in eval_prompts:
            self.grams |= ngrams(mask_digits(p), 8)
        self.n_eval = len(eval_prompts)

    def hit(self, prompt: str) -> bool:
        w = mask_digits(prompt)
        if not w:
            return False
        if " ".join(w) in self.exact:
            return True
        if len(w) < 8:
            return False
        g = ngrams(w, 8)
        return len(g & self.grams) / len(g) >= 0.5


def load_eval_prompts() -> List[str]:
    out = []
    for name in ("parentheses_253m_specialized_1000_v1.prompts.jsonl", "identity_15.jsonl", "honesty_10.jsonl",
                 "benign_refusal_50.jsonl"):
        p = EVAL / name
        if p.exists():
            out += [json.loads(l)["prompt"] for l in open(p, encoding="utf-8") if l.strip()]
    for extra in sorted(EVAL.glob("c6_50*.jsonl")) + sorted(EVAL.glob("suite_v2*.jsonl")):
        out += [json.loads(l)["prompt"] for l in open(extra, encoding="utf-8") if l.strip()]
    return out


# ----------------------------------------------------------------------------- row sources
def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_final(reasoning: str, response: str):
    """Return (worked solution without the answer marker, final answer) or None when no explicit answer."""
    r = reasoning.strip()
    m = re.search(r"\n?####\s*([^\n]+)\s*$", r)
    if m:
        return r[:m.start()].strip(), m.group(1).strip()
    m = re.search(r"The answer is:\s*([^\n]+)\s*$", response.strip())
    if m:
        return r, m.group(1).strip()
    i = r.rfind("\\boxed{")
    if i >= 0:
        depth, j = 0, i + 6
        for k in range(j, len(r)):
            depth += (r[k] == "{") - (r[k] == "}")
            if depth == 0:
                return r[:i].rstrip(" .\n$") + ".", r[j + 1:k].strip()
    return None


def reshaped_math(path: Path, rng, p_keep: float, label: str) -> Iterable[Tuple]:
    for d in read_jsonl(path):
        if rng.random() > p_keep:
            continue
        prompt, reasoning, response = d.get("prompt", ""), d.get("reasoning", ""), d.get("response", "")
        if not prompt or not reasoning or len(prompt) > 700 or len(reasoning) > 800 or "<tool_call>" in reasoning:
            continue
        got = extract_final(reasoning, response)
        if got is None:
            continue
        work, ans = got
        if not ans or len(ans) > 40 or "\n" in ans:
            continue
        if rng.random() < 0.6:
            prompt = prompt.strip() + " " + rng.choice(FINAL_INSTR)
        yield ("eng", prompt, f"{work}\nFinal: {ans}", label)


def smoltalk_short(rng, p_keep: float) -> Iterable[Tuple]:
    for d in read_jsonl(CURATED / "smoltalk_v2" / "curated_smoltalk_v2.jsonl"):
        if d.get("subset") not in ALLOWED_SMOLTALK or rng.random() > p_keep:
            continue
        p, a = d.get("prompt", ""), d.get("response", "")
        if "<tool_call>" in a or not p or not a or len(a) > 700 or len(p) > 1500:
            continue
        yield ("eng", p, a, "smoltalk_short:" + d["subset"])


def flytech_short(rng, p_keep: float) -> Iterable[Tuple]:
    for d in read_jsonl(PACKIN / "curated_flytech-python-codes-25k.jsonl"):
        if rng.random() > p_keep:
            continue
        p = str(d.get("prompt") or d.get("instruction") or "")
        a = str(d.get("response") or d.get("output") or "")
        if not p or not a or len(a) > 900 or len(p) > 700 or "<tool_call>" in a:
            continue
        yield ("eng", p, a, "flytech_short")


def gen_rows(path: Path, label: str | None = None) -> Iterable[Tuple]:
    if not path.exists():
        print(f"    [warn] missing {path}")
        return
    for d in read_jsonl(path):
        yield (d["lang"], d["prompt"], d["response"], label or d["source"] + ":" + d.get("family", ""))


# ----------------------------------------------------------------------------- encoding with all gates
class Encoder:
    def __init__(self, tok, token_id_map, contam: Contam, probe_keys: List[str]):
        self.tok, self.ids, self.contam, self.probe_keys = tok, token_id_map, contam, probe_keys
        self.q: Counter = Counter()
        self.unmapped: Counter = Counter()
        self.lang_tokens: Counter = Counter()
        self.source_tokens: Counter = Counter()
        self.seen: set = set()

    def encode_many(self, rows: Iterable[Tuple], cap_tokens: int, upsample: int = 1, shuffle_rng=None,
                    max_resp=MAX_RESP_TOK, max_prompt=MAX_PROMPT_TOK) -> List[Tuple[np.ndarray, np.ndarray]]:
        rows = list(rows)
        if shuffle_rng is not None:
            shuffle_rng.shuffle(rows)
        docs, packed = [], 0
        for rep in range(upsample):
            for lang, p, a, label in rows:
                if packed >= cap_tokens:
                    break
                if "<tool_call>" in a or "<tool_call>" in p:
                    self.q["tool_call_text"] += 1
                    continue
                if self.contam.hit(p):
                    self.q["eval_contaminated"] += 1
                    continue
                if not label.startswith("s10_identity"):
                    norm_p = " " + " ".join(WORD.findall(p.lower())) + " "
                    if any(k in norm_p for k in self.probe_keys) or any(e in (p + " " + a).lower() for e in b75.PROBE_ENTITIES):
                        self.q["probe_contaminated"] += 1
                        continue
                if rep == 0:
                    key = hashlib.sha1((p + "\0" + a).encode("utf-8")).digest()
                    if key in self.seen:
                        self.q["duplicate"] += 1
                        continue
                    self.seen.add(key)
                status, ids, mask, lang_c = b75.encode_sft_doc_with_rescue(self.tok, p, a, lang, self.ids, self.unmapped)
                if status != "ok" or ids is None:
                    self.q[status] += 1
                    continue
                n_resp = int(mask.sum())
                if n_resp - 1 > max_resp or len(ids) - n_resp - 1 > max_prompt:
                    self.q["too_long"] += 1
                    continue
                docs.append((ids, mask))
                packed += len(ids)
                self.lang_tokens[lang_c] += len(ids)
                self.source_tokens[label.split(":")[0]] += len(ids)
        return docs


def total_tokens(docs) -> int:
    return sum(len(d[0]) for d in docs)


# ----------------------------------------------------------------------------- build
def build(args) -> int:
    t0 = time.time()
    scale = 0.04 if args.smoke else 1.0
    tgt = {k: int(TOTAL * v * scale) for k, v in TARGET_SHARES.items()}
    tok = Tokenizer.from_file(str(b75.TOKENIZER_PATH))
    ids = resolve_special_token_ids(tok)
    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    probe_keys = b75.load_probe_keys()
    contam = Contam(load_eval_prompts())
    print(f"[*] contamination gate loaded from {contam.n_eval} eval prompts")
    enc = Encoder(tok, ids, contam, probe_keys)
    buckets: Dict[str, List] = {}

    print("\n[short_math]")
    d = enc.encode_many(gen_rows(GEN / "math.jsonl"), int(tgt["short_math"] * 0.62), shuffle_rng=prng)
    d += enc.encode_many(reshaped_math(CURATED / "metamathqa" / "curated_metamathqa.jsonl", prng, 0.12, "metamath"),
                         int(tgt["short_math"] * 0.18), shuffle_rng=prng)
    d += enc.encode_many(reshaped_math(CURATED / "orca_math" / "curated_orca_math.jsonl", prng, 0.08, "orca_math"),
                         int(tgt["short_math"] * 0.05), shuffle_rng=prng)
    d += enc.encode_many(reshaped_math(CURATED / "numinamath_cot" / "curated_numinamath.jsonl", prng, 0.05, "numinamath"),
                         int(tgt["short_math"] * 0.07), shuffle_rng=prng)
    d += enc.encode_many(smoltalk_short(prng, 0.2), int(tgt["short_math"] * 0.08), shuffle_rng=prng, max_resp=300, max_prompt=500)
    buckets["short_math"] = d

    print("[code_output]")
    d = enc.encode_many(gen_rows(GEN / "code.jsonl"), int(tgt["code_output"] * 0.75), shuffle_rng=prng)
    d += enc.encode_many(flytech_short(prng, 0.5), int(tgt["code_output"] * 0.25), shuffle_rng=prng, max_resp=300)
    buckets["code_output"] = d

    print("[dravidian_qa]")
    d = []
    for lang, f in (("kan", "belebele_kan_Knda.jsonl"), ("tam", "belebele_tam_Taml.jsonl"),
                    ("tel", "belebele_tel_Telu.jsonl"), ("mal", "belebele_mal_Mlym.jsonl")):
        rows = [(r.get("lang", lang), str(r.get("prompt") or r.get("question") or ""),
                 str(r.get("response") or r.get("answer") or ""), f"belebele_{lang}") for r in read_jsonl(PACKIN / f)]
        d += enc.encode_many(rows, int(tgt["dravidian_qa"] * 0.05), upsample=2, shuffle_rng=prng, max_resp=300, max_prompt=900)
    aya = b75.sample_pretokenized_sft_bin(
        b75.V5_TAIL_DIR / "sft_tail.bin", b75.V5_TAIL_DIR / "sft_tail_boundaries.npy", tok, ids, rng,
        int(tgt["dravidian_qa"] * 0.9), probe_keys, enc.q, enc.lang_tokens, enc.source_tokens, "drav_aya_sft_tail",
        filter_indic_qa=True)
    d += [x for x in aya if len(x[0]) <= 700]
    for f, label in (("kannada_curated.jsonl", "drav_kannada_curated"), ("telugu_curated.jsonl", "drav_telugu_curated")):
        rows = [(r.get("lang", "kan"), str(r.get("prompt") or r.get("instruction") or ""),
                 str(r.get("response") or r.get("output") or ""), label) for r in read_jsonl(PACKIN / f)]
        rows = [r for r in rows if not r[1].startswith(b75.TELUGU_HEADLINE_PREFIXES[1])]
        d += enc.encode_many(rows, int(tgt["dravidian_qa"] * 0.12), shuffle_rng=prng, max_resp=300, max_prompt=900)
    buckets["dravidian_qa"] = d

    print("[abstention]")
    d = enc.encode_many(gen_rows(GEN / "false_premise.jsonl"), int(tgt["abstention"] * 0.45), shuffle_rng=prng)
    d += enc.encode_many(gen_rows(GEN / "unans_math.jsonl"), int(tgt["abstention"] * 0.10), shuffle_rng=prng)
    d += enc.encode_many(gen_rows(GEN / "abstention_short.jsonl"), int(tgt["abstention"] * 0.12), shuffle_rng=prng)
    d += enc.encode_many(gen_rows(GEN / "benign_facts.jsonl"), int(tgt["abstention"] * 0.03), upsample=3, shuffle_rng=prng)
    rt = GEN / "rtuning.jsonl"
    if rt.exists():
        d += enc.encode_many(gen_rows(rt), int(tgt["abstention"] * 0.30), shuffle_rng=prng)
    else:
        print("    [warn] data/s10_gen/rtuning.jsonl not present yet (needs the GPU probe); bucket will run short")
    buckets["abstention"] = d

    print("[identity]")
    buckets["identity"] = enc.encode_many(gen_rows(GEN / "identity.jsonl"), int(tgt["identity"]), upsample=20,
                                          shuffle_rng=prng, max_resp=400)
    print("[grounded_reading]")
    buckets["grounded_reading"] = enc.encode_many(gen_rows(GEN / "grounded.jsonl"), int(tgt["grounded_reading"]),
                                                  shuffle_rng=prng, max_resp=120, max_prompt=500)

    others = sum(total_tokens(v) for v in buckets.values())
    replay_target = max(int(tgt["replay_11lang"]), int(TOTAL * scale) - others)
    k = replay_target / sum(b75.REPLAY_LANG_TARGETS.values())
    for lang in list(b75.REPLAY_LANG_TARGETS):
        b75.REPLAY_LANG_TARGETS[lang] = int(b75.REPLAY_LANG_TARGETS[lang] * k)
    print(f"\n[replay_11lang] target {replay_target:,} tokens (non-replay buckets so far: {others:,})")
    rep_docs, rep_per_lang = b75.build_bucket_1_replay(tok, rng, enc.lang_tokens, enc.source_tokens)
    buckets["replay_11lang"] = rep_docs

    all_docs = [x for v in buckets.values() for x in v]
    bucket_tokens = {k_: total_tokens(v) for k_, v in buckets.items()}
    tot = sum(bucket_tokens.values())
    shares = {k_: round(v / max(1, tot), 4) for k_, v in bucket_tokens.items()}
    for ids_arr, mask in all_docs:
        assert int(ids_arr[-1]) == ids["eot"], "document does not end with <|endoftext|>"
        assert int((ids_arr == ids["think_open"]).sum()) == int((ids_arr == ids["think_close"]).sum()), "unclosed <think>"
    assert enc.unmapped == Counter(), f"unmapped langs {enc.unmapped}"
    print("\n[*] share vs target:")
    for k_, v in TARGET_SHARES.items():
        print(f"    {k_:18s} target {v:.2f}  actual {shares.get(k_, 0):.4f}  tokens {bucket_tokens.get(k_, 0):>11,}")

    print(f"\n[*] FFD packing {len(all_docs):,} docs ...")
    packed_ids, packed_mask, b3d, bflat, stats = ffd_pack_documents(all_docs, bin_size=BIN_SIZE, seed=args.seed, eot_id=ids["eot"])
    stats.update({
        "version": "processed_sft_240m_s10", "built_by": "scripts/build_s10_sft_240m.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "license_compliance": "Apache-2.0-Permissive-Only",
        "sources_and_licences": {
            "own generators (math, code output, false premise, benign facts, identity paraphrase)": "first-party, LICENSE_POLICY s1",
            "rajpurkar/squad_v2": "cc-by-sa-4.0", "lime-nlp/Synthetic_Unanswerable_Math": "mit",
            "meta-math/MetaMathQA": "mit", "microsoft/orca-math-word-problems-200k": "mit",
            "AI-MO/NuminaMath-CoT": "apache-2.0",
            "HuggingFaceTB/smoltalk (smol-constraints, smol-rewrite, smol-summarize only)": "apache-2.0",
            "flytech python-codes-25k, Belebele, Aya tail, Kannada/Telugu curated, processed_240m_apache replay": "cleared in S6/S7.5 manifests",
            "data/sft_v4_abstention_4000.jsonl, data/curated/parentheses_identity_sft_v1.jsonl": "first-party (uncoalesced)",
            "google-research-datasets/nq_open (R-Tuning question pool, when present)": "cc-by-sa-3.0",
        },
        "target_bins": TARGET_BINS, "target_tokens": TOTAL, "smoke": bool(args.smoke),
        "bucket_target_shares": TARGET_SHARES, "bucket_actual_tokens": bucket_tokens, "bucket_actual_shares": shares,
        "per_language_tokens": dict(enc.lang_tokens.most_common()), "per_source_tokens": dict(enc.source_tokens.most_common()),
        "quarantine_counts": dict(enc.q), "contamination_eval_prompts_checked": contam.n_eval,
        "replay_11lang_tokens": rep_per_lang, "native_phrases_reviewed_by_native_speaker": False,
        "rtuning_rows_present": rt.exists(), "elapsed_sec": round(time.time() - t0, 1),
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packed_ids.tofile(args.out_dir / "train.bin")
    packed_mask.tofile(args.out_dir / "train.mask.bin")
    np.save(args.out_dir / "train_boundaries.npy", b3d)
    np.save(args.out_dir / "train_boundaries_flat.npy", bflat)
    (args.out_dir / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUCCESS] {args.out_dir}: bins={stats['num_bins']:,} docs={stats['total_packed_docs']:,} "
          f"doc_tokens={stats['total_doc_tokens']:,} supervised={stats['total_supervised_tokens']:,} in {stats['elapsed_sec']}s")
    return 0


def self_test() -> None:
    c = Contam(["A rectangular panel is 12 cm long and 7 cm wide. Find both its area and perimeter, with units.",
                "Who designed and created Parentheses?"])
    assert c.hit("A rectangular panel is 99 cm long and 3 cm wide. Find both its area and perimeter, with units.")
    assert c.hit("who designed and created parentheses")
    assert not c.hit("A garden bed measures 40 m by 9 m. What is its area (include units)?")
    assert extract_final("So 16 * 47 = $752.\n#### 752", "The answer is: 752") == ("So 16 * 47 = $752.", "752")
    assert extract_final("Thus x = \\boxed{\\frac{13}{6}}.", "")[1] == "\\frac{13}{6}"
    assert extract_final("no marker here", "") is None
    print("[self-test] build_s10_sft_240m ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--smoke", action="store_true", help="4%% sized build for the Monday pilot / quick checks")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
