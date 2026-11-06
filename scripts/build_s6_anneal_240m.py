#!/usr/bin/env python3
# Engineered by uncoalesced
"""
Session 6 (240M) anneal corpus builder: the primary half of the S6 mix (Briefs/BRIEF_SESSION_6_240M_DRAFT.md).
processed_v5 is the 8/16 secondary and is not touched here. The 8 primary slots split 3:3:2:

    stem      : S4 STEM chat docs, sampled from data/processed_s4_stem_240m/train.bin (already tokenized)
                <|user|> <eng> prompt <|assistant|> <think>..</think> answer <|endoftext|>
    chat      : Corrected Path B:
                - Dravidian chat (~3.5M tokens): Aya kan/tam/tel/mal from data/processed_v5/sft_tail.bin
                - Abstention (~1.6M tokens): Multilingual epistemic abstention traces from data/sft_v4_abstention_4000.jsonl
                - SmolTalk v2 (~33.6M tokens): curated smoltalk_v2 (F:), capped smol-summarize <= 30%
                Total chat = 38.7M tokens (37.5% share).
    textbook  : S5 textbook docs from data/processed_s5_textbooks_240m/train.bin, Cosmopedia:FineWeb 2:1
                <eng> text <|endoftext|>

Chat filters: echoed replies, short replies, stray template tokens, probe-prompt contamination,
duplicate prompts, IndicQA eval benchmark removal, and probe entity contamination pruning.
Then a shuffled train.bin / val.bin via build_s4_stem_240m.assemble.
    .venv\\Scripts\\python.exe scripts\\build_s6_anneal_240m.py --out data\\processed_s6_anneal_240m
    .venv\\Scripts\\python.exe scripts\\build_s6_anneal_240m.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.build_s4_stem_240m import WORD, assemble, probe_keys  # noqa: E402

STEM_DIR = REPO_ROOT / "data" / "processed_s4_stem_240m"
TEXT_DIR = REPO_ROOT / "data" / "processed_s5_textbooks_240m"
V5_DIR = REPO_ROOT / "data" / "processed_v5"
SFT_APACHE_DIR = REPO_ROOT / "data" / "processed_sft_240m_v4_apache"
ABSTENTION_FILE = REPO_ROOT / "data" / "sft_v4_abstention_4000.jsonl"
CHAT_FILE = Path(r"F:\Lingual_Data\raw\hf_permissive_stem_code\curated\smoltalk_v2\curated_smoltalk_v2.jsonl")

# 3,000 steps x 16 micro x 4,096 tokens = 196.6M; primary half 98.3M split 3:3:2, +5% margin
CAPS = {"stem": 38_700_000, "chat": 38_700_000, "textbook": 25_800_000}
CAPS_CHAT = {"dravidian": 3_500_000, "abstention": 1_600_000, "smoltalk": 33_600_000}
TEMPLATE_STRINGS = ("<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>")
INDIC_QA_MARKERS = ("ಪ್ರಶ್ನೆ:", "கேள்வி:", "ప్రశ్న:", "ചോദ്യം:")
PROBE_ENTITIES = (
    "velanthor", "hyper-lattice", "vibranium", "alistair finch", "cryo-hepatic",
    "napoleon", "2036", "rahul sharma", "madhya pradesh", "snow tiger", "submarine",
    "ವೇಲಾಂಥೂರ್", "ವೇಲಾಂತೂರ್", "వేలాంతూర్", "വേലാന്തൂർ",
)
MIN_REPLY = 20


def is_echo(prompt: str, response: str) -> bool:
    p, a = prompt.strip(), response.strip()
    if a == p:
        return True
    if len(a) > 40 and a in p:
        return True
    return len(p) > 40 and p in a and len(a) < 1.2 * len(p)


def chat_keep(rec: dict, keys: list[str], seen: set, stats: dict) -> bool:
    p, a = rec.get("prompt", "").strip(), rec.get("response", "").strip()
    if not p or len(a) < MIN_REPLY:
        stats["short"] += 1
        return False
    if is_echo(p, a):
        stats["echo"] += 1
        return False
    if any(t in p or t in a for t in TEMPLATE_STRINGS):
        stats["stray_template"] += 1
        return False
    norm = " " + " ".join(WORD.findall(p.lower())) + " "
    if any(k in norm for k in keys):
        stats["contaminated"] += 1
        return False
    h = hashlib.sha1(norm.encode("utf-8")).digest()
    if h in seen:
        stats["dup_prompt"] += 1
        return False
    seen.add(h)
    return True


def docs_of(d: Path, split: str = "train"):
    a = np.memmap(d / f"{split}.bin", dtype=np.uint16, mode="r")
    s = np.load(d / f"{split}_boundaries.npy").astype(np.int64)
    return a, s, np.append(s[1:], len(a))


def write_src(out_dir: Path, name: str, pieces: list[np.ndarray]) -> int:
    lens = np.array([len(p) for p in pieces], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else np.zeros(0, np.int64)
    (np.concatenate(pieces) if pieces else np.zeros(0, np.uint16)).astype(np.uint16).tofile(out_dir / f"_src_{name}.bin")
    np.save(out_dir / f"_src_{name}_starts.npy", starts)
    return int(lens.sum())


def sample_docs(arr, s, e, order, cap: int) -> list[np.ndarray]:
    out, tot = [], 0
    for j in order:
        if tot >= cap:
            break
        d = np.asarray(arr[s[j]:e[j]])
        out.append(d)
        tot += len(d)
    return out


def build_stem(out_dir: Path, cap: int, rng) -> dict:
    a, s, e = docs_of(STEM_DIR)
    pieces = sample_docs(a, s, e, rng.permutation(len(s)), cap)
    return dict(source=str(STEM_DIR / "train.bin"), docs_available=int(len(s)), kept=len(pieces),
                tokens=write_src(out_dir, "stem", pieces), license="Apache-2.0 / MIT (see S4 manifest)")


def textbook_sources(n_docs: int) -> np.ndarray:
    """Source index (0 cosmopedia_v2, 1 fineweb_edu) of each S5 train doc, by replaying its assemble() shuffle."""
    m = json.loads((TEXT_DIR / "manifest.json").read_text(encoding="utf-8"))
    names = list(m["sources"])
    kept = [m["sources"][n]["kept"] for n in names]
    rng = np.random.default_rng(5001)
    d = np.concatenate([np.stack([np.full(k, i), np.arange(k)], 1) for i, k in enumerate(kept)])
    rng.shuffle(d)
    n_val = max(1, int(len(d) * 0.005))
    src = d[n_val:, 0]
    assert len(src) == n_docs, f"replayed {len(src)} train docs != {n_docs}"
    assert names == ["cosmopedia_v2", "fineweb_edu"], names
    return src


def build_textbook(out_dir: Path, cap: int, rng) -> dict:
    a, s, e = docs_of(TEXT_DIR)
    src = textbook_sources(len(s))
    cos, fw = rng.permutation(np.nonzero(src == 0)[0]), rng.permutation(np.nonzero(src == 1)[0])
    pieces = sample_docs(a, s, e, cos, cap * 2 // 3)
    n_cos = len(pieces)
    pieces += sample_docs(a, s, e, fw, cap - sum(len(p) for p in pieces))
    return dict(source=str(TEXT_DIR / "train.bin"), kept=len(pieces), cosmopedia_docs=n_cos,
                fineweb_docs=len(pieces) - n_cos, tokens=write_src(out_dir, "textbook", pieces),
                license="Cosmopedia-v2 Apache-2.0 rows / ODC-By card; FineWeb-Edu ODC-By")


def build_chat_dravidian(out_dir: Path, cap: int, rng, tok, ids: dict, keys: list[str]) -> dict:
    """Aya kan/tam/tel/mal chat docs from data/processed_v5/sft_tail.bin without IndicQA eval docs."""
    bin_path = V5_DIR / "sft_tail.bin"
    bounds_path = V5_DIR / "sft_tail_boundaries.npy"
    if not bin_path.exists() or not bounds_path.exists():
        raise FileNotFoundError(f"Missing Dravidian chat sources at {V5_DIR}")

    b = np.memmap(bin_path, dtype=np.uint16, mode="r")
    bounds = np.load(bounds_path)
    drav_ids = {ids["kan"]: "kan", ids["tam"]: "tam", ids["tel"]: "tel", ids["mal"]: "mal"}

    seen = set()
    cand = []
    stats = dict(total_tail_docs=len(bounds), drav_candidates=0, indic_qa_dropped=0, dup_dropped=0, kept=0)

    for i in range(len(bounds)):
        s = bounds[i]
        e = bounds[i + 1] if i + 1 < len(bounds) else len(b)
        doc = b[s:e]
        if len(doc) <= 2 or doc[0] != ids["user"] or doc[1] not in drav_ids:
            continue
        stats["drav_candidates"] += 1
        text = tok.decode(doc)
        if any(marker in text for marker in INDIC_QA_MARKERS):
            stats["indic_qa_dropped"] += 1
            continue
        norm = " " + " ".join(WORD.findall(text.lower())) + " "
        if any(k in norm for k in keys):
            continue
        h = hashlib.sha256(doc.tobytes()).digest()
        if h in seen:
            stats["dup_dropped"] += 1
            continue
        seen.add(h)
        cand.append(np.asarray(doc, dtype=np.uint16))

    # Permute deterministically and sample up to cap
    order = rng.permutation(len(cand))
    pieces, total, per_lang = [], 0, {}
    for idx in order:
        if total >= cap:
            break
        d = cand[idx]
        pieces.append(d)
        total += len(d)
        lang_name = drav_ids[d[1]]
        per_lang[lang_name] = per_lang.get(lang_name, 0) + len(d)

    stats["kept"] = len(pieces)
    stats["tokens"] = write_src(out_dir, "chat_dravidian", pieces)
    stats["tokens_by_lang"] = per_lang
    stats["source"] = str(bin_path)
    stats["license"] = "Aya Dataset (Apache-2.0, permissive only)"
    return stats


def build_chat_abstention(out_dir: Path, cap: int, rng, tok, ids: dict, keys: list[str]) -> dict:
    """Multilingual epistemic abstention traces, filtered against eval entity contamination."""
    # Check 2D boundaries handle for processed_sft_240m_v4_apache if present
    apache_dir = SFT_APACHE_DIR
    if (apache_dir / "train_boundaries.npy").exists():
        bounds_2d = np.load(apache_dir / "train_boundaries.npy")
        assert bounds_2d.ndim == 2 and bounds_2d.shape[1] == 3, f"Expected 2-D boundaries [bin, start, end], got {bounds_2d.shape}"

    if not ABSTENTION_FILE.exists():
        raise FileNotFoundError(f"Missing abstention jsonl at {ABSTENTION_FILE}")

    cand = []
    stats = dict(raw_traces=0, probe_entity_dropped=0, probe_key_dropped=0, kept_unique=0)
    with open(ABSTENTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["raw_traces"] += 1
            rec = json.loads(line)
            content = (rec.get("prompt", "") + " " + rec.get("response", "")).lower()
            if any(term in content for term in PROBE_ENTITIES):
                stats["probe_entity_dropped"] += 1
                continue
            if any(k in content for k in keys):
                stats["probe_key_dropped"] += 1
                continue
            cand.append((rec["lang"], rec["prompt"].strip(), rec["response"].strip()))

    stats["kept_unique"] = len(cand)

    # Replicate cleanly to reach cap ~1.6M tokens
    pieces, total, per_lang = [], 0, {}
    # Shuffle order per pass for diversity
    for pass_idx in range(4):
        order = rng.permutation(len(cand))
        for idx in order:
            if total >= cap:
                break
            lang, p, r = cand[idx]
            lang_id = ids.get(lang, ids["eng"])
            p_ids = tok.encode(p, add_special_tokens=False).ids
            r_ids = tok.encode(r, add_special_tokens=False).ids
            doc = np.asarray([ids["user"], lang_id, *p_ids, ids["assistant"], *r_ids, ids["eot"]], dtype=np.uint16)
            pieces.append(doc)
            total += len(doc)
            per_lang[lang] = per_lang.get(lang, 0) + len(doc)
        if total >= cap:
            break

    stats["kept"] = len(pieces)
    stats["tokens"] = write_src(out_dir, "chat_abstention", pieces)
    stats["tokens_by_lang"] = per_lang
    stats["source"] = str(ABSTENTION_FILE)
    stats["license"] = "Apache-2.0 (uncoalesced_abstention_traces)"
    return stats


def build_chat_smoltalk(out_dir: Path, cap: int, max_summarize_tokens: int, tok, ids: dict, keys: list[str], seed: int) -> dict:
    """SmolTalk v2 chat docs with smol-summarize capped at <= 30% of total chat tokens."""
    stats = dict(rows=0, short=0, echo=0, stray_template=0, contaminated=0, dup_prompt=0, kept=0)
    cand: list[tuple[str, str, str]] = []
    seen: set = set()
    with open(CHAT_FILE, encoding="utf-8") as fh:  # sequential read of F:
        for line in fh:
            stats["rows"] += 1
            rec = json.loads(line)
            if chat_keep(rec, keys, seen, stats):
                cand.append((rec["subset"], rec["prompt"].strip(), rec["response"].strip()))

    random.Random(seed).shuffle(cand)
    pieces, total, per_subset = [], 0, {}
    for i in range(0, len(cand), 2048):
        chunk = cand[i:i + 2048]
        pe = tok.encode_batch([p for _, p, _ in chunk], add_special_tokens=False)
        re_ = tok.encode_batch([r for _, _, r in chunk], add_special_tokens=False)
        for (sub, _, _), p, r in zip(chunk, pe, re_):
            if total >= cap:
                break
            doc_len = 1 + 1 + len(p.ids) + 1 + len(r.ids) + 1
            if sub == "smol-summarize" and per_subset.get(sub, 0) + doc_len > max_summarize_tokens:
                continue
            doc = np.asarray([ids["user"], ids["eng"], *p.ids, ids["assistant"], *r.ids, ids["eot"]], dtype=np.uint16)
            pieces.append(doc)
            total += len(doc)
            per_subset[sub] = per_subset.get(sub, 0) + len(doc)
        if total >= cap:
            break

    stats["kept"] = len(pieces)
    stats["candidates"] = len(cand)
    stats["tokens_by_subset"] = per_subset
    stats["tokens"] = write_src(out_dir, "chat_smoltalk", pieces)
    stats["cap_reached"] = total >= cap
    stats["source"] = str(CHAT_FILE)
    stats["license"] = "per smoltalk_v2/SUBSETS.md: Apache-2.0, CC-BY-4.0 (apigen-80k), ODC-By (self-oss-instruct)"
    return stats


def verify(out_dir: Path, ids: dict) -> dict:
    res = {}
    valid_chat_langs = {ids["eng"], ids["kan"], ids["tam"], ids["tel"], ids["mal"]}
    id_to_lang = {ids["eng"]: "eng", ids["kan"]: "kan", ids["tam"]: "tam", ids["tel"]: "tel", ids["mal"]: "mal"}

    for split in ("train", "val"):
        a, s, e = docs_of(out_dir, split)
        bad, kinds = 0, {"chat": 0, "text": 0}
        chat_tags_by_lang = {}
        for i, j in zip(s, e):
            d = np.asarray(a[i:j])
            if len(d) > 2 and d[0] == ids["user"]:
                lang_tok = int(d[1])
                ok = (lang_tok in valid_chat_langs and d[-1] == ids["eot"] and int((d == ids["eot"]).sum()) == 1
                      and int((d == ids["assistant"]).sum()) == 1 and int((d == ids["user"]).sum()) == 1
                      and int((d == ids["think_open"]).sum()) == int((d == ids["think_close"]).sum()))
                kinds["chat"] += 1
                lname = id_to_lang.get(lang_tok, str(lang_tok))
                chat_tags_by_lang[lname] = chat_tags_by_lang.get(lname, 0) + 1
            else:
                ok = (len(d) > 2 and d[0] == ids["eng"] and d[-1] == ids["eot"] and int((d == ids["eot"]).sum()) == 1
                      and int((d == ids["user"]).sum()) == 0 and int((d == ids["assistant"]).sum()) == 0)
                kinds["text"] += 1
            bad += not ok
        res[split] = dict(docs=int(len(s)), tokens=int(len(a)), malformed=int(bad), chat_tags_by_lang=chat_tags_by_lang, **kinds)
        assert bad == 0, f"{split}: {bad} malformed documents"
    return res


def special_ids(tok) -> dict:
    ids = {k: tok.token_to_id(v) for k, v in dict(user="<|user|>", assistant="<|assistant|>", eot="<|endoftext|>",
                                                  eng="<eng>", kan="<kan>", tam="<tam>", tel="<tel>", mal="<mal>",
                                                  think_open="<think>", think_close="</think>").items()}
    assert all(v is not None for v in ids.values()), ids
    return ids


def self_test() -> None:
    assert is_echo("hello there", "hello there")
    assert is_echo("x" * 50 + " tail", "x" * 50)
    assert not is_echo("What is 2+2?", "The answer is 4, because two plus two is four.")
    st = dict(short=0, echo=0, stray_template=0, contaminated=0, dup_prompt=0)
    keys, seen = [" what is 17 multiplied by 6 "], set()
    ok = {"prompt": "Summarise the text.", "response": "A short but complete summary of it."}
    assert chat_keep(ok, keys, seen, st)
    assert not chat_keep(ok, keys, seen, st) and st["dup_prompt"] == 1
    assert not chat_keep({"prompt": "Hi?", "response": "ok"}, keys, seen, st) and st["short"] == 1
    assert not chat_keep({"prompt": "Q", "response": "fine <|endoftext|> more text here"}, keys, seen, st)
    assert st["stray_template"] == 1
    assert not chat_keep({"prompt": "Question: what is 17 multiplied by 6?", "response": "It is one hundred two."},
                         keys, seen, st) and st["contaminated"] == 1
    a = np.arange(10, dtype=np.uint16)
    assert [len(x) for x in sample_docs(a, np.array([0, 3, 7]), np.array([3, 7, 10]), [2, 0, 1], 5)] == [3, 3]

    # Verify Dravidian and Abstention markers
    assert any(m in "ಇಲ್ಲಿ ಪ್ರಶ್ನೆ: ಉತ್ತರ ನೀಡಿ" for m in INDIC_QA_MARKERS)
    assert any(term in "the velanthor maritime accord of 1919" for term in PROBE_ENTITIES)
    print("build_s6_anneal_240m self-test OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "processed_s6_anneal_240m")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json")
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=6001)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    ids = special_ids(tok)
    a.out.mkdir(parents=True, exist_ok=True)
    rng, t0 = np.random.default_rng(a.seed), time.time()
    pkeys = probe_keys()

    per = {}
    print("[1/5] Building STEM...", flush=True)
    per["stem"] = build_stem(a.out, CAPS["stem"], rng)
    print(f"      kept={per['stem']['kept']}, tokens={per['stem']['tokens']:,}", flush=True)

    print("[2/5] Building Dravidian Chat (Aya)...", flush=True)
    per["chat_dravidian"] = build_chat_dravidian(a.out, CAPS_CHAT["dravidian"], rng, tok, ids, pkeys)
    print(f"      kept={per['chat_dravidian']['kept']}, tokens={per['chat_dravidian']['tokens']:,}, by_lang={per['chat_dravidian']['tokens_by_lang']}", flush=True)

    print("[3/5] Building Abstention Traces...", flush=True)
    per["chat_abstention"] = build_chat_abstention(a.out, CAPS_CHAT["abstention"], rng, tok, ids, pkeys)
    print(f"      kept={per['chat_abstention']['kept']}, tokens={per['chat_abstention']['tokens']:,}, by_lang={per['chat_abstention']['tokens_by_lang']}", flush=True)

    # Remaining smoltalk budget
    smol_budget = CAPS["chat"] - per["chat_dravidian"]["tokens"] - per["chat_abstention"]["tokens"]
    max_summarize = int(0.30 * CAPS["chat"])
    print(f"[4/5] Building SmolTalk v2 (budget: {smol_budget:,}, max summarize: {max_summarize:,})...", flush=True)
    per["chat_smoltalk"] = build_chat_smoltalk(a.out, smol_budget, max_summarize, tok, ids, pkeys, a.seed)
    print(f"      kept={per['chat_smoltalk']['kept']}, tokens={per['chat_smoltalk']['tokens']:,}, subsets={per['chat_smoltalk']['tokens_by_subset']}", flush=True)

    print("[5/5] Building Textbooks...", flush=True)
    per["textbook"] = build_textbook(a.out, CAPS["textbook"], rng)
    print(f"      kept={per['textbook']['kept']}, tokens={per['textbook']['tokens']:,}", flush=True)

    names = ["stem", "chat_smoltalk", "chat_dravidian", "chat_abstention", "textbook"]
    print("Assembling splits...", flush=True)
    splits = assemble(a.out, names, a.val_frac, a.seed)
    print("Verifying documents...", flush=True)
    checks = verify(a.out, ids)

    total = sum(per[n]["tokens"] for n in names)
    total_chat = per["chat_smoltalk"]["tokens"] + per["chat_dravidian"]["tokens"] + per["chat_abstention"]["tokens"]

    shares = {
        "stem": round(per["stem"]["tokens"] / total, 4),
        "chat": round(total_chat / total, 4),
        "textbook": round(per["textbook"]["tokens"] / total, 4),
    }
    sub_shares = {n: round(per[n]["tokens"] / total, 4) for n in names}

    manifest = {
        "version": "processed_s6_anneal_240m",
        "license_compliance": "Apache-2.0-Permissive-Only",
        "built_by": "scripts/build_s6_anneal_240m.py",
        "brief": "Briefs/BRIEF_SESSION_6_240M_DRAFT.md (approved 2026-11-04)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "formats": {
            "stem/smoltalk": "<|user|> <eng> prompt <|assistant|> body <|endoftext|>",
            "dravidian": "<|user|> <kan|tam|tel|mal> prompt <|assistant|> body <|endoftext|>",
            "abstention": "<|user|> <lang> prompt <|assistant|> <think>..</think> body <|endoftext|>",
            "textbook": "<eng> text <|endoftext|>",
        },
        "tokenizer": str(a.tokenizer),
        "tokenizer_sha256": hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
        "caps": CAPS,
        "caps_chat": CAPS_CHAT,
        "sources": per,
        "shares": shares,
        "sub_shares": sub_shares,
        "splits": splits,
        "verify": checks,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for n in names:
        (a.out / f"_src_{n}.bin").unlink()
        (a.out / f"_src_{n}_starts.npy").unlink()

    print(json.dumps({k: manifest[k] for k in ("shares", "sub_shares", "splits", "verify")}, indent=2, ensure_ascii=False))
    print(f"[OK] Finished building S6 anneal corpus in {manifest['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
