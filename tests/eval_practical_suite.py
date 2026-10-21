#!/usr/bin/env python3
"""
Practical open-generation eval for Parentheses 0.9-120M (Claude Code, 2026-10-18).

Earlier evals scored multiple-choice logits, which hid total failure in open generation.
Every score here comes from text the model actually generates (or, for bpb, from the
likelihood of a held-out reference), on the model's own domain:

  1. Dravidian continuation: Kannada, Tamil, Telugu, Malayalam (+ Tulu, Koya, Kuvi, SE Kolami)
     prompts cut from held-out val docs (data/processed_v4/val). Every candidate prompt+reference
     window is checked against the training token streams on disk for any shared 32-token span
     (near-dup, not just exact hash) and dropped if one is found.
     Scores: script fidelity of the output, repetition, and bits per UTF-8 byte of the reference
     continuation given the prompt.
  2. Vocabulary / translation pairs, scored by contains-match on the generated answer:
       - Tulu: tcy.wiktionary dump (raw\\wikimedia\\tcy_wiktionary, never parsed into training).
       - Kannada: Alar; Malayalam: Olam. Both WERE in the S4 dictionary mix and SFT, so these
         two measure recall of seen data, not generalisation.
       - Koya / Kuvi: no dictionary source exists in F:\\Lingual_Data (only eBible scripture), so
         they are covered by section 1 only.
  3. English baseline: the stable-answer questions from Joel's vault test-01. The headline mode
     ("product") calls `chat.reply()` with `DEFAULT_SYSTEM_PROMPT` at chat.py defaults, so
     deterministic math/logic intercepts, RAG grounding, and instruction post-processing are
     active. Raw model-only greedy and sampled generations are kept alongside for comparison.
     Scored by exact match and contains match (never multiple-choice logits).

Usage (repo root):
  .venv\\Scripts\\python.exe tests\\eval_practical_suite.py --self-test
  .venv\\Scripts\\python.exe tests\\eval_practical_suite.py build            # prompts + near-dup scan (CPU)
  .venv\\Scripts\\python.exe tests\\eval_practical_suite.py run --device cuda # both checkpoints (GPU, takes GPU.lock)
  .venv\\Scripts\\python.exe tests\\eval_practical_suite.py report           # markdown into the vault Reports folder
  .venv\\Scripts\\python.exe tests\\eval_practical_suite.py all --device cuda
"""

from __future__ import annotations

import argparse
import bz2
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DIR = REPO_ROOT / "out" / "practical_eval"
PROMPTS_PATH = OUT_DIR / "prompts.json"
VAL_DIR = REPO_ROOT / "data" / "processed_v4" / "val"
TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
VAULT_REPORTS = Path(r"C:\Users\Joel Anthony\Documents\Phile\Heafod\Heafod\Parentheses\Reports")

DICT_DIR = Path(r"F:\Lingual_Data\raw\dictionaries")
TCY_WIKTIONARY = Path(r"F:\Lingual_Data\raw\wikimedia\tcy_wiktionary\tcywiktionary-latest-pages-articles.xml.bz2")

# Training token streams still on disk (uint16, v3_32k ids). SFT v3/v4 data dirs are gone from
# disk, so their rows can't be scanned; the report says so.
TRAIN_BINS = [
    "data/processed_v5/train.bin",
    "data/processed_v5/sft_tail.bin",
    "data/processed_qa_mix_s4/train.bin",
    "data/processed_sft_120m_v2/train.bin",
]

CHECKPOINTS = {
    "sft_v12": "checkpoints/parentheses-0.9-120M-sft-v12/step_9835_final.pt",
    "sft_v4": "checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt",
    "sft_v2": "checkpoints/parentheses-0.9-120M-sft-v2/step_9786_final.pt",
}

LANG_NAMES = {"kan": "Kannada", "tam": "Tamil", "tel": "Telugu", "mal": "Malayalam",
              "tcy": "Tulu", "kff": "Koya", "kxv": "Kuvi", "nit": "SE Kolami", "eng": "English"}
MAIN_LANGS = ("kan", "tam", "tel", "mal")
LOWRES_LANGS = ("tcy", "kff", "kxv", "nit")
SCRIPT_RANGES = {
    "kan": (0x0C80, 0x0CFF), "tcy": (0x0C80, 0x0CFF),   # Tulu is written in Kannada script
    "tam": (0x0B80, 0x0BFF),
    "tel": (0x0C00, 0x0C7F), "nit": (0x0C00, 0x0C7F),   # SE Kolami: Telugu script
    "mal": (0x0D00, 0x0D7F),
    "kff": (0x0B00, 0x0B7F), "kxv": (0x0B00, 0x0B7F),   # Koya / Kuvi eBible: Odia script
}

PROMPT_TOKENS = 64      # prompt length (moved forward to the next word start)
REF_TOKENS = 96         # reference continuation length (also word-aligned)
N_MAIN, N_LOWRES = 20, 12
N_VOCAB = 25
W = 32                  # near-dup span length in tokens (same as out/audit_v5_data.py)

CONT_NEW_TOKENS = 128
SAMPLING = dict(temperature=0.7, top_k=40, min_p=0.06, repetition_penalty=1.12, no_repeat_ngram_size=4)  # chat.py defaults
GREEDY = dict(temperature=1e-5, top_k=1, min_p=None, repetition_penalty=1.12, no_repeat_ngram_size=4)

# ---------------------------------------------------------------------------
# English baseline: test-01 questions with a stable gold answer.
# Left out: anything needing live web data (Python/Ubuntu/Node/Git versions, USD-INR rate),
# questions about an unnamed "project", and the open-ended RAM-vs-storage essay.
# Gold entries are regexes matched case-insensitively unless prefixed with "(?-i)".
# ---------------------------------------------------------------------------
ENGLISH_QA = [
    ("disc", "If a product costs ₹1,500 and receives a 20% discount followed by a 10% discount, what is its final price?", [r"1,?080"]),
    ("seq", "What is the next number in this sequence: 3, 6, 11, 18, 27, and what rule did you use?", [r"\b38\b"]),
    ("roses", "If all roses are flowers and some flowers fade quickly, can you conclude that some roses fade quickly? Explain your reasoning.", [r"^\W*no\b", r"\bcannot\b", r"\bcan't\b", r"\bcan not\b", r"\bnot necessarily\b"]),
    ("nvidia", "Who is NVIDIA's CEO? Answer with the name only.", [r"jensen", r"huang"]),
    ("mul", "What is 17 × 6? Answer with the number only.", [r"\b102\b"]),
    ("http429", "What does HTTP 429 mean? Answer in two words.", [r"too many requests"]),
    ("sheep", "If all but 4 sheep run away from 12 sheep, how many remain?", [r"\b4\b", r"\bfour\b"]),
    ("aus", "What is the capital of Australia? Answer with one word.", [r"canberra"]),
    ("larger", "Which is larger: 0.8 or 0.75?", [r"0\.8\b(?!\d)"]),
    ("day", "If yesterday was two days before Sunday, what day is today?", [r"saturday"]),
    ("prime", "What is the smallest prime number?", [r"\b2\b", r"\btwo\b"]),
    ("linux", "Who created Linux? Answer with the person's name.", [r"linus", r"torvalds"]),
    ("potassium", "What is the chemical symbol for potassium? Answer with one symbol only.", [r"(?-i)\bK\b"]),
    ("div", "What is 144 divided by 12? Answer with the number only.", [r"\b12\b(?![.,]\d)", r"\btwelve\b"]),
    ("rings", "Which planet has the most prominent ring system? Answer with one name.", [r"saturn"]),
    ("yen", "What is the currency of Japan? Answer with one word.", [r"\byen\b"]),
    ("sq", "What is 7 squared? Answer with the number only.", [r"\b49\b"]),
    ("kg", "Which is heavier: 1 kg of iron or 1 kg of cotton? Answer briefly.", [r"\bsame\b", r"\bneither\b", r"\bequal", r"\bboth\b"]),
    ("continent", "What is the smallest continent by land area? Answer with one name.", [r"australia", r"oceania"]),
    ("mona", "Who painted the Mona Lisa? Answer with the person's name only.", [r"leonardo", r"da vinci"]),
    ("msft", "What is the current CEO of Microsoft? Verify using an official source and give the name only.", [r"satya", r"nadella"]),
    ("india_pres", "What is the current President of India? Verify using an official government source and give the name only.", [r"murmu", r"droupadi"]),
    ("salt", "What is the chemical formula for table salt? Answer with the formula only.", [r"(?-i)\bNaCl\b"]),
    ("race", "If you overtake the person in second place during a race, what position are you in? Answer with one word.", [r"\bsecond\b", r"\b2nd\b"]),
    ("minutes", "How many minutes are in 2.5 hours? Answer with the number only.", [r"\b150\b"]),
    ("ocean", "Which ocean lies between Africa and Australia? Answer with one name.", [r"\bindian\b"]),
    ("chf", "What is the official currency of Switzerland? Answer with one word.", [r"franc", r"\bCHF\b"]),
    ("pct", "What is 15% of 80? Answer with the number only.", [r"\b12\b(?![.,]\d)", r"\btwelve\b"]),
    ("gosling", "Which programming language was created by James Gosling? Answer with one name.", [r"\bjava\b"]),
    ("clock", "If a clock shows 3:15, what is the smaller angle between its hour and minute hands? Answer in degrees only.", [r"\b7\.5\b"]),
]

# Basic vocabulary (Swadesh-style) used to pick Alar / Olam pairs a 120M model could know.
COMMON_WORDS = """water fire sun moon star rain river sea tree leaf flower fruit seed root stone sand earth
mountain cloud wind smoke ash salt milk rice egg fish bird dog cow cat horse snake elephant tiger monkey
man woman child mother father brother sister king house village road door book money name day night year
head hair eye ear nose mouth tooth tongue hand foot leg knee heart blood bone skin neck belly
red black white green yellow big small hot cold new old good bad long sweet bitter
eat drink sleep see hear walk come give say die kill swim fly sit stand run cry laugh
one two three four five ten hundred""".split()


# ---------------------------------------------------------------------------
# Scoring helpers (pure functions; covered by --self-test)
# ---------------------------------------------------------------------------

def script_fidelity(text: str, lang: str) -> float:
    """Share of letters/marks (Unicode L*/M*) that fall in the language's script block."""
    lo, hi = SCRIPT_RANGES[lang]
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def repetition_rate(text: str, n: int = 16) -> float:
    """Duplicate character n-gram rate (same as scripts/eval_dravidian.py): 0 = no repeats, ~1 = a loop."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def distinct_n(text: str, n: int = 2) -> float:
    words = re.findall(r"\w+", text)
    if len(words) < n:
        return 1.0 if words else 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def is_looping(tokens: list[int], window: int = 16, repeats: int = 3) -> bool:
    need = window * repeats
    if len(tokens) < need:
        return False
    tail = tokens[-need:]
    return all(tail[i * window:(i + 1) * window] == tail[:window] for i in range(1, repeats))


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip().lower()
    s = re.sub(r"^[\s\"'`*_:\-]+|[\s\"'`*_.!?,;:\-]+$", "", s)
    return re.sub(r"\s+", " ", s)


def gold_match(text: str, patterns: list[str]) -> bool:
    for p in patterns:
        flags = 0 if p.startswith("(?-i)") else re.IGNORECASE
        if re.search(p.replace("(?-i)", ""), text, flags | re.MULTILINE):
            return True
    return False


def score_english(output: str, prompt: str, gold: list[str]) -> dict:
    """exact: the first non-empty line, normalised, is (or is just) the gold answer.
    contains: gold appears anywhere -- but if the gold string already appears in the prompt
    (e.g. "0.8 or 0.75", "all but 4 sheep") an echo would score, so contains falls back to exact."""
    first = next((ln for ln in output.splitlines() if ln.strip()), "")
    first_n = _norm(first)
    first_n = re.sub(r"^(the answer is|answer:?|final answer:?)\s*", "", first_n)
    first_n = re.sub(r"\\boxed\{([^}]*)\}", r"\1", first_n)
    exact = False
    for p in gold:
        flags = 0 if p.startswith("(?-i)") else re.IGNORECASE
        if p.startswith("^") and re.search(p, first_n, flags):  # anchored gold: answer leads the line ("No, ...")
            exact = True
            break
        pat = p.replace("(?-i)", "").lstrip("^").replace(r"\W*", "")
        if re.fullmatch(rf"(?:{pat})(?:\s*(?:degrees?|°|minutes|\.))?", first_n, flags) or \
           re.fullmatch(rf"(?:{pat})", first.strip().rstrip("."), flags):
            exact = True
            break
    # a short first line that contains the gold also counts as exact ("Jensen Huang", "Canberra.")
    if not exact and len(first_n.split()) <= 4 and gold_match(first_n, gold):
        exact = True
    echo_risk = gold_match(prompt, [g for g in gold if not g.startswith("^")])
    contains = gold_match(output, gold)
    if echo_risk and contains and not exact:
        # the gold already appears in the prompt, so an echo would score; accept
        # only if the generated text is short enough to be an answer, not a restatement
        contains = len(_norm(output).split()) <= 6
    return {"exact": exact, "contains": contains, "echo_risk": echo_risk}


def contains_word(text: str, words: list[str], latin: bool) -> bool:
    """Whole-word contains for Latin glosses; substring for Indic targets (agglutinative suffixes)."""
    t = unicodedata.normalize("NFC", text)
    for w in words:
        w = unicodedata.normalize("NFC", w).strip()
        if not w:
            continue
        if latin:
            if re.search(rf"\b{re.escape(w)}(s|es)?\b", t, re.IGNORECASE):
                return True
        elif w in t:
            return True
    return False


# ---------------------------------------------------------------------------
# Build step: prompts + near-dup filter
# ---------------------------------------------------------------------------

def span_hashes(arr: np.ndarray) -> np.ndarray:
    """uint64 polynomial hash of every W-token window (same hash as out/audit_v5_data.py)."""
    a = arr.astype(np.uint64)
    n = len(a) - W + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(W):
            h = h * np.uint64(1000003) + a[j:j + n]
    return h


def load_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    tok.encode_special_tokens = True
    return tok


def val_docs(lang: str) -> list[np.ndarray]:
    arr = np.fromfile(VAL_DIR / f"{lang}.bin", dtype=np.uint16)
    bnd = np.load(VAL_DIR / f"{lang}_boundaries.npy").astype(np.int64)
    ends = np.append(bnd[1:], len(arr))
    return [arr[s:e] for s, e in zip(bnd, ends)]


def word_start_after(ids: np.ndarray, pos: int, starts: np.ndarray) -> int | None:
    """First index >= pos whose token begins a new word (byte-level BPE 'Ġ' prefix)."""
    for i in range(pos, len(ids)):
        if starts[int(ids[i])]:
            return i
    return None


def make_continuation_candidates(tok, rng) -> list[dict]:
    starts = np.array([(tok.id_to_token(i) or "").startswith("Ġ") for i in range(tok.get_vocab_size())])
    cands = []
    for lang in (*MAIN_LANGS, *LOWRES_LANGS):
        docs = val_docs(lang)
        order = rng.permutation(len(docs))
        want = (N_MAIN if lang in MAIN_LANGS else N_LOWRES) * 4
        got = 0
        for di in order:
            d = docs[int(di)]
            body = d[1:]                      # drop the <lang> tag
            body = body[body != 0]            # drop <|endoftext|>
            p_end = word_start_after(body, PROMPT_TOKENS, starts)
            if p_end is None:
                continue
            r_end = word_start_after(body, p_end + REF_TOKENS, starts)
            if r_end is None:
                continue
            prompt_ids, ref_ids = body[:p_end], body[p_end:r_end]
            prompt_text, ref_text = tok.decode(prompt_ids.tolist()), tok.decode(ref_ids.tolist())
            if "\ufffd" in prompt_text + ref_text or script_fidelity(prompt_text + ref_text, lang) < 0.6:
                continue  # skip mostly-Latin/garbled docs: they don't test the target script
            cands.append({"id": f"{lang}#{int(di)}", "lang": lang, "doc": int(di),
                          "prompt_ids": prompt_ids.tolist(), "ref_ids": ref_ids.tolist(),
                          "prompt_text": prompt_text, "ref_text": ref_text})
            got += 1
            if got >= want:
                break
    return cands


def near_dup_scan(cands: list[dict]) -> dict:
    """Mark every candidate whose prompt+reference shares ANY 32-token span with a training stream."""
    hs, owner = [], []
    for k, c in enumerate(cands):
        h = span_hashes(np.array(c["prompt_ids"] + c["ref_ids"], dtype=np.uint16))
        hs.append(h)
        owner.append(np.full(len(h), k, dtype=np.int64))
    hs, owner = np.concatenate(hs), np.concatenate(owner)
    order = np.argsort(hs)
    hs, owner = hs[order], owner[order]
    hits = Counter()
    scanned = {}
    for rel in TRAIN_BINS:
        path = REPO_ROOT / rel
        if not path.exists():
            scanned[rel] = "missing"
            continue
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        t0 = time.time()
        matched = set()
        chunk = 50_000_000
        for s in range(0, len(arr), chunk):
            h = span_hashes(np.asarray(arr[s:s + chunk + W - 1]))
            i = np.searchsorted(hs, h)
            i[i == len(hs)] = 0
            matched.update(np.unique(h[hs[i] == h]).tolist())
        for v in matched:
            lo, hi = np.searchsorted(hs, np.uint64(v)), np.searchsorted(hs, np.uint64(v), side="right")
            for o in set(owner[lo:hi].tolist()):
                hits[o] += 1
        scanned[rel] = f"{len(arr):,} tokens, {len(matched)} matching spans, {time.time() - t0:.0f}s"
        print(f"[build] scanned {rel}: {scanned[rel]}", flush=True)
        del arr
    for k, c in enumerate(cands):
        c["train_span_hits"] = int(hits.get(k, 0))
    return scanned


def parse_alar(common: set[str]) -> dict[str, list[str]]:
    """English word -> Kannada words whose FIRST Alar definition is exactly that word.
    (Alar's `head` is the alphabet letter the entry is filed under; the word is the 2-space `entry`.)"""
    out: dict[str, list[str]] = {}
    head, first_def_seen = None, False
    with open(DICT_DIR / "alar.yml", encoding="utf-8") as f:
        for line in f:
            if line.startswith("- id:"):
                head, first_def_seen = None, False
            elif line.startswith("  entry:"):
                head = line.split(":", 1)[1].strip().strip("'\"")
            elif line.startswith("    entry:") and head and not first_def_seen:
                first_def_seen = True
                d = line.split(":", 1)[1].strip().strip("'\"").lower()
                d = re.split(r"[;,]", d)[0]
                d = re.sub(r"^(a|an|the|to)\s+", "", d).rstrip(" ;.,")
                if d in common and " " not in head and head:
                    out.setdefault(d, [])
                    if head not in out[d]:
                        out[d].append(head)
    return out


def parse_olam(common: set[str]) -> dict[str, list[str]]:
    """English word -> single-word Malayalam translations (Olam enml TSV)."""
    out: dict[str, list[str]] = {}
    with open(DICT_DIR / "olam" / "files" / "enml", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            en, ml = parts[0].strip().lower(), parts[2].strip()
            if en in common and ml and " " not in ml and script_fidelity(ml, "mal") > 0.99:
                out.setdefault(en, [])
                if ml not in out[en]:
                    out[en].append(ml)
    return out


def parse_kaikki(lang_name: str, common: set[str]) -> dict[str, set[str]]:
    """English word -> native words with a Kaikki (en.wiktionary) sense whose gloss is that word."""
    out: dict[str, set[str]] = {}
    with open(DICT_DIR / f"kaikki-{lang_name}.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            word = e.get("word", "")
            if " " in word or not word:
                continue
            for s in e.get("senses", []):
                for g in s.get("glosses", [])[:1]:
                    g = re.sub(r"\(.*?\)", "", g.lower())
                    for part in re.split(r"[;,]", g):
                        part = re.sub(r"^(a|an|the|to)\s+", "", part.strip()).rstrip(".")
                        if part in common:
                            out.setdefault(part, set()).add(word)
    return out


def _clean_wiki(s: str) -> str:
    s = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"'''?|&quot;|&amp;|<[^>]+>", "", s)
    return s.strip()


def parse_tcy_wiktionary() -> list[dict]:
    """Tulu headword -> first short English gloss (+ Kannada gloss) from the tcy.wiktionary dump."""
    x = bz2.open(TCY_WIKTIONARY).read().decode("utf-8")
    rows = []
    for page in re.findall(r"<page>(.*?)</page>", x, re.S):
        if "<ns>0</ns>" not in page:
            continue
        title = re.search(r"<title>(.*?)</title>", page).group(1).strip()
        m = re.search(r"<text[^>]*>(.*?)</text>", page, re.S)
        if not m or " " in title or script_fidelity(title, "tcy") < 0.99:
            continue
        text = m.group(1)
        en = re.search(r"^\*\s*(?:English|ಇಂಗ್ಲಿಸ್|ಇಂಗ್ಲೀಷ್|ಇಂಗ್ಲಿಷ್|ಇಂಗ್ಲೀಸ್)\s*:\s*(.+)$", text, re.M)
        kn = re.search(r"^\*\s*ಕನ್ನಡ\s*:\s*(.+)$", text, re.M)
        if not en:
            continue
        gloss = _clean_wiki(en.group(1))
        gloss = re.sub(r"^\(?\d+[.)]\s*", "", gloss)
        first = re.split(r"[;,]|\s\d+[.)]", gloss)[0].strip().rstrip(".").lower()
        first = re.sub(r"^(a|an|the|to)\s+", "", first)
        if not first or len(first.split()) > 2 or not re.fullmatch(r"[a-z][a-z \-]*", first):
            continue
        rows.append({"tcy": title, "en": first, "en_full": gloss[:120],
                     "kn": _clean_wiki(kn.group(1))[:80] if kn else ""})
    return rows


def build_vocab(tok, rng) -> dict:
    common = set(COMMON_WORDS)
    vocab = {}

    # Tulu: prefer headwords that occur in held-out Tulu text (commoner words), then fill randomly.
    tcy_rows = parse_tcy_wiktionary()
    tcy_text = " ".join(tok.decode(d[1:].tolist()) for d in val_docs("tcy"))
    for r in tcy_rows:
        r["val_count"] = len(re.findall(rf"(?<![\u0C80-\u0CFF]){re.escape(r['tcy'])}(?![\u0C80-\u0CFF])", tcy_text))
    common_tcy = [r for r in tcy_rows if r["en"] in common]
    rest = sorted((r for r in tcy_rows if r["en"] not in common), key=lambda r: -r["val_count"])
    seen, picked = set(), []
    for r in common_tcy + rest:
        if r["tcy"] in seen or r["en"] in {p["en"] for p in picked}:
            continue
        seen.add(r["tcy"])
        picked.append(r)
        if len(picked) >= N_VOCAB:
            break
    # all Tulu headwords sharing a gloss count as correct for en->tcy
    by_en: dict[str, list[str]] = {}
    for r in tcy_rows:
        by_en.setdefault(r["en"], []).append(r["tcy"])
    vocab["tcy"] = {"source": "tcy.wiktionary dump (not in any training mix)", "seen_in_training": False,
                    "n_entries_parsed": len(tcy_rows),
                    "pairs": [{"word": r["tcy"], "en": r["en"], "en_full": r["en_full"], "kn": r["kn"],
                               "en_accept": [r["en"]] + [s for s in (re.sub(r"^(a|an|the|to)\s+", "", x.strip().lower()).rstrip(".")
                                                                     for x in re.split(r"[;,]|\d+[.)]", r["en_full"]))
                                                         if s and s != r["en"] and len(s.split()) <= 2 and re.fullmatch(r"[a-z][a-z \-]*", s)],
                               "val_count": r["val_count"], "targets": sorted(set(by_en[r["en"]]))} for r in picked]}

    for lang, parser, src in (("kan", parse_alar, "Alar (in S4 dictionary mix + SFT)"),
                              ("mal", parse_olam, "Olam enml (in S4 dictionary mix + SFT)")):
        m = parser(common)
        kk = parse_kaikki(LANG_NAMES[lang], common)
        # Dictionaries list rare/figurative senses too (Olam "dog" -> "iron hook", Alar's poetic
        # synonyms), so the word shown for <lang>->en must ALSO be glossed as that English word by
        # Kaikki (en.wiktionary), and is the most frequent such word in held-out text.
        # en-><lang> accepts any target from either dictionary.
        val_text = " ".join(tok.decode(d[1:].tolist()) for d in val_docs(lang))
        freq = Counter(re.findall(r"[ಀ-೿ഀ-ൿ]+", val_text))
        confirmed = {w: [t for t in set(m.get(w, [])) | kk.get(w, set()) if t in kk.get(w, set()) and freq[t] >= 3]
                     for w in COMMON_WORDS}
        words = [w for w in COMMON_WORDS if confirmed[w]]
        idx = rng.permutation(len(words))[:N_VOCAB]
        pairs = []
        for i in sorted(idx):
            w = words[int(i)]
            best = max(confirmed[w], key=lambda t: freq[t])
            targets = sorted(set(m.get(w, [])) | kk.get(w, set()), key=lambda t: -freq[t])
            en_accept = [w] + [w2 for w2 in COMMON_WORDS if w2 != w and best in kk.get(w2, set())]
            pairs.append({"word": best, "en": w, "en_accept": en_accept, "targets": targets, "val_count": freq[best]})
        vocab[lang] = {"source": src + "; display word confirmed by Kaikki", "seen_in_training": True,
                       "n_common_matched": len(words), "pairs": pairs}

    vocab["kff"] = vocab["kxv"] = None  # no dictionary source on F: (eBible scripture only)
    return vocab


def build(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    rng = np.random.default_rng(20261018)
    t0 = time.time()
    cands = make_continuation_candidates(tok, rng)
    print(f"[build] {len(cands)} continuation candidates; near-dup scan of training streams...", flush=True)
    scanned = near_dup_scan(cands)
    cont = []
    dropped = Counter()
    for lang in (*MAIN_LANGS, *LOWRES_LANGS):
        n = N_MAIN if lang in MAIN_LANGS else N_LOWRES
        lc = [c for c in cands if c["lang"] == lang]
        clean = [c for c in lc if c["train_span_hits"] == 0]
        dropped[lang] = len(lc) - len(clean)
        cont.extend(clean[:n])
        print(f"[build] {lang}: {len(lc)} candidates, {dropped[lang]} share a 32-token span with train, "
              f"kept {min(n, len(clean))}", flush=True)
    vocab = build_vocab(tok, np.random.default_rng(7))
    data = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "near_dup": {"span_tokens": W, "rule": "drop if ANY 32-token span of prompt+reference occurs in a train stream",
                     "scanned": scanned, "candidates": Counter(c["lang"] for c in cands),
                     "dropped": dict(dropped), "not_scanned": "SFT v3/v4 row files (no longer on disk)"},
        "continuation": cont,
        "vocab": vocab,
        "english": [{"id": i, "prompt": p, "gold": g} for i, p, g in ENGLISH_QA],
    }
    PROMPTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[build] wrote {PROMPTS_PATH} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Run step
# ---------------------------------------------------------------------------

class GPULock:
    """out/locks/GPU.lock contract (AGENTS.md): refuse if held, hold for the run, always release."""

    def __init__(self, name: str):
        self.path = REPO_ROOT / "out" / "locks" / "GPU.lock"
        self.name = name

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise RuntimeError(f"GPU.lock held: {self.path.read_text(encoding='utf-8').strip()}")
        self.path.write_text(f"claude-code {self.name} {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def load_model(ckpt_path: Path, device: str):
    import torch
    from model import Parentheses
    from model.config import ModelConfig
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    model = Parentheses(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    step = ckpt.get("step", 0)
    del ckpt
    return model, step


def generate(model, tok, in_ids: list[int], device: str, max_new: int, decode: dict, seed: int) -> tuple[str, list[int], float]:
    import torch
    torch.manual_seed(seed)
    stops = {tok.token_to_id(t) for t in ("<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>", "<|pad|>", "<esg>")} - {None}
    idx = torch.tensor([in_ids], dtype=torch.long, device=device)
    out: list[int] = []
    t0 = time.time()
    for nxt in model.stream(idx, max_new_tokens=max_new, temperature=decode["temperature"], top_k=decode["top_k"],
                            min_p=decode["min_p"], repetition_penalty=decode["repetition_penalty"],
                            no_repeat_ngram_size=decode["no_repeat_ngram_size"]):
        t = int(nxt.item())
        if t in stops:
            break
        out.append(t)
        if is_looping(out):
            break
    text = tok.decode(out, skip_special_tokens=False)
    text = re.sub(r"</?(?:think|freethink)>|<\|(?:user|assistant|system|endoftext|pad)\|>|<esg>", "", text).strip()
    return text, out, time.time() - t0


def ref_bpb(model, tok, tag_id: int, prompt_ids: list[int], ref_ids: list[int], device: str) -> float:
    """Bits per UTF-8 byte of the reference continuation, conditioned on <lang> + prompt."""
    import torch
    import torch.nn.functional as F
    ids = [tag_id] + prompt_ids + ref_ids
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
    nats = float(nll[-len(ref_ids):].sum())
    nbytes = len(tok.decode(ref_ids).encode("utf-8"))
    return nats / math.log(2) / max(nbytes, 1)


def chat_ids(tok, text: str, lang: str = "eng") -> list[int]:
    return [tok.token_to_id("<|user|>"), tok.token_to_id(f"<{lang}>")] + \
        tok.encode(text, add_special_tokens=False).ids + [tok.token_to_id("<|assistant|>")]


def run_checkpoint(name: str, ckpt: str, data: dict, device: str, limit: int | None) -> dict:
    tok = load_tokenizer()
    model, step = load_model(REPO_ROOT / ckpt, device)
    res = {"name": name, "checkpoint": ckpt, "step": step, "device": device,
           "decode": {"continuation": SAMPLING, "english_greedy": GREEDY, "english_sampled": SAMPLING,
                      "english_product": {**SAMPLING, "system": "DEFAULT_SYSTEM_PROMPT", "fmt": "chat", "lang": "eng",
                                          "note": "chat.reply(): math/logic intercepts, RAG grounding, post-processing active"},
                      "vocab": GREEDY},
           "continuation": [], "vocab": {}, "english": []}
    t_all = time.time()

    items = data["continuation"]
    if limit:
        items = [c for lang in (*MAIN_LANGS, *LOWRES_LANGS) for c in [x for x in items if x["lang"] == lang][:limit]]
    for k, c in enumerate(items):
        tag = tok.token_to_id(f"<{c['lang']}>")
        text, out, dt = generate(model, tok, [tag] + c["prompt_ids"], device, CONT_NEW_TOKENS, SAMPLING, seed=1000 + k)
        r = {"id": c["id"], "lang": c["lang"], "output": text, "n_tokens": len(out), "sec": round(dt, 2),
             "script_fidelity": round(script_fidelity(text, c["lang"]), 4),
             "ref_script_fidelity": round(script_fidelity(c["ref_text"], c["lang"]), 4),
             "rep16": round(repetition_rate(text), 4), "ref_rep16": round(repetition_rate(c["ref_text"]), 4),
             "distinct2": round(distinct_n(text, 2), 4), "loop": is_looping(out),
             "bpb_ref": round(ref_bpb(model, tok, tag, c["prompt_ids"], c["ref_ids"], device), 4)}
        res["continuation"].append(r)
        print(f"[{name}] cont {c['id']}: script={r['script_fidelity']:.2f} rep16={r['rep16']:.2f} "
              f"bpb={r['bpb_ref']:.3f} toks={r['n_tokens']}", flush=True)

    for lang, v in data["vocab"].items():
        if not v:
            continue
        lname = LANG_NAMES[lang]
        rows = []
        pairs = v["pairs"][:limit] if limit else v["pairs"]
        for k, p in enumerate(pairs):
            q1 = f'What does the {lname} word "{p["word"]}" mean in English? Answer with one English word.'
            o1, t1, _ = generate(model, tok, chat_ids(tok, q1), device, 40, GREEDY, seed=k)
            q2 = f'How do you say "{p["en"]}" in {lname}? Answer with the {lname} word only.'
            o2, t2, _ = generate(model, tok, chat_ids(tok, q2), device, 40, GREEDY, seed=k)
            rows.append({"word": p["word"], "en": p["en"], "en_accept": p.get("en_accept", [p["en"]]), "targets": p["targets"],
                         "to_en_prompt": q1, "to_en_output": o1,
                         "to_en_correct": contains_word(o1, p.get("en_accept", [p["en"]]), latin=True),
                         "from_en_prompt": q2, "from_en_output": o2,
                         "from_en_correct": contains_word(o2, p["targets"], latin=False),
                         "from_en_script": round(script_fidelity(o2, lang), 3) if o2 else 0.0})
            print(f"[{name}] vocab {lang} {p['en']}: to_en={rows[-1]['to_en_correct']} from_en={rows[-1]['from_en_correct']}", flush=True)
        res["vocab"][lang] = rows

    qs = data["english"][:limit] if limit else data["english"]
    import chat as chat_mod
    for k, q in enumerate(qs):
        row = {"id": q["id"], "prompt": q["prompt"], "gold": q["gold"]}
        for mode, dec in (("greedy", GREEDY), ("sampled", SAMPLING)):
            text, out, dt = generate(model, tok, chat_ids(tok, q["prompt"]), device, 80, dec, seed=500 + k)
            row[mode] = {"output": text, "n_tokens": len(out), **score_english(text, q["prompt"], q["gold"])}
        t0 = time.time()
        text = chat_mod.reply(model, q["prompt"], device, 80, SAMPLING["temperature"], SAMPLING["top_k"],
                              tok=tok, lang="eng", fmt="chat", stream=False, system=chat_mod.DEFAULT_SYSTEM_PROMPT)
        row["product"] = {"output": text, "n_tokens": len(tok.encode(text, add_special_tokens=False).ids),
                          "sec": round(time.time() - t0, 2), **score_english(text, q["prompt"], q["gold"])}
        res["english"].append(row)
        print(f"[{name}] eng {q['id']}: product exact={row['product']['exact']} contains={row['product']['contains']} "
              f"| raw greedy exact={row['greedy']['exact']} contains={row['greedy']['contains']} "
              f"| raw sampled exact={row['sampled']['exact']} contains={row['sampled']['contains']}", flush=True)

    res["total_sec"] = round(time.time() - t_all, 1)
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


def run(args):
    if not PROMPTS_PATH.exists():
        raise SystemExit(f"{PROMPTS_PATH} missing: run the build step first")
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    names = args.checkpoints or list(CHECKPOINTS)
    with GPULock("eval_practical_suite"):
        for name in names:
            res = run_checkpoint(name, CHECKPOINTS[name], data, args.device, args.limit)
            suffix = "_limit" if args.limit else ""
            (OUT_DIR / f"results_{name}{suffix}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[run] {name} done in {res['total_sec']}s")


# ---------------------------------------------------------------------------
# Report step
# ---------------------------------------------------------------------------

def summarize(res: dict) -> dict:
    s = {"cont": {}, "vocab": {}, "english": {}}
    for lang in (*MAIN_LANGS, *LOWRES_LANGS):
        rows = [r for r in res["continuation"] if r["lang"] == lang]
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        s["cont"][lang] = {"n": len(rows), "script": m("script_fidelity"), "ref_script": m("ref_script_fidelity"),
                           "rep16": m("rep16"), "ref_rep16": m("ref_rep16"), "distinct2": m("distinct2"),
                           "loop": m("loop"), "bpb": m("bpb_ref"), "tokens": m("n_tokens"),
                           "wrong_script_80": float(np.mean([r["script_fidelity"] < 0.8 for r in rows]))}
    for lang, rows in res["vocab"].items():
        s["vocab"][lang] = {"n": len(rows), "to_en": float(np.mean([r["to_en_correct"] for r in rows])),
                            "from_en": float(np.mean([r["from_en_correct"] for r in rows])),
                            "from_en_script": float(np.mean([r["from_en_script"] for r in rows]))}
    for mode in ("product", "greedy", "sampled"):
        rows = [r for r in res["english"] if mode in r]
        if not rows:
            continue
        s["english"][mode] = {"n": len(rows), "exact": float(np.mean([r[mode]["exact"] for r in rows])),
                              "contains": float(np.mean([r[mode]["contains"] for r in rows]))}
    return s


def _clip(s: str, n: int) -> str:
    s = s.replace("\n", " ⏎ ").replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "…"


def report(args):
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    results = {}
    for name in CHECKPOINTS:
        p = OUT_DIR / f"results_{name}.json"
        if p.exists():
            results[name] = json.loads(p.read_text(encoding="utf-8"))
    if not results:
        raise SystemExit("no results_*.json: run the run step first")
    # re-score stored outputs with the current scorer and golds, so scorer fixes never need a GPU re-run
    gold_by_id = {i: g for i, _, g in ENGLISH_QA}
    for q in data["english"]:
        q["gold"] = gold_by_id.get(q["id"], q["gold"])
    for r in results.values():
        for row in r["english"]:
            row["gold"] = gold_by_id.get(row["id"], row["gold"])
            for mode in ("product", "greedy", "sampled"):
                if mode in row:
                    row[mode].update(score_english(row[mode]["output"], row["prompt"], row["gold"]))
        for rows in r["vocab"].values():
            for x in rows:
                x["to_en_correct"] = contains_word(x["to_en_output"], x.get("en_accept", [x["en"]]), latin=True)
                x["from_en_correct"] = contains_word(x["from_en_output"], x["targets"], latin=False)
    sums = {n: summarize(r) for n, r in results.items()}
    names = list(results)
    pct = lambda x: f"{100 * x:.0f}%"
    L = []
    L.append(f"# Practical open-generation eval: Parentheses 0.9-120M SFT {' vs '.join(names)} ({time.strftime('%Y-%m-%d')})\n")
    L.append("Script: `tests/eval_practical_suite.py` (Claude Code). Raw results: `out/practical_eval/results_*.json`; "
             "prompts: `out/practical_eval/prompts.json`.\n")
    L.append("Every score below comes from **generated text** (or, for bpb, the likelihood of a held-out reference). "
             "No multiple-choice logits anywhere.\n")
    L.append("| checkpoint | path | step | wall time |\n|---|---|---|---|")
    for n in names:
        r = results[n]
        L.append(f"| {n} | `{r['checkpoint']}` | {r['step']} | {r['total_sec']:.0f}s |")
    L.append("")

    # headline
    L.append("## Headline\n")
    L.append("| metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    def row(label, f):
        L.append(f"| {label} | " + " | ".join(f(sums[n]) for n in names) + " |")
    row("English exact, product chat.reply (n=%d)" % len(data["english"]),
        lambda s: pct(s["english"]["product"]["exact"]) if "product" in s["english"] else "-")
    row("English contains, product chat.reply",
        lambda s: pct(s["english"]["product"]["contains"]) if "product" in s["english"] else "-")
    row("English exact, raw greedy", lambda s: pct(s["english"]["greedy"]["exact"]))
    row("English contains, raw greedy", lambda s: pct(s["english"]["greedy"]["contains"]))
    row("English exact, raw sampled T=0.7", lambda s: pct(s["english"]["sampled"]["exact"]))
    row("English contains, raw sampled T=0.7", lambda s: pct(s["english"]["sampled"]["contains"]))
    for lang in MAIN_LANGS:
        row(f"{LANG_NAMES[lang]} continuation: script fidelity / bpb",
            lambda s, l=lang: f"{pct(s['cont'][l]['script'])} / {s['cont'][l]['bpb']:.3f}" if l in s["cont"] else "-")
    for lang, v in data["vocab"].items():
        if v:
            row(f"{LANG_NAMES[lang]} vocab {lang}→en / en→{lang}" + (" (seen in training)" if v["seen_in_training"] else " (unseen source)"),
                lambda s, l=lang: f"{pct(s['vocab'][l]['to_en'])} / {pct(s['vocab'][l]['from_en'])}" if l in s["vocab"] else "-")
    L.append("")

    # section 1
    nd = data["near_dup"]
    L.append("## 1. Dravidian open continuation (held-out val, near-dup filtered)\n")
    L.append(f"Prompts: first ~{PROMPT_TOKENS} tokens of a `data/processed_v4/val/<lang>.bin` doc (word-aligned), "
             f"reference = next ~{REF_TOKENS} tokens. Prompt is fed as `<lang>` + text (plain continuation, no chat tags). "
             f"Generation: {CONT_NEW_TOKENS} new tokens, chat.py sampling defaults "
             f"(T={SAMPLING['temperature']}, top_k={SAMPLING['top_k']}, min_p={SAMPLING['min_p']}, "
             f"repetition_penalty={SAMPLING['repetition_penalty']}, no_repeat_ngram={SAMPLING['no_repeat_ngram_size']}), fixed seeds.\n")
    L.append(f"Near-dup filter: {nd['rule']}. Candidates per language: {dict(nd['candidates'])}; dropped: {nd['dropped']}. "
             f"Streams scanned: " + "; ".join(f"`{k}` ({v})" for k, v in nd["scanned"].items()) +
             f". Not scanned: {nd['not_scanned']}.\n")
    L.append("Metrics: **script** = share of letters in the language's script block (reference text shown for calibration); "
             "**rep16** = duplicate char-16-gram rate (0 = none, ~1 = loop); **distinct-2** = unique word bigrams; "
             "**bpb** = bits per UTF-8 byte of the reference continuation given the prompt; **<80%** = share of outputs under 80% script fidelity.\n")
    hdr = "| lang | n | ref script | " + " | ".join(f"{n} script | {n} <80% | {n} rep16 | {n} distinct-2 | {n} bpb" for n in names) + " |"
    L.append(hdr)
    L.append("|---|---|---|" + "---|" * (5 * len(names)))
    for lang in (*MAIN_LANGS, *LOWRES_LANGS):
        if lang not in sums[names[0]]["cont"]:
            continue
        c0 = sums[names[0]]["cont"][lang]
        cells = []
        for n in names:
            c = sums[n]["cont"][lang]
            cells.append(f"{pct(c['script'])} | {pct(c['wrong_script_80'])} | {c['rep16']:.3f} | {c['distinct2']:.2f} | {c['bpb']:.3f}")
        L.append(f"| {LANG_NAMES[lang]} ({lang}) | {c0['n']} | {pct(c0['ref_script'])} | " + " | ".join(cells) + " |")
    L.append(f"\nReference text rep16 for calibration: " +
             ", ".join(f"{l} {sums[names[0]]['cont'][l]['ref_rep16']:.3f}" for l in sums[names[0]]["cont"]) + ".\n")

    L.append("### Sample continuations (first 2 per language)\n")
    cont_by_id = {c["id"]: c for c in data["continuation"]}
    for lang in (*MAIN_LANGS, *LOWRES_LANGS):
        ids = [c["id"] for c in data["continuation"] if c["lang"] == lang][:2]
        for cid in ids:
            c = cont_by_id[cid]
            L.append(f"**{LANG_NAMES[lang]} `{cid}`**  ")
            L.append(f"- prompt: {_clip(c['prompt_text'], 220)}")
            L.append(f"- reference: {_clip(c['ref_text'], 220)}")
            for n in names:
                r = next(x for x in results[n]["continuation"] if x["id"] == cid)
                L.append(f"- **{n}** (script {pct(r['script_fidelity'])}, rep16 {r['rep16']:.2f}, bpb {r['bpb_ref']:.3f}): {_clip(r['output'], 300)}")
            L.append("")

    # section 2
    L.append("## 2. Vocabulary / translation pairs (chat mode, greedy, contains-match)\n")
    L.append("Prompts: `What does the <Lang> word \"X\" mean in English? Answer with one English word.` and "
             "`How do you say \"w\" in <Lang>? Answer with the <Lang> word only.` (chat tags, `<eng>` user tag, greedy, 40 tokens). "
             "to-en correct = an accepted English gloss appears as a whole word (the pair's word, plus any other basic-vocabulary "
             "gloss Kaikki / the Tulu entry gives the same native word); from-en correct = any accepted target word appears as a substring. "
             "Kannada/Malayalam display words must be glossed as the English word by Kaikki (Alar/Olam add accepted en→ targets), "
             "and are the most frequent such word in held-out text.\n")
    L.append("| lang | source | n | " + " | ".join(f"{n} →en | {n} en→ | {n} en→ script" for n in names) + " |")
    L.append("|---|---|---|" + "---|" * (3 * len(names)))
    for lang, v in data["vocab"].items():
        if not v:
            continue
        cells = [f"{pct(sums[n]['vocab'][lang]['to_en'])} | {pct(sums[n]['vocab'][lang]['from_en'])} | {pct(sums[n]['vocab'][lang]['from_en_script'])}" for n in names]
        L.append(f"| {LANG_NAMES[lang]} | {v['source']} | {sums[names[0]]['vocab'][lang]['n']} | " + " | ".join(cells) + " |")
    L.append("| Koya / Kuvi | no dictionary source in F:\\Lingual_Data (eBible scripture only) | 0 | " + " | ".join("- | - | -" for _ in names) + " |")
    L.append("")
    L.append("### Sample vocab answers\n")
    for lang in results[names[0]]["vocab"]:
        L.append(f"**{LANG_NAMES[lang]}**\n")
        L.append("| word | gold | " + " | ".join(f"{n} →en | {n} en→" for n in names) + " |")
        L.append("|---|---|" + "---|" * (2 * len(names)))
        for i in range(min(6, len(results[names[0]]["vocab"][lang]))):
            r0 = results[names[0]]["vocab"][lang][i]
            cells = []
            for n in names:
                r = results[n]["vocab"][lang][i]
                cells.append(f"{'✓' if r['to_en_correct'] else '✗'} {_clip(r['to_en_output'], 60)} | "
                             f"{'✓' if r['from_en_correct'] else '✗'} {_clip(r['from_en_output'], 60)}")
            L.append(f"| {r0['word']} | {r0['en']} / {', '.join(r0['targets'][:3])} | " + " | ".join(cells) + " |")
        L.append("")

    # section 3
    L.append("## 3. English baseline (test-01 stable-answer questions)\n")
    L.append("Excluded from test-01: live-web questions (Python/Ubuntu/Node/Git versions, USD-INR), the unnamed-project questions, "
             "and the open-ended RAM essay. **product** = `chat.reply()` with `DEFAULT_SYSTEM_PROMPT`, chat.py sampling defaults "
             "(min_p 0.06, repetition_penalty 1.12, no_repeat_ngram 4), so deterministic math/logic intercepts, RAG grounding, and "
             "instruction post-processing are active; this is the shipping path. **raw greedy** = model-only top_k 1 without the system "
             "prompt; **raw sampled** = model-only T=0.7/top-k 40 (both kept for comparison). "
             "**exact** = first non-empty line is the answer (a ≤4-word first line containing it counts). "
             "**contains** = gold anywhere in the output, except where the gold already appears in the prompt "
             "(`0.8 or 0.75`, `all but 4 sheep`, `144 divided by 12`), where contains falls back to exact.\n")
    mode_labels = (("product", "product (chat.reply)"), ("greedy", "raw greedy"), ("sampled", "raw sampled T=0.7"))
    for mode, label in mode_labels:
        present = [n for n in names if mode in sums[n]["english"]]
        if not present:
            continue
        L.append(f"### {label}\n")
        L.append("| id | gold | " + " | ".join(present) + " |")
        L.append("|---|---|" + "---|" * len(present))
        for i, q in enumerate(data["english"]):
            cells = []
            for n in present:
                m = results[n]["english"][i][mode]
                mark = "E" if m["exact"] else ("C" if m["contains"] else "✗")
                cells.append(f"{mark} {_clip(m['output'], 90)}")
            L.append(f"| {q['id']} | `{_clip(' / '.join(q['gold']), 30)}` | " + " | ".join(cells) + " |")
        L.append("\nE = exact, C = contains only, ✗ = wrong.\n")

    out = Path(args.report_out) if args.report_out else VAULT_REPORTS / f"Practical Eval 120M SFT {' vs '.join(names)} {time.strftime('%Y-%m-%d')}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] wrote {out}")


# ---------------------------------------------------------------------------

def self_test():
    assert script_fidelity("ನೀರು ಕುಡಿ", "kan") == 1.0
    assert script_fidelity("ನೀರು water", "kan") < 0.5
    assert script_fidelity("123 ...", "tam") == 0.0
    assert script_fidelity("ଜଳ", "kff") == 1.0
    assert repetition_rate("abcdefghijklmnopqrstuvwxyz") == 0.0
    assert repetition_rate("abcabcabc" * 20) > 0.8
    assert distinct_n("a b a b a b", 2) == 2 / 5
    assert is_looping(list(range(16)) * 3) and not is_looping(list(range(48)))
    s = score_english("Canberra.", "What is the capital of Australia?", [r"canberra"])
    assert s["exact"] and s["contains"]
    s = score_english("The capital of Australia is a town in Ukraine.", "What is the capital of Australia?", [r"canberra"])
    assert not s["exact"] and not s["contains"]
    s = score_english("The largest is 0.75. 0.8 is more than", "Which is larger: 0.8 or 0.75?", [r"0\.8\b(?!\d)"])
    assert s["echo_risk"] and not s["contains"] and not s["exact"]
    assert score_english("You are in second place.", "If you overtake the person in second place during a race, what position are you in? Answer with one word.", [r"\bsecond\b", r"\b2nd\b"])["contains"]
    assert score_english("144 divided by 12 = 12", "What is 144 divided by 12? Answer with the number only.", [r"\b12\b(?![.,]\d)", r"\btwelve\b"])["contains"]
    assert score_english("0.8", "Which is larger: 0.8 or 0.75?", [r"0\.8\b(?!\d)"])["exact"]
    assert score_english("\\boxed{102}", "What is 17 × 6?", [r"\b102\b"])["exact"]
    assert score_english("So the answer is \\boxed{48}", "What is 17 × 6?", [r"\b102\b"])["contains"] is False
    assert score_english("K", "symbol for potassium", [r"(?-i)\bK\b"])["exact"]
    assert not score_english("ok, the key", "symbol for potassium", [r"(?-i)\bK\b"])["contains"]
    assert score_english("Jensen Huang", "Who is NVIDIA's CEO?", [r"jensen", r"huang"])["exact"]
    assert score_english("No, we cannot conclude that.", "roses", [r"^\W*no\b", r"\bcannot\b"])["exact"]
    assert score_english("7.5 degrees", "clock", [r"\b7\.5\b"])["exact"]
    assert contains_word("It means water.", ["water"], latin=True)
    assert contains_word("Trees", ["tree"], latin=True)
    assert not contains_word("waterfall", ["water"], latin=True)
    assert contains_word("ನೀರನ್ನು", ["ನೀರ"], latin=False)
    assert _clean_wiki("[https://en.wikipedia.org/wiki/Strain strain]; [[a|b]]") == "strain; b"
    print("eval_practical_suite self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=["build", "run", "report", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", nargs="*", choices=list(CHECKPOINTS))
    ap.add_argument("--limit", type=int, default=None, help="items per section (smoke test; writes results_*_limit.json)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild prompts.json even if it exists")
    ap.add_argument("--report-out", default=None, help="markdown path (default: vault Reports folder)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.step in ("build", "all") and (args.rebuild or not PROMPTS_PATH.exists() or args.step == "build"):
        build(args)
    if args.step in ("run", "all"):
        run(args)
    if args.step in ("report", "all") and not args.limit:
        report(args)


if __name__ == "__main__":
    main()
