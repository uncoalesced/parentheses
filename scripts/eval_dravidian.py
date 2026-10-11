"""
Dravidian eval: val loss (overall, per language, all-sources) and language-tag
conditioning, for one or more checkpoints. Prints a table and writes JSON.

Languages: the four anchors from processed_dravidian_v1 plus the low-resource
set in processed_dravidian_lowres_v1 (one folder per language).

Tag conditioning, two ways:
  - script test: sample from `<lang> ` and check the output's majority Unicode
    script is that language's script. Tells apart languages with different
    scripts only (Koya/Kuvi = Odia, Aheri Gondi = Devanagari).
  - tag-swap test: loss on real val text under its own tag vs a confusable
    tag (Tulu vs <kan>, Kolami vs <tel>, Koya vs <kxv>, ...). A positive gap
    means the model uses the tag; works for languages that share a script.

    python scripts/eval_dravidian.py --checkpoints checkpoints/dravidian-v2-replay/step_760999_final.pt
    python scripts/eval_dravidian.py --self-test

All val.bin files are uint16 token-id arrays (documentation.md, 2026-09-22 §2).
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Parentheses
from train import boundary_reset_mask

LANGS = ("kan", "tam", "tel", "mal")
LOWRES = ("tcy", "kff", "kxv", "nit", "esg")
SCRIPTS = {"Knda": (0x0C80, 0x0CFF), "Taml": (0x0B80, 0x0BFF), "Telu": (0x0C00, 0x0C7F),
           "Mlym": (0x0D00, 0x0D7F), "Orya": (0x0B00, 0x0B7F), "Deva": (0x0900, 0x097F)}
LANG_SCRIPT = {"kan": "Knda", "tam": "Taml", "tel": "Telu", "mal": "Mlym",
               "tcy": "Knda", "kff": "Orya", "kxv": "Orya", "nit": "Telu", "esg": "Deva"}
SWAP = {"kan": "tcy", "tam": "mal", "tel": "nit", "mal": "tam",
        "tcy": "kan", "kff": "kxv", "kxv": "kff", "nit": "tel", "esg": "tel"}
DRAV = "data/processed_dravidian_v1"
LOWRES_DIR = "data/processed_dravidian_lowres_v1"
ALL_SOURCES_VAL = "data/processed_all_sources_v1/val.bin"
TAG_LEN = 6  # "<xxx> "


def script_of(text: str) -> str | None:
    """Majority script (of SCRIPTS) among the letters of `text`, None if it has none."""
    counts = dict.fromkeys(SCRIPTS, 0)
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPTS.items():
            if lo <= cp <= hi:
                counts[name] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] else None


def distinct_n(text: str | list[str], n: int) -> float:
    """Distinct-n: unique word n-grams / total word n-grams in [0, 1].
    If text is str, computes on that string; if list, averages across samples."""
    if isinstance(text, str):
        words = re.findall(r"\w+", text)
        if len(words) < n:
            return 1.0 if words else 0.0
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        return len(set(grams)) / len(grams)
    scores = [distinct_n(t, n) for t in text]
    return float(np.mean(scores)) if scores else 0.0


def repetition_rate(text: str | list[str], n: int = 16) -> float:
    """Duplicate n-gram rate in [0, 1]. 0 = no repeats, ~1 = a loop.
    Uses character n-grams (default n=16). If list, averages across samples."""
    if isinstance(text, str):
        grams = [text[i:i + n] for i in range(len(text) - n + 1)]
        if not grams:
            return 0.0
        return 1.0 - len(set(grams)) / len(grams)
    scores = [repetition_rate(t, n) for t in text]
    return float(np.mean(scores)) if scores else 0.0


repetition = repetition_rate  # alias for eval_persona compatibility


def doc_langs(data: np.memmap, boundaries: np.ndarray) -> np.ndarray:
    """Language of each document from its leading `<xxx> ` tag ('' if untagged)."""
    out = []
    for s in boundaries:
        head = bytes(data[s:s + 5].astype(np.uint8))
        out.append(head[1:4].decode("ascii", "replace") if head[:1] == b"<" and head[4:5] == b">" else "")
    return np.array(out)


def window_starts(rng, bounds_ext: np.ndarray, doc_ids: np.ndarray, block: int, n: int, hi: int) -> np.ndarray:
    """n window starts, each uniform inside a document drawn from doc_ids."""
    picks = rng.choice(doc_ids, size=n)
    lo, ln = bounds_ext[picks], bounds_ext[picks + 1] - bounds_ext[picks]
    return np.minimum(lo + (rng.random(n) * ln).astype(np.int64), hi)


def _loss(model, x, y, rm, device) -> float:
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        _, loss = model(x, y, reset_mask=rm)
    return loss.item()


@torch.no_grad()
def mean_loss(model, data, starts, block, batch, device, boundaries=None) -> float:
    losses = []
    for i in range(0, len(starts), batch):
        ix = starts[i:i + batch]
        x = torch.stack([torch.from_numpy(data[j:j + block].astype(np.int64)) for j in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + block].astype(np.int64)) for j in ix]).to(device)
        rm = None if boundaries is None else boundary_reset_mask(boundaries, ix, block).to(device)
        losses.append(_loss(model, x, y, rm, device))
    return float(np.mean(losses))


def swap_tag(x: torch.Tensor, lang: str) -> torch.Tensor:
    """Same rows with the leading `<lang>` tag bytes replaced by `<SWAP[lang]>`."""
    x = x.clone()
    x[:, 1:4] = torch.tensor(list(SWAP[lang].encode()), dtype=x.dtype)
    return x


@torch.no_grad()
def tag_swap_gap(model, data, bnd, lang, block, n, device, docs=None) -> float:
    """Mean loss on the first `block` bytes of each val doc (tag excluded from scoring)
    under a confusable tag minus under its own tag. Positive = the tag is used.
    `docs` = the doc starts to score (default all of `bnd`); `bnd` stays the full
    boundary set so the reset mask sees every document start."""
    ix = [int(s) for s in (bnd if docs is None else docs) if data[s] == ord("<")][:n]
    ix = [s for s in ix if s + block + 1 <= len(data)]
    x = torch.stack([torch.from_numpy(data[j:j + block].astype(np.int64)) for j in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[j + 1:j + 1 + block].astype(np.int64)) for j in ix]).to(device)
    y[:, :TAG_LEN - 1] = -1                       # don't score predicting the tag itself
    rm = boundary_reset_mask(bnd, np.array(ix), block).to(device)
    return _loss(model, swap_tag(x, lang), y, rm, device) - _loss(model, x, y, rm, device)


@torch.no_grad()
def tag_accuracy(model, langs, device, n: int, new_tokens: int, seed: int) -> dict:
    out = {}
    for lang in langs:
        torch.manual_seed(seed)
        prompt = f"<{lang}> ".encode()
        ids = torch.tensor([list(prompt)] * n, dtype=torch.long, device=device)
        gen = model.generate(ids, new_tokens, temperature=0.8, top_k=40)
        texts = [bytes(row[len(prompt):].tolist()).decode("utf-8", "replace") for row in gen]
        got = [script_of(t) for t in texts]
        out[lang] = {"acc": sum(g == LANG_SCRIPT[lang] for g in got) / n,
                     "got": {str(k): got.count(k) for k in set(got)},
                     "repetition_rate": repetition_rate(texts),
                     "distinct_2": distinct_n(texts, 2),
                     "distinct_3": distinct_n(texts, 3),
                     "sample": texts[0]}
    return out


def evaluate(path, args) -> dict:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = Parentheses(cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    block, n = cfg.block_size, args.batches * args.batch_size
    rng = np.random.default_rng(args.seed)

    val = np.memmap(f"{DRAV}/val.bin", dtype=np.uint16, mode="r")
    bnd = np.load(f"{DRAV}/val_boundaries.npy")
    bounds_ext = np.append(np.asarray(bnd, dtype=np.int64), len(val))
    langs = doc_langs(val, bnd)
    hi = len(val) - block - 1
    res = {"checkpoint": path, "step": ckpt.get("step"),
           "drav_val": mean_loss(model, val, window_starts(rng, bounds_ext, np.arange(len(bnd)), block, n, hi),
                                 block, args.batch_size, args.device, bnd)}
    res["tagswap"] = {}
    for lang in LANGS:
        ids = np.nonzero(langs == lang)[0]
        res[f"{lang}_val"] = mean_loss(model, val, window_starts(rng, bounds_ext, ids, block, n, hi),
                                       block, args.batch_size, args.device, bnd) if len(ids) else None
        if len(ids):
            res["tagswap"][lang] = tag_swap_gap(model, val, bnd, lang, block, args.swap_docs, args.device,
                                                  docs=bnd[ids])
    for lang in LOWRES:
        d = f"{LOWRES_DIR}/{lang}"
        if not os.path.exists(f"{d}/val.bin"):
            continue
        lv = np.memmap(f"{d}/val.bin", dtype=np.uint16, mode="r")
        lb = np.load(f"{d}/val_boundaries.npy")
        lext = np.append(np.asarray(lb, dtype=np.int64), len(lv))
        res[f"{lang}_val"] = mean_loss(model, lv, window_starts(rng, lext, np.arange(len(lb)), block, n,
                                                                len(lv) - block - 1),
                                       block, args.batch_size, args.device, lb)
        res["tagswap"][lang] = tag_swap_gap(model, lv, lb, lang, block, args.swap_docs, args.device)
    if os.path.exists(ALL_SOURCES_VAL):
        av = np.memmap(ALL_SOURCES_VAL, dtype=np.uint16, mode="r")
        res["all_sources_val"] = mean_loss(model, av, rng.integers(0, len(av) - block - 1, n),
                                           block, args.batch_size, args.device)
    present = [l for l in (*LANGS, *LOWRES) if res.get(f"{l}_val") is not None]
    res["tags"] = tag_accuracy(model, present, args.device, args.tag_samples, args.tag_tokens, args.seed)
    res["repetition_rate"] = {l: res["tags"][l]["repetition_rate"] for l in present}
    res["distinct_2"] = {l: res["tags"][l]["distinct_2"] for l in present}
    res["distinct_3"] = {l: res["tags"][l]["distinct_3"] for l in present}
    return res


# ---------------------------------------------------------------- bits/byte mode
# Byte (vocab 256) and BPE checkpoints scored on the SAME text: each language's
# old val split, decoded (build_corpus_v3.val_texts). Every doc is scored on its
# text plus one end marker ("\n" for byte models, <|endoftext|> for BPE); the tag
# is context, never a target. bits/byte = sum nats / (sum utf8 bytes + docs) / ln 2.
# Windows are non-overlapping block_size chunks with document resets, so each
# model gets exactly its own context length.

BPB_LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng")


def load_tokenizer(ckpt):
    meta = ckpt.get("tokenizer")
    if not meta:
        return None
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(meta["path"], "tokenizer.json"))
    tok.encode_special_tokens = True
    return tok


def doc_stream(texts, lang, tok):
    """(ids, targets_mask, doc starts) for docs laid end to end."""
    ids, score, starts = [], [], []
    for t in texts:
        starts.append(len(ids))
        if tok is None:
            tag = b"" if lang == "eng" else f"<{lang}> ".encode()
            body = list(t.encode("utf-8")) + [10]
            ids += list(tag) + body
            score += [False] * len(tag) + [True] * len(body)
        else:
            body = tok.encode(t, add_special_tokens=False).ids + [tok.token_to_id("<|endoftext|>")]
            ids += [tok.token_to_id(f"<{lang}>")] + body
            score += [False] + [True] * len(body)
    return np.array(ids, dtype=np.int64), np.array(score), np.array(starts, dtype=np.int64)


@torch.no_grad()
def sum_nats(model, ids, score, starts, block, batch, device) -> tuple[float, int]:
    x_all, y_all = ids[:-1], np.where(score[1:], ids[1:], -1)
    n = -(-len(x_all) // block) * block
    x_all = np.pad(x_all, (0, n - len(x_all)))
    y_all = np.pad(y_all, (0, n - len(y_all)), constant_values=-1)
    wstarts = np.arange(0, n, block)
    total, count = 0.0, 0
    for i in range(0, len(wstarts), batch):
        ix = wstarts[i:i + batch]
        x = torch.from_numpy(np.stack([x_all[s:s + block] for s in ix])).to(device)
        y = torch.from_numpy(np.stack([y_all[s:s + block] for s in ix])).to(device)
        rm = boundary_reset_mask(starts, ix, block).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(x, None, reset_mask=rm)
        total += torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                                   ignore_index=-1, reduction="sum").item()
        count += int((y != -1).sum())
    return total, count


@torch.no_grad()
def tag_test_bpb(model, tok, lang, device, n, new_tokens, seed):
    torch.manual_seed(seed)
    if tok is None:
        prompt = list(f"<{lang}> ".encode())
        dec = lambda ids: bytes([i for i in ids if i < 256]).decode("utf-8", "replace")
    else:
        prompt = [tok.token_to_id(f"<{lang}>")]
        dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
    gen = model.generate(torch.tensor([prompt] * n, device=device), new_tokens, temperature=0.8, top_k=40)
    texts = [dec(row[len(prompt):].tolist()) for row in gen]
    got = [script_of(t) for t in texts]
    want = LANG_SCRIPT.get(lang)
    rep = repetition_rate(texts)
    d2 = distinct_n(texts, 2)
    d3 = distinct_n(texts, 3)
    return {"acc": None if want is None else sum(g == want for g in got) / n,
            "got": {str(k): got.count(k) for k in set(got)},
            "repetition_rate": rep, "distinct_2": d2, "distinct_3": d3,
            "samples": texts[:3]}


def evaluate_bpb(path, args, texts) -> dict:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = Parentheses(cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    tok = load_tokenizer(ckpt)
    res = {"checkpoint": path, "step": ckpt.get("step"), "vocab": cfg.vocab_size, "block": cfg.block_size,
           "params": model.num_params(), "bpb": {}, "tokens_per_byte": {}, "tags": {}, "tagswap": {},
           "repetition_rate": {}, "distinct_2": {}, "distinct_3": {}}
    batch = max(1, args.bpb_tokens // cfg.block_size)
    for lang in BPB_LANGS:
        ids, score, starts = doc_stream(texts[lang], lang, tok)
        nats, n = sum_nats(model, ids, score, starts, cfg.block_size, batch, args.device)
        nbytes = sum(len(t.encode("utf-8")) + 1 for t in texts[lang])
        res["bpb"][lang] = nats / nbytes / np.log(2)
        res["tokens_per_byte"][lang] = n / nbytes
        if lang == "eng" and tok is None:
            continue          # byte models never saw an <eng> tag
        res["tags"][lang] = tag_test_bpb(model, tok, lang, args.device, args.tag_samples, args.tag_tokens,
                                         args.seed)
        res["repetition_rate"][lang] = res["tags"][lang]["repetition_rate"]
        res["distinct_2"][lang] = res["tags"][lang]["distinct_2"]
        res["distinct_3"][lang] = res["tags"][lang]["distinct_3"]
        if lang in SWAP:      # mean loss under a confusable tag minus under the docs' own tag
            heads = [t[:500] for t in texts[lang][:args.swap_docs]]   # the tag matters most early on
            sw = doc_stream(heads, SWAP[lang], tok)
            own = doc_stream(heads, lang, tok)
            a = sum_nats(model, *sw, cfg.block_size, batch, args.device)
            b = sum_nats(model, *own, cfg.block_size, batch, args.device)
            res["tagswap"][lang] = a[0] / a[1] - b[0] / b[1]
    print(f"[bpb] {path}: " + " ".join(f"{l} {v:.3f}" for l, v in res["bpb"].items()), flush=True)
    return res


def bpb_main(args):
    sys.path.insert(0, "F:/Lingual_Data/scripts")
    from build_corpus_v3 import val_texts
    texts = {}
    for lang, docs in val_texts().items():
        keep, n = [], 0
        for d in docs:
            keep.append(d)
            n += len(d.encode())
            if n >= args.bpb_bytes:
                break
        texts[lang] = keep
    results = [evaluate_bpb(c, args, texts) for c in args.checkpoints]
    print(f"\nbits per UTF-8 byte (same val text for every model; ~{args.bpb_bytes / 1e6:.1f} MB/lang)")
    print(f"{'checkpoint':<52}{'params':>8}" + "".join(f"{l:>7}" for l in BPB_LANGS))
    for r in results:
        print(f"{r['checkpoint'][-52:]:<52}{r['params'] / 1e6:>7.1f}M"
              + "".join(f"{r['bpb'][l]:>7.3f}" for l in BPB_LANGS))
    print("\ntag accuracy (script of a sample from the tag alone) / tag-swap gap (nats/token)")
    for r in results:
        print(f"{r['checkpoint'][-52:]:<52}" + "".join(
            f" {l}:{r['tags'][l]['acc'] if r['tags'][l]['acc'] is not None else float('nan'):.2f}"
            f"/{r['tagswap'].get(l, float('nan')):+.2f}" for l in BPB_LANGS if l in r["tags"]))
    print("\ndiversity on generated samples: distinct-2 / distinct-3 / repetition rate")
    for r in results:
        print(f"{r['checkpoint'][-52:]:<52}" + "".join(
            f" {l}:{r['distinct_2'].get(l, float('nan')):.2f}/{r['distinct_3'].get(l, float('nan')):.2f}"
            f"/rep:{r['repetition_rate'].get(l, float('nan')):.2f}"
            for l in BPB_LANGS if l in r["tags"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[info] wrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+")
    p.add_argument("--bpb", action="store_true",
                   help="bits/byte mode: byte and BPE checkpoints on the same val text (8 langs + English)")
    p.add_argument("--bpb-bytes", type=int, default=1_000_000, help="val text per language in --bpb mode")
    p.add_argument("--bpb-tokens", type=int, default=8192, help="tokens per eval batch in --bpb mode")
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--swap-docs", type=int, default=64, help="val docs per language for the tag-swap test")
    p.add_argument("--tag-samples", type=int, default=32)
    p.add_argument("--tag-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="out/eval_dravidian.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.bpb:
        return bpb_main(args)

    results = [evaluate(c, args) for c in args.checkpoints]
    langs = [l for l in (*LANGS, *LOWRES) if any(r.get(f"{l}_val") is not None for r in results)]
    print(f"{'checkpoint':<44} {'drav':>7} {'EN':>7} " + " ".join(f"{l:>17}" for l in langs))
    print(f"{'':<44} {'':>7} {'':>7} " + " ".join(f"{'loss/tag/swap':>17}" for _ in langs))
    for r in results:
        cells = " ".join(f"{r[f'{l}_val']:6.3f}/{r['tags'][l]['acc']:4.2f}/{r['tagswap'][l]:+5.2f}" for l in langs)
        print(f"{r['checkpoint'][-44:]:<44} {r['drav_val']:7.4f} {r.get('all_sources_val', float('nan')):7.4f} {cells}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[info] wrote {args.out}")


def _self_test():
    assert script_of("ಬೆಂಗಳೂರು news") == "Knda"
    assert script_of("தமிழ்") == "Taml"
    assert script_of("తెలుగు") == "Telu"
    assert script_of("മലയാളം ಕ") == "Mlym"
    assert script_of("ଜିସୁ") == "Orya" and script_of("येसु") == "Deva"
    assert script_of("hello") is None
    assert set(LANG_SCRIPT) == set(LANGS) | set(LOWRES) and set(SWAP) == set(LANG_SCRIPT)
    assert all(len(f"<{l}> ") == TAG_LEN for l in LANG_SCRIPT)
    x = torch.tensor([list(b"<tcy> ab"), list(b"<tcy> cd")])
    assert bytes(swap_tag(x, "tcy")[0].tolist()) == b"<kan> ab" and bytes(x[0].tolist()) == b"<tcy> ab"
    d = np.frombuffer(b"<kan> abc<tam> xy<mal> q", dtype=np.uint8).astype(np.uint16)
    b = np.array([0, 9, 17])
    assert list(doc_langs(d, b)) == ["kan", "tam", "mal"]
    ext = np.append(b, len(d))
    s = window_starts(np.random.default_rng(0), ext, np.array([1]), 2, 50, len(d) - 3)
    assert ((s >= 9) & (s < 17)).all(), s

    # Repetition and distinct tests
    assert repetition_rate("abcdefghij", n=8) == 0.0
    assert repetition_rate("a" * 200, n=8) > 0.99
    assert 0.0 < repetition_rate("abcdefgh" * 3 + "zyxwvuts", n=8) < 1.0
    assert repetition_rate("short", n=8) == 0.0
    assert repetition("abcdefghij", n=8) == 0.0
    assert repetition_rate(["abcdefghij", "short"], n=8) == 0.0
    assert repetition_rate("abcdefghijklmnop" * 3 + "zyxwvuts") > 0.0
    assert distinct_n("the cat sat on the mat", 2) == 1.0
    assert distinct_n("cat cat cat cat", 2) < 0.5
    assert distinct_n("cat cat cat cat", 3) <= 0.5
    assert distinct_n("cat cat cat cat cat", 3) < 0.5
    assert distinct_n(["the cat sat on the mat", "cat cat cat cat"], 2) == 0.5 * (1.0 + distinct_n("cat cat cat cat", 2))
    assert distinct_n("short", 2) == 1.0
    assert distinct_n("", 2) == 0.0
    print("[self-test] eval_dravidian ok")


if __name__ == "__main__":
    main()
