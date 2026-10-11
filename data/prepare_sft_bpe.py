r"""
BPE chat SFT set for Parentheses 0.9 (v3_32k tokenizer), for train.py --sft-spans.

data/prepare_sft.py is the byte-level version (\x01role\x02text\x03 delimiters);
this one uses the tokenizer's reserved chat specials instead:

    <|user|><lang> prompt<|assistant|>answer<|endoftext|>      (turns repeat for multi-turn)

The mask is 1 on assistant tokens and the closing <|endoftext|>, 0 elsewhere,
element-aligned with train.bin (train.py reads mask[j] = "score predicting token j").

Sources (all Apache-2.0, SOURCE.txt in each folder under data/raw/manual/raw-sourced/):
  aya_dataset  kan/tam/tel/mal/eng prompt/completion rows (human-written)
  oasst2       English trees; top-ranked (rank 0) reply path, human turns only
  persona      data/raw/manual/personality-spec/voice-and-persona.md, verbatim, fully
               masked as plain docs (Joel's own writing)

    python data/prepare_sft_bpe.py --out data/processed_sft_v1
    python data/prepare_sft_bpe.py --self-test
"""

import argparse
import gzip
import hashlib
import json
import os
import random

import numpy as np

RAW = "data/raw/manual/raw-sourced"
PERSONA = "data/raw/manual/personality-spec/voice-and-persona.md"
DRAV = ("kan", "tam", "tel", "mal")


def encode_chat(tok, lang: str, turns: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """[(user, assistant), ...] -> (ids, mask). The lang tag opens the first user turn only."""
    sid = tok.token_to_id
    ids, mask = [], []
    for n, (user, asst) in enumerate(turns):
        u = [sid("<|user|>")] + ([sid(f"<{lang}>")] if n == 0 else []) + \
            tok.encode(user, add_special_tokens=False).ids + [sid("<|assistant|>")]
        a = tok.encode(asst, add_special_tokens=False).ids
        ids += u + a
        mask += [0] * len(u) + [1] * len(a)
    ids.append(sid("<|endoftext|>"))
    mask.append(1)
    return ids, mask


def encode_plain(tok, lang: str, text: str) -> tuple[list[int], list[int]]:
    ids = [tok.token_to_id(f"<{lang}>")] + tok.encode(text, add_special_tokens=False).ids + \
          [tok.token_to_id("<|endoftext|>")]
    return ids, [0] + [1] * (len(ids) - 1)


def aya_rows(path: str):
    import pandas as pd
    d = pd.read_parquet(path, columns=["inputs", "targets", "language_code"])
    for r in d[d.language_code.isin(DRAV + ("eng",))].itertuples(index=False):
        yield r.language_code, [(r.inputs.strip(), r.targets.strip())]


def oasst_rows(path: str):
    """English trees: follow the rank-0 reply at every level, stop at the first
    synthetic/deleted message or at a dangling user turn."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)["prompt"]
            if node.get("lang") != "en":
                continue
            turns, user = [], None
            while node and not node.get("deleted") and not node.get("synthetic"):
                if node["role"] == "prompter":
                    user = node["text"].strip()
                else:
                    turns.append((user, node["text"].strip()))
                kids = [k for k in node.get("replies") or [] if not k.get("deleted") and not k.get("synthetic")]
                ranked = [k for k in kids if k.get("rank") is not None]
                node = min(ranked, key=lambda k: k["rank"]) if ranked else (kids[0] if kids else None)
            if turns:
                yield "eng", turns


def text_key(turns) -> str:
    return hashlib.sha1("\n".join(u + "\n" + a for u, a in turns).encode("utf-8")).hexdigest()


def build(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    tok.encode_special_tokens = True  # prompt text never turns into specials

    convs = list(aya_rows(f"{RAW}/aya_dataset/data/train-00000-of-00001.parquet"))
    convs += list(oasst_rows(f"{RAW}/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz"))
    seen, uniq = set(), []
    for lang, turns in convs:
        k = text_key(turns)
        if k not in seen and all(u and a for u, a in turns):
            seen.add(k)
            uniq.append((lang, turns))
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val, train = uniq[:n_val], uniq[n_val:]

    enc = lambda rows: [(lang, *encode_chat(tok, lang, turns)) for lang, turns in rows]
    train_e, val_e = enc(train), enc(val)

    # Mix: Dravidian >= target share by tokens. Upsample the thinnest Dravidian
    # language first (<= max_epochs each), then cap English to fit.
    drav = [r for r in train_e if r[0] in DRAV]
    eng = [r for r in train_e if r[0] == "eng"]
    base = {l: sum(len(r[1]) for r in drav if r[0] == l) for l in DRAV}
    eng_tok = sum(len(r[1]) for r in eng)
    # thin languages (kan has ~330 Aya rows) always get the max upsample
    reps = {l: args.max_epochs if base[l] < 0.05 * (sum(base.values()) + eng_tok) else 1 for l in DRAV}
    target = args.drav_share / (1 - args.drav_share) * eng_tok
    while sum(base[l] * reps[l] for l in DRAV) < target:
        thin = [l for l in DRAV if reps[l] < args.max_epochs]
        if not thin:
            break
        reps[min(thin, key=lambda l: base[l] * reps[l])] += 1
    d_tok = sum(base[l] * reps[l] for l in DRAV)
    eng_cap = int(d_tok * (1 - args.drav_share) / args.drav_share)
    eng_kept, t = [], 0
    for r in eng:
        if t + len(r[1]) > eng_cap:
            break
        eng_kept.append(r)
        t += len(r[1])
    mix = [r for r in drav for _ in range(reps[r[0]])] + eng_kept
    persona = open(PERSONA, encoding="utf-8").read()
    mix += [("eng", *encode_plain(tok, "eng", persona))] * args.persona_copies
    rng.shuffle(mix)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"tokenizer": args.tokenizer, "seed": args.seed,
                "format": "<|user|><lang> p<|assistant|>a<|endoftext|>",
                "sources": {"aya_dataset": "Apache-2.0", "oasst2": "Apache-2.0", "persona": "Joel's own writing"},
                "upsample": reps, "persona_copies": args.persona_copies, "splits": {}, "checksums_sha256": {}}
    for split, rows in (("train", mix), ("val", val_e)):
        ids = np.fromiter((i for r in rows for i in r[1]), dtype=np.uint16)
        mask = np.fromiter((m for r in rows for m in r[2]), dtype=np.uint8)
        starts = np.cumsum([0] + [len(r[1]) for r in rows[:-1]]).astype(np.int64)
        for name, arr in ((f"{split}.bin", ids), (f"{split}.mask.bin", mask), (f"{split}_boundaries.npy", starts)):
            p = os.path.join(args.out, name)
            np.save(p, arr) if name.endswith(".npy") else arr.tofile(p)
            manifest["checksums_sha256"][name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        per = {}
        for r in rows:
            per.setdefault(r[0], [0, 0])
            per[r[0]][0] += 1
            per[r[0]][1] += len(r[1])
        tot = sum(v[1] for v in per.values())
        manifest["splits"][split] = {"docs": len(rows), "tokens": tot, "supervised": int(mask.sum()),
                                     "langs": {l: {"docs": v[0], "tokens": v[1], "share": round(v[1] / tot, 4)}
                                               for l, v in sorted(per.items())}}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest["splits"], indent=1), "\nupsample:", reps)


def _self_test():
    class Tok:
        sp = {"<|endoftext|>": 0, "<|user|>": 13, "<|assistant|>": 14, "<kan>": 2, "<eng>": 10}

        def token_to_id(self, s):
            return self.sp[s]

        def encode(self, s, add_special_tokens=False):
            return type("E", (), {"ids": [100 + ord(c) % 50 for c in s]})()

    t = Tok()
    ids, mask = encode_chat(t, "kan", [("ab", "xyz"), ("c", "de")])
    want = [13, 2, *t.encode("ab").ids, 14, *t.encode("xyz").ids, 13, *t.encode("c").ids, 14, *t.encode("de").ids, 0]
    assert ids == want, ids
    assert mask == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], mask
    ids, mask = encode_plain(t, "eng", "hi")
    assert ids[0] == 10 and ids[-1] == 0 and mask == [0, 1, 1, 1]
    assert text_key([("a", "b")]) != text_key([("a", "c")])
    print("[self-test] prepare_sft_bpe ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed_sft_v1")
    p.add_argument("--tokenizer", default="data/tokenizer/v3_32k")
    p.add_argument("--drav-share", type=float, default=0.6)
    p.add_argument("--max-epochs", type=int, default=3, help="max Dravidian upsample per language")
    p.add_argument("--persona-copies", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return _self_test()
    build(args)


if __name__ == "__main__":
    main()
