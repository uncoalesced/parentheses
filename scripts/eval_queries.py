"""
Phase 3 query eval: run a fixed JSONL query set through a BPE checkpoint and
score each answer with cheap automatic metrics. LLM/human judging reads the
output JSONL afterwards.

    python scripts/eval_queries.py --checkpoint checkpoints/parentheses-0.9-41M-v4/step_16083_final.pt \
        --queries eval/queries_v1.jsonl --out out/queries_v1_base.jsonl --format base
    python scripts/eval_queries.py --validate eval/queries_v1.jsonl
    python scripts/eval_queries.py --self-test

Query line: {"id", "lang", "bucket", "prompt", "max_new_tokens"?, "reference"?}.
Output line: the query plus {"output", "n_tokens", "stop", "script", "script_ok",
"repetition_rate", "distinct_2", "seconds"}. Appends one line per query and skips
ids already in --out, so a killed run resumes where it stopped.

--format base: <lang> + prompt (pretrained model, continuation).
--format chat: <|user|> <lang> prompt <|assistant|> (after SFT).
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_dravidian import LANG_SCRIPT, distinct_n, repetition_rate, script_of  # noqa: E402

LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng")
BUCKETS = ("lang_gen", "english", "reasoning", "instruction", "tool", "code", "freethink", "probe_hin")
STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>")
FIELDS = {"id", "lang", "bucket", "prompt", "max_new_tokens", "reference", "notes"}


def looping(ids: list[int], window: int = 32, repeats: int = 4) -> bool:
    """True once the last `window` tokens have appeared `repeats` times back to back."""
    need = window * repeats
    if len(ids) < need:
        return False
    tail = ids[-need:]
    first = tail[:window]
    return all(tail[i * window:(i + 1) * window] == first for i in range(1, repeats))


def validate(path: str) -> list[str]:
    """Problems in a query file (empty list = valid)."""
    errs, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"line {n}: bad JSON ({e})")
                continue
            extra = set(q) - FIELDS
            if extra:
                errs.append(f"line {n}: unknown fields {sorted(extra)}")
            for k in ("id", "lang", "bucket", "prompt"):
                if not isinstance(q.get(k), str) or not q.get(k).strip():
                    errs.append(f"line {n}: missing/empty {k}")
            if q.get("id") in seen:
                errs.append(f"line {n}: duplicate id {q.get('id')}")
            seen.add(q.get("id"))
            if q.get("lang") not in LANGS + ("hin",):
                errs.append(f"line {n}: lang {q.get('lang')!r} not in {LANGS + ('hin',)}")
            if q.get("bucket") not in BUCKETS:
                errs.append(f"line {n}: bucket {q.get('bucket')!r} not in {BUCKETS}")
            mnt = q.get("max_new_tokens", 256)
            if not isinstance(mnt, int) or not 1 <= mnt <= 2048:
                errs.append(f"line {n}: max_new_tokens must be int 1..2048")
            want = LANG_SCRIPT.get(q.get("lang"))
            prompt = q.get("prompt") or ""
            if want and q.get("bucket") == "lang_gen" and script_of(prompt) != want:
                errs.append(f"line {n}: {q.get('lang')} lang_gen prompt is not in {want} script")
    if len(seen) < 500:
        errs.append(f"only {len(seen)} queries (need 500+)")
    return errs


def prompt_ids(tok, q: dict, fmt: str) -> list[int]:
    tag = tok.token_to_id(f"<{q['lang']}>")
    body = tok.encode(q["prompt"], add_special_tokens=False).ids
    lead = [tag] if tag is not None else []
    if fmt == "chat":
        return [tok.token_to_id("<|user|>")] + lead + body + [tok.token_to_id("<|assistant|>")]
    return lead + body


def score(q: dict, text: str) -> dict:
    want = LANG_SCRIPT.get(q["lang"])
    got = script_of(text)
    if q["lang"] == "eng":
        ok = got is None and any(c.isascii() and c.isalpha() for c in text)
    else:
        ok = None if want is None else got == want
    return {"script": got, "script_ok": ok, "repetition_rate": repetition_rate(text),
            "distinct_2": distinct_n(text, 2)}


@torch.no_grad()
def answer(model, tok, q: dict, fmt: str, device: str, temperature: float, top_k: int) -> dict:
    stops = {tok.token_to_id(s) for s in STOP_TOKENS}
    idx = torch.tensor([prompt_ids(tok, q, fmt)], dtype=torch.long, device=device)
    out, stop, t0 = [], "max_tokens", time.perf_counter()
    for nxt in model.stream(idx, q.get("max_new_tokens", 256), temperature, top_k):
        i = int(nxt)
        if i in stops:
            stop = "eos"
            break
        out.append(i)
        if looping(out):
            stop = "loop"
            break
    text = tok.decode(out, skip_special_tokens=False)
    return {"output": text, "n_tokens": len(out), "stop": stop,
            "seconds": round(time.perf_counter() - t0, 3), **score(q, text)}


def run(args):
    from tokenizers import Tokenizer

    from model import Parentheses

    errs = validate(args.queries)
    if errs:
        sys.exit("query file invalid:\n  " + "\n  ".join(errs[:20]))
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_file(os.path.join(ckpt["tokenizer"]["path"], "tokenizer.json"))
    tok.encode_special_tokens = True  # tags in prompt text stay text; specials go in by id

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    todo = [q for q in queries if q["id"] not in done]
    print(f"[info] {len(done)} done, {len(todo)} to run -> {args.out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for n, q in enumerate(todo, 1):
        torch.manual_seed(args.seed)  # same seed per query: reruns and before/after compare cleanly
        rec = {**q, "checkpoint": args.checkpoint, "format": args.format,
               **answer(model, tok, q, args.format, args.device, args.temperature, args.top_k)}
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 25 == 0:
            print(f"[info] {n}/{len(todo)}")
    summarize(args.out)


def summarize(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"{'lang':<5} {'n':>4} {'script_ok':>9} {'eos':>5} {'loop':>5} {'rep':>6} {'d2':>5}")
    for lang in sorted({r["lang"] for r in rows}):
        rs = [r for r in rows if r["lang"] == lang]
        oks = [r["script_ok"] for r in rs if r["script_ok"] is not None]
        print(f"{lang:<5} {len(rs):>4} {(sum(oks) / len(oks) if oks else float('nan')):>9.2f} "
              f"{sum(r['stop'] == 'eos' for r in rs) / len(rs):>5.2f} "
              f"{sum(r['stop'] == 'loop' for r in rs) / len(rs):>5.2f} "
              f"{sum(r['repetition_rate'] for r in rs) / len(rs):>6.3f} "
              f"{sum(r['distinct_2'] for r in rs) / len(rs):>5.2f}")


def _self_test():
    import tempfile

    assert not looping([1, 2, 3] * 3)
    assert looping([9] + [1, 2] * 4, window=2, repeats=4)
    assert not looping([1, 2] * 3 + [1, 3], window=2, repeats=4)
    assert looping(list(range(32)) * 4)

    assert score({"lang": "kan"}, "ಬೆಂಗಳೂರು")["script_ok"] is True
    assert score({"lang": "kan"}, "தமிழ்")["script_ok"] is False
    assert score({"lang": "eng"}, "hello there")["script_ok"] is True
    assert score({"lang": "eng"}, "ಬೆಂಗಳೂರು")["script_ok"] is False

    good = [{"id": f"q{i}", "lang": "kan", "bucket": "lang_gen", "prompt": "ಬೆಂಗಳೂರು ನಗರ"} for i in range(500)]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "q.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in good)
        assert validate(p) == [], validate(p)
        bad = good[:499] + [{"id": "q0", "lang": "xx", "bucket": "lang_gen", "prompt": "hello", "x": 1}]
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in bad)
        errs = " ".join(validate(p))
        for want in ("duplicate id", "lang 'xx'", "unknown fields", "only 499"):
            assert want in errs, (want, errs)

    class FakeTok:
        ids = {"<kan>": 2, "<|user|>": 13, "<|assistant|>": 14}

        def token_to_id(self, s):
            return self.ids.get(s)

        def encode(self, s, add_special_tokens=False):
            return type("E", (), {"ids": [100, 101]})()

    q = {"lang": "kan", "prompt": "x"}
    assert prompt_ids(FakeTok(), q, "base") == [2, 100, 101]
    assert prompt_ids(FakeTok(), q, "chat") == [13, 2, 100, 101, 14]
    print("[self-test] eval_queries ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--queries", default="eval/queries_v1.jsonl")
    p.add_argument("--out")
    p.add_argument("--format", choices=("base", "chat"), default="base")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--validate", metavar="QUERIES")
    p.add_argument("--summary", metavar="OUT_JSONL")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return _self_test()
    if args.validate:
        errs = validate(args.validate)
        print("\n".join(errs) if errs else f"[ok] {args.validate} valid")
        sys.exit(1 if errs else 0)
    if args.summary:
        return summarize(args.summary)
    if not (args.checkpoint and args.out):
        p.error("--checkpoint and --out are required for a run")
    run(args)


if __name__ == "__main__":
    main()
