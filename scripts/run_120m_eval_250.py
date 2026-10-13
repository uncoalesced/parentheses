"""
Run 250 evaluation queries on Parentheses 0.9-120M step_1601_final.pt.
Stratified distribution:
- 100 English prompts
- 150 Other language prompts (proportional to token size trained on: kan 42, tam 30, tel 30, mal 30, tcy 6, hin 4, kff 3, kxv 3, nit 2)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_dravidian import LANG_SCRIPT, distinct_n, repetition_rate, script_of
from model import Parentheses
from tokenizers import Tokenizer

STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>")


def looping(ids: list[int], window: int = 32, repeats: int = 4) -> bool:
    need = window * repeats
    if len(ids) < need:
        return False
    tail = ids[-need:]
    first = tail[:window]
    return all(tail[i * window:(i + 1) * window] == first for i in range(1, repeats))


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
    return {
        "script": got,
        "script_ok": ok,
        "repetition_rate": repetition_rate(text),
        "distinct_2": distinct_n(text, 2),
    }


@torch.no_grad()
def answer(model, tok, q: dict, fmt: str, device: str, temperature: float, top_k: int,
           min_p: float | None = None, max_cap: int = 256) -> dict:
    stops = {tok.token_to_id(s) for s in STOP_TOKENS}
    idx = torch.tensor([prompt_ids(tok, q, fmt)], dtype=torch.long, device=device)
    out, stop, t0 = [], "max_tokens", time.perf_counter()
    # 256 keeps s250 runs comparable across sessions (11 queries ask for 1024); reason250 code uses --max-cap 320
    max_tokens = min(q.get("max_new_tokens", 256), max_cap)
    for nxt in model.stream(idx, max_tokens, temperature, top_k, min_p):
        i = int(nxt)
        if i in stops:
            stop = "eos"
            break
        out.append(i)
        if looping(out):
            stop = "loop"
            break
    text = tok.decode(out, skip_special_tokens=False)
    return {
        "output": text,
        "n_tokens": len(out),
        "stop": stop,
        "seconds": round(time.perf_counter() - t0, 3),
        **score(q, text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/parentheses-0.9-120M/step_1601_final.pt")
    parser.add_argument("--queries", default="eval/queries_120m_s250.jsonl")
    parser.add_argument("--out", default="out/queries_120m_s250_final.jsonl")
    parser.add_argument("--format", default="base", choices=["base", "chat"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--max-cap", type=int, default=256, help="upper bound on each query's max_new_tokens")
    parser.add_argument("--limit-per-lang", type=int, default=0,
                        help="first N queries of each language (decoding-sweep subset); 0 = all")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    print(f"Loading checkpoint: {args.checkpoint} on {args.device}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    tok_dir = ckpt["tokenizer"]["path"]
    tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
    tok.encode_special_tokens = True

    # Read queries
    with open(args.queries, "r", encoding="utf-8") as f:
        queries = subset([json.loads(line) for line in f if line.strip()], args.limit_per_lang)

    # Resume support
    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            done_ids = {json.loads(line)["id"] for line in f if line.strip()}

    todo = [q for q in queries if q["id"] not in done_ids]
    print(f"Total queries: {len(queries)} ({len(done_ids)} already done, {len(todo)} to run)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t_start = time.time()
    
    for idx, q in enumerate(todo, 1):
        torch.manual_seed(args.seed + idx)
        ans = answer(model, tok, q, args.format, args.device, args.temperature, args.top_k, args.min_p, args.max_cap)
        rec = {
            **q,
            "checkpoint": args.checkpoint,
            "format": args.format,
            "decode": {"temperature": args.temperature, "top_k": args.top_k, "min_p": args.min_p},
            **ans,
        }
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        if idx % 10 == 0 or idx == len(todo):
            elapsed = time.time() - t_start
            rate = idx / elapsed
            eta = (len(todo) - idx) / rate if rate > 0 else 0
            print(f"[{idx}/{len(todo)}] Done: {q['id']} ({q['lang']}) | {ans['n_tokens']} tok in {ans['seconds']}s | ETA: {eta/60:.1f}m")

    # Clean up GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll {len(queries)} queries completed in {time.time()-t_start:.1f}s.")
    print("\n--- Summary by Language ---")
    summarize(args.out)


def subset(queries: list[dict], per_lang: int) -> list[dict]:
    """First per_lang queries of each language, file order kept (deterministic sweep subset)."""
    if not per_lang:
        return queries
    seen: dict[str, int] = {}
    out = []
    for q in queries:
        seen[q["lang"]] = seen.get(q["lang"], 0) + 1
        if seen[q["lang"]] <= per_lang:
            out.append(q)
    return out


def self_test():
    assert looping([1, 2] * 64) and not looping(list(range(200)))
    qs = [{"id": i, "lang": l} for i, l in enumerate("aabbbcaaa")]
    assert [q["id"] for q in subset(qs, 2)] == [0, 1, 2, 3, 5]
    assert subset(qs, 0) == qs
    assert score({"lang": "eng"}, "hello world")["script_ok"] is True
    assert score({"lang": "kan"}, "hello")["script_ok"] is False
    print("[self-test] run_120m_eval_250 ok")


def summarize(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"{'lang':<5} {'n':>4} {'script_ok':>9} {'eos':>5} {'loop':>5} {'rep':>6} {'d2':>5} {'avg_sec':>7}")
    for lang in sorted({r["lang"] for r in rows}):
        rs = [r for r in rows if r["lang"] == lang]
        oks = [r["script_ok"] for r in rs if r["script_ok"] is not None]
        avg_s = sum(r.get("seconds", 0) for r in rs) / len(rs) if rs else 0
        print(f"{lang:<5} {len(rs):>4} {(sum(oks) / len(oks) if oks else float('nan')):>9.2f} "
              f"{sum(r['stop'] == 'eos' for r in rs) / len(rs):>5.2f} "
              f"{sum(r['stop'] == 'loop' for r in rs) / len(rs):>5.2f} "
              f"{sum(r['repetition_rate'] for r in rs) / len(rs):>6.3f} "
              f"{sum(r['distinct_2'] for r in rs) / len(rs):>5.2f} "
              f"{avg_s:>7.2f}s")


if __name__ == "__main__":
    main()
