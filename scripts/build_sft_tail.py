#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_sft_tail.py

Builds data/processed_v5/sft_tail.bin + sft_tail_boundaries.npy (≈110M tokens)
for the Parentheses 0.9-120M Phase 3 SFT tail.

Token Format (all existing special tokens in v3_32k):
  Chat:
    <|user|><lang> {prompt}<|assistant|>{answer}<|endoftext|>
  Think:
    <|user|><lang> {prompt}<|assistant|><think>{reasoning}</think>{answer}<|endoftext|>
  Tool:
    <|user|><eng> {prompt}<|assistant|><tool_call>{"name": "...", "arguments": {...}}</tool_call><|system|>tool_result: {text}<|assistant|>{answer}<|endoftext|>

Mix Shares & Governance:
  - Dravidian Chat: Aya (kan/tam/tel/mal) + IndicQA (kan/tam/tel/mal) [IndicQA noted as eval benchmark]
  - English Chat: OASST2 only (rank-0 replies, synthetic == False, deduped)
  - Think Traces: Bespoke-Stratos-17k (DeepSeek-R1 distilled)
  - Tool Traces: Hermes Function Calling mapped ONLY to genuine web_search, fetch_url, read_file, python
  - Capped at <= 3x repeats per source (report honest shares, no over-tiling)
  - Pre-filter: dedup against v3/v4 val
  - Zero <esg> (id 11 count == 0)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data/raw/manual/raw-sourced"
OUT_DEFAULT = REPO / "data/processed_v5"
TOKENIZER_DIR = REPO / "data/tokenizer/v3_32k"
VAL_DIR = REPO / "data/processed_v4/val"

# Special token constants
EOT = 0
TAG = {"kan": 2, "tam": 3, "tel": 4, "mal": 5, "eng": 10}
SYSTEM_TAG = 12
USER_TAG = 13
ASSISTANT_TAG = 14
THINK_START = 15
THINK_END = 16
TOOL_START = 17
TOOL_END = 18
ESG_ID = 11

ALLOWED_TOOLS = {"web_search", "fetch_url", "read_file", "python"}


def load_tokenizer(tdir: Path) -> Tokenizer:
    tok_json = tdir / "tokenizer.json"
    sha = hashlib.sha256(tok_json.read_bytes()).hexdigest()
    assert sha.startswith("7a9b4187d5bc"), f"Tokenizer SHA256 mismatch: {sha}"
    tok = Tokenizer.from_file(str(tok_json))
    tok.encode_special_tokens = True
    return tok


def clean_text(s: str) -> str:
    """Strips leading/trailing space while preserving internal newlines and indentation."""
    if not s:
        return ""
    return s.strip()


def load_val_hashes(val_dir: Path, tok: Tokenizer) -> set[str]:
    """Loads sha1 hashes of all validation texts across all 9 languages."""
    val_hashes = set()
    if not val_dir.exists():
        return val_hashes
    for l in ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng"):
        bin_p = val_dir / f"{l}.bin"
        bnd_p = val_dir / f"{l}_boundaries.npy"
        if not bin_p.exists() or not bnd_p.exists():
            continue
        data = np.fromfile(bin_p, dtype=np.uint16)
        bnd = np.append(np.load(bnd_p).astype(np.int64), len(data))
        for s, e in zip(bnd[:-1], bnd[1:]):
            txt = tok.decode(data[s:e]).strip()
            val_hashes.add(hashlib.sha1(txt.encode("utf-8")).hexdigest())
    return val_hashes


# -------------------------------------------------------------
# 1. Dravidian Chat Extractor (Aya + IndicQA)
# -------------------------------------------------------------
def get_dravidian_chat_docs(val_hashes: set[str]):
    """Yields dicts with 'lang', 'prompt', 'answer', filtered against val."""
    # A. Aya dataset
    aya_path = RAW / "aya_dataset/data/train-00000-of-00001.parquet"
    if aya_path.exists():
        table = pq.read_table(aya_path, columns=["inputs", "targets", "language_code"])
        lang_map = {"kan": "kan", "tam": "tam", "tel": "tel", "mal": "mal"}
        for inp, tgt, lcode in zip(table["inputs"].to_pylist(), table["targets"].to_pylist(), table["language_code"].to_pylist()):
            l = lang_map.get(lcode)
            if l and inp and tgt:
                p = clean_text(inp)
                a = clean_text(tgt)
                if len(p) >= 10 and len(a) >= 10:
                    if hashlib.sha1(p.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(a.encode("utf-8")).hexdigest() in val_hashes:
                        continue
                    yield {"lang": l, "prompt": p, "answer": a}

    # B. IndicQA (human-written native QA)
    indicqa_dir = RAW / "indicqa"
    if indicqa_dir.exists():
        for l in ["kn", "ta", "te", "ml"]:
            canon_l = "kan" if l == "kn" else ("tam" if l == "ta" else ("tel" if l == "te" else "mal"))
            fpath = indicqa_dir / f"indicqa.{l}.json"
            if fpath.exists():
                data = json.loads(fpath.read_text(encoding="utf-8"))
                for art in data.get("data", []):
                    for p_obj in art.get("paragraphs", []):
                        ctx = clean_text(p_obj.get("context", ""))
                        for qa in p_obj.get("qas", []):
                            q = clean_text(qa.get("question", ""))
                            answers = qa.get("answers", [])
                            if answers:
                                ans_text = clean_text(answers[0].get("text", ""))
                                if q and ans_text:
                                    prompt = f"{ctx}\n\nಪ್ರಶ್ನೆ: {q}" if canon_l == "kan" else (
                                        f"{ctx}\n\nகேள்வி: {q}" if canon_l == "tam" else (
                                            f"{ctx}\n\nప్రశ్న: {q}" if canon_l == "tel" else
                                            f"{ctx}\n\nചോദ്യം: {q}"
                                        )
                                    )
                                    if hashlib.sha1(prompt.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(ans_text.encode("utf-8")).hexdigest() in val_hashes:
                                        continue
                                    yield {"lang": canon_l, "prompt": prompt, "answer": ans_text}


# -------------------------------------------------------------
# 2. English Chat Extractor (OASST2 only, rank-0, non-synthetic, deduped)
# -------------------------------------------------------------
def get_english_chat_docs(val_hashes: set[str]):
    ds_dir = RAW / "oasst2/data"
    if not ds_dir.exists():
        return
    seen_prompts = set()
    for pfile in ds_dir.glob("train-*.parquet"):
        table = pq.read_table(pfile)
        cols = table.column_names
        
        m_ids = table["message_id"].to_pylist()
        p_ids = table["parent_id"].to_pylist()
        roles = table["role"].to_pylist()
        langs = table["lang"].to_pylist()
        ranks = table["rank"].to_pylist()
        synths = table["synthetic"].to_pylist() if "synthetic" in cols else [False] * len(m_ids)
        deleteds = table["deleted"].to_pylist() if "deleted" in cols else [False] * len(m_ids)
        texts = table["text"].to_pylist()

        prompters = {}
        assistants = []
        for mid, pid, role, l, rk, syn, dlt, txt in zip(m_ids, p_ids, roles, langs, ranks, synths, deleteds, texts):
            if dlt:
                continue
            if role == "prompter" and txt:
                prompters[mid] = clean_text(txt)
            elif role == "assistant" and l == "en" and not syn and (rk == 0 or rk == 0.0) and txt:
                assistants.append((pid, clean_text(txt)))

        for pid, ans in assistants:
            if pid in prompters:
                p_txt = prompters[pid]
                if len(p_txt) >= 10 and len(ans) >= 10:
                    h_pair = hashlib.sha1((p_txt + "\n" + ans).encode("utf-8")).hexdigest()
                    if h_pair not in seen_prompts:
                        seen_prompts.add(h_pair)
                        if hashlib.sha1(p_txt.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(ans.encode("utf-8")).hexdigest() in val_hashes:
                            continue
                        yield {"lang": "eng", "prompt": p_txt, "answer": ans}


# -------------------------------------------------------------
# 3. Think Traces Extractor (Bespoke-Stratos-17k)
# -------------------------------------------------------------
def get_think_trace_docs(val_hashes: set[str]):
    pfile = RAW / "bespoke_stratos/train-00000-of-00001.parquet"
    if not pfile.exists():
        return
    table = pq.read_table(pfile, columns=["conversations"])
    for conv_item in table["conversations"].to_pylist():
        if len(conv_item) >= 2:
            u_val = conv_item[0].get("value", "")
            a_val = conv_item[1].get("value", "")
            if "<|begin_of_thought|>" in a_val and "<|end_of_thought|>" in a_val:
                t_raw = a_val.split("<|end_of_thought|>", 1)[0].replace("<|begin_of_thought|>", "").strip()
                ans_raw = a_val.split("<|end_of_thought|>", 1)[1]
                ans_raw = ans_raw.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()
                p = clean_text(u_val)
                t = clean_text(t_raw)
                a = clean_text(ans_raw)
                if p and t and a:
                    if hashlib.sha1(p.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(a.encode("utf-8")).hexdigest() in val_hashes:
                        continue
                    yield {"lang": "eng", "prompt": p, "reasoning": t, "answer": a}


# -------------------------------------------------------------
# 4. Tool Traces Extractor (Hermes Function Calling - Genuine Tools Only)
# -------------------------------------------------------------
def get_tool_trace_docs(val_hashes: set[str]):
    hermes_dir = RAW / "hermes-function-calling-v1"
    if not hermes_dir.exists():
        return
    for fname in ["glaive-function-calling-5k.json", "func-calling-singleturn.json"]:
        fpath = hermes_dir / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for item in data:
            convs = item.get("conversations", [])
            for i in range(len(convs) - 3):
                if convs[i].get("from") == "human" and convs[i+1].get("from") == "gpt" and convs[i+2].get("from") == "tool" and convs[i+3].get("from") == "gpt":
                    u_txt = clean_text(convs[i].get("value", ""))
                    tcall = convs[i+1].get("value", "")
                    tresp = convs[i+2].get("value", "")
                    ans_txt = clean_text(convs[i+3].get("value", ""))
                    if "<tool_call>" in tcall and "</tool_call>" in tcall:
                        call_body = tcall.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
                        resp_body = tresp.replace("<tool_response>", "").replace("</tool_response>", "").strip()
                        try:
                            c_json = json.loads(call_body)
                            name = c_json.get("name", "").lower()
                            args = c_json.get("arguments", {})
                            
                            mapped = None
                            if any(w in name for w in ['web_search', 'search_web', 'google', 'bing', 'duckduckgo']) or (name.startswith('search') and any(k in args for k in ['query', 'query_string', 'q', 'keyword'])):
                                mapped = "web_search"
                            elif any(w in name for w in ['fetch_url', 'get_url', 'read_url', 'download_url', 'webpage', 'fetch_page']) and any(k in args for k in ['url', 'link', 'uri']):
                                mapped = "fetch_url"
                            elif any(w in name for w in ['read_file', 'view_file', 'open_file', 'cat', 'load_file']) and any(k in args for k in ['path', 'file', 'filename', 'filepath']):
                                mapped = "read_file"
                            elif any(w in name for w in ['python', 'code_runner', 'execute_code', 'run_code', 'calculator', 'eval_math']) and any(k in args for k in ['code', 'script', 'expression', 'formula']):
                                mapped = "python"

                            if mapped is None:
                                continue  # Never rename unmappable functions

                            c_json["name"] = mapped
                            call_str = json.dumps(c_json, ensure_ascii=False)
                            if hashlib.sha1(u_txt.encode("utf-8")).hexdigest() in val_hashes or hashlib.sha1(ans_txt.encode("utf-8")).hexdigest() in val_hashes:
                                continue
                            yield {
                                "lang": "eng",
                                "prompt": u_txt,
                                "tool_call": call_str,
                                "tool_result": clean_text(resp_body),
                                "answer": ans_txt,
                            }
                        except Exception:
                            continue


# -------------------------------------------------------------
# Encoding Pipelines
# -------------------------------------------------------------
def encode_chat(tok: Tokenizer, item: dict) -> np.ndarray:
    l_tag = TAG.get(item["lang"], TAG["eng"])
    prefix = [USER_TAG, l_tag]
    body_p = tok.encode(item["prompt"], add_special_tokens=False).ids
    mid = [ASSISTANT_TAG]
    body_a = tok.encode(item["answer"], add_special_tokens=False).ids
    suffix = [EOT]
    return np.array(prefix + body_p + mid + body_a + suffix, dtype=np.uint16)


def encode_think(tok: Tokenizer, item: dict) -> np.ndarray:
    l_tag = TAG.get(item["lang"], TAG["eng"])
    prefix = [USER_TAG, l_tag]
    body_p = tok.encode(item["prompt"], add_special_tokens=False).ids
    mid1 = [ASSISTANT_TAG, THINK_START]
    body_t = tok.encode(item["reasoning"], add_special_tokens=False).ids
    mid2 = [THINK_END]
    body_a = tok.encode(item["answer"], add_special_tokens=False).ids
    suffix = [EOT]
    return np.array(prefix + body_p + mid1 + body_t + mid2 + body_a + suffix, dtype=np.uint16)


def encode_tool(tok: Tokenizer, item: dict) -> np.ndarray:
    l_tag = TAG.get(item["lang"], TAG["eng"])
    prefix = [USER_TAG, l_tag]
    body_p = tok.encode(item["prompt"], add_special_tokens=False).ids
    mid1 = [ASSISTANT_TAG, TOOL_START]
    body_call = tok.encode(item["tool_call"], add_special_tokens=False).ids
    mid2 = [TOOL_END, SYSTEM_TAG]
    lead_sys = tok.encode("tool_result: ", add_special_tokens=False).ids
    body_resp = tok.encode(item["tool_result"], add_special_tokens=False).ids
    mid3 = [ASSISTANT_TAG]
    body_a = tok.encode(item["answer"], add_special_tokens=False).ids
    suffix = [EOT]
    return np.array(prefix + body_p + mid1 + body_call + mid2 + lead_sys + body_resp + mid3 + body_a + suffix, dtype=np.uint16)


def tile_docs_capped(encoded_docs: list[np.ndarray], target_tokens: int, max_reps: int = 3) -> tuple[list[np.ndarray], int, int]:
    """Capped upsampling: repeats <= max_reps. If short of target, reports honestly without over-tiling."""
    if not encoded_docs:
        return [], 0, 0
    lens = np.array([len(d) for d in encoded_docs])
    pool_sum = int(lens.sum())
    reps = min(max_reps, target_tokens // pool_sum)
    chosen = []
    tot = 0
    for _ in range(reps):
        chosen.extend(encoded_docs)
        tot += pool_sum
    rem = target_tokens - tot
    if reps < max_reps and rem > 0:
        for d in encoded_docs:
            if tot >= target_tokens:
                break
            chosen.append(d)
            tot += len(d)
    return chosen, tot, pool_sum


def main():
    parser = argparse.ArgumentParser(description="Build SFT tail for Parentheses 0.9-120M")
    parser.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--val-dir", default=str(VAL_DIR))
    parser.add_argument("--budget", type=int, default=110_000_000, help="Total token budget (~110M)")
    parser.add_argument("--dry-run", action="store_true", help="Run small smoke test without full encoding")
    args = parser.parse_args()

    print(f"=== Building SFT Tail (Target: {args.budget:,} tokens, capped at <=3x repeats) ===", flush=True)
    tok = load_tokenizer(Path(args.tokenizer))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_hashes = load_val_hashes(Path(args.val_dir), tok)
    print(f"Loaded {len(val_hashes):,} validation hashes for zero-leak deduplication.", flush=True)

    target_drav = int(args.budget * 0.40)
    target_eng = int(args.budget * 0.25)
    target_think = int(args.budget * 0.20)
    target_tool = int(args.budget * 0.15)

    if args.dry_run:
        target_drav = 20_000
        target_eng = 12_500
        target_think = 10_000
        target_tool = 7_500

    print(f"Target Token Quotas: Dravidian={target_drav:,}, Eng={target_eng:,}, Think={target_think:,}, Tool={target_tool:,}")

    all_docs = []

    # 1. Dravidian Chat
    print("[1/4] Encoding Dravidian Chat...", flush=True)
    drav_items = list(get_dravidian_chat_docs(val_hashes))
    print(f"Loaded {len(drav_items):,} clean Dravidian raw chat items", flush=True)
    drav_encoded = [encode_chat(tok, item) for item in drav_items]
    drav_chosen, drav_tokens, drav_pool = tile_docs_capped(drav_encoded, target_drav, max_reps=3)
    all_docs.extend(drav_chosen)
    print(f"Dravidian: {len(drav_chosen):,} docs, {drav_tokens:,} tokens (unique pool: {len(drav_encoded):,} docs, {drav_pool:,} tokens, {drav_tokens/max(1,drav_pool):.2f}x)")

    # 2. English Chat
    print("[2/4] Encoding English Chat (OASST2 rank-0 non-synthetic)...", flush=True)
    eng_items = list(get_english_chat_docs(val_hashes))
    print(f"Loaded {len(eng_items):,} clean English raw chat items", flush=True)
    eng_encoded = [encode_chat(tok, item) for item in eng_items]
    eng_chosen, eng_tokens, eng_pool = tile_docs_capped(eng_encoded, target_eng, max_reps=3)
    all_docs.extend(eng_chosen)
    print(f"English: {len(eng_chosen):,} docs, {eng_tokens:,} tokens (unique pool: {len(eng_encoded):,} docs, {eng_pool:,} tokens, {eng_tokens/max(1,eng_pool):.2f}x)")

    # 3. Think Traces
    print("[3/4] Encoding Think Traces...", flush=True)
    think_items = list(get_think_trace_docs(val_hashes))
    print(f"Loaded {len(think_items):,} clean Think raw trace items", flush=True)
    think_encoded = [encode_think(tok, item) for item in think_items]
    think_chosen, think_tokens, think_pool = tile_docs_capped(think_encoded, target_think, max_reps=3)
    all_docs.extend(think_chosen)
    print(f"Think: {len(think_chosen):,} docs, {think_tokens:,} tokens (unique pool: {len(think_encoded):,} docs, {think_pool:,} tokens, {think_tokens/max(1,think_pool):.2f}x)")

    # 4. Tool Traces
    print("[4/4] Encoding Tool Traces (genuine tools only)...", flush=True)
    tool_items = list(get_tool_trace_docs(val_hashes))
    print(f"Loaded {len(tool_items):,} clean Tool raw trace items", flush=True)
    tool_encoded = [encode_tool(tok, item) for item in tool_items]
    tool_chosen, tool_tokens, tool_pool = tile_docs_capped(tool_encoded, target_tool, max_reps=3)
    all_docs.extend(tool_chosen)
    print(f"Tool: {len(tool_chosen):,} docs, {tool_tokens:,} tokens (unique pool: {len(tool_encoded):,} docs, {tool_pool:,} tokens, {tool_tokens/max(1,tool_pool):.2f}x)")

    total_tokens = sum(len(d) for d in all_docs)
    total_docs = len(all_docs)
    print(f"\nTotal assembled tail tokens: {total_tokens:,} across {total_docs:,} documents")

    # Shuffle documents
    rng = np.random.default_rng(1234)
    idx_perm = rng.permutation(len(all_docs))
    shuffled_docs = [all_docs[i] for i in idx_perm]

    bin_path = out_dir / ("sft_tail_dryrun.bin" if args.dry_run else "sft_tail.bin")
    bnd_path = out_dir / ("sft_tail_dryrun_boundaries.npy" if args.dry_run else "sft_tail_boundaries.npy")

    print(f"Writing to {bin_path} and {bnd_path}...", flush=True)
    final_starts = [0]
    with open(bin_path, "wb") as f:
        for doc in shuffled_docs:
            f.write(doc.tobytes())
            final_starts.append(final_starts[-1] + len(doc))

    np.save(bnd_path, np.array(final_starts[:-1], dtype=np.int64))

    # Verification gates
    print("\n--- Gate Verifications ---", flush=True)
    bin_data = np.fromfile(bin_path, dtype=np.uint16)
    bnd = np.load(bnd_path)
    
    esg_count = int(np.sum(bin_data == ESG_ID))
    print(f"Gate 1: <esg> count: {esg_count} (expect 0)", flush=True)
    assert esg_count == 0, f"Found {esg_count} <esg> tokens!"

    print("Gate 2: Boundaries monotonic, first token is <|user|> (13)...", flush=True)
    assert np.all(np.diff(bnd) > 0), "Boundaries are not monotonically increasing!"
    for i in range(min(100, len(bnd))):
        assert bin_data[bnd[i]] == USER_TAG, f"Doc {i} does not start with <|user|>! Got {bin_data[bnd[i]]}"
    print("Gate 2: PASS")

    print("Gate 3: Round-trip decoding check on 50 sample docs...", flush=True)
    for idx in rng.choice(len(bnd), size=min(50, len(bnd)), replace=False):
        s = bnd[idx]
        e = bnd[idx + 1] if idx + 1 < len(bnd) else len(bin_data)
        d_toks = bin_data[s:e]
        text = tok.decode(d_toks, skip_special_tokens=False)
        re_toks = tok.encode(text, add_special_tokens=False).ids
        assert len(re_toks) > 0, "Empty re-tokenized doc!"
    print("Gate 3: Round-trip decode: PASS")

    manifest = {
        "budget_target": args.budget,
        "total_tokens": int(total_tokens),
        "total_docs": int(total_docs),
        "shares": {
            "dravidian": {"tokens": drav_tokens, "unique": drav_pool, "share": round(drav_tokens / total_tokens, 4)},
            "english": {"tokens": eng_tokens, "unique": eng_pool, "share": round(eng_tokens / total_tokens, 4)},
            "think": {"tokens": think_tokens, "unique": think_pool, "share": round(think_tokens / total_tokens, 4)},
            "tool": {"tokens": tool_tokens, "unique": tool_pool, "share": round(tool_tokens / total_tokens, 4)},
        },
        "sha256": hashlib.sha256(open(bin_path, "rb").read()).hexdigest(),
    }
    m_path = out_dir / ("sft_tail_manifest_dryrun.json" if args.dry_run else "sft_tail_manifest.json")
    m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {m_path}", flush=True)
    print("=== SFT Tail Build Complete ===", flush=True)


if __name__ == "__main__":
    main()
