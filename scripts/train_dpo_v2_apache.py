# Engineered by uncoalesced
"""
Parentheses 1.0-240M (slab_v4) - Session 8 On-Policy DPO v2 + Memory-Pinned Reference + SWA-3
100% Apache-2.0 Compliant Preference Alignment Pipeline for 8GB VRAM (RTX 5050).

Modes:
  1. --mode generate       : Sample N=4 on-policy completions from SFT checkpoint & score pairs
  2. --mode precompute-ref : Compute & pin reference log-probs to disk (.npy), freeing 50% VRAM
  3. --mode train          : Run DPO + 0.1 * NLL anchor loss with single model in VRAM
  4. --mode swa3           : Merge final 3 checkpoint anchors into SWA-3 golden weights
  5. --mode all            : Execute precompute-ref -> train -> swa3 end-to-end
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(r"E:\Parentheses\parentheses-0.9")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ModelConfig
from model_slab_v4 import SlabV4Model
from scripts.license_guard import verify_apache_manifest

# Special Token & Formatting Constants
EOT_TOKEN_ID = 0
PAD_TOKEN_ID = 1  # <|pad|>
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Unicode Script Ranges for Dravidian Purity Verification
DRAVIDIAN_RANGES: Dict[str, Tuple[int, int]] = {
    "kn": (0x0C80, 0x0CFF),  # Kannada ( & Tulu in KN script)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "ml": (0x0D00, 0x0D7F),  # Malayalam
}


# ---------------------------------------------------------------------------
# 1. Deterministic On-Policy Candidate Verifiers (Math, <think>, Dravidian, Abstention)
# ---------------------------------------------------------------------------

def has_ngram_loop(token_ids: List[int], n: int = 4, max_repeats: int = 3) -> bool:
    if len(token_ids) < n * max_repeats:
        return False
    counts: Dict[Tuple[int, ...], int] = {}
    for i in range(len(token_ids) - n + 1):
        gram = tuple(token_ids[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= max_repeats:
            return True
    return False


def script_purity_score(text: str, lang: str) -> float:
    """Checks that expected Dravidian script characters dominate non-ASCII alpha chars."""
    if lang not in DRAVIDIAN_RANGES:
        return 1.0
    lo, hi = DRAVIDIAN_RANGES[lang]
    # Strip <think>...</think> block if reasoning was done in English/bilingual
    ans_part = text.split(THINK_CLOSE)[-1] if THINK_CLOSE in text else text
    non_ascii_alpha = [ch for ch in ans_part if ord(ch) > 127 and ch.isalpha()]
    if not non_ascii_alpha:
        return 0.0
    in_script = sum(1 for ch in non_ascii_alpha if lo <= ord(ch) <= hi)
    return float(in_script) / float(len(non_ascii_alpha))


def extract_last_number(text: str) -> Optional[str]:
    ans_part = text.split(THINK_CLOSE)[-1] if THINK_CLOSE in text else text
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", ans_part)
    if not matches:
        return None
    return matches[-1].replace(",", "")


# Prompt files use the repo's 3-letter tags (kan/tam/tel/mal); the purity table uses 2-letter keys.
LANG_ALIASES: Dict[str, str] = {"kan": "kn", "tam": "ta", "tel": "te", "mal": "ml", "tcy": "kn"}


def score_on_policy_candidate(
    text: str,
    token_ids: List[int],
    task_type: str = "general",
    lang: str = "en",
    ground_truth_num: Optional[str] = None,
    requires_think: bool = False,
    requires_abstention: bool = False,
    ground_truth_text: Optional[str] = None,
) -> float:
    """
    Deterministic verifier reward in [-2.0, +2.0] for ranking on-policy completions.
    """
    lang = LANG_ALIASES.get(lang, lang)
    score = 0.0

    # Reading-QA gold answer (Belebele rows): reward answers containing most of the gold option's words,
    # so script purity alone cannot win a pair.
    if ground_truth_text:
        gold = set(re.findall(r"\w+", ground_truth_text.lower()))
        ans = set(re.findall(r"\w+", (text.split(THINK_CLOSE)[-1] if THINK_CLOSE in text else text).lower()))
        if gold and len(gold & ans) / len(gold) >= 0.6:
            score += 0.75
        else:
            score -= 0.25

    # 1. Repetition & Termination Check
    if has_ngram_loop(token_ids, n=4, max_repeats=3):
        score -= 1.25
    if len(token_ids) > 0 and token_ids[-1] == EOT_TOKEN_ID:
        score += 0.25
    else:
        score -= 0.30  # Truncated without EOT

    # 2. <think> Tag Closure & Hygiene
    n_open = text.count(THINK_OPEN)
    n_close = text.count(THINK_CLOSE)
    if requires_think:
        if n_open == 1 and n_close == 1 and text.find(THINK_OPEN) < text.find(THINK_CLOSE):
            think_body = text[text.find(THINK_OPEN) + len(THINK_OPEN) : text.find(THINK_CLOSE)].strip()
            ans_body = text[text.find(THINK_CLOSE) + len(THINK_CLOSE) :].strip()
            if len(think_body) >= 16 and len(ans_body) >= 4:
                score += 0.60
            else:
                score -= 0.40
        else:
            score -= 0.90  # Unclosed or malformed <think> tag
    else:
        if n_open != n_close:
            score -= 0.90

    # 3. Task-Specific Verification
    # S8 target behaviour: no <tool_call> on plain code prompts (S6 greedy did it on 7/10 code probes).
    if task_type == "code" and "<tool_call>" in text:
        score -= 0.75

    if task_type == "math" and ground_truth_num is not None:
        pred_num = extract_last_number(text)
        if pred_num is not None and pred_num == str(ground_truth_num).replace(",", ""):
            score += 1.00
        else:
            score -= 0.75

    if requires_abstention:
        # Native markers are the phrases every S7.5 abstention SFT response uses (data/sft_v4_abstention_4000.jsonl,
        # 800/800 per language), so a correct native abstention is not out-scored by an English one. The old
        # Telugu marker was garbled (Devanagari + Kannada characters) and could never match.
        abstention_markers = [
            "cannot verify", "not provided", "insufficient information", "cannot answer", "i cannot",
            "ಮಾಹಿತಿ ಇಲ್ಲ", "ಗೊತ್ತಿಲ್ಲ", "ದೃಢೀಕರಿಸದ", "ಕಾಲ್ಪನಿಕ", "ದಾಖಲೆಗಳು ಲಭ್ಯವಿಲ್ಲ",
            "தகவல் இல்லை", "உறுதிப்படுத்தப்படாத", "வழங்க இயலாது",
            "సమాచారం లేదు", "ఆధారాలు లేవు", "ధృవీకరించబడని", "కల్పితమైన",
            "വിവരങ്ങൾ ലഭ്യമല്ല", "ലഭ്യമല്ല", "നൽകാൻ കഴിയില്ല", "സ്ഥിരീകരിക്കാത്ത", "സാങ്കൽപ്പിക",
        ]
        lower_t = text.lower()
        if any(m in lower_t for m in abstention_markers):
            score += 0.85
        else:
            score -= 0.85

    # 4. Dravidian Script Purity
    if lang in DRAVIDIAN_RANGES:
        purity = script_purity_score(text, lang)
        score += 0.50 * purity
        if purity < 0.50:
            score -= 0.60

    return score


# ---------------------------------------------------------------------------
# 2. Model Loading & Sequence Log-Prob Computation (Chunked for Low VRAM)
# ---------------------------------------------------------------------------

def load_slab_v4_checkpoint(ckpt_path: Path, device: str, dtype: torch.dtype = torch.float32) -> Tuple[SlabV4Model, ModelConfig]:
    # fp32 by default: train.py keeps fp32 master weights, and AdamW steps on bf16 params at
    # lr ~5e-7 round to zero (bf16 spacing near 0.02 is ~1e-4). Compute runs under bf16 autocast.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt.get("cfg", ckpt.get("config", {}))
    # Any pickled config object is used as-is (it may be model.config.ModelConfig, not root config.ModelConfig).
    cfg = ModelConfig(**cfg_raw) if isinstance(cfg_raw, dict) else cfg_raw

    model = SlabV4Model(cfg)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=True)
    del ckpt, state_dict, clean_state
    gc.collect()

    model = model.to(device=device, dtype=dtype)
    return model, cfg


def compute_response_logprob(
    model: SlabV4Model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    response_mask: torch.Tensor,
    reset_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes token-level log-probabilities restricted strictly to assistant response tokens
    (where response_mask == 1).

    Returns:
      - sum_logprob  : (B,) sum of log pi(y_t | x, y_<t) over response tokens
      - mean_logprob : (B,) length-normalized mean log-prob over response tokens
      - nll_loss     : scalar negative log-likelihood (-mean_logprob.mean()) for 0.1 * NLL anchor
    """
    out = model(input_ids, reset_mask=reset_mask)
    logits = out[0] if isinstance(out, tuple) else out  # (B, T, V)

    # Compute token-wise cross-entropy in FP32 without materializing full (B, T, V) log_softmax in FP32 at once
    B, T, V = logits.shape
    token_nll = F.cross_entropy(
        logits.reshape(B * T, V).float(),
        target_ids.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    mask_f = response_mask.float()
    token_logp = -token_nll * mask_f

    valid_counts = mask_f.sum(dim=-1).clamp(min=1.0)
    sum_logprob = token_logp.sum(dim=-1)
    mean_logprob = sum_logprob / valid_counts
    nll_loss = (-token_logp.sum()) / mask_f.sum().clamp(min=1.0)

    return sum_logprob, mean_logprob, nll_loss


# ---------------------------------------------------------------------------
# 3. Stage 1: On-Policy Preference Pair Generation (--mode generate)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_candidates_recurrent(
    model: SlabV4Model,
    prompt_ids: List[int],
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_ids: set,
    device: str,
) -> List[List[int]]:
    """
    Samples `num_candidates` completions in one batch through the O(1)-per-token recurrent step()
    (the old path re-ran the full prefix every token: O(T^2)). Prompt prefill is fed token by token,
    as in eval_60p_adversarial_suite.stream_slab_v4. Returns one id list per candidate; a list ends
    with EOT only if the model emitted EOT.
    """
    B = num_candidates
    states = model.empty_states(B, device=device)
    ids = torch.tensor([prompt_ids] * B, dtype=torch.long, device=device)
    for i in range(ids.size(1) - 1):
        _, states = model.step(ids[:, i:i + 1], states)
    curr = ids[:, -1:]
    outs: List[List[int]] = [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new_tokens):
        logits, states = model.step(curr, states)
        logits = logits.float() / max(temperature, 1e-4)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > top_p
        cutoff[:, 1:] = cutoff[:, :-1].clone()
        cutoff[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        pick = torch.multinomial(sorted_probs, num_samples=1)
        nxt = sorted_idx.gather(-1, pick)  # (B, 1)
        for b in range(B):
            if done[b]:
                continue
            t = int(nxt[b, 0].item())
            if t in stop_ids:
                done[b] = True
                if t == EOT_TOKEN_ID:
                    outs[b].append(t)
                continue
            outs[b].append(t)
        if all(done):
            break
        curr = nxt
    return outs


@torch.no_grad()
def generate_on_policy_pairs(
    sft_ckpt: Path,
    prompts_jsonl: Path,
    out_pairs_jsonl: Path,
    tokenizer_path: Path,
    num_candidates: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.92,
    min_score_margin: float = 0.35,
    device: str = "cuda",
    seed: int = 20261107,
) -> int:
    verify_apache_manifest(str(prompts_jsonl.parent))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # fp32, no autocast: same numerics as eval_60p_adversarial_suite.py decoding.
    model, _ = load_slab_v4_checkpoint(sft_ckpt, device=device, dtype=torch.float32)
    model.eval()
    torch.manual_seed(seed)

    # Same stop set as the 60-probe harness. EOT is the SFT turn terminator and is kept in the
    # ids (so DPO trains "stop here"); <|user|>/<|system|> stops are dropped and count as unterminated.
    stop_ids = {EOT_TOKEN_ID} | {tokenizer.token_to_id(t) for t in ("<|user|>", "<|system|>")} - {None}

    prompts: List[Dict[str, Any]] = []
    with prompts_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    out_pairs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept_pairs = 0
    t_start = time.time()

    with out_pairs_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, item in enumerate(prompts, start=1):
            prompt_text = item["prompt"]
            prompt_ids = tokenizer.encode(prompt_text).ids
            task_type = item.get("task_type", "general")
            lang = item.get("lang", "en")
            gt_num = item.get("ground_truth_num", None)
            req_think = bool(item.get("requires_think", THINK_OPEN in prompt_text))
            req_abstain = bool(item.get("requires_abstention", False))

            all_gen = sample_candidates_recurrent(
                model, prompt_ids, num_candidates, max_new_tokens, temperature, top_p, stop_ids, device
            )
            candidates: List[Tuple[float, List[int], str]] = []
            for gen_ids in all_gen:
                gen_text = tokenizer.decode([t for t in gen_ids if t != EOT_TOKEN_ID], skip_special_tokens=False)
                sc = score_on_policy_candidate(
                    text=gen_text,
                    token_ids=gen_ids,
                    task_type=task_type,
                    lang=lang,
                    ground_truth_num=gt_num,
                    requires_think=req_think,
                    requires_abstention=req_abstain,
                    ground_truth_text=item.get("ground_truth_text"),
                )
                candidates.append((sc, gen_ids, gen_text))

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_ids, best_txt = candidates[0]
            worst_sc, worst_ids, worst_txt = candidates[-1]

            if (best_sc - worst_sc) >= min_score_margin and best_ids != worst_ids:
                record = {
                    "prompt": prompt_text,
                    "prompt_ids": prompt_ids,
                    "chosen": best_txt,
                    "chosen_ids": best_ids,
                    "rejected": worst_txt,
                    "rejected_ids": worst_ids,
                    "score_chosen": round(best_sc, 4),
                    "score_rejected": round(worst_sc, 4),
                    "lang": lang,
                    "task_type": task_type,
                    "license": "Apache-2.0",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_pairs += 1

            if idx % 25 == 0:
                out_f.flush()
                el = time.time() - t_start
                print(f"[Generate] Processed {idx}/{len(prompts)} prompts | Kept pairs: {kept_pairs} | "
                      f"{el:.0f}s elapsed, ~{el / idx * (len(prompts) - idx):.0f}s left", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"[+] Saved {kept_pairs} on-policy DPO preference pairs to {out_pairs_jsonl}")
    return kept_pairs


# ---------------------------------------------------------------------------
# 4. Stage 2: Precompute & Pin Reference Log-Probs (--mode precompute-ref)
# ---------------------------------------------------------------------------

def build_sequence_tensors(
    prompt_ids: List[int],
    resp_ids: List[int],
    seq_len: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs prompt + response into (1, T) input_ids, target_ids, response_mask, and reset_mask.
    """
    full_ids = (prompt_ids + resp_ids)[: seq_len + 1]
    if len(full_ids) < 2:
        full_ids = [EOT_TOKEN_ID, EOT_TOKEN_ID]

    x_ids = full_ids[:-1]
    y_ids = full_ids[1:]
    p_len = max(0, len(prompt_ids) - 1)

    # Response mask is 0 on prompt tokens, 1 on assistant response tokens
    resp_mask = [0 if i < p_len else 1 for i in range(len(y_ids))]

    # The chunked SSD scan needs T % 64 == 0. Right-pad with <|pad|> (id 1), masked out of the loss:
    # the model is causal, so tokens after the response cannot change the response log-probs.
    pad = (-len(x_ids)) % 64
    x_ids = x_ids + [PAD_TOKEN_ID] * pad
    y_ids = y_ids + [PAD_TOKEN_ID] * pad
    resp_mask = resp_mask + [0] * pad

    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    y = torch.tensor([y_ids], dtype=torch.long, device=device)
    m = torch.tensor([resp_mask], dtype=torch.uint8, device=device)

    # Recurrent boundary reset mask: 1 everywhere inside single pair, 0 at start or EOT
    reset_mask = torch.ones_like(m, dtype=torch.uint8)
    reset_mask[:, 0] = 0
    reset_mask = reset_mask & (x != EOT_TOKEN_ID).to(torch.uint8)
    return x, y, m, reset_mask


@torch.no_grad()
def precompute_reference_logprobs(
    sft_ckpt: Path,
    pairs_jsonl: Path,
    ref_cache_npy: Path,
    tokenizer_path: Optional[Path] = None,
    seq_len: int = 4096,
    device: str = "cuda",
) -> Path:
    verify_apache_manifest(str(pairs_jsonl.parent))
    tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path and tokenizer_path.exists() else None

    print(f"[*] Loading frozen reference model from {sft_ckpt} to precompute log-probs...")
    # Same fp32-weights + bf16-autocast path as the policy in train mode, so log-ratios start at ~0.
    ref_model, _ = load_slab_v4_checkpoint(sft_ckpt, device=device, dtype=torch.float32)
    ref_model.eval()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    pairs: List[Dict[str, Any]] = []
    with pairs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    ref_records = np.zeros((len(pairs), 4), dtype=np.float32)
    # Columns: [chosen_sum_logp, chosen_mean_logp, rejected_sum_logp, rejected_mean_logp]

    for i, item in enumerate(pairs):
        p_ids = item.get("prompt_ids") or tokenizer.encode(item["prompt"]).ids
        c_ids = item.get("chosen_ids") or tokenizer.encode(item["chosen"]).ids
        r_ids = item.get("rejected_ids") or tokenizer.encode(item["rejected"]).ids

        xc, yc, mc, rmc = build_sequence_tensors(p_ids, c_ids, seq_len, device)
        xr, yr, mr, rmr = build_sequence_tensors(p_ids, r_ids, seq_len, device)

        with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16):
            c_sum, c_mean, _ = compute_response_logprob(ref_model, xc, yc, mc, rmc)
            r_sum, r_mean, _ = compute_response_logprob(ref_model, xr, yr, mr, rmr)

        ref_records[i, 0] = float(c_sum.item())
        ref_records[i, 1] = float(c_mean.item())
        ref_records[i, 2] = float(r_sum.item())
        ref_records[i, 3] = float(r_mean.item())

        if (i + 1) % 200 == 0 or (i + 1) == len(pairs):
            print(f"[Precompute-Ref] {i + 1}/{len(pairs)} pairs cached...")

    ref_cache_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(ref_cache_npy, ref_records)

    # Free reference model completely from GPU VRAM before DPO training starts
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[+] Pinned {len(pairs)} reference log-prob pairs to {ref_cache_npy} and freed reference VRAM.")
    return ref_cache_npy


# ---------------------------------------------------------------------------
# 5. Stage 3: Memory-Pinned DPO Training with 0.1 * NLL Anchor (--mode train)
# ---------------------------------------------------------------------------

def train_dpo_with_nll_anchor(
    sft_ckpt: Path,
    pairs_jsonl: Path,
    ref_cache_npy: Path,
    out_dir: Path,
    metrics_jsonl: Path,
    tokenizer_path: Optional[Path] = None,
    max_steps: int = 400,
    grad_accum: int = 8,
    lr: float = 1.5e-6,
    beta: float = 0.15,
    nll_coeff: float = 0.10,
    length_normalize: bool = True,
    grad_clip: float = 1.0,
    save_every: int = 100,
    seq_len: int = 4096,
    device: str = "cuda",
) -> List[Path]:
    verify_apache_manifest(str(pairs_jsonl.parent))
    if not ref_cache_npy.exists():
        precompute_reference_logprobs(sft_ckpt, pairs_jsonl, ref_cache_npy, tokenizer_path, seq_len, device)

    tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path and tokenizer_path.exists() else None
    pairs: List[Dict[str, Any]] = []
    with pairs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # Memory-map pinned reference log-probs from disk (zero GPU VRAM overhead)
    ref_table = np.load(ref_cache_npy, mmap_mode="r")
    assert len(ref_table) == len(pairs), "Mismatch between pairs JSONL and ref_logprobs.npy"

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    if dev_type == "cuda":
        torch.cuda.empty_cache()
    # fp32 master weights (bf16 compute via autocast); see load_slab_v4_checkpoint.
    model, cfg = load_slab_v4_checkpoint(sft_ckpt, device=device, dtype=torch.float32)
    model.train()
    # Belebele prompts reach ~1.5k tokens and chosen+rejected graphs coexist: checkpoint blocks (as S7.5 does).
    model.gradient_checkpointing_enable()

    # Segregated groups: matrix / embeddings+norms / recurrent_dynamics (0.1x LR, applied below).
    optimizer = model.configure_optimizers(
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95), use_fused=(dev_type == "cuda")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metrics_jsonl.write_text("", encoding="utf-8")  # one run per metrics file
    saved_anchors: List[Path] = []

    warmup_steps = max(20, int(0.10 * max_steps))
    rng = np.random.default_rng(seed=20261031)
    num_pairs = len(pairs)

    def _pair_len(it: Dict[str, Any]) -> int:
        p = it.get("prompt_ids") or []
        return len(p) + max(len(it.get("chosen_ids") or []), len(it.get("rejected_ids") or []))

    longest_idx = max(range(num_pairs), key=lambda i: _pair_len(pairs[i]))
    print(f"[Train] {num_pairs} pairs; longest pair idx {longest_idx} = {_pair_len(pairs[longest_idx])} tokens", flush=True)

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()

        # Cosine schedule with warmup
        if step <= warmup_steps:
            curr_lr = lr * (step / warmup_steps)
        else:
            prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            curr_lr = lr * 0.1 + 0.9 * lr * 0.5 * (1.0 + math.cos(math.pi * prog))

        for pg in optimizer.param_groups:
            scale = 0.1 if pg.get("name", "") == "recurrent_dynamics" else 1.0
            pg["lr"] = curr_lr * scale

        optimizer.zero_grad(set_to_none=True)

        step_loss_accum = 0.0
        step_dpo_accum = 0.0
        step_nll_accum = 0.0
        step_margin_accum = 0.0
        step_acc_accum = 0.0

        batch_indices = rng.integers(0, num_pairs, size=grad_accum)
        if step == 1:
            # Longest pair first, so the 2-step smoke run surfaces any OOM before the real run.
            batch_indices[0] = longest_idx

        for micro_idx in batch_indices:
            item = pairs[int(micro_idx)]
            p_ids = item.get("prompt_ids") or tokenizer.encode(item["prompt"]).ids
            c_ids = item.get("chosen_ids") or tokenizer.encode(item["chosen"]).ids
            r_ids = item.get("rejected_ids") or tokenizer.encode(item["rejected"]).ids

            xc, yc, mc, rmc = build_sequence_tensors(p_ids, c_ids, seq_len, device)
            xr, yr, mr, rmr = build_sequence_tensors(p_ids, r_ids, seq_len, device)

            # Read pinned reference log-probs from CPU mmap
            col_c = 1 if length_normalize else 0
            col_r = 3 if length_normalize else 2
            ref_c_logp = torch.tensor([ref_table[int(micro_idx), col_c]], dtype=torch.float32, device=device)
            ref_r_logp = torch.tensor([ref_table[int(micro_idx), col_r]], dtype=torch.float32, device=device)

            with torch.amp.autocast(device_type=dev_type, dtype=torch.bfloat16):
                pi_c_sum, pi_c_mean, nll_chosen = compute_response_logprob(model, xc, yc, mc, rmc)
                pi_r_sum, pi_r_mean, _ = compute_response_logprob(model, xr, yr, mr, rmr)

                pi_c = pi_c_mean if length_normalize else pi_c_sum
                pi_r = pi_r_mean if length_normalize else pi_r_sum

                chosen_logratios = pi_c - ref_c_logp
                rejected_logratios = pi_r - ref_r_logp
                logits_dpo = beta * (chosen_logratios - rejected_logratios)

                dpo_loss = -F.logsigmoid(logits_dpo).mean()
                total_loss = (dpo_loss + nll_coeff * nll_chosen) / grad_accum

            total_loss.backward()

            reward_margin = float((beta * (chosen_logratios - rejected_logratios)).mean().item())
            reward_acc = float((chosen_logratios > rejected_logratios).float().mean().item())

            step_loss_accum += float(total_loss.item())
            step_dpo_accum += float(dpo_loss.item()) / grad_accum
            step_nll_accum += float(nll_chosen.item()) / grad_accum
            step_margin_accum += reward_margin / grad_accum
            step_acc_accum += reward_acc / grad_accum

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dt = time.perf_counter() - t0
        vram_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if dev_type == "cuda" else 0.0
        if not all(math.isfinite(v) for v in (step_loss_accum, step_dpo_accum, step_nll_accum, grad_norm)):
            raise RuntimeError(f"[FAIL] Non-finite DPO telemetry at step {step}: loss={step_loss_accum} "
                               f"dpo={step_dpo_accum} nll={step_nll_accum} gnorm={grad_norm}")

        telemetry = {
            "step": step,
            "loss": round(step_loss_accum, 4),
            "dpo_loss": round(step_dpo_accum, 4),
            "nll_chosen": round(step_nll_accum, 4),
            "reward_margin": round(step_margin_accum, 4),
            "reward_acc": round(step_acc_accum, 4),
            "lr": curr_lr,
            "grad_norm": round(grad_norm, 4),
            "vram_gb": round(vram_gb, 3),
            "step_time_s": round(dt, 2),
        }

        print(
            f"[DPO Step {step:04d}/{max_steps:04d}] "
            f"loss={telemetry['loss']:.4f} (dpo={telemetry['dpo_loss']:.4f}, nll={telemetry['nll_chosen']:.4f}) | "
            f"margin={telemetry['reward_margin']:+.4f} | acc={telemetry['reward_acc']*100:.1f}% | "
            f"gnorm={telemetry['grad_norm']:.4f} | vram={telemetry['vram_gb']:.3f}GB | {dt:.2f}s"
        )

        with metrics_jsonl.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(telemetry) + "\n")

        # A run shorter than save_every (the smoke test) saves nothing.
        if step % save_every == 0 or (step == max_steps and max_steps >= save_every):
            if dev_type == "cuda":
                torch.cuda.empty_cache()
            ckpt_path = out_dir / f"step_{step:04d}.pt"
            tmp_path = out_dir / f"step_{step:04d}.pt.tmp"
            torch.save(
                {
                    "model": model.state_dict(),
                    # cfg object, not a dict: eval_60p / bpb_slab_v4 read cfg.n_embd and cfg.slab_v4.
                    "cfg": cfg,
                    "step": step,
                    "dpo_telemetry": telemetry,
                    "license": "Apache-2.0",
                },
                tmp_path,
            )
            os.replace(tmp_path, ckpt_path)
            saved_anchors.append(ckpt_path)
            print(f"[+] Saved DPO anchor checkpoint: {ckpt_path}")

    del model, optimizer
    torch.cuda.empty_cache()
    return saved_anchors


# ---------------------------------------------------------------------------
# 6. Stage 4: SWA-3 Stochastic Weight Averaging Merge (--mode swa3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def merge_swa3_checkpoints(anchor_paths: List[Path], out_golden_path: Path) -> Path:
    if len(anchor_paths) < 2:
        raise ValueError(f"SWA-3 requires at least 2 anchor checkpoints; got {len(anchor_paths)}")

    selected = anchor_paths[-3:]  # Take final 3 cooldown anchors
    print(f"[*] Merging SWA-{len(selected)} soup across: {[p.name for p in selected]}")

    avg_state: Dict[str, torch.Tensor] = {}
    base_meta: Dict[str, Any] = {}
    k_count = float(len(selected))

    for idx, path in enumerate(selected):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if idx == len(selected) - 1:
            base_meta = {k: v for k, v in ckpt.items() if k != "model"}

        state = ckpt["model"]
        for name, tensor in state.items():
            clean_name = name.replace("_orig_mod.", "")
            if torch.is_floating_point(tensor):
                if clean_name not in avg_state:
                    avg_state[clean_name] = tensor.float() / k_count
                else:
                    avg_state[clean_name] += tensor.float() / k_count
            else:
                if clean_name not in avg_state:
                    avg_state[clean_name] = tensor.clone()
        del ckpt

    # Cast back to BF16 and verify 0 NaNs
    final_state: Dict[str, torch.Tensor] = {}
    for name, tensor in avg_state.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"[FAIL] Non-finite values detected in SWA-3 tensor: {name}")
        final_state[name] = tensor.to(torch.bfloat16) if torch.is_floating_point(tensor) else tensor

    # Enforce tied embedding invariant (tok_emb.weight == head.weight)
    if "tok_emb.weight" in final_state and "head.weight" in final_state:
        final_state["head.weight"] = final_state["tok_emb.weight"]

    out_golden_path.parent.mkdir(parents=True, exist_ok=True)
    base_meta["model"] = final_state
    base_meta["swa3_anchors"] = [str(p) for p in selected]
    base_meta["license"] = "Apache-2.0"
    torch.save(base_meta, out_golden_path)
    print(f"[+] Exported SWA-3 Golden Apache-2.0 Checkpoint to: {out_golden_path}")
    return out_golden_path


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Parentheses 1.0-240M Session 8 On-Policy DPO + SWA-3")
    parser.add_argument("--mode", choices=["generate", "precompute-ref", "train", "swa3", "all"], default="all")
    parser.add_argument("--sft-ckpt", type=Path, default=REPO_ROOT / "checkpoints/parentheses-1.0-240M-s7-sft/latest.pt")
    parser.add_argument("--prompts-jsonl", type=Path, default=REPO_ROOT / "data/dpo_240m_apache/prompts.jsonl")
    parser.add_argument("--pairs-jsonl", type=Path, default=REPO_ROOT / "data/dpo_240m_apache/on_policy_pairs.jsonl")
    parser.add_argument("--ref-cache", type=Path, default=REPO_ROOT / "data/dpo_240m_apache/ref_logprobs.npy")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/v3_32k/tokenizer.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "checkpoints/parentheses-1.0-240M-s8-dpo-swa3")
    parser.add_argument("--metrics-jsonl", type=Path, default=REPO_ROOT / "out/session8_dpo_metrics.jsonl")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-6)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--nll-coeff", type=float, default=0.10)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-score-margin", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()

    if args.mode != "swa3" and not args.sft_ckpt.exists():
        raise FileNotFoundError(f"--sft-ckpt not found: {args.sft_ckpt}")
    if str(args.device).startswith("cuda"):
        # AGENTS.md: Windows silently spills past VRAM into system RAM at ~10x slower.
        torch.cuda.set_per_process_memory_fraction(0.92)

    if args.mode == "generate":
        generate_on_policy_pairs(
            sft_ckpt=args.sft_ckpt,
            prompts_jsonl=args.prompts_jsonl,
            out_pairs_jsonl=args.pairs_jsonl,
            tokenizer_path=args.tokenizer,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            min_score_margin=args.min_score_margin,
            device=args.device,
            seed=args.seed,
        )
    elif args.mode == "precompute-ref":
        precompute_reference_logprobs(
            sft_ckpt=args.sft_ckpt,
            pairs_jsonl=args.pairs_jsonl,
            ref_cache_npy=args.ref_cache,
            tokenizer_path=args.tokenizer,
            seq_len=args.seq_len,
            device=args.device,
        )
    elif args.mode in ("train", "all"):
        if args.mode == "all" and not args.ref_cache.exists():
            precompute_reference_logprobs(
                sft_ckpt=args.sft_ckpt,
                pairs_jsonl=args.pairs_jsonl,
                ref_cache_npy=args.ref_cache,
                tokenizer_path=args.tokenizer,
                seq_len=args.seq_len,
                device=args.device,
            )
        anchors = train_dpo_with_nll_anchor(
            sft_ckpt=args.sft_ckpt,
            pairs_jsonl=args.pairs_jsonl,
            ref_cache_npy=args.ref_cache,
            out_dir=args.out_dir,
            metrics_jsonl=args.metrics_jsonl,
            tokenizer_path=args.tokenizer,
            max_steps=args.max_steps,
            grad_accum=args.grad_accum,
            lr=args.lr,
            beta=args.beta,
            nll_coeff=args.nll_coeff,
            save_every=args.save_every,
            seq_len=args.seq_len,
            device=args.device,
        )
        if args.mode == "all" and len(anchors) >= 3:
            merge_swa3_checkpoints(anchors[-3:], args.out_dir / "parentheses-1.0-240M-dpo-swa3-golden.pt")
    elif args.mode == "swa3":
        ckpts = sorted(args.out_dir.glob("step_*.pt"))
        merge_swa3_checkpoints(ckpts[-3:], args.out_dir / "parentheses-1.0-240M-dpo-swa3-golden.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
