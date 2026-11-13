#!/usr/bin/env python3
# Engineered by uncoalesced
"""Shared O(1)-memory sampling penalties and recurrent decoding helpers."""

from __future__ import annotations

import itertools
from typing import Iterator

import torch
import torch.nn.functional as F


def apply_sampling_penalties(
    logits: torch.Tensor,
    recent_tokens: list[list[int]],
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> None:
    """Apply repetition penalty and no-repeat n-gram blocking in place on (B, vocab) logits."""
    for b, seq in enumerate(recent_tokens):
        if repetition_penalty > 1.0 and seq:
            counts: dict[int, int] = {}
            for tid in seq:
                counts[tid] = counts.get(tid, 0) + 1
            repeated = [(tid, cnt) for tid, cnt in counts.items() if cnt >= 2]
            if repeated:
                tids = torch.tensor([tid for tid, _ in repeated], dtype=torch.long, device=logits.device)
                pens = torch.tensor(
                    [repetition_penalty ** min(cnt - 1, 3) for _, cnt in repeated],
                    dtype=logits.dtype,
                    device=logits.device,
                )
                vals = logits[b, tids]
                logits[b, tids] = torch.where(vals > 0, vals / pens, vals * pens)
        if no_repeat_ngram_size >= 2:
            n = no_repeat_ngram_size
            if len(seq) >= n:
                prefix = seq[-(n - 1):]
                for j in range(len(seq) - n + 1):
                    if seq[j:j + n - 1] == prefix:
                        logits[b, seq[j + n - 1]] = -float("inf")


def sample_next_token(
    logits: torch.Tensor,
    recent_tokens: list[list[int]] | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Sample next token IDs of shape (B, 1) and update rolling 64-token penalty history."""
    use_penalties = repetition_penalty > 1.0 or no_repeat_ngram_size >= 2
    logits = logits.float()
    if use_penalties and recent_tokens is not None:
        apply_sampling_penalties(logits, recent_tokens, repetition_penalty, no_repeat_ngram_size)

    if temperature <= 1e-4:
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if min_p is not None and min_p > 0.0:
            p_max = probs.max(dim=-1, keepdim=True).values
            p_threshold = min_p * p_max
            probs = torch.where(probs >= p_threshold, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        next_id = torch.multinomial(probs, num_samples=1)

    if use_penalties and recent_tokens is not None:
        for b in range(next_id.size(0)):
            recent_tokens[b].append(int(next_id[b, 0]))
            if len(recent_tokens[b]) > 64:
                del recent_tokens[b][:-64]

    return next_id


@torch.no_grad()
def stream_recurrent_step(
    model,
    idx: torch.Tensor,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eot_token_id: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield sampled token IDs of shape (B, 1) using O(1) per-token recurrent state."""
    use_penalties = repetition_penalty > 1.0 or no_repeat_ngram_size >= 2
    recent_tokens = [[] for _ in range(idx.size(0))] if use_penalties else None
    track_boundary = bool(getattr(model.cfg, "slab_v4", False)) and eot_token_id is not None

    states = model.empty_states(
        idx.size(0),
        device=idx.device,
        dtype=model.tok_emb.weight.dtype,
    )

    prev_was_eot = False
    for i in range(idx.size(1) - 1):
        tok_i = idx[:, i:i + 1]
        if track_boundary:
            _, states = model.step(tok_i, states, is_boundary=prev_was_eot)
            prev_was_eot = bool((tok_i[:, 0] == eot_token_id).all().item())
        else:
            _, states = model.step(tok_i, states)

    curr = idx[:, -1:]
    steps = itertools.count() if max_new_tokens is None else range(max_new_tokens)
    for _ in steps:
        if track_boundary:
            logits, states = model.step(curr, states, is_boundary=prev_was_eot)
            prev_was_eot = bool((curr[:, 0] == eot_token_id).all().item())
        else:
            logits, states = model.step(curr, states)

        next_id = sample_next_token(
            logits,
            recent_tokens=recent_tokens,
            temperature=temperature,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        yield next_id
        curr = next_id
