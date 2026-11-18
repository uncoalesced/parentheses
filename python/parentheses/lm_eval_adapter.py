"""lm-evaluation-harness adapter.

Lets the harness score a parentheses checkpoint with its own, unmodified task
definitions. Nothing here computes a metric: the adapter only returns
log-probabilities and generated text, and the harness does the scoring.

Prompts reach the model exactly as the harness builds them. No chat markers or
language tags are added, so results describe plain-text completion.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from . import TOKENIZER_PATH
from .chat import pick_device
from .checkpoint import load_model


@register_model("parentheses")
class ParenthesesLM(TemplateLM):
    def __init__(self, checkpoint: str, tokenizer: Optional[str] = None,
                 device: Optional[str] = None, batch_size: int = 8,
                 max_length: Optional[int] = None, max_gen_toks: int = 256,
                 dtype: str = "float32", **_ignored) -> None:
        super().__init__()
        self._device = torch.device(pick_device(device or "auto"))
        self.model = load_model(checkpoint, device=self._device, dtype=getattr(torch, dtype))
        self.tokenizer = Tokenizer.from_file(str(tokenizer or TOKENIZER_PATH))
        self.batch_size = int(batch_size) if str(batch_size).isdigit() else 8
        self.max_length = int(max_length or self.model.cfg.block_size)
        self.max_gen_toks = int(max_gen_toks)

    # -- harness plumbing ---------------------------------------------------

    @property
    def device(self):
        return self._device

    @property
    def eot_token_id(self) -> int:
        return self.model.cfg.eot_token_id

    @property
    def tokenizer_name(self) -> str:
        return "parentheses-v3_32k"

    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None, **_) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False).ids

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    # -- scoring ------------------------------------------------------------

    @torch.no_grad()
    def _score(self, pairs: Sequence[Tuple[List[int], List[int]]]) -> List[Tuple[float, bool]]:
        """Log-probability of each continuation given its context, one batch."""
        rows, spans = [], []
        for context, continuation in pairs:
            # Keep the right-hand side if the pair is too long: the continuation must survive.
            full = (context + continuation)[-(self.max_length + 1):]
            rows.append(full[:-1])
            spans.append(len(continuation))
        width = max(len(r) for r in rows)
        pad = self.model.cfg.pad_token_id
        ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows],
                           dtype=torch.long, device=self._device)
        # Right padding cannot leak backwards: the model is strictly causal.
        logprobs = F.log_softmax(self.model(ids).float(), dim=-1)
        out = []
        for b, ((context, continuation), n) in enumerate(zip(pairs, spans)):
            n = min(n, len(rows[b]))
            target = torch.tensor(continuation[-n:], dtype=torch.long, device=self._device)
            window = logprobs[b, len(rows[b]) - n:len(rows[b])]
            total = window.gather(-1, target.unsqueeze(-1)).sum().item()
            greedy = bool((window.argmax(-1) == target).all().item())
            out.append((total, greedy))
        return out

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **_):
        # Longest first, so an out-of-memory batch shows up immediately and
        # rows in a batch have similar lengths.
        order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][1]) + len(requests[i][2])))
        results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
        for start in tqdm(range(0, len(order), self.batch_size), disable=disable_tqdm,
                          desc="Scoring"):
            chunk = order[start:start + self.batch_size]
            scored = self._score([(requests[i][1], requests[i][2]) for i in chunk])
            for i, res in zip(chunk, scored):
                results[i] = res
                self.cache_hook.add_partial("loglikelihood", requests[i][0], res)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        totals = []
        for (text,) in tqdm([r.args for r in requests], disable=disable_tqdm, desc="Rolling"):
            windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                token_list=self.tok_encode(text), prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length, context_len=1)))
            total = 0.0
            for start in range(0, len(windows), self.batch_size):
                total += sum(lp for lp, _ in self._score(windows[start:start + self.batch_size]))
            totals.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
        return totals

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        outputs = []
        for context, gen_kwargs in tqdm([r.args for r in requests], disable=disable_tqdm,
                                        desc="Generating"):
            kwargs = dict(gen_kwargs or {})
            until = kwargs.get("until") or []
            until = [until] if isinstance(until, str) else [u for u in until if u]
            max_new = int(kwargs.get("max_gen_toks", self.max_gen_toks))
            temperature = float(kwargs.get("temperature", 0.0) or 0.0)
            if not kwargs.get("do_sample", False):
                temperature = 0.0

            ids = self.tok_encode(context)[-(self.max_length - max_new):] or [self.eot_token_id]
            prompt = torch.tensor([ids], dtype=torch.long, device=self._device)
            produced: List[int] = []
            text = ""
            for token in self.model.stream(prompt, max_new_tokens=max_new, temperature=temperature):
                produced.append(token)
                text = self.tok_decode(produced)
                if any(stop in text for stop in until):
                    break
            for stop in until:
                text = text.split(stop)[0]
            outputs.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
        return outputs
