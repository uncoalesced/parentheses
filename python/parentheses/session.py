"""A chat that keeps the recurrent state between turns.

The state is a fixed-size summary of everything read so far, so a new turn
only has to read its own tokens. History is never fed again.

Turn format, as 1.0 was tuned:

    <|user|><lang> message <|assistant|> reply <|user|><lang> message <|assistant|> ...

Stop tokens end a reply and are not written into the state. In particular
<|endoftext|> is never fed between turns, because the token after it starts a
new document and wipes the state.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .model import LayerState, Parentheses

STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>", "<|pad|>")

# The sampling 1.0 was tuned with: low temperature, a min-p floor, no
# repetition penalty, and a block on repeating any 8-token run.
RELEASE_SAMPLING = {"temperature": 0.2, "top_k": 40, "min_p": 0.1, "no_repeat_ngram_size": 8}


def _block_repeated_ngrams(scores: torch.Tensor, history: List[int], n: int) -> None:
    """Forbid any token that would complete an n-gram already in `history`."""
    if n < 2 or len(history) < n:
        return
    prefix = history[-(n - 1):]
    for j in range(len(history) - n + 1):
        if history[j:j + n - 1] == prefix:
            scores[history[j + n - 1]] = float("-inf")


class ChatSession:
    """Holds one conversation's state. Feed turns in order.

    `ids_seen` is every token written into the state, in order. Reading it in
    one go with `forward_with_state` gives the same state (the tests check it).
    """

    def __init__(self, model: Parentheses, tokenizer=None, lang: str = "eng"):
        self.model = model
        self.tokenizer = tokenizer
        self.lang = lang
        self.device = next(model.parameters()).device
        self.states: Optional[List[LayerState]] = None
        self.ids_seen: List[int] = []
        self.next_logits: Optional[torch.Tensor] = None   # (vocab,) after the last fed token
        self._special = {}

    # -- low level ----------------------------------------------------------

    def _id(self, name: str) -> int:
        if name not in self._special:
            tid = None if self.tokenizer is None else self.tokenizer.token_to_id(name)
            if tid is None:
                raise KeyError(f"token {name} unknown; pass a tokenizer or use feed() with ids")
            self._special[name] = tid
        return self._special[name]

    def _encode(self, text_or_ids: Union[str, Sequence[int]]) -> List[int]:
        if isinstance(text_or_ids, str):
            if self.tokenizer is None:
                raise ValueError("text needs a tokenizer")
            return self.tokenizer.encode(text_or_ids, add_special_tokens=False).ids
        return [int(t) for t in text_or_ids]

    @property
    def _prev_eot(self) -> bool:
        return bool(self.ids_seen) and self.ids_seen[-1] == self.model.cfg.eot_token_id

    @torch.no_grad()
    def feed(self, ids: Sequence[int]) -> torch.Tensor:
        """Write tokens into the state with the parallel form. Returns the
        logits for the token after them."""
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("nothing to feed")
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, self.states = self.model.forward_with_state(x, self.states, prev_eot=self._prev_eot)
        self.ids_seen.extend(ids)
        self.next_logits = logits[0, -1]
        return self.next_logits

    @torch.no_grad()
    def feed_one(self, token_id: int) -> torch.Tensor:
        """Write one token with the recurrent step."""
        token = torch.tensor([int(token_id)], dtype=torch.long, device=self.device)
        logits, self.states = self.model.step(token, self.states, boundary=self._prev_eot)
        self.ids_seen.append(int(token_id))
        self.next_logits = logits[0]
        return self.next_logits

    # -- turns --------------------------------------------------------------

    def user_turn_ids(self, message: Union[str, Sequence[int]], lang: Optional[str] = None) -> List[int]:
        lang = lang or self.lang
        return ([self._id("<|user|>"), self._id(f"<{lang}>")] + self._encode(message)
                + [self._id("<|assistant|>")])

    def add_user(self, message: Union[str, Sequence[int]], lang: Optional[str] = None) -> torch.Tensor:
        """Read a user message, ending on <|assistant|> so the reply comes next."""
        return self.feed(self.user_turn_ids(message, lang))

    def add_assistant(self, reply: Union[str, Sequence[int]]) -> torch.Tensor:
        """Write a fixed assistant reply (for scripted conversations)."""
        return self.feed(self._encode(reply))

    def default_stops(self) -> List[int]:
        if self.tokenizer is None:
            return [self.model.cfg.eot_token_id]
        return [t for t in (self.tokenizer.token_to_id(s) for s in STOP_TOKENS) if t is not None]

    @torch.no_grad()
    def generate(self, max_new_tokens: int = 256, temperature: float = 0.0,
                 top_k: Optional[int] = None, min_p: Optional[float] = None,
                 no_repeat_ngram_size: int = 0,
                 stop_token_ids: Optional[Sequence[int]] = None,
                 on_token: Optional[Callable[[int], None]] = None) -> List[int]:
        """Generate a reply after the last fed token. Each kept token is
        written into the state; the stop token is not.

        `temperature=0` is greedy. `min_p` drops tokens less likely than
        min_p times the most likely one. `no_repeat_ngram_size` blocks
        repeating any n-gram from the last 64 generated tokens.
        `on_token` is called with each kept token, for streaming."""
        if self.next_logits is None:
            raise RuntimeError("feed a turn first")
        stops = set(int(t) for t in (stop_token_ids if stop_token_ids is not None
                                     else self.default_stops()))
        out: List[int] = []
        history: List[int] = []
        scores = self.next_logits
        for _ in range(max_new_tokens):
            scores = scores.float().clone()
            _block_repeated_ngrams(scores, history, no_repeat_ngram_size)
            if temperature <= 1e-4:
                next_id = int(scores.argmax())
            else:
                scores = scores / temperature
                if top_k:
                    kth = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                    scores = scores.masked_fill(scores < kth, float("-inf"))
                probs = F.softmax(scores, dim=-1)
                if min_p:
                    probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
                    probs = probs / probs.sum().clamp_min(1e-9)
                next_id = int(torch.multinomial(probs, 1))
            if next_id in stops:
                break
            out.append(next_id)
            history = (history + [next_id])[-64:]
            if on_token is not None:
                on_token(next_id)
            scores = self.feed_one(next_id)
        return out

    def reply(self, message: Union[str, Sequence[int]], lang: Optional[str] = None, **kw) -> List[int]:
        """add_user then generate."""
        self.add_user(message, lang)
        return self.generate(**kw)

    def state_bytes(self) -> int:
        return 0 if self.states is None else Parentheses.state_bytes(self.states)
