"""Single-token decoding replayed from a CUDA graph.

One token through 16 layers is a few hundred small kernels. Launched one by
one from Python, the launch cost dominates. The recurrent state has a fixed
shape, so the whole step can be captured once and replayed.

The graph runs the same `Parentheses.step` code on the same tensors, so its
output matches eager mode (the tests check it). The step after
<|endoftext|> resets the state; it is rare and runs eagerly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .model import CONV_KERNEL, LayerState, Parentheses, _wide


class GraphStepper:
    """Owns static state buffers and a captured `model.step`.

    Usage:
        g = GraphStepper(model)
        g.load_states(states)         # optional, e.g. after a parallel prefill
        logits = g.step(token_id)     # (vocab,) view into a static buffer; copy to keep
        states = g.read_states()      # clones
    """

    def __init__(self, model: Parentheses, warmup: int = 3):
        if not torch.cuda.is_available():
            raise RuntimeError("GraphStepper needs CUDA")
        self.model = model.eval()
        p = next(model.parameters())
        if p.device.type != "cuda":
            raise RuntimeError("move the model to CUDA first")
        self.device, self.dtype = p.device, p.dtype
        cfg = model.cfg
        H, Dh, D = cfg.n_head, cfg.head_dim, cfg.n_embd
        wide = _wide(p)
        self.static_token = torch.zeros(1, dtype=torch.long, device=self.device)
        self.static_states: List[LayerState] = [
            LayerState(torch.zeros(1, H, Dh, Dh, dtype=wide, device=self.device),
                       torch.zeros(1, CONV_KERNEL - 1, 3 * D, dtype=self.dtype, device=self.device),
                       torch.zeros(1, D, dtype=self.dtype, device=self.device))
            for _ in range(cfg.n_layer)]
        self.prev_eot = False
        self._capture(warmup)

    @torch.no_grad()
    def _body(self) -> torch.Tensor:
        logits, new_states = self.model.step(self.static_token, self.static_states, boundary=False)
        for dst, src in zip(self.static_states, new_states):
            for d, s in zip(dst, src):
                d.copy_(s)
        return logits[0]

    @torch.no_grad()
    def _capture(self, warmup: int) -> None:
        saved = self.read_states()
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self._body()
        self.load_states(saved)          # warm-up and capture must leave no trace

    # -- state in and out ---------------------------------------------------

    @torch.no_grad()
    def load_states(self, states: Optional[Sequence[LayerState]], prev_eot: bool = False) -> None:
        if states is None:
            for layer in self.static_states:
                for t in layer:
                    t.zero_()
        else:
            for dst, src in zip(self.static_states, states):
                for d, s in zip(dst, src):
                    d.copy_(s)
        self.prev_eot = prev_eot

    def read_states(self) -> List[LayerState]:
        return [LayerState(*(t.clone() for t in layer)) for layer in self.static_states]

    # -- decoding -----------------------------------------------------------

    @torch.no_grad()
    def step(self, token_id: int) -> torch.Tensor:
        """Write one token, return next-token logits (vocab,). The result is a
        view of a static buffer, overwritten by the next call."""
        if self.prev_eot:
            token = torch.tensor([int(token_id)], dtype=torch.long, device=self.device)
            logits, new_states = self.model.step(token, self.static_states, boundary=True)
            self.load_states(new_states)
            self.static_logits.copy_(logits[0])
        else:
            self.static_token.fill_(int(token_id))
            self.graph.replay()
        self.prev_eot = int(token_id) == self.model.cfg.eot_token_id
        return self.static_logits
